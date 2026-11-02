"""
src/route_methods/m7_llm_pathfinder.py - STRATA (Neuro-Symbolic Topological Lattice)

M7: LLM Full-Context Pathfinder.

Unlike M4 (one cell at a time, from a 5-candidate shortlist, names only) and
M5 (a same-shape one-shot call, names only, tried only as a last resort),
M7 gives the model the full prompt AND the full RAG-retrieved candidate set
WITH each cell's real input/output type signature, in one call, and asks it
to reason out the entire ordered path itself -- the way a human engineer
reading the node list would. The embedder still does the retrieval (the
`tunnel` argument, already narrowed by LatticeRouter.route()); the LLM is
the one deciding how those retrieved nodes chain together to satisfy the
prompt. Nothing here is domain-specific: the candidate list, and every
op/type name in it, comes entirely from whatever trees are loaded.

The proposed order is never trusted blindly -- each consecutive pair is
re-verified with the same Robinson-unification machinery (step_unifies /
find_bridge) every other route method already uses, and a 1-step bridge
cell is inserted automatically where the model's ordering has a type gap
it didn't (or couldn't) resolve itself.

Falls back to M1 (clause-anchor) when no LLM is loaded, the model's output
doesn't parse, or none of the proposed cell IDs are in the shown candidate
set.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any

from .base import RouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from lattice import Cell, LatticeOrchestrator
from unification import ExecutionContext
from inference import ModelManager

try:
    from utils import extract_json_object
except ImportError:
    from ..utils import extract_json_object

try:
    from node_resolver import DynamicNodeResolver
except (ImportError, ValueError):
    try:
        from ..node_resolver import DynamicNodeResolver
    except (ImportError, ValueError):
        DynamicNodeResolver = None

# How many candidate cells (by relevance) get shown to the model. Keeps the
# prompt within a small local model's context window regardless of how wide
# the retrieved tunnel is; ordering by relevance_map means truncation drops
# the least-relevant cells first, not an arbitrary prefix.
MAX_SHOWN_CANDIDATES = 25

PATH_SCHEMA = {
    "type": "object",
    "required": ["path"],
    "properties": {
        "path": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 14,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a pipeline planner. You are given a user request and a list "
    "of available operations (cells). Each line shows a cell ID, domain, "
    "description, and typed input/output ports.\n"
    "Select an ordered sequence of cell IDs from the available list "
    "that completely fulfills ALL steps of the user request from start to finish:\n"
    "- Select operations for each distinct step in the request "
    "(e.g., loading/initialization, preprocessing/filtering, transformation/computation, "
    "aggregation/analysis, combining/merging parallel branches, terminal formatting/output).\n"
    "- When operations apply to distinct inputs, branches, or sub-components, "
    "select extraction and transformation cells for EACH branch so all required branches exist in the pipeline.\n"
    "- NEVER select multiple redundant alternatives for the same step "
    "(e.g. choose exactly one operation per functional step; do not select redundant alternative implementations of the same transform).\n"
    "- Do NOT select operations that have no evidence in the prompt (e.g. do NOT select terminal sinks or exporters unless the prompt explicitly asks for persistence or export).\n"
    "- Chain transformations in logical execution order from initial ingress to final output.\n"
    "Respond with ONLY a JSON object: {\"path\": [\"CELL_ID\", ...]}"
)


def _fmt_port_sig(p: Any) -> str:
    sig = getattr(p, "signature", None)
    tn = str(getattr(sig, "type_name", "") or getattr(p, "abstract_type", "") or "any")
    st = str(getattr(sig, "state", "") or "")
    return f"{tn}[{st}]" if st and st not in ("any", "none") else tn


def _format_cell_signature(cell: Cell) -> str:
    """One compact, type-bearing line per cell with short summary."""
    req_in = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.inputs.items() if getattr(p, "required", False)]
    opt_in = [f"{n}:{_fmt_port_sig(p)}?" for n, p in cell.inputs.items() if not getattr(p, "required", False)]
    outs = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.outputs.items()]
    in_str = ", ".join(req_in + opt_in) or "none"
    out_str = ", ".join(outs) or "none"
    dom = getattr(cell, "domain_name", "") or "?"
    stage = getattr(cell, "stage", "?")
    doc = getattr(cell, "docstring", "") or ""
    summary = doc.split(".")[0].strip() if doc else ""
    desc_str = f" ({summary})" if summary else ""
    return f"{cell.cell_id} [{dom} s{stage}]{desc_str} in({in_str}) -> out({out_str})"


def _parse_proposed_path(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    data = extract_json_object(raw_text)
    if data and isinstance(data.get("path"), list):
        return [str(x).strip() for x in data["path"] if str(x).strip()]
    return []


class M7LLMPathfinderRouteMethod(RouteMethod):
    name = "m7_llm_pathfinder"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        self._current_prompt = prompt
        try:
            from inference import ModelManager
        except (ImportError, ValueError):
            from ..inference import ModelManager
        if not ModelManager.get_instance().can_synthesize():
            return []
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    getattr(c, "node_type", "") != "macro"
                    and not c.cell_id.startswith("MACRO_")
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ] or [tunnel[0]]
        mm = ModelManager.get_instance()
        has_llm = mm.can_synthesize()

        def _fallback(reason):
            return self._delegate_to_m1(
                reason, prompt, tunnel, relevance_map, orch, ctx=ctx,
                start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms,
            )

        if not has_llm:
            return _fallback("no LLM loaded in this profile")

        shown = self.select_stratified_candidates(prompt, candidates, relevance_map, orchestrator=orch, max_shown=MAX_SHOWN_CANDIDATES)
        cand_map = {c.cell_id.lower(): c for c in shown}

        sig_block = "\n".join(f"- {_format_cell_signature(c)}" for c in shown)
        user_msg = f"Request: {prompt}\n\nAvailable operations:\n{sig_block}"

        try:
            raw = mm.generate_text(
                user_msg, max_tokens=768, schema=PATH_SCHEMA, system_prompt=SYSTEM_PROMPT,
            )
        except Exception as _llm_err:
            return _fallback(f"LLM generation raised {type(_llm_err).__name__}: {_llm_err}")

        raw_ids = _parse_proposed_path(raw)
        proposed: List[Cell] = []
        for cid in raw_ids:
            c = cand_map.get(cid.strip().lower())
            if c is None:
                try:
                    from node_resolver import DynamicNodeResolver
                    c = DynamicNodeResolver.resolve_node(
                        raw_id=cid,
                        prompt=prompt,
                        orchestrator=orch,
                        rag=kwargs.get("rag"),
                    )
                    if c is not None:
                        cand_map[c.cell_id.lower()] = c
                        candidates.append(c)
                except Exception:
                    c = None
            if c is not None and (not proposed or proposed[-1].cell_id != c.cell_id):
                proposed.append(c)

        if not proposed:
            return _fallback("LLM proposed no resolvable cell ids")

        # Prune redundant alternative proposed cells targeting the same prompt clause
        if len(proposed) > 1:
            try:
                from token_evidence import get_evidence
                ev = get_evidence(orch)
                if ev is not None:
                    p_clauses = self.segment_prompt_clauses(prompt)
                    pruned_proposed: List[Cell] = []
                    clause_winners: Dict[int, Cell] = {}
                    for m in proposed:
                        best_cl_idx = None
                        best_f = -1.0
                        for idx, cl in enumerate(p_clauses):
                            toks = ev.clause_tokens(cl)
                            o = ev.identity_overlap(toks, m)
                            if o["f"] > best_f and o["recall"] > 0.0:
                                best_f = o["f"]
                                best_cl_idx = idx
                        if best_cl_idx is not None and best_f > 0.0:
                            if best_cl_idx not in clause_winners:
                                clause_winners[best_cl_idx] = m
                                pruned_proposed.append(m)
                            else:
                                prev_w = clause_winners[best_cl_idx]
                                prev_toks = ev.clause_tokens(p_clauses[best_cl_idx])
                                prev_o = ev.identity_overlap(prev_toks, prev_w)
                                prev_f = prev_o["f"]
                                if best_f > prev_f:
                                    pruned_proposed[pruned_proposed.index(prev_w)] = m
                                    clause_winners[best_cl_idx] = m
                                elif self.step_unifies(prev_w, m) or self.step_unifies(m, prev_w):
                                    pruned_proposed.append(m)
                        else:
                            b_roles = set()
                            try:
                                from lattice import TypeRegistry
                                b_roles = {str(r).lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
                            except Exception:
                                pass
                            if str(getattr(m, "node_role", "") or "").lower() in b_roles or getattr(m, "node_type", "") in ("bridge", "tunnel"):
                                pruned_proposed.append(m)
                    if pruned_proposed:
                        proposed = pruned_proposed
            except Exception:
                pass

        # Re-verify the model's proposed order against real type unification
        # rather than trusting it -- same machinery every other route method
        # uses, so a model that got the order slightly wrong still yields a
        # type-sound pipeline (or a bridge cell fills the gap) instead of
        # unification failing downstream.
        chain = [proposed[0]]
        first = chain[0]
        if not self.is_ingress_node(first, orch) and any(getattr(p, "required", False) and getattr(p, "default_value", None) is None for p in first.inputs.values()):
            s1_candidates = [c for c in candidates if self.is_ingress_node(c, orch)]
            if not s1_candidates and orch and hasattr(orch, "loaded_cells"):
                s1_candidates = [c for c in orch.loaded_cells.values() if self.is_ingress_node(c, orch)]
            if s1_candidates:
                best_s1 = None
                best_bridge = None
                best_score = -1.0
                for s1 in s1_candidates:
                    rel = relevance_map.get(s1.cell_id, 0.0) if relevance_map else 0.0
                    if self.step_unifies(s1, first, prev_path=[s1]):
                        score = 10.0 + rel
                        if score > best_score:
                            best_score = score
                            best_s1 = s1
                            best_bridge = None
                    else:
                        brg = self.find_bridge(s1, first, candidates, orch, prev_path=[s1])
                        if not brg and orch and hasattr(orch, "loaded_cells"):
                            brg = self.find_bridge(s1, first, list(orch.loaded_cells.values()), orch, prev_path=[s1])
                        if brg:
                            score = 5.0 + rel
                            if score > best_score:
                                best_score = score
                                best_s1 = s1
                                best_bridge = brg
                if best_s1 is None:
                    try:
                        p_clauses = self.segment_prompt_clauses(prompt)
                        producer = self._find_type_producer(first, [], candidates, p_clauses, orch)
                        if producer is not None and self.is_ingress_node(producer, orch):
                            best_s1 = producer
                    except Exception:
                        pass
                if best_s1 is not None:
                    if best_bridge is not None:
                        chain = [best_s1, best_bridge] + chain
                    else:
                        chain = [best_s1] + chain

        for nxt in proposed[1:]:
            if any(c.cell_id == nxt.cell_id for c in chain):
                continue
            curr = chain[-1]
            if self.step_unifies(curr, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                chain.append(nxt)
                continue
            bridge = self.find_bridge(curr, nxt, candidates, orch, prev_path=chain)
            if not bridge and orch and hasattr(orch, "loaded_cells"):
                bridge = self.find_bridge(curr, nxt, list(orch.loaded_cells.values()), orch, prev_path=chain)
            if bridge:
                chain.extend([bridge, nxt])
                continue

            # Multi-carrier / DAG branch check: can nxt directly unify or bridge from an earlier ancestor in chain?
            connected_ancestor = False
            for anc in reversed(chain[:-1]):
                if self.step_unifies(anc, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                    chain.append(nxt)
                    connected_ancestor = True
                    break
                b_anc = self.find_bridge(anc, nxt, candidates, orch, prev_path=chain)
                if not b_anc and orch and hasattr(orch, "loaded_cells"):
                    b_anc = self.find_bridge(anc, nxt, list(orch.loaded_cells.values()), orch, prev_path=chain)
                if b_anc:
                    chain.extend([b_anc, nxt])
                    connected_ancestor = True
                    break
            if connected_ancestor:
                continue

            try:
                from node_resolver import DynamicNodeResolver
                dyn_bridge = DynamicNodeResolver.synthesize_adapter(curr, nxt, orch, rag=kwargs.get("rag"))
                can_curr_dyn = self.step_unifies(curr, dyn_bridge, prev_path=chain) or self.step_unifies_dag(dyn_bridge, chain, ctx=ctx)
                can_dyn_nxt = self.step_unifies(dyn_bridge, nxt, prev_path=chain + [dyn_bridge]) or self.step_unifies_dag(nxt, chain + [dyn_bridge], ctx=ctx)
                if dyn_bridge and can_curr_dyn and can_dyn_nxt:
                    chain.extend([dyn_bridge, nxt])
                    continue
            except Exception:
                pass

            try:
                p_clauses = self.segment_prompt_clauses(prompt)
                producer = self._find_type_producer(nxt, chain, candidates, p_clauses, orch)
                if producer is not None:
                    chain.extend([producer, nxt])
                    continue
            except Exception:
                pass
            # If next node cannot be bridged, do not truncate early! Continue attempting remaining nodes
            continue

        if len(chain) < 2:
            return _fallback("verified chain shorter than 2 cells")

        self.tag_cells_with_clause_indices(chain, prompt)

        try:
            from token_evidence import audit
            aud = audit(chain, prompt, orch)
            if aud.get("available") and any(iss.get("kind") == "clause_unserved" for iss in aud.get("issues", [])):
                return _fallback("LLM proposed path has unserved clauses")
        except Exception:
            pass

        clauses = self.segment_prompt_clauses(prompt)
        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return chain[:max_allowed]

