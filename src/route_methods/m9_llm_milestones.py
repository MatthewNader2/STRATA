"""
src/route_methods/m9_llm_milestones.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

M9: LLM Milestone Anchor & Topological Pathfinder.

Unlike M7 (which asks the LLM to output every single node including intermediate
adapters in one call) and M8 (which steps one node at a time):

M9 decomposes the problem into two distinct phases:
1. Milestone Extraction (LLM):
   The LLM inspects the prompt and candidate operations (with their full
   typed signatures) and decides ONLY the core "must-have" operations
   (e.g., data loader, feature transformer, estimator/model, exporter/plot)
   in chronological workflow order. It does not need to worry about low-level
   carrier conversions or intermediate adapter/bridge cells.

2. Topological Path Synthesis (Robinson Unification & Multi-Hop Bridging):
   STRATA connects consecutive milestone pairs (M_i -> M_{i+1}) using:
   - Direct edge if step_unifies(M_i, M_{i+1})
   - 1-step bridge cell via find_bridge(M_i, M_{i+1})
   - Multi-step BFS search through the candidate lattice if a 2-hop adapter
     is required to align types/carriers.

If any gap between milestones cannot be bridged, or if no LLM is loaded,
it gracefully falls back to M1 (clause-anchor).
"""
from __future__ import annotations
import json
from collections import deque
from typing import Dict, List, Optional, Any, Set

from .base import RouteMethod, _bridge_roles, tokenize_alphanumeric
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from lattice import Cell, LatticeOrchestrator, TypeRegistry
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

MAX_SHOWN_CANDIDATES = 25

MILESTONE_SCHEMA = {
    "type": "object",
    "required": ["milestones"],
    "properties": {
        "milestones": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 14,
            "description": "Ordered list of essential, must-have cell IDs"
        },
    },
}

SYSTEM_PROMPT_MILESTONES = (
    "You are an expert pipeline architect. You are given a user request and a list "
    "of available typed operations (cells). Each cell has an ID, domain, stage, "
    "input ports with types, and output ports with types.\n"
    "Identify the essential MUST-HAVE operations (milestones) required to fulfill "
    "all steps of the user request from start to finish:\n"
    "- When operations apply to distinct inputs or branches, include the appropriate selection and transformation cells for EACH branch so all branches exist in the pipeline.\n"
    "- NEVER select multiple redundant alternatives for the same step; choose exactly one operation per requested step.\n"
    "- Do NOT select operations that have no evidence in the prompt.\n"
    "- Do NOT include intermediate adapters or bridge cells -- only the primary operations "
    "in their proper chronological execution order.\n"
    "Respond with ONLY a JSON object of the form: "
    "{\"milestones\": [\"CELL_ID\", \"CELL_ID\", ...]}. No prose, no markdown."
)


def _fmt_port_sig(p: Any) -> str:
    sig = getattr(p, "signature", None)
    tn = str(getattr(sig, "type_name", "") or getattr(p, "abstract_type", "") or "any")
    st = str(getattr(sig, "state", "") or "")
    return f"{tn}[{st}]" if st and st not in ("any", "none") else tn


def _format_cell_signature(cell: Cell) -> str:
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


def _parse_milestones(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    data = extract_json_object(raw_text)
    if data and isinstance(data.get("milestones"), list):
        return [str(x).strip() for x in data["milestones"] if str(x).strip()]
    return []


class M9LLMMilestonePathfinderRouteMethod(RouteMethod):
    name = "m9_llm_milestones"

    def _find_bfs_path(
        self,
        src: Cell,
        dst: Cell,
        candidate_pool: List[Cell],
        max_hops: int = 3,
        prev_path: Optional[List[Cell]] = None
    ) -> Optional[List[Cell]]:
        """
        Searches for a short unifying path from src to dst through candidate_pool.
        Returns the intermediate nodes [B1, B2, ...] excluding src and dst.
        """
        base_scope = prev_path if prev_path else [src]
        queue = deque([(src, [])])
        visited: Set[str] = {src.cell_id}

        prompt_str = getattr(self, "_current_prompt", "") or ""
        prompt_toks = set(tokenize_alphanumeric(str(prompt_str).lower()))
        declared_bridge_roles = _bridge_roles()

        while queue:
            curr, intermediates = queue.popleft()
            if len(intermediates) >= max_hops:
                continue

            current_scope = base_scope + intermediates
            for cand in candidate_pool:
                if cand.cell_id == dst.cell_id:
                    if self.step_unifies(curr, dst, prev_path=current_scope):
                        return intermediates
                    continue

                if cand.cell_id in visited or self.is_ingress_node(cand):
                    continue

                reg = TypeRegistry.get_instance()
                comb_roles = set(reg.get_verification_semantics("combinator_roles") or ())
                comb_topologies = set(reg.get_verification_semantics("combinator_topologies") or ())
                cand_role = str(getattr(cand, "node_role", "") or getattr(cand, "role", "") or "").lower()
                cand_topology = str(getattr(cand, "topology_type", "") or getattr(cand, "node_type", "") or "").lower()
                if (
                    getattr(cand, "is_combinator", False)
                    or cand_role in comb_roles
                    or cand_topology in comb_topologies
                ):
                    continue

                destructive_names = set(reg.get_verification_semantics("destructive_effects"))
                effects = getattr(cand, "effects", []) or []
                has_destructive_effect = any(
                    (isinstance(e, dict) and e.get("name") in destructive_names)
                    or (isinstance(e, str) and e in destructive_names)
                    for e in effects
                )
                if has_destructive_effect:
                    continue

                cand_toks = set(getattr(cand, "token_set", set())) | set(getattr(cand, "keywords", []) or []) | set(getattr(cand, "semantic_tags", []) or [])
                cand_toks = {str(t).lower() for t in cand_toks}
                bridge_tokens = set(reg.get_verification_semantics("bridge_tokens") or ())
                is_declared_bridge = (
                    cand_role in declared_bridge_roles
                    or getattr(cand, "node_type", "") in ("tunnel", "bridge")
                    or bool(bridge_tokens & cand_toks)
                )

                if not is_declared_bridge and not (cand_toks & prompt_toks):
                    continue

                if self.step_unifies(curr, cand, prev_path=current_scope):
                    visited.add(cand.cell_id)
                    # Check if cand can directly reach dst
                    if self.step_unifies(cand, dst, prev_path=current_scope + [cand]):
                        return intermediates + [cand]
                    queue.append((cand, intermediates + [cand]))

        return None

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        ctx: Optional[ExecutionContext] = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        **kwargs
    ) -> List[Cell]:
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

        # Phase 1: Milestone Identification
        sig_block = "\n".join(f"- {_format_cell_signature(c)}" for c in shown)
        user_msg = f"Request: {prompt}\n\nAvailable operations:\n{sig_block}"

        try:
            raw = mm.generate_text(
                user_msg, max_tokens=512, schema=MILESTONE_SCHEMA, system_prompt=SYSTEM_PROMPT_MILESTONES
            )
        except Exception as _llm_err:
            return _fallback(f"LLM generation raised {type(_llm_err).__name__}: {_llm_err}")

        milestone_ids = _parse_milestones(raw)
        milestones: List[Cell] = []
        for mid in milestone_ids:
            c = cand_map.get(mid.strip().lower())
            if c is None:
                try:
                    from node_resolver import DynamicNodeResolver
                    c = DynamicNodeResolver.resolve_node(
                        raw_id=mid,
                        prompt=prompt,
                        orchestrator=orch,
                        rag=kwargs.get("rag"),
                    )
                    if c is not None:
                        cand_map[c.cell_id.lower()] = c
                        candidates.append(c)
                except Exception:
                    c = None
            if c is not None and (not milestones or milestones[-1].cell_id != c.cell_id):
                milestones.append(c)

        if not milestones:
            return _fallback("LLM proposed no milestones")

        # Prune redundant alternative milestones targeting the same prompt clause
        if len(milestones) > 1:
            try:
                from token_evidence import get_evidence
                ev = get_evidence(orch)
                if ev is not None:
                    p_clauses = self.segment_prompt_clauses(prompt)
                    pruned_milestones: List[Cell] = []
                    clause_winners: Dict[int, Cell] = {}
                    for m in milestones:
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
                                pruned_milestones.append(m)
                            else:
                                prev_w = clause_winners[best_cl_idx]
                                prev_toks = ev.clause_tokens(p_clauses[best_cl_idx])
                                prev_o = ev.identity_overlap(prev_toks, prev_w)
                                prev_f = prev_o["f"]
                                if best_f > prev_f:
                                    pruned_milestones[pruned_milestones.index(prev_w)] = m
                                    clause_winners[best_cl_idx] = m
                                elif self.step_unifies(prev_w, m) or self.step_unifies(m, prev_w):
                                    pruned_milestones.append(m)
                        else:
                            pruned_milestones.append(m)
                    if pruned_milestones:
                        milestones = pruned_milestones
            except Exception:
                pass

        # Phase 2: Topological Path Synthesis between Milestones
        chain: List[Cell] = [milestones[0]]
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

        for target in milestones[1:]:
            if any(c.cell_id == target.cell_id for c in chain):
                continue
            curr = chain[-1]

            # 1. Direct edge
            if self.step_unifies(curr, target, prev_path=chain) or self.step_unifies_dag(target, chain, ctx=ctx):
                chain.append(target)
                continue

            # 2. 1-Step Bridge
            bridge = self.find_bridge(curr, target, candidates, orch, prev_path=chain)
            if not bridge and orch and hasattr(orch, "loaded_cells"):
                bridge = self.find_bridge(curr, target, list(orch.loaded_cells.values()), orch, prev_path=chain)
            if bridge:
                chain.extend([bridge, target])
                continue

            # 2b. Multi-carrier / DAG branch check: can target unify or bridge from an earlier ancestor in chain?
            connected = False
            for anc in reversed(chain[:-1]):
                if self.step_unifies(anc, target, prev_path=chain) or self.step_unifies_dag(target, chain, ctx=ctx):
                    chain.append(target)
                    connected = True
                    break
                b_anc = self.find_bridge(anc, target, candidates, orch, prev_path=chain)
                if not b_anc and orch and hasattr(orch, "loaded_cells"):
                    b_anc = self.find_bridge(anc, target, list(orch.loaded_cells.values()), orch, prev_path=chain)
                if b_anc:
                    chain.extend([b_anc, target])
                    connected = True
                    break
            if connected:
                continue

            # 3. Multi-Hop BFS Bridge (up to 2 intermediate nodes)
            bfs_intermediates = self._find_bfs_path(curr, target, candidates, max_hops=2, prev_path=chain)
            if bfs_intermediates is not None:
                chain.extend(bfs_intermediates + [target])
                continue

            # 4. On-demand dynamic adapter synthesis (Dev Mode)
            try:
                from node_resolver import DynamicNodeResolver
                dyn_bridge = DynamicNodeResolver.synthesize_adapter(curr, target, orch, rag=kwargs.get("rag"))
                can_curr_dyn = self.step_unifies(curr, dyn_bridge, prev_path=chain) or self.step_unifies_dag(dyn_bridge, chain, ctx=ctx)
                can_dyn_tgt = self.step_unifies(dyn_bridge, target, prev_path=chain + [dyn_bridge]) or self.step_unifies_dag(target, chain + [dyn_bridge], ctx=ctx)
                if dyn_bridge and can_curr_dyn and can_dyn_tgt:
                    chain.extend([dyn_bridge, target])
                    continue
            except Exception:
                pass

            # 5. Type-directed repair: if milestone target requires a prerequisite type (e.g. model fit producer)
            try:
                p_clauses = self.segment_prompt_clauses(prompt)
                producer = self._find_type_producer(target, chain, shown, p_clauses, orch)
                if producer is not None:
                    chain.extend([producer, target])
                    continue
            except Exception:
                pass

            # If milestone cannot be connected, skip it and continue attempting subsequent milestones
            continue

        # Final verification: verify all consecutive pairs
        if len(chain) > 1:
            verified = [chain[0]]
            for nxt in chain[1:]:
                c_prev = verified[-1]
                if self.step_unifies(c_prev, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                    verified.append(nxt)
                else:
                    brg = self.find_bridge(c_prev, nxt, candidates, orch, prev_path=verified)
                    if not brg and orch and hasattr(orch, "loaded_cells"):
                        brg = self.find_bridge(c_prev, nxt, list(orch.loaded_cells.values()), orch, prev_path=verified)
                    if brg:
                        verified.extend([brg, nxt])
                    else:
                        # DAG branch check: can nxt unify or bridge from any ancestor in verified?
                        anc_ok = False
                        for anc in reversed(verified[:-1]):
                            if self.step_unifies(anc, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                                verified.append(nxt)
                                anc_ok = True
                                break
                            b_anc = self.find_bridge(anc, nxt, candidates, orch, prev_path=verified)
                            if not b_anc and orch and hasattr(orch, "loaded_cells"):
                                b_anc = self.find_bridge(anc, nxt, list(orch.loaded_cells.values()), orch, prev_path=verified)
                            if b_anc:
                                verified.extend([b_anc, nxt])
                                anc_ok = True
                                break
                        if not anc_ok:
                            adapter_added = False
                            try:
                                from node_resolver import DynamicNodeResolver
                                dyn_brg = DynamicNodeResolver.synthesize_adapter(c_prev, nxt, orch, rag=kwargs.get("rag"))
                                can_c_prev = self.step_unifies(c_prev, dyn_brg, prev_path=verified) or self.step_unifies_dag(dyn_brg, verified, ctx=ctx)
                                can_dyn_nxt = self.step_unifies(dyn_brg, nxt, prev_path=verified + [dyn_brg]) or self.step_unifies_dag(nxt, verified + [dyn_brg], ctx=ctx)
                                if dyn_brg and can_c_prev and can_dyn_nxt:
                                    verified.extend([dyn_brg, nxt])
                                    adapter_added = True
                            except Exception:
                                pass
                            if not adapter_added:
                                try:
                                    p_clauses = self.segment_prompt_clauses(prompt)
                                    producer = self._find_type_producer(nxt, verified, candidates, p_clauses, orch)
                                    if producer is not None:
                                        verified.extend([producer, nxt])
                                        continue
                                except Exception:
                                    pass
                                continue
            chain = verified

        if len(chain) < 2:
            return _fallback("verified chain shorter than 2 cells")

        self.tag_cells_with_clause_indices(chain, prompt)
        clauses = self.segment_prompt_clauses(prompt)
        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return chain[:max_allowed]
