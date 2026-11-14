"""
src/route_methods/m8_llm_stepwise.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

M8: LLM Stepwise Edge Pathfinder.

Unlike M4 (which only showed 5 bare cell names at each step) and M7 (which
plans the entire path in one global shot), M8 performs iterative per-node
decision making with full type context and explicit edge visibility:

At each step:
1. Shows the current pipeline prefix built so far.
2. Identifies all topologically valid next-hop transitions from the current
   node (both direct unifying edges and 1-step bridged transitions).
3. Shows the rest of the relevant candidate nodes in the pool with full
   port type signatures so the model maintains foresight of overall goals.
4. Asks the model to choose the next node from the valid transitions, or
   FINISH when the pipeline is complete.

Every step is verified with Robinson unification (step_unifies / find_bridge).
Falls back to M1 (clause-anchor) when no LLM is loaded or parsing fails.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, Any, Tuple, Set

from log_config import get_logger

logger = get_logger("route_methods.m8")

from .base import RouteMethod, is_terminal_sink, _clause_has_egress_intent, is_macro_cell, _fmt_port_sig
from lattice import Cell, LatticeOrchestrator, TypeRegistry
from unification import ExecutionContext
from inference import ModelManager

try:
    from utils import extract_json_object, tokenize_alphanumeric
except ImportError:
    from ..utils import extract_json_object, tokenize_alphanumeric

try:
    from node_resolver import DynamicNodeResolver
except (ImportError, ValueError):
    try:
        from ..node_resolver import DynamicNodeResolver
    except (ImportError, ValueError):
        DynamicNodeResolver = None

MAX_SHOWN_CANDIDATES = 35
MAX_TRANSITIONS_SHOWN = 15

STEP_SCHEMA = {
    "type": "object",
    "required": ["next_node"],
    "properties": {
        "next_node": {"type": "string"},
    },
}

SYSTEM_PROMPT_ENTRY = (
    "You are a pipeline planner. Given a user request and a list of available "
    "typed entry operations (cells), select the best operation to BEGIN the "
    "pipeline.\n"
    "Respond with ONLY a JSON object of the form: {\"next_node\": \"CELL_ID\"}."
)

SYSTEM_PROMPT_STEP = (
    "You are an iterative pipeline planner. You are given a user request, the "
    "current pipeline prefix, a list of topologically VALID next transitions from "
    "the current node, and the remaining available candidate operations in the pool.\n"
    "Select the NEXT operation from the 'Valid Next Transitions' list to advance "
    "toward fulfilling the request, or output 'FINISH' if the pipeline has achieved all goals of the request.\n"
    "- NEVER select terminal sinks or file exporters while intermediate processing clauses are still unserved.\n"
    "- Only select transforms that have positive lexical or structural evidence in the prompt.\n"
    "- Progress through the request clauses sequentially in chronological order.\n"
    "Respond with ONLY a JSON object of the form: {\"next_node\": \"CELL_ID\"} or "
    "{\"next_node\": \"FINISH\"}. No prose, no markdown."
)


def _format_cell_signature(cell: Cell) -> str:
    req_in = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.inputs.items() if getattr(p, "required", False)]
    opt_in = [f"{n}:{_fmt_port_sig(p)}[opt]" for n, p in cell.inputs.items() if not getattr(p, "required", False)]
    outs = [f"{n}:{_fmt_port_sig(p)}" for n, p in cell.outputs.items()]
    in_str = ", ".join(req_in + opt_in) or "none"
    out_str = ", ".join(outs) or "none"
    dom = getattr(cell, "domain_name", "") or "unk"
    stage = getattr(cell, "stage", "unk")
    doc = getattr(cell, "docstring", "") or ""
    summary = doc.split(".")[0].strip() if doc else ""
    desc_str = f" ({summary})" if summary else ""
    return f"{cell.cell_id} [{dom} s{stage}]{desc_str} in({in_str}) -> out({out_str})"


def _parse_next_node(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    data = extract_json_object(raw_text)
    if data and isinstance(data.get("next_node"), str):
        return data["next_node"].strip()
    # Fallback token search without regex
    toks = tokenize_alphanumeric(raw_text)
    if "finish" in toks:
        return "FINISH"
    return toks[0] if toks else None


class M8LLMStepwisePathfinderRouteMethod(RouteMethod):
    name = "m8_llm_stepwise"

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
                    not is_macro_cell(c)
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ] or [tunnel[0]]
        mm = ModelManager.get_instance()
        has_llm = mm.can_synthesize()

        def _fallback(reason):
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        if not has_llm:
            return _fallback("no LLM loaded in this profile")

        shown = self.select_stratified_candidates(prompt, candidates, relevance_map, orchestrator=orch, max_shown=MAX_SHOWN_CANDIDATES)
        cand_map = {c.cell_id.lower(): c for c in shown}

        # Step 0: Choose entry node
        src_file_literals, _ = self.extract_file_literals(prompt or "", ctx=ctx)
        entry_candidates = [
            c for c in shown
            if self.is_ingress_node(c, orch)
        ]
        if not entry_candidates:
            entry_candidates = shown[:5]
        elif src_file_literals:
            file_compat = [
                c for c in entry_candidates
                if self._is_path_consumer(c) and self.is_file_format_compatible(c, str(src_file_literals[0]))
            ]
            if file_compat:
                entry_candidates = file_compat
            else:
                def _entry_compat_score(c: Cell) -> float:
                    is_pc = self._is_path_consumer(c)
                    if is_pc and self.is_file_format_compatible(c, str(src_file_literals[0])):
                        return relevance_map.get(c.cell_id, 0.0) + 10.0
                    elif is_pc and not self.is_file_format_compatible(c, str(src_file_literals[0])):
                        return relevance_map.get(c.cell_id, 0.0) - 20.0
                    elif not is_pc:
                        return relevance_map.get(c.cell_id, 0.0) - 10.0
                    return relevance_map.get(c.cell_id, 0.0)
                entry_candidates.sort(key=_entry_compat_score, reverse=True)

        entry_cand_map = {c.cell_id.lower(): c for c in entry_candidates}
        entry_cell = None
        try:
            entry_lines = "\n".join(f"- {_format_cell_signature(c)}" for c in entry_candidates[:8])
            entry_prompt = f"Request: {prompt}\n\nAvailable Entry Operations:\n{entry_lines}"
            raw = mm.generate_text(
                entry_prompt, max_tokens=128, schema=STEP_SCHEMA, system_prompt=SYSTEM_PROMPT_ENTRY
            )
            node_id = _parse_next_node(raw)
            if node_id:
                entry_cell = entry_cand_map.get(node_id.lower())
                if entry_cell is None:
                    try:
                        from node_resolver import DynamicNodeResolver
                        resolved = DynamicNodeResolver.resolve_node(
                            raw_id=node_id,
                            prompt=prompt,
                            orchestrator=orch,
                            rag=kwargs.get("rag"),
                        )
                        if resolved and resolved.cell_id.lower() in entry_cand_map and getattr(resolved, "verified", True):
                            entry_cell = resolved
                    except Exception:
                        entry_cell = None
        except Exception:
            entry_cell = None

        if entry_cell is None and entry_candidates:
            entry_cell = entry_candidates[0]

        _ev = self._evidence()
        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        if not clauses:
            clauses = [prompt]
        num_clauses = len(clauses)

        path: List[Cell] = [entry_cell]
        visited_ids = {entry_cell.cell_id}
        max_steps = max(24, num_clauses * 3 + 4, max_transforms + 8)

        def _cand_clause(c: Cell) -> Optional[int]:
            idx = getattr(c, "matched_clause_idx", None)
            if idx is not None:
                return idx
            if _ev is not None and clauses:
                c_toks = getattr(c, "identity_tokens", None) or c.token_set
                best_j, best_f = None, 0.0
                for j, cl in enumerate(clauses):
                    cl_t = _ev.clause_tokens(cl)
                    overlap = len(c_toks & cl_t)
                    if overlap > 0 and overlap > best_f:
                        best_j, best_f = j, float(overlap)
                return best_j
            return None

        prompt_tokens = _ev.clause_tokens(prompt) if _ev is not None else set(tokenize_alphanumeric(prompt.lower()))
        covered_tokens: Set[str] = set(getattr(entry_cell, "identity_tokens", entry_cell.token_set) & prompt_tokens)

        # Stepwise expansion loop
        for _ in range(max_steps):
            curr = path[-1]
            curr_stage = getattr(curr, "stage", None)
            curr_role = str(getattr(curr, "node_role", "") or "").lower()

            covered_indices = {
                _cand_clause(c) for c in path if _cand_clause(c) is not None
            }
            if _ev is not None:
                for j, cl in enumerate(clauses):
                    cl_toks = _ev.clause_tokens(cl)
                    if cl_toks and (cl_toks & covered_tokens) and len(cl_toks & covered_tokens) / len(cl_toks) >= 0.5:
                        covered_indices.add(j)
            uncovered_indices = [j for j in range(len(clauses)) if j not in covered_indices]
            target_cl_idx = min(uncovered_indices) if uncovered_indices else (num_clauses - 1)

            target_cl = clauses[target_cl_idx] if target_cl_idx < num_clauses else ""
            target_is_egress = (
                _clause_has_egress_intent(target_cl)
                or (self.requested_sink(target_cl, shown) is not None)
                or (len(uncovered_indices) <= 1 and target_cl_idx >= num_clauses - 1)
            )

            # Identify valid next transitions from curr
            # Tuple: (target_cell, bridge_cell_or_None, transition_type_str)
            transitions: List[Tuple[Cell, Optional[Cell], str]] = []

            for c in shown:
                if c.cell_id in visited_ids:
                    continue
                if self.is_ingress_node(c, orch):
                    continue
                if is_terminal_sink(c, orch) and not target_is_egress and uncovered_indices:
                    continue

                if self.step_unifies(curr, c, prev_path=path):
                    transitions.append((c, None, "DIRECT"))
                elif self.step_unifies_dag(c, path, ctx=ctx):
                    transitions.append((c, None, "DAG_FRONTIER"))
                else:
                    bridge = self.find_bridge(curr, c, candidates, orch, prev_path=path)
                    if not bridge and orch and hasattr(orch, "loaded_cells"):
                        bridge = self.find_bridge(curr, c, list(orch.loaded_cells.values()), orch, prev_path=path)
                    if bridge and bridge.cell_id not in visited_ids:
                        transitions.append((c, bridge, f"BRIDGED via {bridge.cell_id}"))

            if not transitions:
                break

            # Pure Markovian type-mask ranking over E_valid(Gamma_t): no M1 exact_target override
            def _trans_score(t: Tuple[Cell, Optional[Cell], str]) -> Tuple[float, float]:
                tgt, brg, _ = t
                fit_rec, _, _ = self.clause_fit(prompt, tgt) if prompt else (0.0, 0.0, 0.0)
                score = relevance_map.get(tgt.cell_id, 0.0) * 2.0
                score += self.calculate_edge_affinity(curr, brg if brg else tgt, orch, relevance_map=relevance_map)
                if brg:
                    score += self.calculate_edge_affinity(brg, tgt, orch, relevance_map=relevance_map) * 0.8
                return (fit_rec, score)

            cand_transitions = list(transitions)
            cand_transitions.sort(key=_trans_score, reverse=True)
            active_transitions = cand_transitions[:MAX_TRANSITIONS_SHOWN]
            trans_by_id = {t[0].cell_id.lower(): t for t in active_transitions}

            # Remaining unvisited relevant nodes (for model foresight)
            remaining_nodes = [
                c for c in shown
                if c.cell_id not in visited_ids and c.cell_id.lower() not in trans_by_id
            ][:10]

            # Format prompt for the step
            path_str = " -> ".join([c.cell_id for c in path])
            valid_lines = []
            for tgt, brg, mode in active_transitions:
                valid_lines.append(f"- {tgt.cell_id} [{mode}]: {_format_cell_signature(tgt)}")
            valid_block = "\n".join(valid_lines)

            rem_block = ""
            if remaining_nodes:
                rem_lines = [f"- {_format_cell_signature(c)}" for c in remaining_nodes]
                rem_block = f"\n\nOther Candidate Nodes in Pool (Goals / Rest of Pipeline):\n" + "\n".join(rem_lines)

            clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
            finish_hint = " (or 'FINISH' if pipeline is complete)" if (not uncovered_indices and (is_terminal_sink(curr, orch) or (clauses and len(path) >= len(clauses)))) else ""
            step_msg = (
                f"Request: {prompt}\n\n"
                f"Current Pipeline: {path_str}\n\n"
                f"Valid Next Transitions from {curr.cell_id}:\n{valid_block}"
                f"{rem_block}\n\n"
                f"Choose the next node ID from 'Valid Next Transitions'{finish_hint}:"
            )

            try:
                raw_step = mm.generate_text(
                    step_msg, max_tokens=128, schema=STEP_SCHEMA, system_prompt=SYSTEM_PROMPT_STEP
                )
                chosen_id = _parse_next_node(raw_step)
                logger.debug("[M8] curr=%s target_cl=%s LLM=%s", curr.cell_id, target_cl_idx, chosen_id)
            except Exception as e:
                logger.debug("[M8] ModelManager generate_text failed: %s", e)
                chosen_id = None

            reg = TypeRegistry.get_instance()
            stop_tokens = {str(t).upper() for t in (reg.get_verification_semantics("stopping_tokens") or ())}
            if chosen_id and chosen_id.upper() in stop_tokens:
                if uncovered_indices:
                    chosen_id = None
                else:
                    break

            matched_trans = trans_by_id.get(chosen_id.lower()) if chosen_id else None
            if matched_trans is None and chosen_id:
                cand_cell = next((c for c in candidates if c.cell_id.lower() == chosen_id.lower()), None)
                if cand_cell:
                    if is_terminal_sink(cand_cell, orch) and uncovered_indices:
                        chosen_id = None
                    elif cand_cell.cell_id not in visited_ids:
                        if self.step_unifies(curr, cand_cell, prev_path=path):
                            matched_trans = (cand_cell, None, "DIRECT")
                        elif self.step_unifies_dag(cand_cell, path, ctx=ctx):
                            matched_trans = (cand_cell, None, "DAG_FRONTIER")
                        else:
                            brg = self.find_bridge(curr, cand_cell, candidates, orch, prev_path=path)
                            if brg and brg.cell_id not in visited_ids:
                                matched_trans = (cand_cell, brg, f"BRIDGED via {brg.cell_id}")

                if matched_trans is None and chosen_id:
                    try:
                        from node_resolver import DynamicNodeResolver
                        res_cell = DynamicNodeResolver.resolve_node(
                            raw_id=chosen_id,
                            prompt=prompt,
                            orchestrator=orch,
                            rag=kwargs.get("rag"),
                        )
                        if res_cell is not None and res_cell.cell_id not in visited_ids and getattr(res_cell, "verified", True):
                            if self.step_unifies(curr, res_cell, prev_path=path):
                                matched_trans = (res_cell, None, "DYNAMIC_DIRECT")
                            elif self.step_unifies_dag(res_cell, path, ctx=ctx):
                                matched_trans = (res_cell, None, "DYNAMIC_DAG_FRONTIER")
                            else:
                                brg = self.find_bridge(curr, res_cell, candidates, orch)
                                if brg and brg.cell_id not in visited_ids:
                                    matched_trans = (res_cell, brg, f"DYNAMIC_BRIDGED via {brg.cell_id}")
                    except Exception:
                        pass

            if matched_trans is None:
                # If current node is a sink or stage 3 and all clauses covered, complete
                if is_terminal_sink(curr, orch) and not uncovered_indices:
                    break
                # Default to top scored transition that is not visited
                for tr in active_transitions:
                    if tr[0].cell_id not in visited_ids:
                        matched_trans = tr
                        break
                if matched_trans is None:
                    break

            target_cell, bridge_cell, _ = matched_trans
            if target_cell.cell_id in visited_ids:
                break
            if bridge_cell:
                path.append(bridge_cell)
                visited_ids.add(bridge_cell.cell_id)
                covered_tokens.update(getattr(bridge_cell, "identity_tokens", bridge_cell.token_set) & prompt_tokens)
            path.append(target_cell)
            visited_ids.add(target_cell.cell_id)
            covered_tokens.update(getattr(target_cell, "identity_tokens", target_cell.token_set) & prompt_tokens)

            if is_terminal_sink(target_cell, orch) and not uncovered_indices:
                # Egress reached and all clauses covered
                break

        # Verification pass: ensure consecutive unification
        if len(path) > 1:
            verified = [path[0]]
            for nxt in path[1:]:
                c_prev = verified[-1]
                if self.step_unifies(c_prev, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                    verified.append(nxt)
                else:
                    brg = self.find_bridge(c_prev, nxt, candidates, orch)
                    if brg:
                        verified.extend([brg, nxt])
                    else:
                        # DAG branch check: can nxt unify with earlier ancestor in verified?
                        anc_ok = False
                        for anc in reversed(verified[:-1]):
                            if self.step_unifies(anc, nxt, prev_path=verified) or self.step_unifies_dag(nxt, verified, ctx=ctx):
                                verified.append(nxt)
                                anc_ok = True
                                break
                        if not anc_ok:
                            try:
                                from node_resolver import DynamicNodeResolver
                                dyn_brg = DynamicNodeResolver.synthesize_adapter(c_prev, nxt, orch, rag=kwargs.get("rag"))
                                if dyn_brg is not None:
                                    can_c_prev = self.step_unifies(c_prev, dyn_brg, prev_path=verified) or self.step_unifies_dag(dyn_brg, verified, ctx=ctx)
                                    can_dyn_nxt = self.step_unifies(dyn_brg, nxt, prev_path=verified + [dyn_brg]) or self.step_unifies_dag(nxt, verified + [dyn_brg], ctx=ctx)
                                    if can_c_prev and can_dyn_nxt:
                                        verified.extend([dyn_brg, nxt])
                                    else:
                                        return _fallback(f"synthesized adapter between {c_prev.cell_id} and {nxt.cell_id} does not unify")
                                else:
                                    return _fallback(f"could not synthesize adapter between {c_prev.cell_id} and {nxt.cell_id}")
                            except Exception as _brg_err:
                                return _fallback(f"adapter synthesis raised {type(_brg_err).__name__}: {_brg_err}")
            path = verified

        if len(path) < 2:
            return _fallback("verified chain shorter than 2 cells")

        self.tag_cells_with_clause_indices(path, prompt, orchestrator=orch)

        try:
            from token_evidence import audit
            aud = audit(path, prompt, orch)
            if aud.get("available") and any(iss.get("kind") == "clause_unserved" for iss in aud.get("issues", [])):
                return _fallback("LLM proposed path has unserved clauses")
            from unification import UnificationGate, ExecutionContext
            from preflight import PreflightLinter
            _gate = UnificationGate(orchestrator=orch)
            _tctx = ExecutionContext(prompt=prompt)
            _ur = _gate.unify_pipeline(path, _tctx, extract_llm_slots=False)
            if _ur.is_bottom():
                return _fallback("LLM proposed path does not unify (bottom)")
            _code = _gate.emit_code(_ur.value, _tctx, extract_llm_slots=False)
            _lint = PreflightLinter.lint(_ur.value, prompt=prompt, code_str=_code)
            if not _lint.is_valid:
                return _fallback(f"LLM proposed path has pre-flight violations: {_lint.violations[0][:100]}")
        except Exception:
            pass

        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return path[:max_allowed]
