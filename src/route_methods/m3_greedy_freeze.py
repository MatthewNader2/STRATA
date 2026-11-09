"""
src/route_methods/m3_greedy_freeze.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M3: Forward Greedy Beam with Frozen Committed Prefixes.
Greedily commits the best next morphism at each step without backtracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS, is_terminal_sink

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
    from ..log_config import get_logger
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext
    from log_config import get_logger

logger = get_logger("m3_greedy_freeze")


class M3GreedyFreezeRouteMethod(RouteMethod):
    """
    RouteMethod M3: Forward Greedy Freeze.
    Iteratively extends the pipeline forward, permanently committing the best
    valid continuation cell at each step without backtracking across earlier stages.
    """
    name: str = "m3_greedy_freeze"

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
        ]
        if not candidates:
            return [tunnel[0]]

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)

        # 1. Select initial entry cell C_0
        stage1_cands = [c for c in candidates if self.is_ingress_node(c, orch)]
        entry_pool = stage1_cands if stage1_cands else candidates[:5]

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        first_clause = clauses[0] if clauses else prompt
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None
        best_entry = max(
            entry_pool,
            key=lambda c: (self.file_compat(c, src_lit), self.clause_fit(first_clause, c), relevance_map.get(c.cell_id, 0.0)),
        )

        _ev = self._evidence()
        prompt_tokens = _ev.clause_tokens(prompt) if _ev is not None else prompt_tokens

        # Precompute clause anchors using canonical token evidence rules
        clause_anchors: Dict[int, List[Cell]] = {}
        if _ev is not None and clauses:
            try:
                try:
                    from token_evidence import anchors_for_clause
                except (ImportError, ValueError):
                    from ..token_evidence import anchors_for_clause
                for j, cl in enumerate(clauses):
                    stage_pool = [c for c in candidates if (self.is_ingress_node(c, orch) if (j == 0 and len(clauses) > 1) else not self.is_ingress_node(c, orch))] or candidates
                    acc = anchors_for_clause(_ev, cl, stage_pool, relevance_map, clause_idx=j)
                    clause_anchors[j] = [c for c, _ in acc]
            except Exception as _e:
                logger.debug(f"[M3] Error pre-computing clause anchors: {_e}")

        def _cand_clause(c: Cell) -> Optional[int]:
            idx = getattr(c, "matched_clause_idx", None)
            if idx is not None:
                return idx
            for j, anchors in clause_anchors.items():
                if any(a.cell_id == c.cell_id for a in anchors):
                    return j
            if _ev is not None and clauses:
                best_j, best_rank = None, (-1.0, -1.0, -1.0)
                for j, cl in enumerate(clauses):
                    cl_t = _ev.clause_tokens(cl)
                    if not cl_t:
                        continue
                    ov = _ev.identity_overlap(cl_t, c)
                    prim = clause_anchors.get(j, [None])[0] if clause_anchors.get(j) else None
                    if prim is not None:
                        prim_ov = _ev.identity_overlap(cl_t, prim)
                        if ov["rank"] < prim_ov["rank"] and ov["f"] < prim_ov["f"]:
                            continue
                    if ov["hit"] > 0.0 and ov["rank"] > best_rank:
                        best_j, best_rank = j, ov["rank"]
                return best_j
            return None

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        covered_tokens: Set[str] = set(getattr(best_entry, "identity_tokens", best_entry.token_set) & prompt_tokens)

        num_clauses = len(clauses) if clauses else 1
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)

        # 2. Greedily commit steps forward
        for step in range(max_steps):
            curr = committed_path[-1]

            # Terminal condition: terminal evaluator reached after covering clauses
            if self.is_sink_node(curr, orch) and len(committed_path) > 1:
                if step >= max(1, num_clauses - 1) or not (prompt_tokens - covered_tokens):
                    break

            covered_indices = {
                _cand_clause(c) for c in committed_path if _cand_clause(c) is not None
            }
            if _ev is not None:
                for j, cl in enumerate(clauses):
                    cl_toks = _ev.clause_tokens(cl)
                    if cl_toks and (cl_toks & covered_tokens) and len(cl_toks & covered_tokens) / len(cl_toks) >= 0.5:
                        covered_indices.add(j)
            uncovered_indices = [j for j in range(len(clauses)) if j not in covered_indices]
            target_cl_idx = min(uncovered_indices) if uncovered_indices else (num_clauses - 1)
            target_cl = clauses[target_cl_idx] if target_cl_idx < num_clauses else prompt
            target_toks = _ev.clause_tokens(target_cl or prompt) if _ev is not None else set()
            _requested_sink = self.requested_sink(prompt, candidates)

            # Find all valid next candidates that unify with curr or DAG scope
            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids and (self.step_unifies(curr, c, prev_path=committed_path) or self.step_unifies_dag(c, committed_path, ctx=ctx))
            ]

            target_anchors = clause_anchors.get(target_cl_idx, [])
            if not target_anchors:
                target_anchors = [c for c in candidates if c.cell_id not in visited_ids and _cand_clause(c) == target_cl_idx]

            has_target_in_valid = any(
                (_cand_clause(c) == target_cl_idx or any(a.cell_id == c.cell_id for a in target_anchors))
                for c in valid_next
            )
            if not has_target_in_valid or not any(any(a.cell_id == c.cell_id for a in target_anchors) for c in valid_next):
                for target_cand in target_anchors:
                    if target_cand.cell_id in visited_ids:
                        continue
                    prod = self._find_type_producer(target_cand, committed_path, candidates, clauses, orch)
                    if prod is not None and prod.cell_id not in visited_ids and prod not in valid_next:
                        valid_next.append(prod)
                    bridge = self.find_bridge(curr, target_cand, candidates, orch, prev_path=committed_path)
                    if not bridge and orch and hasattr(orch, "loaded_cells"):
                        bridge = self.find_bridge(curr, target_cand, list(orch.loaded_cells.values()), orch, prev_path=committed_path)
                    if bridge is not None and bridge.cell_id not in visited_ids and bridge not in valid_next:
                        if not hasattr(bridge, "matched_clause_idx") or bridge.matched_clause_idx is None:
                            bridge.matched_clause_idx = target_cl_idx
                        valid_next.append(bridge)

            if not valid_next:
                break

            # Greedy key, lexicographic (no magic weights):
            # unrequested egress/inversion check > target clause alignment > linear handle continuity >
            # identity evidence for target clause > newly explained prompt tokens > stage progression > retrieval + edge affinity.
            def _score_cand(cand: Cell):
                rel = relevance_map.get(cand.cell_id, 0.0)
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                c_toks = getattr(cand, "identity_tokens", cand.token_set)
                new_mass = _ev.mass((c_toks & prompt_tokens) - covered_tokens) if _ev is not None else 0.0
                is_cand_sink = self.is_sink_node(cand, orch)
                is_curr_sink = self.is_sink_node(curr, orch)
                is_cand_ing = self.is_ingress_node(cand, orch)
                is_curr_ing = self.is_ingress_node(curr, orch)
                if is_curr_ing and not is_cand_ing:
                    progress = 2
                elif not is_curr_sink and is_cand_sink:
                    progress = 2
                elif is_cand_ing and not is_curr_ing:
                    progress = 0
                else:
                    curr_stage = getattr(curr, "stage", 1) or 1
                    cand_stage = getattr(cand, "stage", 2) or 2
                    try:
                        progress = 2 if int(cand_stage) > int(curr_stage) else (1 if int(cand_stage) == int(curr_stage) else 0)
                    except (ValueError, TypeError):
                        progress = 1
                unrequested_egress = (
                    is_cand_sink and self._is_path_consumer(cand)
                    and not (dst_lit is not None and self.file_compat(cand, dst_lit) > 0)
                ) or (_requested_sink is None and is_cand_sink and not is_curr_sink and self.clause_fit(target_cl, cand)[0] <= 0.0)

                cand_cl = _cand_clause(cand)
                is_inversion = bool(cand_cl is not None and uncovered_indices and cand_cl < min(uncovered_indices))
                is_target_cl = bool(
                    (cand_cl is not None and cand_cl == target_cl_idx)
                    or any(a.cell_id == cand.cell_id for a in target_anchors)
                )

                # Linear handle continuity: +1 if consumes active handle, -1 if competing/abandoning
                handle_score = self.linear_handle_score(cand, committed_path)

                if _ev is not None and target_toks:
                    ov = _ev.identity_overlap(target_toks, cand)
                    fit = ov["rank"]
                else:
                    fit = self.clause_fit(target_cl, cand)

                # Lookahead reachability: ensure candidate can transition/bridge to target anchors
                def _can_reach_target(c: Cell) -> bool:
                    if not target_anchors or any(a.cell_id == c.cell_id for a in target_anchors):
                        return True
                    simulated_path = committed_path + [c]
                    for ta in target_anchors:
                        if self.step_unifies(c, ta, prev_path=simulated_path) or self.step_unifies_dag(ta, simulated_path, ctx=ctx):
                            return True
                        if self.find_bridge(c, ta, candidates, orch, prev_path=simulated_path) is not None:
                            return True
                    return False

                reaches_target = 1 if _can_reach_target(cand) else 0

                return (
                    0 if unrequested_egress else (0 if is_inversion else 1),
                    1 if is_target_cl else 0,
                    reaches_target,
                    handle_score,
                    fit,
                    new_mass,
                    progress,
                    rel + aff
                )

            valid_forward = [
                c for c in valid_next
                if not (_cand_clause(c) is not None and uncovered_indices and _cand_clause(c) < min(uncovered_indices))
            ]
            exact_target = [
                c for c in valid_forward
                if _cand_clause(c) == target_cl_idx or any(a.cell_id == c.cell_id for a in target_anchors)
            ]
            target_forward = exact_target if exact_target else [
                c for c in valid_forward if _cand_clause(c) is None
            ]
            active_next = target_forward if target_forward else valid_forward
            if not active_next:
                active_next = valid_next
            active_next.sort(key=_score_cand, reverse=True)
            best_next = active_next[0]

            # Commit next cell (freeze)
            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(getattr(best_next, "identity_tokens", best_next.token_set) & prompt_tokens)

            # Goal test shared with preflight: once every prompt clause is served by the path the plan is complete
            if step >= num_clauses - 1:
                try:
                    from token_evidence import audit as _audit
                except (ImportError, ValueError):
                    try:
                        from ..token_evidence import audit as _audit
                    except (ImportError, ValueError):
                        _audit = None
                if _audit is not None:
                    _res = _audit(committed_path, prompt, orch)
                    if _res.get("available") and not any(i["kind"] == "clause_unserved" for i in _res.get("issues", [])):
                        self._trace("m3_goal_reached", step=step, path_len=len(committed_path))
                        break

            # Check if all prompt content tokens are covered and cell is a terminal sink
            if (prompt_tokens - covered_tokens) == set() and self.is_sink_node(best_next, orch):
                break

        self.tag_cells_with_clause_indices(committed_path, prompt, orchestrator=orch)

        # 1. Coverage audit check
        try:
            try:
                from token_evidence import audit as _audit
            except (ImportError, ValueError):
                try:
                    from ..token_evidence import audit as _audit
                except (ImportError, ValueError):
                    _audit = None
            if _audit is not None:
                _cov = _audit(committed_path, prompt, orch)
                if _cov.get("available") and any(i["kind"] == "clause_unserved" for i in _cov.get("issues", [])):
                    unserved_reasons = [i.get("message", "unserved clause") for i in _cov.get("issues", []) if i.get("kind") == "clause_unserved"]
                    return self._delegate_to_m1(
                        f"M3 forward greedy path has unserved clauses: {'; '.join(unserved_reasons[:2])}",
                        prompt, tunnel, relevance_map, orch=orch, ctx=ctx,
                        start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms, **kwargs
                    )
        except Exception as _cov_exc:
            logger.debug("[M3] Coverage audit check skipped: %s", _cov_exc)

        # 2. Monadic unification and Preflight lint check
        try:
            try:
                from unification import UnificationGate, ExecutionContext
                from preflight import PreflightLinter
            except (ImportError, ValueError):
                try:
                    from ..unification import UnificationGate, ExecutionContext
                    from ..preflight import PreflightLinter
                except (ImportError, ValueError):
                    UnificationGate = None
                    ExecutionContext = None
                    PreflightLinter = None

            if UnificationGate is not None and PreflightLinter is not None:
                _gate = UnificationGate(orchestrator=orch)
                _tctx = ctx if ctx is not None else ExecutionContext(prompt=prompt)
                _ur = _gate.unify_pipeline(committed_path, _tctx, extract_llm_slots=False)
                if _ur.is_bottom() or not getattr(_ur, "value", None):
                    return self._delegate_to_m1(
                        "M3 path does not unify (bottom)",
                        prompt, tunnel, relevance_map, orch=orch, ctx=ctx,
                        start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms, **kwargs
                    )
                _code = _gate.emit_code(_ur.value, _tctx, extract_llm_slots=False)
                _lint = PreflightLinter.lint(_ur.value, prompt=prompt, code_str=_code, orchestrator=orch)
                if not _lint.is_valid:
                    return self._delegate_to_m1(
                        f"M3 path failed pre-flight lint: {'; '.join(_lint.violations[:2])}",
                        prompt, tunnel, relevance_map, orch=orch, ctx=ctx,
                        start_sig=start_sig, goal_sig=goal_sig, max_transforms=max_transforms, **kwargs
                    )
        except Exception as _chk_exc:
            logger.debug("[M3] Pre-flight verification check skipped: %s", _chk_exc)

        return committed_path
