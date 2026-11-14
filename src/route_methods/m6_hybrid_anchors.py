"""
src/route_methods/m6_hybrid_anchors.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M6: Flagship Hybrid — Clause Anchors + Endpoint Anchors + Beam Bridging + Pre-Flight Linting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, is_macro_cell
from log_config import get_logger

logger = get_logger("route_methods.m6")

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, UnificationGate, ExecutionContext
    from ..preflight import PreflightLinter
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, UnificationGate, ExecutionContext
    from preflight import PreflightLinter


class M6HybridAnchorsRouteMethod(RouteMethod):
    """
    RouteMethod M6: Flagship Hybrid.
    Integrates clause-level waypoint anchoring, source/sink endpoint binding,
    Viterbi beam bridging, and static pre-flight linting into a unified routing strategy.
    """
    name: str = "m6_hybrid_anchors"

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

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant" and not is_macro_cell(c)]
        if not candidates:
            return [tunnel[0]]

        # 1. Extract universal literals (files, identifiers, etc.)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)

        # 2. Identify Source & Sink Endpoints
        source_cell: Optional[Cell] = None
        sink_cell: Optional[Cell] = None

        ingress_cands = [c for c in candidates if self.is_ingress_node(c, orch)]
        sink_cands = [c for c in candidates if self.is_sink_node(c, orch)]
        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        first_clause = clauses[0] if clauses else (prompt or "")
        last_clause = clauses[-1] if clauses else (prompt or "")
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None

        if ingress_cands:
            source_cell = max(
                ingress_cands,
                key=lambda c: (self.file_compat(c, src_lit), self.clause_fit(first_clause, c), relevance_map.get(c.cell_id, 0.0)),
            )
        # A sink exists only if a clause is better explained by a sink than by every non-sink cell
        # (not because the prompt contains a word such as "write"), or an explicit destination file
        # has a format-compatible writer.
        sink_cell = self.requested_sink(prompt, candidates)
        if sink_cell is None and dst_lit is not None:
            destination_sinks = [c for c in sink_cands if self.file_compat(c, dst_lit) > 0]
            if destination_sinks:
                sink_cell = max(destination_sinks, key=lambda c: (self.clause_fit(last_clause, c), relevance_map.get(c.cell_id, 0.0)))
        self._trace("m6_endpoints", source=source_cell.cell_id if source_cell else None,
                    sink=sink_cell.cell_id if sink_cell else None)

        # 3. Identify Clause-Level Waypoint Anchors
        clause_anchors: List[Cell] = []
        from token_evidence import anchors_for_clause
        _ev = self._evidence()
        anchor_pool = [
            c for c in candidates
            if bool(getattr(c, "slots", None))
            or not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator"
                    or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
        ]

        for idx, cl in enumerate(clauses):
            cl_literals = [
                v for _, k, v in ExecutionContext._extract_universal_literals(cl)
                if k in ("identifier", "quoted_str")
            ]
            clause_cells: List[Cell] = []
            if _ev is not None:
                locked_tau = 0.7
                for c, _o in anchors_for_clause(_ev, cl, anchor_pool, relevance_map, on_event=self._trace, clause_idx=idx):
                    conf = max(
                        float(_o.get("name_precision", 0.0)),
                        float(_o.get("precision", 0.0)),
                        float(_o.get("f", 0.0)),
                    )
                    if conf <= locked_tau and len(anchor_pool) > 1:
                        self._trace("m6_anchor_unlocked_low_precision", cell=c.cell_id, clause_idx=idx, confidence=round(conf, 3))
                        continue
                    c_inst = c.clone() if hasattr(c, "clone") else c
                    c_inst.matched_clause_idx = idx
                    c_inst.clause_literals = cl_literals
                    clause_cells.append(c_inst)
                    self._trace("m6_anchor_locked", cell=c.cell_id, clause_idx=idx, confidence=round(conf, 3))
            if not clause_cells and _ev is not None and _ev.clause_tokens(cl):
                self._trace("clause_without_anchor", clause_idx=idx, clause=cl,
                            reason="no candidate cell carries identity evidence for this clause")

            if len(clause_cells) >= 2:
                from functools import cmp_to_key
                def _reach_cmp(a: Cell, b: Cell) -> int:
                    if a.cell_id == b.cell_id:
                        return 0
                    aff_ab = self.calculate_edge_affinity(a, b, orch)
                    aff_ba = self.calculate_edge_affinity(b, a, orch)
                    if aff_ab > aff_ba:
                        return -1
                    if aff_ba > aff_ab:
                        return 1
                    return 0
                clause_cells = sorted(clause_cells, key=cmp_to_key(_reach_cmp))

            for c in clause_cells:
                if not clause_anchors or clause_anchors[-1].cell_id != c.cell_id or getattr(clause_anchors[-1], "clause_literals", None) != getattr(c, "clause_literals", None):
                    clause_anchors.append(c)

        # 4. Assemble Waypoints
        waypoints: List[Cell] = []
        if source_cell:
            waypoints.append(source_cell)

        # If sink_cell is already among clause_anchors, enrich sink_cell with ca's tagging
        if sink_cell:
            matching_ca = next((ca for ca in clause_anchors if ca.cell_id == sink_cell.cell_id), None)
            if matching_ca is not None:
                if getattr(sink_cell, "matched_clause_idx", None) is None:
                    sink_cell.matched_clause_idx = getattr(matching_ca, "matched_clause_idx", None)
                if not getattr(sink_cell, "clause_literals", None):
                    sink_cell.clause_literals = getattr(matching_ca, "clause_literals", None)

        for ca in clause_anchors:
            if not waypoints or waypoints[-1].cell_id != ca.cell_id or getattr(waypoints[-1], "clause_literals", None) != getattr(ca, "clause_literals", None):
                waypoints.append(ca)

        if sink_cell and not any(w.cell_id == sink_cell.cell_id for w in waypoints):
            if getattr(sink_cell, "matched_clause_idx", None) is None and clauses:
                sink_cell.matched_clause_idx = len(clauses) - 1
            waypoints.append(sink_cell)

        if not waypoints:
            waypoints = [candidates[0]]

        # 5. Viterbi beam bridging across multi-hop gaps between locked waypoints
        def _viterbi_beam_bridge(src: Cell, dst: Cell, base_scope: List[Cell], max_hops: int = 4, beam_width: int = 4) -> Optional[List[Cell]]:
            used_ids = {c.cell_id for c in base_scope} | {dst.cell_id}
            # Direct 1-hop bridge check first
            b1 = self.find_bridge(src, dst, candidates, orch, prev_path=base_scope)
            if b1 is not None and b1.cell_id not in used_ids:
                return [b1]
            # Multi-hop Viterbi beam search maximizing cumulative (edge_affinity + relevance)
            beam: List[Tuple[float, List[Cell]]] = [(0.0, [])]
            best_complete: Optional[Tuple[float, List[Cell]]] = None
            pool = [c for c in candidates if c.cell_id not in used_ids]
            for _hop in range(max_hops):
                next_beam: List[Tuple[float, List[Cell]]] = []
                for sc, mids in beam:
                    tail = mids[-1] if mids else src
                    scope = base_scope + mids
                    mids_ids = {m.cell_id for m in mids}
                    for cand in pool:
                        if cand.cell_id in mids_ids:
                            continue
                        if not (self.step_unifies(tail, cand, prev_path=scope) or self.step_unifies_dag(cand, scope, ctx=ctx)):
                            continue
                        step_sc = (
                            self.calculate_edge_affinity(tail, cand, orch, relevance_map=relevance_map)
                            + relevance_map.get(cand.cell_id, 0.0)
                        )
                        new_mids = mids + [cand]
                        new_sc = sc + step_sc
                        new_scope = base_scope + new_mids
                        if self.step_unifies(cand, dst, prev_path=new_scope) or self.step_unifies_dag(dst, new_scope, ctx=ctx):
                            tot_sc = new_sc + self.calculate_edge_affinity(cand, dst, orch, relevance_map=relevance_map)
                            if best_complete is None or tot_sc > best_complete[0]:
                                best_complete = (tot_sc, new_mids)
                        else:
                            next_beam.append((new_sc, new_mids))
                if best_complete is not None:
                    return best_complete[1]
                next_beam.sort(key=lambda x: x[0], reverse=True)
                beam = next_beam[:beam_width]
                if not beam:
                    break
            if orch and hasattr(orch, "find_shortest_bridge_path"):
                shortest = orch.find_shortest_bridge_path(src, dst, max_hops=max_hops, candidate_pool=candidates)
                if shortest:
                    return shortest
            return None

        bridged_path: List[Cell] = [waypoints[0]]
        for nxt in waypoints[1:]:
            curr = bridged_path[-1]
            if curr.cell_id == nxt.cell_id:
                continue
            if self.step_unifies(curr, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                bridged_path.append(nxt)
            else:
                beam_mids = _viterbi_beam_bridge(curr, nxt, bridged_path, max_hops=4, beam_width=4)
                if beam_mids:
                    for bm in beam_mids:
                        if getattr(bm, "matched_clause_idx", None) is None:
                            bm.matched_clause_idx = getattr(nxt, "matched_clause_idx", None)
                    bridged_path.extend(beam_mids)
                    bridged_path.append(nxt)
                    self._trace("m6_beam_bridged", from_cell=curr.cell_id, to_cell=nxt.cell_id,
                                via=[b.cell_id for b in beam_mids])
                else:
                    connected = False
                    for ancestor in reversed(bridged_path[:-1]):
                        if self.step_unifies(ancestor, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                            bridged_path.append(nxt)
                            connected = True
                            break
                        anc_mids = _viterbi_beam_bridge(ancestor, nxt, bridged_path, max_hops=3, beam_width=3)
                        if anc_mids:
                            bridged_path.extend(anc_mids + [nxt])
                            connected = True
                            break
                    if not connected:
                        producer = self._find_type_producer(nxt, bridged_path, candidates, clauses, orch)
                        if producer is not None:
                            self._trace("producer_inserted", cell=producer.cell_id, for_anchor=nxt.cell_id,
                                        clause_idx=getattr(nxt, "matched_clause_idx", None),
                                        reason="anchor input type unmet by scope; producer has same-clause identity evidence and unifies from scope")
                            bridged_path.extend([producer, nxt])
                            connected = True
                    if not connected:
                        self._trace("anchor_dropped", cell=nxt.cell_id, clause_idx=getattr(nxt, "matched_clause_idx", None),
                                    reason="no direct unification, no DAG-scope unification, no Viterbi beam bridge",
                                    ancestors=[a.cell_id for a in bridged_path], needs=[], ancestor_outputs=[])

        # 6. Pre-flight verification via UnificationGate & PreflightLinter
        num_clauses = len(clauses) if clauses else 1
        max_allowed = max(32, num_clauses * 4 + 4, max_transforms + 8)
        bridged_path = bridged_path[:max_allowed]

        if not bridged_path:
            self.fallback_reason = "M6 hybrid anchor assembly produced empty path"
            self._trace("method_failure", requested=type(self).__name__, reason=self.fallback_reason)
            return []

        try:
            _gate = UnificationGate(orchestrator=orch)
            _tctx = ctx if ctx is not None else ExecutionContext(prompt=prompt)
            _ur = _gate.unify_pipeline(bridged_path, _tctx, extract_llm_slots=False)
            if _ur.is_bottom() or not getattr(_ur, "value", None):
                self.fallback_reason = "M6 hybrid path failed UnificationGate (bottom)"
                self._trace("method_failure", requested=type(self).__name__, reason=self.fallback_reason)
                return []
            _code = _gate.emit_code(_ur.value, _tctx, extract_llm_slots=False)
            _lint = PreflightLinter.lint(_ur.value, prompt=prompt, code_str=_code, orchestrator=orch)
            if not _lint.is_valid:
                self.fallback_reason = f"M6 hybrid path failed PreflightLinter: {'; '.join(_lint.violations[:2])}"
                self._trace("method_failure", requested=type(self).__name__, reason=self.fallback_reason)
                return []
        except Exception as _chk_exc:
            logger.debug("[M6] Verification check exception: %s", _chk_exc)

        return bridged_path
