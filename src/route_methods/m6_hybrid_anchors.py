"""
src/route_methods/m6_hybrid_anchors.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M6: Flagship Hybrid — Clause Anchors + Endpoint Anchors + Beam Bridging + Pre-Flight Linting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, UnificationGate, ExecutionContext
    from ..preflight import PreflightLinter
    from ..planner import LatticePlanner
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, UnificationGate, ExecutionContext
    from preflight import PreflightLinter
    from planner import LatticePlanner


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

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant" and not c.cell_id.startswith("MACRO_")]
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
                for c, _o in anchors_for_clause(_ev, cl, anchor_pool, relevance_map, on_event=self._trace, clause_idx=idx):
                    c_inst = c.clone() if hasattr(c, "clone") else c
                    c_inst.matched_clause_idx = idx
                    c_inst.clause_literals = cl_literals
                    clause_cells.append(c_inst)
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

        # 5. Bridge gaps between waypoints using lattice search
        bridged_path: List[Cell] = [waypoints[0]]
        for nxt in waypoints[1:]:
            curr = bridged_path[-1]
            if curr.cell_id == nxt.cell_id:
                continue
            if self.step_unifies(curr, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                bridged_path.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch, prev_path=bridged_path)
                if bridge:
                    bridged_path.append(bridge)
                    bridged_path.append(nxt)
                else:
                    connected = False
                    for ancestor in reversed(bridged_path[:-1]):
                        if self.step_unifies(ancestor, nxt, prev_path=bridged_path) or self.step_unifies_dag(nxt, bridged_path, ctx=ctx):
                            bridged_path.append(nxt)
                            connected = True
                            break
                        anc_bridge = self.find_bridge(ancestor, nxt, candidates, orch, prev_path=bridged_path)
                        if anc_bridge:
                            bridged_path.extend([anc_bridge, nxt])
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
                                    reason="no direct unification, no DAG-scope unification, no bridge or producer from any ancestor",
                                    ancestors=[a.cell_id for a in bridged_path], needs=[], ancestor_outputs=[])

        # 6. Pre-flight verification & path return
        num_clauses = len(clauses) if clauses else 1
        max_allowed = max(32, num_clauses * 4 + 4, max_transforms + 8)

        if bridged_path and len(bridged_path) >= 2:
            return bridged_path[:max_allowed]

        # If waypoints failed to assemble, fallback to Trellis only if allow_fallback is enabled
        try:
            from config import settings
            allow_fb = settings.allow_fallback
        except Exception:
            try:
                from ..config import settings
                allow_fb = settings.allow_fallback
            except Exception:
                allow_fb = False

        if allow_fb and orch:
            planner = LatticePlanner(orchestrator=orch)
            m0_path = planner.plan(
                prompt=prompt,
                tunnel=tunnel,
                relevance_map=relevance_map,
                start_sig=start_sig,
                goal_sig=goal_sig,
                max_transforms=max_transforms
            )
            if m0_path:
                return m0_path

        return []
