"""
src/route_methods/m1_clause_anchor.py - Neuro-Symbolic Topological Lattice (NSTL)
M1: Clause-Anchored Insertion.
Anchors key morphisms for each linguistic clause in the prompt and bridges type gaps.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger
from .base import RouteMethod, STOPWORDS

logger = get_logger("m1_clause_anchor")

try:
    from ..lattice import Cell, LatticeOrchestrator, TypeRegistry
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator, TypeRegistry
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext


def _clause_has_egress_intent(clause: str) -> bool:
    """Language-level materialization intent for a clause: declared egress
    verbs (planner.EGRESS_INTENT_TOKENS) when not an in-memory variable target."""
    try:
        from ..planner import EGRESS_INTENT_TOKENS
    except (ImportError, ValueError):
        from planner import EGRESS_INTENT_TOKENS
    target = ExecutionContext._extract_target_sink(clause)
    toks = set(CellTokenizer.tokenize_prompt(clause.lower()))
    return bool(toks & EGRESS_INTENT_TOKENS) and not target


class M1ClauseAnchorRouteMethod(RouteMethod):
    """
    RouteMethod M1: Clause-Anchored Insertion.
    Segments user prompt into sequential intent clauses, extracts high-confidence anchor cells
    for each clause, and synthesizes/routes type-valid bridges between consecutive anchors.
    """
    name: str = "m1_clause_anchor"

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

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        anchors: List[Cell] = []
        clause_alts: Dict[int, List[Cell]] = {}

        # 1. Identify anchor cells for each clause via semantic set-covering and stage awareness
        for idx, cl in enumerate(clauses):
            cl_tokens = CellTokenizer.tokenize_prompt(cl) - STOPWORDS
            if not cl_tokens:
                continue

            cl_literals = [
                v for _, k, v in ExecutionContext._extract_universal_literals(cl)
                if k in ("identifier", "quoted_str")
            ]

            if idx == 0 and len(clauses) > 1:
                pool = [c for c in candidates if self.is_ingress_node(c, orch)] or candidates
            elif idx == len(clauses) - 1 and len(clauses) > 1 and _clause_has_egress_intent(cl):
                pool = [c for c in candidates if self.is_sink_node(c, orch)] or candidates
                _, dest_lits = self.extract_file_literals(prompt, ctx=ctx)
                dest_lit = dest_lits[0] if dest_lits else None
                if dest_lit:
                    compat_cands = [c for c in pool if self.file_compat(c, dest_lit) > 0]
                    if compat_cands:
                        max_c = max(self.file_compat(c, dest_lit) for c in compat_cands)
                        pool = [c for c in compat_cands if self.file_compat(c, dest_lit) >= max_c]
            else:
                pool = [c for c in candidates if not self.is_ingress_node(c, orch)] or candidates

            pool = [
                c for c in pool
                if bool(getattr(c, "slots", None))
                or not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
            ]

            if idx == 0 and len(clauses) > 1:
                src_lits, _ = self.extract_file_literals(prompt, ctx=ctx)
                src_lit = src_lits[0] if src_lits else None
                if src_lit:
                    compat_cands = [c for c in pool if self.file_compat(c, src_lit) > 0]
                    if compat_cands:
                        max_c = max(self.file_compat(c, src_lit) for c in compat_cands)
                        pool = [c for c in compat_cands if self.file_compat(c, src_lit) >= max_c]
                downstream_cands = [c for c in candidates if not self.is_ingress_node(c, orch)]
                from lattice import TypeRegistry
                reg = TypeRegistry.get_instance()

                _reach_cache: Dict[str, int] = {}

                def _downstream_reach(source_cell: Cell) -> int:
                    if not self.is_ingress_node(source_cell, orch):
                        return 0
                    cid = source_cell.cell_id
                    if cid in _reach_cache:
                        return _reach_cache[cid]
                    cnt = 0
                    s_out = getattr(source_cell, "primary_output", None) or (
                        list(source_cell.outputs.values())[0] if source_cell.outputs else None
                    )
                    s_tname = str(getattr(getattr(s_out, "signature", s_out), "type_name", "")).lower() if s_out else ""
                    if s_tname:
                        for dc in downstream_cands:
                            for d_in in dc.inputs.values():
                                d_tname = str(getattr(getattr(d_in, "signature", d_in), "type_name", "")).lower()
                                if (
                                    s_tname == d_tname
                                    or reg.is_subtype(s_tname, d_tname)
                                    or (reg.is_composite_carrier(s_tname) and (reg.is_composite_carrier(d_tname) or reg.get_abstract_carrier(s_tname) == reg.get_abstract_carrier(d_tname)))
                                ):
                                    cnt += 1
                                    break
                    _reach_cache[cid] = cnt
                    return cnt

                pool.sort(
                    key=lambda c: (
                        self.file_compat(c, src_lit),
                        _downstream_reach(c),
                        self.clause_fit(cl, c),
                        relevance_map.get(c.cell_id, 0.0),
                    ),
                    reverse=True,
                )
            else:
                _downstream_reach = None
                src_lit = None

            # Evidence-ranked anchor selection (corpus-derived, no vocabulary).
            # Every pool cell is scored by identity F (precision x recall of its
            # operation tokens against the clause, token_evidence.py). The best
            # cell is the clause's primary anchor. A clause is one operation (the
            # segmenter already splits on conjunctions/sequencing), so a further
            # anchor in the same clause is accepted ONLY when it explains
            # residual clause tokens at least as well as the primary did.
            from token_evidence import get_evidence, anchors_for_clause
            ev = get_evidence(orch)
            clause_cells: List[Cell] = []
            if ev is not None:
                cl_toks_all = CellTokenizer.tokenize_prompt(cl)
                alt_scored = []
                for c in pool:
                    o = ev.identity_overlap(cl_toks_all, c)
                    if o["hit"] > 0.0:
                        alt_scored.append((o["rank"], relevance_map.get(c.cell_id, 0.0), c))
                alt_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                clause_alts[idx] = []
                for _, _, c in alt_scored:
                    c_inst = c.clone() if hasattr(c, "clone") else c
                    c_inst.matched_clause_idx = idx
                    c_inst.clause_literals = cl_literals
                    clause_alts[idx].append(c_inst)

                for c, _o in anchors_for_clause(ev, cl, pool, relevance_map, on_event=self._trace, clause_idx=idx):
                    c_inst = c.clone() if hasattr(c, "clone") else c
                    c_inst.matched_clause_idx = idx
                    c_inst.clause_literals = cl_literals
                    clause_cells.append(c_inst)

                if len(clause_cells) > 1:
                    from unification import is_edge_valid
                    ordered_clause_cells: List[Cell] = []
                    for c in clause_cells:
                        inserted = False
                        for i, other in enumerate(ordered_clause_cells):
                            if any(is_edge_valid(out, inp, c, other) for out in c.outputs.values() for inp in other.inputs.values()):
                                ordered_clause_cells.insert(i, c)
                                inserted = True
                                break
                        if not inserted:
                            ordered_clause_cells.append(c)
                    clause_cells = ordered_clause_cells
                if not clause_cells and pool:
                    # No lattice cell carries identity evidence for this clause:
                    # fall back to the retrieval score (maximizing reachable downstream anchors for ingress) and SAY so.
                    if idx == 0 and _downstream_reach is not None:
                        best = max(
                            pool,
                            key=lambda c: (
                                self.file_compat(c, src_lit),
                                _downstream_reach(c),
                                self.clause_fit(cl, c),
                                relevance_map.get(c.cell_id, 0.0),
                            ),
                        )
                    else:
                        best = max(pool, key=lambda c: relevance_map.get(c.cell_id, 0.0))
                    if relevance_map.get(best.cell_id, 0.0) > 0.0:
                        c_inst = best.clone() if hasattr(best, "clone") else best
                        c_inst.matched_clause_idx = idx
                        c_inst.clause_literals = cl_literals
                        clause_cells.append(c_inst)
                        self._trace("anchor_fallback_retrieval_only", clause_idx=idx,
                                    cell=best.cell_id,
                                    relevance=round(relevance_map.get(best.cell_id, 0.0), 3))
                    else:
                        self._trace("clause_without_anchor", clause_idx=idx, clause=cl,
                                    reason="no identity evidence and no retrieval score")

            if len(clause_cells) >= 2:
                from unification import is_edge_valid
                nodes = list(clause_cells)
                in_degree: Dict[str, int] = {c.cell_id: 0 for c in nodes}
                adj: Dict[str, List[str]] = {c.cell_id: [] for c in nodes}

                for i, a in enumerate(nodes):
                    for j, b in enumerate(nodes):
                        if i == j:
                            continue
                        a_feeds_b = any(
                            getattr(inp, "required", True) and is_edge_valid(out, inp, a, b)
                            for out in a.outputs.values()
                            for inp in b.inputs.values()
                        )
                        b_feeds_a = any(
                            getattr(inp, "required", True) and is_edge_valid(out, inp, b, a)
                            for out in b.outputs.values()
                            for inp in a.inputs.values()
                        )
                        if a_feeds_b and not b_feeds_a:
                            adj[a.cell_id].append(b.cell_id)
                            in_degree[b.cell_id] += 1

                def _clause_key(c: Cell) -> int:
                    idx_val = getattr(c, "matched_clause_idx", None)
                    return idx_val if idx_val is not None else 0

                cell_by_id = {c.cell_id: c for c in nodes}
                ready = [c for c in nodes if in_degree[c.cell_id] == 0]
                ready.sort(key=_clause_key)

                ordered = []
                while ready:
                    curr = ready.pop(0)
                    ordered.append(curr)
                    for nxt_id in adj[curr.cell_id]:
                        in_degree[nxt_id] -= 1
                        if in_degree[nxt_id] == 0:
                            ready.append(cell_by_id[nxt_id])
                            ready.sort(key=_clause_key)

                if len(ordered) < len(nodes):
                    rem = [c for c in nodes if c not in ordered]
                    rem.sort(key=_clause_key)
                    ordered.extend(rem)

                clause_cells = ordered

            for c in clause_cells:
                if not anchors or anchors[-1].cell_id != c.cell_id or getattr(anchors[-1], "clause_literals", None) != getattr(c, "clause_literals", None):
                    anchors.append(c)

        if not anchors:
            return [candidates[0]]

        # 2. Bridge gaps between consecutive anchors
        routed_path: List[Cell] = [anchors[0]]

        for next_anchor in anchors[1:]:
            curr_cell = routed_path[-1]
            connected = False
            if self.step_unifies(curr_cell, next_anchor, prev_path=routed_path):
                self._trace("anchor_connected", cell=next_anchor.cell_id, via="direct", after=curr_cell.cell_id)
                routed_path.append(next_anchor)
                connected = True
            else:
                producer = None
                if not self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                    # Type-directed repair: check if a candidate producer on the current clause bridges to next_anchor
                    producer = self._find_type_producer(next_anchor, routed_path, candidates, clauses, orch)
                if producer is not None:
                    self._trace("producer_inserted", cell=producer.cell_id, for_anchor=next_anchor.cell_id,
                                clause_idx=getattr(next_anchor, "matched_clause_idx", None),
                                reason="anchor input type unmet by scope; producer has same-clause identity evidence and unifies from scope")
                    if not self.step_unifies(curr_cell, producer, prev_path=routed_path):
                        p_bridges = self.find_bridge_chain(curr_cell, producer, candidates, prev_path=routed_path)
                        if p_bridges:
                            for pb in p_bridges:
                                pb.matched_clause_idx = getattr(producer, "matched_clause_idx", None)
                                pb.clause_literals = getattr(producer, "clause_literals", {})
                            routed_path.extend(p_bridges)
                    routed_path.extend([producer, next_anchor])
                    connected = True
                else:
                    curr_out = getattr(curr_cell, "primary_output", None) or (
                        list(curr_cell.outputs.values())[0] if curr_cell.outputs else None
                    )
                    curr_out_sig = getattr(curr_out, "signature", curr_out)
                    c_rank = getattr(curr_out_sig, "canonical_rank", None)
                    c_sk = str(getattr(curr_out, "structure_kind", "") or getattr(curr_out_sig, "structure_kind", "") or "").lower()
                    c_role = str(getattr(curr_cell, "node_role", "") or "").lower()
                    is_curr_reduction = (c_rank == 0) or (c_sk == "scalar") or (c_role == "reduction")

                    next_in = getattr(next_anchor, "primary_input", None) or (
                        list(next_anchor.inputs.values())[0] if next_anchor.inputs else None
                    )
                    next_in_sig = getattr(next_in, "signature", next_in)
                    n_rank = getattr(next_in_sig, "canonical_rank", None)
                    n_sk = str(getattr(next_in, "structure_kind", "") or getattr(next_in_sig, "structure_kind", "") or "").lower()
                    requires_tensor = (n_rank is not None and n_rank >= 1) or (n_sk in ("tensor", "collection", "composite"))

                    if is_curr_reduction and requires_tensor:
                        # Dimensional rank reduction: curr_cell is a scalar measurement (rank 0).
                        # The carrier tensor remains in the monoidal DAG scope.
                        # Check if next_anchor unifies or bridges from DAG scope.
                        if self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                            self._trace("anchor_connected", cell=next_anchor.cell_id, via="dag_scope", after=curr_cell.cell_id)
                            routed_path.append(next_anchor)
                            connected = True
                        else:
                            for ancestor in reversed(routed_path[:-1]):
                                if self.step_unifies(ancestor, next_anchor, prev_path=routed_path):
                                    routed_path.append(next_anchor)
                                    connected = True
                                    break
                                anc_bridge = self.find_bridge(ancestor, next_anchor, candidates, orch, prev_path=routed_path)
                                if anc_bridge:
                                    anc_bridge.matched_clause_idx = getattr(next_anchor, "matched_clause_idx", None)
                                    anc_bridge.clause_literals = getattr(next_anchor, "clause_literals", {})
                                    routed_path.extend([anc_bridge, next_anchor])
                                    connected = True
                                    break
                    else:
                        # Sequential carrier continuity takes precedence over DAG-scope fallback
                        bridges = self.find_bridge_chain(curr_cell, next_anchor, candidates, prev_path=routed_path)
                        if bridges:
                            for b in bridges:
                                b.matched_clause_idx = getattr(next_anchor, "matched_clause_idx", None)
                                b.clause_literals = getattr(next_anchor, "clause_literals", {})
                            routed_path.extend(bridges)
                            routed_path.append(next_anchor)
                            connected = True
                        elif self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                            self._trace("anchor_connected", cell=next_anchor.cell_id, via="dag_scope", after=curr_cell.cell_id)
                            routed_path.append(next_anchor)
                            connected = True
                        else:
                            # In a DAG, if next_anchor does not unify directly with curr_cell,
                            # check if next_anchor unifies or bridges from any earlier ancestor in routed_path
                            for ancestor in reversed(routed_path[:-1]):
                                if self.step_unifies(ancestor, next_anchor, prev_path=routed_path) or self.step_unifies_dag(next_anchor, routed_path, ctx=ctx):
                                    routed_path.append(next_anchor)
                                    connected = True
                                    break
                                anc_bridge = self.find_bridge(ancestor, next_anchor, candidates, orch, prev_path=routed_path)
                                if anc_bridge:
                                    anc_bridge.matched_clause_idx = getattr(next_anchor, "matched_clause_idx", None)
                                    anc_bridge.clause_literals = getattr(next_anchor, "clause_literals", {})
                                    routed_path.extend([anc_bridge, next_anchor])
                                    connected = True
                                    break

            if not connected:
                cl_idx = getattr(next_anchor, "matched_clause_idx", None)
                alts = clause_alts.get(cl_idx, [])
                for alt in alts:
                    if alt.cell_id == next_anchor.cell_id or any(c.cell_id == alt.cell_id for c in routed_path if getattr(c, "matched_clause_idx", None) == cl_idx):
                        continue
                    if self.step_unifies(curr_cell, alt, prev_path=routed_path):
                        routed_path.append(alt)
                        connected = True
                        self._trace("anchor_alternative_connected", clause_idx=cl_idx, original=next_anchor.cell_id, cell=alt.cell_id, via="direct")
                        break
                    alt_bridges = self.find_bridge_chain(curr_cell, alt, candidates, prev_path=routed_path)
                    if alt_bridges:
                        for b in alt_bridges:
                            b.matched_clause_idx = getattr(alt, "matched_clause_idx", None)
                            b.clause_literals = getattr(alt, "clause_literals", {})
                        routed_path.extend(alt_bridges)
                        routed_path.append(alt)
                        connected = True
                        self._trace("anchor_alternative_connected", clause_idx=cl_idx, original=next_anchor.cell_id, cell=alt.cell_id, via="bridge_chain")
                        break
                    if self.step_unifies_dag(alt, routed_path, ctx=ctx):
                        routed_path.append(alt)
                        connected = True
                        self._trace("anchor_alternative_connected", clause_idx=cl_idx, original=next_anchor.cell_id, cell=alt.cell_id, via="dag_scope")
                        break
                    for ancestor in reversed(routed_path[:-1]):
                        if self.step_unifies(ancestor, alt, prev_path=routed_path) or self.step_unifies_dag(alt, routed_path, ctx=ctx):
                            routed_path.append(alt)
                            connected = True
                            self._trace("anchor_alternative_connected", clause_idx=cl_idx, original=next_anchor.cell_id, cell=alt.cell_id, via="ancestor")
                            break
                        anc_b = self.find_bridge(ancestor, alt, candidates, orch, prev_path=routed_path)
                        if anc_b:
                            anc_b.matched_clause_idx = getattr(alt, "matched_clause_idx", None)
                            anc_b.clause_literals = getattr(alt, "clause_literals", {})
                            routed_path.extend([anc_b, alt])
                            connected = True
                            self._trace("anchor_alternative_connected", clause_idx=cl_idx, original=next_anchor.cell_id, cell=alt.cell_id, via="ancestor_bridge")
                            break
                    if connected:
                        break

            if not connected:
                # Previously dropped without a trace. A clause losing its
                # only anchor here is exactly how "perform FFT" vanished.
                self._trace(
                            "anchor_dropped", cell=next_anchor.cell_id,
                            clause_idx=getattr(next_anchor, "matched_clause_idx", None),
                            reason="no direct unification, no DAG-scope unification, no bridge from any ancestor",
                            ancestors=[a.cell_id for a in routed_path],
                            needs=[(n, str(getattr(getattr(p, "signature", p), "type_name", "")))
                                   for n, p in next_anchor.inputs.items() if getattr(p, "required", False)],
                            ancestor_outputs=[(a.cell_id, str(getattr(getattr(a.primary_output, "signature", a.primary_output), "type_name", "")))
                                              for a in routed_path if getattr(a, "primary_output", None) is not None])

        # 3. Handle endpoint completion if declared in prompt
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)
        has_egress = bool(dest_file_literals) or _clause_has_egress_intent(prompt or "")

        if has_egress and routed_path:
            last_cell = routed_path[-1]
            if not self.is_sink_node(last_cell, orch):
                sinks = [c for c in candidates if self.is_sink_node(c, orch)]
                best_sink = None
                best_s_score = -1.0
                dst_lit = dest_file_literals[0] if dest_file_literals else None
                for s in sinks:
                    if dst_lit is not None and self.file_compat(s, dst_lit) < 0:
                        continue
                    if self.step_unifies(last_cell, s, prev_path=routed_path):
                        sc = relevance_map.get(s.cell_id, 0.0) + self.calculate_edge_affinity(last_cell, s, orch, relevance_map=relevance_map)
                        if sc > best_s_score:
                            best_s_score = sc
                            best_sink = s
                if best_sink:
                    routed_path.append(best_sink)

        # 4. Fallback to LatticePlanner if empty and allow_fallback is enabled
        try:
            from config import settings
            allow_fb = settings.allow_fallback
        except Exception:
            try:
                from ..config import settings
                allow_fb = settings.allow_fallback
            except Exception:
                allow_fb = False

        if orch and not routed_path:
            if allow_fb:
                try:
                    try:
                        from ..planner import LatticePlanner
                    except (ImportError, ValueError):
                        from planner import LatticePlanner
                    planner = LatticePlanner(orchestrator=orch)
                    fb_path = planner.plan(
                        prompt=prompt,
                        tunnel=tunnel,
                        relevance_map=relevance_map,
                        start_sig=start_sig,
                        goal_sig=goal_sig,
                        max_transforms=max_transforms
                    )
                    if fb_path:
                        return fb_path
                except Exception as fb_err:
                    logger.debug(f"[M1] Fallback to LatticePlanner failed: {fb_err}")
            return []

        # Propagate matched_clause_idx forward to any unindexed bridge cells
        for i, c in enumerate(routed_path):
            if getattr(c, "matched_clause_idx", None) is None:
                for nxt in routed_path[i + 1:]:
                    n_idx = getattr(nxt, "matched_clause_idx", None)
                    if n_idx is not None:
                        c.matched_clause_idx = n_idx
                        c.clause_literals = getattr(nxt, "clause_literals", {})
                        break

        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return routed_path[:max_allowed]
