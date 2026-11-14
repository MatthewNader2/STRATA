"""
src/route_methods/m2_endpoint_anchor.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M2: Endpoint-Anchored Coverage (Bidirectional).
Anchors source entry and sink exit morphisms and searches bidirectionally.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, is_macro_cell
from log_config import get_logger

logger = get_logger("m2_endpoint_anchor")

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
    from ..config import settings
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext
    from config import settings


class M2EndpointAnchorRouteMethod(RouteMethod):
    """
    RouteMethod M2: Endpoint-Anchored Coverage.
    Identifies source and destination anchors from prompt assets and constraints,
    then executes bidirectional meeting-in-the-middle pathfinding.
    """
    name: str = "m2_endpoint_anchor"

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
                    not is_macro_cell(c)
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ]
        if not candidates:
            return [tunnel[0]]

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        first_clause = clauses[0] if clauses else prompt
        last_clause = clauses[-1] if clauses else prompt
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None

        # 1. Source anchor: format-compatible (ordinal) > reachable downstream anchors > evidence for the FIRST clause > retrieval relevance.
        source_pool = [c for c in candidates if self.is_ingress_node(c, orch)] or candidates[:5]
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

        best_source: Optional[Cell] = max(
            source_pool,
            key=lambda c: (
                self.file_compat(c, src_lit),
                _downstream_reach(c),
                self.clause_fit(first_clause, c),
                relevance_map.get(c.cell_id, 0.0),
            ),
        )

        # 2. Sink anchor: only a cell the FINAL clause actually names (or a format-compatible writer for
        # an explicit destination file). A path-writing sink with no destination literal is never requested.
        sink_pool = [c for c in candidates if self.is_sink_node(c, orch)]
        requested = self.requested_sink(prompt, sink_pool)   # beats every non-sink cell on some clause
        if requested is not None:
            eligible_sinks = [requested]
        elif dst_lit is not None:
            eligible_sinks = [c for c in sink_pool if self.file_compat(c, dst_lit) > 0]
        else:
            # Generalized co-terminal anchor: in memory-only or open pipelines,
            # terminal cells or evaluators matching the final clause serve as the sink anchor.
            terminal_roles = set(TypeRegistry.get_instance().get_terminal_roles_set())
            eligible_sinks = [
                c for c in candidates
                if (getattr(c, "node_role", "") in terminal_roles or self.clause_fit(last_clause, c)[0] > 0)
                and not self.is_ingress_node(c, orch)
            ]
        best_sink: Optional[Cell] = max(
            eligible_sinks,
            key=lambda c: (
                self.file_compat(c, dst_lit) if dst_lit is not None else 0,
                self.clause_fit(last_clause, c),
                relevance_map.get(c.cell_id, 0.0),
            ),
        ) if eligible_sinks else None
        self._trace("m2_endpoints", source=best_source.cell_id if best_source else None,
                    sink=best_sink.cell_id if best_sink else None,
                    sink_pool=[c.cell_id for c in sink_pool][:8], eligible_sinks=[c.cell_id for c in eligible_sinks][:8])

        if not best_source:
            best_source = candidates[0]

        # Single-node pipeline if source is also sink
        if best_sink and best_source.cell_id == best_sink.cell_id:
            return [best_source]

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        num_clauses = len(clauses) if clauses else 1

        if not best_sink:
            # No explicit/co-terminal sink found: select highest-scoring non-ingress candidate as target endpoint
            non_ing = [c for c in candidates if c.cell_id != best_source.cell_id and not self.is_ingress_node(c, orch)]
            if non_ing:
                best_sink = max(
                    non_ing,
                    key=lambda c: (self.clause_fit(last_clause, c), relevance_map.get(c.cell_id, 0.0)),
                )
            else:
                return [best_source]

        # 3. True Bidirectional Meet-in-the-Middle Search from c_start to c_end for arbitrary K >= 1
        # Uses indexed orchestrator adjacency / reverse_adjacency, restricted to tunnel candidates.
        cand_by_id: Dict[str, Cell] = {c.cell_id: c for c in candidates}
        cand_by_id.setdefault(best_source.cell_id, best_source)
        cand_by_id.setdefault(best_sink.cell_id, best_sink)

        orch_adj = getattr(orch, "_adjacency", None) or {}
        orch_rev = getattr(orch, "_reverse_adjacency", None) or {}

        fwd_adj: Dict[str, List[Cell]] = {}
        rev_adj: Dict[str, List[Cell]] = {}
        pool_cells = list(cand_by_id.values())

        def _get_fwd_neighbors(cid: str) -> List[Cell]:
            cached = fwd_adj.get(cid)
            if cached is not None:
                return cached
            src_c = cand_by_id[cid]
            nbrs: List[Cell] = []
            seen: Set[str] = {cid}
            for nid in orch_adj.get(cid, ()):
                nc = cand_by_id.get(nid)
                if nc is not None and nid not in seen and self.step_unifies(src_c, nc, prev_path=[src_c]):
                    seen.add(nid)
                    nbrs.append(nc)
            for nc in pool_cells:
                if nc.cell_id not in seen and self.step_unifies(src_c, nc, prev_path=[src_c]):
                    seen.add(nc.cell_id)
                    nbrs.append(nc)
            nbrs.sort(
                key=lambda c: (
                    self.calculate_edge_affinity(src_c, c, orch, relevance_map=relevance_map),
                    relevance_map.get(c.cell_id, 0.0),
                ),
                reverse=True,
            )
            fwd_adj[cid] = nbrs
            return nbrs

        def _get_rev_neighbors(cid: str) -> List[Cell]:
            cached = rev_adj.get(cid)
            if cached is not None:
                return cached
            dst_c = cand_by_id[cid]
            nbrs: List[Cell] = []
            seen: Set[str] = {cid}
            for pid in orch_rev.get(cid, ()):
                pc = cand_by_id.get(pid)
                if pc is not None and pid not in seen and self.step_unifies(pc, dst_c, prev_path=[pc]):
                    seen.add(pid)
                    nbrs.append(pc)
            for pc in pool_cells:
                if pc.cell_id not in seen and self.step_unifies(pc, dst_c, prev_path=[pc]):
                    seen.add(pc.cell_id)
                    nbrs.append(pc)
            nbrs.sort(
                key=lambda c: (
                    self.calculate_edge_affinity(c, dst_c, orch, relevance_map=relevance_map),
                    relevance_map.get(c.cell_id, 0.0),
                ),
                reverse=True,
            )
            rev_adj[cid] = nbrs
            return nbrs

        max_total_len = max(4, min(14, max_transforms + num_clauses + 2))
        beam_limit = max(16, getattr(settings, "m2_max_candidate_transforms", 24))
        half_steps = max(2, (max_total_len // 2) + 1)

        # Frontier maps endpoint_id -> list of partial paths ending (or starting) at endpoint_id
        fwd_paths: Dict[str, List[List[Cell]]] = {best_source.cell_id: [[best_source]]}
        bwd_paths: Dict[str, List[List[Cell]]] = {best_sink.cell_id: [[best_sink]]}
        fwd_frontier: List[List[Cell]] = [[best_source]]
        bwd_frontier: List[List[Cell]] = [[best_sink]]

        meeting_candidates: List[List[Cell]] = []
        seen_signatures: Set[Tuple[str, ...]] = set()

        def _record_meeting(fp: List[Cell], bp: List[Cell]) -> None:
            # fp ends with M, bp starts with M
            merged = fp + bp[1:]
            if len(merged) > max_total_len or len(merged) < 2:
                return
            sig = tuple(c.cell_id for c in merged)
            if sig in seen_signatures:
                return
            # Ensure linear/DAG step unification across full merged path
            if len({c.cell_id for c in merged}) != len(merged):
                return
            if self.chain_is_valid(merged, prompt=prompt):
                seen_signatures.add(sig)
                meeting_candidates.append(merged)

        # Check trivial 1-hop bridge edge (direct edge fwd[0] -> bwd[0])
        if self.step_unifies(best_source, best_sink, prev_path=[best_source]):
            _record_meeting([best_source, best_sink], [best_sink])

        for _step in range(half_steps):
            # Check intersections between current fwd_paths and bwd_paths
            for mid_id in set(fwd_paths.keys()) & set(bwd_paths.keys()):
                for fp in fwd_paths[mid_id][:4]:
                    for bp in bwd_paths[mid_id][:4]:
                        _record_meeting(fp, bp)

            if not fwd_frontier and not bwd_frontier:
                break

            # Alternate expanding smaller active frontier (pure Meet-in-the-Middle)
            expand_fwd = bool(fwd_frontier) and (not bwd_frontier or len(fwd_frontier) <= len(bwd_frontier))
            if expand_fwd:
                next_fwd: List[List[Cell]] = []
                for fp in fwd_frontier:
                    tail = fp[-1]
                    fp_ids = {c.cell_id for c in fp}
                    for nxt in _get_fwd_neighbors(tail.cell_id):
                        if nxt.cell_id in fp_ids:
                            continue
                        if not (self.step_unifies(tail, nxt, prev_path=fp) or self.step_unifies_dag(nxt, fp, ctx=ctx)):
                            continue
                        new_fp = fp + [nxt]
                        fwd_paths.setdefault(nxt.cell_id, []).append(new_fp)
                        next_fwd.append(new_fp)
                        if nxt.cell_id in bwd_paths:
                            for bp in bwd_paths[nxt.cell_id][:4]:
                                _record_meeting(new_fp, bp)
                next_fwd.sort(
                    key=lambda p: sum(relevance_map.get(c.cell_id, 0.0) for c in p) / len(p),
                    reverse=True,
                )
                fwd_frontier = next_fwd[:beam_limit]
            else:
                next_bwd: List[List[Cell]] = []
                for bp in bwd_frontier:
                    head = bp[0]
                    bp_ids = {c.cell_id for c in bp}
                    for prev_c in _get_rev_neighbors(head.cell_id):
                        if prev_c.cell_id in bp_ids:
                            continue
                        if not self.step_unifies(prev_c, head, prev_path=[prev_c]):
                            continue
                        new_bp = [prev_c] + bp
                        bwd_paths.setdefault(prev_c.cell_id, []).append(new_bp)
                        next_bwd.append(new_bp)
                        if prev_c.cell_id in fwd_paths:
                            for fp in fwd_paths[prev_c.cell_id][:4]:
                                _record_meeting(fp, new_bp)
                next_bwd.sort(
                    key=lambda p: sum(relevance_map.get(c.cell_id, 0.0) for c in p) / len(p),
                    reverse=True,
                )
                bwd_frontier = next_bwd[:beam_limit]

        # Final sweep of direct edges between any fwd leaf and bwd root (odd-length meets without shared node)
        for fid, fps in fwd_paths.items():
            for fp in fps[:3]:
                for nbr in _get_fwd_neighbors(fid):
                    if nbr.cell_id in bwd_paths:
                        for bp in bwd_paths[nbr.cell_id][:3]:
                            _record_meeting(fp, bp)

        if not meeting_candidates:
            reason = f"Bidirectional Meet-in-the-Middle found no connecting path between {best_source.cell_id} and {best_sink.cell_id}"
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        _ev = self._evidence()
        p_toks = _ev.clause_tokens(prompt) if _ev is not None else prompt_tokens
        cl_tok_sets = [(_ev.clause_tokens(cl) if _ev is not None else CellTokenizer.tokenize_prompt(cl)) for cl in clauses]

        def _score_meeting(m_path: List[Cell]) -> Tuple[Any, ...]:
            served_cl = 0
            for cl_t in cl_tok_sets:
                if cl_t and any((getattr(c, "identity_tokens", c.token_set) & cl_t) for c in m_path):
                    served_cl += 1
            path_toks = set().union(*(getattr(c, "identity_tokens", c.token_set) for c in m_path))
            tok_cov = len(path_toks & p_toks) / max(len(p_toks), 1)
            aff_sum = sum(
                self.calculate_edge_affinity(m_path[i], m_path[i + 1], orch, relevance_map=relevance_map)
                for i in range(len(m_path) - 1)
            )
            rel_sum = sum(relevance_map.get(c.cell_id, 0.0) for c in m_path)
            has_mid = 1 if (len(m_path) > 2 or num_clauses <= 2) else 0
            return (served_cl, has_mid, round(tok_cov, 4), (aff_sum + rel_sum) / len(m_path), -len(m_path))

        meeting_candidates.sort(key=_score_meeting, reverse=True)
        best_path = meeting_candidates[0]
        self._trace(
            "m2_mitm_selected",
            source=best_source.cell_id,
            sink=best_sink.cell_id,
            candidates_found=len(meeting_candidates),
            path=[c.cell_id for c in best_path],
        )
        return best_path[:max(32, num_clauses * 4 + 4, max_transforms + 8)]
