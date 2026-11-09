"""
src/route_methods/m2_endpoint_anchor.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M2: Endpoint-Anchored Coverage (Bidirectional).
Anchors source entry and sink exit morphisms and searches bidirectionally.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS
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
            terminal_roles = set()
            if orch and hasattr(orch, "registry"):
                terminal_roles.update(orch.registry.get_verification_semantics("terminal_roles") or ())
            terminal_roles.update({"evaluator", "metric", "sink", "consumer"})
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

        # Single-node pipeline if source is also sink or only one anchor makes sense
        if best_sink and best_source.cell_id == best_sink.cell_id:
            return [best_source]

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        num_clauses = len(clauses) if clauses else 1

        # 3. For multi-clause compound prompts (> 2 clauses), chain forward through intermediate transformations
        if num_clauses > 2:
            path = [best_source]
            curr = best_source
            max_steps = max(2, min(16, max(max_transforms + 2, num_clauses + 3)))
            _ev = self._evidence()
            _last_toks = _ev.clause_tokens(clauses[-1]) if _ev is not None else set()
            _covered_last: set = set()
            for cl_idx in range(1, num_clauses + 2):
                if self.is_sink_node(curr, orch) or (best_sink and curr.cell_id == best_sink.cell_id):
                    break
                valid_next = [
                    c for c in candidates
                    if c.cell_id != curr.cell_id and c.cell_id not in [p.cell_id for p in path]
                    and (self.step_unifies(curr, c, prev_path=path) or self.step_unifies_dag(c, path, ctx=ctx))
                ]
                if cl_idx >= num_clauses and _ev is not None:
                    # every clause has had its turn: only continue for last-clause tokens nothing explained yet
                    _resid = _last_toks - _covered_last
                    valid_next = [c for c in valid_next if _ev.identity_overlap(_resid, c)["hit"] > 0.0]
                if not valid_next:
                    if best_sink and (self.step_unifies(curr, best_sink, prev_path=path) or self.step_unifies_dag(best_sink, path, ctx=ctx)) and best_sink.cell_id not in [p.cell_id for p in path]:
                        path.append(best_sink)
                    break

                target_cl = clauses[min(cl_idx, num_clauses - 1)] if cl_idx < num_clauses else ""
                target_toks = _ev.clause_tokens(target_cl or prompt) if _ev is not None else set()

                def _score_step(c: Cell):
                    # Identity evidence (recall, name-precision, F) > retrieval + edge affinity > sink tie-break
                    if _ev is not None and target_toks:
                        ov = _ev.identity_overlap(target_toks, c)
                        fit = ov["rank"]
                    else:
                        fit = self.clause_fit(target_cl or prompt, c)
                    rel = relevance_map.get(c.cell_id, 0.0)
                    aff = self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map)
                    is_snk = 1.0 if (best_sink and c.cell_id == best_sink.cell_id and cl_idx >= num_clauses - 1) else 0.0
                    return (fit, rel + aff, is_snk)

                valid_next.sort(key=_score_step, reverse=True)
                nxt = valid_next[0]
                self._trace("m2_step", step=cl_idx, target_clause=min(cl_idx, num_clauses - 1) + 1, chosen=nxt.cell_id,
                            fit=tuple(round(x, 3) for x in _score_step(nxt)[0]),
                            runners_up=[(c.cell_id, tuple(round(x, 3) for x in _score_step(c)[0])) for c in valid_next[1:3]],
                            candidates_after_filter=len(valid_next))
                path.append(nxt)
                curr = nxt
                if _ev is not None and cl_idx >= num_clauses - 1:
                    _covered_last |= set(_ev.identity_overlap(_last_toks, nxt)["tokens"])

            if best_sink and path[-1].cell_id != best_sink.cell_id:
                if self.step_unifies(path[-1], best_sink, prev_path=path) or self.step_unifies_dag(best_sink, path, ctx=ctx):
                    path.append(best_sink)

            self.tag_cells_with_clause_indices(path, prompt, orchestrator=orch)
            try:
                from ..token_evidence import audit as _audit
                _cov = _audit(path, prompt, orch)
                if _cov.get("available") and any(i["kind"] == "clause_unserved" for i in _cov.get("issues", [])):
                    return self._delegate_to_m1(
                        "M2 compound path has unserved clauses", prompt, tunnel, relevance_map,
                        orch=orch, ctx=ctx, **kwargs
                    )
            except Exception as _cov_exc:
                logger.debug("[M2] Coverage audit check skipped: %s", _cov_exc)

            try:
                from ..unification import UnificationGate, ExecutionContext
                from ..preflight import PreflightLinter
                _gate = UnificationGate(orchestrator=orch)
                _ur = _gate.unify_pipeline(path, ctx=ctx or ExecutionContext(prompt=prompt))
                if _ur is not None and getattr(_ur, "value", None):
                    _lint = PreflightLinter.lint(_ur.value, prompt=prompt, orchestrator=orch)
                    if not _lint.is_valid:
                        return self._delegate_to_m1(
                            f"M2 path failed pre-flight lint: {'; '.join(_lint.violations[:2])}",
                            prompt, tunnel, relevance_map, orch=orch, ctx=ctx, **kwargs
                        )
            except Exception as _chk_exc:
                logger.debug("[M2] Pre-flight verification check skipped: %s", _chk_exc)

            return path

        # Bidirectional Meeting-in-the-Middle through intermediate transforms (short prompts)
        transforms = [c for c in candidates if self.is_transform_node(c, orch)]
        transforms.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
        max_tf = settings.m2_max_candidate_transforms
        if len(transforms) > max_tf:
            self._trace("m2_transforms_truncated", total_candidates=len(transforms), cap=max_tf)
            transforms = transforms[:max_tf]

        if best_sink and transforms:
            # Look for 1-hop bridge: source -> T -> sink
            best_mid: Optional[Cell] = None
            best_mid_score = ((-1.0, -1.0, -1.0), -1.0)
            for t in transforms:
                if self.step_unifies(best_source, t) and self.step_unifies(t, best_sink, prev_path=[best_source, t]):
                    sc = (
                        self.clause_fit(prompt, t),
                        relevance_map.get(t.cell_id, 0.0)
                        + self.calculate_edge_affinity(best_source, t, orch, relevance_map=relevance_map)
                        + self.calculate_edge_affinity(t, best_sink, orch, relevance_map=relevance_map)
                        + (1.0 if t.domain_name == best_source.domain_name else 0.0),
                    )
                    if sc > best_mid_score:
                        best_mid_score = sc
                        best_mid = t
            if best_mid:
                return [best_source, best_mid, best_sink]

            # Look for 2-hop bridge: source -> T1 -> T2 -> sink
            best_pair: Optional[Tuple[Cell, Cell]] = None
            best_pair_score = ((-1.0, -1.0, -1.0), -1.0)
            for t1 in transforms:
                if not self.step_unifies(best_source, t1):
                    continue
                for t2 in transforms:
                    if t1.cell_id == t2.cell_id:
                        continue
                    if self.step_unifies(t1, t2, prev_path=[best_source, t1]) and self.step_unifies(t2, best_sink, prev_path=[best_source, t1, t2]):
                        f1, f2 = self.clause_fit(prompt, t1), self.clause_fit(prompt, t2)
                        sc = (
                            tuple(a + b for a, b in zip(f1, f2)),
                            relevance_map.get(t1.cell_id, 0.0) + relevance_map.get(t2.cell_id, 0.0)
                            + self.calculate_edge_affinity(best_source, t1, orch, relevance_map=relevance_map)
                            + self.calculate_edge_affinity(t1, t2, orch, relevance_map=relevance_map)
                            + self.calculate_edge_affinity(t2, best_sink, orch, relevance_map=relevance_map),
                        )
                        if sc > best_pair_score:
                            best_pair_score = sc
                            best_pair = (t1, t2)
            if best_pair:
                return [best_source, best_pair[0], best_pair[1], best_sink]

        # If direct unification is valid when no transform found and prompt is trivial
        if best_sink and self.step_unifies(best_source, best_sink) and num_clauses <= 1:
            return [best_source, best_sink]

        # Fallback forward chaining
        path = [best_source]
        curr = best_source
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)
        for _ in range(max_steps):
            valid_next = [c for c in candidates if c.cell_id != curr.cell_id and self.step_unifies(curr, c)]
            if not valid_next:
                break
            valid_next.sort(
                key=lambda c: relevance_map.get(c.cell_id, 0.0) + self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map),
                reverse=True
            )
            nxt = valid_next[0]
            path.append(nxt)
            curr = nxt
            if self.is_sink_node(curr, orch):
                break

        return path
