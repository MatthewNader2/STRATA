"""
src/route_methods/m3_greedy_freeze.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M3: Forward Greedy Beam with Frozen Committed Prefixes.
Greedily commits the best next morphism at each step without backtracking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, is_terminal_sink, is_macro_cell

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
                    not is_macro_cell(c)
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

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        covered_tokens: Set[str] = set(getattr(best_entry, "identity_tokens", best_entry.token_set) & prompt_tokens)

        num_clauses = len(clauses) if clauses else 1
        max_steps = max(16, num_clauses * 3 + 2, max_transforms + 4)

        # 2. Pure myopic forward greedy freeze: no lookahead, no anchor precomputation, no backtracking
        for step in range(max_steps):
            curr = committed_path[-1]

            # Stop once a terminal sink is reached after at least one continuation
            if self.is_sink_node(curr, orch) and len(committed_path) > 1:
                break

            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids
                and not self.is_ingress_node(c, orch)
                and (self.step_unifies(curr, c, prev_path=committed_path) or self.step_unifies_dag(c, committed_path, ctx=ctx))
            ]

            if not valid_next:
                break

            uncovered_tokens = prompt_tokens - covered_tokens
            _requested_sink = self.requested_sink(prompt, candidates)

            def _local_greedy_score(cand: Cell) -> Tuple[Any, ...]:
                rel = relevance_map.get(cand.cell_id, 0.0)
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                c_toks = getattr(cand, "identity_tokens", cand.token_set)
                new_hits = c_toks & uncovered_tokens
                new_mass = _ev.mass(new_hits) if _ev is not None else float(len(new_hits))

                is_cand_sink = self.is_sink_node(cand, orch)
                is_curr_sink = self.is_sink_node(curr, orch)
                unrequested_egress = (
                    is_cand_sink and self._is_path_consumer(cand)
                    and not (dst_lit is not None and self.file_compat(cand, dst_lit) > 0)
                ) or (_requested_sink is None and is_cand_sink and not is_curr_sink and new_mass <= 0.0)

                handle_score = self.linear_handle_score(cand, committed_path)
                # Purely myopic local compatibility & relevance
                if new_mass <= 0.0 and rel + aff <= 0.0:
                    local_fit = (0.0, 0.0, 0.0)
                elif _ev is not None and uncovered_tokens:
                    local_fit = _ev.identity_overlap(uncovered_tokens, cand)["rank"]
                else:
                    local_fit = self.clause_fit(prompt, cand)

                return (
                    0 if unrequested_egress else 1,
                    handle_score,
                    local_fit,
                    new_mass,
                    rel + aff,
                )

            valid_next.sort(key=_local_greedy_score, reverse=True)
            best_next = valid_next[0]
            best_sc = _local_greedy_score(best_next)

            # Freeze permanently without backtracking; stop if no positive local signal remains
            if best_sc[0] == 0 and best_sc[3] <= 0.0 and best_sc[4] <= 0.1:
                break

            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(getattr(best_next, "identity_tokens", best_next.token_set) & prompt_tokens)

            if self.is_sink_node(best_next, orch) and not (prompt_tokens - covered_tokens):
                break

        # Verify linear/DAG unification without silently delegating to M1
        try:
            try:
                from unification import UnificationGate, ExecutionContext, UNRESOLVED_PORT
            except (ImportError, ValueError):
                from ..unification import UnificationGate, ExecutionContext, UNRESOLVED_PORT
            _gate = UnificationGate(orchestrator=orch)
            _tctx = ctx if ctx is not None else ExecutionContext(prompt=prompt)
            _ur = _gate.unify_pipeline(committed_path, _tctx, extract_llm_slots=False)
            has_unresolved = bool(
                getattr(_ur, "value", None)
                and any(v is UNRESOLVED_PORT for _, b in _ur.value for v in b.values())
            )
            if _ur.is_bottom() or not getattr(_ur, "value", None) or has_unresolved:
                reason = "M3 myopic greedy path failed type unification or left unresolved input ports"
                self.fallback_reason = reason
                self._trace("method_failure", requested=type(self).__name__, reason=reason)
                return []
        except Exception as _chk_exc:
            logger.debug("[M3] Unification verification check skipped: %s", _chk_exc)

        return committed_path
