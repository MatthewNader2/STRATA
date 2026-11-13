"""
src/route_methods/m10_monte_carlo.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

M10: verifier-guided Monte Carlo Tree Search over plan structure.

The deterministic methods (M1-M3, M6) commit to the single best-ranked cell per clause. That cannot settle
questions lexical evidence leaves open: which of several near-equivalent cells to use, and whether a bridge
cell is needed to make the program valid (e.g. a complex FFT result bound to an estimator whose tree
precondition rejects complex data). M10 searches those choices with real Monte Carlo Tree Search
(src/mcts.py) and scores every candidate plan with the same static verifiers the pipeline ends with:
Layer-3 unification, code emission and the preflight linter (coverage audit, row lineage, tree-declared
property contracts). No sandbox execution and no LLM.

Action space (all derived from the lattice and the trees, nothing hardcoded):
  * for each prompt clause: its shared anchor group (primary + residual-serving cells) and up to
    `mcts_max_branch - 1` alternative single cells ranked by identity evidence;
  * optionally, up to `mcts_max_bridges` bridge cells: cells whose tree-declared POSTCONDITION establishes a
    property that some candidate's tree-declared PRECONDITION requires.
Reward: (fraction of clauses served) x (mean evidence quality of the cells serving them) / (1 + number of
preflight violations of the emitted program). A clause counts as served only through the cells assigned to it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import RouteMethod, Cell, is_macro_cell
from lattice import LatticeOrchestrator, TypeRegistry
from unification import ExecutionContext, unify


def _cond_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class M10MonteCarloRouteMethod(RouteMethod):
    """Monte Carlo Tree Search over per-clause cell choices and bridges, scored by Layer-3 + preflight."""

    name: str = "m10_monte_carlo"

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
        from config import settings
        from mcts import MonteCarloTreeSearch
        from token_evidence import anchors_for_clause
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        pool = [c for c in tunnel if getattr(c, "node_type", "") != "constant" and not is_macro_cell(c)]
        ev = self._evidence()
        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        if ev is None or not pool or not clauses:
            reason = "evidence service or clause segmentation unavailable"
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        proto: Dict[str, Cell] = {c.cell_id: c for c in pool}
        literals: List[List[str]] = [
            [v for _, k, v in ExecutionContext._extract_universal_literals(cl) if k in ("identifier", "quoted_str")]
            for cl in clauses
        ]

        # ---- action space: per-clause groups -------------------------------------------------
        groups: List[List[Tuple[List[str], float]]] = []
        max_alts = max(getattr(settings, "mcts_max_branch", 3) - 1, 1)
        for j, cl in enumerate(clauses):
            toks = ev.clause_tokens(cl)
            primary = [c for c, _ in anchors_for_clause(ev, cl, pool, relevance_map, on_event=None, clause_idx=j)]
            gl: List[Tuple[List[str], float]] = []
            if primary:
                _po = ev.identity_overlap(toks, primary[0])
                gl.append(([c.cell_id for c in primary], _po["recall"] * _po["name_precision"]))
                prim_ids = {p.cell_id for p in primary}
                alt_scored = [
                    (o, c)
                    for o, c in ((ev.identity_overlap(toks, c), c) for c in pool if c.cell_id not in prim_ids)
                    if o["hit"] > 0.0 or relevance_map.get(c.cell_id, 0.0) > 0.0
                ]
                alt_scored.sort(key=lambda t: (t[0]["rank"], relevance_map.get(t[1].cell_id, 0.0)), reverse=True)
                for o, c in alt_scored[:max_alts]:
                    score_val = max(o["recall"] * o["name_precision"], 0.1 * relevance_map.get(c.cell_id, 0.0))
                    gl.append(([c.cell_id], score_val))
            else:
                fallback_cands = sorted(
                    pool,
                    key=lambda c: relevance_map.get(c.cell_id, 0.0),
                    reverse=True,
                )[:max_alts + 1]
                for c in fallback_cands:
                    gl.append(([c.cell_id], max(0.05, relevance_map.get(c.cell_id, 0.0))))
            groups.append(gl)

        # ---- bridge candidates: tree postconditions that satisfy tree preconditions ----------
        # Only properties where a producer and a consumer among the CANDIDATE PLAN CELLS actually contradict each
        # other can need a bridge (e.g. one declares output is_complex == True, another requires is_complex == False).
        plan_cells = [proto[cid] for gl in groups for ids, _ in gl for cid in ids]
        produced: Dict[Any, set] = {}
        for c in plan_cells:
            for q in (getattr(c, "postconditions", None) or []):
                if _cond_get(q, "operator") == "==" and _cond_get(q, "target") == "output_var" and _cond_get(q, "property") is not None:
                    produced.setdefault(_cond_get(q, "property"), set()).add(repr(_cond_get(q, "value")))
        needs = set()
        for c in plan_cells:
            for pre in (getattr(c, "preconditions", None) or []):
                prop = _cond_get(pre, "property")
                if _cond_get(pre, "operator") == "==" and prop in produced:
                    if produced[prop] - {repr(_cond_get(pre, "value"))}:
                        needs.add((prop, repr(_cond_get(pre, "value"))))
        bridges: List[Cell] = []
        if settings.mcts_max_bridges > 0 and needs:
            for c in orch.loaded_cells.values():
                if any(_cond_get(q, "operator") == "==" and _cond_get(q, "target") == "output_var"
                       and (_cond_get(q, "property"), repr(_cond_get(q, "value"))) in needs for q in (getattr(c, "postconditions", None) or [])):
                    bridges.append(c)
                    proto.setdefault(c.cell_id, c)
        self._trace("mcts_space", clauses=len(clauses), groups=[[g[0] for g in gl] for gl in groups],
                    bridges=[b.cell_id for b in bridges])

        # ---- state: (next_clause, ((cell_id, clause_idx), ...), bridges_used) ----------------
        N = len(clauses)

        def materialize(path_t) -> List[Cell]:
            out: List[Cell] = []
            for cid, idx in path_t:
                c = proto[cid].clone() if hasattr(proto[cid], "clone") else proto[cid]
                if idx >= 0:
                    c.matched_clause_idx = idx
                    c.clause_literals = list(literals[idx])
                out.append(c)
            return out

        def connect(path_cells: List[Cell], cell_ids: List[str], idx: int) -> Optional[List[Tuple[str, int]]]:
            """Cells (and any bridge/producer) needed to append `cell_ids` after `path_cells`, or None."""
            appended: List[Tuple[str, int]] = []
            cur = list(path_cells)
            for cid in cell_ids:
                nxt = proto[cid]
                if cur:
                    if not (self.step_unifies(cur[-1], nxt, prev_path=cur) or self.step_unifies_dag(nxt, cur)):
                        br = self.find_bridge(cur[-1], nxt, pool, orch, prev_path=cur)
                        if br is not None:
                            cur.append(br)
                            appended.append((br.cell_id, -1))
                            proto.setdefault(br.cell_id, br)
                            if not (self.step_unifies(br, nxt, prev_path=cur) or self.step_unifies_dag(nxt, cur)):
                                return None
                        else:
                            tagged = nxt.clone() if hasattr(nxt, "clone") else nxt
                            tagged.matched_clause_idx = idx
                            prod = self._find_type_producer(tagged, cur, pool, clauses, orch)
                            if prod is None:
                                return None
                            cur.append(prod)
                            appended.append((prod.cell_id, idx))
                            proto.setdefault(prod.cell_id, prod)
                cur.append(nxt)
                appended.append((cid, idx))
            return appended

        def actions(state):
            j, path_t, nb = state
            if j >= N:
                return []
            cells = materialize(path_t)
            acts: List[Tuple] = []
            for gi, (ids, _) in enumerate(groups[j]):
                if connect(cells, ids, j) is not None:
                    acts.append(("group", gi))
            if not groups[j] or not acts:
                acts.append(("skip", 0))
            if cells and nb < settings.mcts_max_bridges:
                tail = cells[-1]
                tail_post = {(_cond_get(q, "property"), repr(_cond_get(q, "value"))) for q in (getattr(tail, "postconditions", None) or [])
                             if _cond_get(q, "operator") == "=="}
                props_of_tail = {p for p, _ in tail_post}
                for b in bridges:
                    flips = any(_cond_get(q, "property") in props_of_tail
                                and (_cond_get(q, "property"), repr(_cond_get(q, "value"))) not in tail_post
                                for q in (getattr(b, "postconditions", None) or []))
                    if not flips:
                        continue
                    if b.cell_id not in {c for c, _ in path_t} and self.step_unifies(cells[-1], b, prev_path=cells):
                        acts.append(("bridge", b.cell_id))
            return acts

        def step(state, action):
            j, path_t, nb = state
            kind, arg = action
            if kind == "skip":
                return (j + 1, path_t, nb)
            if kind == "bridge":
                return (j, path_t + ((arg, path_t[-1][1] if path_t else -1),), nb + 1)   # a bridge extends the previous cell's clause
            ids = groups[j][arg][0]
            add = connect(materialize(path_t), ids, j) or []
            return (j + 1, path_t + tuple(add), nb)

        import math as _math

        def prior(state, action):
            # softmax rollout policy: weight = exp(heuristic / temperature)
            return _math.exp(_score(state, action) / settings.mcts_rollout_temperature)

        def _score(state, action):
            j = state[0]
            kind, arg = action
            if kind == "group":
                return 0.05 + groups[j][arg][1]
            if kind == "bridge":
                # A bridge is promising right after a cell whose tree-declared postcondition asserts the very
                # property the bridge flips (e.g. FFT declares is_complex == True, the bridge establishes False).
                tail = proto.get(state[1][-1][0]) if state[1] else None
                flips = False
                cand = proto.get(arg)
                if tail is not None and cand is not None:
                    tail_post = {(_cond_get(q, "property"), repr(_cond_get(q, "value"))) for q in (getattr(tail, "postconditions", None) or [])
                                 if _cond_get(q, "operator") == "=="}
                    props_of_tail = {p for p, _ in tail_post}
                    flips = any(_cond_get(q, "property") in props_of_tail
                                and (_cond_get(q, "property"), repr(_cond_get(q, "value"))) not in tail_post
                                for q in (getattr(cand, "postconditions", None) or []))
                if not flips:
                    return 0.02
                return 0.9
            return 0.01

        from unification import UnificationGate, UnificationFailure
        from preflight import PreflightLinter
        try:
            from ..token_evidence import audit as _audit
        except (ImportError, ValueError):
            from token_evidence import audit as _audit
        verdicts: Dict[Tuple, Dict[str, Any]] = {}
        reg = TypeRegistry.get_instance()

        unif_reward_cache: Dict[tuple, float] = {}

        def _fast_reward(path_t: tuple) -> float:
            if not path_t:
                return 0.0
            if path_t in unif_reward_cache:
                return unif_reward_cache[path_t]
            mat_cells = materialize(path_t)
            # Fast structural port-compatibility check between adjacent steps
            compat_factor = 1.0
            for i in range(len(mat_cells) - 1):
                c1, c2 = mat_cells[i], mat_cells[i + 1]
                p_out = getattr(c1, "primary_output", None)
                p_in = getattr(c2, "primary_input", None)
                if p_out is not None and p_in is not None:
                    t_out = str(getattr(getattr(p_out, "signature", p_out), "type_name", "") or "").lower()
                    t_in = str(getattr(getattr(p_in, "signature", p_in), "type_name", "") or "").lower()
                    if t_out and t_in and not reg.is_declared_top(t_out) and not reg.is_declared_top(t_in):
                        if not (reg.is_subtype(t_out, t_in) or reg.is_subtype(t_in, t_out)):
                            compat_factor *= 0.5

            # Stage 3 UnificationGate validity check inside rollout reward
            unif_factor = 1.0
            try:
                gate_r = UnificationGate(orchestrator=orch)
                ctx_r = ExecutionContext(prompt=prompt)
                ur_r = gate_r.unify_pipeline(mat_cells, ctx_r, extract_llm_slots=False)
                if ur_r.is_bottom() or not getattr(ur_r, "value", None):
                    unif_factor = 0.05
            except Exception:
                unif_factor = 0.05

            res = _audit(mat_cells, prompt, orch)
            infos = [c for c in res.get("clauses", []) if c["status"] in ("served", "UNSERVED")]
            served = (sum(1 for c in infos if c["status"] == "served") / len(infos)) if infos else 0.0
            quality = (sum(c["quality"] for c in infos) / len(infos)) if infos else 0.0
            n_issues = len(res.get("issues", []))
            val = (served * quality * compat_factor * unif_factor) / (1.0 + n_issues)
            unif_reward_cache[path_t] = val
            return val

        def reward(state) -> float:
            return _fast_reward(state[1])

        search = MonteCarloTreeSearch(
            actions=actions, step=step, reward=reward, prior=prior,
            state_key=lambda s: (s[0], s[1], s[2]),
            describe_action=lambda a: f"{a[0]}:{a[1]}",
            iterations=settings.mcts_iterations, exploration=settings.mcts_exploration,
            widening_alpha=settings.mcts_widening_alpha, max_seconds=settings.mcts_max_seconds,
            max_rollout_depth=N + settings.mcts_max_bridges + 2, seed=settings.mcts_seed,
        )
        result = search.search((0, tuple(), 0))
        best_path = result.best_state[1]
        cur_reward = result.best_reward

        # Final verification pass: run compilation and preflight ONCE on the winning trajectory
        nviol, note = 0, None
        final_cells: List[Cell] = []
        if best_path and cur_reward > 0.0:
            final_cells = materialize(best_path)
            try:
                gate = UnificationGate(orchestrator=orch)
                run_ctx = ExecutionContext(prompt=prompt)
                ur = gate.unify_pipeline(final_cells, run_ctx, extract_llm_slots=False)
                if ur.is_bottom():
                    nviol, note = 10, "unification produced bottom"
                else:
                    code = gate.emit_code(ur.value, run_ctx, extract_llm_slots=False)
                    lint = PreflightLinter.lint(ur.value, prompt=prompt, code_str=code, orchestrator=orch)
                    nviol = len(lint.violations)
                    note = lint.violations[0][:140] if lint.violations else None
            except UnificationFailure as exc:
                nviol, note = 10, f"refused: {str(exc)[:120]}"
            except Exception as exc:
                nviol, note = 10, f"verifier error {type(exc).__name__}: {str(exc)[:100]}"
            verdicts[best_path] = {"served": round(cur_reward, 3), "quality": round(cur_reward, 3), "violations": nviol, "first": note}
            if nviol > 0:
                result.best_reward = cur_reward / (1.0 + nviol)

        v = verdicts.get(best_path, {})
        self._trace("mcts_summary", iterations=result.iterations, evaluations=result.evaluations, cache_hits=result.cache_hits,
                    seed=result.seed, elapsed_ms=round(result.elapsed_ms, 1), stopped_by=result.stopped_by,
                    best_reward=round(result.best_reward, 4), best_plan=[c for c, _ in best_path], verdict=v,
                    root=result.root_stats[:8], best_line=result.best_path_stats)
        if not best_path or result.best_reward <= 0.0 or nviol >= 10:
            reason = "Monte Carlo search found no plan with positive reward"
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []
        return final_cells if final_cells else materialize(best_path)
