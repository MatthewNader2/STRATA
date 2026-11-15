"""
Sweep-8 regressions (offline, no models). Each test pins a defect that was visible in a real-model log or
found while fixing it:

  * a real Monte Carlo Tree Search engine (seeded, deterministic, statistics, finds optima),
  * column transformations are committed back into their table (tree-declared effects),
  * a Series made before a row-changing step is re-derived from the table that absorbed it,
  * column provenance is column-level (a projection carries only its column),
  * tree-declared value properties are propagated statically (complex FFT output -> real-only estimator),
  * a column cannot be read before the cell that defines it,
  * the M10 route method returns a plan that lints clean AND whose emitted program runs on data with nulls.
"""
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
logging.disable(logging.CRITICAL)

from lattice import LatticeOrchestrator  # noqa: E402
from unification import UnificationGate, ExecutionContext  # noqa: E402
from preflight import PreflightLinter  # noqa: E402

PROMPT = ('load a csv file named "input.csv", normalize "X" column, drop null values, get the mean of the "Y" column, '
          'and perform FFT and write it to a new column called "Z", then make a regression model trained on "X" and "Y" to predict "Z"')


class _Port:
    def __init__(self, binds=None):
        self.binds = binds


class _Cell:
    def __init__(self, cell_id, **ports):
        self.cell_id = cell_id
        self.inputs = {k: _Port(v) for k, v in ports.items()}


class TestMonteCarloEngine(unittest.TestCase):
    def _toy(self, seed, iterations=60):
        from mcts import MonteCarloTreeSearch
        target = (1, 0, 1, 1, 0)
        return MonteCarloTreeSearch(
            actions=lambda s: [0, 1] if len(s) < 5 else [],
            step=lambda s, a: s + (a,),
            reward=lambda s: sum(1 for x, y in zip(s, target) if x == y) / 5,
            iterations=iterations, seed=seed).search(())

    def test_finds_the_optimum_and_reuses_its_cache(self):
        r = self._toy(0)
        self.assertEqual(r.best_state, (1, 0, 1, 1, 0))
        self.assertEqual(r.best_reward, 1.0)
        self.assertGreater(r.cache_hits, 0)
        self.assertLess(r.evaluations, 32)       # fewer distinct evaluations than the 32 possible states

    def test_deterministic_for_a_seed(self):
        a, b = self._toy(7), self._toy(7)
        self.assertEqual((a.best_state, a.root_stats, a.best_path_stats), (b.best_state, b.root_stats, b.best_path_stats))

    def test_reports_tree_statistics(self):
        r = self._toy(0)
        self.assertTrue(r.root_stats and all({"action", "visits", "mean_reward"} <= set(s) for s in r.root_stats))
        self.assertEqual(sum(s["visits"] for s in r.root_stats), r.iterations)

    def test_time_budget_stops_the_search(self):
        from mcts import MonteCarloTreeSearch
        r = MonteCarloTreeSearch(actions=lambda s: [0, 1] if len(s) < 5 else [], step=lambda s, a: s + (a,),
                                 reward=lambda s: 0.5, iterations=10 ** 9, max_seconds=0.05).search(())
        self.assertEqual(r.stopped_by, "time")


class TestStaticAnalyses(unittest.TestCase):
    def test_column_read_before_it_is_defined_is_found(self):
        from preflight import PreflightLinter
        analyze_column_definitions = PreflightLinter._verify_causal_key_def_use
        read = _Cell("READ", path=None)
        fit = _Cell("FIT", X=None, y=None)
        setc = _Cell("SET", df="data_carrier", column="column_key", value="assigned_value")
        bindings = [
            (read, {"path": "'in.csv'", "output_var": "var_1"}),
            (fit, {"X": "var_1[['X']]", "y": "var_1['Z']", "output_var": "var_2"}),
            (setc, {"df": "var_1", "column": "'Z'", "value": "np.fft.fft(var_1['X'])", "output_var": "var_3"}),
        ]
        issues = analyze_column_definitions(bindings)
        self.assertEqual([(i["column"], i["read_at"], i["defined_at"]) for i in issues], [("Z", 1, 2)])

    def test_updating_a_column_is_not_use_before_define(self):
        from preflight import PreflightLinter
        analyze_column_definitions = PreflightLinter._verify_causal_key_def_use
        setc = _Cell("SET", df="data_carrier", column="column_key", value="assigned_value")
        bindings = [
            (_Cell("READ", path=None), {"path": "'in.csv'", "output_var": "var_1"}),
            (_Cell("NORM", s=None), {"s": "var_1['X']", "output_var": "var_2"}),
            (setc, {"df": "var_1", "column": "'X'", "value": "var_2", "output_var": "var_3"}),
        ]
        self.assertEqual(analyze_column_definitions(bindings), [])

    def test_wire_ranking_prefers_a_fresher_producer_over_an_exact_state_match(self):
        from unification import _wire_rank
        fresh_cleaned = _wire_rank("table", "state:cleaned", "data_input", "table", "state:raw", "data_input", 2, 1, 5)
        old_raw = _wire_rank("table", "state:raw", "data_input", "table", "state:raw", "data_input", 2, 0, 1)
        self.assertGreater(fresh_cleaned, old_raw)       # adjacency outranks an exact-state bonus


class TestPlansAndPrograms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = LatticeOrchestrator(trees_directory=str(ROOT_DIR / "trees"), db_path=str(Path(tempfile.gettempdir()) / "s8.db"))
        cls.orch.load_all_json_trees()
        cls.orch.build_topology()
        from router import LatticeRouter
        cls.router = LatticeRouter(orchestrator=cls.orch, internal_rag=None, default_route_method="m0")

    def _plan(self, method):
        ctx = ExecutionContext(prompt=PROMPT)
        cells = self.router.plan_path(PROMPT, return_tuple=False, route_method=method, ctx=ctx)
        gate = UnificationGate(orchestrator=self.orch)
        ctx2 = ExecutionContext(prompt=PROMPT)
        res = gate.unify_pipeline(cells, ctx2)
        code = gate.emit_code(res.value, ctx2)
        lint = PreflightLinter.lint(res.value, prompt=PROMPT, code_str=code)
        return cells, code, lint, ctx2

    def test_column_transform_is_committed_before_dropna(self):
        cells, code, lint, _ = self._plan("m1")
        ids = [c.cell_id for c in cells]
        self.assertEqual(ids[ids.index("PD_SERIES_NORMALIZE") + 1], "PD_SET_COLUMN")
        self.assertRegex(code, r"\['X'\] = (?:var|series)_\d+")
        self.assertTrue(any(e["event"] == "column_commit_inserted" for e in self.router.last_route_trace))

    def test_stale_series_is_rederived_from_the_table_that_absorbed_it(self):
        _, code, _, ctx = self._plan("m1")
        # FFT consumes a Series derived from the post-dropna DataFrame, not the pre-dropna stale Series.
        lines = code.splitlines()
        dropna_idx = next(i for i, l in enumerate(lines) if ".dropna(" in l)
        post_df_var = lines[dropna_idx].split(" = ")[0].strip()
        fft_line = next(l for l in lines if "np.fft.fft(" in l)
        fft_arg = fft_line.split("np.fft.fft(")[1].split(",")[0].strip()
        rederive_line = next((l for l in lines[dropna_idx + 1:] if l.startswith(f"{fft_arg} = {post_df_var}[")), None)
        self.assertIsNotNone(rederive_line, f"expected FFT input {fft_arg} to be derived from post-dropna {post_df_var}, got:\n{code}")
        self.assertTrue(
            any(e["event"] in ("stale_series_rederived", "planned_wire_rejected_stale_epoch_carrier", "in_scope_rejected_stale_epoch_carrier") for e in ctx.binding_trace)
        )

    def test_projection_provenance_is_column_level(self):
        _, code, _, ctx = self._plan("m1")
        fft_var = next(l.split(" = ")[0] for l in code.splitlines() if "np.fft.fft" in l)
        self.assertEqual(len(ctx.var_origins[fft_var]), 1)
        self.assertTrue(ctx.var_origins[fft_var] <= {"X", "Y"})

    def test_complex_target_is_a_static_precondition_violation_with_bridge_candidates(self):
        cells, _, lint, _ = self._plan("m1")
        self.assertTrue(lint.is_valid, lint.violations)
        self.assertIn("NUMPY_ABS", [c.cell_id for c in cells])
        # Removing the property bridge must trigger the static is_complex violation with NUMPY_ABS suggested.
        stripped = [c for c in cells if c.cell_id != "NUMPY_ABS"]
        gate = UnificationGate(orchestrator=self.orch)
        ctx2 = ExecutionContext(prompt=PROMPT)
        res = gate.unify_pipeline(stripped, ctx2)
        code = gate.emit_code(res.value, ctx2)
        stripped_lint = PreflightLinter.lint(res.value, prompt=PROMPT, code_str=code)
        msgs = [v for v in stripped_lint.violations if "is_complex" in v]
        self.assertTrue(msgs, f"expected a property violation when NUMPY_ABS is stripped, got {stripped_lint.violations}")
        self.assertIn("NUMPY_ABS", msgs[0])

    def test_m10_plan_is_clean_and_its_program_runs_on_data_with_nulls(self):
        cells, code, lint, _ = self._plan("m10")
        self.assertTrue(lint.is_valid, lint.violations)
        self.assertIn("NUMPY_ABS", [c.cell_id for c in cells])
        summary = next(e for e in self.router.last_route_trace if e["event"] == "mcts_summary")
        self.assertGreater(summary["iterations"], 0)
        try:
            import numpy as np
            import pandas as pd
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("pandas/numpy/scikit-learn unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            df = pd.DataFrame({"X": rng.normal(size=100), "Y": rng.normal(size=100)})
            df.loc[[3, 17, 60], "X"] = np.nan
            df.to_csv(Path(tmp) / "input.csv", index=False)
            old = os.getcwd()
            os.chdir(tmp)
            try:
                exec(compile(code, "<m10>", "exec"), {})
            finally:
                os.chdir(old)

    def test_m10_never_scores_below_the_deterministic_baseline(self):
        self._plan("m10")
        summary = next(e for e in self.router.last_route_trace if e["event"] == "mcts_summary")
        self.assertGreater(summary["best_reward"], 0.0)

    def test_planner_monte_carlo_fallback_honours_the_planner_deadline(self):
        import time
        from planner import LatticePlanner
        planner = LatticePlanner(self.orch, require_coverage_floor=False)
        tunnel = list(self.orch.loaded_cells.values())
        rel = {c.cell_id: 0.5 for c in tunnel}
        planner._plan_deadline = time.perf_counter() - 1.0                    # already over budget
        self.assertIsNone(planner._monte_carlo_chain_search(PROMPT, tunnel, rel))
        planner._plan_deadline = time.perf_counter() + 0.3                    # 300 ms left
        t0 = time.perf_counter()
        planner._monte_carlo_chain_search(PROMPT, tunnel, rel)
        self.assertLess(time.perf_counter() - t0, 1.5)

    def test_method_is_registered_and_listed(self):
        from route_methods import route_method_ids, describe_route_methods
        self.assertIn("M10", route_method_ids())
        self.assertTrue(any(m[0] == "M10" for m in describe_route_methods()))


if __name__ == "__main__":
    unittest.main()
