"""
tests/test_end_to_end_offline.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Round 6 (R6-3): offline end-to-end synthesis tests. No models required.

Builds FIXED cell lists from the loaded lattice (case-insensitive cell_id lookup)
and drives the exact same chain the CLI main path drives:
    ExecutionContext(prompt=...) -> gate.unify_pipeline(cells, ctx)
                                 -> gate.emit_code(pipeline_bindings, ctx)
                                 -> PreflightLinter.lint(pipeline_bindings, prompt, code)
Clause tagging happens inside unify_pipeline (RouteMethod.tag_cells_with_clause_indices),
exactly as in the CLI.

Asserts NO EXCEPTION anywhere in the chain (R6-1 regression gate). Does NOT assert
code text. Also unit-tests the R6-2 qualifier flattening contract.
"""

from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator, PortSignature, AlgebraicSignature
from preflight import PreflightLinter
from unification import (
    UnificationGate,
    ExecutionContext,
    UnificationFailure,
    _shape_compatible,
)

BENCHMARK_PROMPT = (
    'load a csv file named "input.csv", normalize "X" column, drop null values, '
    'get the mean of the "Y" column, and perform FFT and write it to a new column called "Z", '
    'then make a regression model trained on "X" and "Y" to predict "Z"'
)

# Fixed paths (cell ids looked up case-insensitively from the lattice).
CASES = {
    "three_cell_minimal": [
        "PD_READ_CSV",
        "PD_SERIES_NORMALIZE",
        "PD_DROPNA",
    ],
    "m1_path_from_log": [
        "PD_READ_CSV",
        "PD_GET_DUMMIES",
        "PD_GET_DUMMIES",
        "PD_SERIES_NORMALIZE",
        "PD_DROPNA",
        "PD_ROLLING_MEAN",
        "NUMPY_FFT_FFT",
        "PD_SET_COLUMN",
        "sklearn.tree.DecisionTreeRegressor.fit",
        "sklearn.linear_model.LinearRegression.predict",
    ],
    "reference_like": [
        "PD_READ_CSV",
        "PD_SERIES_NORMALIZE",
        "PD_DROPNA",
        "PD_SERIES_MEAN",
        "NUMPY_FFT_FFT",
        "NUMPY_ABS",
        "PD_SET_COLUMN",
        "sklearn.linear_model.LinearRegression.fit",
        "sklearn.linear_model.LinearRegression.predict",
    ],
}


class TestEndToEndOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trees_dir = str(ROOT_DIR / "trees")
        db_path = str(ROOT_DIR / "trees" / "lattice.db")
        cls.orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=db_path)
        # Tests load the JSON trees directly (the source of truth, R6-3): the
        # lattice.db cache is a runtime artifact whose cell construction may
        # diverge from the declared trees.
        cls.orch.load_all_json_trees()
        cls.orch.build_topology()
        # Case-insensitive cell_id index (derived from the lattice, not hardcoded)
        cls._ci_index = {
            str(cid).strip().lower(): cell
            for cid, cell in cls.orch.loaded_cells.items()
        }

    def _cells(self, ids):
        out = []
        for cid in ids:
            cell = self._ci_index.get(str(cid).strip().lower())
            self.assertIsNotNone(cell, f"cell {cid!r} not present in lattice")
            out.append(cell)
        return out

    def _run_cli_chain(self, cell_ids):
        """Mirrors the CLI main path: unify_pipeline -> emit_code -> PreflightLinter.lint.

        Returns (bindings, code, lint_res) on success, or (None, None, None) when the
        chain REFUSES with a declared failure (UnificationFailure family — e.g.
        unresolved ports after estimator-identity gating). Any OTHER exception is an
        internal engine crash and fails the test (R6-1/R6-2 regression gate).
        """
        cells = self._cells(cell_ids)
        # The CLI plans through router.plan_path, whose post-pass tags cells with their clause and writes
        # column transformations back into their table (tree-declared column_transform_effects). Mirror it,
        # otherwise a path that transforms a column before dropping nulls is judged without the step that
        # makes it correct.
        from route_methods.base import RouteMethod
        from route_methods.m1_clause_anchor import M1ClauseAnchorRouteMethod
        RouteMethod.tag_cells_with_clause_indices(cells, BENCHMARK_PROMPT, orchestrator=self.orch)
        cells = M1ClauseAnchorRouteMethod(orchestrator=self.orch).insert_column_commits(cells)
        gate = UnificationGate(orchestrator=self.orch)
        ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
        try:
            unify_res = gate.unify_pipeline(cells, ctx)
            if unify_res.is_bottom():
                return None, None, None
            pipeline_bindings = unify_res.value
            code = gate.emit_code(pipeline_bindings, ctx)
        except UnificationFailure:
            # Declared refusal — honest, structured, category-correct (R6-4/R6-6)
            return None, None, None
        lint_res = PreflightLinter.lint(pipeline_bindings, prompt=BENCHMARK_PROMPT, code_str=code)
        return pipeline_bindings, code, lint_res

    @staticmethod
    def _exec_on_fixture(code):
        """Runs emitted code in a temp dir against a fixture input.csv containing nulls.
        Returns the executed namespace. A crash here is a real defect, not a stale expectation."""
        import os, tempfile
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 40
        x = rng.uniform(10, 50, n)
        df = pd.DataFrame({"X": x, "Y": 2.0 * x + rng.normal(0, 1, n)})
        df.loc[[3, 17], "X"] = np.nan
        df.loc[[8], "Y"] = np.nan
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                df.to_csv("input.csv", index=False)
                ns = {}
                exec(compile(code, "<emitted>", "exec"), ns)
            finally:
                os.chdir(old)
        return ns

    @staticmethod
    def _assert_in_order(testcase, bindings, expected_ids):
        ids = [b[0].cell_id for b in bindings]
        pos = -1
        for e in expected_ids:
            testcase.assertIn(e, ids[pos + 1:], f"{e} missing or out of order in {ids}")
            pos = ids.index(e, pos + 1)

    def test_three_cell_minimal_runs(self):
        bindings, code, lint_res = self._run_cli_chain(CASES["three_cell_minimal"])
        if bindings is None:
            self.fail("three-cell pipeline refused — it must unify and emit")
        # Behavioural contract (not a cell count): the requested cells appear in order, and the emitted
        # program really min-max-normalises X and leaves no nulls. Helper cells the planner adds
        # (column select / commit) are allowed; their number is an implementation detail.
        self._assert_in_order(self, bindings, CASES["three_cell_minimal"])
        import pandas as pd
        ns = self._exec_on_fixture(code)
        out = [v for v in ns.values() if isinstance(v, pd.DataFrame)][-1]
        self.assertFalse(out.isna().any().any(), "dropna left nulls")
        self.assertAlmostEqual(float(out["X"].min()), 0.0)
        self.assertAlmostEqual(float(out["X"].max()), 1.0)

    def test_m1_path_from_log_runs(self):
        bindings, code, lint_res = self._run_cli_chain(CASES["m1_path_from_log"])
        if bindings is None:
            # The M1 log path pairs DecisionTreeRegressor.fit with
            # LinearRegression.predict; with R6-6 estimator-identity gating the
            # cross-class model port cannot be bound and the chain refuses with a
            # declared failure. That is the intended honest outcome.
            return
        self.assertEqual(len(bindings), 10)

    def test_reference_like_path_runs(self):
        bindings, code, lint_res = self._run_cli_chain(CASES["reference_like"])
        if bindings is None:
            self.fail("reference-like pipeline refused — it must unify and emit")
        self._assert_in_order(self, bindings, CASES["reference_like"])
        import numpy as np
        import pandas as pd
        ns = self._exec_on_fixture(code)
        frames = [v for v in ns.values() if isinstance(v, pd.DataFrame)]
        self.assertTrue(any("Z" in f.columns for f in frames), "FFT result was never written to column Z")
        n_rows = int(frames[0].dropna().shape[0]) if frames else 0
        preds = [v for k, v in ns.items() if k.startswith("ndarray_") and getattr(v, "ndim", 0) == 1]
        self.assertTrue(preds and len(preds[-1]) == n_rows, "regression output must have one value per cleaned row")

    def test_reference_like_path_lints_clean(self):
        """R6-9 acceptance: zero false positives on the reference-like path."""
        bindings, code, lint_res = self._run_cli_chain(CASES["reference_like"])
        self.assertIsNotNone(bindings, "reference-like pipeline must unify and emit")
        self.assertTrue(
            lint_res.is_valid,
            "reference-like path must pass preflight with zero violations; got: "
            + "; ".join(lint_res.violations),
        )

    def test_emitted_code_compiles(self):
        """Emitted code must at least be syntactically valid Python for all fixed paths."""
        import ast
        for name in CASES:
            with self.subTest(case=name):
                bindings, code, _ = self._run_cli_chain(CASES[name])
                if bindings is None:
                    continue  # declared refusal — nothing emitted, nothing to parse
                self.assertTrue(str(code).strip(), f"{name}: emitted empty code")
                ast.parse(code)


class TestQualifierFlattening(unittest.TestCase):
    """R6-2: _shape_compatible must flatten qualifier tuples of ANY length."""

    def _out_with_quals(self, quals):
        return PortSignature(
            name="out",
            signature=AlgebraicSignature(
                type_name="ndarray",
                state="scaled_features",
                qualifiers=frozenset({quals}) if not isinstance(quals, str) else quals,
            ),
        )

    def _consumer_rejecting_complex(self):
        return PortSignature(
            name="in",
            signature=AlgebraicSignature(type_name="ndarray", state="scaled_features"),
            rejected_qualifiers=["complex"],
        )

    def test_single_element_tuple_flattens_without_error(self):
        # Regression: q[1] on ("complex",) raised IndexError at HEAD (R6-2)
        out = self._out_with_quals(("complex",))
        self.assertFalse(_shape_compatible(out, self._consumer_rejecting_complex()))

    def test_two_element_tuple_flattens(self):
        out = self._out_with_quals(("a", "b"))
        # Must not raise; a/b are not rejected so shape stays compatible
        self.assertTrue(_shape_compatible(out, self._consumer_rejecting_complex()))

    def test_empty_tuple_flattens(self):
        out = self._out_with_quals(())
        self.assertTrue(_shape_compatible(out, self._consumer_rejecting_complex()))

    def test_three_element_tuple_flattens(self):
        out = self._out_with_quals(("x", "complex", "y"))
        self.assertFalse(_shape_compatible(out, self._consumer_rejecting_complex()))

    def test_bare_string_qualifier_flattens(self):
        out = self._out_with_quals("complex")
        self.assertFalse(_shape_compatible(out, self._consumer_rejecting_complex()))


if __name__ == "__main__":
    unittest.main()
