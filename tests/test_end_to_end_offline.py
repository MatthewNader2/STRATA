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
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
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

    def test_three_cell_minimal_runs(self):
        bindings, code, lint_res = self._run_cli_chain(CASES["three_cell_minimal"])
        if bindings is None:
            self.fail("three-cell pipeline refused — it must unify and emit")
        # +1: normalising X and then dropping nulls plans the tree-declared column commit between them
        self.assertEqual(len(bindings), 3 + 1)

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
        # +1: the column commit written back after normalisation (tree: column_transform_effects)
        self.assertEqual(len(bindings), 9 + 1)

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
                state="nd_tensor",
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
