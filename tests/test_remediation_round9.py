"""
tests/test_remediation_round9.py - regression tests for defects found while analysing the NSTL benchmark run.

1. `_unique_var` was used by unification.py but only defined in synthesis.py (NameError on control-flow prompts).
2. Dimension notation ("5x5") produced no numeric literal, so `CV2_GAUSSIAN_BLUR.ksize` stayed unresolved.
3. A REQUIRED numeric port could take a literal from a sibling clause (blur sigma stole Canny's threshold),
   producing a program that runs but is wrong. Nothing here is keyed to a specific prompt or port name.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator
from unification import UnificationGate, ExecutionContext, _split_dimension_token


class TestHelpersAndLexing(unittest.TestCase):
    def test_unique_var_is_shared_not_duplicated(self):
        import unification, synthesis, utils
        self.assertIs(unification._unique_var, utils.unique_var)
        self.assertIs(synthesis._unique_var, utils.unique_var)

    def test_dimension_token_split(self):
        self.assertEqual(_split_dimension_token("5x5"), [(0, "5"), (2, "5")])
        self.assertEqual(_split_dimension_token("800x600"), [(0, "800"), (4, "600")])
        self.assertEqual(_split_dimension_token("3x3x3"), [(0, "3"), (2, "3"), (4, "3")])
        for not_dim in ("box", "x5", "5x", "5xx5", "0x1F", "10", "1.2.3x4", ".5x5"):
            self.assertEqual(_split_dimension_token(not_dim), [], not_dim)

    def test_dimension_notation_lexes_like_spelled_out_form(self):
        def nums(p):
            return [v for _, k, v in ExecutionContext._extract_universal_literals(p) if k == "numeric"]
        self.assertEqual(nums("blur with kernel size 5x5"), nums("blur with kernel size 5 by 5"))


class TestClauseLocalNumericBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = LatticeOrchestrator(trees_directory=str(ROOT_DIR / "trees"), db_path=str(ROOT_DIR / "trees" / "lattice.db"))
        cls.orch.load_all_json_trees()
        cls.orch.build_topology()
        cls.idx = {str(k).lower(): v for k, v in cls.orch.loaded_cells.items()}

    def _emit(self, ids, prompt):
        from route_methods.base import RouteMethod
        cells = [self.idx[i.lower()] for i in ids]
        RouteMethod.tag_cells_with_clause_indices(cells, prompt, orchestrator=self.orch)
        gate = UnificationGate(orchestrator=self.orch)
        ctx = ExecutionContext(prompt=prompt)
        res = gate.unify_pipeline(cells, ctx)
        self.assertFalse(res.is_bottom(), "pipeline refused")
        return gate.emit_code(res.value, ctx)

    IDS = ["CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY", "CV2_GAUSSIAN_BLUR", "CV2_CANNY"]

    def test_sibling_clause_literals_are_not_stolen(self):
        p = ('read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5, '
             'then detect edges with Canny thresholds 50 and 150')
        code = self._emit(self.IDS, p)
        self.assertIn("GaussianBlur(ndarray_2, (5, 5), 0.0", code)
        self.assertIn("Canny(ndarray_3, 50, 150", code)

    def test_dimension_notation_binds_kernel(self):
        p = 'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5x5'
        self.assertIn("(5, 5)", self._emit(self.IDS[:3], p))


class TestRound9Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = LatticeOrchestrator(trees_directory=str(ROOT_DIR / "trees"), db_path=str(ROOT_DIR / "trees" / "lattice.db"))
        cls.orch.load_all_json_trees()
        cls.orch.build_topology()
        cls.idx = {str(k).lower(): v for k, v in cls.orch.loaded_cells.items()}

    def _cells(self, ids):
        return [self.idx[i.lower()] for i in ids]

    def test_tagger_is_order_optimal_not_greedy(self):
        """A redundant out-of-order cell with a strong hit used to claim clause 1 and block every correct later tag."""
        from route_methods.base import RouteMethod
        P = ('read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
             'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"')
        ids = ["CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY", "CV2_GAUSSIAN_BLUR", "PILLOW_NDARRAY_GENERIC_TO_PILLOW",
               "PILLOW_FILTER_FIND_EDGES", "CV2_CANNY", "PILLOW_CONVERT_TO_GRAYSCALE", "PILLOW_IMAGE_SAVE_PNG"]
        cells = self._cells(ids)
        RouteMethod.tag_cells_with_clause_indices(cells, P, orchestrator=self.orch)
        tag = {c.cell_id: c.matched_clause_idx for c in cells}
        self.assertEqual(tag["CV2_GAUSSIAN_BLUR"], 2)
        self.assertEqual(tag["CV2_CANNY"], 3)
        clauses = [c.matched_clause_idx for c in cells]
        self.assertEqual(clauses, sorted(clauses), "clause tags must be non-decreasing along the path")

    def test_carrier_transform_with_discarded_result_is_a_lint_violation(self):
        """cv2 plan whose Canny / FIND_EDGES results are never read used to lint clean and save the wrong image."""
        from route_methods.base import RouteMethod
        from preflight import PreflightLinter
        P = ('read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
             'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"')
        ids = ["CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY", "CV2_GAUSSIAN_BLUR", "PILLOW_NDARRAY_GENERIC_TO_PILLOW",
               "PILLOW_FILTER_FIND_EDGES", "CV2_CANNY", "PILLOW_CONVERT_TO_GRAYSCALE", "PILLOW_IMAGE_SAVE_PNG"]
        cells = self._cells(ids)
        RouteMethod.tag_cells_with_clause_indices(cells, P, orchestrator=self.orch)
        gate = UnificationGate(orchestrator=self.orch)
        ctx = ExecutionContext(prompt=P)
        res = gate.unify_pipeline(cells, ctx)
        self.assertFalse(res.is_bottom())
        code = gate.emit_code(res.value, ctx)
        lint = PreflightLinter.lint(res.value, prompt=P, code_str=code, orchestrator=self.orch)
        dead = [v for v in lint.violations if "dead output" in v]
        self.assertTrue(any("CV2_CANNY" in v for v in dead), lint.violations)

    def test_scalar_observation_may_be_unused(self):
        """'get the mean of Y' legitimately produces a result nobody consumes; that must stay lint-clean."""
        from route_methods.base import RouteMethod
        from preflight import PreflightLinter
        P = 'load a csv file named "input.csv", normalize "X" column, drop null values, get the mean of the "Y" column'
        ids = ["PD_READ_CSV", "PD_SERIES_NORMALIZE", "PD_DROPNA", "PD_SERIES_MEAN"]
        cells = self._cells(ids)
        RouteMethod.tag_cells_with_clause_indices(cells, P, orchestrator=self.orch)
        gate = UnificationGate(orchestrator=self.orch)
        ctx = ExecutionContext(prompt=P)
        res = gate.unify_pipeline(cells, ctx)
        self.assertFalse(res.is_bottom())
        code = gate.emit_code(res.value, ctx)
        lint = PreflightLinter.lint(res.value, prompt=P, code_str=code, orchestrator=self.orch)
        self.assertFalse([v for v in lint.violations if "MEAN" in v.upper() and "dead output" in v], lint.violations)

    def test_unresolved_ports_are_structured(self):
        from unification import UnresolvedPlaceholderError
        from route_methods.base import RouteMethod
        P = 'read image "input.png", convert to grayscale, apply Gaussian blur'   # no kernel size stated
        cells = self._cells(["CV2_IMREAD", "CV2_CVT_COLOR_BGR2GRAY", "CV2_GAUSSIAN_BLUR"])
        RouteMethod.tag_cells_with_clause_indices(cells, P, orchestrator=self.orch)
        gate = UnificationGate(orchestrator=self.orch)
        ctx = ExecutionContext(prompt=P)
        res = gate.unify_pipeline(cells, ctx)
        self.assertFalse(res.is_bottom())
        with self.assertRaises(UnresolvedPlaceholderError) as cm:
            gate.emit_code(res.value, ctx)
        self.assertIn(("CV2_GAUSSIAN_BLUR", "ksize"), cm.exception.ports)

    def test_emission_defect_is_flagged_as_engine_defect(self):
        from unification import EmissionLivenessError
        self.assertTrue(EmissionLivenessError.is_engine_defect)

    def test_failure_record_kinds(self):
        import cli
        B = cli.PipelineDebugger._build_failure_record
        common = dict(cells=[1], internal_error_msg=None, unify_fail_reason=None, refusal_exc=None,
                      refusal_reason=None, sandbox_res={}, final_code="x")
        self.assertIsNone(B(**{**common, "sandbox_res": {"success": True}}))
        self.assertEqual(B(**{**common, "cells": []})["kind"], "PLANNING_FAILED")
        self.assertEqual(B(**{**common, "internal_error_msg": "NameError: x"})["kind"], "INTERNAL_ERROR")
        self.assertEqual(B(**{**common, "unify_fail_reason": "bad"})["kind"], "UNIFICATION_FAILED")
        self.assertEqual(B(**{**common, "sandbox_res": {"preflight_lint_violations": ["v"]}})["kind"], "LINT_REJECTED")
        self.assertEqual(B(**{**common, "sandbox_res": {"success": False, "error": "boom"}})["kind"], "SANDBOX_FAILED")
        from unification import EmissionLivenessError, UnresolvedPlaceholderError
        e1 = EmissionLivenessError("Emitted code has invalid syntax")
        self.assertEqual(B(**{**common, "refusal_exc": e1, "refusal_reason": str(e1)})["kind"], "EMISSION_DEFECT")
        e2 = UnresolvedPlaceholderError("x", ports=[("A", "b")])
        rec = B(**{**common, "refusal_exc": e2, "refusal_reason": "x"})
        self.assertEqual(rec["kind"], "UNRESOLVED_PORTS")
        self.assertEqual(rec["unresolved_ports"], ["A.b"])


if __name__ == "__main__":
    unittest.main()
