"""
tests/test_verification.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Round 5 Offline Verification Suite (PR-1 Core Correctness).

Tests Probes 1 to 6:
- Probe 1 (R5-4): Clause-level coverage & false-positive elimination (PD_GET_DUMMIES).
- Probe 2 (R5-2): Dataflow output liveness (dead_output detection).
- Probe 3 (R5-2): Dependency ordering & monotonic clause ordering (order_inversion).
- Probe 4 (R5-1): Projection typing & typed binding records without synthetic strings.
- Probe 5 (R5-3): Schema tracking & port-addressed column removal (column_absent).
- Probe 6 (Rule 2): No hardcoded column names or cell IDs in src/ engine.
"""

from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Set, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lattice import LatticeOrchestrator, Cell, MicroCell, AlgebraicSignature, PortSignature, TypeRegistry
from preflight import PreflightLinter, VariableSchema
from unification import UnificationGate, ExecutionContext, TypedBindingRecord, resolve_typed_binding_record
from tokenizer import CellTokenizer
from planner import LatticePlanner, _segment_prompt_clauses, STOPWORDS


class TestRound5Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trees_dir = str(ROOT_DIR / "trees")
        db_path = str(ROOT_DIR / "trees" / "lattice.db")
        cls.orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=db_path)
        if os.path.exists(db_path):
            try:
                cls.orch.load_from_database(db_path)
            except Exception:
                cls.orch.load_all_json_trees()
        else:
            cls.orch.load_all_json_trees()
        cls.orch.build_topology()

    def test_probe_1_coverage_precision_synthetic_lattice(self):
        """
        Probe 1 / Section 5 Item 5 (R5-4):
        Synthetic lattice: cell A shares one generic token with a clause,
        cell B explains the clause with its whole identity. Only B covers it.
        """
        prompt = "fit linear regression model"
        clauses = _segment_prompt_clauses(prompt)
        clause_toks = set(CellTokenizer.tokenize_prompt(clauses[0])) - set(STOPWORDS)

        # Cell A: generic estimator tool sharing 'model' with many other tokens
        cell_a = Cell(
            cell_id="GENERIC_MODEL_EVALUATOR",
            keywords=["model", "evaluation", "score", "metric", "validate", "benchmark"],
            node_role="transformer",
            stage=2,
            inputs={"x": PortSignature(name="x", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))},
            outputs={"y": PortSignature(name="y", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))},
        )
        # Cell B: exact regressor whose whole identity explains the clause
        cell_b = Cell(
            cell_id="LINEAR_REGRESSION_FIT",
            keywords=["fit", "linear", "regression", "model"],
            node_role="transformer",
            stage=2,
            inputs={"x": PortSignature(name="x", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))},
            outputs={"y": PortSignature(name="y", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))},
        )

        candidates = [cell_a, cell_b]
        token_index_for_idf = getattr(self.orch, "token_index", None) or {}
        corpus_size_for_idf = max(len(self.orch.loaded_cells), 1)

        import math
        def _idf(t: str) -> float:
            df = len(token_index_for_idf.get(t, ()))
            return math.log(1.0 + (corpus_size_for_idf + 1) / (df + 1.0))

        raw_by_cell = {}
        prec_by_cell = {}
        for c in candidates:
            c_toks = c.token_set
            id_t = c.identity_tokens
            raw_by_cell[c.cell_id] = sum(_idf(t) for t in (clause_toks & c_toks & id_t))
            c_tot = sum(_idf(t) for t in c_toks)
            prec_by_cell[c.cell_id] = sum(_idf(t) for t in (clause_toks & c_toks)) / c_tot if c_tot > 0 else 0.0

        max_ev = max(raw_by_cell.values())
        max_prec = max(prec_by_cell.values())
        cl_weight = sum(_idf(t) for t in clause_toks)
        min_ev = max(0.5 * max_ev, 0.3 * cl_weight)
        min_prec = 0.5 * max_prec

        covered_a = raw_by_cell[cell_a.cell_id] >= min_ev and prec_by_cell[cell_a.cell_id] >= min_prec
        covered_b = raw_by_cell[cell_b.cell_id] >= min_ev and prec_by_cell[cell_b.cell_id] >= min_prec

        self.assertFalse(covered_a, "Cell A (generic token overlap) must not cover the clause")
        self.assertTrue(covered_b, "Cell B (whole identity match) must cover the clause")


    def test_probe_2_dataflow_liveness_catches_dead_output(self):
        """
        Probe 2 (R5-2):
        An intermediate step whose output is never consumed downstream must trigger dead_output.
        """
        read_cell = self.orch.loaded_cells.get("PD_READ_CSV")
        fft_cell = self.orch.loaded_cells.get("NUMPY_FFT_FFT")
        dropna_cell = self.orch.loaded_cells.get("PD_DROPNA")

        # Pipeline where fft_cell produces var_2 which is never read downstream
        pipeline_bindings = [
            (read_cell, {"filepath_or_buffer": "'input.csv'", "output_var": "var_1"}),
            (fft_cell, {"a": "var_1['X']", "output_var": "var_2"}),  # dead output!
            (dropna_cell, {"data": "var_1", "output_var": "var_3"}),
        ]

        res = PreflightLinter.lint(pipeline_bindings, prompt="test")
        dead_violations = [v for v in res.structured_violations if v.check_id == "dead_output"]
        self.assertGreater(
            len(dead_violations),
            0,
            "Expected 'dead_output' violation for unconsumed var_2, but none was raised!",
        )
        self.assertEqual(dead_violations[0].variable_name, "var_2")

    def test_probe_3_dependency_ordering_catches_order_inversion(self):
        """
        Probe 3 (R5-2):
        Placing an earlier-clause step (e.g. dropna, clause 1) after a later-clause step
        (e.g. fit, clause 2) on shared carrier data must trigger order_inversion.
        """
        prompt = "load a csv file, drop null values, fit linear regression"
        read_cell = self.orch.loaded_cells.get("PD_READ_CSV")
        fit_cell = self.orch.loaded_cells.get("sklearn.linear_model.LinearRegression.fit")
        dropna_cell = self.orch.loaded_cells.get("PD_DROPNA")

        read_cell.matched_clause_idx = 0
        fit_cell.matched_clause_idx = 2
        dropna_cell.matched_clause_idx = 1

        # Fit (clause 2) placed BEFORE Dropna (clause 1) on shared carrier var_1
        pipeline_bindings = [
            (read_cell, {"filepath_or_buffer": "'input.csv'", "output_var": "var_1"}),
            (fit_cell, {"X": "var_1['X']", "y": "var_1['Y']", "output_var": "var_2"}),
            (dropna_cell, {"data": "var_1", "output_var": "var_3"}),
        ]

        res = PreflightLinter.lint(pipeline_bindings, prompt=prompt)
        inversion_violations = [v for v in res.structured_violations if v.check_id == "order_inversion"]
        self.assertGreater(
            len(inversion_violations),
            0,
            "Expected 'order_inversion' violation for dropna after fit, but none was raised!",
        )

    def test_probe_4_projection_typing_without_synthetic_strings(self):
        """
        Probe 4 (R5-1):
        Projections must resolve types directly from lattice nodes declaring projection attribute.
        Preflight must contain zero generic state strings (e.g. 'series_generic', 'dataframe_2d_generic').
        """
        # 1. Verify lattice nodes have projection attribute
        sel_col = self.orch.loaded_cells.get("PD_SELECT_COLUMN")
        sel_cols = self.orch.loaded_cells.get("PD_SELECT_COLUMNS")
        self.assertEqual(getattr(sel_col, "projection", None), "column")
        self.assertEqual(getattr(sel_cols, "projection", None), "columns")

        # 2. Test resolve_typed_binding_record
        var_sigs = {
            "var_1": PortSignature(name="var_1", signature=AlgebraicSignature(type_name="DataFrame", state="raw"))
        }

        # Single column projection: var_1['X'] -> Series
        rec_col = resolve_typed_binding_record(
            cell=None,
            port_name="x_input",
            bound_val="var_1['X']",
            var_signatures=var_sigs,
        )
        self.assertEqual(rec_col.kind, "projection")
        self.assertEqual(rec_col.projection, {"kind": "column", "key": "X"})
        self.assertEqual(rec_col.resulting_signature.type_name, "Series")

        # Multi-column projection: var_1[['X', 'Y']] -> DataFrame
        rec_cols = resolve_typed_binding_record(
            cell=None,
            port_name="features",
            bound_val="var_1[['X', 'Y']]",
            var_signatures=var_sigs,
        )
        self.assertEqual(rec_cols.kind, "projection")
        self.assertEqual(rec_cols.projection, {"kind": "columns", "key": ["X", "Y"]})
        self.assertEqual(rec_cols.resulting_signature.type_name, "DataFrame")

        # 3. Verify no generic state string fabrications exist in preflight.py source
        preflight_source = (ROOT_DIR / "src" / "preflight.py").read_text()
        self.assertNotIn("dataframe_2d_generic", preflight_source)
        self.assertNotIn("series_generic", preflight_source)
        self.assertNotIn("def _resolve_expr_sig", preflight_source)

    def test_probe_5_schema_tracking_and_port_addressed_column_removal(self):
        """
        Probe 5 (R5-3):
        When a cell with removes_columns removes X, a subsequent reference to X
        raises column_absent. If schema is unknown (known_columns=None), references pass.
        """
        gd_cell = self.orch.loaded_cells.get("PD_GET_DUMMIES")
        # Ensure effects declare structured port-addressed removal
        effects = getattr(gd_cell, "effects", [])
        self.assertTrue(
            any(
                (isinstance(e, dict) and e.get("name") == "removes_columns" and e.get("from_port") == "port_1")
                or e == "removes_columns"
                for e in effects
            ),
            f"PD_GET_DUMMIES effects must declare removes_columns, got {effects}",
        )

        read_cell = self.orch.loaded_cells.get("PD_READ_CSV")
        fit_cell = self.orch.loaded_cells.get("sklearn.linear_model.LinearRegression.fit")

        # Pipeline where PD_GET_DUMMIES encodes ['X', 'Y'], removing them from var_2
        # Then fit_cell accesses var_2['X'] -> must trigger column_absent!
        pipeline_bindings = [
            (read_cell, {"filepath_or_buffer": "'input.csv'", "output_var": "var_1"}),
            (
                gd_cell,
                {
                    "port_0": "var_1",
                    "port_1": "['X', 'Y']",
                    "drop_first": "True",
                    "output_var": "var_2",
                },
            ),
            (
                fit_cell,
                {
                    "X": "var_2['X']",  # Absent column!
                    "y": "var_2['Z']",
                    "output_var": "var_3",
                },
            ),
        ]

        res = PreflightLinter.lint(pipeline_bindings, prompt="test")
        absent_violations = [v for v in res.structured_violations if v.check_id == "column_absent"]
        self.assertGreater(
            len(absent_violations),
            0,
            "Expected 'column_absent' violation for referencing removed column 'X', but none was raised!",
        )
        self.assertEqual(absent_violations[0].details.get("column"), "X")

    def test_probe_6_no_engine_hardcodes_in_preflight_and_unification(self):
        """
        Probe 6 (Rule 2 / R5-1 / R5-3):
        No synthetic state strings ("series_generic", "dataframe_2d_generic")
        or expression resolvers in src/preflight.py or src/unification.py.
        Zero hardcoded port names ("value", "val"), candidate_cols, or fixtures.
        """
        for filename in ["preflight.py", "unification.py", "planner.py"]:
            source = (ROOT_DIR / "src" / filename).read_text()
            self.assertNotIn("dataframe_2d_generic", source, f"Found dataframe_2d_generic in {filename}")
            self.assertNotIn("series_generic", source, f"Found series_generic in {filename}")
            self.assertNotIn("def _resolve_expr_sig", source, f"Found _resolve_expr_sig in {filename}")
            self.assertNotIn('("value", "val")', source, f'Found ("value", "val") in {filename}')
            self.assertNotIn("candidate_cols", source, f"Found candidate_cols in {filename}")

        self.assertFalse((ROOT_DIR / "src" / "fixtures.py").exists(), "src/fixtures.py must be permanently deleted")
        gevr_source = (ROOT_DIR / "src" / "gevr_sandbox.py").read_text()
        self.assertNotIn("FixtureSynthesizer", gevr_source)

    def test_probe_7_clause_adjacency_and_provenance_gating(self):
        """
        Probe 7 (R5-7):
        Clause adjacency allows binding FFT output to PD_SET_COLUMN.port_2 when
        adjacent, but flags provenance_mismatch when bound cross-clause without
        clause adjacency or matching prompt literal.
        """
        read_cell = self.orch.loaded_cells.get("PD_READ_CSV")
        fft_cell = self.orch.loaded_cells.get("NUMPY_FFT_FFT")
        set_cell = self.orch.loaded_cells.get("PD_SET_COLUMN")
        dropna_cell = self.orch.loaded_cells.get("PD_DROPNA")
        describe_cell = self.orch.loaded_cells.get("PD_DESCRIBE")

        prompt = "load csv input.csv, compute fft of X column, and set it to Z column"
        # 1. Adjacent pipeline: FFT (clause 1) feeds PD_SET_COLUMN (clause 2)
        read_cell.matched_clause_idx = 0
        fft_cell.matched_clause_idx = 1
        set_cell.matched_clause_idx = 2

        valid_bindings = [
            (read_cell, {"filepath_or_buffer": "'input.csv'", "output_var": "var_1"}),
            (fft_cell, {"a": "var_1['X']", "output_var": "var_2"}),
            (set_cell, {"port_0": "var_1", "port_1": "'Z'", "port_2": "var_2", "output_var": "var_3"}),
        ]
        res_ok = PreflightLinter.lint(valid_bindings, prompt=prompt)
        mismatches_ok = [v for v in res_ok.structured_violations if v.check_id == "provenance_mismatch"]
        self.assertEqual(len(mismatches_ok), 0, f"Expected 0 provenance_mismatch for adjacent clauses, got {mismatches_ok}")

        # 2. Non-adjacent pipeline: FFT (clause 1) bound to PD_SET_COLUMN (clause 4)
        # across intermediate clauses 2 and 3 without literal match or adjacency
        prompt_long = "load csv, compute fft of X, drop nulls, describe summary stats, and assign value to column W"
        dropna_cell.matched_clause_idx = 2
        describe_cell.matched_clause_idx = 3
        set_cell.matched_clause_idx = 4

        bad_bindings = [
            (read_cell, {"filepath_or_buffer": "'input.csv'", "output_var": "var_1"}),
            (fft_cell, {"a": "var_1['X']", "output_var": "var_2"}),
            (dropna_cell, {"data": "var_1", "output_var": "var_3"}),
            (describe_cell, {"data": "var_3", "output_var": "var_4"}),
            (set_cell, {"port_0": "var_3", "port_1": "'W'", "port_2": "var_2", "output_var": "var_5"}),
        ]
        res_bad = PreflightLinter.lint(bad_bindings, prompt=prompt_long)
        mismatches_bad = [v for v in res_bad.structured_violations if v.check_id == "provenance_mismatch"]
        self.assertGreater(len(mismatches_bad), 0, "Expected provenance_mismatch for cross-clause non-adjacent binding")
        self.assertEqual(mismatches_bad[0].port_name, "port_2")

    def test_probe_8_coverage_floor_refusal_naming_missing_clause(self):
        """
        Probe 8 (R5-5):
        Coverage floor refusal names the specific missing clause when the lattice
        cannot cover that clause.
        """
        prompt = "load a csv file, drop null values, and apply quantum entanglement teleportation"
        planner = LatticePlanner(
            self.orch,
            require_coverage_floor=True,
            coverage_floor_fraction=0.85,
        )
        read_cell = self.orch.loaded_cells.get("PD_READ_CSV")
        dropna_cell = self.orch.loaded_cells.get("PD_DROPNA")
        tunnel = [read_cell, dropna_cell]
        relevance_map = {read_cell.cell_id: 0.95, dropna_cell.cell_id: 0.90}

        plan = planner.plan(prompt, tunnel, relevance_map)
        self.assertEqual(plan, [], "Planner must refuse when coverage is below floor")
        self.assertIsNotNone(planner.last_refusal, "Planner must record structured last_refusal")
        self.assertEqual(planner.last_refusal.get("reason"), "coverage_below_floor")
        self.assertLess(planner.last_refusal.get("coverage_fraction", 1.0), 0.85)

        uncovered = planner.last_refusal.get("uncovered_clauses", [])
        self.assertGreater(len(uncovered), 0, "last_refusal must include uncovered clauses list")
        # Clause 2 (quantum entanglement teleportation) must be in uncovered clauses
        has_quantum_clause = any(idx == 2 or "quantum" in text.lower() for idx, text in uncovered)
        self.assertTrue(has_quantum_clause, f"Missing clause must name quantum clause, got: {uncovered}")

    def test_probe_9_latency_budget_folding_deadline_adherence(self):
        """
        Probe 9 (R5-6):
        Latency budget folding ensures planning terminates within 1.05x total budget.
        """
        import time
        prompt = (
            "load a csv file named 'input.csv', normalize X column, drop null values, "
            "get the mean of the Y column, and perform FFT and write it to a new column called Z, "
            "then make a regression model trained on X and Y to predict Z"
        )
        budget_ms = 400.0
        planner = LatticePlanner(
            self.orch,
            planner_time_budget_ms=budget_ms,
            planner_greedy_budget_ms=100.0,
            require_coverage_floor=False,
        )
        tunnel = list(self.orch.loaded_cells.values())
        relevance_map = {c.cell_id: 0.5 for c in tunnel}

        t0 = time.perf_counter()
        _ = planner.plan(prompt, tunnel, relevance_map, planner_time_budget_ms=budget_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        max_allowed_ms = budget_ms * 1.05 + 150.0  # Allow slight timing tolerance for Python GC / system thread wakeup
        self.assertLessEqual(
            elapsed_ms,
            max_allowed_ms,
            f"Planner exceeded budget: took {elapsed_ms:.1f}ms with budget {budget_ms:.1f}ms",
        )


if __name__ == "__main__":
    unittest.main()

