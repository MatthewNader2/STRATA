"""
Sweep-7 regressions. Offline (no models). Each test pins a defect that was visible in a real-model log:

  * the compiled lattice.db must yield the SAME cells as the JSON trees (it silently dropped port `binds`,
    merged semantic tags into retrieval identity, and lost raises/fixture_needs/mutation/topology),
  * tree-declared filler words must not count as lexical evidence,
  * the shared anchor rule must pick fft for "perform FFT" and mean for "get the mean of the Y column",
  * LLM-method fallbacks must be reported, not silent,
  * planning must not add cells no clause asked for.
"""
import argparse
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
logging.disable(logging.CRITICAL)

from lattice import LatticeOrchestrator, TypeRegistry  # noqa: E402
import cli  # noqa: E402

PROMPT = ('load a csv file named "input.csv", normalize "X" column, drop null values, get the mean of the "Y" column, '
          'and perform FFT and write it to a new column called "Z", then make a regression model trained on "X" and "Y" to predict "Z"')


def _norm(v):
    if isinstance(v, (set, frozenset)):
        return sorted(map(str, v))
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items()}
    return v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)[:60]


class TestSweep7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "lat.db")
        trees = str(ROOT_DIR / "trees")
        cli.cmd_compile(argparse.Namespace(trees_dir=trees, output=cls.db, domains=[], clean=True))
        cls.json_orch = LatticeOrchestrator(trees_directory=trees, db_path=cls.db)
        cls.json_orch.load_all_json_trees()
        cls.json_orch.build_topology()
        cls.db_orch = LatticeOrchestrator(trees_directory=trees, db_path=cls.db)
        cls.db_orch.load_from_database(cls.db)
        cls.db_orch.build_topology()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _by_id(orch):
        out = {}
        for v in orch.loaded_cells.values():
            out.setdefault(v.cell_id.upper(), v)
        return out

    def test_db_and_json_loaders_build_identical_cells(self):
        a, b = self._by_id(self.json_orch), self._by_id(self.db_orch)
        self.assertEqual(set(a), set(b))
        cell_attrs = ["stage", "node_type", "node_role", "mutation_type", "keywords", "semantic_tags", "effects",
                      "preconditions", "postconditions", "raises", "fixture_needs", "topology_type",
                      "identity_tokens", "token_set"]
        diffs = []
        for cid, ca in a.items():
            cb = b[cid]
            for at in cell_attrs:
                if _norm(getattr(ca, at, None)) != _norm(getattr(cb, at, None)):
                    diffs.append((cid, at))
            for io in ("inputs", "outputs"):
                for pn, pa in getattr(ca, io).items():
                    pb = getattr(cb, io).get(pn)
                    if pb is None or any(_norm(getattr(pa, k, None)) != _norm(getattr(pb, k, None))
                                         for k in ("binds", "port_role", "required", "default_value", "abstract_type")):
                        diffs.append((cid, f"{io}.{pn}"))
        self.assertEqual(diffs[:5], [], f"{len(diffs)} divergences between JSON and DB cells")

    def test_port_binds_survive_the_database(self):
        cell = self._by_id(self.db_orch)["PD_SET_COLUMN"]
        binds = {getattr(p, "binds", None) for p in cell.inputs.values()}
        self.assertIn("assigned_value", binds)

    def test_filler_tokens_are_not_evidence(self):
        from token_evidence import get_evidence
        ev = get_evidence(self.db_orch)
        self.assertEqual(ev.clause_tokens("perform FFT"), {"fft"})
        toks = ev.clause_tokens("get the mean of the Y column called Z")
        self.assertIn("mean", toks)
        for filler in ("the", "called"):
            self.assertNotIn(filler, toks)

    def test_shared_anchor_rule_picks_the_named_operation(self):
        from token_evidence import get_evidence, anchors_for_clause
        ev = get_evidence(self.db_orch)
        pool = list(self.db_orch.loaded_cells.values())
        fft = [c.cell_id for c, _ in anchors_for_clause(ev, "perform FFT", pool)]
        self.assertEqual(fft[0], "NUMPY_FFT_FFT")
        mean = [c.cell_id for c, _ in anchors_for_clause(ev, "get the mean of the Y column", pool)]
        self.assertEqual(mean[0], "PD_SERIES_MEAN")

    def test_every_method_plans_the_requested_operations(self):
        from router import LatticeRouter
        from unification import ExecutionContext
        router = LatticeRouter(orchestrator=self.db_orch, internal_rag=None, default_route_method="m0")
        must = {"PD_READ_CSV", "PD_SERIES_NORMALIZE", "PD_DROPNA", "PD_SERIES_MEAN", "NUMPY_FFT_FFT", "PD_SET_COLUMN"}
        for method in ("m1", "m6"):
            cells = router.plan_path(PROMPT, return_tuple=False, route_method=method, ctx=ExecutionContext(prompt=PROMPT))
            ids = {c.cell_id for c in cells}
            self.assertTrue(must <= ids, f"{method} omitted {must - ids}")
            self.assertFalse({"PD_TO_CSV", "NUMPY_TOFILE", "PD_GET_DUMMIES"} & ids, f"{method} planned an unrequested sink/junk cell")
        for method in ("m2", "m3"):
            cells = router.plan_path(PROMPT, return_tuple=False, route_method=method, ctx=ExecutionContext(prompt=PROMPT))
            if cells:
                ids = {c.cell_id for c in cells}
                self.assertFalse({"PD_TO_CSV", "NUMPY_TOFILE", "PD_GET_DUMMIES"} & ids, f"{method} planned an unrequested sink/junk cell")
            else:
                self.assertTrue(router.last_fallback_reason, f"{method} returned empty without structured failure reason")

    def test_llm_methods_report_their_fallback(self):
        from router import LatticeRouter
        from unification import ExecutionContext
        router = LatticeRouter(orchestrator=self.db_orch, internal_rag=None, default_route_method="m0")
        for method in ("m5", "m7", "m8", "m9"):
            router.plan_path(PROMPT, return_tuple=False, route_method=method, ctx=ExecutionContext(prompt=PROMPT))
            self.assertEqual(router.last_effective_route, method.upper(), f"{method} failed honestly but reported {router.last_effective_route}")
            self.assertTrue(router.last_fallback_reason and method.upper() in router.last_fallback_reason)
            self.assertTrue(any(e["event"] in ("method_failed", "method_fallback") for e in router.last_route_trace))

    def test_prediction_cells_can_end_a_pipeline(self):
        cells = self._by_id(self.db_orch)
        for cid in ("SKLEARN.LINEAR_MODEL.LINEARREGRESSION.PREDICT",):
            self.assertTrue(getattr(cells[cid], "endable", False), f"{cid} must be endable in the sklearn tree")


if __name__ == "__main__":
    unittest.main()
