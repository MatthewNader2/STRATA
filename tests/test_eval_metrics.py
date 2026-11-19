"""
tests/test_eval_metrics.py - the benchmark's scoring must itself be tested, or its numbers prove nothing.
Covers eval_metrics.py and the way test.py consumes it (via a stub runner; no models needed).
"""
from __future__ import annotations
import importlib.util
import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_metrics as EM


def _load_harness():
    spec = importlib.util.spec_from_file_location("nstl_harness", ROOT / "test.py")
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


class TestStatistics(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = EM.wilson_interval(0, 1)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.7)          # n=1 tells you almost nothing
        lo, hi = EM.wilson_interval(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(EM.wilson_interval(0, 0), (0.0, 1.0))

    def test_small_n_is_flagged(self):
        self.assertIn("too small", EM.format_rate(1, 1))
        self.assertNotIn("too small", EM.format_rate(15, 30))


class TestLiteralConsumption(unittest.TestCase):
    def test_extracts_quoted_and_numeric_not_digits_in_words(self):
        L = EM.extract_prompt_literals('read "input.png" make a 2D game, blur kernel size 5 sigma 1.5, say \'Score: 0\'')
        self.assertEqual(L["strings"], ["input.png", "Score: 0"])
        self.assertEqual(L["numbers"], [5.0, 1.5])      # "2D" is not a number; the 0 inside 'Score: 0' belongs to the string

    def test_apostrophe_is_not_a_quote(self):
        self.assertEqual(EM.extract_prompt_literals("don't load 3 files")["strings"], [])

    def test_consumed(self):
        code = "import cv2\nx = cv2.imread('input.png')\ny = cv2.GaussianBlur(x, (5, 5), 1.5)\n"
        r = EM.literal_consumption('read "input.png", blur kernel size 5 sigma 1.5', code)
        self.assertTrue(r["checked"])
        self.assertEqual(r["missing"], [])

    def test_ignored_literal_is_reported(self):
        code = "x = open('input.csv')\nz = 3\n"
        r = EM.literal_consumption('read "other.csv" and scale by 7', code)
        self.assertEqual(sorted(r["missing"]), ["number 7", "string 'other.csv'"])

    def test_unparseable_code_is_not_verified(self):
        r = EM.literal_consumption('read "a.csv"', "def (:")
        self.assertFalse(r["checked"])


class TestClassification(unittest.TestCase):
    def test_every_kind_maps_to_a_status_and_category(self):
        for kind, status in EM.FAILURE_KIND_TO_STATUS.items():
            self.assertIn(status, EM.STATUS_CATEGORY, kind)

    def test_lint_rejection_lists_all_violations(self):
        rec = {"failure": {"kind": "LINT_REJECTED", "violations": ["a is dead", "b unserved"]}, "code": "x=1"}
        status, detail, _ = EM.classify_from_record(rec)
        self.assertEqual(status, "LINT_HALTED")
        self.assertIn("a is dead", detail)
        self.assertIn("b unserved", detail)

    def test_unresolved_ports_are_named(self):
        rec = {"failure": {"kind": "UNRESOLVED_PORTS", "unresolved_ports": ["CV2_GAUSSIAN_BLUR.ksize"]}}
        status, detail, _ = EM.classify_from_record(rec)
        self.assertEqual(status, "UNRESOLVED_PORT")
        self.assertIn("CV2_GAUSSIAN_BLUR.ksize", detail)

    def test_emitter_defect_is_not_a_theoretical_limit(self):
        rec = {"failure": {"kind": "EMISSION_DEFECT", "detail": "Emitted code has invalid syntax"}}
        status, _, _ = EM.classify_from_record(rec)
        self.assertEqual(EM.category_of(status), "IMPLEMENTATION")

    def test_no_catch_all_architectural_bucket(self):
        self.assertNotIn("ARCHITECTURAL", set(EM.STATUS_CATEGORY.values()))

    def test_sandbox_pass_with_ignored_literal_does_not_count(self):
        status, detail, rep = EM.finalize_success("PASSED", "ok", 'blur kernel size 5, thresholds 50 and 150',
                                                  "import cv2\ny = cv2.GaussianBlur(x, (5, 5), 0.0)\n", {})
        self.assertEqual(status, "PASSED_UNVERIFIED")
        self.assertIn("number 50", detail)
        self.assertFalse(EM.outcome_ok(status, "synthesize"))

    def test_lint_that_did_not_run_blocks_verification(self):
        status, detail, _ = EM.finalize_success("PASSED", "ok", 'read "a.csv"', "open('a.csv')", {"lint_error": "KeyError: x"})
        self.assertEqual(status, "PASSED_UNVERIFIED")

    def test_clarify_accepts_precise_refusal_but_never_a_defect(self):
        self.assertTrue(EM.outcome_ok("UNRESOLVED_PORT", "clarify"))
        self.assertTrue(EM.outcome_ok("LINT_HALTED", "clarify"))
        self.assertFalse(EM.outcome_ok("CRASHED", "clarify"))
        self.assertFalse(EM.outcome_ok("UNRESOLVED_PORT", "synthesize"))

    def test_funnel_is_monotone(self):
        rs = [
            {"status": "PASSED", "code": "x"},
            {"status": "PASSED_UNVERIFIED", "code": "x"},
            {"status": "LINT_HALTED", "code": "x"},
            {"status": "UNRESOLVED_PORT", "code": ""},
        ]
        f = EM.funnel(rs)
        self.assertGreaterEqual(f["attempted"], f["emitted_code"])
        self.assertGreaterEqual(f["emitted_code"], f["lint_clean"])
        self.assertGreaterEqual(f["lint_clean"], f["ran_in_sandbox"])
        self.assertGreaterEqual(f["ran_in_sandbox"], f["verified"])
        self.assertEqual(f["verified"], 1)


class _StubRunner:
    """Returns a canned structured record, as CachedNSTLRunner.run_test would."""
    def __init__(self, rc, code, sandbox, record):
        self.payload = (rc, "stub output", 0.01, code, sandbox, record)
        self.calls = []

    def run_test(self, **kw):
        self.calls.append(kw)
        return self.payload


class TestHarnessIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = _load_harness()

    def _run(self, runner, prompt='read "a.csv"', expected="synthesize"):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            return self.H.run_single_test(
                profile="O", method="OMEGA", prompt=prompt, output_dir=Path(d), total_log_path=Path(d) / "t.log",
                runner=runner, use_cache=True, expected=expected)

    def test_lint_halt_surfaces_the_real_violations(self):
        rec = {"failure": {"kind": "LINT_REJECTED", "stage": "lint", "violations": ["Cell X produced dead output"]},
               "code": "x = 1", "sandbox_result": {"preflight_lint_violations": ["Cell X produced dead output"]},
               "route_ms": 5.0, "plan_ms": 7.0, "synth_ms": 3.0, "latency_ms": 20.0}
        res = self._run(_StubRunner(3, "x = 1", rec["sandbox_result"], rec))
        self.assertEqual(res["status"], "LINT_HALTED")
        self.assertEqual(res["category"], "VERIFIER_REJECTED")
        self.assertIn("dead output", res["detail"])
        self.assertEqual(res["latencies"]["plan_ms"], 7.0)          # per-layer timings survive a halted run

    def test_verified_pass(self):
        code = "p = open('a.csv')\n"
        rec = {"failure": None, "code": code, "sandbox_result": {"success": True}, "latency_ms": 9.0}
        res = self._run(_StubRunner(0, code, {"success": True}, rec))
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["category"], "SUCCESS")

    def test_sandbox_pass_that_drops_a_literal_is_suspect(self):
        code = "p = open('other.csv')\n"
        rec = {"failure": None, "code": code, "sandbox_result": {"success": True}, "latency_ms": 9.0}
        res = self._run(_StubRunner(0, code, {"success": True}, rec))
        self.assertEqual(res["status"], "PASSED_UNVERIFIED")
        self.assertEqual(res["category"], "SUSPECT")

    def test_run_without_history_record_is_not_scored_on_stale_output(self):
        # runner reports "no record": the harness must not invent a pass from leftover code
        res = self._run(_StubRunner(1, "", {}, None))
        self.assertNotEqual(res["status"], "PASSED")

    def test_cold_route_is_the_default_and_is_forwarded(self):
        rec = {"failure": {"kind": "PLANNING_FAILED"}, "code": ""}
        runner = _StubRunner(2, "", {}, rec)
        res = self._run(runner)
        self.assertTrue(runner.calls[0]["cold_route"])
        self.assertEqual(res["status"], "PLANNING_FAILED")

    def test_expected_outcome_is_recorded(self):
        rec = {"failure": {"kind": "UNRESOLVED_PORTS", "unresolved_ports": ["A.b"]}, "code": ""}
        res = self._run(_StubRunner(2, "", {}, rec), expected="clarify")
        self.assertTrue(res["outcome_ok"])
        res = self._run(_StubRunner(2, "", {}, rec), expected="synthesize")
        self.assertFalse(res["outcome_ok"])

    def test_vague_presets_declare_clarify_and_others_default(self):
        P = self.H.PROMPT_PRESETS
        self.assertEqual(P["vague_game"]["expected"], "clarify")
        self.assertEqual(P["vague_ml"]["expected"], "clarify")
        self.assertEqual(P["canonical"].get("expected", "synthesize"), "synthesize")

    def test_no_hardcoded_time_saved_estimate_remains(self):
        src = (ROOT / "test.py").read_text()
        self.assertNotIn("* 8.5", src)
        self.assertNotIn("Estimated Time Saved", src)


if __name__ == "__main__":
    unittest.main()
