"""
tests/test_golden_prompt.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Golden benchmark test for the 7-clause benchmark prompt (offline; no models required).

  "load a csv file named \"input.csv\", normalize X column, drop null values,
   get the mean of the Y column, and perform FFT and write it to a new column called Z,
   then make a regression model trained on X and Y to predict Z"

Contract under test (Definition of Done #4, R6-4):
For every deterministic route method, the pipeline either
  (a) emits code that passes the golden runtime assertions, or
  (b) refuses with a structured, specific reason.
An INTERNAL ERROR (any engine exception) is always a hard failure — it is an
engine defect, never a refusal.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from gevr_sandbox import GEVRSandbox
from lattice import LatticeOrchestrator
from planner import LatticePlanner, _segment_prompt_clauses
from preflight import PreflightLinter
from tokenizer import CellTokenizer
from unification import UnificationGate, ExecutionContext, UnificationFailure

BENCHMARK_PROMPT = (
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
)

NON_LLM_METHODS = ["M0", "M1", "M2", "M3", "M6", "M9"]


@pytest.fixture(scope="module")
def orchestrator():
    trees_dir = str(ROOT_DIR / "trees")
    db_path = str(ROOT_DIR / "trees" / "lattice.db")
    orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=db_path)
    # Load the JSON trees directly (source of truth, R6-3): the lattice.db cache
    # is a runtime artifact whose cell construction may diverge from the trees.
    orch.load_all_json_trees()
    orch.build_topology()
    return orch


def _run_pipeline_for_method(orch: LatticeOrchestrator, method: str, profile_name: str = "A"):
    # SemanticRouteOptimizer accepts any profile-shaped object (str or enum);
    # config.py declares no Profile enum, so the plain profile string is used.
    prof = profile_name
    planner = LatticePlanner(orchestrator=orch)
    gate = UnificationGate(orchestrator=orch)

    # Route and plan
    relevance_map = {}
    tokens = CellTokenizer.tokenize_prompt(BENCHMARK_PROMPT)
    for c in orch.cells:
        overlap = len(tokens & c.token_set)
        if overlap > 0:
            relevance_map[c.cell_id] = float(overlap)

    if method == "M0":
        # Mirror the CLI: the router computes the semantic tunnel first, then the
        # M0 trellis plans inside it (planning over the full lattice exhausts the
        # search budget and is not the CLI's M0 configuration).
        from router import SemanticRouteOptimizer
        optimizer = SemanticRouteOptimizer(orch, prof, use_reranker=False)
        best_path = optimizer.route(BENCHMARK_PROMPT, method_name="M0", relevance_map=relevance_map)
        if not best_path:
            router_refusal = getattr(optimizer.router.planner, "last_refusal", None) if hasattr(optimizer.router, "planner") else None
            if router_refusal:
                return None, None, None, [], f"planner refusal: {router_refusal.get('reason')}; missing: {router_refusal.get('uncovered_clauses')}"
    else:
        # Use route method
        from router import SemanticRouteOptimizer
        optimizer = SemanticRouteOptimizer(orch, prof, use_reranker=False)
        best_path = optimizer.route(BENCHMARK_PROMPT, method_name=method, relevance_map=relevance_map)

    if not best_path:
        return None, None, None, [], None

    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
    try:
        unify_res = gate.unify_pipeline(best_path, context=ctx)
    except Exception as exc:
        # R6-4: internal errors must propagate, never masquerade as refusals
        raise AssertionError(
            f"INTERNAL ERROR ({type(exc).__name__}) in unify_pipeline for {method}: {exc}"
        ) from exc
    if unify_res.is_bottom():
        refusal = getattr(unify_res, "reason", "type bottom")
        return best_path, None, None, [], refusal

    bindings = unify_res.value
    try:
        code = gate.emit_code(bindings, ctx)
    except UnificationFailure as exc:
        # Declared synthesis refusal (e.g. unresolved ports after estimator-
        # identity gating) — the honest refuse arm of the DoD dichotomy.
        return best_path, None, None, [], str(exc)
    except Exception as exc:
        raise AssertionError(
            f"INTERNAL ERROR ({type(exc).__name__}) in emit_code for {method}: {exc}"
        ) from exc

    # Pre-flight lint gate — the same check the CLI applies between emission and
    # execution. A lint-invalid path is REFUSED with specific structured reasons;
    # it must never reach the sandbox.
    lint_res = PreflightLinter.lint(bindings, prompt=BENCHMARK_PROMPT, code_str=code)
    if not lint_res.is_valid:
        return best_path, None, None, [], (
            "preflight lint: " + "; ".join(lint_res.violations[:3])
        )
    return best_path, bindings, code, ctx.ordered_literals, None


def _execute_with_tracing(code: str, bindings: list, literals: list, working_dir: Path):
    # Create test input files derived from literals
    import pandas as pd
    import numpy as np
    csv_files = [lit for lit in literals if isinstance(lit, str) and lit.endswith(".csv")]
    if not csv_files:
        csv_files = ["input.csv"]
    col_names = [lit for lit in literals if isinstance(lit, str) and not lit.endswith(".csv") and len(lit) <= 30]
    if not col_names:
        col_names = ["X", "Y"]
    np.random.seed(42)
    sample_data = {c: np.random.randn(50) * 10.0 + 5.0 for c in col_names}
    if col_names:
        for c in col_names:
            sample_data[c][2] = np.nan
    df_sample = pd.DataFrame(sample_data)
    for f in csv_files:
        p = working_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        df_sample.to_csv(p, index=False)

    # Trace wrapper prepended to code to inspect runtime contracts
    instrumentation_preamble = """
import numpy as _nstl_np
import pandas as _nstl_pd
import sklearn.linear_model as _nstl_slm

_nstl_trace = {
    'series_mean_names': [],
    'fft_input_ndims': [],
    'fft_input_types': [],
    'fit_calls': [],
}

_orig_series_mean = _nstl_pd.Series.mean
def _traced_series_mean(self, *args, **kwargs):
    _nstl_trace['series_mean_names'].append(getattr(self, 'name', None))
    return _orig_series_mean(self, *args, **kwargs)
_nstl_pd.Series.mean = _traced_series_mean

_orig_fft = _nstl_np.fft.fft
def _traced_fft(a, *args, **kwargs):
    _nstl_trace['fft_input_ndims'].append(getattr(a, 'ndim', 1))
    _nstl_trace['fft_input_types'].append(type(a).__name__)
    return _orig_fft(a, *args, **kwargs)
_nstl_np.fft.fft = _traced_fft

_orig_lr_fit = _nstl_slm.LinearRegression.fit
def _traced_lr_fit(self, X, y, *args, **kwargs):
    _nstl_trace['fit_calls'].append({
        'X_ndim': getattr(X, 'ndim', None),
        'X_shape': getattr(X, 'shape', None),
        'y_ndim': getattr(y, 'ndim', None),
        'y_shape': getattr(y, 'shape', None),
        'y_vals': _nstl_np.array(y).tolist() if hasattr(y, '__iter__') else None
    })
    return _orig_lr_fit(self, X, y, *args, **kwargs)
_nstl_slm.LinearRegression.fit = _traced_lr_fit
"""

    instrumented_code = instrumentation_preamble + "\n" + code

    sandbox = GEVRSandbox()
    # Execute inside working_dir
    orig_cwd = os.getcwd()
    try:
        os.chdir(working_dir)
        res = sandbox.execute(
            instrumented_code,
            timeout=10.0,
            pipeline_bindings=bindings,
            extracted_literals=literals,
            fixtures_dir=str(working_dir),
        )
    finally:
        os.chdir(orig_cwd)

    return res


@pytest.mark.parametrize("method", NON_LLM_METHODS)
def test_golden_prompt(orchestrator, method):
    """DoD dichotomy: golden-pass OR structured refusal; internal errors always fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_path = Path(tmpdir)
        best_path, bindings, code, literals, refusal = _run_pipeline_for_method(orchestrator, method)

        if best_path is None:
            # No path AND no structured refusal is never acceptable (R6-4/R6-7)
            assert refusal, f"{method} produced no path and gave no structured refusal reason"
            return
        assert best_path, f"Planner produced no path for {method} (without structured refusal)"

        if code is None or bindings is None:
            # (b) Refusal path: must be a structured, specific reason — never empty
            assert refusal, f"{method} emitted no code and gave no refusal reason"
            return

        # (a) Emission path: the golden runtime assertions must all pass
        # 6. No cell on path has empty clause coverage unless it is a bridge or source
        clauses = _segment_prompt_clauses(BENCHMARK_PROMPT)
        clause_toks = [CellTokenizer.tokenize_prompt(cl) for cl in clauses]
        for cell in best_path:
            role = str(getattr(cell, "node_role", "")).lower()
            if role in ("bridge", "source"):
                continue
            c_toks = cell.token_set
            covered = any(len(c_toks & cl_t) > 0 for cl_t in clause_toks)
            assert covered, f"Cell {cell.cell_id} on path has empty clause coverage and is not bridge/source"

        res = _execute_with_tracing(code, bindings, literals, work_path)

        # 1. Script executes without exception
        assert res.get("success") is True, f"Execution failed for {method}: {res.get('error')}"

        results = res.get("results", {})
        trace = results.get("_nstl_trace", {})

        # 2. Frame has column Z with length matching non-null rows, no NaN
        # Check all DataFrame variables in results
        df_found = False
        for k, v in results.items():
            if hasattr(v, "columns") and "Z" in v.columns:
                df_found = True
                z_col = v["Z"]
                assert not z_col.isna().any(), "Column Z contains NaN values"
                assert len(z_col) > 0, "Column Z is empty"
                break
        assert df_found, "Working frame does not contain column 'Z'"

        # 3. LinearRegression.fit was called with 2-D X of two columns and 1-D y
        fit_calls = trace.get("fit_calls", [])
        assert len(fit_calls) > 0, "LinearRegression.fit was never called"
        fit_info = fit_calls[0]
        assert fit_info["X_ndim"] == 2, f"Expected 2-D X, got {fit_info['X_ndim']}"
        assert fit_info["X_shape"][1] == 2, f"Expected 2 columns in X, got {fit_info['X_shape'][1]}"
        assert fit_info["y_ndim"] == 1, f"Expected 1-D y, got {fit_info['y_ndim']}"

        # 4. Mean computed on Series named Y
        mean_names = trace.get("series_mean_names", [])
        assert any(name in ("Y", "'Y'", '"Y"') for name in mean_names), (
            f"Mean was not computed over Series named Y; captured series: {mean_names}"
        )

        # 5. 1-D FFT applied to single column, not the whole frame
        fft_ndims = trace.get("fft_input_ndims", [])
        assert len(fft_ndims) > 0, "np.fft.fft was never called"
        assert all(ndim == 1 for ndim in fft_ndims), f"np.fft.fft input ndim was not 1: {fft_ndims}"
