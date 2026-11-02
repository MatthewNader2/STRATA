#!/usr/bin/env python3
"""
scripts/sweep.py - STRATA (Neuro-Symbolic Topological Lattice)
Multi-Profile & Multi-Method Sweep Runner for Benchmark Evaluation.

Runs the benchmark prompt across:
  Profiles: A, C, D, E, S
  Methods: M0, M1, M2, M3, M4, M5, M6, M7, M8, M9

Options used: --debug --no-reranker --exec

Emits a structured markdown table:
  | profile | method | effective route | fallback reason | steps | emitted? | refused why | sandbox ok? | golden assertions passed (n/6) | layer-2 ms |
Followed by:
  - Number of distinct emitted scripts
  - Number refused
  - Preflight violation categories with counts
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import settings
from gevr_sandbox import GEVRSandbox
from lattice import LatticeOrchestrator
from planner import LatticePlanner, _segment_prompt_clauses
from preflight import PreflightLinter
from tokenizer import CellTokenizer
from unification import ExecutionContext, UnificationGate

BENCHMARK_PROMPT = (
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
)

PROFILES = ["A", "C", "D", "E", "S"]
METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def evaluate_golden_assertions(
    code: str, bindings: list, literals: list, working_dir: Path, path: list
) -> Tuple[int, Optional[str]]:
    """
    Evaluates runtime assertions on the generated code.
    Returns (score: 0-6, error_message).
    """
    import numpy as np
    import pandas as pd

    # Dynamically create sample dataset for test execution in the temporary sandbox
    csv_files = [lit for lit in literals if isinstance(lit, str) and lit.endswith(".csv")]
    if not csv_files:
        csv_files = ["input.csv"]
    col_names = [lit for lit in literals if isinstance(lit, str) and not lit.endswith(".csv") and len(lit) <= 30]
    if not col_names:
        col_names = ["X", "Y"]

    np.random.seed(42)
    n_rows = 50
    sample_data = {c: np.random.randn(n_rows) * 10.0 + 5.0 for c in col_names}
    if n_rows > 5 and col_names:
        for c in col_names:
            sample_data[c][2] = np.nan
    df_sample = pd.DataFrame(sample_data)

    for f in csv_files:
        p = working_dir / f
        p.parent.mkdir(parents=True, exist_ok=True)
        df_sample.to_csv(p, index=False)

    clauses = _segment_prompt_clauses(BENCHMARK_PROMPT)
    clause_toks = [CellTokenizer.tokenize_prompt(cl) for cl in clauses]

    passed = 0

    # Assertion 6: No cell on path has empty clause coverage unless bridge or source
    a6_pass = True
    for cell in path:
        role = str(getattr(cell, "node_role", "")).lower()
        if role in ("bridge", "source"):
            continue
        c_toks = cell.token_set
        if not any(len(c_toks & cl_t) > 0 for cl_t in clause_toks):
            a6_pass = False
            break
    if a6_pass:
        passed += 1

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

    if not res.get("success"):
        return passed, res.get("error")

    # Assertion 1: Executed without exception
    passed += 1

    results = res.get("results", {})
    trace = results.get("_nstl_trace", {})

    # Assertion 2: Target created column in working frame, length matches non-null rows, no NaN
    target_col = None
    for b in bindings:
        c_id = str(b.get("cell_id", "")).upper()
        if "SET_COLUMN" in c_id:
            for p_k in ("port_1", "column", "key", "target"):
                if p_k in b and isinstance(b[p_k], str):
                    target_col = b[p_k].strip("'\"")
                    break
        if target_col:
            break
    if not target_col:
        col_literals = [lit for lit in literals if isinstance(lit, str) and not lit.endswith(".csv")]
        target_col = col_literals[-1] if col_literals else "Z"

    df_found = False
    for v in results.values():
        if hasattr(v, "columns") and target_col in v.columns:
            z_col = v[target_col]
            if len(z_col) > 0 and not z_col.isna().any():
                df_found = True
                break
    if df_found:
        passed += 1

    # Assertion 3: LinearRegression.fit called with 2D X (2 columns) and 1D y
    fit_calls = trace.get("fit_calls", [])
    if fit_calls and fit_calls[0]["X_ndim"] == 2 and fit_calls[0]["X_shape"][1] == 2 and fit_calls[0]["y_ndim"] == 1:
        passed += 1

    # Assertion 4: Mean computed on target Series
    mean_target_col = None
    for b in bindings:
        c_id = str(b.get("cell_id", "")).upper()
        if "MEAN" in c_id:
            val = b.get("port_0") or b.get("series") or b.get("data")
            if val and isinstance(val, str) and "[" in val:
                import re
                m = re.search(r"\[['\"](.*?)['\"]\]", val)
                if m:
                    mean_target_col = m.group(1)
        if mean_target_col:
            break
    if not mean_target_col:
        for cl in clauses:
            if "mean" in cl.lower():
                for lit in literals:
                    if isinstance(lit, str) and lit in cl and not lit.endswith(".csv"):
                        mean_target_col = lit
                        break
    if not mean_target_col:
        mean_target_col = "Y"

    mean_names = trace.get("series_mean_names", [])
    if any(str(name).strip("'\"") == mean_target_col for name in mean_names if name is not None):
        passed += 1

    # Assertion 5: 1D FFT applied to single column
    fft_ndims = trace.get("fft_input_ndims", [])
    if fft_ndims and all(ndim == 1 for ndim in fft_ndims):
        passed += 1

    return passed, None


def parse_args():
    parser = argparse.ArgumentParser(description="STRATA Multi-Profile & Multi-Method Sweep Runner")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-reranker", action="store_true", help="Disable neural reranker")
    parser.add_argument("--exec", dest="execute_sandbox", action="store_true", default=True, help="Execute sandbox")
    parser.add_argument("--no-exec", dest="execute_sandbox", action="store_false", help="Skip sandbox execution")
    parser.add_argument("--profiles", nargs="+", default=PROFILES, help="Profiles to evaluate")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Route methods to evaluate")
    parser.add_argument("--explain-plan", action="store_true", help="Print clause-level coverage and precision diagnostics for the chosen path")
    return parser.parse_args()


def run_sweep():
    args = parse_args()
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    if getattr(args, "explain_plan", False):
        try:
            settings.explain_plan = True
        except Exception:
            pass

    trees_dir = str(ROOT_DIR / "trees")
    db_path = str(ROOT_DIR / "trees" / "lattice.db")
    orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=db_path)
    # Load the JSON trees directly (source of truth): the lattice.db cache is a
    # runtime artifact whose cell construction may diverge from the declared trees.
    orch.load_all_json_trees()
    orch.build_topology()

    all_cells = orch.cells
    prompt_tokens = CellTokenizer.tokenize_prompt(BENCHMARK_PROMPT)
    relevance_map = {
        c.cell_id: float(len(prompt_tokens & c.token_set))
        for c in all_cells if len(prompt_tokens & c.token_set) > 0
    }

    gate = UnificationGate(orchestrator=orch)

    rows = []
    distinct_scripts = set()
    refused_count = 0
    internal_error_count = 0
    violation_categories = collections.Counter()

    profiles_to_run = [p.upper() for p in args.profiles]
    methods_to_run = [m.upper() for m in args.methods]

    for profile_name in profiles_to_run:
        # config.py declares no Profile enum; SemanticRouteOptimizer accepts the
        # plain profile string (it str()s it and initializes the model profile).
        prof = profile_name.upper()
        for method_name in methods_to_run:
            t0 = time.perf_counter()
            best_path = None
            code = None
            refused_why = "-"
            sandbox_ok = "no"
            golden_score = 0
            steps = 0
            layer2_ms = 0.0
            effective_route = method_name
            fallback_reason = "-"

            try:
                if method_name == "M0":
                    # Mirror the CLI: the router computes the semantic tunnel first,
                    # then the M0 trellis plans inside it (the full lattice is not
                    # the tunnel — planning over it exhausts the search budget).
                    from router import SemanticRouteOptimizer
                    optimizer = SemanticRouteOptimizer(orch, prof, use_reranker=not args.no_reranker)
                    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
                    best_path = optimizer.route(BENCHMARK_PROMPT, method_name="M0", relevance_map=relevance_map, ctx=ctx)
                    effective_route = getattr(optimizer, "last_effective_route", "M0")
                    fallback_reason = getattr(optimizer, "last_fallback_reason", "-") or "-"
                else:
                    from router import SemanticRouteOptimizer
                    optimizer = SemanticRouteOptimizer(orch, prof, use_reranker=not args.no_reranker)
                    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
                    best_path = optimizer.route(BENCHMARK_PROMPT, method_name=method_name, relevance_map=relevance_map, ctx=ctx)
                    effective_route = getattr(optimizer, "last_effective_route", method_name)
                    fallback_reason = getattr(optimizer, "last_fallback_reason", "-") or "-"

                layer2_ms = (time.perf_counter() - t0) * 1000.0

                if not best_path:
                    refused_why = "Planner produced no path"
                    refused_count += 1
                else:
                    steps = len(best_path)
                    ctx = ExecutionContext(prompt=BENCHMARK_PROMPT)
                    try:
                        unify_res = gate.unify_pipeline(best_path, context=ctx)
                    except Exception as exc:
                        # Engine defect: never counted as a refusal (R6-4)
                        import traceback
                        traceback.print_exc()
                        internal_error_count += 1
                        rows.append({
                            "profile": profile_name,
                            "method": method_name,
                            "effective_route": effective_route,
                            "fallback_reason": fallback_reason,
                            "steps": steps,
                            "emitted": "no",
                            "refused_why": "-",
                            "internal_error": f"{type(exc).__name__}",
                            "sandbox_ok": "no",
                            "golden_passed": "0/6",
                            "layer2_ms": f"{layer2_ms:.1f}",
                        })
                        continue
                    if unify_res.is_bottom():
                        refused_why = f"Unification failed: {unify_res.reason if hasattr(unify_res, 'reason') else 'type bottom'}"
                        refused_count += 1
                    else:
                        bindings = unify_res.value
                        try:
                            code = gate.emit_code(bindings, ctx)
                        except Exception as exc:
                            # Engine defect: never counted as a refusal (R6-4)
                            import traceback
                            traceback.print_exc()
                            internal_error_count += 1
                            rows.append({
                                "profile": profile_name,
                                "method": method_name,
                                "effective_route": effective_route,
                                "fallback_reason": fallback_reason,
                                "steps": steps,
                                "emitted": "no",
                                "refused_why": "-",
                                "internal_error": f"{type(exc).__name__}",
                                "sandbox_ok": "no",
                                "golden_passed": "0/6",
                                "layer2_ms": f"{layer2_ms:.1f}",
                            })
                            continue

                        # Preflight check (same parameters the CLI main path uses)
                        lint_res = PreflightLinter.lint(
                            bindings,
                            prompt=BENCHMARK_PROMPT,
                            code_str=code,
                        )

                        if not lint_res.is_valid:
                            refused_count += 1
                            v_first = lint_res.violations[0] if lint_res.violations else "Preflight lint violation"
                            refused_why = f"Preflight: {v_first[:50]}"
                            for v in lint_res.violations:
                                cat = v.split(":")[0] if ":" in v else v.split()[0]
                                violation_categories[cat] += 1
                            code = None
                        else:
                            # Preflight passed, record script and test sandbox
                            script_hash = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:12]
                            distinct_scripts.add(script_hash)

                            if args.execute_sandbox:
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    golden_score, err = evaluate_golden_assertions(
                                        code, bindings, ctx.ordered_literals, Path(tmpdir), best_path
                                    )
                                    if err is None:
                                        sandbox_ok = "yes"
                                    else:
                                        sandbox_ok = f"error: {err.splitlines()[-1][:40]}"
                            else:
                                sandbox_ok = "skipped"

            except Exception as exc:
                # Planning-stage engine defect: reported as internal_error, never a refusal (R6-4)
                layer2_ms = (time.perf_counter() - t0) * 1000.0
                import traceback
                traceback.print_exc()
                internal_error_count += 1
                rows.append({
                    "profile": profile_name,
                    "method": method_name,
                    "effective_route": effective_route,
                    "fallback_reason": fallback_reason,
                    "steps": steps,
                    "emitted": "no",
                    "refused_why": "-",
                    "internal_error": f"{type(exc).__name__}",
                    "sandbox_ok": "no",
                    "golden_passed": "0/6",
                    "layer2_ms": f"{layer2_ms:.1f}",
                })
                continue

            rows.append({
                "profile": profile_name,
                "method": method_name,
                "effective_route": effective_route,
                "fallback_reason": fallback_reason,
                "steps": steps,
                "emitted": "yes" if code else "no",
                "refused_why": refused_why,
                "internal_error": "-",
                "sandbox_ok": sandbox_ok,
                "golden_passed": f"{golden_score}/6",
                "layer2_ms": f"{layer2_ms:.1f}",
            })

    # Print markdown table
    print("\n### Sweep Results Table\n")
    print("| profile | method | effective route | fallback reason | steps | emitted? | refused why | internal error | sandbox ok? | golden assertions passed (n/6) | layer-2 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['profile']} | {r['method']} | {r['effective_route']} | {r['fallback_reason']} | {r['steps']} | {r['emitted']} | {r['refused_why']} | {r['internal_error']} | {r['sandbox_ok']} | {r['golden_passed']} | {r['layer2_ms']} |")

    print(f"\nDistinct emitted scripts: {len(distinct_scripts)}")
    print(f"Refused runs: {refused_count} / {len(rows)}")
    print(f"Internal errors: {internal_error_count} / {len(rows)}")
    if violation_categories:
        print("\nPreflight violation categories:")
        for cat, count in violation_categories.most_common():
            print(f"  - {cat}: {count}")

    # Internal errors are hard failures: the sweep must exit non-zero (R6-4)
    if internal_error_count:
        print("\n[FAIL] Sweep completed with INTERNAL ERRORS - these are engine defects, not refusals.")
        sys.exit(1)


if __name__ == "__main__":
    run_sweep()
