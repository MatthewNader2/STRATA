#!/usr/bin/env python3
"""
test.py — Theory-First Empirical Runner for Unified STRATA (v2).

Legacy routing methods (M0-M11 / OMEGA) were consolidated into STRATA's single
unified type-directed planner (`method="unified"`) parameterized over the
7 orthogonal profiles: `0, A, B, C, D, E, O`.

Execution discipline:
  1. Prints the pre-run Theoretical Passability Matrix across profiles.
  2. Executes all theoretically `PASSABLE` cases first (`0 -> A -> B -> C -> D -> E -> O`),
     halting immediately (`--early-stop`, default True) on the first passable failure
     with full IR / retrieval / plan / code / sandbox root-cause diagnostics.
  3. Executes theoretically `EXPECTED_REFUSAL` cases (`PROOF OF THEORY`), verifying
     that formal type, refinement, asset, and resource guards refuse for the exact
     theoretically predicted reason rather than hallucinating unsound code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import resolve_profile_spec, settings  # noqa: E402
from gevr_sandbox import GEVRSandbox  # noqa: E402
from inference import ModelManager, select_optimal_embedder, select_optimal_llm  # noqa: E402
from lattice_db import ensure_lattice_compiled, load_compiled_lattice  # noqa: E402
from profile_adapters import build_strata_pipeline_for_profile  # noqa: E402
from reranker import LocalReranker, get_available_rerankers  # noqa: E402
from strata_pipeline import PipelineRunResult, StrataPipeline  # noqa: E402


METHOD_NAME = "unified"
DEFAULT_PROFILES: Tuple[str, ...] = ("0", "A", "B", "C", "D", "E", "O")


@dataclass(frozen=True)
class TheoryExpectation:
    mode: str  # "PASSABLE" | "EXPECTED_REFUSAL"
    allowed_refusals: Tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class TestCase:
    task_id: str
    kind: str
    prompt: str
    theory_by_profile: Dict[str, TheoryExpectation]
    check_fn: Optional[Callable[[PipelineRunResult], Optional[str]]] = None
    notes: str = ""


@dataclass
class RunRecord:
    phase: str
    profile: str
    method: str
    task_id: str
    expected_mode: str
    outcome: str  # "PASSED" | "PROOF OF THEORY" | "FAILED" | "UNEXPECTED_PASS" | "UNEXPECTED_REFUSAL"
    ok: bool
    refusal_kind: Optional[str]
    stage_failed: Optional[str]
    elapsed_ms: float
    detail: str = ""
    result: Optional[PipelineRunResult] = None
    extra_check_error: Optional[str] = field(default=None)


def _check_live_bridge(res: PipelineRunResult) -> Optional[str]:
    if res.live_bridge_count < 1:
        return f"Expected live_bridge_count >= 1, got {res.live_bridge_count}"
    return None


def _build_test_catalog() -> List[TestCase]:
    formal_pass_exact = TheoryExpectation(
        mode="PASSABLE",
        rationale="Exact formal IDs + grounded types/assets/refinements/resources",
    )
    formal_nl_refusal = TheoryExpectation(
        mode="EXPECTED_REFUSAL",
        allowed_refusals=("FORMAL_SYNTAX_ERROR",),
        rationale="Non-LLM profile rejects unconstrained natural language at Layer 0",
    )

    p_cv2_edges = (
        'img = cv2::CV2_IMREAD(image_path=@"input.png")\n'
        "gray = cv2::CV2_CVT_COLOR_BGR2GRAY(src=$img)\n"
        "blur = cv2::CV2_GAUSSIAN_BLUR(src=$gray, ksize=(5, 5), sigma_x=1.5)\n"
        "edges = cv2::CV2_CANNY(src=$blur, threshold1=50.0, threshold2=150.0)\n"
        'saved = cv2::CV2_IMWRITE(output_path=@"edges_out.png", image=$edges)'
    )
    p_cv2_bridge = (
        'img = cv2::CV2_IMREAD(image_path=@"input.png")\n'
        "edges = cv2::CV2_CANNY(src=$img, threshold1=50.0, threshold2=150.0)"
    )
    p_sqlite_pd = (
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(port_0="SELECT 1 AS a, 2 AS b", port_1=$db)\n'
        "done = sqlite3::SQLITE_CLOSE(connection=$db)"
    )
    p_mpl = (
        "arr = numpy::NUMPY_LINSPACE(start=0.0, stop=10.0, num=20)\n"
        "fig, ax = matplotlib::matplotlib_plt_subplots_single()\n"
        "matplotlib::matplotlib_ax_plot(ax=$ax, y=$arr)\n"
        "matplotlib::matplotlib_ax_legend(ax=$ax)\n"
        "matplotlib::matplotlib_plt_close(fig=$fig)"
    )
    p_semantic_cv2 = (
        'img = cv2::"load color image from file"(image_path=@"input.png")\n'
        'gray = cv2::"convert bgr image to grayscale"(src=$img)\n'
        'edges = cv2::"detect edges using canny operator"(src=$gray, threshold1=50.0, threshold2=150.0)'
    )
    p_leak = (
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(port_0="SELECT 1 AS a", port_1=$db)'
    )
    p_ref_bad = (
        "a = numpy::NUMPY_ZEROS_1D(length=6)\n"
        "b = numpy::NUMPY_RESHAPE_NCOL(array=$a, n_columns=0)"
    )
    p_missing_lit = (
        'img = cv2::CV2_IMREAD(image_path=@"input.png")\n'
        "edges = cv2::CV2_CANNY(src=$img)"
    )

    p_nl_cv2 = (
        'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
        'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"'
    )
    p_nl_sqlite = (
        'open an in-memory sqlite database connection, run SQL query "SELECT 1 AS a, 2 AS b" into a pandas dataframe, '
        "and close the sqlite connection"
    )
    p_nl_cv2_implicit = 'read image "input.png", convert to grayscale, and detect Canny edges'
    p_nl_vague_ml = "load dataset and train a machine learning model to predict target"
    p_nl_audio = "load audio file, apply high-pass filter, compute spectrogram, and export filtered audio"

    return [
        TestCase(
            task_id="formal_cv2_edges",
            kind="formal_exact",
            prompt=p_cv2_edges,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_cv2_live_bridge",
            kind="formal_bridge",
            prompt=p_cv2_bridge,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
            check_fn=_check_live_bridge,
        ),
        TestCase(
            task_id="formal_sqlite_pandas",
            kind="formal_exact",
            prompt=p_sqlite_pd,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_mpl_multiout",
            kind="formal_exact",
            prompt=p_mpl,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_semantic_cv2",
            kind="formal_semantic",
            prompt=p_semantic_cv2,
            theory_by_profile={
                "0": TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("NO_MATCHING_OPERATION",),
                    rationale="Profile 0 has no embedder, so natural-language intents fail lookup",
                ),
                "A": TheoryExpectation(mode="PASSABLE", rationale="Dense embedder retrieves formal semantic intents"),
                "B": TheoryExpectation(mode="PASSABLE", rationale="Embedder + reranker retrieves formal semantic intents"),
            },
        ),
        TestCase(
            task_id="formal_leak_refusal",
            kind="formal_refusal_proof",
            prompt=p_leak,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNCLOSED_RESOURCE",),
                    rationale="Linear theorem rejects unclosed acquire(+Lin) sqlite connection",
                )
                for p in ("0", "A", "B")
            },
        ),
        TestCase(
            task_id="formal_refinement_refusal",
            kind="formal_refusal_proof",
            prompt=p_ref_bad,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("REFINEMENT_VIOLATION",),
                    rationale="Refinement theorem rejects n_columns=0 against min=1 invariant",
                )
                for p in ("0", "A", "B")
            },
        ),
        TestCase(
            task_id="formal_missing_literal",
            kind="formal_refusal_proof",
            prompt=p_missing_lit,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNBOUND_PORT",),
                    rationale="Non-slot-fill profile rejects missing required thresholds cleanly",
                )
                for p in ("0", "A", "B")
            },
        ),
        TestCase(
            task_id="nl_cv2_params",
            kind="nl_explicit",
            prompt=p_nl_cv2,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                "C": TheoryExpectation(mode="PASSABLE", rationale="SLM compiles NL with explicit literals + dense retrieval"),
                "D": TheoryExpectation(mode="PASSABLE", rationale="SLM + dense + slot-fill handles explicit CV2 task"),
                "E": TheoryExpectation(mode="PASSABLE", rationale="SLM + dense + reranker + slot-fill handles CV2 task"),
                "O": TheoryExpectation(mode="PASSABLE", rationale="Full Omni profile compiles, plans, binds & executes CV2 task"),
            },
        ),
        TestCase(
            task_id="nl_sqlite_pandas",
            kind="nl_explicit",
            prompt=p_nl_sqlite,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                "C": TheoryExpectation(mode="PASSABLE", rationale="SLM compiles SQLite + Pandas + Close pipeline"),
                "D": TheoryExpectation(mode="PASSABLE", rationale="SLM + slot-fill handles SQLite + Pandas + Close pipeline"),
                "E": TheoryExpectation(mode="PASSABLE", rationale="SLM + reranker + slot-fill handles SQLite + Pandas pipeline"),
                "O": TheoryExpectation(mode="PASSABLE", rationale="Full Omni profile handles SQLite + Pandas pipeline"),
            },
        ),
        TestCase(
            task_id="nl_cv2_implicit_params",
            kind="nl_implicit_slots",
            prompt=p_nl_cv2_implicit,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                "C": TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNBOUND_PORT", "SLM_COMPILE_ERROR", "UNCOMPOSABLE_TYPES"),
                    rationale="Profile C lacks slot-fill when required Canny thresholds are omitted",
                ),
                "D": TheoryExpectation(mode="PASSABLE", rationale="SLM + schema slot-fill estimates missing Canny thresholds"),
                "E": TheoryExpectation(mode="PASSABLE", rationale="SLM + reranker + slot-fill estimates missing Canny thresholds"),
                "O": TheoryExpectation(mode="PASSABLE", rationale="Full Omni profile estimates missing Canny thresholds"),
            },
        ),
        TestCase(
            task_id="nl_vague_ml",
            kind="nl_refusal_proof",
            prompt=p_nl_vague_ml,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=(
                            "MISSING_EXTERNAL_ASSET",
                            "AMBIGUITY",
                            "UNBOUND_PORT",
                            "NO_MATCHING_OPERATION",
                            "UNCOMPOSABLE_TYPES",
                            "SLM_COMPILE_ERROR",
                        ),
                        rationale="Underspecified dataset path & target refuses instead of fabricating paths",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
        TestCase(
            task_id="nl_audio_missing_file",
            kind="nl_refusal_proof",
            prompt=p_nl_audio,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=(
                            "MISSING_EXTERNAL_ASSET",
                            "UNBOUND_PORT",
                            "AMBIGUITY",
                            "UNCOMPOSABLE_TYPES",
                            "NO_MATCHING_OPERATION",
                            "SLM_COMPILE_ERROR",
                        ),
                        rationale="Omitted audio input/output asset paths refuse asset-hallucination cleanly",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
    ]


def _print_matrix(catalog: List[TestCase], profiles: List[str]) -> None:
    print("=" * 96)
    print(" PRE-RUN THEORETICAL PASSABILITY MATRIX  (Method: unified | Legacy M0-M11/OMEGA consolidated)")
    print("=" * 96)
    hdr = f"{'Task ID':<26} " + " ".join(f"{p:^8}" for p in profiles)
    print(hdr)
    print("-" * len(hdr))
    for tc in catalog:
        cells: List[str] = []
        for p in profiles:
            exp = tc.theory_by_profile.get(p)
            if exp is None:
                lbl = "-"
            elif exp.mode == "PASSABLE":
                lbl = "PASS"
            else:
                lbl = "REFUSE"
            cells.append(f"{lbl:^8}")
        print(f"{tc.task_id:<26} " + " ".join(cells))
    print("=" * 96 + "\n", flush=True)


def _print_failure_diagnostic(
    rec: RunRecord,
    tc: TestCase,
    pipe: StrataPipeline,
) -> None:
    res = rec.result
    print("\n" + "!" * 96)
    print(f" EARLY STOP ON FIRST FAILURE [{rec.phase}] -> {rec.outcome}")
    print("!" * 96)
    print(f"Profile      : {rec.profile}")
    print(f"Method       : {rec.method}")
    print(f"Task ID      : {rec.task_id} ({tc.kind})")
    print(f"Expected     : {rec.expected_mode}")
    print(f"Outcome      : {rec.outcome} ({rec.detail})")
    print(f"Elapsed      : {rec.elapsed_ms:.1f} ms")
    print(f"Prompt       :\n{tc.prompt}")
    if res is None:
        print("Result       : None")
        print("!" * 96 + "\n", flush=True)
        return

    print(f"stage_failed : {res.stage_failed}")
    print(f"refusal_kind : {res.refusal_kind}")
    print(f"error        : {res.error}")
    print(f"warnings     : {list(res.warnings)}")
    if rec.extra_check_error:
        print(f"check_error  : {rec.extra_check_error}")

    if res.ir is not None:
        print("\n--- Layer 0 TypedIR ---")
        print(json.dumps(res.ir.model_dump(), indent=2, default=str))
        print("\n--- Layer 1 Top-5 Retrieved Candidates per Call Node ---")
        try:
            cands_map = pipe.retriever.retrieve_for_ir(res.ir, top_k=5)
            for nid, matches in cands_map.items():
                summary = [
                    f"{m.cell_id} ({m.domain}, score={m.score:.4f}, mode={m.match_mode})"
                    for m in matches
                ]
                print(f"  [{nid}]: {summary}")
        except Exception as exc:
            print(f"  <retrieval dump failed: {exc}>")

    if res.plan is not None:
        print(f"\n--- Layer 2 Chosen Plan (cost={res.plan.total_cost}, open_res={res.plan.open_resources}) ---")
        for i, st in enumerate(res.plan.steps, start=1):
            print(
                f"  {i}. {st.cell_id} (bridge={st.is_live_bridge}) "
                f"in={st.input_wiring} -> out={st.output_var} ({st.output_sig.carrier}[{st.output_sig.typestate}])"
            )

    if res.code:
        print("\n--- Layer 3 Emitted Code ---")
        print(res.code)

    if res.sandbox_result is not None:
        print("\n--- Layer 5 Sandbox Result ---")
        print(json.dumps(res.sandbox_result, indent=2, default=str))

    print("!" * 96 + "\n", flush=True)


def _init_pipeline_for_profile(
    profile_code: str,
    compiled: Any,
    sandbox: Optional[GEVRSandbox],
    args: argparse.Namespace,
) -> StrataPipeline:
    spec = resolve_profile_spec(profile_code)
    mm = ModelManager.get_instance()
    if spec.pure_symbolic:
        mm.cleanup(release_reranker=True)
        rr_obj = None
    else:
        mm.initialize_profile(spec.code, embedder_name=args.embedder, llm_name=args.llm)
        rr_obj = None
        if spec.uses_reranker and args.use_reranker:
            rr_obj = LocalReranker.get_shared(device="auto")
            if getattr(rr_obj, "model", None) is None:
                raise RuntimeError(f"Profile '{spec.code}' requires LocalReranker, but model failed to load.")
        elif not spec.uses_reranker and os.environ.get("NSTL_LOW_VRAM") == "1":
            LocalReranker.release_shared()

    return build_strata_pipeline_for_profile(
        compiled,
        spec,
        model_manager=mm,
        reranker=rr_obj,
        sandbox=sandbox,
        timeout=args.timeout,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="STRATA v2 Theory-First Empirical Multi-Profile Runner")
    ap.add_argument("--profiles", type=str, default=",".join(DEFAULT_PROFILES), help="Comma-separated profiles (0,A,B,C,D,E,O)")
    ap.add_argument("--early-stop", dest="early_stop", action="store_true", default=True, help="Stop on first failing passable test (default: True)")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="Continue running after failures")
    ap.add_argument("--passable-only", action="store_true", default=False, help="Run only theoretically PASSABLE cases")
    ap.add_argument("--no-exec", action="store_true", default=False, help="Disable live GEVR sandbox execution")
    ap.add_argument("--llm", type=str, default="", help="Override GGUF LLM name")
    ap.add_argument("--embedder", type=str, default="", help="Override embedder model name")
    ap.add_argument(
        "--reranker",
        dest="use_reranker",
        action="store_true",
        default=bool(settings.use_reranker),
        help="Enable neural reranker on reranker-capable profiles (B, E, O)",
    )
    ap.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Disable neural reranker across all profiles",
    )
    ap.add_argument("--timeout", type=float, default=30.0, help="Sandbox timeout in seconds")
    args = ap.parse_args()

    profiles = [resolve_profile_spec(p.strip()).code for p in args.profiles.split(",") if p.strip()]
    catalog = _build_test_catalog()
    _print_matrix(catalog, profiles)

    try:
        emb_sel = select_optimal_embedder(args.embedder)
    except Exception as exc:
        emb_sel = f"N/A ({exc})"
    try:
        llm_sel = select_optimal_llm(args.llm)
    except Exception as exc:
        llm_sel = f"N/A ({exc})"
    avail_rr = get_available_rerankers()
    rr_name = settings.reranker_model or (avail_rr[0] if avail_rr else "auto")
    rr_label = f"enabled ({rr_name})" if args.use_reranker else "disabled (default per A/B test)"
    print(
        f"[env] embedder={emb_sel} | llm={llm_sel} | reranker={rr_label} "
        f"| sandbox={'OFF' if args.no_exec else 'ON'} | early_stop={args.early_stop}"
    )

    db_path = ROOT_DIR / "trees" / "lattice.db"
    ensure_lattice_compiled(trees_dir=str(ROOT_DIR / "trees"), db_path=str(db_path))
    compiled = load_compiled_lattice(db_path)
    sandbox = None if args.no_exec else GEVRSandbox()
    print(f"[env] loaded SQLite v{compiled.schema_version}: {len(compiled.cells)} cells, {len(compiled.state_parents)} typestates\n", flush=True)

    schedule: List[Tuple[str, str, TestCase, TheoryExpectation]] = []
    for p_code in profiles:
        for tc in catalog:
            exp = tc.theory_by_profile.get(p_code)
            if exp is not None and exp.mode == "PASSABLE":
                schedule.append(("PHASE1_PASSABLE", p_code, tc, exp))
    if not args.passable_only:
        for p_code in profiles:
            for tc in catalog:
                exp = tc.theory_by_profile.get(p_code)
                if exp is not None and exp.mode == "EXPECTED_REFUSAL":
                    schedule.append(("PHASE2_PROOF", p_code, tc, exp))

    records: List[RunRecord] = []
    pipe_cache: Dict[str, StrataPipeline] = {}

    try:
        for phase, p_code, tc, exp in schedule:
            if p_code not in pipe_cache:
                t_init0 = time.perf_counter()
                print(f"[init] Initializing Profile {p_code} ({resolve_profile_spec(p_code).display_name})...", flush=True)
                pipe_cache[p_code] = _init_pipeline_for_profile(p_code, compiled, sandbox, args)
                print(f"[init] Profile {p_code} ready in {(time.perf_counter() - t_init0) * 1000.0:.1f} ms", flush=True)

            pipe = pipe_cache[p_code]
            t0 = time.perf_counter()
            res = pipe.run(tc.prompt, execute=not args.no_exec)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if exp.mode == "PASSABLE":
                extra_err = tc.check_fn(res) if (res.ok and tc.check_fn is not None) else None
                if res.ok and extra_err is None:
                    rec = RunRecord(
                        phase=phase,
                        profile=p_code,
                        method=METHOD_NAME,
                        task_id=tc.task_id,
                        expected_mode=exp.mode,
                        outcome="PASSED",
                        ok=True,
                        refusal_kind=res.refusal_kind,
                        stage_failed=res.stage_failed,
                        elapsed_ms=elapsed_ms,
                        detail=f"bridges={res.live_bridge_count}",
                        result=res,
                    )
                    print(f"[PASS]           P={p_code} | {METHOD_NAME} | {tc.task_id:<26} | {elapsed_ms:7.1f} ms | bridges={res.live_bridge_count}", flush=True)
                else:
                    rec = RunRecord(
                        phase=phase,
                        profile=p_code,
                        method=METHOD_NAME,
                        task_id=tc.task_id,
                        expected_mode=exp.mode,
                        outcome="FAILED",
                        ok=False,
                        refusal_kind=res.refusal_kind,
                        stage_failed=res.stage_failed,
                        elapsed_ms=elapsed_ms,
                        detail=extra_err or f"{res.stage_failed}:{res.refusal_kind}: {res.error}",
                        result=res,
                        extra_check_error=extra_err,
                    )
                    records.append(rec)
                    _print_failure_diagnostic(rec, tc, pipe)
                    if args.early_stop:
                        return 1
                    continue
            else:
                if not res.ok and res.refusal_kind in exp.allowed_refusals:
                    rec = RunRecord(
                        phase=phase,
                        profile=p_code,
                        method=METHOD_NAME,
                        task_id=tc.task_id,
                        expected_mode=exp.mode,
                        outcome="PROOF OF THEORY",
                        ok=True,
                        refusal_kind=res.refusal_kind,
                        stage_failed=res.stage_failed,
                        elapsed_ms=elapsed_ms,
                        detail=f"predicted={res.refusal_kind}",
                        result=res,
                    )
                    print(f"[PROOF OF THEORY] P={p_code} | {METHOD_NAME} | {tc.task_id:<26} | {elapsed_ms:7.1f} ms | refusal={res.refusal_kind}", flush=True)
                elif res.ok:
                    rec = RunRecord(
                        phase=phase,
                        profile=p_code,
                        method=METHOD_NAME,
                        task_id=tc.task_id,
                        expected_mode=exp.mode,
                        outcome="UNEXPECTED_PASS",
                        ok=False,
                        refusal_kind=None,
                        stage_failed=None,
                        elapsed_ms=elapsed_ms,
                        detail=f"Soundness bug: expected {exp.allowed_refusals}, but succeeded!",
                        result=res,
                    )
                    records.append(rec)
                    _print_failure_diagnostic(rec, tc, pipe)
                    if args.early_stop:
                        return 1
                    continue
                else:
                    rec = RunRecord(
                        phase=phase,
                        profile=p_code,
                        method=METHOD_NAME,
                        task_id=tc.task_id,
                        expected_mode=exp.mode,
                        outcome="UNEXPECTED_REFUSAL",
                        ok=False,
                        refusal_kind=res.refusal_kind,
                        stage_failed=res.stage_failed,
                        elapsed_ms=elapsed_ms,
                        detail=f"Expected refusal in {exp.allowed_refusals}, got {res.stage_failed}:{res.refusal_kind}",
                        result=res,
                    )
                    records.append(rec)
                    _print_failure_diagnostic(rec, tc, pipe)
                    if args.early_stop:
                        return 1
                    continue

            records.append(rec)

        print("\n" + "=" * 96)
        print(" FINAL TEST SUMMARY")
        print("=" * 96)
        p_ok = sum(1 for r in records if r.outcome == "PASSED")
        p_tot = sum(1 for r in records if r.phase == "PHASE1_PASSABLE")
        t_ok = sum(1 for r in records if r.outcome == "PROOF OF THEORY")
        t_tot = sum(1 for r in records if r.phase == "PHASE2_PROOF")
        f_cnt = sum(1 for r in records if not r.ok)
        print(f"Passable Passed   : {p_ok}/{p_tot}")
        print(f"Proof-of-Theory   : {t_ok}/{t_tot}")
        print(f"Unexpected Fails  : {f_cnt}")
        print("=" * 96, flush=True)
        return 0 if f_cnt == 0 else 1
    finally:
        ModelManager.get_instance().cleanup(release_reranker=True)


if __name__ == "__main__":
    raise SystemExit(main())
