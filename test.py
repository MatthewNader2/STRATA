#!/usr/bin/env python3
"""
test.py — Theory-First Empirical Runner for Unified STRATA (v2).

Legacy routing methods (M0-M11 / OMEGA) were consolidated into STRATA's single
unified type-directed planner (`method="unified"`) parameterized over the
7 orthogonal profiles: `0, A, B, C, D, E, O`.

Test groups:
  - formal       : Formal-syntax synthesis, multi-hop cross-domain live bridging, semantics, and formal refusals
  - nl_core      : Detailed ("unvague") NL tasks contrasting vague/underspecified/missing-asset refusals
  - unseen       : Blind synthetic SDKs (CryoPulse, Aetheris) zero-pretraining & leak-refusal proofs
  - haystack     : Needle-in-a-haystack isolation (11-15 target cells embedded inside 1,038 unrelated noise cells)
  - wrong_domain : Refusal proofs when only completely unrelated/insufficient domains are loaded
  - domain_spec  : Explicit library specification adherence & cross-library substitution refusal (Pillow vs OpenCV)

Execution discipline:
  1. Prints the pre-run Theoretical Passability Matrix across profiles.
  2. Executes all theoretically `PASSABLE` cases first (`0 -> A -> B -> C -> D -> E -> O`),
     halting immediately (`--early-stop`, default True) on the first passable failure
     with full IR / retrieval / plan / code / sandbox root-cause diagnostics.
  3. Executes theoretically `EXPECTED_REFUSAL` cases (`PROOF OF THEORY`), verifying
     that formal type, refinement, asset, resource, and domain guards refuse for
     the exact theoretically predicted reason rather than hallucinating unsound code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.evaluators import audit_sdk_code, verify_literals_consumed  # noqa: E402
from config import resolve_profile_spec, settings  # noqa: E402
from gevr_sandbox import GEVRSandbox  # noqa: E402
from inference import ModelManager, select_optimal_embedder, select_optimal_llm  # noqa: E402
from lattice_db import ensure_lattice_compiled, load_compiled_lattice  # noqa: E402
from profile_adapters import (  # noqa: E402
    build_strata_pipeline_for_profile,
    build_strata_pipeline_from_trees,
    make_embed_fn,
)
from reranker import LocalReranker, get_available_rerankers  # noqa: E402
from schema import get_attr  # noqa: E402
from strata_pipeline import PipelineRunResult, StrataPipeline  # noqa: E402


METHOD_NAME = "unified"
DEFAULT_PROFILES: Tuple[str, ...] = ("0", "A", "B", "C", "D", "E", "O")
VALID_GROUPS: Tuple[str, ...] = (
    "formal",
    "nl_core",
    "unseen",
    "haystack",
    "wrong_domain",
    "domain_spec",
)


@dataclass(frozen=True)
class TheoryExpectation:
    mode: str  # "PASSABLE" | "EXPECTED_REFUSAL"
    allowed_refusals: Tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class TestCase:
    task_id: str
    group: str
    kind: str
    prompt: str
    theory_by_profile: Dict[str, TheoryExpectation]
    domain_set: str = "default"
    execute: bool = True
    check_fn: Optional[Callable[[PipelineRunResult], Optional[str]]] = None
    notes: str = ""


@dataclass
class RunRecord:
    phase: str
    group: str
    domain_set: str
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


def _iter_plan_call_steps(steps: Iterable[Any]) -> Iterable[Any]:
    for step in steps:
        if getattr(step, "kind", "call") == "call":
            yield step
        if getattr(step, "body_steps", None):
            yield from _iter_plan_call_steps(step.body_steps)
        if getattr(step, "else_steps", None):
            yield from _iter_plan_call_steps(step.else_steps)


def _compose_checks(*fns: Callable[[PipelineRunResult], Optional[str]]) -> Callable[[PipelineRunResult], Optional[str]]:
    def _combined(res: PipelineRunResult) -> Optional[str]:
        for fn in fns:
            err = fn(res)
            if err:
                return err
        return None
    return _combined


def _check_live_bridge(min_bridges: int = 1) -> Callable[[PipelineRunResult], Optional[str]]:
    def _chk(res: PipelineRunResult) -> Optional[str]:
        if res.live_bridge_count < min_bridges:
            return f"Expected live_bridge_count >= {min_bridges}, got {res.live_bridge_count}"
        return None
    return _chk


def _check_literals(expected_literals: Sequence[Any]) -> Callable[[PipelineRunResult], Optional[str]]:
    exp_list = list(expected_literals)

    def _chk(res: PipelineRunResult) -> Optional[str]:
        missing = verify_literals_consumed(res.code, exp_list)
        if missing:
            return f"Missing prompt literals in emitted code: {missing}"
        return None
    return _chk


def _check_domain_purity(
    allowed_domains: Iterable[str],
    *,
    forbidden_domains: Iterable[str] = (),
    forbidden_code_substrings: Sequence[str] = (),
) -> Callable[[PipelineRunResult], Optional[str]]:
    allowed_l = {str(d).strip().lower() for d in allowed_domains if str(d).strip()}
    forbidden_l = {str(d).strip().lower() for d in forbidden_domains if str(d).strip()}
    bad_substrs = [str(s) for s in forbidden_code_substrings if s]

    def _chk(res: PipelineRunResult) -> Optional[str]:
        if res.plan is None:
            return "No plan produced to verify domain purity"
        non_bridge_doms: List[str] = []
        all_doms: List[str] = []
        for step in _iter_plan_call_steps(res.plan.steps):
            s_dom = str(get_attr(step.cell, "domain_name", None) or "").strip().lower()
            if not s_dom:
                continue
            all_doms.append(f"{step.cell_id}:{s_dom}")
            if s_dom in forbidden_l:
                return f"Forbidden domain '{s_dom}' leaked via step '{step.cell_id}'"
            if not getattr(step, "is_live_bridge", False):
                non_bridge_doms.append(f"{step.cell_id}:{s_dom}")
                if allowed_l and s_dom not in allowed_l:
                    return (
                        f"Disallowed non-bridge step domain '{s_dom}' in '{step.cell_id}' "
                        f"(expected subset of {sorted(allowed_l)}; steps={all_doms})"
                    )
        if not non_bridge_doms:
            return "Plan contained zero non-bridge domain steps"
        for sub in bad_substrs:
            if sub in res.code:
                return f"Forbidden code substring {sub!r} found in emitted code"
        return None

    return _chk


def _check_sdk_and_purity(
    sdk_name: str,
    required_calls: Sequence[str],
    expected_literals: Sequence[Any],
    allowed_domains: Iterable[str],
) -> Callable[[PipelineRunResult], Optional[str]]:
    task_dict = {
        "sdk": sdk_name,
        "required_calls": list(required_calls),
        "expected_literals": list(expected_literals),
    }
    purity_chk = _check_domain_purity(allowed_domains)

    def _chk(res: PipelineRunResult) -> Optional[str]:
        p_err = purity_chk(res)
        if p_err:
            return p_err
        audit = audit_sdk_code(res.code, task_dict)
        if not audit.get("spec_compliant", False):
            parts = []
            for key in (
                "hallucinated_calls",
                "forbidden_calls_used",
                "invalid_kwargs",
                "missing_required_calls",
                "order_violations",
                "missing_literals",
                "pyflakes_issues",
            ):
                val = audit.get(key)
                if val:
                    parts.append(f"{key}={val}")
            if not audit.get("ast_ok"):
                parts.append("ast_ok=False")
            if not audit.get("imports_sdk"):
                parts.append("imports_sdk=False")
            return f"{sdk_name} SDK audit failed: " + (", ".join(parts) or "non-compliant")
        return None

    return _chk


class _DomainSetLoader:
    """Lazy cached loader for standard compiled lattice and custom/haystack domain manifests."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self._json_cache: Dict[Path, Dict[str, Any]] = {}
        self._compiled: Any = None
        self._tree_sets: Dict[str, List[Dict[str, Any]]] = {}

    def _read_json(self, path: Path) -> Dict[str, Any]:
        resolved = path.resolve()
        if resolved not in self._json_cache:
            self._json_cache[resolved] = json.loads(resolved.read_text(encoding="utf-8"))
        return self._json_cache[resolved]

    def get_compiled_default(self) -> Any:
        if self._compiled is None:
            db_path = self.root_dir / "trees" / "lattice.db"
            ensure_lattice_compiled(trees_dir=str(self.root_dir / "trees"), db_path=str(db_path))
            self._compiled = load_compiled_lattice(db_path)
        return self._compiled

    def _all_standard_tree_paths(self) -> List[Path]:
        return sorted(
            p
            for p in (self.root_dir / "trees").glob("*.json")
            if p.name not in {"manifest.json", "index.json"}
        )

    def get_trees(self, domain_set: str) -> List[Dict[str, Any]]:
        if domain_set in self._tree_sets:
            return self._tree_sets[domain_set]

        trees_dir = self.root_dir / "trees"
        fixtures_dir = self.root_dir / "benchmarks" / "fixtures"

        if domain_set == "cryopulse_only":
            paths = [trees_dir / "python_core.json", fixtures_dir / "cryopulse" / "cryopulse.json"]
        elif domain_set == "aetheris_only":
            paths = [trees_dir / "python_core.json", fixtures_dir / "aetheris" / "aetheris.json"]
        elif domain_set == "haystack_cryopulse":
            paths = self._all_standard_tree_paths() + [fixtures_dir / "cryopulse" / "cryopulse.json"]
        elif domain_set == "haystack_aetheris":
            paths = self._all_standard_tree_paths() + [fixtures_dir / "aetheris" / "aetheris.json"]
        elif domain_set == "sqlite_only":
            paths = [trees_dir / "python_core.json", trees_dir / "sqlite3.json"]
        elif domain_set == "cv2_only":
            paths = [trees_dir / "python_core.json", trees_dir / "numpy.json", trees_dir / "cv2.json"]
        elif domain_set == "cv2_and_pillow":
            paths = [
                trees_dir / "python_core.json",
                trees_dir / "numpy.json",
                trees_dir / "cv2.json",
                trees_dir / "pillow.json",
            ]
        else:
            raise ValueError(f"Unknown domain_set: {domain_set!r}")

        loaded = [self._read_json(p) for p in paths]
        self._tree_sets[domain_set] = loaded
        return loaded


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
    nl_unavail_refusal = TheoryExpectation(
        mode="EXPECTED_REFUSAL",
        allowed_refusals=("UNAVAILABLE_DOMAIN",),
        rationale="Explicitly requested library/domain absent from loaded lattice refuses at Layer 0.5",
    )
    formal_unavail_refusal = TheoryExpectation(
        mode="EXPECTED_REFUSAL",
        allowed_refusals=("UNAVAILABLE_DOMAIN",),
        rationale="Explicitly qualified domain absent from loaded lattice refuses at Layer 0.5",
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
    p_pillow_cv2_bridge = (
        'pil_img = pillow::PILLOW_IMAGE_OPEN(image_path=@"input.png")\n'
        "edges = cv2::CV2_CANNY(src=$pil_img, threshold1=50.0, threshold2=150.0)"
    )
    p_pillow_formal = (
        'img = pillow::PILLOW_IMAGE_OPEN(image_path=@"input.png")\n'
        "gray = pillow::PILLOW_CONVERT_TO_GRAYSCALE(image=$img)\n"
        'done = pillow::PILLOW_IMAGE_SAVE(image=$gray, dest_path=@"pil_formal.png")'
    )
    p_sqlite_pd = (
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(sql="SELECT 1 AS a, 2 AS b", con=$db)\n'
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
        'df = pandas::PD_READ_SQL(sql="SELECT 1 AS a", con=$db)'
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
    p_nl_pillow_spec = (
        'using Pillow, read image "input.png", convert to grayscale, and save to "pil_gray.png"'
    )
    p_nl_cv2_spec = (
        'using OpenCV (cv2), read image "input.png", convert to grayscale, and save to "cv2_gray.png"'
    )
    p_nl_pygame_unvague = (
        "initialize pygame, create a display window of size (640, 480), "
        "flip the display surface, and quit pygame"
    )
    p_nl_pandas_etl = (
        'load csv file "input.csv", drop missing rows, and save the dataframe to "clean_out.csv"'
    )
    p_nl_vague_game = "make a 2d platformer game"
    p_nl_vague_ml = "load dataset and train a machine learning model to predict target"
    p_nl_audio = "load audio file, apply high-pass filter, compute spectrogram, and export filtered audio"
    p_nl_bound_ambig = "build an end-to-end anomaly detection microservice with alerts and automatic retraining"

    cryo_req_calls = [
        "open_cryo_vessel",
        "energize_flux_coil",
        "sample_squid_flux",
        "demodulate_quadrature",
        "quench_tls_defects",
        "compile_pi_schedule",
        "emit_cryo_manifest",
        "vent_cryo_vessel",
    ]
    cryo_lits = ["DR-BLUE-04", 12.5, "RAIL-X2", 38.0, 512, 2048, 6.15, 0.18, 0.998, "calibration.cym"]
    p_cryo_formal_full = (
        'vessel = cryopulse::CRYOPULSE_OPEN_CRYO_VESSEL(chamber_tag="DR-BLUE-04", target_mk=12.5)\n'
        'channel = cryopulse::CRYOPULSE_ENERGIZE_FLUX_COIL(vessel=$vessel, rail_id="RAIL-X2", bias_ua=38.0)\n'
        "trace = cryopulse::CRYOPULSE_SAMPLE_SQUID_FLUX(channel=$channel, window_ns=512, shots=2048)\n"
        "iq_data = cryopulse::CRYOPULSE_DEMODULATE_QUADRATURE(trace=$trace, lo_ghz=6.15)\n"
        "quenched = cryopulse::CRYOPULSE_QUENCH_TLS_DEFECTS(iq_data=$iq_data, damping_gamma=0.18)\n"
        "schedule = cryopulse::CRYOPULSE_COMPILE_PI_SCHEDULE(iq_data=$quenched, target_fidelity=0.998)\n"
        'manifest = cryopulse::CRYOPULSE_EMIT_CRYO_MANIFEST(schedule=$schedule, dest_path=@"calibration.cym")\n'
        "cryopulse::CRYOPULSE_VENT_CRYO_VESSEL(vessel=$vessel)"
    )
    p_cryo_formal_leak = (
        'vessel = cryopulse::CRYOPULSE_OPEN_CRYO_VESSEL(chamber_tag="DR-BLUE-04", target_mk=12.5)\n'
        'channel = cryopulse::CRYOPULSE_ENERGIZE_FLUX_COIL(vessel=$vessel, rail_id="RAIL-X2", bias_ua=38.0)\n'
        "trace = cryopulse::CRYOPULSE_SAMPLE_SQUID_FLUX(channel=$channel, window_ns=512, shots=2048)\n"
        "iq_data = cryopulse::CRYOPULSE_DEMODULATE_QUADRATURE(trace=$trace, lo_ghz=6.15)\n"
        "quenched = cryopulse::CRYOPULSE_QUENCH_TLS_DEFECTS(iq_data=$iq_data, damping_gamma=0.18)\n"
        "schedule = cryopulse::CRYOPULSE_COMPILE_PI_SCHEDULE(iq_data=$quenched, target_fidelity=0.998)\n"
        'manifest = cryopulse::CRYOPULSE_EMIT_CRYO_MANIFEST(schedule=$schedule, dest_path=@"calibration.cym")'
    )
    p_cryo_nl = (
        "open cryo vessel 'DR-BLUE-04' at 12.5 millikelvin, "
        "energize flux coil on rail 'RAIL-X2' at 38.0 microamps, "
        "sample squid flux for 512 ns with 2048 shots, "
        "demodulate quadrature at 6.15 GHz, "
        "quench tls defects with damping gamma 0.18, "
        "compile pi schedule with target fidelity 0.998, "
        "emit cryo manifest to 'calibration.cym', "
        "and vent cryo vessel"
    )

    aeth_req_calls = [
        "mount_optical_table",
        "lock_reference_laser",
        "arm_phase_modulator",
        "sweep_fringe_interferogram",
        "unwrap_phase_residues",
        "solve_zernike_wavefront",
        "export_phase_lut",
        "park_optical_table",
    ]
    aeth_lits = ["OPT-NORTH-2", 1550.0, "TEM00", 18.5, 16, 3.6, 65.0, 1024, "goldstein", 8, "zernike_n2.ael"]
    p_aeth_formal_full = (
        'bench = aetheris::AETHERIS_MOUNT_OPTICAL_TABLE(rack_id="OPT-NORTH-2", wavelength_nm=1550.0)\n'
        'cavity = aetheris::AETHERIS_LOCK_REFERENCE_LASER(bench=$bench, cavity_mode="TEM00", power_mw=18.5)\n'
        "modulator = aetheris::AETHERIS_ARM_PHASE_MODULATOR(cavity=$cavity, lane_count=16, vpi_volts=3.6)\n"
        "fringes = aetheris::AETHERIS_SWEEP_FRINGE_INTERFEROGRAM(modulator=$modulator, span_pm=65.0, frames=1024)\n"
        'phase = aetheris::AETHERIS_UNWRAP_PHASE_RESIDUES(fringes=$fringes, branch_cut="goldstein")\n'
        "model = aetheris::AETHERIS_SOLVE_ZERNIKE_WAVEFRONT(phase=$phase, order=8)\n"
        'lut = aetheris::AETHERIS_EXPORT_PHASE_LUT(model=$model, out_path=@"zernike_n2.ael")\n'
        "aetheris::AETHERIS_PARK_OPTICAL_TABLE(bench=$bench)"
    )
    p_aeth_formal_leak = (
        'bench = aetheris::AETHERIS_MOUNT_OPTICAL_TABLE(rack_id="OPT-NORTH-2", wavelength_nm=1550.0)\n'
        'cavity = aetheris::AETHERIS_LOCK_REFERENCE_LASER(bench=$bench, cavity_mode="TEM00", power_mw=18.5)\n'
        "modulator = aetheris::AETHERIS_ARM_PHASE_MODULATOR(cavity=$cavity, lane_count=16, vpi_volts=3.6)\n"
        "fringes = aetheris::AETHERIS_SWEEP_FRINGE_INTERFEROGRAM(modulator=$modulator, span_pm=65.0, frames=1024)\n"
        'phase = aetheris::AETHERIS_UNWRAP_PHASE_RESIDUES(fringes=$fringes, branch_cut="goldstein")\n'
        "model = aetheris::AETHERIS_SOLVE_ZERNIKE_WAVEFRONT(phase=$phase, order=8)\n"
        'lut = aetheris::AETHERIS_EXPORT_PHASE_LUT(model=$model, out_path=@"zernike_n2.ael")'
    )
    p_aeth_nl = (
        "mount optical table 'OPT-NORTH-2' at wavelength 1550.0 nm, "
        "lock reference laser on cavity mode 'TEM00' at 18.5 mW, "
        "arm phase modulator with 16 lanes and vpi 3.6 volts, "
        "sweep fringe interferogram over 65.0 pm with 1024 frames, "
        "unwrap phase residues with 'goldstein' branch cut, "
        "solve zernike wavefront at order 8, "
        "export phase lut to 'zernike_n2.ael', "
        "and park optical table"
    )

    pillow_only_check = _compose_checks(
        _check_domain_purity({"pillow"}, forbidden_domains={"cv2"}, forbidden_code_substrings=("import cv2", "cv2.")),
        _check_literals(["input.png", "pil_gray.png"]),
    )
    cv2_only_check = _compose_checks(
        _check_domain_purity({"cv2"}, forbidden_domains={"pillow"}, forbidden_code_substrings=("from PIL", "import PIL", "Image.")),
        _check_literals(["input.png", "cv2_gray.png"]),
    )

    nl_standard_refusal_reasons = (
        "MISSING_EXTERNAL_ASSET",
        "AMBIGUITY",
        "UNBOUND_PORT",
        "UNCLOSED_RESOURCE",
        "NO_MATCHING_OPERATION",
        "UNCOMPOSABLE_TYPES",
        "UNAVAILABLE_DOMAIN",
        "SLM_COMPILE_ERROR",
    )

    return [
        # =====================================================================
        # Group 1: formal (10 cases)
        # =====================================================================
        TestCase(
            task_id="formal_cv2_edges",
            group="formal",
            kind="formal_exact",
            prompt=p_cv2_edges,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_cv2_live_bridge",
            group="formal",
            kind="formal_bridge",
            prompt=p_cv2_bridge,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
            check_fn=_check_live_bridge(1),
        ),
        TestCase(
            task_id="formal_cross_pillow_cv2_bridge",
            group="formal",
            kind="formal_bridge",
            prompt=p_pillow_cv2_bridge,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
            check_fn=_check_live_bridge(2),
            notes="2-hop cross-domain bridge: pillow Image -> numpy BGR ndarray -> numpy gray -> cv2 Canny",
        ),
        TestCase(
            task_id="formal_pillow_pipeline",
            group="formal",
            kind="formal_exact",
            prompt=p_pillow_formal,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
            check_fn=_check_domain_purity({"pillow"}, forbidden_domains={"cv2"}),
        ),
        TestCase(
            task_id="formal_sqlite_pandas",
            group="formal",
            kind="formal_exact",
            prompt=p_sqlite_pd,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_mpl_multiout",
            group="formal",
            kind="formal_exact",
            prompt=p_mpl,
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="formal_semantic_cv2",
            group="formal",
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
            group="formal",
            kind="formal_refusal_proof",
            prompt=p_leak,
            execute=False,
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
            group="formal",
            kind="formal_refusal_proof",
            prompt=p_ref_bad,
            execute=False,
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
            group="formal",
            kind="formal_refusal_proof",
            prompt=p_missing_lit,
            execute=False,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNBOUND_PORT",),
                    rationale="Non-slot-fill profile rejects missing required thresholds cleanly",
                )
                for p in ("0", "A", "B")
            },
        ),

        # =====================================================================
        # Group 2: nl_core — Unvague vs Vague NL contrast (10 cases)
        # =====================================================================
        TestCase(
            task_id="nl_cv2_params",
            group="nl_core",
            kind="nl_explicit",
            prompt=p_nl_cv2,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="SLM compiles explicit CV2 pipeline cleanly") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_sqlite_pandas",
            group="nl_core",
            kind="nl_explicit",
            prompt=p_nl_sqlite,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="SLM compiles SQLite + Pandas + Close pipeline") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_cv2_implicit_params",
            group="nl_core",
            kind="nl_implicit_slots",
            prompt=p_nl_cv2_implicit,
            check_fn=lambda res: None if (res.plan is not None and any(s.cell_id == "CV2_CANNY" for s in _iter_plan_call_steps(res.plan.steps))) else f"Expected CV2_CANNY in plan, got {[s.cell_id for s in _iter_plan_call_steps(res.plan.steps)] if res.plan else None}",
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                "C": TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNBOUND_PORT", "SLM_COMPILE_ERROR", "UNCOMPOSABLE_TYPES"),
                    rationale="Profile C lacks slot-fill when required Canny thresholds are omitted",
                ),
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Slot-fill estimates missing non-asset Canny parameters") for p in ("D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_unvague_pillow",
            group="nl_core",
            kind="nl_explicit",
            prompt=p_nl_pillow_spec,
            check_fn=pillow_only_check,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Explicit Pillow pipeline across full lattice uses only Pillow") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_unvague_pygame",
            group="nl_core",
            kind="nl_explicit",
            prompt=p_nl_pygame_unvague,
            check_fn=_compose_checks(_check_domain_purity({"pygame"}), _check_literals([640, 480])),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Unvague pygame lifecycle acquires/releases display surface cleanly") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_unvague_pandas_etl",
            group="nl_core",
            kind="nl_explicit",
            prompt=p_nl_pandas_etl,
            check_fn=_compose_checks(_check_domain_purity({"pandas"}), _check_literals(["input.csv", "clean_out.csv"])),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Unvague CSV clean & export executes in sandbox cleanly") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="nl_vague_game",
            group="nl_core",
            kind="nl_refusal_proof",
            prompt=p_nl_vague_game,
            execute=False,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="Underspecified game prompt refuses instead of fabricating game assets/mechanics",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
        TestCase(
            task_id="nl_vague_ml",
            group="nl_core",
            kind="nl_refusal_proof",
            prompt=p_nl_vague_ml,
            execute=False,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="Underspecified dataset path & target refuses instead of fabricating paths",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
        TestCase(
            task_id="nl_audio_missing_file",
            group="nl_core",
            kind="nl_refusal_proof",
            prompt=p_nl_audio,
            execute=False,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="Omitted audio input/output asset paths refuse asset-hallucination cleanly",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
        TestCase(
            task_id="nl_bound_ambiguous",
            group="nl_core",
            kind="nl_refusal_proof",
            prompt=p_nl_bound_ambig,
            execute=False,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="Underspecified microservice prompt refuses honestly at type/asset/operation boundary",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),

        # =====================================================================
        # Group 3: unseen — Blind Zero-Pretraining Domains (6 cases)
        # =====================================================================
        TestCase(
            task_id="blind_cryopulse_formal",
            group="unseen",
            kind="blind_formal",
            domain_set="cryopulse_only",
            execute=False,
            prompt=p_cryo_formal_full,
            check_fn=_check_sdk_and_purity("cryopulse", cryo_req_calls, cryo_lits, {"cryopulse"}),
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="blind_cryopulse_nl",
            group="unseen",
            kind="blind_nl",
            domain_set="cryopulse_only",
            execute=False,
            prompt=p_cryo_nl,
            check_fn=_check_sdk_and_purity("cryopulse", cryo_req_calls, cryo_lits, {"cryopulse"}),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="0.8B SLM + STRATA synthesizes unseen CryoPulse SDK") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="blind_cryopulse_leak_formal",
            group="unseen",
            kind="blind_refusal_proof",
            domain_set="cryopulse_only",
            execute=False,
            prompt=p_cryo_formal_leak,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNCLOSED_RESOURCE",),
                    rationale="Unseen CryoPulse CryoVessel leak caught by domain-agnostic linear theorem",
                )
                for p in ("0", "A", "B")
            },
        ),
        TestCase(
            task_id="blind_aetheris_formal",
            group="unseen",
            kind="blind_formal",
            domain_set="aetheris_only",
            execute=False,
            prompt=p_aeth_formal_full,
            check_fn=_check_sdk_and_purity("aetheris", aeth_req_calls, aeth_lits, {"aetheris"}),
            theory_by_profile={p: formal_pass_exact for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="blind_aetheris_nl",
            group="unseen",
            kind="blind_nl",
            domain_set="aetheris_only",
            execute=False,
            prompt=p_aeth_nl,
            check_fn=_check_sdk_and_purity("aetheris", aeth_req_calls, aeth_lits, {"aetheris"}),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="0.8B SLM + STRATA synthesizes unseen Aetheris SDK avoiding distractors") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="blind_aetheris_leak_formal",
            group="unseen",
            kind="blind_refusal_proof",
            domain_set="aetheris_only",
            execute=False,
            prompt=p_aeth_formal_leak,
            theory_by_profile={
                p: TheoryExpectation(
                    mode="EXPECTED_REFUSAL",
                    allowed_refusals=("UNCLOSED_RESOURCE",),
                    rationale="Unseen Aetheris OpticalBench leak caught by domain-agnostic linear theorem",
                )
                for p in ("0", "A", "B")
            },
        ),

        # =====================================================================
        # Group 4: haystack — Small Blind Domain in 1,038 Noise Cells (2 cases)
        # =====================================================================
        TestCase(
            task_id="haystack_cryopulse_1049_noise",
            group="haystack",
            kind="noise_isolation",
            domain_set="haystack_cryopulse",
            execute=False,
            prompt=p_cryo_nl,
            check_fn=_check_sdk_and_purity("cryopulse", cryo_req_calls, cryo_lits, {"cryopulse"}),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Finds 11-cell CryoPulse chain inside 1,049-cell lattice with zero noise leak") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="haystack_aetheris_1049_noise",
            group="haystack",
            kind="noise_isolation",
            domain_set="haystack_aetheris",
            execute=False,
            prompt=p_aeth_nl,
            check_fn=_check_sdk_and_purity("aetheris", aeth_req_calls, aeth_lits, {"aetheris"}),
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Finds 15-cell Aetheris chain inside 1,053-cell lattice avoiding traps & noise") for p in ("C", "D", "E", "O")},
            },
        ),

        # =====================================================================
        # Group 5: wrong_domain — Unrelated / Missing Domains (2 cases)
        # =====================================================================
        TestCase(
            task_id="wrong_domain_image_in_sqlite",
            group="wrong_domain",
            kind="wrong_domain_refusal",
            domain_set="sqlite_only",
            execute=False,
            prompt=p_nl_cv2,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="Image pipeline refuses cleanly when only sqlite3 + python_core are loaded",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),
        TestCase(
            task_id="wrong_domain_cryo_not_loaded",
            group="wrong_domain",
            kind="wrong_domain_refusal",
            domain_set="default",
            execute=False,
            prompt=p_cryo_nl,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{
                    p: TheoryExpectation(
                        mode="EXPECTED_REFUSAL",
                        allowed_refusals=nl_standard_refusal_reasons,
                        rationale="CryoPulse pipeline refuses cleanly across standard domains when cryopulse is unloaded",
                    )
                    for p in ("C", "D", "E", "O")
                },
            },
        ),

        # =====================================================================
        # Group 6: domain_spec — Explicit Library Adherence (4 cases)
        # =====================================================================
        TestCase(
            task_id="spec_pillow_only_cv2_formal",
            group="domain_spec",
            kind="domain_spec_refusal",
            domain_set="cv2_only",
            execute=False,
            prompt=p_pillow_formal,
            theory_by_profile={p: formal_unavail_refusal for p in ("0", "A", "B")},
        ),
        TestCase(
            task_id="spec_pillow_only_cv2_nl",
            group="domain_spec",
            kind="domain_spec_refusal",
            domain_set="cv2_only",
            execute=False,
            prompt=p_nl_pillow_spec,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: nl_unavail_refusal for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="spec_pillow_both_loaded_nl",
            group="domain_spec",
            kind="domain_spec_pass",
            domain_set="cv2_and_pillow",
            execute=True,
            prompt=p_nl_pillow_spec,
            check_fn=pillow_only_check,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Explicit Pillow spec uses strictly Pillow even when cv2 is loaded") for p in ("C", "D", "E", "O")},
            },
        ),
        TestCase(
            task_id="spec_cv2_both_loaded_nl",
            group="domain_spec",
            kind="domain_spec_pass",
            domain_set="cv2_and_pillow",
            execute=True,
            prompt=p_nl_cv2_spec,
            check_fn=cv2_only_check,
            theory_by_profile={
                "0": formal_nl_refusal,
                "A": formal_nl_refusal,
                "B": formal_nl_refusal,
                **{p: TheoryExpectation(mode="PASSABLE", rationale="Explicit OpenCV spec uses strictly cv2 even when Pillow is loaded") for p in ("C", "D", "E", "O")},
            },
        ),
    ]


def _print_matrix(catalog: List[TestCase], profiles: List[str]) -> None:
    print("=" * 112)
    print(" PRE-RUN THEORETICAL PASSABILITY MATRIX  (Method: unified | STRATA v2 Unified Type-Directed Planner)")
    print("=" * 112)
    hdr = f"{'Group':<13} {'Task ID':<32} {'DomainSet':<18} " + " ".join(f"{p:^6}" for p in profiles)
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
            cells.append(f"{lbl:^6}")
        print(f"{tc.group:<13} {tc.task_id:<32} {tc.domain_set:<18} " + " ".join(cells))
    print("=" * 112 + "\n", flush=True)


def _print_failure_diagnostic(
    rec: RunRecord,
    tc: TestCase,
    pipe: StrataPipeline,
) -> None:
    res = rec.result
    print("\n" + "!" * 112)
    print(f" EARLY STOP ON FIRST FAILURE [{rec.phase}] -> {rec.outcome}")
    print("!" * 112)
    print(f"Profile      : {rec.profile}")
    print(f"Method       : {rec.method}")
    print(f"Group        : {tc.group} | DomainSet: {tc.domain_set} ({len(pipe.cells)} cells)")
    print(f"Task ID      : {rec.task_id} ({tc.kind})")
    print(f"Expected     : {rec.expected_mode}")
    print(f"Outcome      : {rec.outcome} ({rec.detail})")
    print(f"Elapsed      : {rec.elapsed_ms:.1f} ms")
    print(f"Prompt       :\n{tc.prompt}")
    if res is None:
        print("Result       : None")
        print("!" * 112 + "\n", flush=True)
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
            _, allowed_doms, strict_map, _ = pipe._resolve_domain_constraints(res.ir)
            cands_map = pipe.retriever.retrieve_for_ir(
                res.ir,
                top_k=5,
                allowed_domains=allowed_doms,
                strict_domains_by_node=strict_map,
            )
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

    print("!" * 112 + "\n", flush=True)


def _get_or_init_pipeline(
    profile_code: str,
    domain_set: str,
    domain_loader: _DomainSetLoader,
    sandbox: Optional[GEVRSandbox],
    args: argparse.Namespace,
    initialized_profiles: Set[str],
    pipe_cache: Dict[Tuple[str, str], StrataPipeline],
    embed_vec_cache: Dict[Tuple[str, str, str], List[float]],
    embed_cache_id: str,
) -> StrataPipeline:
    cache_key = (profile_code, domain_set)
    if cache_key in pipe_cache:
        return pipe_cache[cache_key]

    spec = resolve_profile_spec(profile_code)
    mm = ModelManager.get_instance()
    if profile_code not in initialized_profiles:
        t_init0 = time.perf_counter()
        print(f"[init] Initializing Profile {profile_code} ({spec.display_name})...", flush=True)
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
        initialized_profiles.add(profile_code)
        print(f"[init] Profile {profile_code} models ready in {(time.perf_counter() - t_init0) * 1000.0:.1f} ms", flush=True)
    else:
        rr_obj = LocalReranker.get_shared(device="auto") if (spec.uses_reranker and args.use_reranker) else None

    embed_override = None
    if spec.uses_embedder:
        raw_embed = make_embed_fn(mm)
        if raw_embed is not None:
            def _cached_embed(texts: List[str], *, mode: str = "document") -> List[List[float]]:
                missing_texts: List[str] = []
                seen_miss: Set[str] = set()
                for txt in texts:
                    ek = (embed_cache_id, mode, txt)
                    if ek not in embed_vec_cache and txt not in seen_miss:
                        missing_texts.append(txt)
                        seen_miss.add(txt)
                if missing_texts:
                    vecs = raw_embed(missing_texts, mode=mode)
                    for txt, vec in zip(missing_texts, vecs):
                        embed_vec_cache[(embed_cache_id, mode, txt)] = list(vec)
                return [embed_vec_cache[(embed_cache_id, mode, txt)] for txt in texts]

            embed_override = _cached_embed

    t_pipe0 = time.perf_counter()
    if domain_set == "default":
        compiled = domain_loader.get_compiled_default()
        pipe = build_strata_pipeline_for_profile(
            compiled,
            spec,
            model_manager=mm,
            reranker=rr_obj,
            sandbox=sandbox,
            timeout=args.timeout,
            embed_fn_override=embed_override,
        )
    else:
        trees = domain_loader.get_trees(domain_set)
        pipe = build_strata_pipeline_from_trees(
            trees,
            spec,
            model_manager=mm,
            reranker=rr_obj,
            sandbox=sandbox,
            timeout=args.timeout,
            embed_fn_override=embed_override,
        )
    if spec.uses_embedder:
        pipe.retriever._ensure_doc_embeddings()
    print(
        f"[domain] Profile {profile_code} | {domain_set:<18} ({len(pipe.cells)} cells) built in "
        f"{(time.perf_counter() - t_pipe0) * 1000.0:.1f} ms",
        flush=True,
    )
    pipe_cache[cache_key] = pipe
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser(description="STRATA v2 Theory-First Empirical Multi-Profile Runner")
    ap.add_argument("--profiles", type=str, default=",".join(DEFAULT_PROFILES), help="Comma-separated profiles (0,A,B,C,D,E,O)")
    ap.add_argument(
        "--group",
        type=str,
        default="all",
        help=f"Comma-separated test groups or 'all' ({','.join(VALID_GROUPS)})",
    )
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
    all_catalog = _build_test_catalog()
    raw_groups = [g.strip().lower() for g in args.group.split(",") if g.strip()]
    if not raw_groups or "all" in raw_groups:
        selected_groups = set(VALID_GROUPS)
    else:
        bad_groups = [g for g in raw_groups if g not in VALID_GROUPS]
        if bad_groups:
            raise ValueError(f"Unknown group(s) {bad_groups}. Valid choices: {', '.join(VALID_GROUPS)}, all")
        selected_groups = set(raw_groups)
    catalog = [tc for tc in all_catalog if tc.group in selected_groups]
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
        f"| sandbox={'OFF' if args.no_exec else 'ON'} | early_stop={args.early_stop} "
        f"| cases={len(catalog)}"
    )

    domain_loader = _DomainSetLoader(ROOT_DIR)
    compiled = domain_loader.get_compiled_default()
    sandbox = (
        None
        if args.no_exec
        else GEVRSandbox(asset_dirs=[ROOT_DIR / "benchmarks" / "fixtures" / "media"])
    )
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
    initialized_profiles: Set[str] = set()
    pipe_cache: Dict[Tuple[str, str], StrataPipeline] = {}
    embed_vec_cache: Dict[Tuple[str, str, str], List[float]] = {}
    embed_cache_id = str(emb_sel)

    try:
        for phase, p_code, tc, exp in schedule:
            pipe = _get_or_init_pipeline(
                p_code,
                tc.domain_set,
                domain_loader,
                sandbox,
                args,
                initialized_profiles,
                pipe_cache,
                embed_vec_cache,
                embed_cache_id,
            )

            t0 = time.perf_counter()
            res = pipe.run(tc.prompt, execute=(not args.no_exec) and tc.execute)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if exp.mode == "PASSABLE":
                extra_err = tc.check_fn(res) if (res.ok and tc.check_fn is not None) else None
                if res.ok and extra_err is None:
                    rec = RunRecord(
                        phase=phase,
                        group=tc.group,
                        domain_set=tc.domain_set,
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
                    print(
                        f"[PASS]           P={p_code} | {tc.group:<12} | {tc.task_id:<32} | "
                        f"{elapsed_ms:7.1f} ms | bridges={res.live_bridge_count}",
                        flush=True,
                    )
                else:
                    rec = RunRecord(
                        phase=phase,
                        group=tc.group,
                        domain_set=tc.domain_set,
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
                        group=tc.group,
                        domain_set=tc.domain_set,
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
                    print(
                        f"[PROOF OF THEORY] P={p_code} | {tc.group:<12} | {tc.task_id:<32} | "
                        f"{elapsed_ms:7.1f} ms | refusal={res.refusal_kind}",
                        flush=True,
                    )
                elif res.ok:
                    rec = RunRecord(
                        phase=phase,
                        group=tc.group,
                        domain_set=tc.domain_set,
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
                        group=tc.group,
                        domain_set=tc.domain_set,
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

        print("\n" + "=" * 112)
        print(" FINAL TEST SUMMARY")
        print("=" * 112)
        for grp in VALID_GROUPS:
            if grp not in selected_groups:
                continue
            g_recs = [r for r in records if r.group == grp]
            if not g_recs:
                continue
            gp_ok = sum(1 for r in g_recs if r.outcome == "PASSED")
            gp_tot = sum(1 for r in g_recs if r.phase == "PHASE1_PASSABLE")
            gt_ok = sum(1 for r in g_recs if r.outcome == "PROOF OF THEORY")
            gt_tot = sum(1 for r in g_recs if r.phase == "PHASE2_PROOF")
            gf = sum(1 for r in g_recs if not r.ok)
            print(f"  {grp:<14}: Passable {gp_ok:2d}/{gp_tot:<2d} | Proof {gt_ok:2d}/{gt_tot:<2d} | Fails {gf}")
        print("-" * 112)
        p_ok = sum(1 for r in records if r.outcome == "PASSED")
        p_tot = sum(1 for r in records if r.phase == "PHASE1_PASSABLE")
        t_ok = sum(1 for r in records if r.outcome == "PROOF OF THEORY")
        t_tot = sum(1 for r in records if r.phase == "PHASE2_PROOF")
        f_cnt = sum(1 for r in records if not r.ok)
        print(f"Passable Passed   : {p_ok}/{p_tot}")
        print(f"Proof-of-Theory   : {t_ok}/{t_tot}")
        print(f"Total Verified    : {p_ok + t_ok}/{p_tot + t_tot}")
        print(f"Unexpected Fails  : {f_cnt}")
        print("=" * 112, flush=True)
        return 0 if f_cnt == 0 else 1
    finally:
        ModelManager.get_instance().cleanup(release_reranker=True)


if __name__ == "__main__":
    raise SystemExit(main())
