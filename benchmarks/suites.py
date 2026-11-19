"""
benchmarks/suites.py - Canonical benchmark task suites for STRATA vs Pure LLMs.

Groups:
  - core8:      Original 8 presets from test.py (sandbox-verified)
  - cryopulse:  Unseen domain #1 (linear docs, 3 tasks)
  - core9:      core8 + cryopulse (the 9 benchmark suites / 11 tasks)
  - aetheris:   Unseen domain #2 (alphabetical docs + deprecated/typestate traps, 3 tasks)
  - user_real:  Realistic multi-library user workflows (3 tasks, sandbox-verified)
  - boundary:   Honest limitation controls (under-specified refusal + non-lattice algorithm)
  - stress:     aetheris + user_real + boundary (8 tasks)
  - all:        core9 + stress (19 tasks total)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

BENCH_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCH_DIR / "fixtures"

CORE_8_TASKS: List[Dict[str, Any]] = [
    {
        "id": "canonical",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Canonical Multi-Stage Pipeline (Pandas + FFT + Sklearn Regression)",
        "prompt": (
            'load a csv file named "input.csv", normalize X column, drop null values, '
            'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
            'then make a regression model trained on X and Y to predict Z'
        ),
        "expected_literals": ["input.csv", "X", "Y", "Z"],
    },
    {
        "id": "vague_game",
        "group": "core8",
        "kind": "sandbox",
        "expected": "clarify",
        "name": "Vague / Abstract 2D Game Loop (Pygame)",
        "prompt": "make a 2D game in pygame with a display window, game loop, clock, and quit",
        "expected_literals": [],
    },
    {
        "id": "unvague_game",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Detailed Closed-Domain Snake Game (Pygame)",
        "prompt": (
            "initialize pygame and font system, create an 800 by 600 display window with title 'Snake Game', "
            "initialize clock, create an Arial size 24 font, create snake head rect at x 400 y 300 width 20 height 20, "
            "create food rect at x 200 y 200 width 20 height 20, create boundary rect at x 0 y 0 width 800 height 600, "
            "create background color black 0 0 0 255, green snake color 0 255 0 255, red food color 255 0 0 255, and white text color 255 255 255 255, "
            "run main game loop while running is True: get events and key presses, move snake rect by dx 20 and dy 0, "
            "clamp snake rect to boundary rect, check rect collision between snake rect and food rect, "
            "fill screen surface with black color, draw green snake rect and red food rect on screen surface, "
            "render score text surface 'Score: 0' with white color and blit it at position (10, 10) onto screen, "
            "tick clock at 15 fps, flip display, and finally quit font and quit pygame"
        ),
        "expected_literals": [800, 600, "Snake Game", 24, 400, 300, 20, 200, 255, 10, 15, "Score: 0"],
    },
    {
        "id": "vague_ml",
        "group": "core8",
        "kind": "sandbox",
        "expected": "clarify",
        "name": "Vague / Abstract Machine Learning Model",
        "prompt": "load dataset and train a machine learning model to predict target",
        "expected_literals": [],
    },
    {
        "id": "cv2_params",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Deep Single-Domain Computer Vision with Explicit Parameters (CV2)",
        "prompt": (
            'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
            'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"'
        ),
        "expected_literals": ["input.png", 5, 1.5, 50, 150, "edges.png"],
    },
    {
        "id": "control_flow",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Intensive Python Control Flow (def, while, if/else, try/except)",
        "prompt": (
            "define a function that processes data with a while loop, "
            "branches on condition with if else, and handles exceptions with try except"
        ),
        "expected_literals": [],
    },
    {
        "id": "crypto",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Cryptographic Pipeline (SHA256, AES Cipher, Base64)",
        "prompt": "read file 'input.csv', hash data using sha256, encrypt payload with aes cipher, and encode with base64",
        "expected_literals": ["input.csv"],
    },
    {
        "id": "audio",
        "group": "core8",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Audio Signal Processing Pipeline (Scipy)",
        "prompt": "load audio file, apply high-pass filter, compute spectrogram, and export filtered audio",
        "expected_literals": [],
    },
]

CRYOPULSE_SPEC: Dict[str, Any] = {
    "module": "cryopulse",
    "docs_path": str(FIXTURES_DIR / "cryopulse" / "cryopulse_docs.md"),
    "tree_path": str(FIXTURES_DIR / "cryopulse" / "cryopulse.json"),
    "api_specs": {
        "open_cryo_vessel": {"pos": ["chamber_tag", "target_mk"], "kw": {"chamber_tag", "target_mk"}},
        "energize_flux_coil": {"pos": ["vessel", "rail_id", "bias_ua"], "kw": {"vessel", "rail_id", "bias_ua"}},
        "sample_squid_flux": {"pos": ["channel", "window_ns", "shots"], "kw": {"channel", "window_ns", "shots"}},
        "demodulate_quadrature": {"pos": ["trace", "lo_ghz"], "kw": {"trace", "lo_ghz"}},
        "quench_tls_defects": {"pos": ["iq_data", "damping_gamma"], "kw": {"iq_data", "damping_gamma"}},
        "compile_pi_schedule": {"pos": ["iq_data", "target_fidelity"], "kw": {"iq_data", "target_fidelity"}},
        "emit_cryo_manifest": {"pos": ["schedule", "dest_path"], "kw": {"schedule", "dest_path"}},
        "measure_mixing_plate": {"pos": ["vessel"], "kw": {"vessel"}},
        "audit_squid_snr": {"pos": ["trace"], "kw": {"trace"}},
        "normalize_iq_gain": {"pos": ["iq_data"], "kw": {"iq_data"}},
        "vent_cryo_vessel": {"pos": ["vessel"], "kw": {"vessel"}},
    },
    "prereqs": [
        ("open_cryo_vessel", "energize_flux_coil"),
        ("energize_flux_coil", "sample_squid_flux"),
        ("sample_squid_flux", "demodulate_quadrature"),
        ("demodulate_quadrature", "quench_tls_defects"),
        ("quench_tls_defects", "compile_pi_schedule"),
        ("compile_pi_schedule", "emit_cryo_manifest"),
        ("open_cryo_vessel", "measure_mixing_plate"),
        ("sample_squid_flux", "audit_squid_snr"),
    ],
}

CRYOPULSE_TASKS: List[Dict[str, Any]] = [
    {
        "id": "cryo_explicit",
        "group": "cryopulse",
        "kind": "sdk",
        "sdk": "cryopulse",
        "expected": "synthesize",
        "name": "Explicit 7-Stage CryoPulse Calibration Pipeline",
        "prompt": (
            "open cryo vessel 'DR-BLUE-04' at 12.5 millikelvin, "
            "energize flux coil on rail 'RAIL-X2' at 38.0 microamps, "
            "sample squid flux for 512 ns with 2048 shots, "
            "demodulate quadrature at 6.15 GHz, "
            "quench tls defects with damping gamma 0.18, "
            "compile pi schedule with target fidelity 0.998, "
            "and emit cryo manifest to 'calibration.cym'"
        ),
        "required_calls": [
            "open_cryo_vessel",
            "energize_flux_coil",
            "sample_squid_flux",
            "demodulate_quadrature",
            "quench_tls_defects",
            "compile_pi_schedule",
            "emit_cryo_manifest",
        ],
        "expected_literals": ["DR-BLUE-04", 12.5, "RAIL-X2", 38.0, 512, 2048, 6.15, 0.18, 0.998, "calibration.cym"],
    },
    {
        "id": "cryo_implicit",
        "group": "cryopulse",
        "kind": "sdk",
        "sdk": "cryopulse",
        "expected": "synthesize",
        "name": "Implicit Prerequisite / Typestate Bridging Pipeline",
        "prompt": (
            "initialize cryo vessel 'DR-RED-09', "
            "capture squid flux trace, "
            "demodulate quadrature at 5.90 GHz, "
            "compile pi pulse schedule at fidelity 0.997, "
            "and emit cryo manifest to 'red09.cym'"
        ),
        "required_calls": [
            "open_cryo_vessel",
            "energize_flux_coil",
            "sample_squid_flux",
            "demodulate_quadrature",
            "quench_tls_defects",
            "compile_pi_schedule",
            "emit_cryo_manifest",
        ],
        "expected_literals": ["DR-RED-09", 5.9, 0.997, "red09.cym"],
    },
    {
        "id": "cryo_diag",
        "group": "cryopulse",
        "kind": "sdk",
        "sdk": "cryopulse",
        "expected": "synthesize",
        "name": "Telemetry + Diagnostic SNR Audit Pipeline",
        "prompt": (
            "open cryo vessel 'QX-ALPHA', "
            "measure mixing plate temperature, "
            "energize flux coil on rail 'FLUX-01', "
            "sample squid flux, "
            "audit squid snr, "
            "and demodulate quadrature"
        ),
        "required_calls": [
            "open_cryo_vessel",
            "measure_mixing_plate",
            "energize_flux_coil",
            "sample_squid_flux",
            "audit_squid_snr",
            "demodulate_quadrature",
        ],
        "expected_literals": ["QX-ALPHA", "FLUX-01"],
    },
]

AETHERIS_SPEC: Dict[str, Any] = {
    "module": "aetheris",
    "docs_path": str(FIXTURES_DIR / "aetheris" / "aetheris_docs.md"),
    "tree_path": str(FIXTURES_DIR / "aetheris" / "aetheris.json"),
    "api_specs": {
        "mount_optical_table": {"pos": ["rack_id", "wavelength_nm"], "kw": {"rack_id", "wavelength_nm"}},
        "lock_reference_laser": {"pos": ["bench", "cavity_mode", "power_mw"], "kw": {"bench", "cavity_mode", "power_mw"}},
        "arm_phase_modulator": {"pos": ["cavity", "lane_count", "vpi_volts"], "kw": {"cavity", "lane_count", "vpi_volts"}},
        "sweep_fringe_interferogram": {"pos": ["modulator", "span_pm", "frames"], "kw": {"modulator", "span_pm", "frames"}},
        "unwrap_phase_residues": {"pos": ["fringes", "branch_cut"], "kw": {"fringes", "branch_cut"}},
        "solve_zernike_wavefront": {"pos": ["phase", "order"], "kw": {"phase", "order"}},
        "export_phase_lut": {"pos": ["model", "out_path"], "kw": {"model", "out_path"}},
        "measure_bench_drift": {"pos": ["bench"], "kw": {"bench"}},
        "audit_fringe_contrast": {"pos": ["fringes"], "kw": {"fringes"}},
        "calibrate_thermal_shroud": {"pos": ["cavity", "target_k"], "kw": {"cavity", "target_k"}},
        "park_optical_table": {"pos": ["bench"], "kw": {"bench"}},
    },
    "forbidden_calls": {
        "legacy_mount_bench",
        "tune_laser_coarse",
        "quick_phase_estimate",
        "dump_raw_fringes_csv",
    },
    "prereqs": [
        ("mount_optical_table", "lock_reference_laser"),
        ("lock_reference_laser", "arm_phase_modulator"),
        ("arm_phase_modulator", "sweep_fringe_interferogram"),
        ("sweep_fringe_interferogram", "unwrap_phase_residues"),
        ("unwrap_phase_residues", "solve_zernike_wavefront"),
        ("solve_zernike_wavefront", "export_phase_lut"),
        ("mount_optical_table", "measure_bench_drift"),
        ("sweep_fringe_interferogram", "audit_fringe_contrast"),
    ],
}

AETHERIS_TASKS: List[Dict[str, Any]] = [
    {
        "id": "aetheris_explicit",
        "group": "aetheris",
        "kind": "sdk",
        "sdk": "aetheris",
        "expected": "synthesize",
        "name": "Explicit 7-Stage Aetheris Interferometry (Alphabetical Docs + Distractors)",
        "prompt": (
            "mount optical table 'OPT-NORTH-2' at wavelength 1550.0 nm, "
            "lock reference laser on cavity mode 'TEM00' at 18.5 mW, "
            "arm phase modulator with 16 lanes and vpi 3.6 volts, "
            "sweep fringe interferogram over 65.0 pm with 1024 frames, "
            "unwrap phase residues with 'goldstein' branch cut, "
            "solve zernike wavefront at order 8, "
            "and export phase lut to 'zernike_n2.ael'"
        ),
        "required_calls": [
            "mount_optical_table",
            "lock_reference_laser",
            "arm_phase_modulator",
            "sweep_fringe_interferogram",
            "unwrap_phase_residues",
            "solve_zernike_wavefront",
            "export_phase_lut",
        ],
        "expected_literals": ["OPT-NORTH-2", 1550.0, "TEM00", 18.5, 16, 3.6, 65.0, 1024, "goldstein", 8, "zernike_n2.ael"],
    },
    {
        "id": "aetheris_implicit",
        "group": "aetheris",
        "kind": "sdk",
        "sdk": "aetheris",
        "expected": "synthesize",
        "name": "Implicit Typestate Bridge (Skips Arming & Unwrapping in Prompt)",
        "prompt": (
            "mount optical table 'OPT-SOUTH-7', "
            "lock reference laser cavity 'HE11', "
            "sweep fringe interferogram, "
            "solve zernike wavefront at order 5, "
            "and export phase lut to 'south7.ael'"
        ),
        "required_calls": [
            "mount_optical_table",
            "lock_reference_laser",
            "arm_phase_modulator",
            "sweep_fringe_interferogram",
            "unwrap_phase_residues",
            "solve_zernike_wavefront",
            "export_phase_lut",
        ],
        "expected_literals": ["OPT-SOUTH-7", "HE11", 5, "south7.ael"],
    },
    {
        "id": "aetheris_diag",
        "group": "aetheris",
        "kind": "sdk",
        "sdk": "aetheris",
        "expected": "synthesize",
        "name": "Non-Linear Diagnostic Branch + Fringe Contrast Audit",
        "prompt": (
            "mount optical table 'LAB-9', "
            "measure bench drift, "
            "lock reference laser on cavity 'TE01', "
            "sweep fringe interferogram, "
            "and audit fringe contrast"
        ),
        "required_calls": [
            "mount_optical_table",
            "measure_bench_drift",
            "lock_reference_laser",
            "arm_phase_modulator",
            "sweep_fringe_interferogram",
            "audit_fringe_contrast",
        ],
        "expected_literals": ["LAB-9", "TE01"],
    },
]

USER_REALISTIC_TASKS: List[Dict[str, Any]] = [
    {
        "id": "user_etl_sqlite",
        "group": "user_real",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Realistic Cross-Domain ETL: CSV Clean + Normalize + SQLite Persist",
        "prompt": (
            "load csv file 'input.csv', drop missing rows, normalize column 'X', "
            "compute correlation matrix, and save the cleaned dataframe to sqlite database 'metrics.db' table 'clean_data'"
        ),
        "expected_literals": ["input.csv", "X", "metrics.db", "clean_data"],
    },
    {
        "id": "user_cv_contour",
        "group": "user_real",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Realistic Vision Pipeline: Grayscale + Median Blur + Threshold + Contours",
        "prompt": (
            "read image 'input.png', convert to grayscale, apply median blur with kernel size 5, "
            "apply Otsu binary threshold, find contours, and save visualization to 'contours.png'"
        ),
        "expected_literals": ["input.png", 5, "contours.png"],
    },
    {
        "id": "user_audio_fft_export",
        "group": "user_real",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Realistic DSP Pipeline: WAV Load + Bandpass + Spectrogram Figure",
        "prompt": (
            "read wav audio file 'input_1.wav', apply low-pass Butterworth filter, "
            "compute spectrogram, and save spectrogram plot to 'spec_out.png'"
        ),
        "expected_literals": ["input_1.wav", "spec_out.png"],
    },
]

HONEST_BOUNDARY_TASKS: List[Dict[str, Any]] = [
    {
        "id": "bound_ambiguous",
        "group": "boundary",
        "kind": "sandbox",
        "expected": "clarify",
        "name": "Honest Refusal Control: Underspecified Production Service Request",
        "prompt": "build an end-to-end anomaly detection microservice with alerts and automatic retraining",
        "expected_literals": [],
    },
    {
        "id": "bound_custom_algo",
        "group": "boundary",
        "kind": "sandbox",
        "expected": "synthesize",
        "name": "Honest Frontier Boundary: Custom Recursive A* Grid Pathfinder from Scratch",
        "prompt": (
            "write a Python function astar_grid(grid, start, goal) implementing A* shortest-path search "
            "with Manhattan distance heuristic using heapq and verify the shortest path on a 5x5 obstacle grid"
        ),
        "expected_literals": [5],
    },
]

SDK_SPECS: Dict[str, Dict[str, Any]] = {
    "cryopulse": CRYOPULSE_SPEC,
    "aetheris": AETHERIS_SPEC,
}

SUITE_GROUPS: Dict[str, List[Dict[str, Any]]] = {
    "core8": CORE_8_TASKS,
    "cryopulse": CRYOPULSE_TASKS,
    "core9": CORE_8_TASKS + CRYOPULSE_TASKS,
    "aetheris": AETHERIS_TASKS,
    "user_real": USER_REALISTIC_TASKS,
    "boundary": HONEST_BOUNDARY_TASKS,
    "stress": AETHERIS_TASKS + USER_REALISTIC_TASKS + HONEST_BOUNDARY_TASKS,
    "all": CORE_8_TASKS + CRYOPULSE_TASKS + AETHERIS_TASKS + USER_REALISTIC_TASKS + HONEST_BOUNDARY_TASKS,
}


def get_suite_tasks(suite_name: str = "core9") -> List[Dict[str, Any]]:
    if suite_name not in SUITE_GROUPS:
        raise ValueError(f"Unknown suite '{suite_name}'. Valid choices: {sorted(SUITE_GROUPS.keys())}")
    return list(SUITE_GROUPS[suite_name])
