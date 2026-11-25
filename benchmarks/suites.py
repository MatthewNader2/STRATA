"""
benchmarks/suites.py - Canonical benchmark task suites for STRATA vs Pure / SKILL.md LLMs.

Legacy Groups (preserved for backward compatibility):
  - core8:      Original 8 presets (sandbox-verified)
  - cryopulse:  Unseen domain #1 (linear docs, 3 tasks)
  - core9:      core8 + cryopulse (11 tasks)
  - aetheris:   Unseen domain #2 (alphabetical docs + deprecated/typestate traps, 3 tasks)
  - user_real:  Realistic multi-library user workflows (3 tasks)
  - boundary:   Honest limitation controls (under-specified refusal + non-lattice algorithm)
  - stress:     aetheris + user_real + boundary (8 tasks)
  - all:        core9 + stress (19 tasks total)

Expanded v2 Groups (aligned with expanded 34-case v2 test.py):
  - v2_nl_core:      10 NL unvague-vs-vague tasks (6 synthesize + 4 clarify refusals)
  - v2_unseen:       2 full-lifecycle blind SDK tasks (CryoPulse + Aetheris)
  - v2_haystack:     2 needle-in-a-haystack blind SDK tasks inside 15-domain catalog
  - v2_wrong_domain: 2 wrong/missing-domain refusal controls
  - v2_domain_spec:  3 explicit library adherence & substitution-refusal tasks
  - v2_expanded:     All 19 v2 comparative tasks above
  - v2_full:         v2_expanded + cryo_implicit/diag + aetheris_implicit/diag + user_real + bound_custom_algo (27 tasks)
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
        "artifact_spec": {
            "kind": "sqlite_clean_table",
            "db_path": "metrics.db",
            "table": "clean_data",
            "source_csv": "input.csv",
            "reject_square_identity_matrix": True,
        },
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

_V2_NL_CV2_PROMPT = (
    'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
    'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"'
)
_V2_NL_PILLOW_SPEC_PROMPT = (
    'using Pillow, read image "input.png", convert to grayscale, and save to "pil_gray.png"'
)
_V2_NL_CV2_SPEC_PROMPT = (
    'using OpenCV (cv2), read image "input.png", convert to grayscale, and save to "cv2_gray.png"'
)
_V2_CRYO_NL_FULL_PROMPT = (
    "open cryo vessel 'DR-BLUE-04' at 12.5 millikelvin, "
    "energize flux coil on rail 'RAIL-X2' at 38.0 microamps, "
    "sample squid flux for 512 ns with 2048 shots, "
    "demodulate quadrature at 6.15 GHz, "
    "quench tls defects with damping gamma 0.18, "
    "compile pi schedule with target fidelity 0.998, "
    "emit cryo manifest to 'calibration.cym', "
    "and vent cryo vessel"
)
_V2_AETH_NL_FULL_PROMPT = (
    "mount optical table 'OPT-NORTH-2' at wavelength 1550.0 nm, "
    "lock reference laser on cavity mode 'TEM00' at 18.5 mW, "
    "arm phase modulator with 16 lanes and vpi 3.6 volts, "
    "sweep fringe interferogram over 65.0 pm with 1024 frames, "
    "unwrap phase residues with 'goldstein' branch cut, "
    "solve zernike wavefront at order 8, "
    "export phase lut to 'zernike_n2.ael', "
    "and park optical table"
)

V2_NL_CORE_TASKS: List[Dict[str, Any]] = [
    {
        "id": "v2_nl_cv2_params",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: Explicit OpenCV Canny Edge Pipeline",
        "prompt": _V2_NL_CV2_PROMPT,
        "input_assets": ["input.png"],
        "egress_paths": ["edges.png"],
        "required_imports": ["cv2"],
        "expected_literals": ["input.png", 5, 1.5, 50, 150, "edges.png"],
    },
    {
        "id": "v2_nl_sqlite_pandas",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: SQLite Query -> Pandas DataFrame -> Close",
        "prompt": (
            'open an in-memory sqlite database connection, run SQL query "SELECT 1 AS a, 2 AS b" '
            "into a pandas dataframe, and close the sqlite connection"
        ),
        "required_imports": ["sqlite3", "pandas"],
        "expected_literals": ["SELECT 1 AS a, 2 AS b"],
    },
    {
        "id": "v2_nl_cv2_implicit_params",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: Implicit Slot Estimation for Canny Thresholds",
        "prompt": 'read image "input.png", convert to grayscale, and detect Canny edges',
        "input_assets": ["input.png"],
        "required_imports": ["cv2"],
        "required_calls_attr": ["Canny"],
        "expected_literals": ["input.png"],
    },
    {
        "id": "v2_nl_unvague_pillow",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: Unvague Pillow Grayscale Pipeline (Full Lattice)",
        "prompt": _V2_NL_PILLOW_SPEC_PROMPT,
        "input_assets": ["input.png"],
        "egress_paths": ["pil_gray.png"],
        "required_imports": ["PIL"],
        "forbidden_imports": ["cv2"],
        "expected_literals": ["input.png", "pil_gray.png"],
    },
    {
        "id": "v2_nl_unvague_pygame",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: Unvague Pygame Display Lifecycle (640x480)",
        "prompt": (
            "initialize pygame, create a display window of size (640, 480), "
            "flip the display surface, and quit pygame"
        ),
        "required_imports": ["pygame"],
        "expected_literals": [640, 480],
    },
    {
        "id": "v2_nl_unvague_pandas_etl",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "synthesize",
        "name": "v2 NL: Unvague Pandas CSV Clean & Export",
        "prompt": 'load csv file "input.csv", drop missing rows, and save the dataframe to "clean_out.csv"',
        "input_assets": ["input.csv"],
        "egress_paths": ["clean_out.csv"],
        "required_imports": ["pandas"],
        "expected_literals": ["input.csv", "clean_out.csv"],
    },
    {
        "id": "v2_nl_vague_game",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "clarify",
        "name": "v2 Refusal Control: Underspecified 2D Platformer Game",
        "prompt": "make a 2d platformer game",
        "expected_literals": [],
    },
    {
        "id": "v2_nl_vague_ml",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "clarify",
        "name": "v2 Refusal Control: Underspecified ML Dataset & Target",
        "prompt": "load dataset and train a machine learning model to predict target",
        "expected_literals": [],
    },
    {
        "id": "v2_nl_audio_missing_file",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "clarify",
        "name": "v2 Refusal Control: Audio Pipeline Missing External Assets",
        "prompt": "load audio file, apply high-pass filter, compute spectrogram, and export filtered audio",
        "expected_literals": [],
    },
    {
        "id": "v2_nl_bound_ambiguous",
        "group": "v2_nl_core",
        "kind": "sandbox",
        "domain_set": "default",
        "expected": "clarify",
        "name": "v2 Refusal Control: Underspecified Anomaly Microservice",
        "prompt": "build an end-to-end anomaly detection microservice with alerts and automatic retraining",
        "expected_literals": [],
    },
]

V2_UNSEEN_TASKS: List[Dict[str, Any]] = [
    {
        "id": "v2_blind_cryopulse_nl",
        "group": "v2_unseen",
        "kind": "sdk",
        "sdk": "cryopulse",
        "domain_set": "cryopulse_only",
        "strict_sdk_only": True,
        "expected": "synthesize",
        "name": "v2 Unseen: Full-Lifecycle CryoPulse Calibration + Vent",
        "prompt": _V2_CRYO_NL_FULL_PROMPT,
        "required_calls": [
            "open_cryo_vessel",
            "energize_flux_coil",
            "sample_squid_flux",
            "demodulate_quadrature",
            "quench_tls_defects",
            "compile_pi_schedule",
            "emit_cryo_manifest",
            "vent_cryo_vessel",
        ],
        "expected_literals": ["DR-BLUE-04", 12.5, "RAIL-X2", 38.0, 512, 2048, 6.15, 0.18, 0.998, "calibration.cym"],
    },
    {
        "id": "v2_blind_aetheris_nl",
        "group": "v2_unseen",
        "kind": "sdk",
        "sdk": "aetheris",
        "domain_set": "aetheris_only",
        "strict_sdk_only": True,
        "expected": "synthesize",
        "name": "v2 Unseen: Full-Lifecycle Aetheris Interferometry + Park",
        "prompt": _V2_AETH_NL_FULL_PROMPT,
        "required_calls": [
            "mount_optical_table",
            "lock_reference_laser",
            "arm_phase_modulator",
            "sweep_fringe_interferogram",
            "unwrap_phase_residues",
            "solve_zernike_wavefront",
            "export_phase_lut",
            "park_optical_table",
        ],
        "expected_literals": ["OPT-NORTH-2", 1550.0, "TEM00", 18.5, 16, 3.6, 65.0, 1024, "goldstein", 8, "zernike_n2.ael"],
    },
]

V2_HAYSTACK_TASKS: List[Dict[str, Any]] = [
    {
        "id": "v2_haystack_cryopulse_1049_noise",
        "group": "v2_haystack",
        "kind": "sdk",
        "sdk": "cryopulse",
        "domain_set": "haystack_cryopulse",
        "strict_sdk_only": True,
        "expected": "synthesize",
        "name": "v2 Haystack: CryoPulse (11 cells) Embedded in 1,049 Distractor Cells",
        "prompt": _V2_CRYO_NL_FULL_PROMPT,
        "required_calls": list(V2_UNSEEN_TASKS[0]["required_calls"]),
        "expected_literals": list(V2_UNSEEN_TASKS[0]["expected_literals"]),
    },
    {
        "id": "v2_haystack_aetheris_1049_noise",
        "group": "v2_haystack",
        "kind": "sdk",
        "sdk": "aetheris",
        "domain_set": "haystack_aetheris",
        "strict_sdk_only": True,
        "expected": "synthesize",
        "name": "v2 Haystack: Aetheris (15 cells) Embedded in 1,049 Distractor Cells",
        "prompt": _V2_AETH_NL_FULL_PROMPT,
        "required_calls": list(V2_UNSEEN_TASKS[1]["required_calls"]),
        "expected_literals": list(V2_UNSEEN_TASKS[1]["expected_literals"]),
    },
]

V2_WRONG_DOMAIN_TASKS: List[Dict[str, Any]] = [
    {
        "id": "v2_wrong_domain_image_in_sqlite",
        "group": "v2_wrong_domain",
        "kind": "refusal",
        "domain_set": "sqlite_only",
        "expected": "refuse",
        "name": "v2 Wrong-Domain: Image Edge Task on sqlite3-Only Environment",
        "prompt": _V2_NL_CV2_PROMPT,
        "available_libraries_note": (
            "Environment restriction: ONLY python_core and sqlite3 are loaded/available. "
            "Image processing libraries (cv2, PIL, etc.) are NOT available. "
            "If the request requires unavailable libraries, refuse cleanly without writing code."
        ),
        "expected_literals": [],
    },
    {
        "id": "v2_wrong_domain_cryo_not_loaded",
        "group": "v2_wrong_domain",
        "kind": "refusal",
        "domain_set": "default",
        "expected": "refuse",
        "name": "v2 Wrong-Domain: CryoPulse Task When cryopulse Is Unloaded",
        "prompt": _V2_CRYO_NL_FULL_PROMPT,
        "available_libraries_note": (
            "Environment restriction: ONLY standard Python data/vision/audio/web libraries "
            "(audio, control_flow, cryptography, cv2, matplotlib, numpy, pandas, pathlib, "
            "pillow, pygame, python_core, requests, seaborn, sklearn, sqlite3) are available. "
            "'cryopulse' is NOT installed or available. If the request requires an unavailable "
            "library, refuse cleanly without writing code."
        ),
        "expected_literals": [],
    },
]

V2_DOMAIN_SPEC_TASKS: List[Dict[str, Any]] = [
    {
        "id": "v2_spec_pillow_only_cv2_nl",
        "group": "v2_domain_spec",
        "kind": "refusal",
        "domain_set": "cv2_only",
        "expected": "refuse",
        "name": "v2 Domain-Spec: Pillow Requested When Only OpenCV (cv2) Is Loaded",
        "prompt": _V2_NL_PILLOW_SPEC_PROMPT,
        "available_libraries_note": (
            "Environment restriction: ONLY python_core, numpy, and cv2 (OpenCV) are loaded/available. "
            "Pillow (PIL) is NOT installed. Do NOT substitute OpenCV when the user explicitly requested "
            "Pillow; refuse cleanly without writing code."
        ),
        "expected_literals": [],
    },
    {
        "id": "v2_spec_pillow_both_loaded_nl",
        "group": "v2_domain_spec",
        "kind": "sandbox",
        "domain_set": "cv2_and_pillow",
        "expected": "synthesize",
        "name": "v2 Domain-Spec: Strict Pillow Adherence When Both cv2 & Pillow Loaded",
        "prompt": _V2_NL_PILLOW_SPEC_PROMPT,
        "input_assets": ["input.png"],
        "egress_paths": ["pil_gray.png"],
        "required_imports": ["PIL"],
        "forbidden_imports": ["cv2"],
        "available_libraries_note": (
            "Available libraries: python_core, numpy, cv2 (OpenCV), and pillow (PIL). "
            "Strictly honor the library requested by the user."
        ),
        "expected_literals": ["input.png", "pil_gray.png"],
    },
    {
        "id": "v2_spec_cv2_both_loaded_nl",
        "group": "v2_domain_spec",
        "kind": "sandbox",
        "domain_set": "cv2_and_pillow",
        "expected": "synthesize",
        "name": "v2 Domain-Spec: Strict OpenCV Adherence When Both cv2 & Pillow Loaded",
        "prompt": _V2_NL_CV2_SPEC_PROMPT,
        "input_assets": ["input.png"],
        "egress_paths": ["cv2_gray.png"],
        "required_imports": ["cv2"],
        "forbidden_imports": ["PIL"],
        "available_libraries_note": (
            "Available libraries: python_core, numpy, cv2 (OpenCV), and pillow (PIL). "
            "Strictly honor the library requested by the user."
        ),
        "expected_literals": ["input.png", "cv2_gray.png"],
    },
]

V2_EXPANDED_TASKS: List[Dict[str, Any]] = (
    V2_NL_CORE_TASKS
    + V2_UNSEEN_TASKS
    + V2_HAYSTACK_TASKS
    + V2_WRONG_DOMAIN_TASKS
    + V2_DOMAIN_SPEC_TASKS
)

V2_FULL_TASKS: List[Dict[str, Any]] = (
    V2_EXPANDED_TASKS
    + [t for t in CRYOPULSE_TASKS if t["id"] != "cryo_explicit"]
    + [t for t in AETHERIS_TASKS if t["id"] != "aetheris_explicit"]
    + USER_REALISTIC_TASKS
    + [t for t in HONEST_BOUNDARY_TASKS if t["id"] == "bound_custom_algo"]
)

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
    "v2_nl_core": V2_NL_CORE_TASKS,
    "v2_unseen": V2_UNSEEN_TASKS,
    "v2_haystack": V2_HAYSTACK_TASKS,
    "v2_wrong_domain": V2_WRONG_DOMAIN_TASKS,
    "v2_domain_spec": V2_DOMAIN_SPEC_TASKS,
    "v2_expanded": V2_EXPANDED_TASKS,
    "v2_full": V2_FULL_TASKS,
}


def get_suite_tasks(suite_name: str = "core9") -> List[Dict[str, Any]]:
    if suite_name not in SUITE_GROUPS:
        raise ValueError(f"Unknown suite '{suite_name}'. Valid choices: {sorted(SUITE_GROUPS.keys())}")
    return list(SUITE_GROUPS[suite_name])
