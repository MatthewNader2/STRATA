"""Deterministic tests for benchmarks v2 suites, SKILL.md builders, evaluators, and AI Studio tooling."""
from __future__ import annotations

import json
import subprocess
import sys

from benchmarks.evaluators import evaluate_task_response
from benchmarks.skill_docs import build_messages_for_arm
from benchmarks.suites import get_suite_tasks
from tests.conftest import ROOT_DIR


def test_benchmark_suite_counts_and_backward_compatibility() -> None:
    assert len(get_suite_tasks("v2_expanded")) == 19
    assert len(get_suite_tasks("v2_full")) == 27
    assert len(get_suite_tasks("core8")) == 8
    assert len(get_suite_tasks("core9")) == 11
    assert len(get_suite_tasks("stress")) == 8
    assert len(get_suite_tasks("all")) == 19


def test_context_mode_arm_isolation() -> None:
    by_id = {t["id"]: t for t in get_suite_tasks("v2_expanded")}

    cryo_task = by_id["v2_blind_cryopulse_nl"]
    m_pure, meta_pure = build_messages_for_arm(cryo_task, context_mode="pure")
    assert "open_cryo_vessel" not in m_pure[0]["content"]
    assert meta_pure["ops_documented"] == 0

    m_skill, meta_skill = build_messages_for_arm(cryo_task, context_mode="skill_md")
    assert "open_cryo_vessel" in m_skill[0]["content"]
    assert "CV2_CANNY" not in m_skill[0]["content"]
    assert meta_skill["domain_set"] == "cryopulse_only"

    m_hay, meta_hay = build_messages_for_arm(by_id["v2_haystack_cryopulse_1049_noise"], context_mode="skill_md_haystack")
    assert meta_hay["haystack"] is True
    assert "open_cryo_vessel" in m_hay[0]["content"]
    assert "## Domain: `cv2`" in m_hay[0]["content"] and "## Domain: `sqlite3`" in m_hay[0]["content"]

    m_cv2_only, _ = build_messages_for_arm(by_id["v2_spec_pillow_only_cv2_nl"], context_mode="skill_md")
    assert "Loaded domains ONLY: python_core, numpy, cv2" in m_cv2_only[0]["content"]
    assert "## Domain: `pillow`" not in m_cv2_only[0]["content"]

    m_sql_only, _ = build_messages_for_arm(by_id["v2_wrong_domain_image_in_sqlite"], context_mode="skill_md")
    assert "Loaded domains ONLY: python_core, sqlite3" in m_sql_only[0]["content"]
    assert "## Domain: `cv2`" not in m_sql_only[0]["content"]


def test_benchmark_evaluators_domain_spec_refusal_and_haystack_noise() -> None:
    by_id = {t["id"]: t for t in get_suite_tasks("v2_expanded")}
    pil_both = by_id["v2_spec_pillow_both_loaded_nl"]
    pil_good_code = (
        'from PIL import Image\n'
        'img = Image.open("input.png").convert("L")\n'
        'img.save("pil_gray.png")\n'
    )
    ev_ok = evaluate_task_response(f"```python\n{pil_good_code}```", pil_good_code, pil_both, timeout=30.0)
    assert ev_ok["passed"] is True and ev_ok["status"] == "PASSED"

    cv2_subst = 'import cv2\ng = cv2.imread("input.png", 0)\ncv2.imwrite("pil_gray.png", g)\n'
    ev_bad = evaluate_task_response(f"```python\n{cv2_subst}```", cv2_subst, pil_both, timeout=30.0)
    assert ev_bad["acceptable"] is False and ev_bad["status"] == "DOMAIN_VIOLATION"

    pil_missing = by_id["v2_spec_pillow_only_cv2_nl"]
    ev_ref = evaluate_task_response("Pillow (PIL) is unavailable; refusing cleanly.", "", pil_missing)
    assert ev_ref["acceptable"] is True and ev_ref["status"] == "REFUSED_OK"

    cryo_good = (
        "import cryopulse\n"
        "v = cryopulse.open_cryo_vessel('DR-BLUE-04', target_mk=12.5)\n"
        "c = cryopulse.energize_flux_coil(v, rail_id='RAIL-X2', bias_ua=38.0)\n"
        "tr = cryopulse.sample_squid_flux(c, window_ns=512, shots=2048)\n"
        "iq = cryopulse.demodulate_quadrature(tr, lo_ghz=6.15)\n"
        "iq2 = cryopulse.quench_tls_defects(iq, damping_gamma=0.18)\n"
        "sched = cryopulse.compile_pi_schedule(iq2, target_fidelity=0.998)\n"
        "cryopulse.emit_cryo_manifest(sched, dest_path='calibration.cym')\n"
        "cryopulse.vent_cryo_vessel(v)\n"
    )
    cryo_hay = by_id["v2_haystack_cryopulse_1049_noise"]
    ev_cryo = evaluate_task_response(f"```python\n{cryo_good}```", cryo_good, cryo_hay)
    assert ev_cryo["passed"] is True and ev_cryo["status"] == "PASSED"

    cryo_noisy = "import cv2\n" + cryo_good
    ev_noisy = evaluate_task_response(f"```python\n{cryo_noisy}```", cryo_noisy, cryo_hay)
    assert ev_noisy["acceptable"] is False and ev_noisy["status"] == "SPEC_FAILED"
    assert ev_noisy["audit"]["noise_imports_used"] == ["cv2"]


def test_aistudio_pack_ingest_and_visualize_roundtrip(tmp_path) -> None:
    py_exe = sys.executable
    packs_dir = tmp_path / "packs"
    subprocess.run(
        [
            py_exe,
            "benchmarks/ingest_aistudio.py",
            "--export-prompts",
            str(packs_dir),
            "--suite",
            "v2_expanded",
            "--context-mode",
            "skill_md",
        ],
        cwd=str(ROOT_DIR),
        check=True,
    )
    skill_pack_dir = packs_dir / "skill_md"
    assert (skill_pack_dir / "ALL_PROMPTS_INDEX.md").is_file()
    assert len(list(skill_pack_dir.glob("[0-9][0-9]_*.md"))) == 19

    pil_code = 'from PIL import Image\nimg = Image.open("input.png").convert("L")\nimg.save("pil_gray.png")\n'
    cryo_code = (
        "import cryopulse\n"
        "v = cryopulse.open_cryo_vessel('DR-BLUE-04', target_mk=12.5)\n"
        "c = cryopulse.energize_flux_coil(v, rail_id='RAIL-X2', bias_ua=38.0)\n"
        "tr = cryopulse.sample_squid_flux(c, window_ns=512, shots=2048)\n"
        "iq = cryopulse.demodulate_quadrature(tr, lo_ghz=6.15)\n"
        "iq2 = cryopulse.quench_tls_defects(iq, damping_gamma=0.18)\n"
        "sched = cryopulse.compile_pi_schedule(iq2, target_fidelity=0.998)\n"
        "cryopulse.emit_cryo_manifest(sched, dest_path='calibration.cym')\n"
        "cryopulse.vent_cryo_vessel(v)\n"
    )
    dump_path = tmp_path / "mock_gemma4.txt"
    dump_path.write_text(
        f"=== TASK: v2_spec_pillow_both_loaded_nl ===\n```python\n{pil_code}```\n=== END ===\n\n"
        "=== TASK: v2_spec_pillow_only_cv2_nl ===\nPillow is unavailable; cannot comply cleanly.\n=== END ===\n\n"
        f"=== TASK: v2_haystack_cryopulse_1049_noise ===\n```python\n{cryo_code}```\n=== END ===\n",
        encoding="utf-8",
    )

    res_json = tmp_path / "results" / "results_gemma4_mock.json"
    subprocess.run(
        [
            py_exe,
            "benchmarks/ingest_aistudio.py",
            "--input",
            str(dump_path),
            "--suite",
            "v2_expanded",
            "--context-mode",
            "skill_md",
            "--out",
            str(res_json),
        ],
        cwd=str(ROOT_DIR),
        check=True,
    )
    parsed = json.loads(res_json.read_text(encoding="utf-8"))
    assert parsed["arm"] == "ai_studio_skill_md"
    assert len(parsed["records"]) == 3
    assert all(r["acceptable"] is True for r in parsed["records"])

    local_json = tmp_path / "results" / "results_strata_mock.json"
    local_json.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": "strata_v2_O",
                "model_id": "strata_v2_o_qwen3.5-0.8b",
                "model_display": "STRATA v2 Profile O (qwen3.5-0.8b)",
                "params_b": 0.8,
                "records": [
                    {
                        "task_id": "v2_spec_pillow_both_loaded_nl",
                        "group": "v2_domain_spec",
                        "status": "PASSED",
                        "passed": True,
                        "acceptable": True,
                        "wall_ms": 1150.0,
                        "total_tokens": 320,
                        "mem_mb": {"hwm_mb": 1400.0},
                        "gpu_mb": {"total_used_mb": 2100.0},
                    },
                    {
                        "task_id": "v2_haystack_cryopulse_1049_noise",
                        "group": "v2_haystack",
                        "status": "PASSED",
                        "passed": True,
                        "acceptable": True,
                        "wall_ms": 2300.0,
                        "total_tokens": 540,
                        "mem_mb": {"hwm_mb": 1450.0},
                        "gpu_mb": {"total_used_mb": 2100.0},
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    viz_dir = tmp_path / "viz"
    subprocess.run(
        [py_exe, "benchmarks/visualize_results.py", str(res_json), str(local_json), "--out-dir", str(viz_dir)],
        cwd=str(ROOT_DIR),
        check=True,
    )
    assert "Per-Category Acceptable Rate Breakdown" in (viz_dir / "comparison_summary.md").read_text(encoding="utf-8")
    assert (viz_dir / "comparison_summary_by_group.csv").is_file()
    for fig_idx in range(1, 5):
        assert any(viz_dir.glob(f"fig{fig_idx}_*.png"))
