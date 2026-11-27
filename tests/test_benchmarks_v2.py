"""Deterministic tests for benchmarks v2 suites, SKILL.md builders, evaluators, and AI Studio tooling."""
from __future__ import annotations

import json
import subprocess
import sys

from benchmarks.colab_eval import (
    _build_cell_naming,
    _dedupe_model_sweep_cells,
    build_chat_completion_payload,
)
from benchmarks.evaluators import (
    build_react_feedback,
    evaluate_task_response,
    extract_code_clean,
    strip_reasoning_traces,
)
from benchmarks.external_bfcl import bfcl_functions_to_strata_tree, evaluate_bfcl_response
from benchmarks.gemini_api_eval import (
    _is_daily_quota_error,
    _sanitize_error_text,
    build_config_grid,
    build_gemini_payload,
    parse_gemini_response,
)
from benchmarks.openrouter_eval import (
    GATEWAY_ENDPOINTS,
    OPENROUTER_MODEL_REGISTRY,
    _build_cell_metadata,
    build_openrouter_payload,
    is_openrouter_daily_quota_error,
    parse_openrouter_response,
    sanitize_openrouter_error,
)
from benchmarks.skill_docs import build_messages_for_arm
from benchmarks.suites import get_suite_tasks
from benchmarks.visualize_results import aggregate_by_model, wilson_ci_95, write_tables
from schema import TreeSchema
from strata_pipeline import StrataPipeline
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


def test_wilson_ci_and_multiseed_aggregation(tmp_path) -> None:
    assert wilson_ci_95(0, 0) == (0.0, 0.0)
    lo0, hi0 = wilson_ci_95(0, 10)
    assert lo0 == 0.0 and 20.0 < hi0 < 35.0
    lo10, hi10 = wilson_ci_95(10, 10)
    assert 65.0 < lo10 < 80.0 and hi10 == 100.0
    lo19, hi19 = wilson_ci_95(19, 19)
    assert 80.0 < lo19 < 86.0 and hi19 == 100.0
    lo5, hi5 = wilson_ci_95(5, 10)
    assert round(lo5 + hi5, 1) == 100.0

    rows = [
        {"display": "DemoModel", "params_b": 1.0, "task_id": "t1", "seed": 1, "group": "g", "passed": True, "acceptable": True, "wall_ms": 100.0, "total_tokens": 50, "hwm_mb": 100.0, "vram_mb": 0.0},
        {"display": "DemoModel", "params_b": 1.0, "task_id": "t2", "seed": 1, "group": "g", "passed": False, "acceptable": True, "wall_ms": 120.0, "total_tokens": 60, "hwm_mb": 110.0, "vram_mb": 0.0},
        {"display": "DemoModel", "params_b": 1.0, "task_id": "t1", "seed": 2, "group": "g", "passed": True, "acceptable": True, "wall_ms": 110.0, "total_tokens": 55, "hwm_mb": 105.0, "vram_mb": 0.0},
        {"display": "DemoModel", "params_b": 1.0, "task_id": "t2", "seed": 2, "group": "g", "passed": True, "acceptable": True, "wall_ms": 130.0, "total_tokens": 65, "hwm_mb": 115.0, "vram_mb": 0.0},
    ]
    summary = aggregate_by_model(rows)
    stat = summary["DemoModel"]
    assert stat["seeds"] == 2
    assert stat["tasks"] == 2
    assert stat["trials"] == 4
    assert stat["passed"] == 3 and stat["acceptable"] == 4
    assert stat["seed_std_pct"] == 0.0
    assert stat["strict_ci_lo"] < stat["strict_pct"] <= stat["strict_ci_hi"]

    md_p, csv_p = write_tables(summary, tmp_path / "tables")
    csv_hdr = csv_p.read_text(encoding="utf-8").splitlines()[0]
    assert "strict_ci_lo" in csv_hdr and "acc_ci_hi" in csv_hdr and "seeds" in csv_hdr
    assert "trials" in csv_hdr and "seed_std_pct" in csv_hdr
    assert "95% CI" in md_p.read_text(encoding="utf-8")


def test_react_feedback_zero_oracle_leakage() -> None:
    synth_task = {"id": "t_sb", "kind": "sandbox", "expected": "synthesize"}
    refuse_task = {"id": "t_ref", "kind": "refusal", "expected": "refuse"}
    clarify_task = {"id": "t_clar", "kind": "sandbox", "expected": "clarify"}
    sdk_task = {"id": "t_sdk", "kind": "sdk", "expected": "synthesize"}

    # Genuine sandbox failure / missing code / security error trigger feedback without leaking literals
    fb_err = build_react_feedback(
        "print(1/0)",
        {"status": "SANDBOX_FAILED", "stderr": "ZeroDivisionError: division by zero"},
        synth_task,
    )
    assert fb_err is not None and "ZeroDivisionError" in fb_err
    assert build_react_feedback("", {"status": "NO_CODE"}, synth_task) is not None
    assert build_react_feedback("import os", {"status": "SECURITY_BLOCKED", "detail": "Blocked os"}, synth_task) is not None

    # Oracle checks, refusal tasks, and clarify tasks must NOT receive hints
    assert build_react_feedback("print(1)", {"status": "PASSED_UNVERIFIED", "detail": "Missing literals: ['foo']"}, synth_task) is None
    assert build_react_feedback("import cv2", {"status": "DOMAIN_VIOLATION", "detail": "Forbidden PIL"}, synth_task) is None
    assert build_react_feedback("pass", {"status": "SEMANTIC_FAILED", "detail": "Wrong SQLite rows"}, synth_task) is None
    assert build_react_feedback("", {"status": "REFUSED_OK"}, refuse_task) is None
    assert build_react_feedback("", {"status": "NO_CODE"}, refuse_task) is None
    assert build_react_feedback("", {"status": "NO_CODE"}, clarify_task) is None

    # SDK tasks only surface syntax/undefined-name compiler errors, never required/forbidden call oracle lists
    assert build_react_feedback(
        "x = (",
        {"status": "SPEC_FAILED", "audit": {"ast_ok": False, "pyflakes_issues": ["SyntaxError: '(' was never closed"]}},
        sdk_task,
    ) is not None
    assert build_react_feedback(
        "import cryopulse\ncryopulse.foo()",
        {
            "status": "SPEC_FAILED",
            "audit": {
                "ast_ok": True,
                "pyflakes_issues": [],
                "missing_required_calls": ["open_cryo_vessel"],
            },
        },
        sdk_task,
    ) is None


def test_bfcl_tree_compiler_profile0_and_evaluator() -> None:
    functions = [
        {
            "name": "triangle_properties.get",
            "description": "Retrieve triangle area and perimeter.",
            "parameters": {
                "type": "dict",
                "required": ["side1", "side2", "side3"],
                "properties": {
                    "side1": {"type": "integer", "description": "First side"},
                    "side2": {"type": "integer", "description": "Second side"},
                    "side3": {"type": "integer", "description": "Third side"},
                    "result": {"type": "boolean", "description": "Return summary dict"},
                },
            },
        }
    ]
    bfcl_tree = bfcl_functions_to_strata_tree(functions)
    parsed = TreeSchema(**bfcl_tree)
    assert len(parsed.cells) == 1
    cell = parsed.cells[0]
    assert "out_val" in cell.outputs  # avoided collision with input param 'result'
    assert "{out_val} = triangle_properties.get" in cell.code_template

    py_core = json.loads((ROOT_DIR / "trees" / "python_core.json").read_text(encoding="utf-8"))
    py_core_base = {**py_core, "cells": []}
    pipe = StrataPipeline.from_trees([py_core_base, bfcl_tree])
    formal = (
        'step1 = bfcl_tools::triangle_properties.get(side1=5, side2=4, side3=3, result=True)'
    )
    res = pipe.run(formal, execute=False)
    assert res.ok, res.error

    task = {
        "id": "multiple_demo",
        "kind": "bfcl_call",
        "expected": "synthesize",
        "functions": functions,
        "ground_truth": [
            {
                "triangle_properties.get": {
                    "side1": [5],
                    "side2": [4.0],
                    "side3": [3],
                    "result": ["", True],
                }
            }
        ],
    }
    ev_pipe = evaluate_bfcl_response("", res.code, task)
    assert ev_pipe["passed"] is True and ev_pipe["status"] == "PASSED"

    # Optional omit / explicit None match "" in accepted list
    code_omit = "out_val = triangle_properties.get(5, 4, 3)"
    assert evaluate_bfcl_response("", code_omit, task)["passed"] is True

    # Wrong numeric argument fails strictly
    code_bad = "out_val = triangle_properties.get(side1=9, side2=4, side3=3)"
    assert evaluate_bfcl_response("", code_bad, task)["passed"] is False

    # Wrong / unknown function name fails
    code_wrong_fn = "out_val = other_fn(side1=5, side2=4, side3=3)"
    assert evaluate_bfcl_response("", code_wrong_fn, task)["passed"] is False

    # Irrelevance task refusal vs emission check
    irrel_task = {"id": "irrel_demo", "kind": "bfcl_refusal", "expected": "refuse", "functions": functions, "ground_truth": []}
    assert evaluate_bfcl_response("No matching function is available for BMI calculation.", "", irrel_task)["acceptable"] is True
    ev_viol = evaluate_bfcl_response("```python\ntriangle_properties.get(5, 4, 3)\n```", "triangle_properties.get(5, 4, 3)", irrel_task)
    assert ev_viol["acceptable"] is False and ev_viol["status"] == "IRRELEVANCE_VIOLATION"


def test_gemini_api_payload_and_response_parser() -> None:
    msgs = [
        {"role": "system", "content": "Follow rules strictly."},
        {"role": "user", "content": "Write add(a, b)."},
    ]
    p_hi = build_gemini_payload(msgs, temperature=0.0, thinking=True, max_tokens=4096)
    assert p_hi["systemInstruction"] == {"parts": [{"text": "Follow rules strictly."}]}
    assert p_hi["contents"] == [{"role": "user", "parts": [{"text": "Write add(a, b)."}]}]
    assert p_hi["generationConfig"]["temperature"] == 0.0
    assert p_hi["generationConfig"]["topP"] == 0.95
    assert p_hi["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert p_hi["generationConfig"]["maxOutputTokens"] == 4096

    p_lo_unlim = build_gemini_payload(msgs, temperature=1.0, thinking=False, max_tokens=None)
    assert p_lo_unlim["generationConfig"]["temperature"] == 1.0
    assert p_lo_unlim["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    assert "maxOutputTokens" not in p_lo_unlim["generationConfig"]

    parsed = parse_gemini_response(
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Step 1: check constraints."},
                            {"text": "```python\nprint(1 + 2)\n```"},
                        ]
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "thoughtsTokenCount": 14,
                "candidatesTokenCount": 9,
                "totalTokenCount": 43,
            },
        }
    )
    assert parsed["thought_text"] == "Step 1: check constraints."
    assert parsed["answer_text"] == "```python\nprint(1 + 2)\n```"
    assert parsed["raw_reply"].startswith("<think>Step 1: check constraints.</think>\n```python")
    assert parsed["thoughts_tokens"] == 14
    assert parsed["candidates_tokens"] == 9
    assert parsed["completion_tokens"] == 23
    assert parsed["total_tokens"] == 43
    assert parsed["finish_reason"] == "STOP"

    assert _sanitize_error_text("Header secret-123 failed", "secret-123") == "Header *** failed"
    assert _is_daily_quota_error(429, '{"error": {"details": [{"quotaId": "GenerateRequestsPerDay"}]}}') is True
    assert _is_daily_quota_error(429, '{"error": {"message": "Quota exceeded for RequestsPerMinute"}}') is False
    grid = build_config_grid(temperatures=[0.0, 1.0], thinking_modes=["on", "off"], token_limits=[4096, None])
    assert len(grid) == 8 and grid[0] == (0.0, True, 4096) and grid[-1] == (1.0, False, None)


def test_colab_eval_decoding_payload_and_sweep_naming() -> None:
    msgs = [{"role": "user", "content": "hello"}]

    p_qwen_off = build_chat_completion_payload(
        msgs, max_tokens=4096, temperature=1.0, disable_thinking=True, thinking_mode="qwen_toggle"
    )
    assert p_qwen_off["max_tokens"] == 4096
    assert p_qwen_off["temperature"] == 1.0
    assert p_qwen_off["chat_template_kwargs"] == {"enable_thinking": False}

    p_unlim = build_chat_completion_payload(
        msgs, max_tokens=None, temperature=0.0, disable_thinking=False, thinking_mode="qwen_toggle"
    )
    assert "max_tokens" not in p_unlim
    assert "chat_template_kwargs" not in p_unlim

    p_gpt_hi = build_chat_completion_payload(
        msgs, max_tokens=None, temperature=0.0, disable_thinking=False, thinking_mode="harmony_effort"
    )
    p_gpt_lo = build_chat_completion_payload(
        msgs, max_tokens=4096, temperature=1.0, disable_thinking=True, thinking_mode="harmony_effort"
    )
    assert p_gpt_hi["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert p_gpt_lo["chat_template_kwargs"] == {"reasoning_effort": "low"}

    p_inst = build_chat_completion_payload(
        msgs, max_tokens=4096, temperature=0.0, disable_thinking=True, thinking_mode="none"
    )
    assert "chat_template_kwargs" not in p_inst

    assert len(_dedupe_model_sweep_cells(thinking_mode="none", temperatures=[0.0, 1.0], thinking_modes=["on", "off"], token_limits=[4096, None])) == 4
    assert len(_dedupe_model_sweep_cells(thinking_mode="qwen_toggle", temperatures=[0.0, 1.0], thinking_modes=["on", "off"], token_limits=[4096, None])) == 8

    spec_q = {"display": "Qwen3.5-35B-A3B MoE", "thinking_mode": "qwen_toggle"}
    legacy = _build_cell_naming(
        model_key="qwen3.5-35b-a3b",
        spec=spec_q,
        context_mode="skill_md",
        react_turns=1,
        temperature=0.0,
        disable_thinking=False,
        max_tokens=4096,
        use_detailed_tags=False,
    )
    assert legacy["filename"] == "results_qwen3.5-35b-a3b_skill_md.json"

    detailed = _build_cell_naming(
        model_key="qwen3.5-35b-a3b",
        spec=spec_q,
        context_mode="skill_md",
        react_turns=1,
        temperature=1.0,
        disable_thinking=True,
        max_tokens=None,
        use_detailed_tags=True,
    )
    assert detailed["filename"] == "results_qwen3.5-35b-a3b_skill_md_t1_nothink_unlim.json"
    assert "T=1 | NoThink | NoLimit" in detailed["display"]

    harmony_raw = (
        "<|channel|>analysis<|message|>plan<|end|>"
        "<|start|>assistant<|channel|>final <|constrain|>python code<|message|>```python\nx=1\n```"
    )
    assert strip_reasoning_traces(harmony_raw) == "```python\nx=1\n```"
    assert extract_code_clean(harmony_raw) == "x=1"


def test_openrouter_payload_parser_and_quota_classifier() -> None:
    msgs = [
        {"role": "system", "content": "Only use documented SDKs."},
        {"role": "user", "content": "Say ok"},
    ]
    p_on = build_openrouter_payload(
        msgs,
        model="nvidia/nemotron-3.5-super-vl:free",
        temperature=0.0,
        thinking=True,
        max_tokens=4096,
    )
    assert p_on["model"] == "nvidia/nemotron-3.5-super-vl:free"
    assert p_on["temperature"] == 0.0
    assert p_on["top_p"] == 0.95
    assert p_on["reasoning"] == {"enabled": True}
    assert p_on["max_tokens"] == 4096

    p_unlim = build_openrouter_payload(
        msgs,
        model="google/gemma-4-31b-it:free",
        temperature=1.0,
        thinking=False,
        max_tokens=None,
    )
    assert p_unlim["temperature"] == 1.0
    assert p_unlim["reasoning"] == {"enabled": False}
    assert "max_tokens" not in p_unlim

    parsed = parse_openrouter_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "```python\nprint(42)\n```",
                        "reasoning": "Check constant 42.",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 18,
                "total_tokens": 48,
                "completion_tokens_details": {"reasoning_tokens": 11},
            },
        }
    )
    assert parsed["raw_reply"] == "<think>Check constant 42.</think>\n```python\nprint(42)\n```"
    assert parsed["thoughts_tokens"] == 11
    assert parsed["completion_tokens"] == 18
    assert parsed["prompt_tokens"] == 30
    assert parsed["total_tokens"] == 48
    assert parsed["finish_reason"] == "stop"

    assert sanitize_openrouter_error("Bearer sk-or-v1-secret failed", "sk-or-v1-secret") == "Bearer *** failed"
    assert is_openrouter_daily_quota_error(429, "Rate limit exceeded: free-models-per-day") is True
    assert is_openrouter_daily_quota_error(402, "Insufficient credits") is True
    assert is_openrouter_daily_quota_error(429, "google/gemma-4-31b-it:free is temporarily rate-limited upstream") is False

    # Multi-gateway support (SandBase DeepSeek thinking + OrcaRouter/TeamoRouter quota/balance classifiers)
    sb_on = build_openrouter_payload(
        msgs,
        model="deepseek/deepseek-v4.1-flash:free",
        thinking=True,
        max_tokens=4096,
        thinking_style="deepseek_thinking",
    )
    sb_off = build_openrouter_payload(
        msgs,
        model="deepseek/deepseek-v4.1-flash:free",
        thinking=False,
        max_tokens=None,
        thinking_style="deepseek_thinking",
    )
    assert sb_on["thinking"] == {"type": "enabled"}
    assert "reasoning" not in sb_on
    assert sb_off["thinking"] == {"type": "disabled"}
    assert "max_tokens" not in sb_off

    hy_on = build_openrouter_payload(
        msgs,
        model="tencent/hy3-free",
        thinking=True,
        thinking_style="reasoning_effort",
    )
    hy_off = build_openrouter_payload(
        msgs,
        model="tencent/hy3-free",
        thinking=False,
        thinking_style="reasoning_effort",
    )
    assert hy_on["reasoning_effort"] == "high"
    assert hy_off["reasoning_effort"] == "none"

    assert GATEWAY_ENDPOINTS["sandbase"]["default_api_key_env"] == "SANDBASE_API_KEY"
    assert GATEWAY_ENDPOINTS["orcarouter"]["default_api_key_env"] == "ORCAROUTER_API_KEY"
    assert GATEWAY_ENDPOINTS["teamorouter"]["default_api_key_env"] == "TEAMOROUTER_API_KEY"
    sb_meta = _build_cell_metadata(
        model_key="deepseek-v4.1-flash-sb",
        spec=OPENROUTER_MODEL_REGISTRY["deepseek-v4.1-flash-sb"],
        context_mode="skill_md",
        temperature=0.0,
        thinking=False,
        max_tokens=None,
    )
    assert sb_meta["arm_id"].startswith("sb_")

    assert is_openrouter_daily_quota_error(429, '{"error": "daily free API limit exceeded"}') is True
    assert is_openrouter_daily_quota_error(429, '{"error": {"code": "free_rate_limited", "metadata": {"reason": "err_free_access_denied"}}}') is True
    assert is_openrouter_daily_quota_error(402, '{"error": {"code": "free_quota_exhausted"}}') is True
    assert is_openrouter_daily_quota_error(400, '{"error": {"code": "enterprise_insufficient_balance"}}') is True
    assert is_openrouter_daily_quota_error(429, '{"error": "organization rate limit exceeded"}') is False


def test_visualize_headline_and_ablation_layouts(tmp_path) -> None:
    import csv as _csv

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    def _make_run(fname: str, *, display: str, arm: str, params_b: float, active_b: float, vram: float, api_err: bool = False, partial: bool = False):
        recs = [
            {
                "task_id": "v2_nl_cv2_params",
                "group": "v2_nl_core",
                "expected": "synthesize",
                "status": "PASSED",
                "passed": True,
                "acceptable": True,
                "wall_ms": 2400.0,
                "total_tokens": 1100,
                "mem_mb": {"hwm_mb": 420.0},
                "gpu_mb": {"total_used_mb": vram},
            },
            {
                "task_id": "v2_blind_cryopulse_nl",
                "group": "v2_unseen",
                "expected": "synthesize",
                "status": "PASSED" if "STRATA" in display or "[SKILL" in display else "FAILED",
                "passed": "STRATA" in display or "[SKILL" in display,
                "acceptable": "STRATA" in display or "[SKILL" in display,
                "wall_ms": 3100.0,
                "total_tokens": 1800,
                "mem_mb": {"hwm_mb": 430.0},
                "gpu_mb": {"total_used_mb": vram},
            },
            {
                "task_id": "v2_nl_vague_game",
                "group": "v2_nl_core",
                "expected": "clarify",
                "status": "API_ERROR" if api_err else "REFUSED_OK",
                "passed": False,
                "acceptable": not api_err,
                "wall_ms": 900.0,
                "total_tokens": 400,
                "mem_mb": {"hwm_mb": 400.0},
                "gpu_mb": {"total_used_mb": vram},
            },
        ]
        if partial:
            recs = recs[:2]
        payload = {
            "schema_version": "strata_bench_v2",
            "arm": arm,
            "model_id": fname.removesuffix(".json"),
            "model_display": display,
            "params_b": params_b,
            "active_params_b": active_b,
            "records": recs,
        }
        (logs_dir / fname).write_text(json.dumps(payload), encoding="utf-8")

    _make_run("results_strata_d.json", display="STRATA v2 Profile D (demo)", arm="strata_v2_D", params_b=0.8, active_b=0.8, vram=1913.0)
    _make_run("results_alpha_skill.json", display="Alpha-30B-A3B [SKILL.md | T=1 | Think | Max4096]", arm="skill_md", params_b=30.5, active_b=3.3, vram=13600.0)
    _make_run("results_alpha_pure.json", display="Alpha-30B-A3B [Pure | T=0 | NoThink | NoLimit]", arm="pure", params_b=30.5, active_b=3.3, vram=13600.0)
    _make_run("results_beta_api.json", display="Beta-120B (API) [SKILL.md | T=0 | Think | Max4096]", arm="or_skill", params_b=120.0, active_b=12.0, vram=0.0)
    _make_run("results_errored_run.json", display="Broken-99B (API) [Pure | T=0]", arm="err", params_b=99.0, active_b=99.0, vram=0.0, api_err=True)
    _make_run("results_partial_run.json", display="Partial-70B (API) [Pure | T=0]", arm="part", params_b=70.0, active_b=70.0, vram=0.0, partial=True)

    py_exe = sys.executable or "python3"
    import re as _re

    def _assert_valid_gfm_tables(md_text: str, label: str) -> None:
        in_table = False
        expected_cols = 0
        for ln_no, raw_ln in enumerate(md_text.splitlines(), 1):
            s = raw_ln.strip()
            if s.startswith("|") and s.endswith("|"):
                cols = len(_re.findall(r"(?<!\\)\|", s)) - 1
                if not in_table:
                    in_table = True
                    expected_cols = cols
                else:
                    assert cols == expected_cols, f"{label}:{ln_no} expected {expected_cols} cols, got {cols}: {s[:90]}"
            else:
                in_table = False
                expected_cols = 0

    for layout_mode in ("headline", "ablation"):
        out_dir = tmp_path / f"viz_{layout_mode}"
        cmd = [
            py_exe,
            "benchmarks/visualize_results.py",
            str(logs_dir),
            "--clean-only",
            "--layout",
            layout_mode,
            "--out-dir",
            str(out_dir),
            "--title-prefix",
            f"Test {layout_mode}: ",
        ]
        if layout_mode == "headline":
            cmd.append("--dark-variant")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "UserWarning" not in proc.stderr
        assert "findfont:" not in proc.stderr
        md_txt = (out_dir / "comparison_summary.md").read_text(encoding="utf-8")
        assert "Headline Cross-Model Leaderboard" in md_txt
        assert "Per-Category Acceptable Rate Breakdown" in md_txt
        assert "Broken-99B" not in md_txt and "Partial-70B" not in md_txt
        _assert_valid_gfm_tables(md_txt, f"viz_{layout_mode}")

        with open(out_dir / "comparison_summary.csv", encoding="utf-8") as fh:
            csv_rows = list(_csv.DictReader(fh))
        assert {"syn_passed", "safety_acc", "safety_refused"}.issubset(csv_rows[0].keys())
        for fig_idx in range(1, 5):
            assert any(p for p in out_dir.glob(f"fig{fig_idx}_*.png") if not p.name.endswith("_dark.png"))
            if layout_mode == "headline":
                assert any(out_dir.glob(f"fig{fig_idx}_*_dark.png"))

    for doc_rel in (
        "README.md",
        "docs/benchmarks/comparison_summary.md",
        "docs/benchmarks/frontier_ablations/comparison_summary.md",
    ):
        doc_path = ROOT_DIR / doc_rel
        _assert_valid_gfm_tables(doc_path.read_text(encoding="utf-8"), doc_rel)
