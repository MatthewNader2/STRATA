#!/usr/bin/env python3
"""
benchmarks/run_local.py - Unified local runner for STRATA Profile O + OMEGA
and local GGUF Pure Baselines across Core and Tier-2 Stress suites.

Replaces ad-hoc scratch scripts with repo-relative fixtures and standardized v2 JSON output.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (str(ROOT_DIR / "src"), str(ROOT_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import test as test_mod  # noqa: E402
from benchmarks.evaluators import (  # noqa: E402
    evaluate_task_response,
    extract_code_clean,
    load_sdk_docs,
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
)
from benchmarks.suites import SDK_SPECS, get_suite_tasks  # noqa: E402


class LlamaTelemetry:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_wall_ms = 0.0
        self._installed = False

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_wall_ms = 0.0

    def install(self) -> None:
        if self._installed:
            return
        from llama_cpp import Llama

        hook = self
        orig_chat = Llama.create_chat_completion
        orig_call = Llama.__call__

        def _rec(resp: Any, dt_ms: float) -> None:
            hook.calls += 1
            hook.llm_wall_ms += dt_ms
            if isinstance(resp, dict):
                usage = resp.get("usage") or {}
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                hook.prompt_tokens += pt
                hook.completion_tokens += ct
                hook.total_tokens += int(usage.get("total_tokens") or (pt + ct))

        def wrapped_chat(self_l, *args, **kwargs):
            t0 = time.perf_counter()
            r = orig_chat(self_l, *args, **kwargs)
            _rec(r, (time.perf_counter() - t0) * 1000.0)
            return r

        def wrapped_call(self_l, *args, **kwargs):
            t0 = time.perf_counter()
            r = orig_call(self_l, *args, **kwargs)
            _rec(r, (time.perf_counter() - t0) * 1000.0)
            return r

        Llama.create_chat_completion = wrapped_chat
        Llama.__call__ = wrapped_call
        self._installed = True


def _compile_combined_trees(extra_sdks: List[str], work_trees_dir: Path) -> Path:
    if work_trees_dir.exists():
        shutil.rmtree(work_trees_dir)
    work_trees_dir.mkdir(parents=True, exist_ok=True)
    for jf in sorted((ROOT_DIR / "trees").glob("*.json")):
        shutil.copy2(jf, work_trees_dir / jf.name)
    for sdk in extra_sdks:
        tpath = Path(SDK_SPECS[sdk]["tree_path"])
        shutil.copy2(tpath, work_trees_dir / tpath.name)
    db_path = work_trees_dir / "lattice.db"
    from cli import ensure_lattice_compiled

    ensure_lattice_compiled(trees_dir=str(work_trees_dir), db_path=str(db_path))
    return db_path


def main() -> None:
    ap = argparse.ArgumentParser(description="STRATA Local Standardized Benchmark Runner")
    ap.add_argument("--arm", choices=["omega_08b", "pure_gguf"], required=True)
    ap.add_argument("--suite", default="stress", help="Suite name from benchmarks/suites.py")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--feedback", action="store_true", help="Enable LLM feedback repair for STRATA O+OMEGA")
    ap.add_argument("--out-dir", default="test_logs/local_bench")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_mod.configure_headless_env()

    hook = LlamaTelemetry()
    hook.install()

    tasks = get_suite_tasks(args.suite)
    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    records: List[Dict[str, Any]] = []

    if args.arm == "pure_gguf":
        pure_runner = test_mod.PureLLMRunner(ROOT_DIR)
        t_i0 = time.perf_counter()
        _, _, resolved_llm = pure_runner._get_llm(llm_name=args.llm)
        init_ms = round((time.perf_counter() - t_i0) * 1000.0, 2)
        gpu_after_init = read_gpu_vram_mb()

        from benchmarks.colab_eval import DEFAULT_PURE_SYS

        pure_log = out_dir / "pure_local_total.log"
        pure_log.write_text("", encoding="utf-8")

        for task in tasks:
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            if task["kind"] == "sdk":
                sys_p = (
                    "You are a Python coding assistant. Use ONLY the documented API below to write a single "
                    "Python script in ```python ... ```.\n\n" + load_sdk_docs(task["sdk"])
                )
            else:
                sys_p = DEFAULT_PURE_SYS
            raw_reply, _, _, resolved_llm = pure_runner.generate_chat_code(
                prompt=task["prompt"],
                llm_name=args.llm,
                system_prompt=sys_p,
                max_tokens=1536,
            )
            code = extract_code_clean(raw_reply)
            ev = evaluate_task_response(raw_reply, code, task, timeout=45.0)
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            records.append(
                {
                    "arm": "pure_local",
                    "model_id": resolved_llm,
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "expected": task.get("expected", "synthesize"),
                    "prompt": task["prompt"],
                    "status": ev["status"],
                    "passed": ev["passed"],
                    "acceptable": ev["acceptable"],
                    "detail": ev["detail"],
                    "wall_ms": wall_ms,
                    "llm_calls": hook.calls,
                    "prompt_tokens": hook.prompt_tokens,
                    "completion_tokens": hook.completion_tokens,
                    "total_tokens": hook.total_tokens,
                    "llm_wall_ms": round(hook.llm_wall_ms, 2),
                    "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                    "mem_mb": read_proc_mem_mb(),
                    "gpu_mb": read_gpu_vram_mb(),
                    "audit": ev.get("audit"),
                    "code": code,
                }
            )
            print(f"[{resolved_llm}] {task['id']:<18} -> {ev['status']} | acc={ev['acceptable']} | {wall_ms:.0f}ms")

        out_path = out_dir / f"results_pure_{resolved_llm}.json"
        payload = {
            "schema_version": "strata_bench_v2",
            "arm": "pure_local",
            "model_id": resolved_llm,
            "init": {
                "init_ms": init_ms,
                "mem_before": mem_before,
                "gpu_before": gpu_before,
                "gpu_after_init": gpu_after_init,
            },
            "records": records,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {out_path}")
        return

    # STRATA Profile O + OMEGA
    needed_sdks = sorted({t["sdk"] for t in tasks if t["kind"] == "sdk"})
    tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
    try:
        if needed_sdks:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="strata_trees_")
            trees_dir = Path(tmp_ctx.name) / "trees"
            db_path = _compile_combined_trees(needed_sdks, trees_dir)
            os.environ["NSTL_TREES_DIR"] = str(trees_dir)
            from config import settings

            settings.trees_dir = trees_dir
            runner = test_mod.CachedNSTLRunner(ROOT_DIR, db_path=db_path)
            runner.trees_dir = str(trees_dir)
        else:
            runner = test_mod.CachedNSTLRunner.get_instance(ROOT_DIR)

        t_i0 = time.perf_counter()
        shell = runner.get_or_create_shell(profile="O", llm=args.llm, llm_feedback=args.feedback)
        init_ms = round((time.perf_counter() - t_i0) * 1000.0, 2)
        gpu_after_init = read_gpu_vram_mb()
        resolved_llm = getattr(shell, "llm_name", "") or args.llm or "qwen3.5-0.8b-instruct"

        omega_log = out_dir / "omega_total.log"
        omega_log.write_text("", encoding="utf-8")

        for task in tasks:
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            no_exec = task["kind"] == "sdk"
            res = test_mod.run_single_test(
                profile="O",
                method="OMEGA",
                prompt=task["prompt"],
                output_dir=out_dir,
                total_log_path=omega_log,
                no_exec=no_exec,
                llm_feedback=args.feedback,
                llm=args.llm,
                pass_label=f"omega_{task['id']}",
                use_cache=True,
                runner=runner,
                verbose_plain=False,
                show_code=False,
            )
            code = res.get("code", "") or ""
            if no_exec:
                ev = evaluate_task_response("", code, task, timeout=45.0)
            else:
                st = res.get("status", "FAILED")
                exp = task.get("expected", "synthesize")
                ev = {
                    "kind": "sandbox",
                    "expected": exp,
                    "status": st,
                    "passed": st == "PASSED",
                    "acceptable": (st == "PASSED") or (exp == "clarify" and st == "LINT_HALTED"),
                    "detail": res.get("detail", ""),
                }
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            records.append(
                {
                    "arm": "omega",
                    "model_id": f"strata_omega_{resolved_llm}",
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "expected": task.get("expected", "synthesize"),
                    "prompt": task["prompt"],
                    "status": ev["status"],
                    "passed": ev["passed"],
                    "acceptable": ev["acceptable"],
                    "detail": ev["detail"],
                    "wall_ms": wall_ms,
                    "latencies": res.get("latencies") or {},
                    "llm_calls": hook.calls,
                    "prompt_tokens": hook.prompt_tokens,
                    "completion_tokens": hook.completion_tokens,
                    "total_tokens": hook.total_tokens,
                    "llm_wall_ms": round(hook.llm_wall_ms, 2),
                    "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                    "mem_mb": read_proc_mem_mb(),
                    "gpu_mb": read_gpu_vram_mb(),
                    "audit": ev.get("audit"),
                    "code": code,
                }
            )
            print(f"[OMEGA] {task['id']:<18} -> {ev['status']} | acc={ev['acceptable']} | {wall_ms:.0f}ms | calls={hook.calls}")

        out_path = out_dir / "results_strata_omega.json"
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "strata_bench_v2",
                    "arm": "omega",
                    "model_id": f"strata_omega_{resolved_llm}",
                    "model_display": f"STRATA O+OMEGA ({resolved_llm})",
                    "family": "STRATA-OMEGA",
                    "params_b": 0.8,
                    "active_params_b": 0.8,
                    "quant": "Q4_K_M",
                    "init": {
                        "init_ms": init_ms,
                        "mem_before": mem_before,
                        "gpu_before": gpu_before,
                        "gpu_after_init": gpu_after_init,
                    },
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved {out_path}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
