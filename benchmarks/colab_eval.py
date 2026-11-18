#!/usr/bin/env python3
"""
benchmarks/colab_eval.py - Google Colab T4 (15-16 GB VRAM) evaluation harness.

Runs requested frontier GGUF models via `llama-server` (latest prebuilt CUDA binary
from ggml-org/llama.cpp, avoiding Python wheel lag on Qwen3.8 / Qwen3.5 / GPT-OSS)
plus optional STRATA Profile O + OMEGA (`0.8B`) on the same tasks.

All results are written in STRATA's unified v2 JSON schema for direct comparison
and graphing via `benchmarks/visualize_results.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from benchmarks.evaluators import (  # noqa: E402
    evaluate_task_response,
    extract_code_clean,
    load_sdk_docs,
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
)
from benchmarks.suites import get_suite_tasks  # noqa: E402

COLAB_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3.8-27b": {
        "display": "Qwen3.8-27B (UD-IQ3_S, 12.0GB)",
        "family": "Qwen3.8",
        "params_b": 27.0,
        "active_params_b": 27.0,
        "quant": "UD-IQ3_S",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-UD-IQ3_S.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "max_tokens": 2048,
    },
    "gpt-oss-20b": {
        "display": "GPT-OSS-20B (Q4_K_M, 11.6GB)",
        "family": "GPT-OSS",
        "params_b": 20.0,
        "active_params_b": 3.6,
        "quant": "Q4_K_M",
        "repo_id": "unsloth/gpt-oss-20b-GGUF",
        "filename": "gpt-oss-20b-Q4_K_M.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "max_tokens": 2048,
    },
    "qwen3-coder-30b-a3b": {
        "display": "Qwen3-Coder-30B-A3B-Instruct (Q3_K_S, 13.3GB)",
        "family": "Qwen3-Coder",
        "params_b": 30.5,
        "active_params_b": 3.3,
        "quant": "Q3_K_S",
        "repo_id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q3_K_S.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "max_tokens": 2048,
    },
    "qwen3.5-35b-a3b": {
        "display": "Qwen3.5-35B-A3B MoE (UD-IQ3_XXS, 13.1GB)",
        "family": "Qwen3.5-MoE",
        "params_b": 35.0,
        "active_params_b": 3.0,
        "quant": "UD-IQ3_XXS",
        "repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "filename": "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "max_tokens": 2048,
    },
    "qwen3.5-27b": {
        "display": "Qwen3.5-27B Dense (Q3_K_S, 12.3GB)",
        "family": "Qwen3.5-Dense",
        "params_b": 27.0,
        "active_params_b": 27.0,
        "quant": "Q3_K_S",
        "repo_id": "unsloth/Qwen3.5-27B-GGUF",
        "filename": "Qwen3.5-27B-Q3_K_S.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 8192,
        "max_tokens": 2048,
    },
}

DEFAULT_PURE_SYS = (
    "You are a concise, precise Python developer. Write ONLY clean executable Python code inside a single "
    "```python ... ``` code block solving the user's request. Use standard/installed libraries only. "
    "Preserve every literal value, filename, column name, and parameter specified in the prompt. "
    "If the prompt is genuinely too underspecified to write concrete working code without guessing critical missing rules, "
    "ask 2 concise clarifying questions instead of writing speculative code."
)


def build_task_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    if task["kind"] == "sdk":
        docs_md = load_sdk_docs(task["sdk"])
        sys_msg = (
            "You are a Python coding assistant. Use ONLY the API documented below to write a single "
            "executable Python script inside a ```python ... ``` code block solving the user's request.\n\n"
            + docs_md
        )
    else:
        sys_msg = DEFAULT_PURE_SYS
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": task["prompt"]},
    ]


def download_gguf(repo_id: str, filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
    )
    return Path(local_path)


def wait_for_server(base_url: str, proc: subprocess.Popen, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited prematurely with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if data.get("status") == "ok":
                        return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"llama-server did not become ready within {timeout_s}s")


def query_llama_server(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = str(msg.get("content") or "")
    # Some servers separate reasoning_content from content; handle both gracefully
    usage = data.get("usage") or {}
    return {
        "raw_reply": content,
        "wall_ms": dt_ms,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def run_model_via_server(
    model_key: str,
    spec: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    llama_server_bin: str,
    cache_dir: Path,
    out_dir: Path,
    port: int = 8091,
    cleanup_cache: bool = False,
) -> Path:
    model_cache = cache_dir / model_key
    t_dl0 = time.perf_counter()
    print(f"\n[COLAB] Downloading/locating {spec['repo_id']} :: {spec['filename']} ...")
    model_path = download_gguf(spec["repo_id"], spec["filename"], model_cache)
    dl_ms = round((time.perf_counter() - t_dl0) * 1000.0, 2)

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    server_log = out_dir / f"{model_key}_server.log"

    env = os.environ.copy()
    bin_dir = str(Path(llama_server_bin).resolve().parent)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    cmd = [
        str(llama_server_bin),
        "-m", str(model_path),
        "-ngl", str(spec["n_gpu_layers"]),
        "-c", str(spec["n_ctx"]),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--flash-attn", "on",
    ]
    print(f"[COLAB] Starting llama-server: {' '.join(cmd)}")
    log_fh = open(server_log, "w", encoding="utf-8")
    t_init0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_server(base_url, proc, timeout_s=240.0)
        init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
        gpu_after_init = read_gpu_vram_mb()

        records: List[Dict[str, Any]] = []
        for task in tasks:
            messages = build_task_messages(task)
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            res_llm = query_llama_server(
                base_url,
                messages=messages,
                max_tokens=spec.get("max_tokens", 2048),
            )
            code = extract_code_clean(res_llm["raw_reply"])
            ev = evaluate_task_response(res_llm["raw_reply"], code, task, timeout=30.0)
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            rec = {
                "arm": "pure",
                "model_id": model_key,
                "model_display": spec["display"],
                "family": spec["family"],
                "params_b": spec["params_b"],
                "active_params_b": spec["active_params_b"],
                "quant": spec["quant"],
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
                "llm_calls": 1,
                "prompt_tokens": res_llm["prompt_tokens"],
                "completion_tokens": res_llm["completion_tokens"],
                "total_tokens": res_llm["total_tokens"],
                "llm_wall_ms": res_llm["wall_ms"],
                "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                "mem_mb": read_proc_mem_mb(),
                "gpu_mb": read_gpu_vram_mb(),
                "audit": ev.get("audit"),
                "code": code,
                "raw_reply_preview": res_llm["raw_reply"][:1200],
            }
            records.append(rec)
            print(
                f"  [{model_key}] {task['id']:<18} | status={ev['status']:<17} "
                f"| acc={ev['acceptable']} | wall={wall_ms:7.1f}ms "
                f"| tok={res_llm['total_tokens']:4d}"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log_fh.close()
        if cleanup_cache and model_cache.exists():
            shutil.rmtree(model_cache, ignore_errors=True)
            print(f"[COLAB] Cleaned cached weights at {model_cache}")

    out_path = out_dir / f"results_{model_key}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": "pure",
                "model_id": model_key,
                "model_display": spec["display"],
                "family": spec["family"],
                "params_b": spec["params_b"],
                "active_params_b": spec["active_params_b"],
                "quant": spec["quant"],
                "init": {
                    "download_ms": dl_ms,
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
    print(f"[COLAB] Saved {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="STRATA Colab T4 Multi-Model Benchmark Runner")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(COLAB_MODEL_REGISTRY.keys()),
        choices=list(COLAB_MODEL_REGISTRY.keys()),
        help="Model keys to evaluate sequentially",
    )
    ap.add_argument(
        "--suite",
        default="core9",
        help="Suite group from benchmarks/suites.py (core8, cryopulse, core9, aetheris, stress, all)",
    )
    ap.add_argument("--llama-server", default="llama-server", help="Path to llama-server CUDA binary")
    ap.add_argument("--cache-dir", default="/content/hf_gguf_cache", help="Directory for GGUF downloads")
    ap.add_argument("--out-dir", default="test_logs/colab_bench", help="Output directory for result JSONs")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--cleanup-cache", action="store_true", help="Delete GGUF file after each model finishes")
    ap.add_argument("--resume", action="store_true", help="Skip models whose output JSON already exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    tasks = get_suite_tasks(args.suite)
    print(f"[COLAB] Suite='{args.suite}' ({len(tasks)} tasks) | Models={args.models}")

    for mkey in args.models:
        out_json = out_dir / f"results_{mkey}.json"
        if args.resume and out_json.exists():
            print(f"[COLAB] Skipping {mkey} (--resume, exists: {out_json})")
            continue
        run_model_via_server(
            model_key=mkey,
            spec=COLAB_MODEL_REGISTRY[mkey],
            tasks=tasks,
            llama_server_bin=args.llama_server,
            cache_dir=cache_dir,
            out_dir=out_dir,
            port=args.port,
            cleanup_cache=args.cleanup_cache,
        )


if __name__ == "__main__":
    main()
