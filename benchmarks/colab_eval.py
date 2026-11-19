#!/usr/bin/env python3
"""
benchmarks/colab_eval.py - Google Colab T4 (15-16 GB VRAM) evaluation harness.

Runs requested frontier GGUF models via `llama-server` (latest prebuilt CUDA binary
from ggml-org/llama.cpp, avoiding Python wheel lag on Qwen3.8 / Qwen3.5 / GPT-OSS)
with live unbuffered logging, server stderr diagnostics, and automatic `-ngl` fallback
if a large checkpoint overflows T4 VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (str(ROOT_DIR / "src"), str(ROOT_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

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
        "n_ctx": 4096,
        "max_tokens": 2560,
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
        "n_ctx": 4096,
        "max_tokens": 2560,
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
        "n_ctx": 4096,
        "max_tokens": 2560,
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
        "n_ctx": 4096,
        "max_tokens": 2560,
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
        "n_ctx": 4096,
        "max_tokens": 2560,
    },
}

DEFAULT_PURE_SYS = (
    "You are a concise, precise Python developer. Write ONLY clean executable Python code inside a single "
    "```python ... ``` code block solving the user's request. Use standard/installed libraries only. "
    "Preserve every literal value, filename, column name, and parameter specified in the prompt. "
    "If the prompt is genuinely too underspecified to write concrete working code without guessing critical missing rules, "
    "ask 2 concise clarifying questions instead of writing speculative code."
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tail_text(path: Path, n_lines: int = 45) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def _build_server_env(llama_server_bin: str) -> Dict[str, str]:
    env = os.environ.copy()
    bin_path = Path(llama_server_bin).resolve()
    search_roots = [bin_path.parent, bin_path.parent.parent]
    so_dirs: List[str] = [str(bin_path.parent)]
    for root in search_roots:
        if root.exists():
            for so in root.rglob("*.so*"):
                p_str = str(so.parent)
                if p_str not in so_dirs:
                    so_dirs.append(p_str)
    for cuda_dir in ("/usr/local/cuda/lib64", "/usr/local/cuda-12.8/lib64", "/usr/lib64-nvidia"):
        if os.path.isdir(cuda_dir) and cuda_dir not in so_dirs:
            so_dirs.append(cuda_dir)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(so_dirs + ([existing] if existing else []))
    return env


def verify_llama_server(llama_server_bin: str) -> Dict[str, str]:
    env = _build_server_env(llama_server_bin)
    proc = subprocess.run([str(llama_server_bin), "--version"], env=env, capture_output=True, text=True, timeout=15)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-server --version failed (exit {proc.returncode}) at '{llama_server_bin}'.\n"
            f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}\nSTDOUT: {out}\nSTDERR: {err}"
        )
    _log(f"[COLAB] llama-server verified: {out or err}")
    return env


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


def wait_for_server(base_url: str, proc: subprocess.Popen, server_log: Path, timeout_s: float = 240.0) -> None:
    deadline = time.time() + timeout_s
    last_dot = time.time()
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = _tail_text(server_log, 50)
            raise RuntimeError(f"llama-server exited with code {rc}.\n--- server log tail ---\n{tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if data.get("status") == "ok":
                        return
        except Exception:
            pass
        if time.time() - last_dot >= 15.0:
            _log("[COLAB] Still loading weights into VRAM...")
            last_dot = time.time()
        time.sleep(1.5)
    tail = _tail_text(server_log, 50)
    raise TimeoutError(f"llama-server did not become ready within {timeout_s}s.\n--- server log tail ---\n{tail}")


def query_llama_server(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 2560,
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or "")
    raw_combined = f"<think>{reasoning}</think>\n{content}" if (reasoning and not content.strip().startswith("<think>")) else (content or reasoning)
    usage = data.get("usage") or {}
    return {
        "raw_reply": raw_combined.strip(),
        "wall_ms": dt_ms,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _launch_server_with_fallback(
    llama_server_bin: str,
    model_path: Path,
    spec: Dict[str, Any],
    env: Dict[str, str],
    server_log: Path,
    port: int,
) -> tuple[subprocess.Popen, Any, int, float]:
    base_ngl = int(spec["n_gpu_layers"])
    ngl_candidates = [base_ngl] + [v for v in (42, 32, 24, 0) if v != base_ngl]
    base_url = f"http://127.0.0.1:{port}"
    last_err: Exception | None = None

    for ngl in ngl_candidates:
        cmd = [
            str(llama_server_bin),
            "-m", str(model_path),
            "-ngl", str(ngl),
            "-c", str(spec["n_ctx"]),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--no-webui",
        ]
        _log(f"[COLAB] Starting llama-server (ngl={ngl}, n_ctx={spec['n_ctx']}): {' '.join(cmd)}")
        log_fh = open(server_log, "w", encoding="utf-8")
        t_init0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        try:
            wait_for_server(base_url, proc, server_log=server_log, timeout_s=240.0)
            init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
            _log(f"[COLAB] Server ready in {init_ms/1000.0:.1f}s (ngl={ngl})")
            return proc, log_fh, ngl, init_ms
        except Exception as exc:
            last_err = exc
            _log(f"[COLAB] Startup failed with ngl={ngl}:\n{exc}")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_fh.close()
    raise RuntimeError(f"All ngl attempts failed for {model_path.name}: {last_err}")


def run_model_via_server(
    model_key: str,
    spec: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    llama_server_bin: str,
    env: Dict[str, str],
    cache_dir: Path,
    out_dir: Path,
    port: int = 8091,
    cleanup_cache: bool = False,
) -> Path:
    model_cache = cache_dir / model_key
    t_dl0 = time.perf_counter()
    _log(f"\n==================== {spec['display']} ====================")
    _log(f"[COLAB] Downloading {spec['repo_id']} :: {spec['filename']} ...")
    model_path = download_gguf(spec["repo_id"], spec["filename"], model_cache)
    dl_ms = round((time.perf_counter() - t_dl0) * 1000.0, 2)
    _log(f"[COLAB] Model ready at {model_path} ({dl_ms/1000.0:.1f}s)")

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    server_log = out_dir / f"{model_key}_server.log"
    base_url = f"http://127.0.0.1:{port}"

    proc = None
    log_fh = None
    records: List[Dict[str, Any]] = []
    init_ms = 0.0
    used_ngl = spec["n_gpu_layers"]
    gpu_after_init: Dict[str, Any] = {}

    try:
        proc, log_fh, used_ngl, init_ms = _launch_server_with_fallback(
            llama_server_bin=llama_server_bin,
            model_path=model_path,
            spec=spec,
            env=env,
            server_log=server_log,
            port=port,
        )
        gpu_after_init = read_gpu_vram_mb()

        out_path = out_dir / f"results_{model_key}.json"

        def _flush_checkpoint() -> None:
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
                            "used_ngl": used_ngl,
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

        for idx, task in enumerate(tasks, 1):
            messages = build_task_messages(task)
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            try:
                res_llm = query_llama_server(
                    base_url,
                    messages=messages,
                    max_tokens=spec.get("max_tokens", 2560),
                )
                code = extract_code_clean(res_llm["raw_reply"])
                ev = evaluate_task_response(res_llm["raw_reply"], code, task, timeout=45.0)
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
            except Exception as task_exc:
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
                    "status": "TIMEOUT",
                    "passed": False,
                    "acceptable": False,
                    "detail": f"Request failed: {task_exc}",
                    "wall_ms": wall_ms,
                    "llm_calls": 1,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "llm_wall_ms": wall_ms,
                    "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                    "mem_mb": read_proc_mem_mb(),
                    "gpu_mb": read_gpu_vram_mb(),
                    "audit": None,
                    "code": "",
                    "raw_reply_preview": "",
                }
            records.append(rec)
            _flush_checkpoint()
            _log(
                f"  [{idx:02d}/{len(tasks):02d}] {task['id']:<18} | {rec['status']:<16} "
                f"| acc={str(rec['acceptable']):<5} | {wall_ms/1000.0:5.1f}s "
                f"| tok={rec['prompt_tokens']}+{rec['completion_tokens']}"
            )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        if log_fh is not None:
            log_fh.close()
        if cleanup_cache and model_cache.exists():
            shutil.rmtree(model_cache, ignore_errors=True)
            _log(f"[COLAB] Freed disk cache: {model_cache}")

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
                    "used_ngl": used_ngl,
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
    _log(f"[COLAB] Saved {out_path}")
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
    _log(f"[COLAB] Suite='{args.suite}' ({len(tasks)} tasks) | Models={args.models}")

    env = verify_llama_server(args.llama_server)

    failures: List[str] = []
    for mkey in args.models:
        out_json = out_dir / f"results_{mkey}.json"
        if args.resume and out_json.exists():
            _log(f"[COLAB] Skipping {mkey} (--resume, exists: {out_json})")
            continue
        try:
            run_model_via_server(
                model_key=mkey,
                spec=COLAB_MODEL_REGISTRY[mkey],
                tasks=tasks,
                llama_server_bin=args.llama_server,
                env=env,
                cache_dir=cache_dir,
                out_dir=out_dir,
                port=args.port,
                cleanup_cache=args.cleanup_cache,
            )
        except Exception as exc:
            _log(f"[COLAB][ERROR] Model '{mkey}' failed: {exc}")
            traceback.print_exc()
            failures.append(mkey)

    if failures:
        _log(f"\n[COLAB] Completed with failures on: {failures}")
        sys.exit(1)
    _log("\n[COLAB] All requested models finished cleanly!")


if __name__ == "__main__":
    main()
