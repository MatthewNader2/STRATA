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
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
)
from benchmarks.skill_docs import (  # noqa: E402
    VALID_CONTEXT_MODES,
    build_messages_for_arm,
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
        "max_tokens": 4096,
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
        "max_tokens": 4096,
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
        "max_tokens": 4096,
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
        "max_tokens": 4096,
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
        "max_tokens": 4096,
    },
}

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


def build_task_messages(
    task: Dict[str, Any],
    *,
    context_mode: str = "skill_md",
    max_distractor_chars: int = 18000,
    full_haystack: bool = False,
) -> List[Dict[str, str]]:
    messages, _ = build_messages_for_arm(
        task,
        context_mode=context_mode,
        max_distractor_chars=max_distractor_chars,
        full_haystack=full_haystack,
    )
    return messages


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


def _positive_int(raw: str) -> int:
    try:
        val = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value: {raw!r}") from exc
    if val <= 0:
        raise argparse.ArgumentTypeError(f"value must be a positive integer, got {val}")
    return val


def build_chat_completion_payload(
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.95,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def query_llama_server(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    disable_thinking: bool = False,
) -> Dict[str, Any]:
    payload_dict = build_chat_completion_payload(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
    )
    payload = json.dumps(payload_dict).encode("utf-8")
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
    disable_thinking: bool = False,
    max_tokens_override: int | None = None,
    n_ctx_override: int | None = None,
    context_mode: str = "skill_md",
    haystack_chars: int = 18000,
    full_haystack: bool = False,
) -> Path:
    eff_spec = dict(spec)
    if n_ctx_override is not None:
        eff_spec["n_ctx"] = int(n_ctx_override)
    if max_tokens_override is not None:
        eff_spec["max_tokens"] = int(max_tokens_override)

    mode_tag = {
        "pure": "Pure",
        "skill_md": "SKILL.md",
        "skill_md_haystack": "SKILL.md+Haystack",
    }[context_mode]
    display_with_mode = f"{eff_spec['display']} [{mode_tag}]"

    model_cache = cache_dir / model_key
    t_dl0 = time.perf_counter()
    _log(f"\n==================== {display_with_mode} ====================")
    _log(f"[COLAB] Downloading {eff_spec['repo_id']} :: {eff_spec['filename']} ...")
    model_path = download_gguf(eff_spec["repo_id"], eff_spec["filename"], model_cache)
    dl_ms = round((time.perf_counter() - t_dl0) * 1000.0, 2)
    _log(f"[COLAB] Model ready at {model_path} ({dl_ms/1000.0:.1f}s)")

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    server_log = out_dir / f"{model_key}_{context_mode}_server.log"
    base_url = f"http://127.0.0.1:{port}"

    proc = None
    log_fh = None
    records: List[Dict[str, Any]] = []
    init_ms = 0.0
    used_ngl = eff_spec["n_gpu_layers"]
    gpu_after_init: Dict[str, Any] = {}

    def _serialize_results() -> str:
        return json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": context_mode,
                "model_id": f"{model_key}_{context_mode}",
                "model_display": display_with_mode,
                "family": eff_spec["family"],
                "params_b": eff_spec["params_b"],
                "active_params_b": eff_spec["active_params_b"],
                "quant": eff_spec["quant"],
                "init": {
                    "context_mode": context_mode,
                    "haystack_chars": int(haystack_chars),
                    "full_haystack": bool(full_haystack),
                    "download_ms": dl_ms,
                    "init_ms": init_ms,
                    "used_ngl": used_ngl,
                    "n_ctx": eff_spec["n_ctx"],
                    "max_tokens": eff_spec["max_tokens"],
                    "disable_thinking": bool(disable_thinking),
                    "mem_before": mem_before,
                    "gpu_before": gpu_before,
                    "gpu_after_init": gpu_after_init,
                },
                "records": records,
            },
            indent=2,
        )

    out_path = out_dir / f"results_{model_key}_{context_mode}.json"
    try:
        proc, log_fh, used_ngl, init_ms = _launch_server_with_fallback(
            llama_server_bin=llama_server_bin,
            model_path=model_path,
            spec=eff_spec,
            env=env,
            server_log=server_log,
            port=port,
        )
        gpu_after_init = read_gpu_vram_mb()

        def _flush_checkpoint() -> None:
            out_path.write_text(_serialize_results(), encoding="utf-8")

        for idx, task in enumerate(tasks, 1):
            messages, ctx_meta = build_messages_for_arm(
                task,
                context_mode=context_mode,
                max_distractor_chars=haystack_chars,
                full_haystack=full_haystack,
            )
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            try:
                res_llm = query_llama_server(
                    base_url,
                    messages=messages,
                    max_tokens=eff_spec.get("max_tokens", 4096),
                    disable_thinking=disable_thinking,
                )
                code = extract_code_clean(res_llm["raw_reply"])
                ev = evaluate_task_response(res_llm["raw_reply"], code, task, timeout=45.0)
                wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                cpu1 = read_cpu_times()
                rec = {
                    "arm": context_mode,
                    "model_id": f"{model_key}_{context_mode}",
                    "model_display": display_with_mode,
                    "family": eff_spec["family"],
                    "params_b": eff_spec["params_b"],
                    "active_params_b": eff_spec["active_params_b"],
                    "quant": eff_spec["quant"],
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "domain_set": ctx_meta["domain_set"],
                    "context_mode": context_mode,
                    "context_meta": ctx_meta,
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
                    "arm": context_mode,
                    "model_id": f"{model_key}_{context_mode}",
                    "model_display": display_with_mode,
                    "family": eff_spec["family"],
                    "params_b": eff_spec["params_b"],
                    "active_params_b": eff_spec["active_params_b"],
                    "quant": eff_spec["quant"],
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "domain_set": ctx_meta["domain_set"],
                    "context_mode": context_mode,
                    "context_meta": ctx_meta,
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
                f"  [{idx:02d}/{len(tasks):02d}] {task['id']:<30} | {rec['status']:<18} "
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

    out_path.write_text(_serialize_results(), encoding="utf-8")
    _log(f"[COLAB] Saved {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
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
        default="v2_expanded",
        help=(
            "Suite group from benchmarks/suites.py "
            "(v2_expanded, v2_full, v2_nl_core, v2_unseen, v2_haystack, v2_wrong_domain, v2_domain_spec, "
            "core8, cryopulse, core9, aetheris, stress, all)"
        ),
    )
    ap.add_argument(
        "--context-mode",
        choices=list(VALID_CONTEXT_MODES),
        default="skill_md",
        help="Prompt baseline arm: pure (no .md docs), skill_md (loaded domain_set docs), skill_md_haystack (15-domain noise)",
    )
    ap.add_argument(
        "--haystack-chars",
        type=_positive_int,
        default=18000,
        help="Character budget for multi-domain SKILL.md catalog (positive int)",
    )
    ap.add_argument(
        "--full-haystack",
        action="store_true",
        help="Include every cell in all loaded domain trees without truncation",
    )
    ap.add_argument("--llama-server", default="llama-server", help="Path to llama-server CUDA binary")
    ap.add_argument("--cache-dir", default="/content/hf_gguf_cache", help="Directory for GGUF downloads")
    ap.add_argument("--out-dir", default="test_logs/colab_bench", help="Output directory for result JSONs")
    ap.add_argument("--port", type=_positive_int, default=8091)
    ap.add_argument("--cleanup-cache", action="store_true", help="Delete GGUF file after each model finishes")
    ap.add_argument("--resume", action="store_true", help="Skip models whose output JSON already exists")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs={'enable_thinking': False} to llama-server chat completions",
    )
    ap.add_argument(
        "--n-ctx",
        type=_positive_int,
        default=None,
        help="Override model context size (-c) for all launched servers (positive int)",
    )
    ap.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Override per-request max completion tokens (positive int)",
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    tasks = get_suite_tasks(args.suite)
    _log(f"[COLAB] Suite='{args.suite}' ({len(tasks)} tasks) | Context='{args.context_mode}' | Models={args.models}")

    env = verify_llama_server(args.llama_server)

    failures: List[str] = []
    for mkey in args.models:
        out_json = out_dir / f"results_{mkey}_{args.context_mode}.json"
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
                disable_thinking=args.disable_thinking,
                max_tokens_override=args.max_tokens,
                n_ctx_override=args.n_ctx,
                context_mode=args.context_mode,
                haystack_chars=args.haystack_chars,
                full_haystack=args.full_haystack,
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
