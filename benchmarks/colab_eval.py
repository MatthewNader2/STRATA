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
from typing import Any, Dict, List, Optional, Tuple

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
    build_react_feedback,
    evaluate_task_response,
    extract_code_clean,
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
    strip_reasoning_traces,
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
        "thinking_mode": "qwen_toggle",
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
        "thinking_mode": "harmony_effort",
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
        "thinking_mode": "none",
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
        "thinking_mode": "qwen_toggle",
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
        "thinking_mode": "qwen_toggle",
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


def _format_temp_tag(temp: float) -> Tuple[str, str]:
    val = float(temp)
    if val.is_integer():
        iv = str(int(val))
        return iv, iv
    s = f"{val:g}"
    return s.replace(".", "p"), s


def _thinking_tags(thinking_mode: str, disable_thinking: bool) -> Tuple[str, str]:
    if thinking_mode == "harmony_effort":
        return ("effort_low", "EffortLow") if disable_thinking else ("effort_high", "EffortHigh")
    if thinking_mode == "none":
        return ("instruct", "Instruct")
    return ("nothink", "NoThink") if disable_thinking else ("think", "Think")


def _parse_token_limits(raw_limits: List[str]) -> List[Optional[int]]:
    parsed: List[Optional[int]] = []
    for item in raw_limits:
        norm = str(item).strip().lower()
        if norm in ("unlimited", "none", "nolimit", "inf", "-1", "0"):
            if None not in parsed:
                parsed.append(None)
        else:
            val = _positive_int(norm)
            if val not in parsed:
                parsed.append(val)
    return parsed


def _build_cell_naming(
    *,
    model_key: str,
    spec: Dict[str, Any],
    context_mode: str,
    react_turns: int,
    temperature: float,
    disable_thinking: bool,
    max_tokens: Optional[int],
    use_detailed_tags: bool,
) -> Dict[str, Any]:
    turns_limit = max(1, int(react_turns or 1))
    mode_tag = {
        "pure": "Pure",
        "skill_md": "SKILL.md",
        "skill_md_haystack": "SKILL.md+Haystack",
    }[context_mode]
    if turns_limit > 1:
        mode_tag = f"{mode_tag}+ReAct-{turns_limit}T"

    thinking_mode = str(spec.get("thinking_mode") or "qwen_toggle")
    if not use_detailed_tags:
        arm_tag = context_mode if turns_limit == 1 else f"{context_mode}_react{turns_limit}t"
        return {
            "arm_tag": arm_tag,
            "model_id": f"{model_key}_{arm_tag}",
            "display": f"{spec['display']} [{mode_tag}]",
            "filename": f"results_{model_key}_{arm_tag}.json",
        }

    t_file, t_disp = _format_temp_tag(temperature)
    th_file, th_disp = _thinking_tags(thinking_mode, disable_thinking)
    lim_file = "unlim" if max_tokens is None else f"lim{int(max_tokens)}"
    lim_disp = "NoLimit" if max_tokens is None else f"Max{int(max_tokens)}"
    react_suffix = "" if turns_limit == 1 else f"_react{turns_limit}t"
    arm_tag = f"{context_mode}_t{t_file}_{th_file}_{lim_file}{react_suffix}"
    return {
        "arm_tag": arm_tag,
        "model_id": f"{model_key}_{arm_tag}",
        "display": f"{spec['display']} [{mode_tag} | T={t_disp} | {th_disp} | {lim_disp}]",
        "filename": f"results_{model_key}_{arm_tag}.json",
    }


def _dedupe_model_sweep_cells(
    *,
    thinking_mode: str,
    temperatures: List[float],
    thinking_modes: List[str],
    token_limits: List[Optional[int]],
) -> List[Tuple[float, bool, Optional[int]]]:
    priority = [
        (0.0, False, 4096),
        (0.0, True, 4096),
        (1.0, False, 4096),
        (0.0, False, None),
        (1.0, False, None),
        (1.0, True, 4096),
        (0.0, True, None),
        (1.0, True, None),
    ]
    requested: List[Tuple[float, bool, Optional[int]]] = []
    th_bools = [False] if thinking_mode == "none" else [m.lower() == "off" for m in thinking_modes]
    for temp in [float(t) for t in temperatures]:
        for dis_th in th_bools:
            for lim in token_limits:
                cell = (temp, dis_th, lim)
                if cell not in requested:
                    requested.append(cell)
    req_set = set(requested)
    ordered = [c for c in priority if c in req_set]
    for c in requested:
        if c not in ordered:
            ordered.append(c)
    return ordered


def build_chat_completion_payload(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = 4096,
    temperature: float = 0.0,
    disable_thinking: bool = False,
    thinking_mode: str = "qwen_toggle",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": float(temperature),
        "top_p": 0.95,
    }
    if max_tokens is not None and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    if thinking_mode == "harmony_effort":
        payload["chat_template_kwargs"] = {"reasoning_effort": "low" if disable_thinking else "high"}
    elif thinking_mode == "qwen_toggle" and disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def query_llama_server(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = 4096,
    temperature: float = 0.0,
    disable_thinking: bool = False,
    thinking_mode: str = "qwen_toggle",
) -> Dict[str, Any]:
    payload_dict = build_chat_completion_payload(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        disable_thinking=disable_thinking,
        thinking_mode=thinking_mode,
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
    kv_quant: bool = False,
) -> tuple[subprocess.Popen, Any, int, int, float]:
    base_ngl = int(spec["n_gpu_layers"])
    req_ctx = int(spec["n_ctx"])
    base_url = f"http://127.0.0.1:{port}"
    last_err: Exception | None = None

    attempts: List[Tuple[int, bool, int]] = [(req_ctx, bool(kv_quant), base_ngl)]
    if req_ctx > 8192:
        for ctx_cand in (12288, 8192):
            if ctx_cand < req_ctx:
                cand = (ctx_cand, bool(kv_quant), base_ngl)
                if cand not in attempts:
                    attempts.append(cand)
        cand_no_q = (8192, False, base_ngl)
        if cand_no_q not in attempts:
            attempts.append(cand_no_q)
    elif kv_quant:
        attempts.append((req_ctx, False, base_ngl))

    for ngl_val in (42, 32, 24, 0):
        if ngl_val != base_ngl:
            cand = (min(req_ctx, 8192), False, ngl_val)
            if cand not in attempts:
                attempts.append(cand)

    for ctx_val, use_kvq, ngl in attempts:
        cmd = [
            str(llama_server_bin),
            "-m", str(model_path),
            "-ngl", str(ngl),
            "-c", str(ctx_val),
            "--jinja",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--no-webui",
        ]
        if use_kvq:
            cmd.extend(["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"])
        if str(spec.get("thinking_mode") or "") == "harmony_effort":
            cmd.append("--skip-chat-parsing")
        _log(f"[COLAB] Starting llama-server (ngl={ngl}, n_ctx={ctx_val}, kv_quant={use_kvq}): {' '.join(cmd)}")
        log_fh = open(server_log, "w", encoding="utf-8")
        t_init0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        try:
            wait_for_server(base_url, proc, server_log=server_log, timeout_s=240.0)
            init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
            _log(f"[COLAB] Server ready in {init_ms/1000.0:.1f}s (ngl={ngl}, n_ctx={ctx_val})")
            return proc, log_fh, ngl, ctx_val, init_ms
        except Exception as exc:
            last_err = exc
            _log(f"[COLAB] Startup failed (ngl={ngl}, n_ctx={ctx_val}, kv_quant={use_kvq}):\n{exc}")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_fh.close()
    raise RuntimeError(f"All launch attempts failed for {model_path.name}: {last_err}")


def _load_existing_completed(out_path: Path) -> Dict[str, Dict[str, Any]]:
    if not out_path.is_file():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        existing: Dict[str, Dict[str, Any]] = {}
        for rec in data.get("records") or []:
            if not isinstance(rec, dict):
                continue
            tid = str(rec.get("task_id") or "")
            if tid and str(rec.get("status") or "") != "TIMEOUT":
                existing[tid] = rec
        return existing
    except Exception:
        return {}


def _run_cell_against_server(
    *,
    model_key: str,
    eff_spec: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    base_url: str,
    out_dir: Path,
    dl_ms: float,
    init_ms: float,
    used_ngl: int,
    used_n_ctx: int,
    kv_quant: bool,
    mem_before: Dict[str, Any],
    gpu_before: Dict[str, Any],
    gpu_after_init: Dict[str, Any],
    context_mode: str,
    temperature: float,
    disable_thinking: bool,
    max_tokens: Optional[int],
    haystack_chars: int,
    full_haystack: bool,
    react_turns: int,
    use_detailed_tags: bool,
    resume: bool,
) -> Path:
    turns_limit = max(1, int(react_turns or 1))
    thinking_mode = str(eff_spec.get("thinking_mode") or "qwen_toggle")
    naming = _build_cell_naming(
        model_key=model_key,
        spec=eff_spec,
        context_mode=context_mode,
        react_turns=turns_limit,
        temperature=temperature,
        disable_thinking=disable_thinking,
        max_tokens=max_tokens,
        use_detailed_tags=use_detailed_tags,
    )
    out_path = out_dir / naming["filename"]
    records_by_id: Dict[str, Dict[str, Any]] = _load_existing_completed(out_path) if resume else {}
    if resume and len(records_by_id) >= len(tasks):
        _log(f"[COLAB] Skipping completed cell (--resume): {out_path.name} ({len(records_by_id)}/{len(tasks)})")
        return out_path

    def _serialize_results() -> str:
        ordered = [records_by_id[t["id"]] for t in tasks if t["id"] in records_by_id]
        return json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": naming["arm_tag"],
                "model_id": naming["model_id"],
                "model_display": naming["display"],
                "family": eff_spec["family"],
                "params_b": eff_spec["params_b"],
                "active_params_b": eff_spec["active_params_b"],
                "quant": eff_spec["quant"],
                "init": {
                    "context_mode": context_mode,
                    "react_turns": turns_limit,
                    "temperature": float(temperature),
                    "disable_thinking": bool(disable_thinking),
                    "thinking_mode": thinking_mode,
                    "max_tokens": max_tokens,
                    "haystack_chars": int(haystack_chars),
                    "full_haystack": bool(full_haystack),
                    "download_ms": dl_ms,
                    "init_ms": init_ms,
                    "used_ngl": used_ngl,
                    "n_ctx": used_n_ctx,
                    "kv_quant": bool(kv_quant),
                    "mem_before": mem_before,
                    "gpu_before": gpu_before,
                    "gpu_after_init": gpu_after_init,
                },
                "records": ordered,
            },
            indent=2,
        )

    _log(f"\n-------------------- {naming['display']} ({len(records_by_id)}/{len(tasks)} cached) --------------------")
    for idx, task in enumerate(tasks, 1):
        tid = task["id"]
        if tid in records_by_id:
            continue

        messages, ctx_meta = build_messages_for_arm(
            task,
            context_mode=context_mode,
            max_distractor_chars=haystack_chars,
            full_haystack=full_haystack,
        )
        cpu0 = read_cpu_times()
        t0 = time.perf_counter()
        conv_messages = [dict(m) for m in messages]
        turns_used = 0
        sum_pt = 0
        sum_ct = 0
        sum_tt = 0
        sum_llm_ms = 0.0
        raw_reply = ""
        code = ""
        ev: Dict[str, Any] = {}
        try:
            for turn in range(1, turns_limit + 1):
                turns_used = turn
                res_llm = query_llama_server(
                    base_url,
                    messages=conv_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    disable_thinking=disable_thinking,
                    thinking_mode=thinking_mode,
                )
                sum_pt += res_llm["prompt_tokens"]
                sum_ct += res_llm["completion_tokens"]
                sum_tt += res_llm["total_tokens"]
                sum_llm_ms += res_llm["wall_ms"]
                raw_reply = res_llm["raw_reply"]
                code = extract_code_clean(raw_reply)
                ev = evaluate_task_response(raw_reply, code, task, timeout=45.0)
                if turn >= turns_limit:
                    break
                fb = build_react_feedback(code, ev, task)
                if fb is None:
                    break
                clean_prev = strip_reasoning_traces(raw_reply)
                assistant_turn = clean_prev or (f"```python\n{code}\n```" if code else "(empty)")
                conv_messages.append({"role": "assistant", "content": assistant_turn})
                conv_messages.append({"role": "user", "content": fb})
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            rec = {
                "arm": naming["arm_tag"],
                "model_id": naming["model_id"],
                "model_display": naming["display"],
                "family": eff_spec["family"],
                "params_b": eff_spec["params_b"],
                "active_params_b": eff_spec["active_params_b"],
                "quant": eff_spec["quant"],
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "disable_thinking": bool(disable_thinking),
                "max_tokens": max_tokens,
                "expected": task.get("expected", "synthesize"),
                "react_turns": turns_used,
                "max_turns": turns_limit,
                "prompt": task["prompt"],
                "status": ev["status"],
                "passed": ev["passed"],
                "acceptable": ev["acceptable"],
                "detail": ev["detail"],
                "wall_ms": wall_ms,
                "llm_calls": turns_used,
                "prompt_tokens": sum_pt,
                "completion_tokens": sum_ct,
                "total_tokens": sum_tt,
                "llm_wall_ms": round(sum_llm_ms, 2),
                "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                "mem_mb": read_proc_mem_mb(),
                "gpu_mb": read_gpu_vram_mb(),
                "audit": ev.get("audit"),
                "code": code,
                "raw_reply_preview": raw_reply[:1200],
            }
        except Exception as task_exc:
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            rec = {
                "arm": naming["arm_tag"],
                "model_id": naming["model_id"],
                "model_display": naming["display"],
                "family": eff_spec["family"],
                "params_b": eff_spec["params_b"],
                "active_params_b": eff_spec["active_params_b"],
                "quant": eff_spec["quant"],
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "disable_thinking": bool(disable_thinking),
                "max_tokens": max_tokens,
                "expected": task.get("expected", "synthesize"),
                "react_turns": turns_used,
                "max_turns": turns_limit,
                "prompt": task["prompt"],
                "status": "TIMEOUT",
                "passed": False,
                "acceptable": False,
                "detail": f"Request failed: {task_exc}",
                "wall_ms": wall_ms,
                "llm_calls": max(1, turns_used),
                "prompt_tokens": sum_pt,
                "completion_tokens": sum_ct,
                "total_tokens": sum_tt,
                "llm_wall_ms": round(sum_llm_ms, 2),
                "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                "mem_mb": read_proc_mem_mb(),
                "gpu_mb": read_gpu_vram_mb(),
                "audit": None,
                "code": "",
                "raw_reply_preview": "",
            }
        records_by_id[tid] = rec
        out_path.write_text(_serialize_results(), encoding="utf-8")
        _log(
            f"  [{idx:02d}/{len(tasks):02d}] {tid:<30} | {rec['status']:<18} "
            f"| acc={str(rec['acceptable']):<5} | {wall_ms/1000.0:5.1f}s "
            f"| tok={rec['prompt_tokens']}+{rec['completion_tokens']}"
        )

    out_path.write_text(_serialize_results(), encoding="utf-8")
    _log(f"[COLAB] Saved {out_path}")
    return out_path


def run_model_sweep_via_server(
    model_key: str,
    spec: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    llama_server_bin: str,
    env: Dict[str, str],
    cache_dir: Path,
    out_dir: Path,
    *,
    context_modes: List[str],
    temperatures: List[float],
    thinking_modes: List[str],
    token_limits: List[Optional[int]],
    port: int = 8091,
    cleanup_cache: bool = False,
    n_ctx_override: int | None = None,
    kv_quant: bool = False,
    haystack_chars: int = 18000,
    full_haystack: bool = False,
    react_turns: int = 1,
    resume: bool = False,
    use_detailed_tags: bool = True,
) -> List[Path]:
    eff_spec = dict(spec)
    if n_ctx_override is not None:
        eff_spec["n_ctx"] = int(n_ctx_override)

    thinking_mode = str(eff_spec.get("thinking_mode") or "qwen_toggle")
    decoding_cells = _dedupe_model_sweep_cells(
        thinking_mode=thinking_mode,
        temperatures=temperatures,
        thinking_modes=thinking_modes,
        token_limits=token_limits,
    )

    pending_plan: List[Tuple[str, float, bool, Optional[int], Path]] = []
    out_paths: List[Path] = []
    for ctx_mode in context_modes:
        for temp, dis_th, max_tok in decoding_cells:
            naming = _build_cell_naming(
                model_key=model_key,
                spec=eff_spec,
                context_mode=ctx_mode,
                react_turns=react_turns,
                temperature=temp,
                disable_thinking=dis_th,
                max_tokens=max_tok,
                use_detailed_tags=use_detailed_tags,
            )
            p = out_dir / naming["filename"]
            out_paths.append(p)
            cached = _load_existing_completed(p) if resume else {}
            if resume and len(cached) >= len(tasks):
                _log(f"[COLAB] Pre-skip completed cell (--resume): {p.name}")
                continue
            pending_plan.append((ctx_mode, temp, dis_th, max_tok, p))

    if not pending_plan:
        _log(f"[COLAB] All {len(out_paths)} cells already completed for {model_key}; skipping server launch.")
        return out_paths

    model_cache = cache_dir / model_key
    t_dl0 = time.perf_counter()
    _log(f"\n==================== {eff_spec['display']} ({len(pending_plan)} cell(s)) ====================")
    _log(f"[COLAB] Downloading {eff_spec['repo_id']} :: {eff_spec['filename']} ...")
    model_path = download_gguf(eff_spec["repo_id"], eff_spec["filename"], model_cache)
    dl_ms = round((time.perf_counter() - t_dl0) * 1000.0, 2)
    _log(f"[COLAB] Model ready at {model_path} ({dl_ms/1000.0:.1f}s)")

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    server_log = out_dir / f"{model_key}_server.log"
    base_url = f"http://127.0.0.1:{port}"

    proc = None
    log_fh = None
    try:
        proc, log_fh, used_ngl, used_n_ctx, init_ms = _launch_server_with_fallback(
            llama_server_bin=llama_server_bin,
            model_path=model_path,
            spec=eff_spec,
            env=env,
            server_log=server_log,
            port=port,
            kv_quant=kv_quant,
        )
        gpu_after_init = read_gpu_vram_mb()

        for ctx_mode, temp, dis_th, max_tok, _ in pending_plan:
            _run_cell_against_server(
                model_key=model_key,
                eff_spec=eff_spec,
                tasks=tasks,
                base_url=base_url,
                out_dir=out_dir,
                dl_ms=dl_ms,
                init_ms=init_ms,
                used_ngl=used_ngl,
                used_n_ctx=used_n_ctx,
                kv_quant=kv_quant,
                mem_before=mem_before,
                gpu_before=gpu_before,
                gpu_after_init=gpu_after_init,
                context_mode=ctx_mode,
                temperature=temp,
                disable_thinking=dis_th,
                max_tokens=max_tok,
                haystack_chars=haystack_chars,
                full_haystack=full_haystack,
                react_turns=react_turns,
                use_detailed_tags=use_detailed_tags,
                resume=resume,
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

    return out_paths


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
    react_turns: int = 1,
    temperature: float = 0.0,
    no_token_limit: bool = False,
    kv_quant: bool = False,
    resume: bool = False,
) -> Path:
    max_tok: Optional[int] = None if no_token_limit else int(
        max_tokens_override if max_tokens_override is not None else spec.get("max_tokens", 4096)
    )
    use_detailed = bool(
        float(temperature) != 0.0
        or disable_thinking
        or no_token_limit
        or max_tokens_override is not None
    )
    paths = run_model_sweep_via_server(
        model_key=model_key,
        spec=spec,
        tasks=tasks,
        llama_server_bin=llama_server_bin,
        env=env,
        cache_dir=cache_dir,
        out_dir=out_dir,
        context_modes=[context_mode],
        temperatures=[float(temperature)],
        thinking_modes=["off" if disable_thinking else "on"],
        token_limits=[max_tok],
        port=port,
        cleanup_cache=cleanup_cache,
        n_ctx_override=n_ctx_override,
        kv_quant=kv_quant,
        haystack_chars=haystack_chars,
        full_haystack=full_haystack,
        react_turns=react_turns,
        resume=resume,
        use_detailed_tags=use_detailed,
    )
    return paths[0]


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
        "--context-modes",
        nargs="+",
        choices=list(VALID_CONTEXT_MODES),
        default=None,
        help="Multiple context arms for --sweep-grid (defaults to --context-mode)",
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
    ap.add_argument("--resume", action="store_true", help="Skip completed tasks/cells whose output JSON already exists")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for single-arm runs")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking/set low reasoning effort for single-arm runs",
    )
    ap.add_argument(
        "--no-token-limit",
        action="store_true",
        help="Omit max_tokens from llama-server requests (generate up to n_ctx)",
    )
    ap.add_argument(
        "--kv-quant",
        action="store_true",
        help="Enable -fa on -ctk q8_0 -ctv q8_0 when launching llama-server",
    )
    ap.add_argument(
        "--sweep-grid",
        action="store_true",
        help="Evaluate a decoding factorial grid per loaded model before unloading",
    )
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 1.0], help="Temperatures for --sweep-grid")
    ap.add_argument(
        "--thinking-modes",
        nargs="+",
        choices=["on", "off"],
        default=["on", "off"],
        help="Thinking toggles for --sweep-grid",
    )
    ap.add_argument(
        "--token-limits",
        nargs="+",
        default=["4096", "unlimited"],
        help="Token limits for --sweep-grid (int or 'unlimited')",
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
    ap.add_argument(
        "--react-turns",
        type=_positive_int,
        default=1,
        help="Max zero-oracle execution/compiler feedback turns per synthesis task (1 = single-shot)",
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    tasks = get_suite_tasks(args.suite)
    ctx_modes = list(args.context_modes) if args.context_modes else [args.context_mode]
    _log(
        f"[COLAB] Suite='{args.suite}' ({len(tasks)} tasks) | Contexts={ctx_modes} "
        f"| Turns={args.react_turns} | Sweep={args.sweep_grid} | Models={args.models}"
    )

    env = verify_llama_server(args.llama_server)
    failures: List[str] = []

    if args.sweep_grid:
        parsed_limits = _parse_token_limits(args.token_limits)
        for mkey in args.models:
            try:
                run_model_sweep_via_server(
                    model_key=mkey,
                    spec=COLAB_MODEL_REGISTRY[mkey],
                    tasks=tasks,
                    llama_server_bin=args.llama_server,
                    env=env,
                    cache_dir=cache_dir,
                    out_dir=out_dir,
                    context_modes=ctx_modes,
                    temperatures=list(args.temperatures),
                    thinking_modes=list(args.thinking_modes),
                    token_limits=parsed_limits,
                    port=args.port,
                    cleanup_cache=args.cleanup_cache,
                    n_ctx_override=args.n_ctx,
                    kv_quant=args.kv_quant,
                    haystack_chars=args.haystack_chars,
                    full_haystack=args.full_haystack,
                    react_turns=args.react_turns,
                    resume=args.resume,
                    use_detailed_tags=True,
                )
            except Exception as exc:
                _log(f"[COLAB][ERROR] Model '{mkey}' sweep failed: {exc}")
                traceback.print_exc()
                failures.append(mkey)
    else:
        for mkey in args.models:
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
                    context_mode=ctx_modes[0],
                    haystack_chars=args.haystack_chars,
                    full_haystack=args.full_haystack,
                    react_turns=args.react_turns,
                    temperature=args.temperature,
                    no_token_limit=args.no_token_limit,
                    kv_quant=args.kv_quant,
                    resume=args.resume,
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
