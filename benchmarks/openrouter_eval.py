#!/usr/bin/env python3
"""
benchmarks/openrouter_eval.py - Multi-gateway free-tier API benchmark harness.

Runs controlled decoding ablations (temperature x reasoning toggle x max_tokens)
for hosted OpenAI-compatible free-tier frontier models (OpenRouter, SandBase,
OrcaRouter, TeamoRouter) against STRATA v2 suites using stdlib HTTP + STRATA's
live GEVR sandbox & SDK auditors.

Security: Reads API keys exclusively from environment variables (`OPENROUTER_API_KEY`,
`SANDBASE_API_KEY`, `ORCAROUTER_API_KEY`, `TEAMOROUTER_API_KEY`) and never writes or
logs credentials into output files or stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
import urllib.error
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
    evaluate_task_response,
    extract_code_clean,
    read_cpu_times,
    read_proc_mem_mb,
)
from benchmarks.skill_docs import (  # noqa: E402
    VALID_CONTEXT_MODES,
    build_messages_for_arm,
)
from benchmarks.suites import get_suite_tasks  # noqa: E402

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

GATEWAY_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "chat_url": OPENROUTER_CHAT_URL,
        "default_api_key_env": "OPENROUTER_API_KEY",
        "arm_prefix": "or",
    },
    "sandbase": {
        "chat_url": "https://api.sandbase.ai/v1/chat/completions",
        "default_api_key_env": "SANDBASE_API_KEY",
        "arm_prefix": "sb",
    },
    "orcarouter": {
        "chat_url": "https://api.orcarouter.ai/v1/chat/completions",
        "default_api_key_env": "ORCAROUTER_API_KEY",
        "arm_prefix": "orca",
    },
    "teamorouter": {
        "chat_url": "https://api.teamorouter.com/v1/chat/completions",
        "default_api_key_env": "TEAMOROUTER_API_KEY",
        "arm_prefix": "teamo",
    },
}

OPENROUTER_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nemotron-3.5-super-vl": {
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3.5-super-vl:free",
        "display": "Nemotron 3.5 Super VL 120B-A12B (OR)",
        "family": "Nemotron-3.5",
        "params_b": 120.0,
        "active_params_b": 12.0,
        "thinking_style": "openrouter_reasoning",
    },
    "gemma-4-31b": {
        "provider": "openrouter",
        "model_id": "google/gemma-4-31b-it:free",
        "display": "Gemma 4 31B IT (OR)",
        "family": "Gemma-4",
        "params_b": 30.7,
        "active_params_b": 30.7,
        "thinking_style": "openrouter_reasoning",
    },
    "deepseek-v4.1-flash-sb": {
        "provider": "sandbase",
        "model_id": "deepseek/deepseek-v4.1-flash:free",
        "display": "DeepSeek-V4.1-Flash 552B-A16B (SB)",
        "family": "DeepSeek-V4.1",
        "params_b": 552.0,
        "active_params_b": 16.0,
        "thinking_style": "deepseek_thinking",
    },
    "orca-deepseek-v4.1-flash": {
        "provider": "orcarouter",
        "model_id": "deepseek/deepseek-v4.1-flash-free",
        "display": "DeepSeek-V4.1-Flash 552B-A16B (Orca)",
        "family": "DeepSeek-V4.1",
        "params_b": 552.0,
        "active_params_b": 16.0,
        "thinking_style": "deepseek_thinking",
    },
    "orca-deepseek-v4-flash": {
        "provider": "orcarouter",
        "model_id": "deepseek/deepseek-v4-flash-free",
        "display": "DeepSeek-V4-Flash 284B-A13B (Orca)",
        "family": "DeepSeek-V4",
        "params_b": 284.0,
        "active_params_b": 13.0,
        "thinking_style": "deepseek_thinking",
    },
    "orca-hy3": {
        "provider": "orcarouter",
        "model_id": "tencent/hy3-free",
        "display": "Hunyuan Hy3 295B-A21B (Orca)",
        "family": "Hunyuan-3",
        "params_b": 295.0,
        "active_params_b": 21.0,
        "thinking_style": "reasoning_effort",
    },
    "orca-hy4-preview": {
        "provider": "orcarouter",
        "model_id": "tencent/hy4-preview-free",
        "display": "Hunyuan Hy4 Preview 770B-A49B (Orca)",
        "family": "Hunyuan-4",
        "params_b": 770.0,
        "active_params_b": 49.0,
        "thinking_style": "reasoning_effort",
    },
    "orca-glm-5.3-flash": {
        "provider": "orcarouter",
        "model_id": "z-ai/glm-5.3-flash-free",
        "display": "GLM 5.3 Flash 320B-A18B (Orca)",
        "family": "GLM-5.3",
        "params_b": 320.0,
        "active_params_b": 18.0,
        "thinking_style": "openrouter_reasoning",
    },
    "teamo-deepseek-v4.1-flash": {
        "provider": "teamorouter",
        "model_id": "deepseek-flash-free",
        "display": "DeepSeek-V4.1-Flash 552B-A16B (Teamo)",
        "family": "DeepSeek-V4.1",
        "params_b": 552.0,
        "active_params_b": 16.0,
        "thinking_style": "deepseek_thinking",
    },
    "teamo-deepseek-v4-flash": {
        "provider": "teamorouter",
        "model_id": "deepseek-v4-flash-free",
        "display": "DeepSeek-V4-Flash 284B-A13B (Teamo)",
        "family": "DeepSeek-V4",
        "params_b": 284.0,
        "active_params_b": 13.0,
        "thinking_style": "deepseek_thinking",
    },
    "teamo-glm-5.3-flash": {
        "provider": "teamorouter",
        "model_id": "glm-5.3-flash-free",
        "display": "GLM 5.3 Flash 320B-A18B (Teamo)",
        "family": "GLM-5.3",
        "params_b": 320.0,
        "active_params_b": 18.0,
        "thinking_style": "openrouter_reasoning",
    },
}

PRIORITY_CELLS_ORDER: List[Tuple[float, bool, Optional[int]]] = [
    (0.0, True, 4096),
    (0.0, False, 4096),
    (1.0, True, 4096),
    (0.0, True, None),
    (1.0, True, None),
    (1.0, False, 4096),
    (0.0, False, None),
    (1.0, False, None),
]

MODE_DISPLAY_TAGS: Dict[str, str] = {
    "pure": "Pure",
    "skill_md": "SKILL.md",
    "skill_md_haystack": "SKILL.md+Haystack",
}

TRANSIENT_INCOMPLETE_STATUSES = {"TIMEOUT", "API_ERROR", "RATE_LIMITED"}


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when OpenRouter signals an account-level daily free-tier or credit cap."""


class ModelUpstreamSaturatedError(RuntimeError):
    """Raised when repeated upstream failures indicate a single model is currently saturated."""


def sanitize_openrouter_error(text: str, api_key: str) -> str:
    clean = str(text or "")
    if api_key:
        clean = clean.replace(api_key, "***")
    return clean


def _format_temp_tag(temp: float) -> Tuple[str, str]:
    val = float(temp)
    if val.is_integer():
        iv = str(int(val))
        return iv, iv
    s = f"{val:g}"
    return s.replace(".", "p"), s


def _parse_token_limits(raw_limits: List[str]) -> List[Optional[int]]:
    parsed: List[Optional[int]] = []
    for item in raw_limits:
        norm = str(item).strip().lower()
        if norm in ("unlimited", "none", "nolimit", "inf", "-1", "0"):
            if None not in parsed:
                parsed.append(None)
        else:
            try:
                val = int(norm)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid token limit {item!r}; use int or 'unlimited'") from exc
            if val <= 0:
                if None not in parsed:
                    parsed.append(None)
            elif val not in parsed:
                parsed.append(val)
    return parsed


def build_config_grid(
    *,
    temperatures: List[float],
    thinking_modes: List[str],
    token_limits: List[Optional[int]],
    priority_order: bool = True,
) -> List[Tuple[float, bool, Optional[int]]]:
    t_vals = [float(t) for t in temperatures]
    think_vals = [m.lower() == "on" for m in thinking_modes]
    requested: List[Tuple[float, bool, Optional[int]]] = []
    for t in t_vals:
        for th in think_vals:
            for lim in token_limits:
                cell = (t, th, lim)
                if cell not in requested:
                    requested.append(cell)
    if not priority_order:
        return requested
    req_set = set(requested)
    ordered = [c for c in PRIORITY_CELLS_ORDER if c in req_set]
    for c in requested:
        if c not in ordered:
            ordered.append(c)
    return ordered


def build_openrouter_payload(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    thinking: bool = True,
    max_tokens: Optional[int] = 4096,
    thinking_style: str = "openrouter_reasoning",
) -> Dict[str, Any]:
    """Build OpenAI-compatible Chat Completions payload with gateway-specific thinking toggle."""
    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": [dict(m) for m in messages],
        "temperature": float(temperature),
        "top_p": 0.95,
    }
    if thinking_style == "deepseek_thinking":
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif thinking_style == "reasoning_effort":
        payload["reasoning_effort"] = "high" if thinking else "none"
    else:
        payload["reasoning"] = {"enabled": bool(thinking)}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    return payload


def parse_openrouter_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content, reasoning traces, finish reason, and token counts from OpenRouter JSON."""
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning") or msg.get("reasoning_content") or "")

    content_s = content.strip()
    reasoning_s = reasoning.strip()
    if reasoning_s and not content_s.startswith("<think>"):
        raw_reply = f"<think>{reasoning_s}</think>\n{content_s}".strip()
    else:
        raw_reply = content_s

    usage = data.get("usage") or {}
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    completion_tok = int(usage.get("completion_tokens") or 0)
    total_tok = int(usage.get("total_tokens") or (prompt_tok + completion_tok))
    comp_details = usage.get("completion_tokens_details") or {}
    thoughts_tok = int(comp_details.get("reasoning_tokens") or 0)

    return {
        "raw_reply": raw_reply,
        "thought_text": reasoning_s,
        "answer_text": content_s,
        "finish_reason": str(choice.get("finish_reason") or ""),
        "prompt_tokens": prompt_tok,
        "thoughts_tokens": thoughts_tok,
        "completion_tokens": completion_tok,
        "total_tokens": total_tok,
    }


def is_openrouter_daily_quota_error(http_code: int, err_body: str) -> bool:
    """Distinguish account-level daily free-tier or balance limits from transient upstream 429s."""
    if http_code not in (400, 401, 402, 403, 429):
        return False
    low = str(err_body or "").lower()
    if "temporarily rate-limited upstream" in low:
        return False
    return any(
        marker in low
        for marker in (
            "free-models-per-day",
            "requests per day",
            "per-day",
            "per day",
            "daily limit",
            "daily free api limit exceeded",
            "insufficient credits",
            "err_free_access_denied",
            "free_quota_exhausted",
            "enterprise_insufficient_balance",
            "insufficient_balance",
        )
    )


def call_openrouter_chat(
    *,
    api_model_id: str,
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
    timeout_s: float,
    max_transient_retries: int,
    endpoint_url: str = OPENROUTER_CHAT_URL,
    thinking_style: str = "openrouter_reasoning",
) -> Dict[str, Any]:
    payload = build_openrouter_payload(
        messages,
        model=api_model_id,
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
        thinking_style=thinking_style,
    )
    body_bytes = json.dumps(payload).encode("utf-8")
    transient_backoff = [3.0, 6.0, 12.0, 20.0, 30.0]
    attempts = 0
    retry_cap = max(0, int(max_transient_retries))

    while True:
        req = urllib.request.Request(
            endpoint_url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "strata-bench/2.0",
                "X-Title": "STRATA-v2-Benchmark",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_json = resp.read().decode("utf-8", errors="replace")
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            data = json.loads(raw_json)
            if "error" in data and not data.get("choices"):
                err_obj = data["error"]
                err_code = int((err_obj or {}).get("code") or 500) if isinstance(err_obj, dict) else 500
                err_msg = sanitize_openrouter_error(json.dumps(err_obj), api_key)
                if is_openrouter_daily_quota_error(err_code, err_msg):
                    raise DailyQuotaExhaustedError(f"Gateway quota/balance limit ({err_code}): {err_msg[:260]}")
                if err_code in (429, 500, 502, 503, 504) and attempts < retry_cap:
                    wait_s = transient_backoff[min(attempts, len(transient_backoff) - 1)] + random.uniform(0.0, 1.5)
                    attempts += 1
                    print(
                        f"  [API][RETRY {attempts}/{retry_cap}] Error {err_code}; sleeping {wait_s:.1f}s ...",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Gateway error payload ({err_code}): {err_msg[:300]}")
            parsed = parse_openrouter_response(data)
            parsed["wall_ms"] = dt_ms
            return parsed
        except urllib.error.HTTPError as exc:
            err_body = sanitize_openrouter_error(exc.read().decode("utf-8", errors="replace"), api_key)
            if is_openrouter_daily_quota_error(exc.code, err_body):
                raise DailyQuotaExhaustedError(f"Gateway quota/balance limit (HTTP {exc.code}): {err_body[:260]}") from exc
            if exc.code in (429, 500, 502, 503, 504) and attempts < retry_cap:
                wait_s = transient_backoff[min(attempts, len(transient_backoff) - 1)] + random.uniform(0.0, 1.5)
                attempts += 1
                print(
                    f"  [API][RETRY {attempts}/{retry_cap}] HTTP {exc.code}; sleeping {wait_s:.1f}s ...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {err_body[:300]}") from exc
        except DailyQuotaExhaustedError:
            raise
        except Exception as exc:
            msg = sanitize_openrouter_error(str(exc), api_key)
            if attempts < retry_cap:
                wait_s = transient_backoff[min(attempts, len(transient_backoff) - 1)] + random.uniform(0.0, 1.5)
                attempts += 1
                print(
                    f"  [API][RETRY {attempts}/{retry_cap}] {type(exc).__name__}: {msg[:120]}; sleeping {wait_s:.1f}s ...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(msg) from exc


def _build_cell_metadata(
    *,
    model_key: str,
    spec: Dict[str, Any],
    context_mode: str,
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    t_file, t_disp = _format_temp_tag(temperature)
    think_file = "think" if thinking else "nothink"
    think_disp = "Think" if thinking else "NoThink"
    lim_file = "unlim" if max_tokens is None else f"lim{int(max_tokens)}"
    lim_disp = "NoLimit" if max_tokens is None else f"Max{int(max_tokens)}"
    ctx_disp = MODE_DISPLAY_TAGS[context_mode]

    provider = str(spec.get("provider") or "openrouter")
    gw = GATEWAY_ENDPOINTS.get(provider, GATEWAY_ENDPOINTS["openrouter"])
    arm_id = f"{gw['arm_prefix']}_{context_mode}_t{t_file}_{think_file}_{lim_file}"
    model_id = f"{model_key}_{context_mode}_t{t_file}_{think_file}_{lim_file}"
    display = f"{spec['display']} [{ctx_disp} | T={t_disp} | {think_disp} | {lim_disp}]"
    filename = f"results_{model_key}_{context_mode}_t{t_file}_{think_file}_{lim_file}.json"
    return {
        "arm_id": arm_id,
        "model_id": model_id,
        "display": display,
        "filename": filename,
    }


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
            status = str(rec.get("status") or "")
            if tid and status not in TRANSIENT_INCOMPLETE_STATUSES:
                existing[tid] = rec
        return existing
    except Exception:
        return {}


def _write_cell_checkpoint(
    out_path: Path,
    *,
    meta: Dict[str, Any],
    spec: Dict[str, Any],
    context_mode: str,
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
    haystack_chars: int,
    full_haystack: bool,
    records_by_id: Dict[str, Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> None:
    ordered_records = [records_by_id[t["id"]] for t in tasks if t["id"] in records_by_id]
    payload = {
        "schema_version": "strata_bench_v2",
        "arm": meta["arm_id"],
        "model_id": meta["model_id"],
        "model_display": meta["display"],
        "family": spec["family"],
        "params_b": float(spec["params_b"]),
        "active_params_b": float(spec["active_params_b"]),
        "quant": "API/Free",
        "init": {
            "provider": str(spec.get("provider") or "openrouter"),
            "api_model": spec["model_id"],
            "context_mode": context_mode,
            "temperature": float(temperature),
            "thinking": bool(thinking),
            "max_tokens": max_tokens,
            "haystack_chars": int(haystack_chars),
            "full_haystack": bool(full_haystack),
        },
        "records": ordered_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cell(
    *,
    model_key: str,
    spec: Dict[str, Any],
    api_key: str,
    tasks: List[Dict[str, Any]],
    context_mode: str,
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
    out_dir: Path,
    resume: bool,
    min_interval_s: float,
    max_transient_retries: int,
    timeout_s: float,
    haystack_chars: int,
    full_haystack: bool,
) -> Path:
    meta = _build_cell_metadata(
        model_key=model_key,
        spec=spec,
        context_mode=context_mode,
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
    )
    out_path = out_dir / meta["filename"]
    records_by_id: Dict[str, Dict[str, Any]] = _load_existing_completed(out_path) if resume else {}

    if resume and len(records_by_id) >= len(tasks):
        print(f"[OR] Skipping completed cell (--resume): {out_path.name} ({len(records_by_id)}/{len(tasks)})", flush=True)
        return out_path

    print(
        f"\n==================== {meta['display']} ({len(records_by_id)}/{len(tasks)} cached) ====================",
        flush=True,
    )
    consecutive_failures = 0
    last_call_ts = 0.0

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
        if last_call_ts > 0.0:
            elapsed = time.time() - last_call_ts
            if elapsed < float(min_interval_s):
                time.sleep(float(min_interval_s) - elapsed)

        cpu0 = read_cpu_times()
        t0 = time.perf_counter()
        provider = str(spec.get("provider") or "openrouter")
        gw = GATEWAY_ENDPOINTS.get(provider, GATEWAY_ENDPOINTS["openrouter"])
        endpoint_url = gw["chat_url"]
        thinking_style = str(spec.get("thinking_style") or "openrouter_reasoning")
        try:
            last_call_ts = time.time()
            or_res = call_openrouter_chat(
                api_model_id=spec["model_id"],
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_transient_retries=max_transient_retries,
                endpoint_url=endpoint_url,
                thinking_style=thinking_style,
            )
            consecutive_failures = 0
            raw_reply = or_res["raw_reply"]
            code = extract_code_clean(raw_reply)
            ev = evaluate_task_response(raw_reply, code, task, timeout=45.0)
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            rec = {
                "arm": meta["arm_id"],
                "model_id": meta["model_id"],
                "model_display": meta["display"],
                "family": spec["family"],
                "params_b": float(spec["params_b"]),
                "active_params_b": float(spec["active_params_b"]),
                "quant": "API/Free",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "thinking": bool(thinking),
                "max_tokens": max_tokens,
                "finish_reason": or_res.get("finish_reason", ""),
                "expected": task.get("expected", "synthesize"),
                "prompt": task["prompt"],
                "status": ev["status"],
                "passed": ev["passed"],
                "acceptable": ev["acceptable"],
                "detail": ev["detail"],
                "wall_ms": wall_ms,
                "llm_calls": 1,
                "prompt_tokens": or_res["prompt_tokens"],
                "thoughts_tokens": or_res["thoughts_tokens"],
                "completion_tokens": or_res["completion_tokens"],
                "total_tokens": or_res["total_tokens"],
                "llm_wall_ms": or_res["wall_ms"],
                "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                "mem_mb": read_proc_mem_mb(),
                "gpu_mb": {"total_used_mb": None, "proc_used_mb": None},
                "audit": ev.get("audit"),
                "code": code,
                "raw_reply_preview": raw_reply[:1200],
            }
            records_by_id[tid] = rec
            _write_cell_checkpoint(
                out_path,
                meta=meta,
                spec=spec,
                context_mode=context_mode,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                haystack_chars=haystack_chars,
                full_haystack=full_haystack,
                records_by_id=records_by_id,
                tasks=tasks,
            )
            print(
                f"  [{idx:02d}/{len(tasks):02d}] {tid:<30} | {rec['status']:<18} "
                f"| acc={str(rec['acceptable']):<5} | {wall_ms / 1000.0:5.1f}s "
                f"| tok={rec['prompt_tokens']}+{rec['thoughts_tokens']}th+{rec['completion_tokens']}",
                flush=True,
            )
        except DailyQuotaExhaustedError:
            _write_cell_checkpoint(
                out_path,
                meta=meta,
                spec=spec,
                context_mode=context_mode,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                haystack_chars=haystack_chars,
                full_haystack=full_haystack,
                records_by_id=records_by_id,
                tasks=tasks,
            )
            raise
        except Exception as exc:
            consecutive_failures += 1
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            err_msg = sanitize_openrouter_error(str(exc), api_key)
            rec = {
                "arm": meta["arm_id"],
                "model_id": meta["model_id"],
                "model_display": meta["display"],
                "family": spec["family"],
                "params_b": float(spec["params_b"]),
                "active_params_b": float(spec["active_params_b"]),
                "quant": "API/Free",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "thinking": bool(thinking),
                "max_tokens": max_tokens,
                "expected": task.get("expected", "synthesize"),
                "prompt": task["prompt"],
                "status": "API_ERROR",
                "passed": False,
                "acceptable": False,
                "detail": f"Request failed: {err_msg}",
                "wall_ms": wall_ms,
                "llm_calls": 1,
                "prompt_tokens": 0,
                "thoughts_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_wall_ms": 0.0,
                "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                "mem_mb": read_proc_mem_mb(),
                "gpu_mb": {"total_used_mb": None, "proc_used_mb": None},
                "audit": None,
                "code": "",
                "raw_reply_preview": "",
            }
            records_by_id[tid] = rec
            _write_cell_checkpoint(
                out_path,
                meta=meta,
                spec=spec,
                context_mode=context_mode,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                haystack_chars=haystack_chars,
                full_haystack=full_haystack,
                records_by_id=records_by_id,
                tasks=tasks,
            )
            print(f"  [{idx:02d}/{len(tasks):02d}] {tid:<30} | API_ERROR | {err_msg[:90]}", flush=True)
            if consecutive_failures >= 3:
                raise ModelUpstreamSaturatedError(
                    f"Model '{model_key}' saturated ({consecutive_failures} consecutive API errors): {err_msg[:180]}"
                ) from exc

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="STRATA Multi-Gateway Free-Tier Decoding Ablation Runner")
    ap.add_argument(
        "--models",
        nargs="+",
        choices=list(OPENROUTER_MODEL_REGISTRY.keys()),
        default=["nemotron-3.5-super-vl", "gemma-4-31b"],
        help="Registered free-tier model keys to evaluate sequentially",
    )
    ap.add_argument(
        "--api-key-env",
        default=None,
        help="Optional override env var holding gateway API key (defaults per model provider)",
    )
    ap.add_argument("--suite", default="v2_expanded", help="Suite name from benchmarks/suites.py")
    ap.add_argument(
        "--context-modes",
        nargs="+",
        choices=list(VALID_CONTEXT_MODES),
        default=["skill_md", "pure"],
        help="Context arms to evaluate sequentially",
    )
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 1.0], help="Temperatures to sweep")
    ap.add_argument(
        "--thinking-modes",
        nargs="+",
        choices=["on", "off"],
        default=["on", "off"],
        help="Reasoning enabled vs disabled",
    )
    ap.add_argument(
        "--token-limits",
        nargs="+",
        default=["4096", "unlimited"],
        help="Max completion tokens (int or 'unlimited')",
    )
    ap.add_argument(
        "--priority-order",
        dest="priority_order",
        action="store_true",
        default=True,
        help="Order decoding cells by scientific priority first (default)",
    )
    ap.add_argument(
        "--no-priority-order",
        dest="priority_order",
        action="store_false",
        help="Keep cartesian grid order instead of priority order",
    )
    ap.add_argument("--haystack-chars", type=int, default=18000, help="Char budget for SKILL.md catalog")
    ap.add_argument("--full-haystack", action="store_true", help="Include every cell without truncation")
    ap.add_argument("--out-dir", default="test_logs/openrouter_v2", help="Output directory for JSON results")
    ap.add_argument("--resume", action="store_true", help="Skip completed tasks in existing output JSONs")
    ap.add_argument("--min-interval-s", type=float, default=3.5, help="Pacing interval in seconds (respects 20 RPM cap)")
    ap.add_argument("--max-transient-retries", type=int, default=12, help="Max fast retries on upstream 429/5xx")
    ap.add_argument("--timeout", type=float, default=240.0, help="HTTP request timeout in seconds")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    resolved_keys: Dict[str, str] = {}
    for mkey in args.models:
        spec = OPENROUTER_MODEL_REGISTRY[mkey]
        provider = str(spec.get("provider") or "openrouter")
        gw = GATEWAY_ENDPOINTS.get(provider, GATEWAY_ENDPOINTS["openrouter"])
        env_name = str(args.api_key_env or gw["default_api_key_env"])
        val = str(os.environ.get(env_name) or "").strip()
        if not val:
            print(
                f"[API][ERROR] Missing gateway API key in environment variable {env_name!r} for model {mkey!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        resolved_keys[mkey] = val

    token_limits = _parse_token_limits(args.token_limits)
    cells = build_config_grid(
        temperatures=args.temperatures,
        thinking_modes=args.thinking_modes,
        token_limits=token_limits,
        priority_order=bool(args.priority_order),
    )
    tasks = get_suite_tasks(args.suite)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[API] Models={args.models} | Suite={args.suite} ({len(tasks)} tasks) "
        f"| Contexts={args.context_modes} | DecodingCells={len(cells)}",
        flush=True,
    )

    saturated_models: List[str] = []
    last_key = ""
    try:
        for mkey in args.models:
            spec = OPENROUTER_MODEL_REGISTRY[mkey]
            api_key = resolved_keys[mkey]
            last_key = api_key
            try:
                for ctx_mode in args.context_modes:
                    for temp, thinking, max_tok in cells:
                        run_cell(
                            model_key=mkey,
                            spec=spec,
                            api_key=api_key,
                            tasks=tasks,
                            context_mode=ctx_mode,
                            temperature=temp,
                            thinking=thinking,
                            max_tokens=max_tok,
                            out_dir=out_dir,
                            resume=bool(args.resume),
                            min_interval_s=float(args.min_interval_s),
                            max_transient_retries=int(args.max_transient_retries),
                            timeout_s=float(args.timeout),
                            haystack_chars=int(args.haystack_chars),
                            full_haystack=bool(args.full_haystack),
                        )
            except ModelUpstreamSaturatedError as sat_exc:
                print(f"\n[API][MODEL_SKIP] {sat_exc} — advancing to next model.", file=sys.stderr, flush=True)
                saturated_models.append(mkey)
    except DailyQuotaExhaustedError as q_exc:
        print(f"\n[API][QUOTA_HALT] {q_exc} (checkpoints preserved for --resume)", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as exc:
        print(f"\n[API][ERROR] {sanitize_openrouter_error(str(exc), last_key)}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    if saturated_models and len(saturated_models) == len(args.models):
        print(f"\n[OR] All requested models hit upstream saturation: {saturated_models}", file=sys.stderr, flush=True)
        sys.exit(3)
    if saturated_models:
        print(f"\n[OR] Completed with partial upstream skip on: {saturated_models}")
    else:
        print("\n[OR] All requested OpenRouter sweeps finished cleanly!")


if __name__ == "__main__":
    main()
