#!/usr/bin/env python3
"""
benchmarks/gemini_api_eval.py - Google Generative Language REST API benchmark harness.

Runs controlled decoding ablations (temperature x thinking level x output token limit)
for hosted models such as Gemma 4 31B IT (`gemma-4-31b-it`) against STRATA v2's
expanded suite using stdlib HTTP + STRATA's live GEVR sandbox & SDK typestate auditors.

Security: Reads the API key exclusively from an environment variable (default:
`GEMINI_API_KEY`) and never writes or logs credentials into output files or stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

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


class QuotaExhaustedError(RuntimeError):
    """Raised when the remote API signals daily quota exhaustion or repeated rate limits."""


def _sanitize_error_text(text: str, api_key: str) -> str:
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


def _safe_model_slug(model: str) -> str:
    low = model.strip().lower()
    if low == "gemma-4-31b-it":
        return "gemma4_31b"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip()).strip("_") or "gemini_model"


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


def build_gemini_payload(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    thinking: bool = True,
    max_tokens: Optional[int] = 4096,
) -> Dict[str, Any]:
    """Convert OpenAI-style chat messages to Gemini API `generateContent` JSON payload."""
    sys_parts: List[Dict[str, str]] = []
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        text = str(msg.get("content") or "")
        if role == "system":
            if text:
                sys_parts.append({"text": text})
        else:
            api_role = "model" if role == "assistant" else "user"
            contents.append({"role": api_role, "parts": [{"text": text}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    gen_cfg: Dict[str, Any] = {
        "temperature": float(temperature),
        "topP": 0.95,
        "thinkingConfig": {"thinkingLevel": "HIGH" if thinking else "MINIMAL"},
    }
    if max_tokens is not None:
        gen_cfg["maxOutputTokens"] = int(max_tokens)

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": gen_cfg,
    }
    if sys_parts:
        payload["systemInstruction"] = {"parts": sys_parts}
    return payload


def parse_gemini_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract answer text, thinking trace, finish reason, and token counts from API JSON."""
    cand = (data.get("candidates") or [{}])[0] or {}
    parts = (cand.get("content") or {}).get("parts") or []

    thought_parts: List[str] = []
    answer_parts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        txt = str(part.get("text") or "")
        if not txt:
            continue
        if part.get("thought") is True:
            thought_parts.append(txt)
        else:
            answer_parts.append(txt)

    thought_text = "\n".join(thought_parts).strip()
    answer_text = "\n".join(answer_parts).strip()
    raw_reply = f"<think>{thought_text}</think>\n{answer_text}".strip() if thought_text else answer_text

    usage = data.get("usageMetadata") or {}
    prompt_tok = int(usage.get("promptTokenCount") or 0)
    thoughts_tok = int(usage.get("thoughtsTokenCount") or 0)
    cand_tok = int(usage.get("candidatesTokenCount") or 0)
    total_tok = int(usage.get("totalTokenCount") or (prompt_tok + cand_tok + thoughts_tok))
    completion_tok = cand_tok + thoughts_tok

    return {
        "raw_reply": raw_reply,
        "thought_text": thought_text,
        "answer_text": answer_text,
        "finish_reason": str(cand.get("finishReason") or ""),
        "prompt_tokens": prompt_tok,
        "thoughts_tokens": thoughts_tok,
        "candidates_tokens": cand_tok,
        "completion_tokens": completion_tok,
        "total_tokens": total_tok,
    }


def _is_daily_quota_error(http_code: int, err_body: str) -> bool:
    if http_code != 429:
        return False
    low = err_body.lower()
    if "perminute" in low or "per_minute" in low:
        return False
    return any(
        marker in low
        for marker in (
            "perday",
            "per_day",
            "generaterequestsperday",
            "requestsperday",
            "tokensperday",
            "daily limit",
        )
    )


def _extract_retry_delay_s(err_body: str, headers: Any = None) -> Optional[float]:
    if headers is not None:
        ra = headers.get("Retry-After")
        if ra:
            try:
                return max(1.0, float(ra))
            except ValueError:
                pass
    m = re.search(r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"', err_body)
    if m:
        return max(1.0, float(m.group(1)))
    m2 = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_body, flags=re.IGNORECASE)
    if m2:
        return max(1.0, float(m2.group(1)))
    return None


def call_gemini_generate_content(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
    timeout_s: float,
    max_retries: int,
    max_503_retries: int = 20,
) -> Dict[str, Any]:
    url = f"{GEMINI_API_BASE}/{model}:generateContent"
    payload = build_gemini_payload(
        messages,
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
    )
    body_bytes = json.dumps(payload).encode("utf-8")
    rate_backoff = [15.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    overload_backoff = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0]

    rate_attempts = 0
    overload_attempts = 0
    while True:
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": api_key,
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_json = resp.read().decode("utf-8", errors="replace")
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            parsed = parse_gemini_response(json.loads(raw_json))
            parsed["wall_ms"] = dt_ms
            return parsed
        except urllib.error.HTTPError as exc:
            err_body = _sanitize_error_text(exc.read().decode("utf-8", errors="replace"), api_key)
            if _is_daily_quota_error(exc.code, err_body):
                raise QuotaExhaustedError(f"Daily Gemini API quota exhausted (HTTP {exc.code}): {err_body[:260]}") from exc
            if exc.code in (500, 502, 503, 504) and overload_attempts < max(0, int(max_503_retries)):
                wait_s = overload_backoff[min(overload_attempts, len(overload_backoff) - 1)]
                overload_attempts += 1
                print(
                    f"  [API][503-RETRY {overload_attempts}/{max_503_retries}] HTTP {exc.code}; sleeping {wait_s:.1f}s ...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            if exc.code == 429 and rate_attempts < max(0, int(max_retries)):
                sched_s = rate_backoff[min(rate_attempts, len(rate_backoff) - 1)]
                hint_s = _extract_retry_delay_s(err_body, exc.headers)
                wait_s = max(sched_s, hint_s + 1.5) if hint_s else sched_s
                rate_attempts += 1
                print(
                    f"  [API][429-RETRY {rate_attempts}/{max_retries}] HTTP 429; sleeping {wait_s:.1f}s ...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {err_body[:300]}") from exc
        except Exception as exc:
            msg = _sanitize_error_text(str(exc), api_key)
            if rate_attempts < max(0, int(max_retries)):
                wait_s = rate_backoff[min(rate_attempts, len(rate_backoff) - 1)]
                rate_attempts += 1
                print(
                    f"  [API][NET-RETRY {rate_attempts}/{max_retries}] {type(exc).__name__}: {msg[:120]}; sleeping {wait_s:.1f}s ...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(msg) from exc


def _build_cell_metadata(
    *,
    model: str,
    model_display_base: str,
    params_b: float,
    context_mode: str,
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    safe_model = _safe_model_slug(model)
    t_file, t_disp = _format_temp_tag(temperature)
    think_file = "think" if thinking else "nothink"
    think_disp = "Think" if thinking else "NoThink"
    lim_file = "unlim" if max_tokens is None else f"lim{int(max_tokens)}"
    lim_disp = "NoLimit" if max_tokens is None else f"Max{int(max_tokens)}"
    ctx_disp = MODE_DISPLAY_TAGS[context_mode]

    arm_id = f"api_{context_mode}_t{t_file}_{think_file}_{lim_file}"
    model_key = f"{safe_model}_{context_mode}_t{t_file}_{think_file}_{lim_file}"
    display = f"{model_display_base} [{ctx_disp} | T={t_disp} | {think_disp} | {lim_disp}]"
    filename = f"results_{safe_model}_{context_mode}_t{t_file}_{think_file}_{lim_file}.json"
    return {
        "arm_id": arm_id,
        "model_key": model_key,
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
    model: str,
    params_b: float,
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
        "model_id": meta["model_key"],
        "model_display": meta["display"],
        "family": "Gemma-4" if "gemma-4" in model.lower() else "Gemini-API",
        "params_b": float(params_b),
        "active_params_b": float(params_b),
        "quant": "BF16/API",
        "init": {
            "api_model": model,
            "context_mode": context_mode,
            "temperature": float(temperature),
            "thinking": bool(thinking),
            "thinking_level": "HIGH" if thinking else "MINIMAL",
            "max_output_tokens": max_tokens,
            "haystack_chars": int(haystack_chars),
            "full_haystack": bool(full_haystack),
        },
        "records": ordered_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cell(
    *,
    model: str,
    model_display_base: str,
    params_b: float,
    api_key: str,
    tasks: List[Dict[str, Any]],
    context_mode: str,
    temperature: float,
    thinking: bool,
    max_tokens: Optional[int],
    out_dir: Path,
    resume: bool,
    min_interval_s: float,
    max_retries: int,
    max_503_retries: int,
    timeout_s: float,
    haystack_chars: int,
    full_haystack: bool,
) -> Path:
    meta = _build_cell_metadata(
        model=model,
        model_display_base=model_display_base,
        params_b=params_b,
        context_mode=context_mode,
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
    )
    out_path = out_dir / meta["filename"]
    records_by_id: Dict[str, Dict[str, Any]] = _load_existing_completed(out_path) if resume else {}

    if resume and len(records_by_id) >= len(tasks):
        print(f"[API] Skipping completed cell (--resume): {out_path.name} ({len(records_by_id)}/{len(tasks)})", flush=True)
        return out_path

    print(
        f"\n==================== {meta['display']} ({len(records_by_id)}/{len(tasks)} cached) ====================",
        flush=True,
    )
    consecutive_api_failures = 0
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
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        req_interval = max(float(min_interval_s), 12.0) if prompt_chars > 12000 else float(min_interval_s)
        if prompt_chars > 100000:
            req_interval = max(req_interval, 25.0)
        if last_call_ts > 0.0:
            elapsed = time.time() - last_call_ts
            if elapsed < req_interval:
                time.sleep(req_interval - elapsed)

        cpu0 = read_cpu_times()
        t0 = time.perf_counter()
        raw_reply = ""
        code = ""
        api_res: Dict[str, Any] = {}
        try:
            last_call_ts = time.time()
            api_res = call_gemini_generate_content(
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
                max_503_retries=max_503_retries,
            )
            consecutive_api_failures = 0
            raw_reply = api_res["raw_reply"]
            code = extract_code_clean(raw_reply)
            ev = evaluate_task_response(raw_reply, code, task, timeout=45.0)
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            rec = {
                "arm": meta["arm_id"],
                "model_id": meta["model_key"],
                "model_display": meta["display"],
                "family": "Gemma-4" if "gemma-4" in model.lower() else "Gemini-API",
                "params_b": float(params_b),
                "active_params_b": float(params_b),
                "quant": "BF16/API",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "thinking": bool(thinking),
                "max_output_tokens": max_tokens,
                "finish_reason": api_res.get("finish_reason", ""),
                "expected": task.get("expected", "synthesize"),
                "prompt": task["prompt"],
                "status": ev["status"],
                "passed": ev["passed"],
                "acceptable": ev["acceptable"],
                "detail": ev["detail"],
                "wall_ms": wall_ms,
                "llm_calls": 1,
                "prompt_tokens": api_res["prompt_tokens"],
                "thoughts_tokens": api_res["thoughts_tokens"],
                "candidates_tokens": api_res["candidates_tokens"],
                "completion_tokens": api_res["completion_tokens"],
                "total_tokens": api_res["total_tokens"],
                "llm_wall_ms": api_res["wall_ms"],
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
                model=model,
                params_b=params_b,
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
                f"| tok={rec['prompt_tokens']}+{rec['thoughts_tokens']}th+{rec['candidates_tokens']}",
                flush=True,
            )
        except QuotaExhaustedError:
            _write_cell_checkpoint(
                out_path,
                meta=meta,
                model=model,
                params_b=params_b,
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
            consecutive_api_failures += 1
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            err_msg = _sanitize_error_text(str(exc), api_key)
            rec = {
                "arm": meta["arm_id"],
                "model_id": meta["model_key"],
                "model_display": meta["display"],
                "family": "Gemma-4" if "gemma-4" in model.lower() else "Gemini-API",
                "params_b": float(params_b),
                "active_params_b": float(params_b),
                "quant": "BF16/API",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": ctx_meta["domain_set"],
                "context_mode": context_mode,
                "context_meta": ctx_meta,
                "temperature": float(temperature),
                "thinking": bool(thinking),
                "max_output_tokens": max_tokens,
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
                "candidates_tokens": 0,
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
                model=model,
                params_b=params_b,
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
            if consecutive_api_failures >= 3:
                raise QuotaExhaustedError(
                    f"Aborting cell after {consecutive_api_failures} consecutive API failures: {err_msg[:200]}"
                ) from exc

    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="STRATA Gemini/Gemma REST API Decoding Ablation Runner")
    ap.add_argument("--model", default="gemma-4-31b-it", help="Gemini API model id")
    ap.add_argument("--model-display", default="Gemma 4 31B IT (API)", help="Base display name")
    ap.add_argument("--params-b", type=float, default=30.7, help="Model total parameters in billions")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Env var holding the API key")
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
        help="Thinking modes: 'on' (HIGH) vs 'off' (MINIMAL)",
    )
    ap.add_argument(
        "--token-limits",
        nargs="+",
        default=["4096", "unlimited"],
        help="Output token limits (integer or 'unlimited')",
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
        help="Keep cartesian grid order instead of scientific priority order",
    )
    ap.add_argument("--haystack-chars", type=int, default=18000, help="Char budget for SKILL.md catalog")
    ap.add_argument("--full-haystack", action="store_true", help="Include every cell without truncation")
    ap.add_argument("--out-dir", default="test_logs/gemma4_api_v2", help="Output directory for JSON results")
    ap.add_argument("--resume", action="store_true", help="Skip completed tasks in existing output JSONs")
    ap.add_argument("--min-interval-s", type=float, default=3.0, help="Min seconds between calls (scaled up to >=12s for >12K char prompts)")
    ap.add_argument("--max-retries", type=int, default=6, help="Max backoff retries on 429/network errors")
    ap.add_argument("--max-503-retries", type=int, default=20, help="Max fast retries on transient HTTP 500/502/503/504 overload")
    ap.add_argument("--timeout", type=float, default=300.0, help="HTTP request timeout in seconds")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    api_key = str(os.environ.get(args.api_key_env) or "").strip()
    if not api_key:
        print(
            f"[API][ERROR] Missing API key in environment variable {args.api_key_env!r}. Export it before running.",
            file=sys.stderr,
        )
        sys.exit(1)

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
        f"[API] Model={args.model} | Suite={args.suite} ({len(tasks)} tasks) "
        f"| Contexts={args.context_modes} | DecodingCells={len(cells)}",
        flush=True,
    )

    try:
        for ctx_mode in args.context_modes:
            for temp, thinking, max_tok in cells:
                run_cell(
                    model=args.model,
                    model_display_base=args.model_display,
                    params_b=float(args.params_b),
                    api_key=api_key,
                    tasks=tasks,
                    context_mode=ctx_mode,
                    temperature=temp,
                    thinking=thinking,
                    max_tokens=max_tok,
                    out_dir=out_dir,
                    resume=bool(args.resume),
                    min_interval_s=float(args.min_interval_s),
                    max_retries=int(args.max_retries),
                    max_503_retries=int(args.max_503_retries),
                    timeout_s=float(args.timeout),
                    haystack_chars=int(args.haystack_chars),
                    full_haystack=bool(args.full_haystack),
                )
    except QuotaExhaustedError as q_exc:
        print(f"\n[API][QUOTA_HALT] {q_exc} (completed tasks saved for --resume)", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as exc:
        print(f"\n[API][ERROR] {_sanitize_error_text(str(exc), api_key)}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("\n[API] All requested cells finished cleanly!", flush=True)


if __name__ == "__main__":
    main()
