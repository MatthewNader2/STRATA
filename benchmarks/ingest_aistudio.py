#!/usr/bin/env python3
"""
benchmarks/ingest_aistudio.py - Evaluates responses exported/pasted from Google AI Studio
(e.g., Gemma 4 31B IT) against STRATA's exact sandbox and SDK typestate verifiers.

Supported input formats:
  1. JSON object: {"canonical": "```python ... ```", "vague_game": "...", ...}
     or {"records": [{"task_id": "...", "raw_reply": "...", "wall_ms": ..., "total_tokens": ...}]}
  2. Delimited text/markdown file:
     === TASK: canonical ===
     <paste model output>
     === END ===
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.evaluators import evaluate_task_response, extract_code_clean  # noqa: E402
from benchmarks.suites import get_suite_tasks  # noqa: E402


def parse_delimited_text(text: str) -> Dict[str, Dict[str, Any]]:
    pattern = re.compile(
        r"===\s*TASK:\s*([A-Za-z0-9_.\-]+)\s*===\s*\n(.*?)(?====\s*(?:END|TASK:)|\Z)",
        flags=re.DOTALL,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for m in pattern.finditer(text):
        tid = m.group(1).strip()
        body = m.group(2).strip()
        out[tid] = {"raw_reply": body}
    return out


def load_responses(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            return {str(r["task_id"]): r for r in data["records"]}
        if isinstance(data, dict):
            return {
                str(k): (v if isinstance(v, dict) else {"raw_reply": str(v)})
                for k, v in data.items()
            }
    return parse_delimited_text(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest & Verify Google AI Studio Responses (Gemma 4 31B)")
    ap.add_argument("--input", required=True, help="Path to JSON or === TASK: id === text dump")
    ap.add_argument("--suite", default="all", help="Suite name from benchmarks/suites.py")
    ap.add_argument("--model-id", default="gemma-4-31b-it")
    ap.add_argument("--model-display", default="Gemma 4 31B IT (AI Studio)")
    ap.add_argument("--params-b", type=float, default=30.7)
    ap.add_argument("--out", default="test_logs/colab_bench/results_gemma4_31b.json")
    args = ap.parse_args()

    responses = load_responses(Path(args.input))
    tasks = get_suite_tasks(args.suite)
    records: List[Dict[str, Any]] = []

    for task in tasks:
        tid = task["id"]
        if tid not in responses:
            continue
        entry = responses[tid]
        raw_reply = str(entry.get("raw_reply") or entry.get("response") or "")
        code = extract_code_clean(raw_reply)
        ev = evaluate_task_response(raw_reply, code, task, timeout=45.0)
        records.append(
            {
                "arm": "ai_studio",
                "model_id": args.model_id,
                "model_display": args.model_display,
                "family": "Gemma-4",
                "params_b": args.params_b,
                "active_params_b": args.params_b,
                "quant": "BF16/API",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "expected": task.get("expected", "synthesize"),
                "prompt": task["prompt"],
                "status": ev["status"],
                "passed": ev["passed"],
                "acceptable": ev["acceptable"],
                "detail": ev["detail"],
                "wall_ms": float(entry.get("wall_ms") or 0.0),
                "llm_calls": 1,
                "prompt_tokens": int(entry.get("prompt_tokens") or 0),
                "completion_tokens": int(entry.get("completion_tokens") or 0),
                "total_tokens": int(entry.get("total_tokens") or 0),
                "llm_wall_ms": float(entry.get("wall_ms") or 0.0),
                "mem_mb": {"rss_mb": None, "hwm_mb": None},
                "gpu_mb": {"total_used_mb": None, "proc_used_mb": None},
                "audit": ev.get("audit"),
                "code": code,
            }
        )
        print(f"[{args.model_id}] {tid:<18} -> {ev['status']:<16} | acc={ev['acceptable']} | {ev['detail'][:80]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": "ai_studio",
                "model_id": args.model_id,
                "model_display": args.model_display,
                "family": "Gemma-4",
                "params_b": args.params_b,
                "active_params_b": args.params_b,
                "quant": "BF16/API",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved verified AI Studio results ({len(records)} tasks) -> {out_path}")


if __name__ == "__main__":
    main()
