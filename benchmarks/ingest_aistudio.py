#!/usr/bin/env python3
"""
benchmarks/ingest_aistudio.py - Export prompt packs & verify responses from Google AI Studio
(e.g., Gemma 4 31B IT) against STRATA's exact sandbox and SDK typestate verifiers.

Modes:
  1. --export-prompts <dir> : Generate copy-pasteable per-task markdown packs for
     `pure`, `skill_md`, or `skill_md_haystack` arms across any suite in benchmarks/suites.py.
  2. --input <file>         : Verify JSON or `=== TASK: <id> ===` delimited raw responses
     into standardized `strata_bench_v2` result JSON.
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
from benchmarks.skill_docs import VALID_CONTEXT_MODES, build_messages_for_arm  # noqa: E402
from benchmarks.suites import get_suite_tasks  # noqa: E402

MODE_TAGS: Dict[str, str] = {
    "pure": "Pure",
    "skill_md": "SKILL.md",
    "skill_md_haystack": "SKILL.md+Haystack",
}


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


def export_prompt_packs(
    *,
    export_dir: Path,
    tasks: List[Dict[str, Any]],
    suite_name: str,
    context_mode: str,
    haystack_chars: int,
    full_haystack: bool,
) -> Path:
    mode_dir = export_dir / context_mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    index_lines: List[str] = [
        f"# AI Studio Prompt Pack — Suite `{suite_name}` | Arm `{context_mode}`",
        "",
        f"- Total tasks: `{len(tasks)}`",
        f"- Full haystack (untruncated): `{full_haystack}` | Haystack char budget: `{haystack_chars}`",
        "",
        "| # | Task ID | Group | Expected | DomainSet | Docs Ops | Prompt Chars | Pack File |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]

    for idx, task in enumerate(tasks, 1):
        messages, meta = build_messages_for_arm(
            task,
            context_mode=context_mode,
            max_distractor_chars=haystack_chars,
            full_haystack=full_haystack,
        )
        sys_txt = messages[0]["content"]
        usr_txt = messages[1]["content"]
        fname = f"{idx:02d}_{task['id']}.md"
        fpath = mode_dir / fname

        # Use 5-backtick outer fence safely even if system text contains triple backticks
        doc_lines = [
            f"# Task {idx:02d}: `{task['id']}` — {task.get('name', task['id'])}",
            "",
            f"- **Group:** `{task['group']}` | **Kind:** `{task['kind']}` | **Expected:** `{task.get('expected', 'synthesize')}`",
            f"- **DomainSet:** `{meta['domain_set']}` | **Context Mode:** `{context_mode}` | **Ops Documented:** `{meta['ops_documented']}` / `{meta['total_lattice_cells']}`",
            "",
            "## System Instructions (Copy verbatim into Google AI Studio -> System Instructions)",
            "",
            "`````text",
            sys_txt,
            "`````",
            "",
            "## User Prompt (Copy verbatim into Fresh Chat)",
            "",
            "`````text",
            usr_txt,
            "`````",
            "",
            "## Response Capture Wrapper (Paste model output inside for ingestion)",
            "",
            "`````text",
            f"=== TASK: {task['id']} ===",
            "<paste raw response here>",
            "=== END ===",
            "`````",
            "",
        ]
        fpath.write_text("\n".join(doc_lines), encoding="utf-8")
        index_lines.append(
            f"| {idx} | `{task['id']}` | `{task['group']}` | `{task.get('expected', 'synthesize')}` | "
            f"`{meta['domain_set']}` | {meta['ops_documented']} | {len(sys_txt)} | [{fname}]({fname}) |"
        )

    idx_path = mode_dir / "ALL_PROMPTS_INDEX.md"
    idx_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Exported {len(tasks)} AI Studio prompt pack(s) -> {mode_dir} (index: {idx_path})")
    return mode_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Export AI Studio Prompt Packs & Verify Responses (Gemma 4 31B)")
    ap.add_argument("--input", default="", help="Path to JSON or === TASK: id === text dump")
    ap.add_argument("--export-prompts", default="", help="Optional output directory to generate per-task System+User markdown packs")
    ap.add_argument("--suite", default="v2_expanded", help="Suite name from benchmarks/suites.py")
    ap.add_argument(
        "--context-mode",
        choices=list(VALID_CONTEXT_MODES),
        default="skill_md",
        help="Evaluation arm tag: pure, skill_md, or skill_md_haystack",
    )
    ap.add_argument("--haystack-chars", type=int, default=18000, help="Char budget when exporting multi-domain SKILL.md")
    ap.add_argument("--full-haystack", action="store_true", help="Include every cell without truncation (ideal for 1M-context AI Studio)")
    ap.add_argument("--model-id", default="gemma-4-31b-it")
    ap.add_argument("--model-display", default="Gemma 4 31B IT (AI Studio)")
    ap.add_argument("--params-b", type=float, default=30.7)
    ap.add_argument("--out", default="test_logs/colab_bench/results_gemma4_31b.json")
    args = ap.parse_args()

    if not args.input and not args.export_prompts:
        ap.error("Provide at least one of --export-prompts <dir> or --input <responses_file>.")

    tasks = get_suite_tasks(args.suite)

    if args.export_prompts:
        export_prompt_packs(
            export_dir=Path(args.export_prompts),
            tasks=tasks,
            suite_name=args.suite,
            context_mode=args.context_mode,
            haystack_chars=int(args.haystack_chars),
            full_haystack=bool(args.full_haystack),
        )
        if not args.input:
            return

    responses = load_responses(Path(args.input))
    mode_tag = MODE_TAGS[args.context_mode]
    display = args.model_display
    if f"[{mode_tag}]" not in display:
        display = f"{display} [{mode_tag}]"
    arm_id = f"ai_studio_{args.context_mode}"
    model_key = f"{args.model_id}_{args.context_mode}"
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
                "arm": arm_id,
                "model_id": model_key,
                "model_display": display,
                "family": "Gemma-4",
                "params_b": args.params_b,
                "active_params_b": args.params_b,
                "quant": "BF16/API",
                "task_id": tid,
                "group": task["group"],
                "kind": task["kind"],
                "domain_set": task.get("domain_set", "default"),
                "context_mode": args.context_mode,
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
        print(f"[{model_key}] {tid:<30} -> {ev['status']:<18} | acc={ev['acceptable']} | {ev['detail'][:80]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "arm": arm_id,
                "model_id": model_key,
                "model_display": display,
                "family": "Gemma-4",
                "params_b": args.params_b,
                "active_params_b": args.params_b,
                "quant": "BF16/API",
                "context_mode": args.context_mode,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved verified AI Studio results ({len(records)} tasks) -> {out_path}")


if __name__ == "__main__":
    main()
