"""
tools/test_cli_pandas_hard.py

Executes hard, multi-step pandas pipeline prompts directly through STRATAInteractiveShell
with zero templates, zero pre-cooked parameters, and zero helper wrappers.
Simulates real user interaction in the CLI.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cli import STRATAInteractiveShell

# Ensure realistic dummy input CSV exists for sandbox verification
DUMMY_CSV = PROJECT_ROOT / "input_data.csv"
if not DUMMY_CSV.exists():
    DUMMY_CSV.write_text(
        "id,name,department,region,sales,salary,age\n"
        "1,Alice,Engineering,North,150.5,85000,32\n"
        "2,Bob,Marketing,South,,62000,28\n"
        "3,Charlie,Engineering,East,200.0,,45\n"
        "4,David,Sales,North,95.2,54000,24\n"
        "5,Eve,Marketing,West,,,39\n"
        "6,Frank,Sales,South,310.0,71000,50\n"
        "7,Grace,Engineering,West,120.0,92000,29\n"
    )

HARD_PROMPTS = [
    (
        "Hard Test 1: Clean, Deduplicate, Fill Missing, Reset Index",
        "read sales data from csv, drop duplicate rows, fill missing values with 0, and reset the index"
    ),
    (
        "Hard Test 2: Grouping, Aggregation & Sorting",
        "load employee records from csv, group by department, calculate total sales sum, and sort values descending"
    ),
    (
        "Hard Test 3: Multi-Stage Filtering, Missing Values & Reset Index",
        "read dataset from csv, drop missing values, sort by age descending, and reset the index"
    ),
    (
        "Hard Test 4: Descriptive Statistics & Export",
        "read transactions from csv, fill missing values with 0, compute descriptive statistics, and save output to csv"
    ),
]


def run_hard_cli_tests():
    print("=" * 80)
    print("STRATA HARD PANDAS EVALUATION VIA CLI (Profile C: Jina-nano + Qwen-1.5B)")
    print("Testing pure natural language prompts through STRATAInteractiveShell.default()")
    print("=" * 80)

    # Initialize the CLI shell exactly as launched via `python3 src/cli.py shell --profile C ...`
    shell = STRATAInteractiveShell(
        initial_profile="C",
        embedder="jina-embeddings-v5-text-nano",
        llm="qwen2.5-coder-1.5b-instruct"
    )

    results = []

    for label, prompt in HARD_PROMPTS:
        print("\n" + "=" * 80)
        print(f"[*] {label}")
        print(f"    Prompt: \"{prompt}\"")
        print("-" * 80)

        t0 = time.perf_counter()
        # Feed prompt directly into the CLI's default command handler
        shell.default(prompt)
        dt = (time.perf_counter() - t0) * 1000.0

        # Retrieve the latest item from shell history
        latest = shell.history[-1] if shell.history else {}
        results.append({
            "label": label,
            "prompt": prompt,
            "path": latest.get("path", []),
            "sandbox_status": latest.get("sandbox_status", "UNKNOWN"),
            "latency_ms": latest.get("latency_ms", dt)
        })

    print("\n" + "=" * 80)
    print("HARD PANDAS CLI EVALUATION SUMMARY")
    print("=" * 80)
    for r in results:
        status = r["sandbox_status"]
        path_str = " -> ".join(r["path"]) if r["path"] else "NONE"
        print(f"[{status:<12}] {r['label']}")
        print(f"              Routed: {path_str}")
        print(f"              Total Latency: {r['latency_ms']:.1f} ms\n")


if __name__ == "__main__":
    run_hard_cli_tests()
