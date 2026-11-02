"""
tools/test_profiles_pandas.py

Multi-profile benchmark harness for testing all profiles except B:
- Profile 0: Pure Symbolic
- Profile A: Dense Embeddings RAG (jina-embeddings-v5-text-nano)
- Profile C: Hybrid Neuro-Symbolic (jina-embeddings-v5-text-nano + qwen2.5-coder-1.5b-instruct)
- Profile D: Routing Benchmark (jina-embeddings-v5-text-nano + qwen2.5-coder-1.5b-instruct)
- Profile E: Translator Pass + Neuro-Symbolic (jina-embeddings-v5-text-nano + qwen2.5-coder-1.5b-instruct)

Tested against a pure pandas task requiring only pandas.json:
"Load dataset from CSV file and remove rows with missing values"
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import ModelManager
from lattice import LatticeOrchestrator
from internal_rag import LocalRAG
from router import LatticeRouter, HardwareProfiler
from unification import UnificationGate
from gevr_sandbox import GEVRSandbox

EMBEDDER_MODEL = "jina-embeddings-v5-text-nano"
LLM_MODEL = "qwen2.5-coder-1.5b-instruct"
TEST_PROMPT = "Load dataset from CSV file and remove rows with missing values"
DB_PATH = str(PROJECT_ROOT / "trees" / "lattice.db")


def run_benchmark():
    print("=" * 70)
    print("STRATA MULTI-PROFILE PANDAS BENCHMARK")
    print(f"Embedder : {EMBEDDER_MODEL}")
    print(f"LLM      : {LLM_MODEL}")
    print(f"Prompt   : \"{TEST_PROMPT}\"")
    print(f"Database : {DB_PATH}")
    print("=" * 70)

    # 1. Initialize Orchestrator and Gate
    print("\n[*] Initializing Lattice Orchestrator from lattice.db...")
    t0 = time.perf_counter()
    orchestrator = LatticeOrchestrator()
    orchestrator.load_from_database(DB_PATH)
    orchestrator.build_topology()
    gate = UnificationGate()
    sandbox = GEVRSandbox()
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"[✓] Orchestrator ready with {len(orchestrator.loaded_cells):,} nodes ({t_load:.1f}ms)")

    profiles_to_test = [
        ("0", "Profile 0 (Pure Symbolic Layer)"),
        ("A", "Profile A (Dense Embeddings RAG)"),
        ("C", "Profile C (Hybrid Neuro-Symbolic + GGUF LLM)"),
        ("D", "Profile D (Routing Benchmark)"),
        ("E", "Profile E (Translator Pass + Neuro-Symbolic)"),
    ]

    results: List[Dict[str, Any]] = []

    dummy_csv = PROJECT_ROOT / "input_data.csv"
    if not dummy_csv.exists():
        dummy_csv.write_text("a,b,c\n1,2,3\n4,,6\n7,8,9\n")

    for prof_key, prof_label in profiles_to_test:
        print(f"\n" + "-" * 70)
        print(f"[*] Testing {prof_label}...")
        t_prof_start = time.perf_counter()

        mm = ModelManager.get_instance()
        rag = None

        try:
            # Initialize profile
            if prof_key == "0":
                mm.active_profile = None
                mm.current_profile_name = "0"
                rag = None
            else:
                HardwareProfiler.set_config(embedder_device="auto", llm_device="auto")
                mm.initialize_profile(
                    profile_type=prof_key,
                    embedder_name=EMBEDDER_MODEL,
                    llm_name=LLM_MODEL if prof_key in ("C", "D", "E") else ""
                )
                print(f"    [*] Initializing LocalRAG vector space with enriched semantic embeddings...")
                rag = LocalRAG(trees_dir=str(PROJECT_ROOT / "trees"), orchestrator=orchestrator)

            router = LatticeRouter(orchestrator=orchestrator, internal_rag=rag)

            # Step 1: Optional Translator Pass (Profile E)
            effective_prompt = TEST_PROMPT
            t_trans = 0.0
            if prof_key == "E" and mm.has_translator_pass():
                t0_trans = time.perf_counter()
                trans_system = (
                    "You are a precise technical translator. Convert the following user request "
                    "into a concise canonical pipeline specification stating input source, exact transforms, "
                    "and destination sink. Output ONLY the canonical query."
                )
                effective_prompt = mm.generate_text(TEST_PROMPT, max_tokens=128, system_prompt=trans_system)
                t_trans = (time.perf_counter() - t0_trans) * 1000.0
                print(f"    [Profile E Translator] ({t_trans:.1f}ms): {effective_prompt}")

            # Step 2: Routing
            t0_route = time.perf_counter()
            cells = router.plan_path(effective_prompt, return_tuple=False)
            route_ms = (time.perf_counter() - t0_route) * 1000.0

            path_ids = [c.cell_id for c in cells] if cells else []
            print(f"    [Routing] ({route_ms:.2f}ms): {' -> '.join(path_ids) if path_ids else 'NO PATH FOUND'}")

            # Step 3: Synthesis (unless Profile D which is routing-only)
            synth_ms = 0.0
            code = ""
            sandbox_res = None

            can_synth = (prof_key != "D") and bool(cells)
            if can_synth:
                t0_synth = time.perf_counter()
                code = gate.unify_and_emit(cells, TEST_PROMPT)
                synth_ms = (time.perf_counter() - t0_synth) * 1000.0
                print(f"    [Synthesis] ({synth_ms:.2f}ms):")
                for line in code.strip().splitlines()[:8]:
                    print(f"        {line}")
                if len(code.strip().splitlines()) > 8:
                    print("        ...")

                # Step 4: Sandbox execution check
                t0_exec = time.perf_counter()
                sandbox_res = sandbox.execute(code, timeout=5.0)
                exec_ms = (time.perf_counter() - t0_exec) * 1000.0
                status = "SUCCESS" if sandbox_res.get("success") else f"FAIL ({sandbox_res.get('error')})"
                print(f"    [Sandbox Execution] ({exec_ms:.2f}ms): {status}")

            total_ms = (time.perf_counter() - t_prof_start) * 1000.0

            results.append({
                "profile": prof_key,
                "label": prof_label,
                "status": "PASS" if path_ids else "FAIL",
                "path": path_ids,
                "route_ms": round(route_ms, 2),
                "synth_ms": round(synth_ms, 2),
                "trans_ms": round(t_trans, 2),
                "total_ms": round(total_ms, 2),
                "sandbox_ok": sandbox_res.get("success", False) if sandbox_res else None,
                "code": code
            })

        except Exception as e:
            print(f"    [!] Error testing {prof_label}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "profile": prof_key,
                "label": prof_label,
                "status": f"ERROR: {e}",
                "path": [],
                "route_ms": 0,
                "synth_ms": 0,
                "trans_ms": 0,
                "total_ms": 0,
                "sandbox_ok": False,
                "code": ""
            })

    # Summary Report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    for r in results:
        status_flag = "[✓ PASS]" if r["status"] == "PASS" else f"[X {r['status']}]"
        path_str = " -> ".join(r["path"]) if r["path"] else "none"
        print(f"{r['profile']:<3} | {r['label']:<42} | {status_flag:<8} | Route: {r['route_ms']:>6.1f}ms | Synth: {r['synth_ms']:>6.1f}ms | Path: {path_str}")

    return results


if __name__ == "__main__":
    run_benchmark()
