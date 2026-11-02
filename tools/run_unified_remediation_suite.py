import sys
import os
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import ModelManager
from lattice import LatticeOrchestrator
from internal_rag import LocalRAG
from router import LatticeRouter
import tests.eval_routing_accuracy as eval_routing
import run_comprehensive_eval as eval_system
import tools.check_lattice_connectivity as connectivity_checker
import tools.find_semantic_duplicates as duplicate_finder

def run_unified_suite():
    print("=" * 80)
    print(" STRATA UNIFIED REMEDIATION & EVALUATION SUITE")
    print(" (Single-Process Persistent Weight Loading)")
    print("=" * 80)

    start_total = time.time()

    # Step 1: Persistent Single-Process Model Loading
    print("\n[Step 1/5] Loading Persistent Models & Orchestrator into RAM/VRAM...")
    ModelManager.get_instance().initialize_profile("D")
    orch = LatticeOrchestrator("trees")
    rag = LocalRAG(trees_dir="trees", orchestrator=orch)
    print("  [+] Model weights and 36,686 node FAISS index loaded ONCE persistently.")

    # Step 2: Phase 1 & 7 Routing Accuracy Evaluation
    print("\n[Step 2/5] Running Routing & Retrieval Accuracy Harness...")
    eval_routing.evaluate_routing_accuracy()

    # Step 3: Phase 7 End-to-End System Benchmark Suite
    print("\n[Step 3/5] Running End-to-End System Evaluation Suite...")
    eval_system.run_comprehensive_evaluation()

    # Step 4: Phase 5 Lattice Connectivity Health Check
    print("\n[Step 4/5] Running Lattice Connectivity Health Check...")
    connectivity_checker.check_connectivity()

    # Step 5: Phase 6 Semantic Duplicate Clustering Report
    print("\n[Step 5/5] Running Semantic Duplicate Clustering Engine...")
    duplicate_finder.find_semantic_duplicates()

    total_duration = time.time() - start_total
    print("\n" + "=" * 80)
    print(f" UNIFIED SUITE EXECUTION COMPLETE (Total Duration: {total_duration:.2f}s)")
    print("=" * 80)

if __name__ == "__main__":
    run_unified_suite()
