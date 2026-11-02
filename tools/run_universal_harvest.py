"""
tools/run_universal_harvest.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Universal Harvest CLI over the adapter-based unified pipeline.

Usage:
  python -m tools.run_universal_harvest --library sklearn
  python -m tools.run_universal_harvest --all
  python -m tools.run_universal_harvest --library <any_package> --output-dir trees/
  python -m tools.run_universal_harvest --library sklearn --compile

Pipeline per domain:
  1. Adapter selection by implementation kind (source / stubs / runtime)
  2. Unified harvest: constants, constructors, methods (with stateful
     lifecycle modeling), module functions
  3. Knowledge merge from existing tree + LLM checkpoint (tags/docstrings)
  4. Wiring-invariant repair + AST validation
  5. Save clean tree JSON (optionally compile to lattice.db)
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.universal_harvester import UniversalHarvester
from src.schema import TreeSchema, CellSchema
from src.template_wiring import repair_wiring_invariant

PRIMARY_DOMAINS: Dict[str, str] = {
    "cv2": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "sklearn",
    "matplotlib": "matplotlib.pyplot",
    "python_core": "builtins",
}


def is_cell_wiring_valid(cell_dict: Dict[str, Any]) -> bool:
    template = cell_dict.get("code_template", "")
    placeholders = set(re.findall(r"\{(\w+)\}", template)) - {"output_var"}
    inputs = set(cell_dict.get("inputs", {}).keys())
    return placeholders.issubset(inputs)


def harvest_single_domain(
        domain_name: str,
        package_name: str,
        output_dir: Path,
        enrich: bool = True,
        compile_db: bool = False
) -> Path:
    print(f"\n========================================================")
    print(f"[*] Universally Harvesting Domain: {domain_name} (pkg: {package_name})")
    print(f"========================================================")

    harvester = UniversalHarvester(
        domain_name=domain_name,
        package_name=package_name,
    )
    print(f"[i] Adapter chain: {harvester.adapter.describe()}")

    cells = harvester.harvest_all()
    print(f"[+] Harvested {len(cells)} raw morphism nodes for {domain_name}")

    if enrich:
        enrichment_sources = [
            output_dir / f"{domain_name}.json",
            PROJECT_ROOT / "strata_enrichment" / "checkpoints" / f"{domain_name}.json"
        ]
        cells = UniversalHarvester.enrich_from_existing_trees(
            domain_name=domain_name,
            new_cells=cells,
            existing_tree_paths=enrichment_sources
        )
        print(f"[+] Merged knowledge from {len(enrichment_sources)} prior sources")

    # Validate wiring and syntax
    valid_cells: List[CellSchema] = []
    wiring_issues = 0
    ast_issues = 0

    for c in cells:
        cid_upper = c.cell_id.upper()
        dom_prefix = f"{domain_name.upper()}_"
        sub_name = cid_upper[len(dom_prefix):] if cid_upper.startswith(dom_prefix) else cid_upper
        if (
            sub_name.startswith("TEST_")
            or sub_name.endswith("_TEST")
            or sub_name in ("TEST", "TESTS", "CONFTEST", "TYPE_CHECKING")
            or "ESTIMATOR_CHECKS" in cid_upper
            or "MODULETESTER" in cid_upper
            or "SKIPTEST" in cid_upper
            or "ESTIMATORCHECKFAILED" in cid_upper
            or "estimator_checks" in c.code_template
            or "testutils" in c.code_template
            or (sub_name.startswith("SET_") and sub_name.endswith("_REQUEST"))
            or ("_SET_" in cid_upper and cid_upper.endswith("_REQUEST"))
            or cid_upper.endswith("_GET_METADATA_ROUTING")
        ):
            continue

        c_dict = c.model_dump()
        if not is_cell_wiring_valid(c_dict):
            repair_wiring_invariant(c_dict, domain_name)
            if not is_cell_wiring_valid(c_dict):
                wiring_issues += 1
                continue
            c = CellSchema(**c_dict)

        # AST syntax check
        dummy_code = re.sub(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', 'dummy_var', c.code_template)
        try:
            ast.parse(dummy_code)
            valid_cells.append(c)
        except SyntaxError:
            ast_issues += 1

    print(f"[+] Validated {len(valid_cells)} / {len(cells)} cells (wiring dropped: {wiring_issues}, AST dropped: {ast_issues})")

    # Lifecycle summary (stateful modeling report)
    endomorphisms = sum(
        1 for c in valid_cells
        if c.outputs.get("output_data", c.outputs and next(iter(c.outputs.values()), None))
        and (c.outputs.get("output_data") or next(iter(c.outputs.values()))).state == "mutated"
        and (c.inputs.get("data") or next(iter(c.inputs.values()), None)) is not None
        and c.inputs.get("data") is not None
    )
    constrained_receivers = sum(
        1 for c in valid_cells
        if c.inputs.get("data") is not None and c.inputs["data"].state == "mutated"
    )
    print(f"[i] Stateful lifecycle: {endomorphisms} endomorphisms, {constrained_receivers} state-dependent receivers")

    # Output clean tree JSON
    tree_dict = {
        "domain": domain_name,
        "version": "2.0.0",
        "cells": [c.model_dump() for c in valid_cells]
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{domain_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=2)

    print(f"[✓] Saved clean categorized domain tree to: {out_file} ({len(valid_cells)} nodes)")

    if compile_db:
        import subprocess
        db_target = output_dir / "lattice.db"
        print(f"[*] Compiling lattice DB at {db_target} (--clean)...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "compile_trees.py"),
             "--output", str(db_target), "--domains", domain_name],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            print(f"[!] Compile step returned {result.returncode}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Universal Library Harvester & Morphism Generator")
    parser.add_argument("--library", type=str, help="Name of the library to harvest (e.g. cv2, numpy, pandas)")
    parser.add_argument("--all", action="store_true", help="Harvest all 7 primary STRATA domain libraries")
    parser.add_argument("--package", type=str, help="Underlying Python package name if different from library name")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "trees"), help="Target output directory")
    parser.add_argument("--enrich", dest="enrich", action="store_true", default=True,
                        help="Merge knowledge from existing trees/checkpoints (default on)")
    parser.add_argument("--no-enrich", dest="enrich", action="store_false",
                        help="Skip knowledge merge (fully fresh harvest)")
    parser.add_argument("--compile", action="store_true", help="Automatically compile into trees/lattice.db")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.all:
        for dom, pkg in PRIMARY_DOMAINS.items():
            harvest_single_domain(
                domain_name=dom,
                package_name=pkg,
                output_dir=out_dir,
                enrich=args.enrich,
                compile_db=args.compile
            )
    elif args.library:
        pkg = args.package or args.library
        harvest_single_domain(
            domain_name=args.library,
            package_name=pkg,
            output_dir=out_dir,
            enrich=args.enrich,
            compile_db=args.compile
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
