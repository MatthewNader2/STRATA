#!/usr/bin/env python3
"""
tools/validate_tree_json.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Validates AI-generated domain tree JSON files against TreeSchema and compilation invariants.

Usage:
    python3 tools/validate_tree_json.py trees/<domain>.json
    python3 tools/validate_tree_json.py trees/*.json
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from schema import TreeSchema, CellSchema
    from lattice import TypeRegistry
except ImportError as e:
    print(f"[!] Import error: {e}")
    sys.exit(1)


def get_existing_cell_ids(exclude_file: Path = None) -> Set[str]:
    """Collect all cell_ids across existing trees to detect collisions."""
    existing_ids = set()
    trees_dir = PROJECT_ROOT / "trees"
    if not trees_dir.exists():
        return existing_ids

    for p in trees_dir.glob("*.json"):
        if exclude_file and p.resolve() == exclude_file.resolve():
            continue
        if p.name.startswith("_"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            cells = data.get("cells") or (data if isinstance(data, list) else [])
            for c in cells:
                if isinstance(c, dict) and "cell_id" in c:
                    existing_ids.add(c["cell_id"])
        except Exception:
            pass
    return existing_ids


def validate_template_ast(cell_id: str, template: str, inputs: dict, outputs: dict) -> List[str]:
    """Validates Python AST syntax and placeholder alignment."""
    issues = []
    if not template or not template.strip():
        return [f"Cell '{cell_id}' has empty code_template."]

    # Extract placeholders
    declared_ports = set(inputs.keys()) | set(outputs.keys())
    placeholders = set()
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{" and i + 1 < n:
            end = template.find("}", i + 1)
            if end != -1:
                inner = template[i + 1 : end]
                if inner.isidentifier():
                    placeholders.add(inner)
                i = end + 1
                continue
        i += 1

    # Check for undeclared placeholders
    undeclared = placeholders - declared_ports
    if undeclared:
        issues.append(f"Cell '{cell_id}' template uses undeclared placeholders: {sorted(undeclared)}")

    # Check for AST syntax by replacing {ph} with valid identifier
    ast_ready = template
    for ph in placeholders:
        ast_ready = ast_ready.replace(f"{{{ph}}}", f"_ph_{ph}")

    try:
        parsed = ast.parse(ast_ready)
    except SyntaxError as syn_err:
        issues.append(f"Cell '{cell_id}' template has invalid Python syntax: '{template}'. Error: {syn_err}")
        return issues

    # Check for hardcoded bare unquoted filenames
    known_exts = {'csv', 'tsv', 'json', 'jpg', 'jpeg', 'png', 'bmp', 'txt', 'db', 'h5', 'hdf5', 'wav', 'mp3', 'parquet'}
    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if any(val.lower().endswith(f".{ext}") for ext in known_exts):
                if not ("{" in val and "}" in val) and "/" not in val and "\\" not in val:
                    issues.append(f"Cell '{cell_id}' contains hardcoded filename: '{val}' (must use a placeholder port).")

    return issues


def validate_tree_file(file_path: Path, existing_ids: Set[str]) -> Tuple[bool, List[str], List[str], Dict[str, int]]:
    """Validates a single tree JSON file."""
    errors = []
    warnings = []
    stats = {"stage_1": 0, "stage_2": 0, "stage_3": 0, "macros": 0, "types": 0, "total_cells": 0}

    if not file_path.exists():
        return False, [f"File not found: {file_path}"], [], stats

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON syntax in {file_path.name}: {e}"], [], stats

    # Schema Validation
    try:
        tree = TreeSchema.model_validate(raw_data)
    except Exception as e:
        errors.append(f"TreeSchema validation failure: {e}")
        return False, errors, warnings, stats

    # Types & Typestates counts
    types_data = raw_data.get("types", [])
    if isinstance(types_data, list):
        stats["types"] = len(types_data)
    elif isinstance(types_data, dict):
        stats["types"] = len(types_data)

    cells = tree.cells
    stats["total_cells"] = len(cells)

    seen_in_file = set()
    for cell in cells:
        cid = cell.cell_id

        # Duplicate checking
        if cid in seen_in_file:
            errors.append(f"Duplicate cell_id inside file: '{cid}'")
        seen_in_file.add(cid)

        if cid in existing_ids:
            warnings.append(f"cell_id '{cid}' already exists in another tree (will override upon compile).")

        # Stage accounting
        if cell.stage == 1:
            stats["stage_1"] += 1
        elif cell.stage == 2:
            stats["stage_2"] += 1
        elif cell.stage == 3:
            stats["stage_3"] += 1

        if getattr(cell, "node_type", "") == "macro" or getattr(cell, "topology_type", "") != "sequential" or cell.slots:
            stats["macros"] += 1

        # Template validation
        inputs_dict = cell.inputs or {}
        outputs_dict = cell.outputs or {}
        tmpl_issues = validate_template_ast(cid, cell.code_template, inputs_dict, outputs_dict)
        for issue in tmpl_issues:
            errors.append(issue)

        # Primary port checks
        if cell.primary_in and cell.primary_in not in inputs_dict:
            warnings.append(f"Cell '{cid}' primary_in '{cell.primary_in}' not declared in inputs.")
        if cell.primary_out and cell.primary_out not in outputs_dict:
            warnings.append(f"Cell '{cid}' primary_out '{cell.primary_out}' not declared in outputs.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/validate_tree_json.py <path_to_tree.json> [...]")
        sys.exit(1)

    targets = [Path(p) for p in sys.argv[1:]]
    existing_cell_ids = get_existing_cell_ids()

    all_valid = True
    for target in targets:
        print("=" * 70)
        print(f"Validating Tree: {target.name}")
        print("=" * 70)

        is_valid, errors, warnings, stats = validate_tree_file(target, existing_cell_ids)

        print(f"Total Cells   : {stats['total_cells']}")
        print(f"  Stage 1 (Ingress/Source) : {stats['stage_1']}")
        print(f"  Stage 2 (Transform/Step) : {stats['stage_2']}")
        print(f"  Stage 3 (Egress/Sink)   : {stats['stage_3']}")
        print(f"  Higher-Order Macros     : {stats['macros']}")
        print(f"Declared Types: {stats['types']}")
        print("-" * 70)

        if warnings:
            print(f"[!] {len(warnings)} Warning(s):")
            for w in warnings[:10]:
                print(f"    - {w}")
            if len(warnings) > 10:
                print(f"    ... and {len(warnings) - 10} more warnings.")

        if not is_valid:
            all_valid = False
            print(f"\n[X] VALIDATION FAILED with {len(errors)} error(s):")
            for e in errors[:15]:
                print(f"    - {e}")
            if len(errors) > 15:
                print(f"    ... and {len(errors) - 15} more errors.")
        else:
            print("\n[✓] TREE IS VALID AND READY TO COMPILE!")
            print(f"    Compile with: python3 tools/compile_trees.py")

        print("=" * 70)
        print()

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
