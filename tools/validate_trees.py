#!/usr/bin/env python3
"""
tools/validate_trees.py - Universal Domain Tree Schema & Structural Invariant Validator

Validates domain tree JSON definitions against TreeSchema, ensuring:
1. Pure structural typing (structure_kind, rank, element_type).
2. Domain-declared resource formats (supported_formats) on path/resource ports.
3. Tree-declared capability manifests for capability-gated modules.
4. Non-overridable security floor compliance in code templates.
5. Template AST correctness and placeholder-to-port alignment.
6. Explicit Silent Degradation Reporting detailing engine fallback behavior.

Usage:
    python3 tools/validate_trees.py --dir trees/
    python3 tools/validate_trees.py trees/cv2.json trees/pandas.json
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from schema import TreeSchema, CellSchema, PortSchema
    from security_policy import (
        NON_OVERRIDABLE_BLOCKED_MODULES,
        NON_OVERRIDABLE_BLOCKED_ATTRS,
        NON_OVERRIDABLE_BLOCKED_CALLS,
        DEFAULT_CAPABILITY_MODULES,
    )
except ImportError as e:
    print(f"[!] Import error in validate_trees: {e}")
    sys.exit(1)


class TreeValidationResult:
    def __init__(self, tree_path: Path):
        self.tree_path = tree_path
        self.domain = ""
        self.total_cells = 0
        self.total_ports = 0
        self.structurally_typed_ports = 0
        self.errors: List[str] = []
        self.degradation_warnings: List[str] = []
        self.anti_pattern_warnings: List[str] = []
        self.capabilities: Dict[str, Any] = {}
        self.supported_formats: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def structural_coverage(self) -> float:
        if self.total_ports == 0:
            return 100.0
        return (self.structurally_typed_ports / self.total_ports) * 100.0


def extract_template_placeholders(template: str) -> Set[str]:
    """Extracts {placeholder} names from a code template."""
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
    return placeholders


def validate_code_template(
    cell_id: str,
    template: str,
    declared_ports: Set[str],
    bound_slots: Set[str],
    cell_slots: Set[str],
    additional_identifiers: Optional[Set[str]] = None,
) -> List[str]:
    """Validates Python syntax and placeholder coverage in template."""
    errors = []
    if not template or not template.strip():
        errors.append(f"Cell '{cell_id}' code_template is empty.")
        return errors

    placeholders = extract_template_placeholders(template)
    allowed_names = declared_ports | bound_slots | cell_slots | (additional_identifiers or set())
    undeclared = placeholders - allowed_names
    if undeclared:
        errors.append(f"Cell '{cell_id}' template references undeclared placeholders: {sorted(undeclared)}")

    # Replace placeholders with valid identifiers to parse AST
    dummy_code = template
    for ph in placeholders:
        dummy_code = dummy_code.replace(f"{{{ph}}}", f"_ph_{ph}")

    try:
        tree = ast.parse(dummy_code)
    except SyntaxError as syn_err:
        errors.append(f"Cell '{cell_id}' template contains invalid Python syntax: {syn_err}")
        return errors

    # Check non-overridable security floor violations
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in NON_OVERRIDABLE_BLOCKED_MODULES or alias.name in NON_OVERRIDABLE_BLOCKED_MODULES:
                    errors.append(
                        f"Cell '{cell_id}' violates non-overridable security floor by importing forbidden module '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in NON_OVERRIDABLE_BLOCKED_MODULES or node.module in NON_OVERRIDABLE_BLOCKED_MODULES:
                    errors.append(
                        f"Cell '{cell_id}' violates non-overridable security floor by importing from forbidden module '{node.module}'"
                    )
        elif isinstance(node, ast.Attribute):
            if node.attr in NON_OVERRIDABLE_BLOCKED_ATTRS:
                errors.append(
                    f"Cell '{cell_id}' violates security floor with forbidden reflection attribute access '{node.attr}'"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in NON_OVERRIDABLE_BLOCKED_CALLS:
                errors.append(
                    f"Cell '{cell_id}' violates security floor with forbidden dangerous call '{node.func.id}()'"
                )

    return errors


def validate_tree_json(file_path: Path) -> TreeValidationResult:
    """Validates a single domain tree file against all structural and security invariants."""
    res = TreeValidationResult(file_path)

    if not file_path.exists():
        res.errors.append(f"File not found: {file_path}")
        return res

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as jde:
        res.errors.append(f"Invalid JSON syntax: {jde}")
        return res
    except Exception as e:
        res.errors.append(f"Failed to read file: {e}")
        return res

    # Pydantic schema validation
    try:
        tree = TreeSchema.model_validate(raw_data)
    except Exception as e:
        res.errors.append(f"TreeSchema validation failure: {e}")
        return res

    res.domain = tree.domain
    res.capabilities = tree.capabilities or {}
    res.supported_formats = tree.supported_formats or []
    res.total_cells = len(tree.cells)

    seen_cell_ids: Set[str] = set()
    tree_requires_caps: Set[str] = set()
    has_resource_ports = False

    for cell in tree.cells:
        cid = cell.cell_id

        if cid in seen_cell_ids:
            res.errors.append(f"Duplicate cell_id detected within tree: '{cid}'")
        seen_cell_ids.add(cid)

        # Check for legacy anti-patterns
        if getattr(cell, "edges", None):
            res.anti_pattern_warnings.append(
                f"Cell '{cid}' contains pre-baked static edges ({len(cell.edges)}). "
                "These will be ignored by dynamic edge synthesis. Remove manual edges."
            )
        if getattr(cell, "source_priority", 100) > 100:
            res.anti_pattern_warnings.append(
                f"Cell '{cid}' has manual source_priority={cell.source_priority}. "
                "Manual priority overrides are deprecated; routing uses structural type specificity."
            )

        # Validate code_template AST & Placeholders
        declared_ports = set(cell.inputs.keys()) | set(cell.outputs.keys())
        bound_slots = set(cell.bound_slots.keys()) if isinstance(cell.bound_slots, dict) else set(cell.bound_slots or [])
        cell_slots = set(cell.slots.keys()) if isinstance(cell.slots, dict) else set(cell.slots or [])
        extra_ids: Set[str] = set()
        if cell.primary_in:
            extra_ids.add(cell.primary_in)
        if cell.primary_out:
            extra_ids.add(cell.primary_out)
        for p in list(cell.inputs.values()) + list(cell.outputs.values()):
            if getattr(p, "binds", None):
                extra_ids.add(p.binds)
        tmpl_errors = validate_code_template(cid, cell.code_template, declared_ports, bound_slots, cell_slots, extra_ids)
        res.errors.extend(tmpl_errors)

        # Port structural typing checks
        all_ports = list(cell.inputs.items()) + list(cell.outputs.items())
        res.total_ports += len(all_ports)

        for port_name, port in all_ports:
            has_structural = bool(port.structure_kind and (port.rank is not None or port.structure_kind == "handle"))
            if has_structural:
                res.structurally_typed_ports += 1
            else:
                res.degradation_warnings.append(
                    f"Port '{port_name}' in cell '{cid}' omits low-level structural typing (`structure_kind` / `rank`). "
                    "Silent Degradation: Dynamic edge synthesis will fall back to nominal type name matching."
                )

            # Check resource/path ports
            is_path_port = (
                port.structure_kind == "path_spec"
                or (port.param_kind or "").lower() == "file_path"
                or (port.type_name or "").lower() in ("path", "filepath", "uri", "url")
                or any(k in port_name.lower() for k in ("path", "filename", "filepath", "uri", "source_file", "dest_file"))
            )
            if is_path_port:
                has_resource_ports = True
                port_formats = port.supported_formats or res.supported_formats
                if not port_formats:
                    res.degradation_warnings.append(
                        f"Resource port '{port_name}' in cell '{cid}' omits `supported_formats`. "
                        "Silent Degradation: Clause-grounded literal binding will accept any file extension without domain validation."
                    )

        # Check required capabilities from dependencies or template
        dep_modules = set(cell.dependencies or [])
        if cell.code_template:
            for line in cell.code_template.splitlines():
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 2:
                        dep_modules.add(parts[1].split(".")[0])

        for mod in dep_modules:
            cap = DEFAULT_CAPABILITY_MODULES.get(mod)
            if cap:
                tree_requires_caps.add(cap)

    # Tree-level silent degradation warnings
    if has_resource_ports and not res.supported_formats:
        res.degradation_warnings.append(
            f"Tree '{res.domain}' contains resource/file ports but omits root-level `supported_formats`. "
            "Silent Degradation: Literal asset resolution cannot filter candidate files by domain formats."
        )

    for cap in tree_requires_caps:
        if cap not in res.capabilities:
            res.degradation_warnings.append(
                f"Tree '{res.domain}' uses capability-gated operations requiring '{cap}', but 'capabilities' is not declared. "
                f"Silent Degradation: Sandboxed execution will block these operations under default-deny unless explicitly granted."
            )

    return res


def main():
    parser = argparse.ArgumentParser(
        description="Validate domain tree JSON files against TreeSchema and structural invariants."
    )
    parser.add_argument("files", nargs="*", help="Specific tree JSON files to validate")
    parser.add_argument("--dir", type=str, default="", help="Directory containing tree JSON files")
    parser.add_argument("--strict", action="store_true", help="Treat silent degradation warnings as errors")
    args = parser.parse_args()

    target_paths: List[Path] = []
    if args.files:
        for f in args.files:
            target_paths.append(Path(f))
    elif args.dir:
        dir_p = Path(args.dir)
        if dir_p.exists():
            target_paths = sorted([p for p in dir_p.glob("*.json") if not p.name.startswith("_")])
    else:
        default_dir = PROJECT_ROOT / "trees"
        if default_dir.exists():
            target_paths = sorted([p for p in default_dir.glob("*.json") if not p.name.startswith("_")])

    if not target_paths:
        print("[!] No tree JSON files found to validate.")
        sys.exit(1)

    print("=" * 80)
    print(f"STRATA Domain Tree Structural Validator ({len(target_paths)} trees)")
    print("=" * 80)

    total_cells = 0
    total_ports = 0
    total_struct_ports = 0
    all_clean = True
    total_degradations = 0

    for path in target_paths:
        result = validate_tree_json(path)
        total_cells += result.total_cells
        total_ports += result.total_ports
        total_struct_ports += result.structurally_typed_ports
        total_degradations += len(result.degradation_warnings)

        status_str = "[PASS]" if result.is_valid else "[FAIL]"
        cov_str = f"{result.structural_coverage:.1f}%"
        caps_str = f"[{', '.join(result.capabilities.keys())}]" if result.capabilities else "none"
        fmts_str = f"[{', '.join(result.supported_formats[:3])}{'...' if len(result.supported_formats) > 3 else ''}]" if result.supported_formats else "none"

        print(f"{status_str:6} {path.name:22} | Cells: {result.total_cells:4} | Struct Typ: {cov_str:6} | Caps: {caps_str:12} | Formats: {fmts_str}")

        if not result.is_valid:
            all_clean = False
            for err in result.errors[:10]:
                print(f"       ERROR: {err}")
            if len(result.errors) > 10:
                print(f"       ... and {len(result.errors) - 10} more error(s)")

        if result.anti_pattern_warnings:
            for w in result.anti_pattern_warnings[:3]:
                print(f"       ANTI-PATTERN: {w}")

        if result.degradation_warnings:
            for dw in result.degradation_warnings[:2]:
                print(f"       WARNING: {dw}")
            if len(result.degradation_warnings) > 2:
                print(f"       ... and {len(result.degradation_warnings) - 2} more silent degradation warning(s)")

    print("-" * 80)
    overall_cov = (total_struct_ports / total_ports * 100.0) if total_ports > 0 else 100.0
    print(f"Summary: {len(target_paths)} trees, {total_cells:,} cells, {total_ports:,} ports ({overall_cov:.1f}% structurally typed)")
    print(f"Total Degradation Warnings: {total_degradations}")
    print("=" * 80)

    if not all_clean:
        print("[!] Validation failed with schema or invariant errors.")
        sys.exit(1)
    elif args.strict and total_degradations > 0:
        print("[!] Strict mode enabled: failed due to silent degradation warnings.")
        sys.exit(1)
    else:
        print("[✓] All domain trees passed schema & structural integrity verification.")
        sys.exit(0)


if __name__ == "__main__":
    main()
