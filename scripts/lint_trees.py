#!/usr/bin/env python3
"""
tools/lint_trees.py

Continuous Integration (CI) linter for STRATA domain trees:
  1. Validates each JSON file in trees/*.json against TreeSchema.
  2. Ensures shape_contract is unified into canonical ShapeContractSchema (no raw strings).
  3. Enforces Pillow RGB/RGBA rank 3 and grayscale rank 2 conventions.
  4. Requires every non-ingress required scalar input port to have a port_role and default value.
  5. Enforces that all port typestates are declared in domain typestates.
  6. Validates structural kind integrity (no collection with rank 0, no handle as tensor/scalar).

Returns exit code 0 on success, or 1 on any schema violation.
"""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from schema import TreeSchema, PortSchema, ShapeContractSchema
except ImportError as e:
    print(f"[lint_trees] ERROR: Failed to import schema: {e}")
    sys.exit(1)


VALID_SHAPE_CONTRACT_KEYS = {
    "ndim", "shape", "rank_polymorphic", "channel_count", "channel_axis", "len"
}

VALID_PORT_ROLES = {
    "configuration", "literal_parameter", "column_projection", "data_input",
    "source_data", "source_identifier", "feature_data", "target_data", "model",
    "destination", "result", "receiver", "parameter"
}

UNIVERSAL_STATES = {"any", "*", "default", "unregistered", "unbound", ""}


def lint_tree_file(path: str) -> List[str]:
    violations: List[str] = []
    rel_path = os.path.relpath(path, REPO_ROOT)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"[{rel_path}] JSON decode error: {e}"]

    # 1. Pydantic TreeSchema validation
    try:
        tree = TreeSchema(**data)
    except Exception as e:
        violations.append(f"[{rel_path}] TreeSchema validation error: {e}")
        return violations

    domain = tree.domain

    # 2. Extract registered typestates and type metadata
    registered_states: Set[str] = set()
    state_props_map: Dict[str, Dict[str, Any]] = {}
    ts_block = data.get("typestates")
    if isinstance(ts_block, dict):
        for s in ts_block.get("states", []):
            if isinstance(s, dict) and "name" in s:
                sname = str(s["name"]).strip().lower()
                registered_states.add(sname)
                if isinstance(s.get("properties"), dict):
                    state_props_map[sname] = s["properties"]
            elif isinstance(s, str):
                registered_states.add(s.strip().lower())
    elif isinstance(ts_block, list):
        for s in ts_block:
            if isinstance(s, dict) and "name" in s:
                sname = str(s["name"]).strip().lower()
                registered_states.add(sname)
                if isinstance(s.get("properties"), dict):
                    state_props_map[sname] = s["properties"]
            elif isinstance(s, str):
                registered_states.add(s.strip().lower())

    type_kind_map: Dict[str, str] = {}
    raw_types = data.get("types")
    if isinstance(raw_types, dict):
        for tk, tv in raw_types.items():
            if isinstance(tv, dict) and tv.get("kind"):
                type_kind_map[str(tk).strip().lower()] = str(tv["kind"]).strip().lower()
    elif isinstance(raw_types, list):
        for tv in raw_types:
            if isinstance(tv, dict) and (tv.get("name") or tv.get("type_name")) and tv.get("kind"):
                type_kind_map[str(tv.get("name") or tv.get("type_name")).strip().lower()] = str(tv["kind"]).strip().lower()

    path_tokens = {str(t).strip().lower() for t in (tree.path_port_tokens or [])}
    for tok_list in (tree.carrier_roles or {}).get("source_data", ()):
        path_tokens.add(str(tok_list).strip().lower())

    # 3. Check cells and ports
    for cell in tree.cells:
        cid = cell.cell_id
        stage = cell.stage

        # Inputs and outputs
        for port_type, ports in [("in", cell.inputs), ("out", cell.outputs)]:
            for p_name, port in ports.items():
                p_loc = f"[{rel_path}] {cid}.{port_type}:{p_name}"

                # Check shape_contract
                raw_sc = port.shape_contract
                if raw_sc is not None:
                    if not isinstance(raw_sc, (dict, ShapeContractSchema)):
                        violations.append(f"{p_loc}: shape_contract must be a structured schema/dict, got {type(raw_sc).__name__}: {raw_sc}")
                    else:
                        sc_dict = raw_sc.model_dump() if isinstance(raw_sc, ShapeContractSchema) else raw_sc
                        unknown_keys = set(sc_dict.keys()) - VALID_SHAPE_CONTRACT_KEYS
                        if unknown_keys:
                            violations.append(f"{p_loc}: shape_contract contains unrecognized keys: {unknown_keys}")
                        if not isinstance(sc_dict.get("rank_polymorphic", False), bool):
                            violations.append(f"{p_loc}: shape_contract.rank_polymorphic must be a boolean")

                # Check port typestate
                st = str(port.state or "").strip().lower()
                if st and st not in UNIVERSAL_STATES and st not in registered_states:
                    violations.append(f"{p_loc}: state '{port.state}' is not registered in domain typestates")

                # Structural kind integrity
                sk = str(port.structure_kind or "").strip().lower()
                rank = port.rank
                tn = str(port.type_name or "").strip()

                if (type_kind_map.get(tn.lower()) == "collection" or tn.startswith("List[") or tn in ("list", "sequence")) and rank == 0:
                    violations.append(f"{p_loc}: collection type '{tn}' cannot have rank 0 (rank 0 denotes scalar)")

                if (type_kind_map.get(tn.lower()) == "collection" or tn.startswith("List[") or tn in ("list", "sequence")) and sk == "handle":
                    violations.append(f"{p_loc}: collection type '{tn}' cannot have structure_kind 'handle'")

                if type_kind_map.get(tn.lower()) == "visual" and sk == "tensor":
                    violations.append(f"{p_loc}: visual handle '{tn}' cannot have structure_kind 'tensor'")

                # Declarative state expected_rank check (driven cleanly by typestates[].properties.expected_rank)
                expected_rank = state_props_map.get(st, {}).get("expected_rank")
                if expected_rank is not None:
                    is_poly = False
                    if isinstance(raw_sc, (dict, ShapeContractSchema)):
                        sc_d = raw_sc.model_dump() if isinstance(raw_sc, ShapeContractSchema) else raw_sc
                        is_poly = bool(sc_d.get("rank_polymorphic"))
                    if not is_poly and rank != expected_rank:
                        violations.append(f"{p_loc}: state '{port.state}' must have rank {expected_rank}, got rank {rank}")

                # Required non-ingress scalar ports
                if port_type == "in" and port.required:
                    at = str(port.abstract_type or "").strip().lower()
                    is_scalar = (sk == "scalar") or (rank == 0 and at in ("scalar", "primitive", "standard", "logical", "text"))
                    if is_scalar:
                        role = str(port.port_role or port.role or "").strip().lower()
                        # Exclude S1 ingress source data / path ports
                        if (
                            role in ("source_data", "source_identifier")
                            or at == "path"
                            or (stage == 1 and (p_name.lower() in path_tokens or tn.lower() in path_tokens or type_kind_map.get(tn.lower()) == "path"))
                        ):
                            continue
                        if not role:
                            violations.append(f"{p_loc}: required scalar port missing port_role")
                        elif role in ("configuration", "literal_parameter", "column_projection"):
                            if port.default_value is None:
                                violations.append(f"{p_loc}: required configuration/parameter port missing default_value")

    return violations


def main() -> int:
    tree_files = sorted(glob.glob(str(REPO_ROOT / "trees" / "*.json")))
    if not tree_files:
        print("[lint_trees] ERROR: No tree files found in trees/*.json")
        return 1

    total_violations = 0
    checked_trees = 0

    print(f"[lint_trees] Linting {len(tree_files)} domain trees...")
    for path in tree_files:
        violations = lint_tree_file(path)
        checked_trees += 1
        if violations:
            total_violations += len(violations)
            for v in violations:
                print(f"  VIOLATION: {v}")

    if total_violations > 0:
        print(f"\n[lint_trees] FAILED: {total_violations} violation(s) across {checked_trees} tree(s).")
        return 1
    else:
        print(f"\n[lint_trees] SUCCESS: All {checked_trees} domain trees passed lint checks cleanly.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
