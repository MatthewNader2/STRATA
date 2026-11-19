#!/usr/bin/env python3
"""
scripts/lint_trees.py

Continuous Integration (CI) linter for STRATA domain trees:
  1. Validates each JSON file in trees/*.json strictly against TreeSchema (extra="forbid").
  2. Ensures shape_contract is unified into canonical ShapeContractSchema (no raw strings).
  3. Fails if any cell code_template in a core/library tree defines an inline `def ` or contains > 2 statements.
  4. Fails if any cell has `required: false` on a positional argument with no default in `inspect.signature` of the target callable.
  5. Enforces that all port typestates are declared in domain typestates and structural kind integrity.

Returns exit code 0 on success, or 1 on any schema violation.
"""

import ast
import glob
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Macro/combinator domain exempt from the core/library 2-statement / no-def rule.
STATEMENT_LIMIT_EXEMPT_DOMAINS = {"control_flow"}

_RUNTIME_NS_CACHE: Optional[Dict[str, Any]] = None


def _get_runtime_ns() -> Dict[str, Any]:
    global _RUNTIME_NS_CACHE
    if _RUNTIME_NS_CACHE is not None:
        return _RUNTIME_NS_CACHE
    ns: Dict[str, Any] = {}
    for mod_name, alias in (
        ("numpy", "np"),
        ("pandas", "pd"),
        ("cv2", "cv2"),
        ("matplotlib.pyplot", "plt"),
        ("matplotlib.figure", "Figure"),
        ("matplotlib.axes", "Axes"),
        ("seaborn", "sns"),
        ("sqlite3", "sqlite3"),
        ("requests", "requests"),
        ("pygame", "pygame"),
        ("pathlib", "pathlib"),
        ("PIL.Image", "Image"),
        ("PIL.ImageDraw", "ImageDraw"),
        ("PIL.ImageFilter", "ImageFilter"),
        ("PIL.ImageEnhance", "ImageEnhance"),
        ("scipy.signal", "signal"),
        ("cryptography.fernet", "fernet"),
        ("joblib", "joblib"),
    ):
        try:
            mod = importlib.import_module(mod_name)
            ns[alias] = mod
            ns[mod_name] = mod
        except Exception:
            pass
    if "pathlib" in ns:
        ns["Path"] = ns["pathlib"].Path
    _RUNTIME_NS_CACHE = ns
    return ns


def _resolve_call_target(cell: Any) -> Tuple[Optional[Any], Optional[ast.Call], bool]:
    code = (cell.code_template or "").strip()
    cid = cell.cell_id or ""
    san = re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_P_\1_", code)
    try:
        tree = ast.parse(san)
    except Exception:
        return None, None, False
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if not calls:
        return None, None, False
    ns = _get_runtime_ns()

    if "." in cid and " " not in cid:
        parts = cid.split(".")
        for k in range(len(parts), 0, -1):
            try:
                m = importlib.import_module(".".join(parts[:k]))
                obj: Any = m
                for a in parts[k:]:
                    obj = getattr(obj, a)
                if callable(obj):
                    return obj, calls[0], False
            except Exception:
                pass

    for c in calls:
        if isinstance(c.func, ast.Attribute):
            if isinstance(c.func.value, ast.Name) and c.func.value.id.startswith("_P_"):
                pname = c.func.value.id[3:-1]
                pmeta = (cell.inputs or {}).get(pname)
                tn = str(getattr(pmeta, "type_name", "") or "").lower()
                cls: Any = None
                for key in (tn, pname.lower()):
                    if "dataframe" in key and "pd" in ns:
                        cls = ns["pd"].DataFrame
                        break
                    if "series" in key and "pd" in ns:
                        cls = ns["pd"].Series
                        break
                    if ("axes" in key or key == "ax") and "Axes" in ns:
                        cls = ns["Axes"].Axes
                        break
                    if ("figure" in key or key == "fig") and "Figure" in ns:
                        cls = ns["Figure"].Figure
                        break
                    if "path" in key and "Path" in ns:
                        cls = ns["Path"]
                        break
                    if ("image" in key or "img" in key) and "Image" in ns:
                        cls = ns["Image"].Image
                        break
                    if "session" in key and "requests" in ns:
                        cls = ns["requests"].Session
                        break
                    if ("connection" in key or "conn" in key) and "sqlite3" in ns:
                        cls = ns["sqlite3"].Connection
                        break
                    if ("cursor" in key or "cur" in key) and "sqlite3" in ns:
                        cls = ns["sqlite3"].Cursor
                        break
                    if "rect" in key and "pygame" in ns:
                        cls = ns["pygame"].Rect
                        break
                    if ("surface" in key or "surf" in key) and "pygame" in ns:
                        cls = ns["pygame"].Surface
                        break
                if cls is not None and hasattr(cls, c.func.attr):
                    return getattr(cls, c.func.attr), c, True
            else:
                chain: List[str] = []
                cur: Any = c.func
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    chain.append(cur.id)
                    chain.reverse()
                    if chain[0] in ns:
                        obj = ns[chain[0]]
                        try:
                            for a in chain[1:]:
                                obj = getattr(obj, a)
                            if callable(obj):
                                return obj, c, False
                        except Exception:
                            pass
    return None, None, False


def lint_tree_file(path: str) -> List[str]:
    violations: List[str] = []
    rel_path = os.path.relpath(path, REPO_ROOT)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [f"[{rel_path}] JSON decode error: {e}"]

    # 1. Strict Pydantic TreeSchema validation (extra="forbid" rejects unknown keys)
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

    # 3. Check cells and ports
    for cell in tree.cells:
        cid = cell.cell_id
        stage = cell.stage

        # 3a. Atomic code_template check for core/library trees (no inline def, <= 2 statements)
        code = (cell.code_template or "").strip()
        if code and domain not in STATEMENT_LIMIT_EXEMPT_DOMAINS:
            san = re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_P_\1_", code)
            try:
                c_ast = ast.parse(san)
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(c_ast)):
                    violations.append(f"[{rel_path}] {cid}: core/library cell code_template defines inline 'def'")
                if len(c_ast.body) > 2:
                    violations.append(f"[{rel_path}] {cid}: core/library cell code_template has {len(c_ast.body)} statements (max 2)")
            except SyntaxError:
                pass

        # 3b. Verify required=False positional args actually have defaults in inspect.signature
        obj, call_node, is_method = _resolve_call_target(cell)
        if obj is not None and call_node is not None:
            try:
                sig = inspect.signature(obj)
            except Exception:
                sig = None
            if sig is not None:
                params = list(sig.parameters.values())
                if is_method and params and params[0].name == "self":
                    params = params[1:]
                for idx, arg_node in enumerate(call_node.args):
                    if (
                        isinstance(arg_node, ast.Name)
                        and arg_node.id.startswith("_P_")
                        and arg_node.id.endswith("_")
                        and idx < len(params)
                    ):
                        pname = arg_node.id[3:-1]
                        sig_p = params[idx]
                        port_obj = cell.inputs.get(pname)
                        if (
                            port_obj is not None
                            and not port_obj.required
                            and port_obj.default_value is None
                            and sig_p.default is inspect.Parameter.empty
                            and sig_p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                        ):
                            violations.append(
                                f"[{rel_path}] {cid}.in:{pname}: required=False on positional arg '{sig_p.name}' with no default in inspect.signature"
                            )

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
                            or sk == "path_spec"
                            or (stage == 1 and type_kind_map.get(tn.lower()) == "path")
                        ):
                            continue
                        if not role:
                            violations.append(f"{p_loc}: required scalar port missing port_role")

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
