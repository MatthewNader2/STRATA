#!/usr/bin/env python3
"""
scripts/lint_trees.py — Strict Canonical v2 CI Linter for STRATA Domain Trees.

Enforces:
  1. Zero legacy/non-canonical keys on raw tree, cell, port, and layout dicts prior to validation.
  2. Strict Pydantic `TreeSchema` (`extra="forbid"`) conformance across all manifests.
  3. Atomic `code_template` rules (no inline imports; non-combinators limited to <= 2 statements, no inline def/walrus).
  4. Consistency of optional positional arguments (`required=False`, `default_value=None`) against live Python signatures.
  5. Typestate registration (including transitive `imports`), `LayoutContract` integrity, `expected_rank`, and literal `port_role`.
"""
from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from schema import (  # noqa: E402
    CellSchema,
    LayoutContract,
    PortSchema,
    TOP_STATES,
    TreeSchema,
)
from type_system import BoundedPoset  # noqa: E402

CANONICAL_TREE_KEYS: Set[str] = set(TreeSchema.model_fields.keys())
CANONICAL_CELL_KEYS: Set[str] = set(CellSchema.model_fields.keys())
CANONICAL_PORT_KEYS: Set[str] = set(PortSchema.model_fields.keys())
CANONICAL_LAYOUT_KEYS: Set[str] = set(LayoutContract.model_fields.keys())
UNIVERSAL_STATES: Set[str] = set(TOP_STATES)
FORBIDDEN_NOISE_INVARIANTS: Set[str] = {
    "primary",
    "optional",
    "scalar",
    "estimator",
    "collection",
    "const",
}


def _is_dotted_type_noise_invariant(inv_l: str) -> bool:
    """Detect dotted package/class type tags (e.g. pkg.type) mistakenly stored in invariants."""
    if "." not in inv_l or any(ch in inv_l for ch in "=<>:()"):
        return False
    parts = inv_l.split(".")
    return len(parts) >= 2 and all(seg.isidentifier() for seg in parts)


def _check_raw_canonical_keys(raw_data: Any, rel_path: str) -> List[str]:
    """Verify raw JSON contains zero legacy keys before Pydantic normalizers run."""
    errs: List[str] = []
    if not isinstance(raw_data, dict):
        return [f"[{rel_path}] Root JSON value must be an object"]

    extra_tree = sorted(set(raw_data.keys()) - CANONICAL_TREE_KEYS)
    if extra_tree:
        errs.append(f"[{rel_path}] Non-canonical tree key(s): {extra_tree}")

    for idx, cell in enumerate(raw_data.get("cells") or ()):
        if not isinstance(cell, dict):
            errs.append(f"[{rel_path}] Cell[{idx}] must be an object")
            continue
        cid = str(cell.get("cell_id") or f"cell_{idx}")
        extra_cell = sorted(set(cell.keys()) - CANONICAL_CELL_KEYS)
        if extra_cell:
            errs.append(f"[{rel_path}] {cid}: non-canonical cell key(s): {extra_cell}")
        for direction in ("inputs", "outputs"):
            ports = cell.get(direction) or {}
            if not isinstance(ports, dict):
                errs.append(f"[{rel_path}] {cid}.{direction} must be an object")
                continue
            for pname, pval in ports.items():
                if not isinstance(pval, dict):
                    errs.append(f"[{rel_path}] {cid}.{direction}:{pname} must be an object")
                    continue
                extra_port = sorted(set(pval.keys()) - CANONICAL_PORT_KEYS)
                if extra_port:
                    errs.append(f"[{rel_path}] {cid}.{direction}:{pname}: non-canonical port key(s): {extra_port}")
                layout = pval.get("layout")
                if isinstance(layout, dict):
                    extra_layout = sorted(set(layout.keys()) - CANONICAL_LAYOUT_KEYS)
                    if extra_layout:
                        errs.append(f"[{rel_path}] {cid}.{direction}:{pname}.layout: non-canonical key(s): {extra_layout}")
    return errs


def _build_cell_ns(cell: Any) -> Dict[str, Any]:
    """Resolve cell's declared `dependencies` import statements dynamically via importlib."""
    ns: Dict[str, Any] = {}
    for raw_dep in (getattr(cell, "dependencies", None) or ()):
        dep_str = str(raw_dep or "").strip()
        if not dep_str:
            continue
        if not dep_str.startswith(("import ", "from ")):
            dep_str = f"import {dep_str}"
        try:
            dep_ast = ast.parse(dep_str)
        except SyntaxError:
            continue
        for node in ast.walk(dep_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        mod = importlib.import_module(alias.name)
                        if alias.asname:
                            ns[alias.asname] = mod
                        else:
                            top_pkg = alias.name.split(".", 1)[0]
                            try:
                                ns[top_pkg] = importlib.import_module(top_pkg)
                            except Exception:
                                pass
                            ns[alias.name] = mod
                    except Exception:
                        pass
            elif isinstance(node, ast.ImportFrom) and node.module:
                try:
                    mod = importlib.import_module(node.module)
                except Exception:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bind_name = alias.asname or alias.name
                    if hasattr(mod, alias.name):
                        ns[bind_name] = getattr(mod, alias.name)
                    else:
                        try:
                            ns[bind_name] = importlib.import_module(f"{node.module}.{alias.name}")
                        except Exception:
                            pass
    return ns


def _resolve_type_class_from_ns(ns: Dict[str, Any], type_hint: str) -> Optional[Any]:
    if not type_hint:
        return None
    clean_hint = type_hint.split("::", 1)[-1].split("[", 1)[0].strip()
    if not clean_hint:
        return None
    hint_low = clean_hint.lower()

    if clean_hint in ns and isinstance(ns[clean_hint], type):
        return ns[clean_hint]
    for obj in ns.values():
        if isinstance(obj, type) and obj.__name__.lower() == hint_low:
            return obj
        if hasattr(obj, "__dict__"):
            direct = getattr(obj, clean_hint, None)
            if isinstance(direct, type):
                return direct
            for attr_val in vars(obj).values():
                if isinstance(attr_val, type) and attr_val.__name__.lower() == hint_low:
                    return attr_val
    return None


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

    ns = _build_cell_ns(cell)
    for c in calls:
        if isinstance(c.func, ast.Name):
            obj = ns.get(c.func.id)
            if callable(obj):
                return obj, c, False
        elif isinstance(c.func, ast.Attribute):
            if isinstance(c.func.value, ast.Name) and c.func.value.id.startswith("_P_") and c.func.value.id.endswith("_"):
                pname = c.func.value.id[3:-1]
                pmeta = (cell.inputs or {}).get(pname)
                tn = str(getattr(pmeta, "carrier", "") or "").strip()
                cls = _resolve_type_class_from_ns(ns, tn) or _resolve_type_class_from_ns(ns, pname)
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


def _extract_state_decls(data: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    states_set: Set[str] = set()
    props_map: Dict[str, Dict[str, Any]] = {}
    ts_block = data.get("typestates")
    items: List[Any] = []
    if isinstance(ts_block, dict):
        items = list(ts_block.get("states", []) or [])
    elif isinstance(ts_block, list):
        items = list(ts_block)
    for s in items:
        if isinstance(s, dict) and s.get("name"):
            sname = str(s["name"]).strip().lower()
            states_set.add(sname)
            if isinstance(s.get("properties"), dict):
                props_map[sname] = s["properties"]
        elif isinstance(s, str) and s.strip():
            states_set.add(s.strip().lower())
    return states_set, props_map


def _load_imported_tree_dicts(
    tree_path: Path,
    direct_imports: List[str],
) -> Dict[str, Dict[str, Any]]:
    search_dirs: List[Path] = []
    for d in (tree_path.resolve().parent, REPO_ROOT / "trees"):
        if d.exists() and d not in search_dirs:
            search_dirs.append(d)

    dom_data_map: Dict[str, Dict[str, Any]] = {}
    for d in search_dirs:
        for fp in sorted(d.glob("*.json")):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(raw, dict):
                dom_k = str(raw.get("domain") or fp.stem).strip().lower()
                if dom_k and dom_k not in dom_data_map:
                    dom_data_map[dom_k] = raw

    imported: Dict[str, Dict[str, Any]] = {}
    queue: List[str] = [str(imp).strip().lower() for imp in (direct_imports or ()) if str(imp).strip()]
    visited: Set[str] = set()
    while queue:
        imp_dom = queue.pop(0)
        if imp_dom in visited:
            continue
        visited.add(imp_dom)
        raw = dom_data_map.get(imp_dom)
        if not raw:
            continue
        imported[imp_dom] = raw
        for sub_imp in raw.get("imports") or ():
            sub_k = str(sub_imp).strip().lower()
            if sub_k and sub_k not in visited:
                queue.append(sub_k)
    return imported


def _load_imported_typestates(
    imported_trees: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Dict[str, Any]]]]:
    imported_by_dom: Dict[str, Set[str]] = {}
    imported_props_by_dom: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for imp_dom, raw in imported_trees.items():
        st_set, p_map = _extract_state_decls(raw)
        imported_by_dom[imp_dom] = st_set
        imported_props_by_dom[imp_dom] = p_map
    return imported_by_dom, imported_props_by_dom


def _extract_tree_poset_edges(
    dom: str,
    raw_data: Dict[str, Any],
    type_parents: Dict[str, Set[str]],
    state_parents: Dict[str, Set[str]],
    top_types: Set[str],
) -> None:
    dom_l = str(dom or "").strip().lower()
    for t in (raw_data.get("top_types") or ()):
        tl = str(t or "").strip().lower()
        if tl:
            top_types.add(tl)

    raw_types = raw_data.get("types")
    type_items: List[Tuple[str, Any]] = []
    if isinstance(raw_types, dict):
        type_items = [(str(k), v) for k, v in raw_types.items()]
    elif isinstance(raw_types, list):
        for entry in raw_types:
            if isinstance(entry, dict) and (entry.get("name") or entry.get("type_name")):
                type_items.append((str(entry.get("name") or entry.get("type_name")), entry))

    for tname, tdef in type_items:
        tn = tname.strip().lower()
        if not tn:
            continue
        type_parents.setdefault(tn, set())
        if isinstance(tdef, dict):
            if tdef.get("parent"):
                type_parents[tn].add(str(tdef["parent"]).strip().lower())
            for p_item in (tdef.get("parents") or ()):
                ps = str(p_item or "").strip().lower()
                if ps:
                    type_parents[tn].add(ps)

    ts_block = raw_data.get("typestates")
    ts_dom = dom_l
    states_list: List[Any] = []
    if isinstance(ts_block, dict):
        ts_dom = str(ts_block.get("domain") or dom_l).strip().lower()
        states_list = list(ts_block.get("states") or ())
    elif isinstance(ts_block, list):
        states_list = list(ts_block)

    def _qual_st(st_val: str) -> str:
        s = st_val.strip().lower()
        if not s or s in UNIVERSAL_STATES:
            return "any"
        return s if "::" in s or not ts_dom else f"{ts_dom}::{s}"

    for s_entry in states_list:
        if not isinstance(s_entry, dict) or not s_entry.get("name"):
            continue
        sn = _qual_st(str(s_entry["name"]))
        if sn == "any":
            continue
        state_parents.setdefault(sn, set())
        p_raw = s_entry.get("parent_state")
        if p_raw:
            ps = _qual_st(str(p_raw))
            if ps != "any":
                state_parents[sn].add(ps)
        for p_item in (s_entry.get("parent_states") or ()):
            ps = _qual_st(str(p_item))
            if ps != "any":
                state_parents[sn].add(ps)


def _extract_type_kinds(raw_types: Any) -> Dict[str, str]:
    type_kind_map: Dict[str, str] = {}
    if isinstance(raw_types, dict):
        for tk, tv in raw_types.items():
            if isinstance(tv, dict) and tv.get("kind"):
                type_kind_map[str(tk).strip().lower()] = str(tv["kind"]).strip().lower()
    elif isinstance(raw_types, list):
        for tv in raw_types:
            if isinstance(tv, dict) and (tv.get("name") or tv.get("type_name")) and tv.get("kind"):
                name = str(tv.get("name") or tv.get("type_name")).strip().lower()
                type_kind_map[name] = str(tv["kind"]).strip().lower()
    return type_kind_map


def lint_tree_file(path: str) -> List[str]:
    violations: List[str] = []
    rel_path = os.path.relpath(path, REPO_ROOT)

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"[{rel_path}] JSON decode error: {exc}"]

    raw_errs = _check_raw_canonical_keys(data, rel_path)
    if raw_errs:
        return raw_errs

    try:
        tree = TreeSchema(**data)
    except Exception as exc:
        return [f"[{rel_path}] TreeSchema validation error: {exc}"]

    domain_low = str(tree.domain or "").strip().lower()
    is_python_lang = tree.language_spec is None or str(tree.language_spec.ast_backend).strip().lower() == "python_ast"

    registered_states, state_props_map = _extract_state_decls(data)
    imported_trees = _load_imported_tree_dicts(Path(path), list(tree.imports or ()))
    imported_by_dom, imported_props_by_dom = _load_imported_typestates(imported_trees)
    all_imported_states: Set[str] = set().union(*imported_by_dom.values()) if imported_by_dom else set()
    type_kind_map = _extract_type_kinds(tree.types)

    poset_tp: Dict[str, Set[str]] = {}
    poset_sp: Dict[str, Set[str]] = {}
    poset_tt: Set[str] = set()
    _extract_tree_poset_edges(domain_low, data, poset_tp, poset_sp, poset_tt)
    for imp_dom, imp_raw in imported_trees.items():
        _extract_tree_poset_edges(imp_dom, imp_raw, poset_tp, poset_sp, poset_tt)
    for dag_err in BoundedPoset(type_parents=poset_tp, state_parents=poset_sp, top_types=poset_tt).validate_acyclic():
        violations.append(f"[{rel_path}] DAG cycle violation: {dag_err}")

    for cell in tree.cells:
        cid = cell.cell_id
        is_combinator = str(cell.mutation_type or "").strip().lower() == "combinator"

        code = (cell.code_template or "").strip()
        if not code:
            violations.append(f"[{rel_path}] {cid}: empty code_template")
        elif is_python_lang:
            san = re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_P_\1_", code)
            try:
                c_ast = ast.parse(san)
                if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(c_ast)):
                    violations.append(f"[{rel_path}] {cid}: cell code_template contains inline import (declare imports in 'dependencies')")
                if not is_combinator and domain_low != "control_flow":
                    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(c_ast)):
                        violations.append(f"[{rel_path}] {cid}: library cell code_template defines inline 'def'")
                    if len(c_ast.body) > 2:
                        violations.append(f"[{rel_path}] {cid}: library cell code_template has {len(c_ast.body)} statements (max 2)")
                    if any(isinstance(n, ast.NamedExpr) for n in ast.walk(c_ast)):
                        violations.append(f"[{rel_path}] {cid}: library cell code_template uses walrus operator ':='")
                    if any(
                        isinstance(n, ast.Subscript)
                        and isinstance(n.value, ast.Tuple)
                        and any(isinstance(sub, ast.Call) for sub in ast.walk(n.value))
                        for n in ast.walk(c_ast)
                    ):
                        violations.append(f"[{rel_path}] {cid}: library cell code_template packs calls into subscripted tuple")
            except SyntaxError:
                pass

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

        for port_type, ports in (("in", cell.inputs), ("out", cell.outputs)):
            for p_name, port in ports.items():
                p_loc = f"[{rel_path}] {cid}.{port_type}:{p_name}"

                if port.layout is not None and not isinstance(port.layout, LayoutContract):
                    violations.append(f"{p_loc}: layout must be a LayoutContract instance")

                # Validate primary typestate & accepted_states declarations (own or transitive imports)
                state_checks = [("typestate", port.typestate)] + [
                    ("accepted_states", acc_s) for acc_s in (port.accepted_states or ())
                ]
                for field_lbl, raw_st in state_checks:
                    st_l = str(raw_st or "").strip().lower()
                    if not st_l or st_l in UNIVERSAL_STATES:
                        continue
                    if "::" in st_l:
                        st_dom, st_bare = (s.strip().lower() for s in st_l.split("::", 1))
                        if st_bare in UNIVERSAL_STATES:
                            continue
                        if st_dom == domain_low:
                            if st_bare not in registered_states:
                                violations.append(f"{p_loc}: {field_lbl} '{raw_st}' is not registered in domain typestates")
                        elif st_dom not in imported_by_dom or st_bare not in imported_by_dom[st_dom]:
                            violations.append(f"{p_loc}: {field_lbl} '{raw_st}' is not registered in declared imports")
                    elif st_l not in registered_states and st_l not in all_imported_states:
                        violations.append(f"{p_loc}: {field_lbl} '{raw_st}' is not registered in domain typestates")

                carrier = str(port.carrier or "").strip()
                carrier_low = carrier.lower()
                rank = port.layout.rank if port.layout is not None else None
                is_poly = bool(port.layout.rank_polymorphic) if port.layout is not None else False

                is_coll = (
                    type_kind_map.get(carrier_low) == "collection"
                    or carrier.startswith("List[")
                    or carrier_low in ("list", "sequence")
                )
                if is_coll and rank == 0:
                    violations.append(f"{p_loc}: collection carrier '{carrier}' cannot have layout.rank == 0")
                if is_coll and port.resource_mode == "acquire":
                    violations.append(f"{p_loc}: collection carrier '{carrier}' cannot have resource_mode 'acquire'")

                ts_low = str(port.typestate or "").strip().lower()
                ts_dom, ts_bare = (
                    (s.strip().lower() for s in ts_low.split("::", 1))
                    if "::" in ts_low
                    else (domain_low, ts_low)
                )
                st_props = (
                    state_props_map.get(ts_bare)
                    if ts_dom == domain_low
                    else imported_props_by_dom.get(ts_dom, {}).get(ts_bare)
                ) or {}
                expected_rank = st_props.get("expected_rank")
                if expected_rank is not None and not is_poly and rank != expected_rank:
                    violations.append(f"{p_loc}: typestate '{port.typestate}' expects rank {expected_rank}, got {rank}")

                if port_type == "in" and port.required and port.value_class == "literal":
                    if not (port.port_role and port.port_role.strip()):
                        violations.append(f"{p_loc}: required literal port missing port_role")

                for inv in (port.invariants or ()):
                    inv_l = str(inv or "").strip().lower()
                    if inv_l in FORBIDDEN_NOISE_INVARIANTS or _is_dotted_type_noise_invariant(inv_l):
                        violations.append(f"{p_loc}: forbidden noise/legacy invariant {inv!r}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="STRATA Canonical v2 Domain Tree CI Linter")
    parser.add_argument("--include-fixtures", action="store_true", help="Also lint held-out fixture trees under benchmarks/fixtures/*/*.json")
    args = parser.parse_args()

    tree_files = sorted(glob.glob(str(REPO_ROOT / "trees" / "*.json")))
    if args.include_fixtures:
        fixture_files = sorted(glob.glob(str(REPO_ROOT / "benchmarks" / "fixtures" / "*" / "*.json")))
        seen = {str(Path(p).resolve()) for p in tree_files}
        for fp in fixture_files:
            if str(Path(fp).resolve()) not in seen:
                tree_files.append(fp)

    if not tree_files:
        print("[lint_trees] ERROR: No tree files found in trees/*.json")
        return 1

    total_violations = 0
    print(f"[lint_trees] Linting {len(tree_files)} domain trees...")
    for path in tree_files:
        violations = lint_tree_file(path)
        if violations:
            total_violations += len(violations)
            for v in violations:
                print(f"  VIOLATION: {v}")

    if total_violations > 0:
        print(f"\n[lint_trees] FAILED: {total_violations} violation(s) across {len(tree_files)} tree(s).")
        return 1
    print(f"\n[lint_trees] SUCCESS: All {len(tree_files)} domain trees passed lint checks cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
