#!/usr/bin/env python3
"""
scripts/audit_trees.py — Universal Canonical v2 & Domain-Neutrality Auditor for trees/*.json.

Checks all domain trees for:
  1. Strict canonical v2 `TreeSchema` (`extra="forbid"`) conformance and zero legacy keys.
  2. Cross-domain typestate collisions (same bare name with conflicting carrier/parent/properties).
  3. Undeclared port typestates / accepted_states (resolved strictly via own or declared imports).
  4. Completeness of canonical 5-axis port contracts (`carrier`, `typestate`, `accepted_states`,
     `layout`, `invariants`) plus `value_class` and `resource_mode`.
  5. Strict domain neutrality: zero foreign domain metadata leaks or undeclared foreign imports,
     with domain alias resolution inferred dynamically from loaded dependencies (no domain hardcodes).
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from schema import (  # noqa: E402
    CellSchema,
    LayoutContract,
    PortSchema,
    TOP_STATES,
    TreeSchema,
)

CANONICAL_TREE_KEYS: Set[str] = set(TreeSchema.model_fields.keys())
CANONICAL_CELL_KEYS: Set[str] = set(CellSchema.model_fields.keys())
CANONICAL_PORT_KEYS: Set[str] = set(PortSchema.model_fields.keys())
CANONICAL_LAYOUT_KEYS: Set[str] = set(LayoutContract.model_fields.keys())
VALID_VALUE_CLASSES: Set[str] = {"data", "literal", "external_asset", "identifier", "enum"}
VALID_RESOURCE_MODES: Set[str] = {"value", "acquire", "borrow", "release"}
UNIVERSAL_STATES: Set[str] = set(TOP_STATES)


def _extract_meta_tokens(obj: Any) -> List[str]:
    tokens: List[str] = []
    if isinstance(obj, str):
        buf = "".join(ch if (ch.isalnum() or ch == "_") else " " for ch in obj.lower())
        tokens.extend(tok for tok in buf.split() if tok)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tokens.extend(_extract_meta_tokens(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            tokens.extend(_extract_meta_tokens(v))
    return tokens


def _parse_dep_roots(deps: List[Any]) -> Set[str]:
    roots: Set[str] = set()
    for raw in deps or ():
        s = str(raw or "").strip()
        if not s:
            continue
        if not s.startswith(("import ", "from ")):
            s = f"import {s}"
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".", 1)[0].strip().lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0].strip().lower())
    return roots


def _build_dynamic_alias_map(parsed_trees: List[Tuple[Path, Dict[str, Any]]]) -> Dict[str, str]:
    """Infer peer package aliases strictly from loaded domain names and dependency roots."""
    known_domains: Set[str] = set()
    dom_dep_counts: Dict[str, Dict[str, int]] = {}
    for p, raw in parsed_trees:
        dom = str(raw.get("domain") or p.stem).strip().lower()
        if not dom:
            continue
        known_domains.add(dom)
        counts: Dict[str, int] = {}
        for c in raw.get("cells") or ():
            if isinstance(c, dict):
                for root in _parse_dep_roots(c.get("dependencies") or ()):
                    counts[root] = counts.get(root, 0) + 1
        dom_dep_counts[dom] = counts

    alias_to_dom: Dict[str, str] = {}
    core_exempt = {"python_core", "control_flow"}
    for dom in known_domains:
        if dom not in core_exempt:
            alias_to_dom[dom] = dom

    for dom, counts in dom_dep_counts.items():
        if dom in core_exempt or not counts:
            continue
        foreign_only = {r: cnt for r, cnt in counts.items() if r not in known_domains}
        if foreign_only:
            top_pkg = max(foreign_only, key=lambda r: (foreign_only[r], r))
            if counts.get(dom, 0) == 0 and foreign_only[top_pkg] >= max(1, sum(foreign_only.values()) // 2):
                alias_to_dom.setdefault(top_pkg, dom)
    return alias_to_dom


def _iter_port_entries(cell: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    entries: List[Tuple[str, str, Dict[str, Any]]] = []
    for direction in ("inputs", "outputs"):
        ports = cell.get(direction) or {}
        if isinstance(ports, dict):
            for pname, pval in ports.items():
                if isinstance(pval, dict):
                    entries.append((direction, str(pname), pval))
    return entries


def audit_directory(
    trees_dir: Path,
    include_fixtures: bool = False,
) -> Dict[str, Any]:
    files = sorted(glob.glob(str(trees_dir / "*.json")))
    if include_fixtures:
        fixture_files = sorted(glob.glob(str(ROOT / "benchmarks" / "fixtures" / "*" / "*.json")))
        seen_paths = {str(Path(p).resolve()) for p in files}
        for fp in fixture_files:
            if str(Path(fp).resolve()) not in seen_paths:
                files.append(fp)

    declared_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}
    state_defs_global: Dict[str, List[Tuple[str, Any, Any, Any]]] = collections.defaultdict(list)
    parsed_trees: List[Tuple[Path, Dict[str, Any]]] = []
    schema_errors: List[str] = []

    for fpath in files:
        p = Path(fpath)
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            schema_errors.append(f"{p.name}: root JSON must be an object")
            continue

        extra_tree = sorted(set(raw.keys()) - CANONICAL_TREE_KEYS)
        if extra_tree:
            schema_errors.append(f"{p.name}: non-canonical tree key(s) {extra_tree}")

        try:
            TreeSchema(**raw)
        except Exception as exc:
            schema_errors.append(f"{p.name}: {exc}")

        dom = str(raw.get("domain") or p.stem).strip().lower()
        dom_states: Dict[str, Dict[str, Any]] = {}
        ts_block = raw.get("typestates") or {}
        states_list = ts_block.get("states", []) if isinstance(ts_block, dict) else []
        for entry in states_list:
            if isinstance(entry, dict) and entry.get("name"):
                sname = str(entry["name"]).strip().lower()
                dom_states[sname] = entry
                state_defs_global[sname].append((
                    dom,
                    entry.get("carrier_type"),
                    entry.get("parent_state"),
                    entry.get("properties") or {},
                ))
            elif isinstance(entry, str) and entry.strip():
                sname = entry.strip().lower()
                dom_states[sname] = {"name": sname}
                state_defs_global[sname].append((dom, None, None, {}))

        declared_by_domain[dom] = dom_states
        parsed_trees.append((p, raw))

    colliding_states: Dict[str, List[Tuple[str, Any, Any, Any]]] = {}
    for sname, defs in sorted(state_defs_global.items()):
        if len(defs) > 1:
            carriers = {str(d[1] or "").lower() for d in defs}
            parents = {str(d[2] or "").lower() for d in defs}
            props = {json.dumps(d[3], sort_keys=True) for d in defs}
            if len(carriers) > 1 or len(parents) > 1 or len(props) > 1:
                colliding_states[sname] = defs

    alias_to_peer = _build_dynamic_alias_map(parsed_trees)
    per_domain_report: Dict[str, Dict[str, Any]] = {}
    totals: collections.Counter[str] = collections.Counter()

    for p, raw in parsed_trees:
        dom = str(raw.get("domain") or p.stem).strip().lower()
        cells = raw.get("cells") or []
        d_counts: collections.Counter[str] = collections.Counter()
        undeclared_samples: List[str] = []
        five_axis_ports = 0
        total_ports = 0
        raw_imports = raw.get("imports")
        imports_declared = isinstance(raw_imports, list)
        allowed_foreign_domains: Set[str] = {dom, "python_core"}
        if imports_declared:
            allowed_foreign_domains.update(
                str(imp).strip().lower() for imp in raw_imports if str(imp).strip()
            )

        for cell in cells:
            if not isinstance(cell, dict):
                continue
            cid = str(cell.get("cell_id") or "<unknown>")
            extra_cell = sorted(set(cell.keys()) - CANONICAL_CELL_KEYS)
            if extra_cell:
                d_counts["legacy_cell_keys"] += len(extra_cell)

            has_effects = isinstance(cell.get("effects"), list)
            mut = str(cell.get("mutation_type") or "").strip().lower()
            if not (has_effects and mut):
                d_counts["cells_missing_effects"] += 1

            bp = cell.get("bridge_profile")
            if bp is not None:
                d_counts["bridge_cells"] += 1
                if not isinstance(bp, dict) or not bp:
                    d_counts["bridges_missing_profile"] += 1

            ins_d = cell.get("inputs") or {}
            outs_d = cell.get("outputs") or {}
            if not ins_d and mut == "pure" and isinstance(outs_d, dict):
                for op in outs_d.values():
                    if isinstance(op, dict) and str(op.get("resource_mode") or "").lower() in ("borrow", "release"):
                        d_counts["zero_in_pure_borrowed_handle"] += 1
                        break

            for _, pname, port in _iter_port_entries(cell):
                total_ports += 1
                extra_port = sorted(set(port.keys()) - CANONICAL_PORT_KEYS)
                if extra_port:
                    d_counts["legacy_port_keys"] += len(extra_port)

                vc = str(port.get("value_class") or "").strip().lower()
                rm = str(port.get("resource_mode") or "").strip().lower()
                if vc in VALID_VALUE_CLASSES:
                    d_counts["explicit_value_class_ports"] += 1
                else:
                    d_counts["invalid_value_class_ports"] += 1
                if rm in VALID_RESOURCE_MODES:
                    d_counts["explicit_resource_mode_ports"] += 1
                else:
                    d_counts["invalid_resource_mode_ports"] += 1

                if imports_declared:
                    state_refs: List[Any] = [port.get("typestate")]
                    state_refs.extend(port.get("accepted_states") or ())
                    for s_ref in state_refs:
                        s_ref_s = str(s_ref or "").strip().lower()
                        if "::" in s_ref_s:
                            dom_prefix = s_ref_s.split("::", 1)[0].strip().lower()
                            if dom_prefix and dom_prefix not in allowed_foreign_domains:
                                d_counts["undeclared_foreign_imports"] += 1

                carrier_val = port.get("carrier")
                typestate_val = port.get("typestate")
                layout_present = "layout" in port
                layout_val = port.get("layout")
                invariants_val = port.get("invariants")

                carrier_ok = isinstance(carrier_val, str) and bool(carrier_val.strip())
                typestate_ok = isinstance(typestate_val, str) and bool(typestate_val.strip())
                layout_ok = layout_present and (
                    layout_val is None
                    or (isinstance(layout_val, dict) and set(layout_val.keys()) <= CANONICAL_LAYOUT_KEYS)
                )
                invariants_ok = isinstance(invariants_val, list)

                if not carrier_ok:
                    d_counts["ports_empty_carrier"] += 1
                if not typestate_ok:
                    d_counts["ports_empty_typestate"] += 1
                if not layout_ok:
                    d_counts["ports_invalid_layout"] += 1
                if not invariants_ok:
                    d_counts["ports_invalid_invariants"] += 1

                if typestate_ok:
                    tsv = str(typestate_val).strip().lower()
                    ts_bare = tsv.split("::", 1)[1] if "::" in tsv else tsv
                    if tsv not in UNIVERSAL_STATES and ts_bare not in UNIVERSAL_STATES:
                        if "::" in tsv:
                            ts_dom = tsv.split("::", 1)[0].strip().lower()
                            if ts_bare not in declared_by_domain.get(ts_dom, {}):
                                d_counts["ports_undeclared_state"] += 1
                                if len(undeclared_samples) < 5:
                                    undeclared_samples.append(f"{cid}.{pname}:typestate={tsv}")
                        elif tsv not in declared_by_domain.get(dom, {}):
                            d_counts["ports_undeclared_state"] += 1
                            if len(undeclared_samples) < 5:
                                undeclared_samples.append(f"{cid}.{pname}:typestate={tsv}")
                        if ts_bare in colliding_states and "::" not in tsv:
                            d_counts["unnamespaced_colliding_ports"] += 1

                for tok in _extract_meta_tokens(invariants_val):
                    tgt_dom = alias_to_peer.get(tok)
                    if tgt_dom and tgt_dom not in allowed_foreign_domains:
                        d_counts["foreign_domain_metadata_leaks"] += 1

                if port.get("accepted_states"):
                    d_counts["ports_with_accepted_states"] += 1

                if carrier_ok and typestate_ok and layout_ok and invariants_ok and vc in VALID_VALUE_CLASSES and rm in VALID_RESOURCE_MODES:
                    five_axis_ports += 1

                car_low = str(carrier_val or "").strip().lower()
                if (car_low.startswith("list[") or car_low in ("list", "sequence")) and isinstance(layout_val, dict) and layout_val.get("rank") == 0:
                    d_counts["ports_sequence_marked_rank_zero"] += 1

        d_counts["total_cells"] = len(cells)
        d_counts["total_ports"] = total_ports
        d_counts["five_axis_ports"] = five_axis_ports
        d_counts["imports_declared"] = 1 if imports_declared else 0
        totals.update(d_counts)
        per_domain_report[dom] = {
            "file": p.name,
            "imports_declared": imports_declared,
            "counts": dict(d_counts),
            "undeclared_samples": undeclared_samples,
        }

    return {
        "files_scanned": len(parsed_trees),
        "schema_errors": schema_errors,
        "colliding_states": colliding_states,
        "totals": dict(totals),
        "per_domain": per_domain_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit trees/*.json for canonical v2 5-axis & neutrality integrity.")
    parser.add_argument("--trees-dir", type=Path, default=ROOT / "trees", help="Path to trees directory")
    parser.add_argument("--strict-five-axis", action="store_true", help="Fail if any port lacks explicit canonical 5-axis fields")
    parser.add_argument("--strict-neutral", action="store_true", help="Fail on any legacy keys, undeclared foreign imports, collisions, or leaks")
    parser.add_argument("--include-fixtures", action="store_true", help="Also audit held-out fixture trees under benchmarks/fixtures/*/*.json")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    rep = audit_directory(args.trees_dir, include_fixtures=args.include_fixtures)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"Scanned {rep['files_scanned']} tree files")
        print(f"Schema errors: {len(rep['schema_errors'])}")
        for err in rep["schema_errors"]:
            print(f"  [SCHEMA ERROR] {err}")
        print(f"Cross-domain colliding typestates: {len(rep['colliding_states'])}")
        for st, defs in sorted(rep["colliding_states"].items()):
            print(f"  [COLLISION] {st}: {[d[0] for d in defs]}")
        print("Totals:", json.dumps(rep["totals"], sort_keys=True))
        print("\nPer-domain summary:")
        for dom, info in sorted(rep["per_domain"].items()):
            c = info["counts"]
            print(
                f"  {dom:14s} cells={c.get('total_cells', 0):3d} "
                f"ports={c.get('total_ports', 0):3d} "
                f"5axis={c.get('five_axis_ports', 0):3d} "
                f"legacy={c.get('legacy_cell_keys', 0) + c.get('legacy_port_keys', 0):2d} "
                f"bad_imp={c.get('undeclared_foreign_imports', 0):2d} "
                f"leak={c.get('foreign_domain_metadata_leaks', 0):2d} "
                f"undec_st={c.get('ports_undeclared_state', 0):2d} "
                f"miss_eff={c.get('cells_missing_effects', 0):2d} "
                f"bridges={c.get('bridge_cells', 0):2d}"
            )

    if rep["schema_errors"]:
        return 1
    t = rep["totals"]
    if args.strict_five_axis:
        if (
            t.get("five_axis_ports", 0) < t.get("total_ports", 0)
            or t.get("ports_empty_carrier", 0) > 0
            or t.get("ports_empty_typestate", 0) > 0
            or t.get("ports_invalid_layout", 0) > 0
            or t.get("ports_invalid_invariants", 0) > 0
            or t.get("invalid_value_class_ports", 0) > 0
            or t.get("invalid_resource_mode_ports", 0) > 0
            or t.get("unnamespaced_colliding_ports", 0) > 0
            or t.get("ports_undeclared_state", 0) > 0
            or t.get("cells_missing_effects", 0) > 0
            or t.get("bridges_missing_profile", 0) > 0
            or t.get("ports_sequence_marked_rank_zero", 0) > 0
            or t.get("zero_in_pure_borrowed_handle", 0) > 0
        ):
            return 2
    if args.strict_neutral:
        if (
            t.get("legacy_cell_keys", 0) > 0
            or t.get("legacy_port_keys", 0) > 0
            or t.get("undeclared_foreign_imports", 0) > 0
            or t.get("unnamespaced_colliding_ports", 0) > 0
            or t.get("foreign_domain_metadata_leaks", 0) > 0
        ):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
