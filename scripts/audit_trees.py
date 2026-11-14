#!/usr/bin/env python3
"""
scripts/audit_trees.py - Universal 5-Axis & Schema Integrity Auditor for trees/*.json.

Checks all domain trees for:
  1. TreeSchema Pydantic compliance.
  2. Cross-domain typestate collisions (same bare state name with conflicting carrier/parent/properties).
  3. Undeclared port typestates (ports referencing states missing from typestates declarations).
  4. Manual source_priority overrides (magic priority bias != 100).
  5. 5-axis / relational / effect completeness and legacy accepted_states / hardcoded layout smells.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from schema import TreeSchema  # noqa: E402


def _iter_port_entries(cell: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    entries: List[Tuple[str, str, Dict[str, Any]]] = []
    for direction in ("inputs", "outputs"):
        ports = cell.get(direction) or {}
        if isinstance(ports, dict):
            for pname, pval in ports.items():
                if isinstance(pval, dict):
                    entries.append((direction, pname, pval))
        elif isinstance(ports, list):
            for idx, pval in enumerate(ports):
                if isinstance(pval, dict):
                    entries.append((direction, str(pval.get("name") or f"port_{idx}"), pval))
    return entries


def audit_directory(trees_dir: Path, strict_five_axis: bool = False) -> Dict[str, Any]:
    files = sorted(glob.glob(str(trees_dir / "*.json")))
    declared_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}
    state_defs_global: Dict[str, List[Tuple[str, Any, Any, Any]]] = collections.defaultdict(list)
    all_declared_states: Set[str] = set()

    parsed_trees: List[Tuple[Path, Dict[str, Any]]] = []
    schema_errors: List[str] = []

    for fpath in files:
        p = Path(fpath)
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        try:
            TreeSchema(**raw)
        except Exception as exc:
            schema_errors.append(f"{p.name}: {exc}")

        dom = str(raw.get("domain") or p.stem).strip().lower()
        dom_states: Dict[str, Dict[str, Any]] = {}
        ts_block = raw.get("typestates") or {}
        states_list = ts_block.get("states", []) if isinstance(ts_block, dict) else (ts_block if isinstance(ts_block, list) else [])
        for entry in states_list:
            if isinstance(entry, dict) and entry.get("name"):
                sname = str(entry["name"]).strip().lower()
                dom_states[sname] = entry
                all_declared_states.add(sname)
                state_defs_global[sname].append((
                    dom,
                    entry.get("carrier_type"),
                    entry.get("parent_state"),
                    entry.get("properties") or {},
                ))
            elif isinstance(entry, str) and entry.strip():
                sname = entry.strip().lower()
                dom_states[sname] = {"name": sname}
                all_declared_states.add(sname)
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

    per_domain_report: Dict[str, Dict[str, Any]] = {}
    totals = collections.Counter()

    for p, raw in parsed_trees:
        dom = str(raw.get("domain") or p.stem).strip().lower()
        cells = raw.get("cells") or []
        d_counts = collections.Counter()
        undeclared_samples: List[str] = []
        five_axis_ports = 0
        total_ports = 0

        for cell in cells:
            cid = cell.get("cell_id", "<unknown>")
            sp = cell.get("source_priority")
            if sp is not None and sp != 100:
                d_counts["manual_source_priority"] += 1

            role = str(cell.get("node_role") or cell.get("role") or "").lower()
            if role == "bridge":
                d_counts["bridge_cells"] += 1
                if not cell.get("bridge_profile"):
                    d_counts["bridges_missing_profile"] += 1

            has_effects = ("effects" in cell or "postconditions" in cell) and ("mutation_type" in cell)
            if not has_effects:
                d_counts["cells_missing_effects"] += 1

            for direction, pname, port in _iter_port_entries(cell):
                total_ports += 1
                st = str(port.get("state") or "").strip().lower()
                if not st:
                    d_counts["ports_missing_state"] += 1
                elif st not in ("any", "*", "default") and st not in declared_by_domain.get(dom, {}) and st not in all_declared_states:
                    d_counts["ports_undeclared_state"] += 1
                    if len(undeclared_samples) < 5:
                        undeclared_samples.append(f"{cid}.{pname}={st}")

                if port.get("accepted_states"):
                    d_counts["ports_with_accepted_states"] += 1

                if all(k in port for k in ("carrier", "layout", "semantics", "typestate", "invariants")):
                    five_axis_ports += 1
                    ts_val = str(port.get("typestate") or "").strip().lower()
                    if st in colliding_states and "::" not in ts_val and ts_val != "any":
                        d_counts["unnamespaced_colliding_ports"] += 1

        d_counts["total_cells"] = len(cells)
        d_counts["total_ports"] = total_ports
        d_counts["five_axis_ports"] = five_axis_ports
        totals.update(d_counts)
        per_domain_report[dom] = {
            "file": p.name,
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
    parser = argparse.ArgumentParser(description="Audit trees/*.json for 5-axis & universal schema integrity.")
    parser.add_argument("--trees-dir", type=Path, default=ROOT / "trees", help="Path to trees directory")
    parser.add_argument("--strict-five-axis", action="store_true", help="Fail if any port lacks explicit 5-axis fields")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    rep = audit_directory(args.trees_dir, strict_five_axis=args.strict_five_axis)
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
                f"prio_bias={c.get('manual_source_priority', 0):2d} "
                f"acc_st={c.get('ports_with_accepted_states', 0):3d} "
                f"undec_st={c.get('ports_undeclared_state', 0):3d} "
                f"miss_eff={c.get('cells_missing_effects', 0):3d} "
                f"br_no_prof={c.get('bridges_missing_profile', 0):2d}"
            )

    if rep["schema_errors"]:
        return 1
    if args.strict_five_axis:
        t = rep["totals"]
        if (
            t.get("five_axis_ports", 0) < t.get("total_ports", 0)
            or t.get("manual_source_priority", 0) > 0
            or t.get("unnamespaced_colliding_ports", 0) > 0
            or t.get("ports_undeclared_state", 0) > 0
            or t.get("cells_missing_effects", 0) > 0
            or t.get("bridges_missing_profile", 0) > 0
        ):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
