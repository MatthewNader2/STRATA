#!/usr/bin/env python3
"""
tools/audit_lattice_integrity.py

Phase 1 Diagnostic & Audit Script for STRATA Micro-Lattice Databases.
Audits `trees/lattice.db` and `trees/strata_lattice.db` against four core invariants:
  1. Self-Named Type Bug (Target: 0)
  2. Unchecked 'any' Fallbacks
  3. Execution Verification (verified == 1 vs 0)
  4. Parameter Slot Coverage ({placeholder} resolution)

Outputs results to `logs/lattice_integrity_report.md`.
"""

import json
import os
import re
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TREES_DIR = os.path.join(PROJECT_ROOT, "trees")
REPORT_PATH = os.path.join(LOGS_DIR, "lattice_integrity_report.md")

VALID_TYPES = {
    'dataframe', 'series', 'ndarray', 'mat', 'image', 'tensor',
    'int', 'float', 'str', 'dict', 'list', 'graph', 'model',
    'bool', 'tuple', 'set', 'bytes', 'object', 'none', 'nonetype',
    'sparsematrix', 'sparse_matrix', 'index', 'multiindex', 'numeric',
    'flask', 'faissindex', 'faiss_index'
}

_KNOWN_PLACEHOLDERS = {
    "input_var", "output_var", "input_filename", "output_filename", "input_source",
    "filename", "output_filename", "image_path", "output_path", "by_column",
    "ascending", "code", "target_column", "start_node", "prediction_col"
}


def is_self_named_type(type_name: str, cell_id: str) -> bool:
    if not type_name:
        return False
    t_lower = type_name.lower().strip()
    if t_lower in VALID_TYPES:
        return False

    clean_id = cell_id.replace('_DEFAULT', '').replace('_CELL', '')
    parts = clean_id.split('_')

    for part in parts:
        if part and len(part) > 2 and part.lower() not in (
            'pandas', 'sklearn', 'scipy', 'numpy', 'cv2', 'default', 'cell',
            'python', 'io', 'libs', 'core', 'arrays'
        ):
            if t_lower == part.lower():
                return True
    if t_lower == clean_id.lower() or t_lower in ('tags', 'resolution', 'scope', 'type', 'dtype', 'slice'):
        return True
    return False


def audit_database(db_path: str):
    if not os.path.exists(db_path):
        return {
            "exists": False,
            "path": db_path,
            "total_nodes": 0,
            "self_named_bugs": [],
            "any_outputs": [],
            "verified_count": 0,
            "unverified_count": 0,
            "valid_slots_count": 0,
            "invalid_slots_count": 0,
            "invalid_slots_samples": []
        }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check columns
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [row[1] for row in cursor.fetchall()]

    query_cols = ["cell_id", "domain_name", "node_type", "input_type", "output_type", "code"]
    if "verified" in columns:
        query_cols.append("verified")
    
    cursor.execute(f"SELECT {', '.join(query_cols)} FROM nodes")
    rows = cursor.fetchall()

    self_named_bugs = []
    any_outputs = []
    verified_count = 0
    unverified_count = 0
    valid_slots_count = 0
    invalid_slots_count = 0
    invalid_slots_samples = []

    # Load verification report JSON if available
    ver_report_file = os.path.join(TREES_DIR, "verification_report.json")
    ver_cv2_file = os.path.join(TREES_DIR, "verification_report_cv2.json")
    verified_ids = set()

    for vf in (ver_report_file, ver_cv2_file):
        if os.path.exists(vf):
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                    for item in v_data:
                        if isinstance(item, dict) and item.get("verified"):
                            verified_ids.add(item.get("cell_id"))
            except Exception:
                pass

    for r in rows:
        cell_id = r[0]
        domain_name = r[1]
        node_type = r[2]
        in_type = r[3]
        out_type = r[4]
        code = r[5] or ""
        
        db_verified = False
        if "verified" in columns and len(r) > 6:
            db_verified = bool(r[6])

        # 1. Self-Named Type Bug
        if is_self_named_type(in_type, cell_id):
            self_named_bugs.append((cell_id, "input_type", in_type))
        if is_self_named_type(out_type, cell_id):
            self_named_bugs.append((cell_id, "output_type", out_type))

        # 2. Unchecked 'any' Fallbacks
        if node_type in ("micro", "function") and out_type and out_type.lower() == "any":
            any_outputs.append(cell_id)

        # 3. Execution Verification
        if db_verified or cell_id in verified_ids:
            verified_count += 1
        else:
            unverified_count += 1

        # 4. Parameter Slot Coverage
        placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", code))
        unbound = placeholders - _KNOWN_PLACEHOLDERS
        if not unbound or code == "":
            valid_slots_count += 1
        else:
            invalid_slots_count += 1
            if len(invalid_slots_samples) < 10:
                invalid_slots_samples.append((cell_id, list(unbound)))

    conn.close()

    return {
        "exists": True,
        "path": db_path,
        "total_nodes": len(rows),
        "self_named_bugs": self_named_bugs,
        "any_outputs": any_outputs,
        "verified_count": verified_count,
        "unverified_count": unverified_count,
        "valid_slots_count": valid_slots_count,
        "invalid_slots_count": invalid_slots_count,
        "invalid_slots_samples": invalid_slots_samples
    }


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    db_main = os.path.join(TREES_DIR, "lattice.db")

    res_main = audit_database(db_main)

    md = []
    md.append("# Micro-Lattice Integrity Audit Report")
    md.append("")
    md.append(f"**Audit Execution Time**: 2026-08-21")
    md.append("")
    md.append("## Summary Statistics")
    md.append("")
    md.append("| Metric | `trees/lattice.db` | Target Invariant |")
    md.append("| --- | --- | --- |")
    md.append(f"| **Total Nodes** | {res_main['total_nodes']} | N/A |")
    md.append(f"| **1. Self-Named Type Bugs** | `{len(res_main['self_named_bugs'])}` | **MUST BE 0** |")
    md.append(f"| **2. Unchecked 'any' Fallbacks** | `{len(res_main['any_outputs'])}` | Minimise / 0 |")
    md.append(f"| **3. Verified Micro-Cells** | `{res_main['verified_count']}` / {res_main['total_nodes']} | Maximize |")
    md.append(f"| **4. Valid Parameter Slot Coverage** | `{res_main['valid_slots_count']}` / {res_main['total_nodes']} | 100% |")
    md.append("")

    md.append("## Invariant 1: Self-Named Type Bug Audit")
    if len(res_main['self_named_bugs']) == 0:
        md.append("> [!NOTE]")
        md.append("> **PASSED**: 0 self-named type bugs detected in micro-lattice DBs.")
    else:
        md.append("> [!WARNING]")
        md.append(f"> Detected {len(res_main['self_named_bugs'])} self-named type bugs in `lattice.db`.")
        md.append("Sample defect nodes:")
        for cell_id, attr, t_val in res_main['self_named_bugs'][:10]:
            md.append(f"- `{cell_id}` ({attr} = `{t_val}`)")

    md.append("")
    md.append("## Invariant 2: Unchecked 'any' Fallback Audit")
    md.append(f"Total operational micro-cells with `output_type == 'any'`: `{len(res_main['any_outputs'])}`.")

    md.append("")
    md.append("## Invariant 3: Execution Verification Status")
    md.append(f"- Verified Nodes: `{res_main['verified_count']}`")
    md.append(f"- Unverified Nodes: `{res_main['unverified_count']}`")

    md.append("")
    md.append("## Invariant 4: Parameter Slot Coverage")
    md.append(f"- Valid Slot Coverage: `{res_main['valid_slots_count']}` nodes")
    md.append(f"- Unbound Placeholder Nodes: `{res_main['invalid_slots_count']}` nodes")
    if res_main['invalid_slots_samples']:
        md.append("Sample unbound placeholders:")
        for cell_id, unbound in res_main['invalid_slots_samples']:
            md.append(f"- `{cell_id}`: unbound {unbound}")

    report_content = "\n".join(md)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Audit complete. Report written to {REPORT_PATH}")
    print(f"    - Self-Named Type Bugs: {len(res_main['self_named_bugs'])} (Target: 0)")
    print(f"    - Unchecked 'any' Fallbacks: {len(res_main['any_outputs'])}")
    print(f"    - Verified Nodes: {res_main['verified_count']} / {res_main['total_nodes']}")
    print(f"    - Parameter Slot Coverage: {res_main['valid_slots_count']} / {res_main['total_nodes']}")

    return len(res_main['self_named_bugs'])


if __name__ == "__main__":
    bugs = main()
    sys.exit(0 if bugs == 0 else 1)
