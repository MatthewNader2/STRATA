#!/usr/bin/env python3
"""
tools/verify_macros.py

Phase 2 Macro Topology Validator.
Loads all JSON macro files in `trees/macro/` and validates every MacroCell against live `trees/lattice.db`:

Validation Rules:
1. Existence: Every ID in macro.sub_cells MUST exist as a valid MicroCell in lattice.db.
2. Topology Completeness: Every node in macro.sub_cells must be connected in internal_topology (except terminal sink nodes).
3. Typestate Flow Consistency: For every directed edge (A -> B) in internal_topology:
   Assert AlgebraicSignature(A.output_type, A.output_state).matches(
       AlgebraicSignature(B.input_type, B.input_state)
   ) is True.
4. Semantic Integrity: The macro's declared input_type/input_state must match the first sub-cell's input_type/input_state,
   and the macro's declared output_type/output_state must match the terminal sub-cell's output_type/output_state.
"""

import glob
import json
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lattice import AlgebraicSignature


def get_micro_cell(cell_id: str, conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT cell_id, input_type, input_state, output_type, output_state FROM nodes WHERE cell_id = ?",
        (cell_id,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "cell_id": row[0],
        "input_type": row[1],
        "input_state": row[2],
        "output_type": row[3],
        "output_state": row[4]
    }


def verify_macro_cell(macro: dict, conn: sqlite3.Connection) -> list:
    errors = []
    macro_id = macro.get("cell_id", "UNKNOWN_MACRO")
    sub_cells = macro.get("sub_cells", [])
    internal_topology = macro.get("internal_topology", {})

    if not sub_cells:
        errors.append(f"[{macro_id}] Macro has empty sub_cells list.")
        return errors

    # 1. Existence Check
    fetched_micro_cells = {}
    for sc_id in sub_cells:
        mc = get_micro_cell(sc_id, conn)
        if not mc:
            errors.append(f"[{macro_id}] Rule 1 (Existence) Failure: Sub-cell '{sc_id}' does not exist in lattice.db.")
        else:
            fetched_micro_cells[sc_id] = mc

    if len(fetched_micro_cells) != len(sub_cells):
        # Stop further validation for this macro if sub-cells are missing
        return errors

    # 2. Topology Completeness Check
    all_targets = set()
    for src, targets in internal_topology.items():
        if src not in sub_cells:
            errors.append(f"[{macro_id}] Rule 2 (Topology) Failure: Topology source '{src}' is not in sub_cells list.")
        for tgt in targets:
            if tgt not in sub_cells:
                errors.append(f"[{macro_id}] Rule 2 (Topology) Failure: Topology target '{tgt}' is not in sub_cells list.")
            all_targets.add(tgt)

    # Determine start node(s) and sink node(s)
    sources = set(internal_topology.keys())
    sink_nodes = [sc for sc in sub_cells if sc not in sources or not internal_topology.get(sc)]
    start_nodes = [sc for sc in sub_cells if sc not in all_targets]

    if not start_nodes:
        start_nodes = [sub_cells[0]]
    if not sink_nodes:
        sink_nodes = [sub_cells[-1]]

    # Check non-sink nodes have outgoing topology connections
    for sc in sub_cells:
        if sc not in sink_nodes and sc not in internal_topology:
            errors.append(f"[{macro_id}] Rule 2 (Topology) Failure: Intermediate node '{sc}' has no outgoing edges in internal_topology.")

    # 3. Typestate Flow Consistency Check
    for src_id, targets in internal_topology.items():
        src_cell = fetched_micro_cells[src_id]
        sig_src_out = AlgebraicSignature.from_string(src_cell["output_type"], src_cell["output_state"])

        for tgt_id in targets:
            tgt_cell = fetched_micro_cells[tgt_id]
            sig_tgt_in = AlgebraicSignature.from_string(tgt_cell["input_type"], tgt_cell["input_state"])

            if not sig_src_out.matches(sig_tgt_in):
                errors.append(
                    f"[{macro_id}] Rule 3 (Typestate Flow) Failure on edge ({src_id} -> {tgt_id}): "
                    f"Source output '{src_cell['output_type']}:{src_cell['output_state']}' does not match "
                    f"Target input '{tgt_cell['input_type']}:{tgt_cell['input_state']}'."
                )

    # 4. Semantic Integrity Check
    macro_inputs = macro.get("inputs", {})
    macro_outputs = macro.get("outputs", {})

    macro_in_sig = AlgebraicSignature.from_string(
        macro_inputs.get("type_name", macro_inputs.get("type", "")),
        macro_inputs.get("state", "any")
    )
    macro_out_sig = AlgebraicSignature.from_string(
        macro_outputs.get("type_name", macro_outputs.get("type", "")),
        macro_outputs.get("state", "any")
    )

    start_cell = fetched_micro_cells[start_nodes[0]]
    start_in_sig = AlgebraicSignature.from_string(start_cell["input_type"], start_cell["input_state"])

    sink_cell = fetched_micro_cells[sink_nodes[-1]]
    sink_out_sig = AlgebraicSignature.from_string(sink_cell["output_type"], sink_cell["output_state"])

    if not macro_in_sig.matches(start_in_sig):
        errors.append(
            f"[{macro_id}] Rule 4 (Semantic Integrity) Failure: Declared macro input "
            f"'{macro_inputs.get('type_name')}:{macro_inputs.get('state')}' does not match first sub-cell "
            f"'{start_cell['cell_id']}' input '{start_cell['input_type']}:{start_cell['input_state']}'."
        )

    if not sink_out_sig.matches(macro_out_sig):
        errors.append(
            f"[{macro_id}] Rule 4 (Semantic Integrity) Failure: Terminal sub-cell "
            f"'{sink_cell['cell_id']}' output '{sink_cell['output_type']}:{sink_cell['output_state']}' "
            f"does not match declared macro output '{macro_outputs.get('type_name')}:{macro_outputs.get('state')}'."
        )

    return errors


def verify_all_macros(db_path: str = None) -> bool:
    if not db_path:
        db_path = os.path.join(PROJECT_ROOT, "trees", "lattice.db")

    if not os.path.exists(db_path):
        print(f"[-] Error: SQLite database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    trees_dir = os.path.join(PROJECT_ROOT, "trees")
    macro_files = glob.glob(os.path.join(trees_dir, "macro", "*.json"))
    if os.path.exists(os.path.join(trees_dir, "macro_tree.json")):
        macro_files.append(os.path.join(trees_dir, "macro_tree.json"))

    if not macro_files:
        print("[-] Warning: No macro JSON files found in trees/")
        return False

    total_macros = 0
    passed_macros = 0
    all_errors = []

    print(f"[*] Validating Macro Trees across {len(macro_files)} files against {os.path.basename(db_path)}...")

    for mf in macro_files:
        filename = os.path.basename(mf)
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            cells = data.get("cells", data) if isinstance(data, dict) else data
            if not isinstance(cells, list):
                continue

            for macro in cells:
                if not isinstance(macro, dict) or macro.get("type") != "macro":
                    continue
                total_macros += 1
                errors = verify_macro_cell(macro, conn)
                if errors:
                    all_errors.extend(errors)
                    print(f"  [X] {macro.get('cell_id')} - FAILED ({len(errors)} errors)")
                    for err in errors:
                        print(f"      - {err}")
                else:
                    passed_macros += 1
                    print(f"  [v] {macro.get('cell_id')} - PASSED")
        except Exception as e:
            all_errors.append(f"Failed to read/parse {filename}: {e}")

    conn.close()

    print("\n" + "=" * 60)
    print(f"Macro Validation Summary: {passed_macros}/{total_macros} Macros PASSED")
    print("=" * 60)

    if all_errors:
        print(f"\n[-] {len(all_errors)} Validation Errors Found:")
        for err in all_errors:
            print(f"  - {err}")
        return False

    print("\n[+] 100% of Macro Cells passed all 4 validation rules successfully!")
    return True


if __name__ == "__main__":
    success = verify_all_macros()
    sys.exit(0 if success else 1)
