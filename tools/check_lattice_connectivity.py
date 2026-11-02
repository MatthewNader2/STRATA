import sys
import os
import sqlite3
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unification import types_unify

def check_connectivity():
    db_path = PROJECT_ROOT / "trees" / "lattice.db"
    if not db_path.exists():
        print(f"[ERROR] Database file not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT cell_id, input_type, output_type, node_type FROM nodes")
    rows = cursor.fetchall()
    conn.close()

    print(f"[+] Loaded {len(rows)} nodes from {db_path.name}")

    input_types = set()
    output_types = set()

    for cell_id, in_t, out_t, node_type in rows:
        in_str = (in_t or "any").lower()
        out_str = (out_t or "any").lower()
        input_types.add(in_str)
        output_types.add(out_str)

    print(f"[+] Unique Input Types: {len(input_types)}")
    print(f"[+] Unique Output Types: {len(output_types)}")

    # Check output-to-input type reachability
    reachable_outputs = 0
    orphan_outputs = []

    for out_t in output_types:
        if out_t in ("any", "", "none"):
            reachable_outputs += 1
            continue
        
        can_reach = any(types_unify(out_t, in_t) for in_t in input_types)
        if can_reach:
            reachable_outputs += 1
        else:
            orphan_outputs.append(out_t)

    reachability_pct = (reachable_outputs / max(len(output_types), 1)) * 100.0
    print("\n" + "="*80)
    print(" LATTICE CONNECTIVITY HEALTH REPORT")
    print("="*80)
    print(f" Type Reachability Rate: {reachability_pct:.2f}% ({reachable_outputs}/{len(output_types)})")
    print(f" Total Orphan Output Types: {len(orphan_outputs)}")
    if orphan_outputs[:10]:
        print(" Sample Orphan Output Types:", orphan_outputs[:10])
    print("="*80)

    # Write health report to logs/lattice_connectivity_report.json
    report_file = PROJECT_ROOT / "logs" / "lattice_connectivity_report.json"
    report_data = {
        "total_nodes": len(rows),
        "unique_input_types": len(input_types),
        "unique_output_types": len(output_types),
        "reachability_percentage": reachability_pct,
        "orphan_output_types": orphan_outputs
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved connectivity report to: {report_file}")

if __name__ == "__main__":
    check_connectivity()
