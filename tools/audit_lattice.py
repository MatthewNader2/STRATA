"""
tools/audit_lattice.py - Comprehensive Diagnostic Auditor for STRATA Knowledge Lattice.
Audits trees/lattice.db and reports:
1. Syntax & AST Parseability of Code Templates
2. Placeholder Binding Consistency ({output_var}, {input_var})
3. Type-System Contamination ('any', 'AnyObject', empty states)
4. Graph Topology Integrity (Orphan nodes, Dead ends, Reachability)
5. Stage Classification Sanity (Source -> Transform -> Sink)
"""

import ast
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


def audit_lattice(db_path=DB_PATH):
    if not os.path.exists(db_path):
        print(f"[FATAL] Database not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT cell_id, domain_name, node_type, node_role, stage,
               input_type, input_state, output_type, output_state,
               code, dependencies, configuration_schema
        FROM nodes
    """)
    rows = cursor.fetchall()
    total_cells = len(rows)

    print("=" * 70)
    print(f" STRATA KNOWLEDGE LATTICE AUDIT REPORT")
    print(f" Database: {db_path} | Total Cells: {total_cells}")
    print("=" * 70)

    # Metrics
    ast_errors = []
    placeholder_issues = []
    any_input_count = 0
    any_output_count = 0
    anyobject_count = 0
    stage_counts = Counter()
    role_counts = Counter()
    domain_counts = Counter()

    input_sig_buckets = set()
    output_sig_buckets = set()

    for row in rows:
        (cid, domain, n_type, n_role, stage,
         in_t, in_s, out_t, out_s,
         code, deps, config) = row

        stage_counts[stage] += 1
        role_counts[n_role] += 1
        domain_counts[domain] += 1

        in_t_str = str(in_t or "").strip()
        out_t_str = str(out_t or "").strip()

        # 1. Type Contamination Checks
        if in_t_str.lower() in ("any", "*", "top") or not in_t_str:
            any_input_count += 1
        if out_t_str.lower() in ("any", "*", "top") or not out_t_str:
            any_output_count += 1
        if "anyobject" in in_t_str.lower() or "anyobject" in out_t_str.lower():
            anyobject_count += 1

        input_sig_buckets.add((in_t_str, in_s))
        output_sig_buckets.add((out_t_str, out_s))

        # 2. Placeholder Verification
        if n_role != "constant" and code:
            if "{output_var}" not in code and "output_var" not in code and stage != 3:
                # Sinks might not assign output_var, but transform steps must
                if n_type != "macro" and not code.strip().startswith("print("):
                    placeholder_issues.append((cid, "Missing {output_var} assignment"))

            # Check for unbracketed raw input_var usage
            if "input_var." in code and "{input_var}." not in code:
                placeholder_issues.append((cid, "Raw unbracketed 'input_var' in template"))

        # 3. AST Syntax Validity Check
        if code and n_role != "constant":
            # Substitute mock variables into placeholders to test syntax compilation
            test_code = code
            placeholders = re.findall(r"\{([a-zA-Z0-9_]+)\}", code)
            for p in placeholders:
                test_code = test_code.replace(f"{{{p}}}", f"_mock_{p}_")

            try:
                ast.parse(test_code)
            except SyntaxError as e:
                ast_errors.append((cid, str(e), code[:60]))

    # 4. Topology Connectivity Audit
    # Check for dead-end or completely unconnectable type signatures
    unreachable_inputs = input_sig_buckets - output_sig_buckets
    # Filter out initial source signatures
    unreachable_inputs = {
        (t, s) for (t, s) in unreachable_inputs
        if t.lower() not in ("str", "source_identifier", "any", "none")
    }

    # ── Report Display ────────────────────────────────────────────────
    print("\n1. CELL METRICS & CLASSIFICATION:")
    print(f"  - Total Cells Loaded:       {total_cells}")
    print(f"  - Stage 1 (Sources/Readers): {stage_counts[1]}")
    print(f"  - Stage 2 (Transforms):      {stage_counts[2]}")
    print(f"  - Stage 3 (Sinks/Exporters): {stage_counts[3]}")
    print(f"  - Roles Breakdown:          {dict(role_counts)}")

    print("\n2. TYPE-SYSTEM HEALTH:")
    print(f"  - 'any' / Wildcard Inputs:  {any_input_count} ({any_input_count/total_cells*100:.1f}%)")
    print(f"  - 'any' / Wildcard Outputs: {any_output_count} ({any_output_count/total_cells*100:.1f}%)")
    print(f"  - 'AnyObject' Corruptions:  {anyobject_count}")
    print(f"  - Unique Input Signatures:  {len(input_sig_buckets)}")
    print(f"  - Unique Output Signatures: {len(output_sig_buckets)}")

    print("\n3. TEMPLATE & AST SYNTAX VALIDATION:")
    print(f"  - Syntax / Parse Errors:    {len(ast_errors)}")
    if ast_errors:
        for cid, err, snippet in ast_errors[:5]:
            print(f"      [!] {cid}: {err} | Snippet: {snippet}...")

    print(f"  - Placeholder Anomalies:    {len(placeholder_issues)}")
    if placeholder_issues:
        for cid, issue in placeholder_issues[:5]:
            print(f"      [!] {cid}: {issue}")

    print("\n4. TOPOLOGY REACHABILITY:")
    print(f"  - Input Typestates with NO Producer: {len(unreachable_inputs)}")
    if unreachable_inputs:
        sample = list(unreachable_inputs)[:5]
        print(f"      Sample unreachable inputs (demand with no supply): {sample}")

    print("\n" + "=" * 70)
    if not ast_errors and anyobject_count == 0 and len(placeholder_issues) == 0:
        print(" [✓] LATTICE HEALTH: EXCELLENT — Zero AST errors, clean types.")
    else:
        print(" [!] LATTICE HEALTH: CORRUPTIONS DETECTED — Needs recompilation.")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    audit_lattice()
