#!/usr/bin/env python3
"""
tools/repair_template_wiring.py

Repairs broken code_template / inputs wiring in place across the corpus:
- Bug A: Single source of truth for input keys vs template placeholders (_FAMILY_ & methods).
- Bug B: Repairs cv2 enum-variant argument-less calls using sibling parameterization.
- Invariant: Every {placeholder} has a matching key in `inputs`, and every declared input
  is referenced in `code_template`.

Preserves existing docstrings (except clearing LLM docstrings for semantic changes in cv2).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_trees import check_template_wiring
from semantic_repair_engine import repair_cell_semantics
from src.template_wiring import clean_malformed_template_braces, transform_call_ast_with_flag
import re


def repair_cv2_variant_cell(
    cell: Dict[str, Any],
    cell_map: Dict[str, Dict[str, Any]],
    base_default_map: Dict[str, Dict[str, Any]],
) -> bool:
    """Repairs cv2 enum-variant cells with argument-less fake calls (Bug B)."""
    cid = cell.get("cell_id", "")
    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    cell["code_template"] = tmpl

    m = re.match(r"\{output_var\}\s*=\s*cv2\.([A-Za-z0-9_]+)\(\)", tmpl)
    if not m:
        return False

    parts = cid.rsplit("_", 1)
    flag = parts[1] if len(parts) > 1 else ""
    grp_def_id = parts[0] + "_DEFAULT"

    target_template = None
    source_cell = None

    if grp_def_id in cell_map and not cell_map[grp_def_id].get("code_template", "").endswith("()"):
        grp_tmpl = clean_malformed_template_braces(cell_map[grp_def_id].get("code_template", ""))
        m_flag = re.search(r"cv2\.([A-Za-z0-9_]+_DEFAULT)", grp_tmpl)
        if m_flag:
            def_flag = m_flag.group(1)
            prefix = def_flag.rsplit("_", 1)[0]
            variant_flag = f"{prefix}_{flag}"
            target_template = grp_tmpl.replace(f"cv2.{def_flag}", f"cv2.{variant_flag}")
            source_cell = cell_map[grp_def_id]
        else:
            target_template = grp_tmpl
            source_cell = cell_map[grp_def_id]

    if not target_template:
        found_base = None
        for b in sorted(base_default_map.keys(), key=len, reverse=True):
            if cid.startswith(b + "_") or cid == b:
                found_base = b
                break
        if found_base:
            base_cell = base_default_map[found_base]
            base_tmpl = clean_malformed_template_braces(base_cell.get("code_template", ""))
            flag_attr = cid[len(found_base) + 1 :] if cid != found_base else ""
            if flag_attr:
                target_template = transform_call_ast_with_flag(base_tmpl, "cv2", flag_attr)
            else:
                target_template = base_tmpl
            source_cell = base_cell

    if not target_template:
        if "_GROUP_" in cid:
            fn_name = cid.split("_GROUP_")[-1]
            target_template = f"{{output_var}} = cv2.{fn_name}({{input_var}})"
            source_cell = cell
        elif cid.endswith("_GROUP"):
            fn_name = cid.replace("CV2_", "").replace("_GROUP", "")
            target_template = f"{{output_var}} = cv2.{fn_name}({{input_var}})"
            source_cell = cell

    if not target_template:
        return False

    cell["code_template"] = clean_malformed_template_braces(target_template)

    if source_cell and source_cell is not cell:
        cell["inputs"] = dict(source_cell.get("inputs", {}))
        cell["outputs"] = dict(source_cell.get("outputs", {}))

    return True

TARGET_FILES = [
    Path("strata_enrichment/checkpoints/cv2.json"),
    Path("strata_enrichment/checkpoints/matplotlib.json"),
    Path("strata_enrichment/checkpoints/numpy.json"),
    Path("strata_enrichment/checkpoints/pandas.json"),
    Path("strata_enrichment/checkpoints/python_core.json"),
    Path("strata_enrichment/checkpoints/scipy.json"),
    Path("strata_enrichment/checkpoints/sklearn.json"),
    Path("trees/cv2.json"),
    Path("trees/matplotlib.json"),
    Path("trees/numpy.json"),
    Path("trees/pandas.json"),
    Path("trees/python_core.json"),
    Path("trees/scipy.json"),
    Path("trees/sklearn.json"),
]


def repair_file(path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Repairs a single tree or checkpoint JSON file in place."""
    if not path.exists():
        print(f"[!] File not found: {path}")
        return {"file": str(path), "status": "missing"}

    print(f"[*] Processing {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = data.get("cells", [])
    domain = data.get("domain", path.stem)
    if domain == "checkpoints":
        domain = path.stem

    cell_map = {c.get("cell_id", ""): c for c in cells}
    base_default_map = {}
    for cid, c in cell_map.items():
        if cid.endswith("_DEFAULT"):
            base = cid[:-8]
            tmpl = clean_malformed_template_braces(c.get("code_template", ""))
            if not tmpl.endswith("()") and ("cv2." in tmpl or f"{domain}." in tmpl):
                base_default_map[base] = c

    modified_count = 0
    repaired_cells = []

    for cell in cells:
        c_copy = json.loads(json.dumps(cell))
        mod_this = False
        if domain == "cv2":
            if repair_cv2_variant_cell(c_copy, cell_map, base_default_map):
                mod_this = True
        if repair_cell_semantics(c_copy, domain):
            mod_this = True
        if mod_this:
            modified_count += 1
        repaired_cells.append(c_copy)

    # Verification check on repaired cells
    problems = check_template_wiring(repaired_cells)

    if not dry_run:
        data["cells"] = repaired_cells
        # Atomic write via temporary file
        temp_dir = path.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        Path(temp_name).replace(path)
        print(f"[+] Saved {path}: {modified_count}/{len(cells)} cells modified. Remaining wiring issues: {len(problems)}")
    else:
        print(f"[*] (DRY-RUN) {path}: {modified_count}/{len(cells)} cells would be modified. Remaining wiring issues: {len(problems)}")

    return {
        "file": str(path),
        "total_cells": len(cells),
        "modified_cells": modified_count,
        "remaining_wiring_problems": len(problems),
        "problems": problems[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Repair code_template / inputs wiring across corpus JSON files.")
    parser.add_argument("files", nargs="*", help="Specific files to repair (defaults to all target files).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files.")
    args = parser.parse_args()

    files_to_process = [Path(f) for f in args.files] if args.files else TARGET_FILES

    results = []
    for f in files_to_process:
        res = repair_file(f, dry_run=args.dry_run)
        results.append(res)

    print("\n" + "=" * 60)
    print("REPAIR SUMMARY REPORT")
    print("=" * 60)
    all_zero = True
    for r in results:
        if r.get("status") == "missing":
            continue
        status = "PASSED (0 problems)" if r["remaining_wiring_problems"] == 0 else f"FAILED ({r['remaining_wiring_problems']} problems)"
        if r["remaining_wiring_problems"] != 0:
            all_zero = False
        print(f"  {r['file']:<45} | Modified: {r['modified_cells']:>5}/{r['total_cells']:<5} | Status: {status}")

    print("=" * 60)
    if all_zero:
        print("[SUCCESS] All files satisfy the template wiring invariant (0 problems everywhere)!")
    else:
        print("[WARNING] Some files still have template wiring issues. Check details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
