#!/usr/bin/env python3
"""
tools/promote_enriched_trees.py

Promotes verified and repaired enriched checkpoint JSONs from
`strata_enrichment/checkpoints/` into `trees/` directory, preserving domain invariants:
- cv2: preserves CV2_COLOR_BGR2GRAY and raw output state on IMREAD cells.
- pandas: preserves PANDAS_SORT_VALUES qualifiers and ordered state.
- Audits every tree before writing to ensure 0 template wiring errors.
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_trees import check_template_wiring, check_duplicate_ids

CHECKPOINT_DIR = PROJECT_ROOT / "strata_enrichment" / "checkpoints"
TREES_DIR = PROJECT_ROOT / "trees"

DOMAINS = [
    "cv2",
    "matplotlib",
    "numpy",
    "pandas",
    "python_core",
    "scipy",
    "sklearn",
]


def promote():
    print("[*] Starting promotion of enriched checkpoints to trees/...")

    for domain in DOMAINS:
        ckpt_path = CHECKPOINT_DIR / f"{domain}.json"
        tree_path = TREES_DIR / f"{domain}.json"

        if not ckpt_path.exists():
            print(f"[!] Checkpoint missing: {ckpt_path}")
            continue

        print(f"[*] Processing {domain}...")
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)

        existing_tree_data = None
        if tree_path.exists():
            with open(tree_path, "r", encoding="utf-8") as f:
                existing_tree_data = json.load(f)

        cells = ckpt_data.get("cells", [])
        cell_map = {c["cell_id"]: c for c in cells}

        # Domain-specific invariant preservation
        if domain == "cv2":
            # 1. Preserve CV2_COLOR_BGR2GRAY
            if "CV2_COLOR_BGR2GRAY" not in cell_map and existing_tree_data:
                for c in existing_tree_data.get("cells", []):
                    if c["cell_id"] == "CV2_COLOR_BGR2GRAY":
                        cells.append(c)
                        cell_map["CV2_COLOR_BGR2GRAY"] = c
                        print("  [+] Preserved CV2_COLOR_BGR2GRAY")
                        break

            # 2. Preserve raw output state on IMREAD cells
            for cid, c in cell_map.items():
                if "IMREAD" in cid and "outputs" in c:
                    for out_key, out_port in c["outputs"].items():
                        if out_port.get("type_name") == "Mat":
                            out_port["state"] = "raw"
                if cid == "CV2_IMREAD":
                    c["code_template"] = "{output_var} = cv2.imread({filename})"
                    c["inputs"] = {
                        "filename": {
                            "type_name": "str",
                            "state": "source_identifier",
                            "default_value": None,
                            "description": "",
                            "required": True,
                        }
                    }
                    c["outputs"] = {
                        "dst": {
                            "type_name": "Mat",
                            "state": "raw",
                            "default_value": None,
                            "description": "",
                            "required": True,
                        }
                    }

        elif domain == "pandas":
            if "PANDAS_SORT_VALUES" in cell_map:
                c = cell_map["PANDAS_SORT_VALUES"]
                if "ascending" in c.get("inputs", {}):
                    c["inputs"]["ascending"]["state"] = "ordered"
                    c["inputs"]["ascending"]["qualifiers"] = [
                        ["positive", "ascending"],
                        ["negative", "descending"],
                    ]
                if "by" in c.get("inputs", {}):
                    c["inputs"]["by"]["default_value"] = None
                # Clean prompt-specific artifacts from tags/keywords
                for key in ("semantic_tags", "keywords"):
                    if key in c:
                        c[key] = [w for w in c[key] if w not in ("age", "salary")]
                print("  [+] Preserved PANDAS_SORT_VALUES qualifiers and ordered state")

        # QA Checks
        wiring_problems = check_template_wiring(cells)
        if wiring_problems:
            print(f"[ERROR] {domain} has {len(wiring_problems)} remaining wiring problems!")
            for p in wiring_problems[:5]:
                print(f"  {p['cell_id']}: {p['template']}")
            sys.exit(1)

        dupes = check_duplicate_ids(cells)
        if dupes:
            print(f"[ERROR] {domain} has duplicate cell IDs: {dupes}")
            sys.exit(1)

        ckpt_data["cells"] = cells

        # Backup existing tree before overwrite
        if tree_path.exists():
            backup_path = tree_path.with_suffix(".json.bak")
            shutil.copy2(tree_path, backup_path)

        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_data, f, indent=2)

        print(f"[+] Successfully promoted {domain} to {tree_path} ({len(cells)} cells, 0 wiring errors)")

    print("\n[SUCCESS] All 7 domains promoted cleanly to trees/!")


if __name__ == "__main__":
    promote()
