#!/usr/bin/env python3
"""
tools/inject_universal_tree_keys.py - Injects universal configuration keys
into trees/*.json and new_trees/*.json if missing, ensuring domain-agnostic
engine fallbacks can remain empty.
"""

import glob
import json
import os
from pathlib import Path

UNIVERSAL_KEYS = {
    "egress_intent_tokens": [
        "save", "export", "write", "dump", "persist", "store", "plot", "show", "display"
    ],
    "materialization_states": [
        "destination_written", "filepath_written", "saved", "exported", "written_to_disk"
    ],
    "polarity_hints": {
        "ascending": [
            "bottom", "lowest", "smallest", "minimum", "min", "worst", "least", "ascending", "asc", "fewest"
        ],
        "descending": [
            "top", "highest", "largest", "biggest", "greatest", "maximum", "max", "best", "most", "descending", "desc", "newest", "latest"
        ]
    }
}


def inject_keys_into_file(filepath: Path) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return False

    if not isinstance(data, dict):
        return False

    modified = False
    for k, v in UNIVERSAL_KEYS.items():
        if k not in data:
            data[k] = v
            modified = True

    if modified:
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        print(f"Injected universal keys into {filepath}")
        return True
    else:
        print(f"Already complete: {filepath}")
        return False


def main():
    repo_root = Path(__file__).resolve().parent.parent
    target_dirs = [repo_root / "trees", repo_root / "new_trees"]

    total_modified = 0
    total_scanned = 0

    for d in target_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            total_scanned += 1
            if inject_keys_into_file(p):
                total_modified += 1

    print(f"Done! Scanned {total_scanned} files, updated {total_modified} files.")


if __name__ == "__main__":
    main()
