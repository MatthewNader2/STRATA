"""
tools/audit_stateful_reachability.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)

Offline static audit over compiled trees/*.json (no engine, no chat profile):

For every harvested class that owns a constructor, at least one mutator
(endomorphism emitting C[mutated]) and at least one state-dependent method
(receiver declares C[mutated]), verify the full lifecycle is type-reachable:

    C_INIT  ->  C_FIT        (constructed unifies with mutator receiver)
    C_FIT   ->  C_USE        (mutated output unifies with dependent receiver)

and report classes whose fit->predict-like edges are structurally disconnected
(the root-cause-B failure class: a mutator modeled as a str-sink or a consumer
whose receiver state does not accept the mutator's post-state). This turns
"here is one broken example" into "here is the fraction of the corpus affected
by this exact defect class".

Usage:
  python tools/audit_stateful_reachability.py                 # all domains
  python tools/audit_stateful_reachability.py --domains sklearn pandas
  python tools/audit_stateful_reachability.py --json          # machine report
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.unification import unify, AlgebraicSignature  # noqa: E402

TREES_DIR = PROJECT_ROOT / "trees"

STATE_RX = re.compile(r"^(?P<cls>.+)\[(?P<state>.+)\]$")
MUTATED = "mutated"


def _sig(type_name: str, state: str) -> AlgebraicSignature:
    return AlgebraicSignature(type_name, state)


def audit_domain(tree_path: Path) -> Dict[str, Any]:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    cells = tree.get("cells", [])
    domain = tree.get("domain", tree_path.stem)

    # Index instance-method cells by (class, method): CELL ids look like
    # <DOMAIN>_<CLASS>_<METHOD>. Constructors end with _INIT. Structural
    # parse via the cell's receiver port (type_name == class) is the truth
    # source; id parsing is only the grouping hint.
    classes: Dict[str, Dict[str, Any]] = {}

    for c in cells:
        inputs = c.get("inputs", {}) or {}
        outputs = c.get("outputs", {}) or {}
        receiver = inputs.get("data")
        out_port = outputs.get("output_data") or next(iter(outputs.values()), None)
        cid = c.get("cell_id", "")

        # Constructors have no receiver port: their OUTPUT carrier names the class.
        is_ctor = c.get("node_type") == "constructor" or cid.upper().endswith("_INIT")
        if is_ctor:
            cls_name = str((out_port or {}).get("type_name", ""))
            if not cls_name or cls_name.lower() in ("any", "str", "none", "*", "top", ""):
                continue
            entry = classes.setdefault(cls_name, {"ctor": None, "mutators": [], "consumers": [], "cells": 0})
            entry["cells"] += 1
            entry["ctor"] = {
                "cell_id": cid,
                "out_state": str((out_port or {}).get("state", "")) or "constructed",
            }
            continue

        if receiver is None:
            continue
        cls_name = str(receiver.get("type_name", ""))
        if not cls_name or cls_name.lower() in ("any", "str", "none", "*", "top", ""):
            continue
        entry = classes.setdefault(cls_name, {"ctor": None, "mutators": [], "consumers": [], "cells": 0})
        entry["cells"] += 1

        out_state = str((out_port or {}).get("state", ""))
        recv_state = str(receiver.get("state", ""))

        if out_state == MUTATED and str((out_port or {}).get("type_name", "")) == cls_name:
            entry["mutators"].append({"cell_id": cid, "recv_state": recv_state or "any"})
        else:
            entry["consumers"].append({
                "cell_id": cid,
                "recv_state": recv_state or "any",
                "out_type": str((out_port or {}).get("type_name", "")),
            })

    report = {
        "domain": domain,
        "classes_with_lifecycle": 0,
        "fully_reachable": 0,
        "disconnected": [],
        "note": "",
    }

    for cls_name, entry in classes.items():
        if not entry["ctor"] or not entry["mutators"] or not entry["consumers"]:
            continue
        report["classes_with_lifecycle"] += 1

        ctor_out = _sig(cls_name, entry["ctor"]["out_state"] or "constructed")
        problems: List[str] = []

        for mut in entry["mutators"]:
            mut_in = _sig(cls_name, mut["recv_state"] or "any")
            if unify(ctor_out, mut_in) is None:
                problems.append(f"DISCONNECT ctor->{mut['cell_id']} (ctor[{entry['ctor']['out_state']}] vs recv[{mut['recv_state']}])")

        mut_out = _sig(cls_name, MUTATED)
        for cons in entry["consumers"]:
            if cons["recv_state"] != MUTATED:
                continue  # permissive receiver: always composable
            cons_in = _sig(cls_name, cons["recv_state"])
            if unify(mut_out, cons_in) is None:
                problems.append(f"DISCONNECT {entry['mutators'][0]['cell_id']}->{cons['cell_id']} (mutated vs recv[{cons['recv_state']}])")

        if problems:
            report["disconnected"].append({"class": cls_name, "problems": problems})
        else:
            report["fully_reachable"] += 1

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit stateful lifecycle reachability across the corpus")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    paths = sorted(TREES_DIR.glob("*.json"))
    if args.domains:
        want = set(args.domains)
        paths = [p for p in paths if p.stem in want]

    total_lifecycle = 0
    total_reachable = 0
    total_broken = 0
    lines: List[str] = []

    for p in paths:
        if p.name.endswith((".pre-curation.json", ".tmp")):
            continue
        try:
            rep = audit_domain(p)
        except Exception as e:
            print(f"[!] {p.name}: {e}")
            continue
        total_lifecycle += rep["classes_with_lifecycle"]
        total_reachable += rep["fully_reachable"]
        total_broken += len(rep["disconnected"])
        if args.json:
            lines.append(json.dumps(rep))
            continue
        print(f"\n=== {rep['domain']} ===")
        print(f"  classes with full lifecycle (ctor+mutator+consumers): {rep['classes_with_lifecycle']}")
        print(f"  fully reachable lifecycles:  {rep['fully_reachable']}")
        print(f"  DISCONNECTED lifecycles:     {len(rep['disconnected'])}")
        for d in rep["disconnected"][:8]:
            print(f"    - {d['class']}:")
            for pr in d["problems"][:4]:
                print(f"        {pr}")

    if not args.json:
        print(f"\n[✓] Corpus totals: {total_lifecycle} stateful classes, "
              f"{total_reachable} fully reachable, {total_broken} with disconnections")


if __name__ == "__main__":
    main()
