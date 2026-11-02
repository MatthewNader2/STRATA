"""
tools/merge_enrichment.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)

Field-wise merge of LLM-enriched checkpoints (strata_enrichment/checkpoints/)
into the freshly harvested trees/ — the "merge the enriched trees" step of
the regeneration pipeline.

Merge contract (provenance discipline):
  LLM checkpoints OWN documentation fields only:
      docstring, enrichment_source ("llm"), enriched_at,
      semantic_tags / keywords (UNION with existing — the harvester's
      structural tokens are never dropped).
  The harvester ALWAYS OWNS structure:
      cell_id, stage, inputs/outputs (types, states, qualifiers, domains),
      code_template, dependencies, node_type/node_role, source_priority.
  A checkpoint cell whose cell_id no longer exists in the fresh tree is
  reported as stale and skipped — stale LLM knowledge can never resurrect
  an old schema.

Usage:
  python tools/merge_enrichment.py                 # merge all available domains
  python tools/merge_enrichment.py --domains sklearn pandas
  python tools/merge_enrichment.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "strata_enrichment" / "checkpoints"
TREES_DIR = PROJECT_ROOT / "trees"

DOC_FIELDS = ("docstring",)
TAG_FIELDS = ("semantic_tags", "keywords")
SET_FIELDS = {"enrichment_source": "llm"}


def merge_domain(domain: str, dry_run: bool = False) -> Dict[str, int]:
    ckpt_path = CHECKPOINT_DIR / f"{domain}.json"
    tree_path = TREES_DIR / f"{domain}.json"
    stats = {"checkpoint_cells": 0, "merged_docs": 0, "merged_tags": 0, "stale": 0, "tree_cells": 0}

    if not ckpt_path.exists():
        print(f"[merge:{domain}] no checkpoint ({ckpt_path.name}); skipping")
        return stats
    if not tree_path.exists():
        print(f"[merge:{domain}] no fresh tree ({tree_path.name}); run the harvester first")
        return stats

    try:
        checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[merge:{domain}] checkpoint unreadable: {e}")
        return stats

    ck_cells = checkpoint.get("cells", [])
    stats["checkpoint_cells"] = len(ck_cells)

    ck_map: Dict[str, Dict[str, Any]] = {}
    for c in ck_cells:
        if isinstance(c, dict) and c.get("cell_id"):
            ck_map[str(c["cell_id"]).upper()] = c

    tree = json.loads(tree_path.read_text(encoding="utf-8"))

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    matched = 0
    for cell in tree.get("cells", []):
        stats["tree_cells"] += 1
        cid = str(cell.get("cell_id", "")).upper()
        if not cid:
            continue
        enriched = ck_map.get(cid) or ck_map.get(cid.replace("_", ""))
        if enriched is None:
            continue
        matched += 1
        merged_any = False
        # Docstrings: prefer the longer, more informative text
        for f in DOC_FIELDS:
            new_doc = str(enriched.get(f, "") or "").strip()
            old_doc = str(cell.get(f, "") or "").strip()
            if new_doc and len(new_doc) > len(old_doc):
                cell[f] = new_doc
                stats["merged_docs"] += 1
                merged_any = True
        # Tags: union, order-preserving
        for f in TAG_FIELDS:
            new_tags = [str(t) for t in (enriched.get(f) or []) if str(t).strip()]
            if not new_tags:
                continue
            old_tags = [str(t) for t in (cell.get(f) or [])]
            union = old_tags + [t for t in new_tags if t not in old_tags]
            if union != old_tags:
                cell[f] = union
                stats["merged_tags"] += 1
                merged_any = True
        # Declared semantic roles (from the enrichment LLM): DATA, not engine
        # logic. Applied ONLY to wildcard-state ports — harvester-declared
        # constraints (mutator receivers, typestates) are never regressed.
        roles = enriched.get("roles") or {}
        if isinstance(roles, dict):
            for p_name, role in roles.items():
                port = (cell.get("inputs") or {}).get(p_name)
                if (
                    isinstance(port, dict) and isinstance(role, str)
                    and str(port.get("state", "")).lower() in ("any", "", "default")
                ):
                    port["state"] = role
                    merged_any = True
        # Provenance: only stamp when this checkpoint actually contributed —
        # fresh harvester documentation must not be mislabeled as LLM output.
        if merged_any:
            for f, v in SET_FIELDS.items():
                cell[f] = v
            cell["enriched_at"] = enriched.get("enriched_at") or now_iso

    stats["stale"] = stats["checkpoint_cells"] - matched

    if dry_run:
        print(f"[merge:{domain}] DRY RUN: {matched}/{stats['checkpoint_cells']} checkpoint cells matched, "
              f"{stats['merged_docs']} docs, {stats['merged_tags']} tag-merges, {stats['stale']} stale")
        return stats

    tmp = tree_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    tmp.replace(tree_path)
    print(f"[merge:{domain}] {matched}/{stats['checkpoint_cells']} checkpoint cells merged "
          f"({stats['merged_docs']} docs, {stats['merged_tags']} tag-merges, {stats['stale']} stale skipped)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge LLM enrichment checkpoints into fresh trees")
    parser.add_argument("--domains", nargs="*", default=None, help="Domains to merge (default: all checkpoints)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    if args.domains:
        domains = args.domains
    else:
        domains = sorted(p.stem for p in CHECKPOINT_DIR.glob("*.json") if not p.name.endswith("_progress.jsonl"))

    print(f"[*] Merging LLM enrichment into trees/ for: {', '.join(domains) or '(none)'}")
    totals = {"merged_docs": 0, "merged_tags": 0, "stale": 0}
    for d in domains:
        s = merge_domain(d, dry_run=args.dry_run)
        for k in totals:
            totals[k] += s.get(k, 0)
    print(f"[✓] Totals: {totals['merged_docs']} docstrings, {totals['merged_tags']} tag-merges, {totals['stale']} stale-skipped")


if __name__ == "__main__":
    main()
