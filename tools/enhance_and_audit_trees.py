"""
tools/enhance_and_audit_trees.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Deterministic, reflection-driven tree enhancement + full-corpus audit.

Enhancement (root-cause-III remediation: missing enum-domain declarations):
  For every cell in trees/<domain>.json, ports that are enum-ish (required,
  no default, str/any/enum carrier) receive a DECLARED `domain` spec
  ("<module>.<FAMILY>_*") whenever the LIBRARY'S OWN NAMING proves the
  relation: the cell's identity vocabulary shares a stem with a module
  constant family (e.g. cell CVTCOLOR ~ constant family COLOR_*). At runtime
  ExecutionContext._resolve_enum_constant grounds such ports dynamically by
  embedding/token similarity over the family members — no engine hardcodes.

Audit:
  Reuses audit_trees.py checks (template wiring, duplicate ids, schema
  fingerprints, type consistency, docstrings) and adds a stateful-lifecycle
  summary. Reports land in logs/.

Usage:
  python tools/enhance_and_audit_trees.py                    # enhance + audit all
  python tools/enhance_and_audit_trees.py --domains sklearn pandas
  python tools/enhance_and_audit_trees.py --audit-only
  python tools/enhance_and_audit_trees.py --dry-run
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.tokenizer import CellTokenizer, normalize_token  # noqa: E402

TREES_DIR = PROJECT_ROOT / "trees"
LOGS_DIR = PROJECT_ROOT / "logs"

# Tree file layout (domain -> underlying module for constant reflection).
# Names here are the TREE FILE NAMES in this repo's pipeline, not engine logic.
TREE_MODULES: Dict[str, str] = {
    "cv2": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "sklearn",
    "matplotlib": "matplotlib",
    "python_core": "builtins",
    "functools": "functools",
    "gzip": "gzip",
    "itertools": "itertools",
    "operator": "operator",
    "sqlite3": "sqlite3",
    "statistics": "statistics",
}

WILDCARD_CARRIERS = {"any", "", "*", "top", "unknown", "none"}


# =====================================================================
# Constant-family discovery (pure reflection)
# =====================================================================

def discover_constant_families(module: Any) -> Dict[str, List[str]]:
    """
    Groups a module's UPPER_CASE constants into families by their first
    underscore prefix (cv2.COLOR_BGR2GRAY -> family "COLOR"). Families with
    >= 2 members are enum-ish candidate domains. Zero name patterns.
    """
    import importlib
    families: Dict[str, List[str]] = {}
    if module is None:
        return families
    for attr in dir(module):
        if not attr.isupper() or "_" not in attr or attr.startswith("_"):
            continue
        try:
            if callable(getattr(module, attr, None)) or isinstance(getattr(module, attr), type):
                continue
        except Exception:
            continue
        prefix = attr.split("_")[0]
        families.setdefault(prefix, []).append(attr)
    return {k: sorted(v) for k, v in families.items() if len(v) >= 2}


def _module_of(spec: str) -> Optional[Any]:
    import importlib
    if not spec:
        return None
    try:
        return importlib.import_module(spec)
    except Exception:
        return None


def _stemmed_tokens(text: str) -> Set[str]:
    out: Set[str] = set()
    for tok in CellTokenizer.tokenize_identifier(text):
        out.add(tok)
        st = normalize_token(tok)
        if len(st) >= 2:
            out.add(st)
    return out


def build_family_index(domains: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    domain -> {family_prefix: {"module": spec, "members": [...]}}
    Family prefixes additionally indexed by STEM so "cvtColor" matches "COLOR".
    """
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for domain in domains:
        spec = TREE_MODULES.get(domain, domain)
        mod = _module_of(spec)
        if mod is None:
            continue
        fams = discover_constant_families(mod)
        index[domain] = {
            f: {"module": spec, "members": members}
            for f, members in fams.items()
        }
    return index


# =====================================================================
# Port enhancement
# =====================================================================

def _param_doc_sections(doc: str) -> Dict[str, str]:
    """
    Extracts per-parameter description text from a numpydoc-style docstring
    ("Parameters\n----------\nname : type\n    description ..."). Works for
    cv2's embedded '@param name: description' style too. Structural parsing,
    zero name patterns.
    """
    out: Dict[str, str] = {}
    if not doc:
        return out
    lines = doc.splitlines()
    i = 0
    n = len(lines)
    # numpydoc style
    while i < n:
        line = lines[i]
        if line.strip().lower() in ("parameters", "params", "arguments", "args") \
                and i + 1 < n and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
            i += 2
            cur_name = None
            buf: List[str] = []
            while i < n and lines[i].strip() and not set(lines[i].strip()) <= {"-"}:
                ln = lines[i]
                stripped = ln.strip()
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_\[\]]*)\s*:\s*(.*)$", stripped)
                if m and not ln.startswith("     "):
                    if cur_name:
                        out[cur_name] = " ".join(buf)
                    cur_name = m.group(1).strip("[]")
                    buf = [m.group(2)]
                elif cur_name:
                    buf.append(stripped)
                i += 1
            if cur_name:
                out[cur_name] = " ".join(buf)
            continue
        # cv2 '@param name: description' style
        m = re.search(r"@param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
        i += 1
    return out


def enhance_domain(domain: str, family_index: Dict[str, Dict[str, Dict[str, Any]]],
                   dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Declares enum domains on enum-ish ports where the LIBRARY'S OWN NAMING
    proves the relation. Evidence channels, in precision order:
      1. the port's own name stem matches a constant family;
      2. the port's OWN docstring parameter description stem-matches a family;
      3. the consuming cell's identity vocabulary stem-matches a family.
    Candidate ports are integer/enum-carried REQUIRED ports without defaults:
    that is the Python constant-flag convention (cv2.COLOR_BGR2GRAY is an
    int); string/path-typed ports are never enum choices (measured exploit:
    every `filename` port wrongly declared as an enum family).
    Returns (cells_scanned, ports_declared, ports_skipped_existing).
    """
    tree_path = TREES_DIR / f"{domain}.json"
    if not tree_path.exists() or domain not in family_index:
        return (0, 0, 0)

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    fams = family_index[domain]
    fam_stems: Dict[str, str] = {}
    for fam in fams:
        for st in _stemmed_tokens(fam):
            fam_stems.setdefault(st, fam)

    # Stems derived from the domain/package name itself (e.g. "cv" from
    # "cv2") are UNIVERSAL prefixes of every cell in the tree, never cell
    # semantics. Excluding them prevents the package prefix from matching a
    # constant family of the same name (measured: every cv2 cell's `filename`
    # port wrongly declared as cv2.CV_* via the "cv" stem).
    banned_stems: Set[str] = set()
    for piece in re.findall(r"[A-Za-z0-9]+", domain):
        banned_stems |= _stemmed_tokens(piece)
    mod_spec = TREE_MODULES.get(domain, domain)
    for piece in mod_spec.split("."):
        banned_stems |= _stemmed_tokens(piece)
    fam_stems = {st: fam for st, fam in fam_stems.items() if st not in banned_stems}

    scanned = 0
    declared = 0
    existing = 0
    changed = False

    for cell in tree.get("cells", []):
        scanned += 1
        cell_stems = _stemmed_tokens(cell.get("cell_id", ""))
        for kw in cell.get("keywords", []) or []:
            cell_stems |= _stemmed_tokens(kw)
        cell_stems -= banned_stems

        param_docs = _param_doc_sections(str(cell.get("docstring", "") or ""))

        for p_name, port in (cell.get("inputs", {}) or {}).items():
            if not isinstance(port, dict):
                continue
            if port.get("domain"):
                existing += 1
                continue
            if not port.get("required", True) or port.get("default_value") is not None:
                continue
            t_name = str(port.get("type_name", "")).lower()
            # Constant-flag convention: enum choices ride integer (or already-
            # enum) carriers. String carriers are references (paths, names).
            if t_name not in ("int", "integer", "enum", ""):
                continue

            p_stems = _stemmed_tokens(p_name) - banned_stems
            doc_stems = _stemmed_tokens(param_docs.get(p_name, "")) - banned_stems

            # Port-side evidence first, then the port's own doc section, then
            # the cell's identity. A port stem matching a DIFFERENT family
            # blocks the declaration (the port belongs elsewhere).
            evidence: Set[str] = set()
            for st in p_stems | doc_stems:
                if st in fam_stems:
                    evidence.add(fam_stems[st])
            blocking = {fam_stems[st] for st in p_stems if st in fam_stems} - evidence
            if not evidence and not blocking:
                for st in cell_stems:
                    if st in fam_stems:
                        evidence.add(fam_stems[st])
            if not evidence:
                continue
            if blocking:
                continue

            fam = sorted(evidence, key=lambda f: -len(fams[f]))[0]
            spec = fams[fam]["module"]
            port["domain"] = f"{spec}.{fam}_*"
            declared += 1
            changed = True

    if changed and not dry_run:
        tree["version"] = tree.get("version", "2.0.0")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tree["enhanced_at"] = now
        tree["enhancement_source"] = "reflection"
        tmp = tree_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(tree, indent=2), encoding="utf-8")
        tmp.replace(tree_path)

    return (scanned, declared, existing)


# =====================================================================
# Audit (reuses the repo's own audit kernel)
# =====================================================================

def audit_domain(domain: str) -> Dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from audit_trees import (  # noqa: E402
        check_template_wiring, check_duplicate_ids, check_docstrings,
        check_type_consistency, check_schema_fingerprints
    )
    tree_path = TREES_DIR / f"{domain}.json"
    if not tree_path.exists():
        return {"domain": domain, "exists": False}
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    cells = tree.get("cells", [])
    report: Dict[str, Any] = {
        "domain": domain,
        "exists": True,
        "cells": len(cells),
        "wiring_issues": len(check_template_wiring(cells)),
        "duplicate_ids": len(check_duplicate_ids(cells)),
        "docstring_missing": sum(len(v) for v in check_docstrings(cells).values()),
        "type_issues": len(check_type_consistency(cells)),
        "fingerprint_dupes": len(check_schema_fingerprints(cells)),
    }
    endos = sum(1 for c in cells
                for p in [((c.get("outputs") or {}).get("output_data") or {})]
                if p.get("state") == "mutated")
    constrained = sum(1 for c in cells
                      if ((c.get("inputs") or {}).get("data") or {}).get("state") == "mutated")
    report["endomorphisms"] = endos
    report["state_dependent_receivers"] = constrained
    report["enum_domains_declared"] = sum(
        1 for c in cells for p in (c.get("inputs") or {}).values()
        if isinstance(p, dict) and p.get("domain")
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Reflection-driven tree enhancement + corpus audit")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    domains = args.domains or sorted(p.stem for p in TREES_DIR.glob("*.json")
                                     if not p.name.endswith(".pre-curation.json"))

    LOGS_DIR.mkdir(exist_ok=True)
    reports: List[Dict[str, Any]] = []

    if not args.audit_only:
        fam_index = build_family_index(domains)
        total_fams = {d: len(f) for d, f in fam_index.items()}
        print(f"[*] Discovered constant families: {total_fams}")
        total_declared = 0
        for d in domains:
            scanned, declared, existing = enhance_domain(d, fam_index, dry_run=args.dry_run)
            total_declared += declared
            if declared or scanned:
                print(f"[enhance:{d}] scanned {scanned} cells -> declared {declared} enum-domain ports "
                      f"({existing} already declared)")
        verb = "would declare" if args.dry_run else "declared"
        print(f"[✓] Enhancement complete ({verb} {total_declared} enum-domain port declarations)")

    print("\n[*] Running corpus audit...")
    for d in domains:
        r = audit_domain(d)
        reports.append(r)
        if r.get("exists"):
            print(f"[audit:{d}] cells={r['cells']} wiring={r['wiring_issues']} dup_ids={r['duplicate_ids']} "
                  f"type_issues={r['type_issues']} endos={r['endomorphisms']} "
                  f"state_recv={r['state_dependent_receivers']} enum_ports={r['enum_domains_declared']}")

    report_path = LOGS_DIR / "tree_enhancement_summary.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\n[✓] Audit reports written to: {report_path}")


if __name__ == "__main__":
    main()
