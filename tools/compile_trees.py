#!/usr/bin/env python3
"""
tools/compile_trees.py — Compact SQLite v4 Compiler for STRATA Domain Trees.

Validates canonical domain manifests against `TreeSchema` (`extra="forbid"`),
checks code templates through the tree's `LanguageBackend`, and writes ONLY the
5 normalized schema v4 tables defined by `src/lattice_db.py` (zero precomputed
static edges, zero legacy triple cell storage).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from language_backend import get_language_backend  # noqa: E402
from lattice_db import (  # noqa: E402
    compute_trees_fingerprint,
    init_sqlite_db,
    record_trees_fingerprint,
)
from schema import TreeSchema  # noqa: E402

DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


def _resolve_tree_files(trees_dir: Path, domain_filter: Optional[List[str]]) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(trees_dir / "*.json")))
    files = [p for p in files if p.name not in {"manifest.json", "index.json"}]
    if domain_filter:
        wanted = {d.strip().lower() for d in domain_filter if d.strip()}
        files = [p for p in files if p.stem.lower() in wanted]
    return files


def _extract_type_edges(tree: TreeSchema) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    raw_types = tree.types
    if isinstance(raw_types, list):
        for item in raw_types:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type_name")
            if not name:
                continue
            parents = item.get("parents")
            if isinstance(parents, list) and parents:
                for par in parents:
                    if par:
                        entries.append((str(name), str(par)))
            else:
                par = item.get("parent") or item.get("parent_type") or ""
                if par:
                    entries.append((str(name), str(par)))
    elif isinstance(raw_types, dict):
        for name, meta in raw_types.items():
            if isinstance(meta, dict):
                parents = meta.get("parents")
                if isinstance(parents, list) and parents:
                    for par in parents:
                        if par:
                            entries.append((str(name), str(par)))
                else:
                    par = meta.get("parent") or meta.get("parent_type") or ""
                    if par:
                        entries.append((str(name), str(par)))
            elif isinstance(meta, list):
                for par in meta:
                    if par:
                        entries.append((str(name), str(par)))
            elif meta:
                entries.append((str(name), str(meta)))
    return entries


def compile_database(
    output_db: str = DB_PATH,
    domain_filter: Optional[List[str]] = None,
    trees_dir: Optional[Union[str, Path]] = None,
) -> int:
    t_dir = Path(trees_dir) if trees_dir else (PROJECT_ROOT / "trees")
    tree_files = _resolve_tree_files(t_dir, domain_filter)

    if not tree_files:
        print(
            f"[compile_trees] No tree files found in {t_dir}"
            + (f" for domains={domain_filter}" if domain_filter else "")
        )
        return 0

    out_path = Path(output_db)
    init_sqlite_db(out_path, clean=(domain_filter is None))

    parsed_trees: List[Tuple[Path, TreeSchema]] = []
    default_lang_spec: Optional[Any] = None
    core_lang_spec: Optional[Any] = None

    for path in tree_files:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        tree = TreeSchema(**raw_data)
        parsed_trees.append((path, tree))
        if tree.language_spec is not None:
            if tree.is_core and core_lang_spec is None:
                core_lang_spec = tree.language_spec
            elif default_lang_spec is None:
                default_lang_spec = tree.language_spec

    fallback_lang_spec = core_lang_spec or default_lang_spec

    written = 0
    types_written = 0
    typestates_written = 0
    warnings: List[str] = []

    conn = sqlite3.connect(str(out_path))
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()

        for path, tree in parsed_trees:
            dom = str(tree.domain or path.stem).strip().lower()
            if domain_filter is not None:
                cur.execute("DELETE FROM nodes WHERE domain_name = ?", (dom,))
                cur.execute("DELETE FROM types WHERE domain_name = ?", (dom,))
                cur.execute("DELETE FROM typestates WHERE domain_name = ?", (dom,))
                cur.execute("DELETE FROM domain_metadata WHERE domain_name = ?", (dom,))

            rs_json = (
                json.dumps(tree.runtime_setup.model_dump(mode="json"), ensure_ascii=False)
                if tree.runtime_setup is not None
                else None
            )
            ls_json = (
                json.dumps(tree.language_spec.model_dump(mode="json"), ensure_ascii=False)
                if tree.language_spec is not None
                else None
            )
            cur.execute(
                "INSERT OR REPLACE INTO domain_metadata "
                "(domain_name, version, is_core, imports_json, top_types_json, runtime_setup_json, language_spec_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dom,
                    tree.version,
                    1 if tree.is_core else 0,
                    json.dumps(list(tree.imports or ()), ensure_ascii=False),
                    json.dumps(list(tree.top_types or ()), ensure_ascii=False),
                    rs_json,
                    ls_json,
                ),
            )

            for t_name, p_type in _extract_type_edges(tree):
                cur.execute(
                    "INSERT OR REPLACE INTO types (type_name, parent_type, domain_name) VALUES (?, ?, ?)",
                    (t_name, p_type, dom),
                )
                types_written += 1

            if tree.typestates is not None:
                for st in tree.typestates.states:
                    if isinstance(st, str):
                        row_ts = (st, None, None, None, dom)
                    else:
                        props = st.properties or {}
                        row_ts = (
                            st.name,
                            st.parent_state,
                            st.carrier_type,
                            json.dumps(props, ensure_ascii=False) if props else None,
                            dom,
                        )
                    cur.execute(
                        "INSERT OR REPLACE INTO typestates "
                        "(state_name, parent_state, carrier_type, properties_json, domain_name) "
                        "VALUES (?, ?, ?, ?, ?)",
                        row_ts,
                    )
                    typestates_written += 1

            backend = get_language_backend(tree.language_spec or fallback_lang_spec)
            for cell in tree.cells:
                tmpl = backend.select_cell_template(cell)
                if not tmpl or not tmpl.strip():
                    warnings.append(f"{path.name}::{cell.cell_id}: empty code_template")
                else:
                    ok, err = backend.validate_syntax(tmpl, allow_placeholders=True)
                    if not ok:
                        warnings.append(f"{path.name}::{cell.cell_id}: {err}")

                cell_payload = cell.model_dump(mode="json")
                if not cell_payload.get("domain_name"):
                    cell_payload["domain_name"] = dom
                cur.execute(
                    "INSERT OR REPLACE INTO nodes "
                    "(cell_id, domain_name, docstring, code_template, mutation_type, cell_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        cell.cell_id,
                        dom,
                        cell.docstring or "",
                        tmpl or "",
                        cell.mutation_type or "pure",
                        json.dumps(cell_payload, ensure_ascii=False),
                    ),
                )
                written += 1

        conn.commit()
    finally:
        conn.close()

    if domain_filter is None:
        record_trees_fingerprint(out_path, compute_trees_fingerprint(t_dir))

    print(
        f"[compile_trees] Schema v4: compiled {written} node(s), {types_written} type edge(s), "
        f"{typestates_written} typestate(s) from {len(tree_files)} file(s) into {out_path}"
    )
    if warnings:
        print(f"[compile_trees] {len(warnings)} template warning(s):", file=sys.stderr)
        for w in warnings[:20]:
            print(f"  WARN {w}", file=sys.stderr)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="STRATA Tree Compiler (Schema v4)")
    parser.add_argument("--output", type=str, default=DB_PATH)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--dir", "--trees-dir", dest="trees_dir", type=str, default=None, help="Directory containing domain tree JSON files")
    args = parser.parse_args()
    compile_database(output_db=args.output, domain_filter=args.domains, trees_dir=args.trees_dir)


if __name__ == "__main__":
    main()
