"""
src/lattice_db.py — Lightweight stdlib-only SQLite v4 schema, compiler fingerprint,
and CompiledLattice snapshot loader for STRATA.

Schema v4 stores only 5 normalized tables with zero precomputed static edges
and 100% read coverage in `load_compiled_lattice`:
  1. `_compilation_meta`
  2. `nodes`
  3. `types`
  4. `typestates`
  5. `domain_metadata`
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from .schema import CellSchema, LanguageSpec, normalize_typestate
except (ImportError, ValueError):
    from schema import CellSchema, LanguageSpec, normalize_typestate

COMPILER_SCHEMA_VERSION = "4"


def init_sqlite_db(db_path: Path, clean: bool = False) -> None:
    """Initialize SQLite schema v4 (5 compact tables), recreating automatically on schema mismatch."""
    if clean and db_path.exists():
        os.remove(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()

        existing_tables = {
            row[0]
            for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        needs_reset = bool(existing_tables & {"aliases", "artifact_readers", "structural_metadata"})
        if not needs_reset and "nodes" in existing_tables:
            node_cols = {row[1] for row in cur.execute("PRAGMA table_info(nodes)").fetchall()}
            if "cell_json" not in node_cols:
                needs_reset = True
        if not needs_reset and "domain_metadata" not in existing_tables and existing_tables:
            needs_reset = True

        if needs_reset:
            for tbl in (
                "nodes",
                "types",
                "typestates",
                "domain_metadata",
                "aliases",
                "artifact_readers",
                "structural_metadata",
                "_compilation_meta",
            ):
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS _compilation_meta (
                key       TEXT PRIMARY KEY,
                value     TEXT,
                timestamp REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                cell_id       TEXT PRIMARY KEY,
                domain_name   TEXT NOT NULL,
                docstring     TEXT NOT NULL DEFAULT '',
                code_template TEXT NOT NULL DEFAULT '',
                mutation_type TEXT NOT NULL DEFAULT 'pure',
                cell_json     TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain_name)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS types (
                type_name   TEXT NOT NULL,
                parent_type TEXT NOT NULL,
                domain_name TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (type_name, parent_type)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS typestates (
                state_name      TEXT NOT NULL,
                parent_state    TEXT,
                carrier_type    TEXT,
                properties_json TEXT,
                domain_name     TEXT NOT NULL,
                PRIMARY KEY (state_name, domain_name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS domain_metadata (
                domain_name        TEXT PRIMARY KEY,
                version            TEXT NOT NULL DEFAULT '2.0.0',
                is_core            INTEGER NOT NULL DEFAULT 0,
                imports_json       TEXT NOT NULL DEFAULT '[]',
                top_types_json     TEXT NOT NULL DEFAULT '[]',
                runtime_setup_json TEXT,
                language_spec_json TEXT
            )
        """)
        conn.commit()


def compute_trees_fingerprint(trees_dir: Union[str, Path] = "trees") -> str:
    """Compute a fast SHA-256 fingerprint over all tree JSON files in the directory."""
    td = Path(trees_dir)
    if not td.exists():
        return "MISSING"
    try:
        entries = sorted(
            [e for e in os.scandir(td) if e.is_file() and e.name.endswith(".json")],
            key=lambda x: x.name,
        )
    except OSError:
        return "MISSING"

    if not entries:
        return "EMPTY"

    h = hashlib.sha256()
    h.update(f"compiler:{COMPILER_SCHEMA_VERSION}".encode("utf-8"))
    for entry in entries:
        h.update(entry.name.encode("utf-8"))
        try:
            stat = entry.stat()
            h.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        except OSError:
            pass
    return h.hexdigest()


def get_stored_meta(db_path: Union[str, Path], key: str) -> Optional[str]:
    """Read one value from `_compilation_meta` (None when absent)."""
    target = Path(db_path)
    if not target.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            row = conn.execute("SELECT value FROM _compilation_meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def get_stored_fingerprint(db_path: Union[str, Path]) -> Optional[str]:
    """Retrieve the tree compilation fingerprint recorded in the SQLite database."""
    return get_stored_meta(db_path, "trees_fingerprint")


def record_trees_fingerprint(db_path: Union[str, Path], fingerprint: str) -> None:
    """Record tree compilation fingerprint and schema version in `_compilation_meta`."""
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS _compilation_meta (key TEXT PRIMARY KEY, value TEXT, timestamp REAL)")
            now = time.time()
            cur.execute(
                "INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('trees_fingerprint', ?, ?)",
                (fingerprint, now),
            )
            cur.execute(
                "INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('compiler_schema', ?, ?)",
                (COMPILER_SCHEMA_VERSION, now),
            )
            conn.commit()
    except Exception:
        pass


def ensure_lattice_compiled(trees_dir: Union[str, Path] = "trees", db_path: Union[str, Path] = "trees/lattice.db") -> bool:
    """Ensure target SQLite database is up-to-date with domain trees."""
    trees_path = Path(trees_dir)
    target_db = Path(db_path)

    schema_current = target_db.exists() and get_stored_meta(target_db, "compiler_schema") == COMPILER_SCHEMA_VERSION

    if target_db.exists() and schema_current:
        try:
            db_mtime = target_db.stat().st_mtime_ns
            dir_mtime = trees_path.stat().st_mtime_ns
            if db_mtime >= dir_mtime:
                stored_fp = get_stored_fingerprint(target_db)
                if stored_fp is not None:
                    entries = [e for e in os.scandir(trees_path) if e.is_file() and e.name.endswith(".json")]
                    if not entries and stored_fp == "EMPTY":
                        return False
                    if entries and all(e.stat().st_mtime_ns <= db_mtime for e in entries):
                        return False
        except OSError:
            pass

    current_fp = compute_trees_fingerprint(trees_path)
    stored_fp = get_stored_fingerprint(target_db)

    if target_db.exists() and schema_current and stored_fp is not None and stored_fp == current_fp:
        return False

    if current_fp == "EMPTY":
        init_sqlite_db(target_db, clean=True)
        record_trees_fingerprint(target_db, current_fp)
        return True

    print(f"[*] Tree update detected in '{trees_path}'. Auto-compiling lattice database '{target_db}'...")
    root_dir = str(Path(__file__).resolve().parent.parent)
    try:
        from tools.compile_trees import compile_database
    except ImportError:
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from tools.compile_trees import compile_database

    init_sqlite_db(target_db, clean=True)
    written = compile_database(output_db=str(target_db), domain_filter=None, trees_dir=str(trees_path))
    record_trees_fingerprint(target_db, current_fp)
    print(f"[*] Compilation Complete: {written} total verified nodes compiled into '{target_db}'.")
    return True


@dataclass
class CompiledLattice:
    """Lightweight Pydantic-validated lattice snapshot loaded directly from compiled SQLite v4 storage."""

    cells: List[CellSchema] = field(default_factory=list)
    type_parents: Dict[str, Set[str]] = field(default_factory=dict)
    state_parents: Dict[str, Set[str]] = field(default_factory=dict)
    top_types: List[str] = field(default_factory=list)
    domain_imports: Dict[str, List[str]] = field(default_factory=dict)
    domain_runtime_setups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    language_spec: Optional[LanguageSpec] = None
    schema_version: str = COMPILER_SCHEMA_VERSION
    trees_fingerprint: Optional[str] = None

    @property
    def loaded_cells(self) -> Dict[str, CellSchema]:
        return {c.cell_id: c for c in self.cells if c.cell_id}


def _load_json_safe(raw: Optional[str], fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def load_compiled_lattice(db_path: Union[str, Path]) -> CompiledLattice:
    """Load CellSchema nodes, posets, domain metadata, and LanguageSpec with 100% v4 table coverage."""
    target = Path(db_path)
    if not target.exists():
        raise FileNotFoundError(f"Compiled lattice database not found: {target}")

    cells: List[CellSchema] = []
    type_parents: Dict[str, Set[str]] = {}
    state_parents: Dict[str, Set[str]] = {}
    top_types: List[str] = []
    domain_imports: Dict[str, List[str]] = {}
    domain_runtime_setups: Dict[str, Dict[str, Any]] = {}
    language_spec: Optional[LanguageSpec] = None
    schema_version = COMPILER_SCHEMA_VERSION
    trees_fingerprint: Optional[str] = None

    with contextlib.closing(sqlite3.connect(str(target))) as conn:
        cur = conn.cursor()
        tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        if "_compilation_meta" in tables:
            for k, v in cur.execute("SELECT key, value FROM _compilation_meta").fetchall():
                if k == "compiler_schema" and v is not None:
                    schema_version = str(v)
                elif k == "trees_fingerprint" and v is not None:
                    trees_fingerprint = str(v)

        if "nodes" in tables:
            for cid, dom_name, cell_json in cur.execute(
                "SELECT cell_id, domain_name, cell_json FROM nodes ORDER BY cell_id"
            ).fetchall():
                cell_dict = _load_json_safe(cell_json, None)
                if not isinstance(cell_dict, dict):
                    continue
                if cid and not cell_dict.get("cell_id"):
                    cell_dict["cell_id"] = cid
                dom_l = str(dom_name or "").strip().lower()
                if dom_l and not cell_dict.get("domain_name"):
                    cell_dict["domain_name"] = dom_l
                cells.append(CellSchema(**cell_dict))

        if "types" in tables:
            for t_name, p_type, dom_name in cur.execute(
                "SELECT type_name, parent_type, domain_name FROM types"
            ).fetchall():
                t_l = str(t_name or "").strip().lower()
                p_l = str(p_type or "").strip().lower()
                dom_l = str(dom_name or "").strip().lower()
                if t_l and p_l:
                    type_parents.setdefault(t_l, set()).add(p_l)
                    if dom_l and "::" not in t_l:
                        type_parents.setdefault(f"{dom_l}::{t_l}", set()).add(p_l)

        if "typestates" in tables:
            for s_name, s_parent, s_dom in cur.execute(
                "SELECT state_name, parent_state, domain_name FROM typestates"
            ).fetchall():
                dom_l = str(s_dom or "").strip().lower()
                sn = normalize_typestate(str(s_name or ""), dom_l)
                sp = normalize_typestate(str(s_parent or ""), dom_l)
                if sn != "any" and sp != "any":
                    state_parents.setdefault(sn, set()).add(sp)

        if "domain_metadata" in tables:
            for dom_name, _is_core, imp_json, top_json, rs_json, ls_json in cur.execute(
                "SELECT domain_name, is_core, imports_json, top_types_json, runtime_setup_json, language_spec_json "
                "FROM domain_metadata ORDER BY is_core DESC, domain_name ASC"
            ).fetchall():
                dom_l = str(dom_name or "").strip().lower()
                if not dom_l:
                    continue
                imps = _load_json_safe(imp_json, [])
                if isinstance(imps, list):
                    domain_imports[dom_l] = [str(x).strip().lower() for x in imps if str(x).strip()]
                tts = _load_json_safe(top_json, [])
                if isinstance(tts, list):
                    for tt in tts:
                        tt_l = str(tt or "").strip().lower()
                        if tt_l and tt_l not in top_types:
                            top_types.append(tt_l)
                rs = _load_json_safe(rs_json, None)
                if isinstance(rs, dict):
                    domain_runtime_setups[dom_l] = rs
                if language_spec is None and ls_json:
                    ls = _load_json_safe(ls_json, None)
                    if isinstance(ls, dict):
                        language_spec = LanguageSpec(**ls)

    return CompiledLattice(
        cells=cells,
        type_parents=type_parents,
        state_parents=state_parents,
        top_types=top_types,
        domain_imports=domain_imports,
        domain_runtime_setups=domain_runtime_setups,
        language_spec=language_spec,
        schema_version=schema_version,
        trees_fingerprint=trees_fingerprint,
    )


__all__ = [
    "COMPILER_SCHEMA_VERSION",
    "CompiledLattice",
    "compute_trees_fingerprint",
    "ensure_lattice_compiled",
    "get_stored_fingerprint",
    "get_stored_meta",
    "init_sqlite_db",
    "load_compiled_lattice",
    "record_trees_fingerprint",
]
