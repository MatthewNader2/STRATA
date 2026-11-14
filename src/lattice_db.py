"""
src/lattice_db.py - Lightweight stdlib-only SQLite schema & fingerprint utilities
for STRATA domain tree compilation verification.

Kept free of heavy runtime imports (torch, transformers, router, inference, rich)
so cold-booting `LatticeOrchestrator` does not transitively load neural stacks.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, Union

# Bump whenever the compiled node serialization changes: forces stale databases to rebuild.
COMPILER_SCHEMA_VERSION = "2"


def init_sqlite_db(db_path: Path, clean: bool = False) -> None:
    """Initializes standard STRATA SQLite schema without destroying existing data unless clean=True."""
    if clean and db_path.exists():
        os.remove(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                cell_id              TEXT PRIMARY KEY,
                domain_name          TEXT,
                node_type            TEXT,
                node_role            TEXT DEFAULT 'function',
                stage                INTEGER,
                keywords             TEXT,
                input_type           TEXT,
                input_state          TEXT,
                output_type          TEXT,
                output_state         TEXT,
                code                 TEXT,
                dependencies         TEXT,
                configuration_schema TEXT,
                slots                TEXT DEFAULT '{}',
                verified             INTEGER DEFAULT 0,
                docstring            TEXT DEFAULT '',
                enrichment_source    TEXT DEFAULT NULL,
                enriched_at          TEXT DEFAULT NULL,
                source_provenance    TEXT DEFAULT 'unknown',
                source_priority      INTEGER DEFAULT 100,
                raw_cell             TEXT DEFAULT NULL
            )
        """)
        cur.execute("PRAGMA table_info(nodes)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "slots" not in existing_cols:
            try:
                cur.execute("ALTER TABLE nodes ADD COLUMN slots TEXT DEFAULT '{}'")
            except Exception:
                pass
        if "raw_cell" not in existing_cols:
            try:
                cur.execute("ALTER TABLE nodes ADD COLUMN raw_cell TEXT DEFAULT NULL")
            except Exception:
                pass

        cur.execute("CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_type ON nodes(node_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_slots ON nodes(slots)")
        cur.execute("CREATE TABLE IF NOT EXISTS types (type_name TEXT, parent_type TEXT, domain_name TEXT, PRIMARY KEY (type_name, parent_type))")
        cur.execute("CREATE TABLE IF NOT EXISTS typestates (state_name TEXT PRIMARY KEY, parent_state TEXT, carrier_type TEXT, properties TEXT, domain_name TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY, canonical TEXT, domain_name TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS artifact_readers (category TEXT, module_name TEXT, function_name TEXT, domain_name TEXT, PRIMARY KEY (category, module_name, function_name))")
        cur.execute("CREATE TABLE IF NOT EXISTS structural_metadata (category TEXT, item TEXT, extra TEXT, domain_name TEXT, PRIMARY KEY (category, item, extra, domain_name))")
        cur.execute("CREATE TABLE IF NOT EXISTS _compilation_meta (key TEXT PRIMARY KEY, value TEXT, timestamp REAL)")
        conn.commit()


def compute_trees_fingerprint(trees_dir: Union[str, Path] = "trees") -> str:
    """Computes a fast SHA-256 fingerprint over all tree JSON files in the directory."""
    td = Path(trees_dir)
    if not td.exists():
        return "MISSING"
    try:
        entries = sorted([e for e in os.scandir(td) if e.is_file() and e.name.endswith(".json")], key=lambda x: x.name)
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
    """Reads one value from the compilation metadata table (None when absent)."""
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
    """Retrieves the compilation fingerprint recorded in the SQLite database."""
    target = Path(db_path)
    if not target.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cur = conn.cursor()
            cur.execute("SELECT value FROM _compilation_meta WHERE key = 'trees_fingerprint'")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def record_trees_fingerprint(db_path: Union[str, Path], fingerprint: str) -> None:
    """Records the tree compilation fingerprint into the SQLite database."""
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS _compilation_meta (key TEXT PRIMARY KEY, value TEXT, timestamp REAL)")
            cur.execute(
                "INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('trees_fingerprint', ?, ?)",
                (fingerprint, time.time()),
            )
            cur.execute(
                "INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('compiler_schema', ?, ?)",
                (COMPILER_SCHEMA_VERSION, time.time()),
            )
            conn.commit()
    except Exception:
        pass


def ensure_lattice_compiled(trees_dir: Union[str, Path] = "trees", db_path: Union[str, Path] = "trees/lattice.db") -> bool:
    """
    Ensures target SQLite database is up-to-date with domain trees.
    Performs fast O(1) directory mtime checks on startup to eliminate stat storm bottlenecks.
    """
    trees_path = Path(trees_dir)
    target_db = Path(db_path)

    schema_current = (not target_db.exists()) or get_stored_meta(target_db, "compiler_schema") == COMPILER_SCHEMA_VERSION

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
