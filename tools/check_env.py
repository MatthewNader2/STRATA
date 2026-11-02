#!/usr/bin/env python3
"""
tools/check_env.py - STRATA Environment & Database Sync Verifier (Track 4.3 & 4.4).

Usage:
  python3 tools/check_env.py --write-lock   # Freeze current tested versions into requirements.lock
  python3 tools/check_env.py --check        # Verify environment against requirements.lock and check lattice.db sync
"""

from __future__ import annotations
import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CRITICAL_PACKAGES: Tuple[str, ...] = (
    "numpy",
    "pydantic",
    "pydantic-settings",
    "fastapi",
    "rich",
    "torch",
    "transformers",
    "sentence-transformers",
    "llama-cpp-python",
    "faiss-cpu",
)


def get_installed_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in CRITICAL_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Check faiss-gpu if faiss-cpu is not installed
            if pkg == "faiss-cpu":
                try:
                    versions["faiss-gpu"] = importlib.metadata.version("faiss-gpu")
                    continue
                except importlib.metadata.PackageNotFoundError:
                    pass
            versions[pkg] = "MISSING"
    return versions


def write_lockfile(lock_path: Path) -> int:
    versions = get_installed_versions()
    lines = [
        "# STRATA Pinned Environment Lockfile (Track 4.3)",
        f"# Python: {sys.version.split()[0]}",
    ]
    missing: List[str] = []
    for pkg, ver in versions.items():
        if ver == "MISSING":
            missing.append(pkg)
            lines.append(f"# {pkg}==MISSING")
        else:
            lines.append(f"{pkg}=={ver}")

    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] Wrote pinned dependencies to '{lock_path}'.")
    if missing:
        print(f"[!] Warning: optional/neural packages not installed in current env: {', '.join(missing)}")
    return 0


def check_lockfile(lock_path: Path) -> bool:
    if not lock_path.exists():
        print(f"[!] Lockfile '{lock_path}' does not exist. Run with --write-lock first.")
        return False

    installed = get_installed_versions()
    mismatches: List[str] = []
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pkg, expected_ver = line.split("==", 1)
        actual_ver = installed.get(pkg.strip(), "MISSING")
        if actual_ver != expected_ver.strip():
            mismatches.append(f"{pkg}: expected {expected_ver}, found {actual_ver}")

    if mismatches:
        print("[X] Environment lockfile mismatch:")
        for m in mismatches:
            print(f"  - {m}")
        return False

    print("[✓] All pinned packages match requirements.lock.")
    return True


def verify_database_sync(trees_dir: Path, db_path: Path, auto_compile: bool = False) -> bool:
    from cli import compute_trees_fingerprint, get_stored_fingerprint, ensure_lattice_compiled

    if auto_compile:
        rebuilt = ensure_lattice_compiled(trees_dir=trees_dir, db_path=db_path)
        if rebuilt:
            print(f"[+] Recompiled '{db_path}' to match '{trees_dir}'.")

    current_fp = compute_trees_fingerprint(trees_dir)
    stored_fp = get_stored_fingerprint(db_path)
    if current_fp in ("MISSING", "EMPTY"):
        print(f"[!] Trees directory '{trees_dir}' is {current_fp}.")
        return False

    if stored_fp != current_fp:
        print(
            f"[X] Database out of sync! '{db_path}' fingerprint={stored_fp}, "
            f"'{trees_dir}' fingerprint={current_fp}."
        )
        print("    Run: python3 -m src.cli compile --clean")
        return False

    print(f"[✓] Database '{db_path}' is strictly synchronized with '{trees_dir}' (sha256={current_fp[:12]}...).")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="STRATA Environment & Lattice DB Sync Checker")
    parser.add_argument("--lockfile", type=Path, default=PROJECT_ROOT / "requirements.lock")
    parser.add_argument("--trees-dir", type=Path, default=PROJECT_ROOT / "trees")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "trees" / "lattice.db")
    parser.add_argument("--write-lock", action="store_true", help="Generate requirements.lock from current environment")
    parser.add_argument("--auto-compile", action="store_true", help="Auto-compile lattice.db if out of sync")
    args = parser.parse_args()

    if args.write_lock:
        write_lockfile(args.lockfile)

    env_ok = check_lockfile(args.lockfile)
    db_ok = verify_database_sync(args.trees_dir, args.db_path, auto_compile=args.auto_compile)
    return 0 if (env_ok and db_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
