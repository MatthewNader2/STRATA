"""Shared pytest configuration and deterministic session fixtures for STRATA v2."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
MEDIA_FIXTURES_DIR = ROOT_DIR / "benchmarks" / "fixtures" / "media"

for _p in (str(SRC_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gevr_sandbox import GEVRSandbox  # noqa: E402
from lattice_db import ensure_lattice_compiled, load_compiled_lattice  # noqa: E402
from strata_pipeline import StrataPipeline  # noqa: E402


@pytest.fixture(scope="session")
def compiled_lattice():
    """Compile (if needed) and load the canonical SQLite v4 lattice."""
    db_path = ROOT_DIR / "trees" / "lattice.db"
    ensure_lattice_compiled(trees_dir=str(ROOT_DIR / "trees"), db_path=str(db_path))
    return load_compiled_lattice(db_path)


@pytest.fixture(scope="session")
def sandbox():
    """Return a headless GEVRSandbox configured with tracked media fixtures."""
    return GEVRSandbox(
        default_timeout=30.0,
        allowed_capabilities=set(),
        asset_dirs=[MEDIA_FIXTURES_DIR],
    )


@pytest.fixture(scope="session")
def profile0_pipeline(compiled_lattice, sandbox):
    """Return a pure-symbolic Profile 0 pipeline wired to the headless sandbox."""
    return StrataPipeline.from_compiled(
        compiled_lattice,
        sandbox_exec_fn=lambda code, **kw: sandbox.execute(code, **kw),
    )
