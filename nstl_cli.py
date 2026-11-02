#!/usr/bin/env python3
"""
nstl_cli.py - Root entry point for STRATA Toolchain CLI (backward compatibility alias)
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cli import main

if __name__ == "__main__":
    main()
