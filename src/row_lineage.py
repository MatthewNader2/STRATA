"""
src/row_lineage.py - Compatibility shim delegating to PreflightLinter causal key and extent checks.
"""

from __future__ import annotations

try:
    from .preflight import PreflightLinter
except (ImportError, ValueError):
    from preflight import PreflightLinter

analyze_column_definitions = PreflightLinter._verify_causal_key_def_use
analyze = PreflightLinter._verify_carrier_extent_epochs

__all__ = ["analyze_column_definitions", "analyze"]
