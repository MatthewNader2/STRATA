"""
src/harvester.py - STRATA (Neuro-Symbolic Topological Lattice)
Universal Harvester Module.
Eliminates redundant subclass shims and delegates category-theoretic reflection
directly to UniversalHarvester using native domain_name resolution.
"""

from typing import Any, Optional

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from universal_harvester import UniversalHarvester
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .universal_harvester import UniversalHarvester

# Direct alias: UniversalHarvester natively manages domain_name and package introspection.
IntelligentHarvester = UniversalHarvester

# Ensure backward-compatible property access if legacy callers query .domain
if not hasattr(UniversalHarvester, "domain"):
    UniversalHarvester.domain = property(  # type: ignore[attr-defined]
        lambda self: getattr(self, "domain_name", ""),
        lambda self, val: setattr(self, "domain_name", val),
    )

__all__ = [
    "CellSchema",
    "PortSchema",
    "TreeSchema",
    "UniversalHarvester",
    "IntelligentHarvester",
]
