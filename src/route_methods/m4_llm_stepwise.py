"""
src/route_methods/m4_llm_stepwise.py - STRATA (Neuro-Symbolic Topological Lattice)

DEPRECATED: M4 (bare-name stepwise) has been superseded by M8 (type-aware stepwise pathfinder).
M4LLMStepwiseRouteMethod is maintained as a backwards-compatible alias to M8LLMStepwisePathfinderRouteMethod.
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Any

from .m8_llm_stepwise import M8LLMStepwisePathfinderRouteMethod


class M4LLMStepwiseRouteMethod(M8LLMStepwisePathfinderRouteMethod):
    name = "m4_llm_stepwise"

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "M4LLMStepwiseRouteMethod is deprecated and aliased to M8LLMStepwisePathfinderRouteMethod.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
