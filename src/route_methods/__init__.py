"""
src/route_methods/__init__.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Pluggable RouteMethods registry and factory (M0 - M10).
"""

from typing import Dict, Type, Optional

from .base import RouteMethod
from .m0_trellis import M0TrellisRouteMethod
from .m1_clause_anchor import M1ClauseAnchorRouteMethod
from .m2_endpoint_anchor import M2EndpointAnchorRouteMethod
from .m3_greedy_freeze import M3GreedyFreezeRouteMethod
from .m4_llm_stepwise import M4LLMStepwiseRouteMethod
from .m5_llm_oneshot import M5LLMOneShotRouteMethod
from .m6_hybrid_anchors import M6HybridAnchorsRouteMethod
from .m7_llm_pathfinder import M7LLMPathfinderRouteMethod
from .m8_llm_stepwise import M8LLMStepwisePathfinderRouteMethod
from .m9_llm_milestones import M9LLMMilestonePathfinderRouteMethod
from .m10_monte_carlo import M10MonteCarloRouteMethod
from .m11_latent_cognitive import M11LatentCognitiveRouteMethod

ROUTE_METHOD_REGISTRY: Dict[str, Type[RouteMethod]] = {
    "m0": M0TrellisRouteMethod,
    "m0_trellis": M0TrellisRouteMethod,
    "trellis": M0TrellisRouteMethod,
    "viterbi": M0TrellisRouteMethod,

    "m1": M1ClauseAnchorRouteMethod,
    "m1_clause_anchor": M1ClauseAnchorRouteMethod,
    "clause": M1ClauseAnchorRouteMethod,
    "clause_anchor": M1ClauseAnchorRouteMethod,

    "m2": M2EndpointAnchorRouteMethod,
    "m2_endpoint_anchor": M2EndpointAnchorRouteMethod,
    "endpoint": M2EndpointAnchorRouteMethod,
    "endpoint_anchor": M2EndpointAnchorRouteMethod,

    "m3": M3GreedyFreezeRouteMethod,
    "m3_greedy_freeze": M3GreedyFreezeRouteMethod,
    "greedy": M3GreedyFreezeRouteMethod,
    "greedy_freeze": M3GreedyFreezeRouteMethod,

    "m4": M4LLMStepwiseRouteMethod,
    "m4_llm_stepwise": M4LLMStepwiseRouteMethod,
    "llm_stepwise": M4LLMStepwiseRouteMethod,

    "m5": M5LLMOneShotRouteMethod,
    "m5_llm_oneshot": M5LLMOneShotRouteMethod,
    "llm_oneshot": M5LLMOneShotRouteMethod,

    "m6": M6HybridAnchorsRouteMethod,
    "m6_hybrid_anchors": M6HybridAnchorsRouteMethod,
    "hybrid": M6HybridAnchorsRouteMethod,
    "hybrid_anchors": M6HybridAnchorsRouteMethod,

    "m7": M7LLMPathfinderRouteMethod,
    "m7_llm_pathfinder": M7LLMPathfinderRouteMethod,
    "pathfinder": M7LLMPathfinderRouteMethod,
    "llm_pathfinder": M7LLMPathfinderRouteMethod,

    "m8": M8LLMStepwisePathfinderRouteMethod,
    "m8_llm_stepwise": M8LLMStepwisePathfinderRouteMethod,
    "stepwise_pathfinder": M8LLMStepwisePathfinderRouteMethod,

    "m9": M9LLMMilestonePathfinderRouteMethod,
    "m9_llm_milestones": M9LLMMilestonePathfinderRouteMethod,
    "milestone_pathfinder": M9LLMMilestonePathfinderRouteMethod,
    "milestones": M9LLMMilestonePathfinderRouteMethod,

    "m10": M10MonteCarloRouteMethod,
    "m10_monte_carlo": M10MonteCarloRouteMethod,
    "monte_carlo": M10MonteCarloRouteMethod,
    "mcts": M10MonteCarloRouteMethod,

    "m11": M11LatentCognitiveRouteMethod,
    "m11_latent_cognitive": M11LatentCognitiveRouteMethod,
    "latent": M11LatentCognitiveRouteMethod,
    "latent_cognitive": M11LatentCognitiveRouteMethod,
}


def route_method_ids() -> list:
    """Canonical ids (M0, M1, ... in numeric order) derived from the registry, so no caller keeps its own list."""
    ids = {k.upper() for k in ROUTE_METHOD_REGISTRY if k[:1] == "m" and k[1:].isdigit()}
    return sorted(ids, key=lambda t: int(t[1:]))


def describe_route_methods() -> list:
    """[(id, module-style name, one-line description)] from each method's own class."""
    out = []
    for mid in route_method_ids():
        cls = ROUTE_METHOD_REGISTRY[mid.lower()]
        doc = next((ln.strip() for ln in (cls.__doc__ or "").strip().splitlines() if ln.strip()), cls.__name__)
        out.append((mid, getattr(cls, "name", cls.__name__), doc))
    return out


def get_route_method(name: str = "m0", **kwargs) -> RouteMethod:
    """
    Factory function returning an instantiated RouteMethod by name or alias.
    """
    clean_name = str(name).strip().lower()
    cls = ROUTE_METHOD_REGISTRY.get(clean_name)
    if not cls:
        raise ValueError(
            f"Unknown route method '{name}'. Available: {list(ROUTE_METHOD_REGISTRY.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "RouteMethod",
    "M0TrellisRouteMethod",
    "M1ClauseAnchorRouteMethod",
    "M2EndpointAnchorRouteMethod",
    "M3GreedyFreezeRouteMethod",
    "M4LLMStepwiseRouteMethod",
    "M5LLMOneShotRouteMethod",
    "M6HybridAnchorsRouteMethod",
    "M7LLMPathfinderRouteMethod",
    "M8LLMStepwisePathfinderRouteMethod",
    "M9LLMMilestonePathfinderRouteMethod",
    "M10MonteCarloRouteMethod",
    "ROUTE_METHOD_REGISTRY",
    "route_method_ids",
    "describe_route_methods",
    "get_route_method",
]
