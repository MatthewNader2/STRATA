"""
src/route_methods/m0_trellis.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
M0: Classical Viterbi / Trellis dynamic programming over the semantic tunnel.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any

from .base import RouteMethod

try:
    from ..lattice import Cell, LatticeOrchestrator
    from ..planner import LatticePlanner
    from ..unification import ExecutionContext
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from planner import LatticePlanner
    from unification import ExecutionContext


class M0TrellisRouteMethod(RouteMethod):
    """
    RouteMethod M0: Classical Trellis / Viterbi Dynamic Programming.
    Baseline monadic trellis search over the active semantic tunnel G|_T.
    """
    name: str = "m0_trellis"

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        ctx: Optional[ExecutionContext] = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        **kwargs
    ) -> List[Cell]:
        orch = orchestrator or self.orchestrator
        if not orch:
            raise ValueError("[M0] LatticeOrchestrator is required for Trellis planning.")

        planner = LatticePlanner(orchestrator=orch)
        return planner.plan(
            prompt=prompt,
            tunnel=tunnel,
            relevance_map=relevance_map,
            start_sig=start_sig,
            goal_sig=goal_sig,
            max_transforms=max_transforms,
            ctx=ctx,
            **kwargs
        )
