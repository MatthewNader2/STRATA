"""
src/lattice_auditor.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Static Reachability, Dead-End, and Disconnected Node Auditor (Phase 3 / T3.3).
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell

logger = get_logger("lattice_auditor")


@dataclass
class LatticeAuditReport:
    """Detailed audit metrics and diagnostic findings for the lattice topology."""
    total_cells: int
    stage_counts: Dict[int, int]
    domain_counts: Dict[str, int]
    entry_nodes: List[str]
    terminal_nodes: List[str]
    reachable_nodes: List[str]
    unreachable_nodes: List[str]
    dead_end_nodes: List[str]
    disconnected_nodes: List[str]
    cross_tree_edge_count: int
    summary: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self, max_unreachable_ratio: float = 0.25) -> bool:
        """Evaluates whether the lattice topology satisfies minimum health invariants."""
        if self.total_cells == 0:
            return False
        if not self.entry_nodes or not self.terminal_nodes:
            return False
        unreachable_ratio = len(self.unreachable_nodes) / max(self.total_cells, 1)
        return unreachable_ratio <= max_unreachable_ratio


class LatticeAuditor:
    """
    Static topological reachability and graph integrity auditor.
    Analyzes the loaded lattice category G = (V, E) without executing code.
    """
    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator

    def audit(self) -> LatticeAuditReport:
        """
        Performs full reachability, dead-end, and cross-tree analysis.
        """
        cells = self.orchestrator.loaded_cells
        total_cells = len(cells)
        adj = getattr(self.orchestrator, "_adjacency", {})
        rev_adj = getattr(self.orchestrator, "_reverse_adjacency", {})

        stage_counts: Dict[int, int] = defaultdict(int)
        domain_counts: Dict[str, int] = defaultdict(int)

        entry_nodes: List[str] = []
        terminal_nodes: List[str] = []
        constants: Set[str] = set()

        for cid, cell in cells.items():
            st = getattr(cell, "stage", 2) or 2
            dom = getattr(cell, "domain_name", "generic") or "generic"
            ntype = getattr(cell, "node_type", "function")

            stage_counts[st] += 1
            domain_counts[dom] += 1

            if ntype == "constant":
                constants.add(cid)
                continue

            if st == 1 or ntype == "constructor":
                entry_nodes.append(cid)
            if st == 3 or getattr(cell, "endable", False):
                terminal_nodes.append(cid)

        # BFS Reachability from all entry nodes
        visited: Set[str] = set()
        queue: deque[str] = deque(entry_nodes)
        visited.update(entry_nodes)

        cross_tree_edges = 0

        for u_id, neighbors in adj.items():
            u_cell = cells.get(u_id)
            u_dom = getattr(u_cell, "domain_name", "") if u_cell else ""
            for v_id in neighbors:
                v_cell = cells.get(v_id)
                v_dom = getattr(v_cell, "domain_name", "") if v_cell else ""
                if u_dom and v_dom and u_dom != v_dom:
                    cross_tree_edges += 1

        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, ()):
                if neighbor not in visited and neighbor in cells:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Categorize anomalies
        unreachable_nodes: List[str] = []
        dead_end_nodes: List[str] = []
        disconnected_nodes: List[str] = []

        for cid, cell in cells.items():
            if cid in constants:
                continue

            in_deg = len(rev_adj.get(cid, ()))
            out_deg = len(adj.get(cid, ()))
            st = getattr(cell, "stage", 2) or 2

            # Completely disconnected
            if in_deg == 0 and out_deg == 0:
                disconnected_nodes.append(cid)

            # Dead ends: has input flow, no outgoing edge, but not marked terminal (stage 3)
            if in_deg > 0 and out_deg == 0 and st != 3:
                dead_end_nodes.append(cid)

            # Unreachable from any Stage 1 entry node
            if cid not in visited and st != 1:
                unreachable_nodes.append(cid)

        summary = {
            "total_cells": total_cells,
            "entry_node_count": len(entry_nodes),
            "terminal_node_count": len(terminal_nodes),
            "reachable_ratio": round(len(visited) / max(total_cells, 1), 4),
            "cross_tree_edges": cross_tree_edges,
            "unreachable_count": len(unreachable_nodes),
            "dead_end_count": len(dead_end_nodes),
            "disconnected_count": len(disconnected_nodes),
        }

        return LatticeAuditReport(
            total_cells=total_cells,
            stage_counts=dict(stage_counts),
            domain_counts=dict(domain_counts),
            entry_nodes=entry_nodes,
            terminal_nodes=terminal_nodes,
            reachable_nodes=sorted(list(visited)),
            unreachable_nodes=unreachable_nodes,
            dead_end_nodes=dead_end_nodes,
            disconnected_nodes=disconnected_nodes,
            cross_tree_edge_count=cross_tree_edges,
            summary=summary
        )
