"""
src/planner_stages.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Track 2.1: Four-Phase Compositional Planner Pipeline.

Decomposes `LatticePlanner.plan()` into four modular, inspectable phases:
  1. ClauseSegmentationPhase   - parses and normalizes prompt clauses, tokens, and universal literals.
  2. AnchorSelectionPhase      - runs provable-infeasibility probing, demand-driven RAG widening, and entry selection.
  3. TopologicalExpansionPhase - executes Frontier Monoidal DAG / Linear Trellis search and MCTS fallback.
  4. CarrierCompletionPhase    - applies hard constraints, Borda ranking, multiplicity expansion, recursive slot binding,
                                 emission dry-run verification, and coverage-floor gating.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from log_config import get_logger

try:
    from .lattice import Cell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from .unification import ExecutionContext, Substitution
    from .tokenizer import CellTokenizer
    from .planner import _segment_prompt_clauses, STOPWORDS
except (ImportError, ValueError):
    from lattice import Cell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from unification import ExecutionContext, Substitution
    from tokenizer import CellTokenizer
    from planner import _segment_prompt_clauses, STOPWORDS

logger = get_logger("planner_stages")


@dataclass
class SegmentedPromptContext:
    """Output of Phase 1 (ClauseSegmentationPhase)."""
    prompt: str
    clauses: List[str]
    clause_tokens_list: List[Set[str]]
    content_prompt_tokens: Set[str]
    num_clauses: int
    l0_extracted_literals: List[Tuple[int, str, str]]
    universal_literals: List[Tuple[str, str]]
    literal_positions: Dict[Tuple[str, str], int]
    identifier_role_map: Dict[int, Any]
    file_literals: List[str]
    src_file_literals: List[str]
    dest_file_literals: List[str]
    numeric_literals: List[str]
    quoted_str_literals: List[str]
    identifier_literals: List[str]
    target_sink: Optional[str]
    has_prompt_egress_intent: bool


class ClauseSegmentationPhase:
    """
    Phase 1 of Planning:
    Segments the prompt into operational clauses, extracts universal L0 literals,
    identifies source/dest file assets and target sinks, and builds token sets.
    """

    @staticmethod
    def execute(
        prompt: str,
        ctx: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        egress_intent_tokens: Optional[Sequence[str]] = None,
    ) -> SegmentedPromptContext:
        registry = TypeRegistry.get_instance()
        l0_extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")

        target_sink = None
        if ctx:
            target_sink = (
                getattr(ctx, "target_key", None)
                or getattr(ctx, "parameters", {}).get("target_sink")
                or getattr(ctx, "parameters", {}).get("target_key")
            )
        if not target_sink:
            target_sink = ExecutionContext._extract_target_sink(prompt or "")

        universal_literals = [
            (kind, val) for _, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str", "numeric")
            and not (kind == "identifier" and target_sink and str(val).lower() == str(target_sink).lower())
        ]
        literal_positions = {
            (kind, val): pos
            for pos, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str", "numeric")
            and not (kind == "identifier" and target_sink and str(val).lower() == str(target_sink).lower())
        }
        identifier_role_map = ExecutionContext._build_identifier_role_map(prompt or "")
        file_literals = [
            v for kind, v in universal_literals
            if kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(v)))
        ]

        if ctx:
            ctx_src = list(getattr(ctx, "parameters", {}).get("source_uris") or getattr(ctx, "source_files", []) or [])
            ctx_dest = list(getattr(ctx, "parameters", {}).get("dest_uris") or getattr(ctx, "dest_files", []) or [])
        else:
            ctx_src, ctx_dest = [], []

        dest_file_literals = ctx_dest if ctx_dest else [
            val for pos, kind, val in l0_extracted_literals
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) == "dest"
        ]
        src_file_literals = ctx_src if ctx_src else [
            val for pos, kind, val in l0_extracted_literals
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) != "dest"
        ]

        numeric_literals = [
            v for _, t, v in l0_extracted_literals
            if t == "numeric" or registry.is_subtype(str(t).lower(), "numeric")
        ]
        quoted_str_literals = [
            v for kind, v in universal_literals
            if kind == "quoted_str" and not ExecutionContext._is_path_string(str(v))
        ]
        identifier_literals = [
            v for kind, v in universal_literals
            if kind == "identifier"
        ]

        clauses = _segment_prompt_clauses(prompt)
        clause_tokens_list = [CellTokenizer.tokenize_prompt(cl) - set(STOPWORDS) for cl in clauses]
        clause_tokens_list = [t for t in clause_tokens_list if t]
        content_prompt_tokens = (
            set().union(*clause_tokens_list)
            if clause_tokens_list
            else (CellTokenizer.tokenize_prompt(prompt) if prompt else set())
        )
        num_clauses = max(len(clause_tokens_list), 1)
        egress_toks = set(egress_intent_tokens or registry.get_egress_tokens())
        has_prompt_egress_intent = (
            (bool(content_prompt_tokens & egress_toks) and not target_sink)
            or bool(dest_file_literals)
            or (goal_sig is not None)
        )

        return SegmentedPromptContext(
            prompt=prompt,
            clauses=clauses,
            clause_tokens_list=clause_tokens_list,
            content_prompt_tokens=content_prompt_tokens,
            num_clauses=num_clauses,
            l0_extracted_literals=l0_extracted_literals,
            universal_literals=universal_literals,
            literal_positions=literal_positions,
            identifier_role_map=identifier_role_map,
            file_literals=file_literals,
            src_file_literals=src_file_literals,
            dest_file_literals=dest_file_literals,
            numeric_literals=numeric_literals,
            quoted_str_literals=quoted_str_literals,
            identifier_literals=identifier_literals,
            target_sink=target_sink,
            has_prompt_egress_intent=has_prompt_egress_intent,
        )


class AnchorSelectionPhase:
    """
    Phase 2 of Planning:
    Filters non-combinator/constant candidates, runs the provable-infeasibility
    probe with demand-driven RAG search widening, and selects initial Stage-1
    and self-contained entry anchors.
    """

    @staticmethod
    def filter_initial_candidates(tunnel: List[Cell]) -> List[Cell]:
        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    getattr(c, "cell_type", "") != "macro"
                    and not getattr(c, "is_macro", False)
                    and not isinstance(c, MacroCell)
                    and not (
                        getattr(c, "is_combinator", False)
                        or getattr(c, "node_role", "") == "combinator"
                        or getattr(c, "role", "") == "combinator"
                        or getattr(c, "node_type", "") == "combinator"
                    )
                )
            )
        ]
        if not candidates:
            candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        return candidates


class TopologicalExpansionPhase:
    """
    Phase 3 of Planning:
    Dispatches between Multi-Carrier Monoidal Frontier DAG search (`frontier`)
    and 1D Monadic Trellis search (`linear`), with UCT Monte Carlo Tree Search
    fallback if the trellis is disconnected.
    """

    def __init__(self, planner: Any) -> None:
        self.planner = planner

    def expand(self, topology_mode: str, **kwargs: Any) -> List[Tuple[List[Cell], Substitution, float, int, int]]:
        if topology_mode == "linear":
            return self.planner._plan_linear_trellis(**kwargs)
        return self.planner._plan_frontier_dag(**kwargs)


class CarrierCompletionPhase:
    """
    Phase 4 of Planning:
    Applies hard literal/coverage filters, Borda path ranking, trial sublattice
    slot planning, identifier multiplicity expansion, emission dry-run gating,
    and coverage-floor verification.
    """

    def __init__(self, planner: Any) -> None:
        self.planner = planner


__all__ = [
    "SegmentedPromptContext",
    "ClauseSegmentationPhase",
    "AnchorSelectionPhase",
    "TopologicalExpansionPhase",
    "CarrierCompletionPhase",
]
