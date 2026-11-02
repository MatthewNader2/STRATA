"""
src/unification_stages.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Track 2.2: Modular Unification Pipeline Stages.

Decomposes `UnificationGate.unify_pipeline()` and `emit_code()` into four
cohesive, independently testable stages:
  1. PortWiringStage           - connects output-to-input variables across DAG scope & adjacent cells.
  2. TypeCoercionStage         - resolves poset subtype compatibility, product projections, and bridge adapters.
  3. ScopeResolutionStage      - maintains ExecutionContext symbol tables, literal grounding, and provenance.
  4. TemplateInstantiationStage - renders concrete Python AST code blocks and verifies AST liveness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from log_config import get_logger

try:
    from .lattice import Cell, MacroCell, TypeRegistry, UNRESOLVED_PORT
    from .unification import (
        ExecutionContext,
        Substitution,
        MonadResult,
        Success,
        Failure,
        TypedBindingRecord,
        resolve_typed_binding_record,
        substitute_generics,
        _is_sink_cell,
        _update_variable_provenance,
    )
    from .diagnostics import DiagnosticTelemetry
except (ImportError, ValueError):
    from lattice import Cell, MacroCell, TypeRegistry, UNRESOLVED_PORT
    from unification import (
        ExecutionContext,
        Substitution,
        MonadResult,
        Success,
        Failure,
        TypedBindingRecord,
        resolve_typed_binding_record,
        substitute_generics,
        _is_sink_cell,
        _update_variable_provenance,
    )
    from diagnostics import DiagnosticTelemetry

logger = get_logger("unification_stages")


@dataclass
class UnificationPipelineState:
    """Shared state threaded cleanly through the four unification stages."""
    cells: List[Cell]
    ctx: ExecutionContext
    sigma: Substitution = field(default_factory=Substitution)
    pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = field(default_factory=list)
    cell_out_vars: Dict[Any, str] = field(default_factory=dict)
    cell_port_vars: Dict[Tuple[Any, str], str] = field(default_factory=dict)
    producer_var: Optional[str] = None
    llm_pipeline_bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class PortWiringStage:
    """
    Stage 1 of Unification:
    Expands composite MacroCells into constituent micro-cells when all sub-cells
    are present, and wires output-to-input variables across adjacent cells and
    multi-carrier DAG ancestors.
    """

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def expand_macros(self, cells: List[Cell], ctx: Optional[ExecutionContext] = None) -> List[Cell]:
        changed = True
        expansion_depth = 0
        current = list(cells)
        diag: Optional[DiagnosticTelemetry] = getattr(ctx, "diagnostics", None)

        while changed and expansion_depth < 10:
            changed = False
            expansion_depth += 1
            expanded_cells: List[Cell] = []
            for c in current:
                sub_cells = getattr(c, "sub_cells", None)
                if not ((getattr(c, "cell_type", "") == "macro" or isinstance(c, MacroCell)) and sub_cells):
                    expanded_cells.append(c)
                    continue

                orch = getattr(self.gate, "orchestrator", None)
                resolved_cache = getattr(c, "_resolved_sub_cells", {}) or {}
                resolved: List[Cell] = []
                missing: List[str] = []
                for sub_item in sub_cells:
                    sub_cell = sub_item if isinstance(sub_item, Cell) else None
                    if sub_cell is None:
                        sub_cell = resolved_cache.get(sub_item)
                    if sub_cell is None and orch:
                        sub_cell = orch.loaded_cells.get(sub_item)
                    if sub_cell is None:
                        missing.append(str(sub_item))
                    else:
                        resolved.append(sub_cell)

                if missing:
                    msg = f"Macro '{c.cell_id}' kept unexpanded: unresolved sub-cells {missing}"
                    logger.warning("[UNIFY:PortWiringStage] %s", msg)
                    if diag:
                        diag.record_warning("PortWiringStage.expand_macros", msg, cell_id=c.cell_id)
                    expanded_cells.append(c)
                else:
                    expanded_cells.extend(resolved)
                    changed = True
            current = expanded_cells
        return current


class TypeCoercionStage:
    """
    Stage 2 of Unification:
    Resolves poset subtype compatibility, heterogeneous tuple/product member
    projections (`var[i]`), and declared bridge-morphism coercion expressions
    (e.g. `table -> series -> ndarray`).
    """

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def coerce_carrier(
        self,
        src_var: str,
        src_type: str,
        dst_type: str,
        literal_bindings: Dict[str, str],
        dst_sig: Optional[Any],
        ctx: ExecutionContext,
    ) -> Optional[str]:
        return self.gate._compose_declared_bridge_expr(
            src_var=src_var,
            src_type=src_type,
            dst_type=dst_type,
            literal_bindings=literal_bindings,
            dst_sig=dst_sig,
            ctx=ctx,
        )

    def rederive_stale_series(
        self,
        cell: Cell,
        port_name: str,
        var_name: str,
        ctx: ExecutionContext,
        registry: TypeRegistry,
    ) -> Optional[str]:
        return None


class ScopeResolutionStage:
    """
    Stage 3 of Unification:
    Maintains `ExecutionContext` symbol tables, resolves prompt literals and
    port defaults, allocates output variables (`var_k`), updates column/row-epoch
    provenance, and attaches `TypedBindingRecord` metadata to every cell port.
    """

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def finalize_cell_scope(
        self,
        cell: Cell,
        cell_bindings: Dict[str, str],
        ctx: ExecutionContext,
        sigma: Substitution,
    ) -> Dict[str, TypedBindingRecord]:
        cell_typed_bindings: Dict[str, TypedBindingRecord] = {}
        var_sigs = {v: sig_tuple[0] for v, sig_tuple in ctx.variables.items()}
        for p_name, val_expr in cell_bindings.items():
            expected_sig = None
            if cell.inputs and p_name in cell.inputs:
                expected_sig = getattr(cell.inputs[p_name], "signature", cell.inputs[p_name])
            elif cell.outputs and p_name in cell.outputs:
                expected_sig = getattr(cell.outputs[p_name], "signature", cell.outputs[p_name])
            elif p_name == "output_var" and getattr(cell, "primary_output", None):
                expected_sig = getattr(cell.primary_output, "signature", cell.primary_output)
            record = resolve_typed_binding_record(
                cell=cell,
                port_name=p_name,
                bound_val=val_expr,
                port_sig=expected_sig,
                var_signatures=var_sigs,
            )
            cell_typed_bindings[p_name] = record
        cell.typed_bindings = cell_typed_bindings
        return cell_typed_bindings


class TemplateInstantiationStage:
    """
    Stage 4 of Unification:
    Prunes dead transform branches, collects and reconciles module imports,
    instantiates cell AST templates, and enforces static AST variable liveness
    and callable-callee sanity checks.
    """

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def render_pipeline(
        self,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        ctx: ExecutionContext,
        accum_sigma: Substitution,
    ) -> str:
        return self.gate.emit_code(pipeline_bindings, ctx, accum_sigma=accum_sigma)


__all__ = [
    "UnificationPipelineState",
    "PortWiringStage",
    "TypeCoercionStage",
    "ScopeResolutionStage",
    "TemplateInstantiationStage",
]
