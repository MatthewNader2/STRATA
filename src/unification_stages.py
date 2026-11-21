"""
src/unification_stages.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Track 2.2: Modular Unification Pipeline Stages.

Active modular stages used by UnificationGate:
  1. PortWiringStage           - expands composite MacroCells into constituent micro-cells.
  2. ScopeResolutionStage      - attaches TypedBindingRecord metadata to resolved cell ports.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from log_config import get_logger

try:
    from .lattice import Cell, MacroCell
    from .unification import (
        ExecutionContext,
        Substitution,
        TypedBindingRecord,
        resolve_typed_binding_record,
    )
    from .diagnostics import DiagnosticTelemetry
except (ImportError, ValueError):
    from lattice import Cell, MacroCell
    from unification import (
        ExecutionContext,
        Substitution,
        TypedBindingRecord,
        resolve_typed_binding_record,
    )
    from diagnostics import DiagnosticTelemetry

logger = get_logger("unification_stages")


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


__all__ = [
    "PortWiringStage",
    "ScopeResolutionStage",
]
