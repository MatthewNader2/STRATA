"""
src/typed_proof.py - Single authoritative typed DAG proof across pipeline stages.

Eliminates the contract gap where Layer 3 propagated polymorphic typestates
into `ctx.variables` while Layer 3.5 (`PreflightLinter`) discarded them and
re-read static cell JSON output templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TypedStepProof:
    """Resolved input bindings, typed records, and propagated output signatures for one step."""

    step_idx: int
    cell: Any
    bindings: Dict[str, Any]
    input_records: Dict[str, Any] = field(default_factory=dict)
    output_signatures: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TypedPipelineProof:
    """Complete propagated type proof emitted by UnificationGate for PreflightLinter and Synthesis."""

    steps: List[TypedStepProof]
    var_signatures: Dict[str, Any]
    var_producers: Dict[str, Tuple[Any, str]]
    sigma: Any
    unresolved_ports: List[Tuple[str, str]] = field(default_factory=list)

    def as_bindings(self) -> List[Tuple[Any, Dict[str, Any]]]:
        return [(step.cell, step.bindings) for step in self.steps]

    def get_var_signature(self, var_name: str) -> Any:
        return self.var_signatures.get(var_name)


def build_typed_pipeline_proof(
    pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
    ctx: Any,
    sigma: Any,
) -> TypedPipelineProof:
    """Build a TypedPipelineProof from live propagated `ctx.variables` and step bindings."""
    var_signatures: Dict[str, Any] = {}
    ctx_vars = getattr(ctx, "variables", None) or {}
    for v_name, entry in ctx_vars.items():
        if isinstance(entry, tuple) and entry:
            var_signatures[v_name] = entry[0]
        elif entry is not None:
            var_signatures[v_name] = entry

    var_producers: Dict[str, Tuple[Any, str]] = {}
    steps: List[TypedStepProof] = []
    for idx, (cell, bnd) in enumerate(pipeline_bindings):
        bindings_d = bnd if isinstance(bnd, dict) else {}
        step_out_sigs: Dict[str, Any] = {}
        cell_outputs = getattr(cell, "outputs", None) or {}

        for out_name, static_sig in cell_outputs.items():
            bound_v = bindings_d.get(out_name)
            if isinstance(bound_v, str) and bound_v.isidentifier():
                live_sig = var_signatures.get(bound_v) or static_sig
                step_out_sigs[out_name] = live_sig
                var_signatures.setdefault(bound_v, live_sig)
                var_producers[bound_v] = (cell, out_name)

        out_v = bindings_d.get("output_var")
        if isinstance(out_v, str) and out_v.isidentifier():
            prim_out = getattr(cell, "primary_output", None)
            prim_name = getattr(prim_out, "name", None) if prim_out is not None else None
            fallback_sig = (
                step_out_sigs.get(prim_name)
                if prim_name
                else (next(iter(step_out_sigs.values()), None) or next(iter(cell_outputs.values()), None))
            )
            live_out_sig = var_signatures.get(out_v) or fallback_sig
            if live_out_sig is not None:
                step_out_sigs["output_var"] = live_out_sig
                var_signatures.setdefault(out_v, live_out_sig)
                var_producers.setdefault(out_v, (cell, "output_var"))

        try:
            setattr(cell, "_propagated_outputs", dict(step_out_sigs))
        except Exception:
            pass

        typed_map = dict(getattr(cell, "typed_bindings", None) or {})
        steps.append(
            TypedStepProof(
                step_idx=idx,
                cell=cell,
                bindings=bindings_d,
                input_records=typed_map,
                output_signatures=step_out_sigs,
            )
        )

    unresolved_raw = getattr(ctx, "unresolved_ports", None) or ()
    unresolved_ports: List[Tuple[str, str]] = [
        (str(item[0]), str(item[1]))
        for item in unresolved_raw
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]

    return TypedPipelineProof(
        steps=steps,
        var_signatures=var_signatures,
        var_producers=var_producers,
        sigma=sigma,
        unresolved_ports=unresolved_ports,
    )


__all__ = [
    "TypedPipelineProof",
    "TypedStepProof",
    "build_typed_pipeline_proof",
]
