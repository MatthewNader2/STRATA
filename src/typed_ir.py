"""
src/typed_ir.py — Unified Typed Intermediate Representation (TypedIR) contract.

Separates frontend parsing (formal syntax vs. SLM semantic compilation) from
lattice retrieval, structured control-flow planning, and algebraic port binding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IRParamValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any = None
    provenance: Literal["prompt", "estimated", "default", "missing"] = "prompt"
    value_class: Literal["literal", "external_asset", "identifier", "enum"] = "literal"

    @model_validator(mode="after")
    def _validate_asset_honesty(self) -> "IRParamValue":
        if self.value_class == "external_asset" and self.provenance == "estimated":
            raise ValueError("External assets cannot be estimated or fabricated; use provenance='missing'.")
        if self.provenance == "missing" and self.value is not None:
            raise ValueError("Missing parameter must have value=None.")
        return self


class IRNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["call", "loop", "branch"] = "call"
    intent: str
    domain_hint: Optional[str] = None
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    params: Dict[str, IRParamValue] = Field(default_factory=dict)
    condition: Optional[str] = None
    body: List["IRNode"] = Field(default_factory=list)
    else_body: List["IRNode"] = Field(default_factory=list)


def infer_loop_var(node: IRNode) -> Optional[str]:
    """Infer bound iteration variable name from a loop node's condition, falling back to outputs[0]."""
    cond = str(node.condition or "").strip()
    for sep in (" in ", " of "):
        if sep in cond:
            lhs = cond.split(sep, 1)[0].strip()
            if lhs.startswith("for "):
                lhs = lhs[4:].strip()
            if lhs.isidentifier():
                return lhs
    if node.outputs:
        out0 = str(node.outputs[0]).strip()
        if out0.isidentifier():
            return out0
    return None


class TypedIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mode: Literal["formal_syntax", "slm_compiled"]
    raw_prompt: str
    nodes: List[IRNode] = Field(default_factory=list)
    assets: Dict[str, IRParamValue] = Field(default_factory=dict)
    deliverable: Optional[str] = None
    ambiguities: List[str] = Field(default_factory=list)

    def flatten_nodes(self) -> List[IRNode]:
        """Deterministic pre-order DFS traversal over top-level nodes, loop bodies, and branches."""
        flat: List[IRNode] = []

        def _visit(node_list: List[IRNode]) -> None:
            for node in node_list:
                flat.append(node)
                if node.body:
                    _visit(node.body)
                if node.else_body:
                    _visit(node.else_body)

        _visit(self.nodes)
        return flat

    def grounded_assets(self) -> Dict[str, Any]:
        """Return strictly prompt-grounded external assets."""
        return {
            k: v.value
            for k, v in self.assets.items()
            if v.provenance == "prompt" and v.value is not None
        }

    def validate_dag(self) -> List[str]:
        """Validate non-empty nodes, unique IDs, lexically scoped DAG dependencies, and control blocks."""
        errs: List[str] = []
        flat = self.flatten_nodes()
        if not flat:
            errs.append("TypedIR contains zero nodes.")
            return errs

        for ak, av in self.assets.items():
            if av.value_class != "external_asset":
                errs.append(f"Asset '{ak}' must have value_class='external_asset', got '{av.value_class}'.")

        seen_ids: Set[str] = set()
        all_defined: Set[str] = set(self.assets.keys())
        for node in flat:
            if node.id:
                all_defined.add(node.id)
            for out_sym in node.outputs:
                if out_sym:
                    all_defined.add(out_sym)

        def _validate_block(node_list: List[IRNode], outer_defined: Set[str]) -> None:
            defined_symbols = set(outer_defined)
            for node in node_list:
                if not node.id or not str(node.id).strip():
                    errs.append("IRNode has empty id.")
                elif node.id in seen_ids:
                    errs.append(f"Duplicate IRNode id '{node.id}'.")
                seen_ids.add(node.id)

                if node.kind in ("loop", "branch"):
                    if not (node.condition and node.condition.strip()):
                        errs.append(f"{node.kind} node '{node.id}' requires non-empty condition.")
                    if not node.body:
                        errs.append(f"{node.kind} node '{node.id}' requires non-empty body.")
                elif node.body or node.else_body:
                    errs.append(f"Call node '{node.id}' cannot contain nested body/else_body blocks.")

                own_syms: Set[str] = {node.id} | {s for s in node.outputs if s}
                for inp in node.inputs:
                    if inp in own_syms:
                        errs.append(f"Cyclic self-dependency on node '{node.id}' input '{inp}'.")
                    elif inp not in defined_symbols:
                        if inp in all_defined:
                            errs.append(f"Cyclic or out-of-scope dependency on node '{node.id}' input '{inp}'.")
                        else:
                            errs.append(f"Undefined input reference '{inp}' in node '{node.id}'.")

                if node.body:
                    body_scope = defined_symbols | own_syms
                    if node.kind == "loop":
                        lv = infer_loop_var(node)
                        if lv:
                            body_scope = body_scope | {lv}
                    _validate_block(node.body, body_scope)
                if node.else_body:
                    _validate_block(node.else_body, defined_symbols | {node.id})

                defined_symbols.update(own_syms)

        _validate_block(self.nodes, set(self.assets.keys()))
        return errs


IRNode.model_rebuild()


__all__ = [
    "IRParamValue",
    "IRNode",
    "TypedIR",
    "infer_loop_var",
]
