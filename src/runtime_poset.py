"""
src/runtime_poset.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Foundational Host Runtime Poset & Protocol Reflection Engine.

Operates strictly at the lowest level of computation:
  - Universal algebraic top/bottom symbols.
  - Python runtime MRO reflection via `inspect.getmro`.
  - Python standard Abstract Base Classes (ABCs): numbers, collections, paths, callables.
  - Category-theoretic structural morphism classification (ingress, egress, lens setter, bridge).

Zero high-level domain vocabulary (no 'table', 'tensor', 'model', 'regressor', 'classifier').
All domain-specific vocabulary and semantics live in domain trees (trees/*.json).
"""

from __future__ import annotations
import builtins
import collections.abc
import inspect
import numbers
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Type


_RUNTIME_ABC_ROOTS: Tuple[Tuple[Type[Any], str], ...] = (
    (bool, "logical"),
    (numbers.Number, "numeric"),
    (os.PathLike, "path"),
    (collections.abc.Mapping, "mapping"),
    (collections.abc.Sequence, "sequence"),
    (collections.abc.Set, "set"),
    (collections.abc.Collection, "collection"),
    (collections.abc.Callable, "callable"),
    (BaseException, "exception"),
)


def reflect_class_carrier(cls: Type[Any]) -> Optional[str]:
    """
    Infers the foundational runtime carrier of a live Python class strictly via
    Python language-level ABCs and built-in type primitives (lowest-level representation).
    """
    if not inspect.isclass(cls):
        return None
    try:
        if issubclass(cls, bool):
            return "logical"
        if issubclass(cls, numbers.Number):
            return "numeric"
        if issubclass(cls, os.PathLike):
            return "path"
        if issubclass(cls, (str,)):
            return "text"
        if issubclass(cls, (bytes, bytearray, memoryview)):
            return "bytes"
        if issubclass(cls, collections.abc.Mapping):
            return "mapping"
        if issubclass(cls, collections.abc.Sequence):
            return "sequence"
        if issubclass(cls, collections.abc.Set):
            return "set"
        if issubclass(cls, collections.abc.Callable):
            return "callable"
        if issubclass(cls, collections.abc.Collection):
            return "collection"
    except TypeError:
        pass
    return None


def reflect_mro_edges(cls: Type[Any], registry: Any) -> None:
    """
    Walks `inspect.getmro(cls)` for any live class and registers its genuine inheritance
    edges and lowest-level runtime ABCs into `TypeRegistry`.
    """
    if not inspect.isclass(cls):
        return
    cls_name = cls.__name__.lower()
    try:
        mro = inspect.getmro(cls)
    except Exception:
        return

    for base in mro[1:]:
        if base is object:
            registry.register_type(cls_name, "object")
            continue
        base_name = base.__name__.lower()
        registry.register_type(cls_name, base_name)

    for abc_cls, proto_name in _RUNTIME_ABC_ROOTS:
        try:
            if issubclass(cls, abc_cls) and not (
                abc_cls is collections.abc.Sequence and issubclass(cls, (str, bytes, bytearray, memoryview))
            ):
                registry.register_type(cls_name, proto_name)
        except TypeError:
            continue

    carrier = reflect_class_carrier(cls)
    if carrier:
        registry.register_type(cls_name, carrier)
        if carrier in ("numeric", "logical", "text"):
            registry.register_type(carrier, "scalar")
            registry.register_type("scalar", "object")
        else:
            registry.register_type(carrier, "object")


def bootstrap_runtime_poset(registry: Any) -> None:
    """
    Bootstraps `TypeRegistry` strictly from host Python runtime introspection
    (`builtins`, `collections.abc`, `numbers`, `os.PathLike`, and loaded modules).
    Operates purely at the lowest level of types: zero domain vocabulary, zero hardcoded strings.
    """
    # 1. Universal top and algebraic symbols
    for sym in ("any", "object", "*", "top", "unknown", "⊤"):
        registry.register_top(sym)
        registry.register_top_state(sym)
    registry.register_type_vars(["T", "U", "V", "R", "K", "State"])
    for prod_c in ("tuple", "product", "pair"):
        registry.register_product_constructor(prod_c)

    # 2. Lowest-level runtime ABC roots
    for _, proto_name in _RUNTIME_ABC_ROOTS:
        registry.register_type(proto_name, "object")
    for abc_cls_a, proto_a in _RUNTIME_ABC_ROOTS:
        for abc_cls_b, proto_b in _RUNTIME_ABC_ROOTS:
            if abc_cls_a is not abc_cls_b and issubclass(abc_cls_a, abc_cls_b):
                registry.register_type(proto_a, proto_b)

    # 3. Introspect all Python built-in classes via live MRO
    for obj in builtins.__dict__.values():
        if inspect.isclass(obj):
            reflect_mro_edges(obj, registry)

    # 4. Introspect already-imported root packages in sys.modules
    for mod_name, mod in list(sys.modules.items()):
        if not mod or "." in mod_name or mod_name.startswith("_"):
            continue
        try:
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name, None)
                if inspect.isclass(obj):
                    reflect_mro_edges(obj, registry)
        except Exception:
            continue


def infer_abstract_carrier(type_name: str, registry: Any = None) -> Optional[str]:
    """
    Infers abstract categorical carrier for any type name strictly via
    registered abstract carriers in the lattice (populated by domain trees).
    """
    tn = str(type_name).strip().lower()
    if not tn or tn in ("any", "object", "*", "top", "unknown", "none"):
        return None
    if registry:
        for abs_c in registry.get_abstract_carriers():
            if tn == abs_c or registry.is_subtype(tn, abs_c):
                return abs_c
    return None


def classify_cell_structural_role(cell: Any, registry: Any = None) -> str:
    """
    Classifies a cell's structural role from first principles via its signature
    morphism f: tau_in -> tau_out:
      - Ingress (source): Consumes an external locator or has no inputs, produces in-memory carrier.
      - Egress (sink): Consumes an in-memory carrier and produces None/void or terminal state,
        or has no LHS variable assignment in its template.
      - Lens Setter (setter): Morphism mu: S x K x V -> S that takes composite carrier S,
        key/index port K, and value port V, and outputs S.
      - Coercion / Projection Bridge (bridge): Pure morphism (|required_inputs| <= 2, no side effects)
        projecting a component from composite carrier or converting representation between carriers.
      - Parametric Morphism: Morphism carrying internal parameters or declared model/estimator role.
    """
    if registry is None:
        try:
            from .lattice import TypeRegistry
            registry = TypeRegistry.get_instance()
        except Exception:
            try:
                from lattice import TypeRegistry
                registry = TypeRegistry.get_instance()
            except Exception:
                registry = None

    inputs = getattr(cell, "inputs", {}) or {}
    outputs = getattr(cell, "outputs", {}) or {}
    template = getattr(cell, "code_template", "") or ""
    effects = getattr(cell, "effects", []) or []

    key_roles = set(registry.get_verification_semantics("key_roles")) if registry else set()
    key_roles |= {"key", "column_key", "field_key", "attr_key", "index_key", "coordinate"}
    val_roles = set(registry.get_verification_semantics("assigned_value_roles")) if registry else set()
    val_roles |= {"assigned_value", "column_value", "field_value", "value"}

    has_carrier_in = False
    has_key_in = False
    has_val_in = False
    carrier_in_type = None

    for p in inputs.values():
        bind = getattr(p, "binds", None)
        role = getattr(p, "port_role", None) or getattr(p, "derived_role", "")
        if bind in ("data_carrier", "receiver", "carrier", "primary_operand", "state") or role in ("receiver", "carrier", "primary_operand", "state"):
            has_carrier_in = True
            carrier_in_type = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        if bind in key_roles or role in key_roles:
            has_key_in = True
        if bind in val_roles or role in val_roles:
            has_val_in = True

    prim_out = getattr(cell, "primary_output", None)
    out_tn = str(getattr(getattr(prim_out, "signature", prim_out), "type_name", "") or "").lower() if prim_out else ""
    out_state = str(getattr(getattr(prim_out, "signature", prim_out), "state", "") or "").lower() if prim_out else ""

    # 1. Lens Setter: mu: S x K x V -> S
    if has_carrier_in and has_key_in and has_val_in:
        if carrier_in_type and out_tn and registry:
            if registry.is_subtype(out_tn, carrier_in_type) or registry.is_subtype(carrier_in_type, out_tn):
                return "setter"
        elif has_carrier_in and has_key_in and has_val_in:
            return "setter"

    # 2. Ingress (source) / Initial Morphism: consumes external locator or has no inputs, produces in-memory carrier
    if not inputs and prim_out:
        return "source"

    path_tokens = registry.get_path_port_tokens() if registry else set()
    source_roles = set(registry.get_verification_semantics("source_roles")) if registry else {"source", "ingress", "source_data"}
    in_is_locator = False
    for p in inputs.values():
        p_tn = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        p_st = str(getattr(getattr(p, "signature", p), "state", "") or "").lower()
        p_role = getattr(p, "port_role", None) or getattr(p, "derived_role", "")
        if p_st in ("source_identifier", "input_path", "source_path") or p_role in source_roles or p_tn in path_tokens:
            in_is_locator = True
            break
        if registry and any(registry.is_subtype(p_tn, pt) for pt in path_tokens):
            in_is_locator = True
            break

    if in_is_locator and prim_out:
        if out_tn not in path_tokens and (not registry or not any(registry.is_subtype(out_tn, pt) for pt in path_tokens)):
            return "source"

    # 3. Egress (sink) / Terminal Morphism
    mat_states = registry.get_materialization_states() if registry else set()
    term_states = registry.get_terminal_states() if registry else set()
    has_lhs = ("=" in template and any(ch not in ("=", "!", "<", ">") for ch in template[:template.index("=")].strip())) if template else False

    if not outputs or out_tn in ("none", "void", "unit", ""):
        return "sink"
    if out_state in mat_states or out_state in term_states:
        return "sink"
    if template and not has_lhs:
        return "sink"

    # 4. Coercion / Projection Bridge
    req_inputs = [p for p in inputs.values() if getattr(p, "required", True)]
    is_pure = not bool(effects) or all(str(e).lower().startswith("preserves") for e in effects)
    if is_pure and len(req_inputs) <= 2:
        node_role = str(getattr(cell, "node_role", "") or "").lower()
        bridge_roles = set(registry.get_verification_semantics("bridge_roles")) if registry else set()
        if node_role in bridge_roles or getattr(cell, "projection", None):
            return "bridge"

    # 5. Parametric / Stateful Morphism (Theta x X -> Y)
    declared_role = str(getattr(cell, "node_role", "") or "").lower()
    model_roles = set(registry.get_verification_semantics("model_roles")) if registry else set()
    if declared_role in model_roles:
        return declared_role

    return declared_role or "function"


__all__ = [
    "reflect_class_carrier",
    "reflect_mro_edges",
    "bootstrap_runtime_poset",
    "infer_abstract_carrier",
    "classify_cell_structural_role",
]
