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
import contextlib
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
    reflected = getattr(registry, "_reflected_python_classes", None)
    if isinstance(reflected, set):
        cid = id(cls)
        if cid in reflected:
            return
        reflected.add(cid)
    cls_name = cls.__name__.lower()
    try:
        mro = inspect.getmro(cls)
    except Exception:
        return

    for base in mro[1:]:
        if base is object:
            registry.register_type(cls_name, "object")
            continue
        if cls is bool and base is int:
            continue
        base_name = base.__name__.lower()
        registry.register_type(cls_name, base_name)

    for abc_cls, proto_name in _RUNTIME_ABC_ROOTS:
        try:
            if cls is bool and abc_cls is numbers.Number:
                continue
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


def bootstrap_python_runtime_poset(registry: Any) -> None:
    """
    Bootstraps Python host runtime type edges (`builtins`, `collections.abc`,
    `numbers`, `os.PathLike`, and loaded modules) strictly when active runtime is Python.
    """
    batch_ctx = registry.batch_mutations() if hasattr(registry, "batch_mutations") else contextlib.nullcontext()
    with batch_ctx:
        registry.register_type("object", None)
        registry.register_product_constructor(tuple.__name__)

        # 2. Lowest-level runtime ABC roots
        for _, proto_name in _RUNTIME_ABC_ROOTS:
            registry.register_type(proto_name, "object")
        for abc_cls_a, proto_a in _RUNTIME_ABC_ROOTS:
            for abc_cls_b, proto_b in _RUNTIME_ABC_ROOTS:
                if abc_cls_a is bool and abc_cls_b is numbers.Number:
                    continue
                if abc_cls_a is not abc_cls_b and issubclass(abc_cls_a, abc_cls_b):
                    registry.register_type(proto_a, proto_b)

        # 3. Introspect all Python built-in classes via live MRO
        for obj in builtins.__dict__.values():
            if inspect.isclass(obj):
                reflect_mro_edges(obj, registry)

        # 4. Introspect already-loaded classes in sys.modules without invoking lazy __getattr__/__dir__
        for mod_name, mod in list(sys.modules.items()):
            if not mod or "." in mod_name or mod_name.startswith("_"):
                continue
            try:
                ns = object.__getattribute__(mod, "__dict__")
            except AttributeError:
                continue
            if not isinstance(ns, dict):
                continue
            for attr_name, obj in list(ns.items()):
                if not isinstance(attr_name, str) or attr_name.startswith("_"):
                    continue
                if inspect.isclass(obj):
                    reflect_mro_edges(obj, registry)


def bootstrap_runtime_poset(registry: Any) -> None:
    """
    Bootstraps language-neutral algebraic top symbols and universal type vars.
    Python MRO reflection runs only if the active LanguageSpec runtime is ``python``.
    """
    try:
        from .language_backend import UNIVERSAL_TOP_SYMBOLS
    except (ImportError, ValueError):
        from language_backend import UNIVERSAL_TOP_SYMBOLS

    for sym in UNIVERSAL_TOP_SYMBOLS:
        registry.register_top(sym)
        registry.register_top_state(sym)

    spec = registry.get_language_spec() if hasattr(registry, "get_language_spec") else None
    if spec is not None and str(getattr(spec, "runtime", "") or "").strip().lower() == "python":
        bootstrap_python_runtime_poset(registry)


def infer_abstract_carrier(type_name: str, registry: Any = None) -> Optional[str]:
    """
    Infers abstract categorical carrier for any type name strictly via
    registered abstract carriers in the lattice (populated by domain trees).
    """
    tn = str(type_name).strip().lower()
    if not tn or tn == "object":
        return None
    if registry is not None:
        if registry.is_declared_top(tn) or registry.is_void_type(tn):
            return None
        for abs_c in registry.get_abstract_carriers():
            if tn == abs_c or registry.is_subtype(tn, abs_c):
                return abs_c
    return None


_FALLBACK_STRUCTURAL_ROLE_CACHE: Dict[Tuple[Any, ...], str] = {}


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
    reg_gen = int(getattr(registry, "_generation", 0)) if registry is not None else 0
    cid = getattr(cell, "cell_id", None) or id(cell)
    cache_key = (
        id(registry),
        reg_gen,
        cid,
        id(inputs),
        len(inputs),
        id(outputs),
        len(outputs),
        str(getattr(cell, "optic_role", "") or ""),
        str(getattr(cell, "node_role", "") or ""),
    )
    # Note: MicroCell uses __slots__ = () so per-object setattr silently fails;
    # store the structural-role cache on the registry (or module fallback when None).
    role_cache = None
    if registry is not None:
        role_cache = getattr(registry, "_cell_structural_role_cache", None)
        if role_cache is None:
            role_cache = {}
            try:
                setattr(registry, "_cell_structural_role_cache", role_cache)
            except Exception:
                role_cache = None
        else:
            cached_gen = getattr(registry, "_cell_structural_role_cache_gen", -1)
            if cached_gen != reg_gen:
                role_cache.clear()
                try:
                    setattr(registry, "_cell_structural_role_cache_gen", reg_gen)
                except Exception:
                    pass
    if role_cache is None:
        role_cache = _FALLBACK_STRUCTURAL_ROLE_CACHE
    hit = role_cache.get(cache_key)
    if hit is not None:
        return hit

    res = _compute_cell_structural_role(cell, registry, inputs, outputs)
    if len(role_cache) >= 8192:
        role_cache.clear()
    role_cache[cache_key] = res
    return res


def _compute_cell_structural_role(cell: Any, registry: Any, inputs: Any, outputs: Any) -> str:
    template = getattr(cell, "code_template", "") or ""
    effects = getattr(cell, "effects", []) or []

    explicit_optic = str(getattr(cell, "optic_role", "") or "").strip().lower()
    optic_roles = registry.get_roles_for_category("optic") if registry else set()
    if explicit_optic and (not optic_roles or explicit_optic in optic_roles):
        return explicit_optic

    key_roles = registry.get_roles_for_category("key") if registry else set()
    val_roles = registry.get_roles_for_category("assigned_value") if registry else set()
    carrier_binds = (registry.get_roles_for_category("receiver") | registry.get_roles_for_category("data")) if registry else set()
    receiver_roles = registry.get_roles_for_category("receiver") if registry else set()

    has_carrier_in = False
    has_key_in = False
    has_val_in = False
    carrier_in_type = None

    for p in inputs.values():
        bind = str(getattr(p, "binds", None) or "").lower()
        role = str(getattr(p, "port_role", None) or getattr(p, "derived_role", "") or "").lower()
        if (bind and bind in carrier_binds) or (role and role in receiver_roles) or (role and role in carrier_binds):
            has_carrier_in = True
            carrier_in_type = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        if (bind and bind in key_roles) or (role and role in key_roles):
            has_key_in = True
        if (bind and bind in val_roles) or (role and role in val_roles):
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

    source_roles = (registry.get_ingress_roles_set() | registry.get_roles_for_category("source") | {"source_data"}) if registry else {"source_data"}
    source_states = registry.get_states_by_property("is_source_identifier", True) if registry else set()
    try:
        from .lattice import is_path_port
    except (ImportError, ValueError):
        from lattice import is_path_port
    in_is_locator = False
    for p in inputs.values():
        p_sig = getattr(p, "signature", p)
        p_tn = str(getattr(p_sig, "type_name", "") or "").lower()
        p_st = str(getattr(p_sig, "state", "") or "").lower()
        p_role = str(getattr(p, "port_role", None) or getattr(p, "derived_role", "") or "").lower()
        if (p_st and p_st in source_states) or (p_role and p_role in source_roles) or is_path_port(p, registry):
            in_is_locator = True
            break

    if in_is_locator and prim_out:
        if not is_path_port(prim_out, registry):
            return "source"

    # 3. Egress (sink) / Terminal Morphism
    mat_states = (
        registry.get_states_by_property("is_materialized", True)
        | registry.get_states_by_property("is_terminal", True)
    ) if registry else set()
    term_states = registry.get_terminal_states() if registry else set()
    void_types = registry.get_void_types_set() if registry else {""}
    try:
        from .language_backend import get_active_language_backend
    except (ImportError, ValueError):
        from language_backend import get_active_language_backend
    has_lhs = bool(template and get_active_language_backend(registry).extract_assignment_targets(template))

    if not outputs or out_tn in void_types:
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
        bridge_roles = (registry.get_bridge_roles_set() | registry.get_roles_for_category("bridge")) if registry else set()
        if node_role in bridge_roles or getattr(cell, "projection", None):
            return "bridge"

    # 5. Parametric / Stateful Morphism (Theta x X -> Y)
    declared_role = str(getattr(cell, "node_role", "") or "").lower()
    model_roles = (registry.get_roles_for_category("model") | registry.get_roles_for_category("estimator")) if registry else set()
    if declared_role in model_roles:
        return declared_role

    return declared_role or "function"


__all__ = [
    "reflect_class_carrier",
    "reflect_mro_edges",
    "bootstrap_runtime_poset",
    "bootstrap_python_runtime_poset",
    "infer_abstract_carrier",
    "classify_cell_structural_role",
]
