"""
src/unification.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Formal Type-Monadic Unification Gate and Deterministic Composition Synthesizer.

Conforms strictly to Section 3.2 of the STRATA paper:
  M_T(A) = { (a, sigma) : a in A, sigma a type substitution } U { bottom }
  bind(m, k) = k(a) with sigma_new if sigma_new = unify(tau_out of m, tau_in of k) succeeds;
               otherwise bottom.
"""

from __future__ import annotations
import os
import ast
import collections
import copy
import sys
import builtins
import importlib.util
import json
import math
import keyword
import threading
import warnings
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import collections.abc
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Generic, TypeVar

from log_config import get_logger

try:
    from utils import (
        tokenize_alphanumeric,
        extract_template_placeholders,
        safe_substitute_template,
        unique_var as _unique_var,
    )
    from errors import (
        UnificationError,
        PlaceholderResolutionError,
        TemplateValidationError,
        DataflowExecutionError,
        PostconditionVerificationError,
    )
except ImportError:
    from .utils import (
        tokenize_alphanumeric,
        extract_template_placeholders,
        safe_substitute_template,
        unique_var as _unique_var,
    )
    from .errors import (
        UnificationError,
        PlaceholderResolutionError,
        TemplateValidationError,
        DataflowExecutionError,
        PostconditionVerificationError,
    )

try:
    from .lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        LatticeType, _CONSTRUCTOR_VARIANCE,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port, _normalize_qualifiers,
        compute_port_residual, apply_polymorphic_propagation,
    )
    from .tokenizer import CellTokenizer, normalize_token
    from .language_backend import get_active_language_backend
    from .schema import (
        EXTENT_EFFECT_KINDS,
        VALUE_LEAF_KINDS,
        cell_has_effect_kind,
        is_bridge_cell as schema_is_bridge_cell,
        is_unresolved_value,
    )
except (ImportError, ValueError):
    from lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        LatticeType, _CONSTRUCTOR_VARIANCE,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port, _normalize_qualifiers,
        compute_port_residual, apply_polymorphic_propagation,
    )
    from tokenizer import CellTokenizer, normalize_token
    from language_backend import get_active_language_backend
    from schema import (
        EXTENT_EFFECT_KINDS,
        VALUE_LEAF_KINDS,
        cell_has_effect_kind,
        is_bridge_cell as schema_is_bridge_cell,
        is_unresolved_value,
    )

try:
    from .diagnostics import DiagnosticTelemetry, record_diagnostic_exception
except (ImportError, ValueError):
    from diagnostics import DiagnosticTelemetry, record_diagnostic_exception

try:
    from .type_predicates import (
        format_literal_value as _tp_format_literal_value,
        is_sequence_or_tuple_type as _tp_is_sequence_or_tuple_type,
        is_true_scalar_port as _tp_is_true_scalar_port,
        validate_slot_value as _tp_validate_slot_value,
    )
    from .typed_proof import TypedPipelineProof, build_typed_pipeline_proof
    from .slot_binder import (
        PipelineSlotBinder,
        consume_matching_literals as _sb_consume_matching_literals,
        is_slot_grounded_in_clause as _sb_is_slot_grounded_in_clause,
    )
except (ImportError, ValueError):
    from type_predicates import (
        format_literal_value as _tp_format_literal_value,
        is_sequence_or_tuple_type as _tp_is_sequence_or_tuple_type,
        is_true_scalar_port as _tp_is_true_scalar_port,
        validate_slot_value as _tp_validate_slot_value,
    )
    from typed_proof import TypedPipelineProof, build_typed_pipeline_proof
    from slot_binder import (
        PipelineSlotBinder,
        consume_matching_literals as _sb_consume_matching_literals,
        is_slot_grounded_in_clause as _sb_is_slot_grounded_in_clause,
    )

logger = get_logger('unification')

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class TypedBindingRecord:
    """
    Structured typed binding record emitted for every cell port during unification.
    Conforms to Round 5 specification (R5-1).
    """
    port: str
    value_expr: str
    kind: str  # "variable", "projection", "literal", "default"
    root_var: Optional[str] = None
    projection: Optional[Dict[str, Any]] = None
    resulting_signature: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "value_expr": self.value_expr,
            "kind": self.kind,
            "root_var": self.root_var,
            "projection": self.projection,
            "resulting_signature": self.resulting_signature,
        }


def _is_unbound(value: Any) -> bool:
    if value is None or is_unresolved_value(value):
        return True
    try:
        s = str(value).strip()
    except Exception:
        return False
    return get_active_language_backend().is_null_literal(s)


@functools.lru_cache(maxsize=2048)
def _cached_ast_parse_ro(source: str, mode: str = "exec") -> Tuple[Optional[ast.AST], Optional[str]]:
    """Read-only LRU-cached ast.parse returning (tree, error_msg). Never mutate returned AST nodes."""
    try:
        return ast.parse(source, mode=mode), None
    except Exception as exc:
        return None, str(exc)


@functools.lru_cache(maxsize=1024)
def _parse_expr_ast(expr_str: str) -> Optional[ast.AST]:
    tree, _ = _cached_ast_parse_ro(expr_str, "eval")
    return tree.body if tree is not None else None


_cached_col_proj_sig: Optional[Any] = None
_cached_cols_proj_sig: Optional[Any] = None
_cached_attr_proj_sigs: Dict[str, Any] = {}
_PROJECTION_CACHE_LOCK = threading.RLock()

_ENUM_EMBEDDING_CACHE: Dict[Tuple[str, str], Tuple[Tuple[str, ...], Tuple[str, ...], Any]] = {}
_TEXT_EMBEDDING_CACHE: Dict[Tuple[str, str, str], Any] = {}
_ENUM_EMBEDDING_LOCK = threading.RLock()


def _embedder_cache_identity(mm: Any) -> str:
    """Stable key identifying the active embedding model and vector space."""
    prof_name = getattr(mm, "current_profile_name", "") or ""
    act_prof = getattr(mm, "active_profile", None)
    emb_name = getattr(act_prof, "embedder_name", "") or ""
    return f"{prof_name}:{emb_name}:{id(act_prof)}"


def _is_product_sig(sig_obj: Any, registry: Optional[Any] = None) -> bool:
    """Checks whether a type signature is a product/tuple/record constructor."""
    if sig_obj is None:
        return False
    if registry is None:
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            from lattice import TypeRegistry
        registry = TypeRegistry.get_instance()
    t_str = str(getattr(getattr(sig_obj, "signature", None), "type_name", "") or getattr(sig_obj, "type_name", sig_obj)).strip()
    ctor = t_str.split("[", 1)[0].strip().lower()
    return registry.is_product_constructor(ctor)

_is_product = _is_product_sig


def resolve_typed_binding_record(
    cell: Any,
    port_name: str,
    bound_val: Any,
    port_sig: Optional[Any] = None,
    var_signatures: Optional[Dict[str, Any]] = None,
    expected_sig: Optional[Any] = None,
) -> TypedBindingRecord:
    """
    Computes a structured TypedBindingRecord for a bound port directly from the lattice,
    without inventing arbitrary synthetic state names or guessing types from strings.
    """
    var_sigs = var_signatures or {}
    if expected_sig is None:
        expected_sig = getattr(port_sig, "signature", port_sig) if port_sig else None

    if bound_val is None or _is_unbound(bound_val) or is_unresolved_value(bound_val):
        return TypedBindingRecord(
            port=port_name,
            value_expr=str(bound_val) if bound_val is not None else "",
            kind="default",
            root_var=None,
            projection=None,
            resulting_signature=expected_sig,
        )

    if not isinstance(bound_val, str):
        return TypedBindingRecord(
            port=port_name,
            value_expr=repr(bound_val),
            kind="literal",
            root_var=None,
            projection=None,
            resulting_signature=expected_sig,
        )

    val_str = bound_val.strip()
    if not val_str:
        return TypedBindingRecord(
            port=port_name,
            value_expr="",
            kind="default",
            root_var=None,
            projection=None,
            resulting_signature=expected_sig,
        )

    if cell is not None and _classify_template_identifier_port(cell, port_name)[0]:
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="literal",
            root_var=None,
            projection=None,
            resulting_signature=expected_sig,
        )

    has_known_scope = var_signatures is not None
    # 1. Plain pipeline variable
    if val_str in var_sigs:
        sig = var_sigs[val_str]
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="variable",
            root_var=val_str,
            projection=None,
            resulting_signature=getattr(sig, "signature", sig),
        )
    elif val_str.isidentifier() and not keyword.iskeyword(val_str):
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="variable",
            root_var=val_str,
            projection=None,
            resulting_signature=None if has_known_scope else expected_sig,
        )

    # 2. Parse expression via AST
    parsed = _parse_expr_ast(val_str)

    if isinstance(parsed, ast.Name):
        v = parsed.id
        sig = var_sigs.get(v)
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="variable",
            root_var=v,
            projection=None,
            resulting_signature=(
                getattr(sig, "signature", sig) if sig is not None else (None if has_known_scope else expected_sig)
            ),
        )

    subscript_node: Optional[ast.Subscript] = None
    if isinstance(parsed, ast.Subscript):
        subscript_node = parsed
    elif isinstance(parsed, ast.Call) and isinstance(parsed.func, ast.Attribute) and isinstance(parsed.func.value, ast.Subscript):
        subscript_node = parsed.func.value
    elif isinstance(parsed, ast.Attribute) and isinstance(parsed.value, ast.Subscript):
        subscript_node = parsed.value

    if subscript_node is not None:
        root_v = subscript_node.value.id if isinstance(subscript_node.value, ast.Name) else None
        base_sig = var_sigs.get(root_v) if root_v else None
        sl = subscript_node.slice
        is_wrapped = (subscript_node is not parsed)

        def _lookup_projection(category: str, cache_name: str) -> Tuple[Optional[str], Any]:
            cached_val = globals().get(cache_name)
            kind_val = None
            try:
                try:
                    from .lattice import LatticeOrchestrator, TypeRegistry
                except (ImportError, ValueError):
                    from lattice import LatticeOrchestrator, TypeRegistry
                orch = LatticeOrchestrator.get_active_instance()
                if orch:
                    proj_kinds = TypeRegistry.get_instance().get_roles_for_category(category)
                    for c in orch.cells:
                        c_proj = getattr(c, "projection", None)
                        if c_proj in proj_kinds:
                            kind_val = str(c_proj)
                            if cached_val is None:
                                p_out = getattr(c, "primary_output", None) or next(iter(c.outputs.values()), None)
                                if p_out:
                                    cached_val = getattr(p_out, "signature", p_out)
                                    with _PROJECTION_CACHE_LOCK:
                                        globals()[cache_name] = cached_val
                            break
            except Exception as _exc:
                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
            return kind_val, cached_val

        root_unknown = has_known_scope and (root_v is None or root_v not in var_sigs)

        # (A) Single-key selection: var['X']
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            col_kind_val, col_sig = _lookup_projection("single_projection", "_cached_col_proj_sig")
            if root_unknown:
                res_sig_a = None
            else:
                if col_sig is None:
                    col_sig = expected_sig or (
                        getattr(base_sig, "signature", base_sig) if base_sig else None
                    ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
                res_sig_a = (expected_sig or col_sig) if is_wrapped else col_sig
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": col_kind_val or "subscript", "key": sl.value},
                resulting_signature=res_sig_a,
            )

        # (B) Multi-key selection: var[['X', 'Y']]
        elif isinstance(sl, (ast.List, ast.Tuple)):
            cols_kind_val, cols_sig = _lookup_projection("multi_projection", "_cached_cols_proj_sig")
            if root_unknown:
                res_sig_b = None
            else:
                if cols_sig is None:
                    cols_sig = expected_sig or (
                        getattr(base_sig, "signature", base_sig) if base_sig else None
                    ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
                res_sig_b = (expected_sig or cols_sig) if is_wrapped else cols_sig
            keys = [elt.value for elt in sl.elts if isinstance(elt, ast.Constant)]
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": cols_kind_val or "subscript", "key": keys},
                resulting_signature=res_sig_b,
            )

        # (C) Index projection: var[0]
        elif isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            idx_val = sl.value
            if root_unknown:
                idx_sig = None
            else:
                idx_sig = None
                if base_sig:
                    raw = str(getattr(getattr(base_sig, "signature", None), "type_name", "") or getattr(base_sig, "type_name", ""))
                    if "[" in raw and raw.endswith("]"):
                        inner = raw.split("[", 1)[1][:-1]
                        members = [m.strip() for m in inner.split(",")]
                        if 0 <= idx_val < len(members):
                            idx_sig = AlgebraicSignature(type_name=members[idx_val], state="any", abstract_type="any")
                if idx_sig is None:
                    idx_sig = AlgebraicSignature(type_name="any", state="any", abstract_type="any")
                idx_sig = (expected_sig or idx_sig) if is_wrapped else idx_sig
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": "index", "key": idx_val},
                resulting_signature=idx_sig,
            )

    # (D) Method calls on variable (e.g. var.to_numpy()) or module functions (e.g. np.asarray(var))
    if isinstance(parsed, ast.Call) and isinstance(parsed.func, ast.Attribute):
        root_v = None
        for node in ast.walk(parsed.func.value):
            if isinstance(node, ast.Name) and node.id in var_sigs:
                root_v = node.id
                break
        if root_v is None and isinstance(parsed.func.value, ast.Name):
            root_v = parsed.func.value.id
        attr_name = parsed.func.attr

        # Check if root_v is a module or library alias rather than a pipeline variable
        try:
            from .lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            known_aliases = set(reg.get_all_aliases().keys())
            known_modules = set(reg.get_all_aliases().values())
        except Exception:
            try:
                from lattice import TypeRegistry
                reg = TypeRegistry.get_instance()
                known_aliases = set(reg.get_all_aliases().keys())
                known_modules = set(reg.get_all_aliases().values())
            except Exception:
                reg = None
                known_aliases = set()
                known_modules = set()

        import sys
        import importlib.util
        def _is_module_or_alias(name: Optional[str]) -> bool:
            if not name or not isinstance(name, str) or not name.isidentifier():
                return False
            if name in known_aliases or name in known_modules or name in sys.modules:
                return True
            try:
                return importlib.util.find_spec(name) is not None
            except (ValueError, AttributeError):
                return False

        is_module_call = _is_module_or_alias(root_v) and root_v not in var_sigs

        if is_module_call:
            carrier_var = None
            if parsed.args:
                arg0 = parsed.args[0]
                if isinstance(arg0, ast.Name):
                    carrier_var = arg0.id
                elif isinstance(arg0, ast.Subscript) and isinstance(arg0.value, ast.Name):
                    carrier_var = arg0.value.id
                else:
                    for node in ast.walk(arg0):
                        if isinstance(node, ast.Name) and node.id in var_sigs:
                            carrier_var = node.id
                            break
            base_sig = var_sigs.get(carrier_var) if carrier_var else None
            if has_known_scope and carrier_var is not None and carrier_var not in var_sigs:
                mod_sig = None
            else:
                mod_sig = expected_sig or (
                    getattr(base_sig, "signature", base_sig) if base_sig else None
                ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection" if carrier_var else "literal",
                root_var=carrier_var,
                projection={"kind": "module_call", "module": root_v, "func": attr_name},
                resulting_signature=mod_sig,
            )

        base_sig = var_sigs.get(root_v) if root_v else None
        with _PROJECTION_CACHE_LOCK:
            method_sig = _cached_attr_proj_sigs.get(attr_name)
        if method_sig is None:
            try:
                try:
                    from .lattice import LatticeOrchestrator
                except (ImportError, ValueError):
                    from lattice import LatticeOrchestrator
                orch = LatticeOrchestrator.get_active_instance()
                if orch:
                    for c in orch.cells:
                        if getattr(c, "projection", None) == attr_name or getattr(c, "method_name", None) == attr_name:
                            p_out = getattr(c, "primary_output", None) or next(iter(c.outputs.values()), None)
                            if p_out:
                                method_sig = getattr(p_out, "signature", p_out)
                                with _PROJECTION_CACHE_LOCK:
                                    _cached_attr_proj_sigs[attr_name] = method_sig
                                break
            except Exception as _exc:
                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

        if method_sig is None and has_known_scope and (root_v is None or root_v not in var_sigs):
            res_meth_sig = None
        else:
            if method_sig is None and expected_sig:
                method_sig = expected_sig
            if method_sig is None:
                method_sig = (
                    getattr(base_sig, "signature", base_sig) if base_sig else None
                ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
            res_meth_sig = method_sig
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="projection",
            root_var=root_v,
            projection={"kind": "method", "key": attr_name},
            resulting_signature=res_meth_sig,
        )

    # 3. Literal
    return TypedBindingRecord(
        port=port_name,
        value_expr=val_str,
        kind="literal",
        root_var=None,
        projection=None,
        resulting_signature=expected_sig,
    )


def _resolve_cell_port(cell: Any, port_ref: Optional[str], prefer: str = "inputs") -> Optional[Any]:
    """Resolves a declarative port reference (canonical name or alias like 'port_0') on a Cell."""
    if cell is None or not port_ref:
        return None
    ref_s = str(port_ref).strip()
    if not ref_s:
        return None
    inputs = getattr(cell, "inputs", None) or {}
    outputs = getattr(cell, "outputs", None) or {}
    order = (outputs, inputs) if prefer == "outputs" else (inputs, outputs)
    for mapping in order:
        if not mapping:
            continue
        try:
            found = mapping.get(ref_s)
            if found is not None:
                return found
        except Exception:
            pass
        if isinstance(mapping, dict):
            for k, p in mapping.items():
                if k == ref_s or getattr(p, "name", None) == ref_s:
                    return p
            if ref_s.startswith("port_") and ref_s[5:].isdigit():
                idx = int(ref_s[5:])
                vals = list(mapping.values())
                if 0 <= idx < len(vals):
                    return vals[idx]
    if prefer == "outputs" and ref_s in ("output", "output_var"):
        prim = getattr(cell, "primary_output", None)
        if prim is not None:
            return prim
    return None


def _port_matches_declared_ref(cell: Any, attr_name: str, port_sig: Any) -> bool:
    """Checks whether `port_sig` corresponds to a declarative port reference (e.g. key_port) on `cell`,
    matching by either object identity or canonical port name after generic substitution."""
    if cell is None or port_sig is None:
        return False
    ref = getattr(cell, attr_name, None)
    if not ref:
        return False
    resolved = _resolve_cell_port(cell, ref, prefer="inputs")
    if resolved is None:
        return False
    p_name = getattr(port_sig, "name", None)
    return resolved is port_sig or (p_name is not None and getattr(resolved, "name", None) == p_name)


def _is_key_or_complement_port(cell: Any, port_sig: Any, reg: Optional[Any] = None) -> bool:
    """True when `port_sig` is a declared key/projection/complement/upstream-key port (never satisfied by empty default)."""
    if port_sig is None:
        return False
    if reg is None:
        reg = TypeRegistry.get_instance()
    key_binds = reg.get_roles_for_category("key")
    proj_roles = (
        set(reg.get_projection_tokens() or ())
        | reg.get_roles_for_category("projection")
        | reg.get_roles_for_category("key")
    )
    p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "role", "") or "").strip().lower()
    p_binds = str(getattr(port_sig, "binds", "") or "").strip().lower()
    return bool(
        p_binds in key_binds
        or p_role in proj_roles
        or _port_matches_declared_ref(cell, "key_port", port_sig)
        or getattr(port_sig, "complement_of_port", None)
        or getattr(port_sig, "key_from_upstream_role", None)
    )


def _var_ancestors(v_name: str, ctx: Any) -> Set[str]:
    """Computes reflexive transitive ancestor set of a variable in ctx.var_parents."""
    visited: Set[str] = set()
    if not v_name or not ctx:
        return visited
    var_parents = getattr(ctx, "var_parents", None) or {}
    stack: List[str] = [v_name]
    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        for par in var_parents.get(curr, ()) or ():
            if isinstance(par, str) and par not in visited:
                stack.append(par)
    return visited


def _extract_keys_from_val(val: Any) -> Set[str]:
    """Extracts symbolic key names from a scalar literal or list/tuple binding."""
    keys: Set[str] = set()
    if val is None or val is UNRESOLVED_PORT or isinstance(val, bool):
        return keys
    if isinstance(val, (int, float)):
        return keys
    if isinstance(val, (list, tuple, set)):
        for item in val:
            keys.update(_extract_keys_from_val(item))
        return keys
    s = str(val).strip()
    be = get_active_language_backend()
    if not s or is_unresolved_value(s) or be.is_null_or_bool_literal(s) or be.is_empty_collection_literal(s):
        return keys
    ok_lit, parsed = be.parse_literal(s)
    if ok_lit and isinstance(parsed, (list, tuple, set)):
        return _extract_keys_from_val(parsed)
    cl = s.strip("'\"").strip()
    if cl and not be.is_null_or_bool_literal(cl):
        try:
            float(cl)
        except ValueError:
            keys.add(cl)
    return keys


def _var_restricted_keys(v_name: str, ctx: Any) -> Set[str]:
    """Returns restricted fiber keys if `v_name` carries an isolated subset of a composite carrier."""
    if not ctx or not v_name:
        return set()
    restricted = getattr(ctx, "var_restricted_origins", None) or {}
    if v_name in restricted and restricted[v_name]:
        return set(restricted[v_name])
    prod_cell = (getattr(ctx, "var_sources", None) or {}).get(v_name)
    if prod_cell is not None and str(getattr(prod_cell, "optic_role", "") or "").lower() == "getter":
        return set((getattr(ctx, "var_origins", None) or {}).get(v_name, ()) or ())
    return set()


def _match_multi_output_inputs(
    out_port_name: Optional[str],
    out_sig: Any,
    cell: Any,
    cell_bindings: Dict[str, Any],
    ctx: Any,
) -> Optional[Set[str]]:
    """Partitions provenance across distinct outputs of a multi-output cell when inputs declare disjoint/complement ports."""
    if not out_port_name or not cell:
        return None
    outputs = getattr(cell, "outputs", None) or {}
    inputs = getattr(cell, "inputs", None) or {}
    if len(outputs) <= 1 or len(inputs) <= 1:
        return None
    has_rel = any(
        getattr(p, "disjoint_from_port", None) or getattr(p, "complement_of_port", None)
        for p in inputs.values()
    )
    if not has_rel:
        return None
    bound_var_ports = [
        (p_n, inputs[p_n])
        for p_n, p_val in cell_bindings.items()
        if p_n in inputs and isinstance(p_val, str) and p_val in getattr(ctx, "variables", {})
    ]
    if len(bound_var_ports) <= 1:
        return None
    def _ident_parts(raw: str) -> Set[str]:
        toks = {t.lower() for t in CellTokenizer.tokenize_identifier(raw)}
        clean = str(raw or "").strip()
        if clean:
            toks.add(clean.lower())
            seg: List[str] = []
            for idx_ch, ch in enumerate(clean):
                if not ch.isalnum():
                    if seg:
                        toks.add("".join(seg).lower())
                        seg = []
                    continue
                if seg:
                    prev_ch = seg[-1]
                    if (
                        (prev_ch.islower() and ch.isupper())
                        or (prev_ch.isupper() and ch.isupper() and idx_ch + 1 < len(clean) and clean[idx_ch + 1].islower())
                    ):
                        toks.add("".join(seg).lower())
                        seg = [ch]
                        continue
                seg.append(ch)
            if seg:
                toks.add("".join(seg).lower())
        return {t for t in toks if t}

    out_toks = _ident_parts(out_port_name)
    out_state = str(getattr(getattr(out_sig, "signature", out_sig), "state", "") or "")
    out_state_low = out_state.strip().lower()
    if out_state_low and out_state_low != "default" and not TypeRegistry.get_instance().is_declared_top(out_state_low):
        out_toks.update(_ident_parts(out_state))
    scored: List[Tuple[int, str]] = []
    for in_name, _in_sig in bound_var_ports:
        in_toks = _ident_parts(in_name)
        ov = len(out_toks & in_toks)
        scored.append((ov, in_name))
    best_ov = max(ov for ov, _ in scored)
    if best_ov <= 0:
        return None
    chosen = {in_name for ov, in_name in scored if ov == best_ov}
    if 0 < len(chosen) < len(bound_var_ports):
        return chosen
    return None


def _port_lead_sym_dim(p_obj: Any) -> Optional[str]:
    """Returns leading symbolic shape dimension (e.g. 'n_samples') declared on a port, if any."""
    for attr in ("shape_contract", "layout"):
        sc_o = getattr(p_obj, attr, None)
        if sc_o is None and hasattr(p_obj, "signature"):
            sc_o = getattr(p_obj.signature, attr, None)
        shp = sc_o.get("shape") if isinstance(sc_o, dict) else getattr(sc_o, "shape", None)
        if isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], str) and shp[0].strip():
            return shp[0].strip().lower()
    return None


def _port_nonlead_sym_dims(p_obj: Any) -> Tuple[str, ...]:
    """Returns non-leading symbolic shape dimensions (e.g. ('n_features',)) declared on a port."""
    for attr in ("shape_contract", "layout"):
        sc_o = getattr(p_obj, attr, None)
        if sc_o is None and hasattr(p_obj, "signature"):
            sc_o = getattr(p_obj.signature, attr, None)
        shp = sc_o.get("shape") if isinstance(sc_o, dict) else getattr(sc_o, "shape", None)
        if isinstance(shp, (list, tuple)) and len(shp) > 1:
            dims = tuple(str(d).strip().lower() for d in shp[1:] if isinstance(d, str) and d.strip())
            if dims:
                return dims
    return ()


def _wire_rejection(cell: Any, p_name: str, port: Any, v_name: str, ctx: Any, registry: Any = None,
                    cell_bindings: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Universal admissibility test for "may variable `v_name` feed port `p_name` of `cell`?".
    Checks universal dataflow availability, typestate compatibility, and declarative
    disjointness / complement constraints without domain-specific assumptions.
    """
    if not ctx or not hasattr(ctx, "variables"):
        return None
    if v_name not in ctx.variables:
        return ("unknown_variable", {"v_name": v_name, "port": p_name})

    reg = registry if registry is not None else TypeRegistry.get_instance()

    if port is not None:
        p_role = str(getattr(port, "port_role", "") or getattr(port, "role", "") or getattr(port, "derived_role", "") or "").lower()
        cfg_roles = reg.get_config_roles_set()
        if p_role in cfg_roles or str(getattr(port, "param_kind", "")).lower() in cfg_roles:
            str_kind = str(getattr(port, "structure_kind", "") or "").lower()
            c_rank = getattr(port, "canonical_rank", None) if getattr(port, "canonical_rank", None) is not None else getattr(port, "rank", None)
            t_name = str(getattr(getattr(port, "signature", port), "type_name", "") or getattr(port, "type_name", "") or "").strip().lower()
            is_scalar = (
                c_rank == 0
                or str_kind == "scalar"
                or (reg is not None and reg.is_subtype(t_name, "scalar"))
            )
            if is_scalar:
                # Allow a non-primitive domain carrier variable (e.g. `sample_rate` [frequency_hz])
                # whose declared type exactly matches the configuration port; reject arbitrary
                # bare primitive scalars (`int`/`float`/`bool`/`str`) from stealing config ports.
                v_entry_cfg = ctx.variables.get(v_name)
                v_sig_cfg = v_entry_cfg[0] if isinstance(v_entry_cfg, tuple) else v_entry_cfg
                v_sig_inner = getattr(v_sig_cfg, "signature", v_sig_cfg)
                p_sig_inner = getattr(port, "signature", port)
                v_t_cfg = str(getattr(v_sig_inner, "type_name", "") or "").strip().lower()
                is_domain_carrier = bool(
                    t_name
                    and v_t_cfg
                    and (v_t_cfg == t_name or (reg is not None and reg.is_subtype(v_t_cfg, t_name)))
                    and (reg is None or (not reg.is_builtin_scalar_primitive(t_name) and not reg.is_builtin_scalar_primitive(v_t_cfg)))
                    and (reg is None or not reg.is_declared_top(t_name))
                    and (reg is None or not reg.is_void_type(t_name))
                    and unify(v_sig_inner, p_sig_inner) is not None
                )
                if not is_domain_carrier:
                    return ("config_port_rejects_dataflow_wire", {"v_name": v_name, "port": p_name})

        v_entry = ctx.variables[v_name]
        v_sig = v_entry[0] if isinstance(v_entry, tuple) else v_entry
        v_actual_sig = getattr(v_sig, "signature", v_sig)
        p_actual_sig = getattr(port, "signature", port)
        prod_cell = getattr(ctx, "var_sources", {}).get(v_name) if hasattr(ctx, "var_sources") else None

        if not _shape_compatible(v_sig, port, producer_cell=prod_cell, consumer_cell=cell):
            return ("shape_incompatible", {"v_name": v_name, "port": p_name, "rejected_sig": str(v_actual_sig)})
        res = unify(v_actual_sig, p_actual_sig)
        if res is None:
            return ("type_incompatible", {"v_name": v_name, "port": p_name, "rejected_sig": str(v_actual_sig)})

        # Reject stale component/derived carriers from an earlier row-set epoch when an unconsumed active
        # container at a newer epoch is in scope (forces optic re-derivation from the post-mutation container).
        _act_v = getattr(ctx, "active_carrier_var", None)
        _sup_vs = getattr(ctx, "superseded_vars", set()) or set()
        if (
            cell is not None
            and _act_v
            and _act_v in ctx.variables
            and _act_v not in _sup_vs
            and v_name != _act_v
            and str(getattr(cell, "optic_role", "") or "").strip().lower() != "setter"
        ):
            _v_ep = (getattr(ctx, "var_epochs", None) or {}).get(v_name, 0)
            _a_ep = (getattr(ctx, "var_epochs", None) or {}).get(_act_v, 0)
            _v_sk = str(getattr(v_sig, "structure_kind", "") or getattr(port, "structure_kind", "") or "").lower()
            if _a_ep > _v_ep and _v_sk not in VALUE_LEAF_KINDS:
                _act_sig = getattr(ctx.variables[_act_v][0], "signature", ctx.variables[_act_v][0])
                if unify(_act_sig, p_actual_sig) is None:
                    return ("stale_epoch_carrier", {"v_name": v_name, "port": p_name, "v_epoch": _v_ep, "active_epoch": _a_ep, "active_carrier": _act_v})

            # Reject focus wires whose column provenance contradicts explicit non-target literals on this clause
            _v_origins = {str(o).strip().lower() for o in ((getattr(ctx, "var_origins", None) or {}).get(v_name, set()) or set()) if str(o).strip()}
            _c_lits = [str(l).strip("'\"").strip().lower() for l in (getattr(cell, "clause_literals", None) or ()) if str(l).strip()]
            if _v_origins and _c_lits:
                _tgt_lits = {str(tl).strip().lower() for tl in (getattr(ctx, "target_literals", set()) or set()) if str(tl).strip()}
                if getattr(ctx, "target_key", None):
                    _tgt_lits.add(str(ctx.target_key).strip().lower())
                _tgt_lits.update(str(wk).strip().lower() for wk in (getattr(ctx, "written_keys", set()) or set()) if str(wk).strip())
                _tgt_lits.update(str(pk).strip().lower() for pk in (getattr(ctx, "_pending_setter_keys", set()) or set()) if str(pk).strip())
                _req_lits = {cl for cl in _c_lits if cl not in _tgt_lits and not ExecutionContext._is_path_string(cl)}
                if _req_lits and not (_req_lits & _v_origins):
                    _act_port = ctx.variables[_act_v][0]
                    _act_sig = getattr(_act_port, "signature", _act_port)
                    _act_tn = str(getattr(_act_sig, "type_name", "") or "").strip().lower()
                    _act_sk = str(getattr(_act_port, "structure_kind", "") or "").strip().lower()
                    _act_abs = str(getattr(_act_port, "abstract_type", "") or getattr(_act_sig, "abstract_type", "") or "").strip().lower()
                    _act_is_handle = _act_sk == "handle" or _act_abs == "handle" or reg.is_subtype(_act_tn, "handle")
                    if not _act_is_handle and reg.is_composite_carrier(_act_tn) and unify(_act_sig, p_actual_sig) is None:
                        return ("clause_literal_origin_mismatch", {"v_name": v_name, "port": p_name, "expected": sorted(_req_lits), "var_origins": sorted(_v_origins)})

        bindings = cell_bindings if cell_bindings is not None else kwargs.get("bindings")
        if isinstance(bindings, dict) and cell is not None:
            v_restricted = _var_restricted_keys(v_name, ctx)
            v_excluded = set((getattr(ctx, "var_excluded_origins", None) or {}).get(v_name, ()) or ())
            v_anc = _var_ancestors(v_name, ctx)

            def _bound_var_for(port_ref: str) -> Optional[Tuple[str, str]]:
                p_obj = _resolve_cell_port(cell, port_ref, prefer="inputs")
                key = getattr(p_obj, "name", None) if p_obj is not None else port_ref
                b_val = bindings.get(key) or bindings.get(port_ref)
                if isinstance(b_val, str) and b_val in ctx.variables:
                    return (key or str(port_ref), b_val)
                return None

            def _check_disjoint(partner_ref: str) -> Optional[Tuple[str, Dict[str, Any]]]:
                hit = _bound_var_for(partner_ref)
                if not hit:
                    return None
                partner_key, partner_var = hit
                if partner_var == v_name:
                    return ("disjoint_port_same_variable", {"v_name": v_name, "port": p_name, "other_port": partner_key})
                partner_restr = _var_restricted_keys(partner_var, ctx)
                if v_restricted and partner_restr and (v_restricted & partner_restr):
                    return (
                        "disjoint_port_overlapping_keys",
                        {"v_name": v_name, "port": p_name, "other_port": partner_key, "keys": sorted(v_restricted & partner_restr)},
                    )
                return None

            def _partner_container_keys(comp_var: str, partner_var: str) -> Tuple[Set[str], bool]:
                sf_map = getattr(ctx, "setter_focus_keys", None) or {}
                sf_keys = sf_map.get((comp_var, partner_var))
                if sf_keys:
                    return (set(sf_keys), True)
                vp_map = getattr(ctx, "var_parents", None) or {}
                seen_vp: Set[str] = {comp_var}
                queue_vp = list(vp_map.get(comp_var, ()) or ())
                while queue_vp:
                    anc_v = queue_vp.pop(0)
                    if anc_v in seen_vp:
                        continue
                    seen_vp.add(anc_v)
                    anc_keys = sf_map.get((anc_v, partner_var))
                    if anc_keys:
                        return (set(anc_keys), True)
                    queue_vp.extend(vp_map.get(anc_v, ()) or ())
                return (_var_restricted_keys(partner_var, ctx), False)

            def _check_complement_pair(
                comp_var: str,
                comp_port_key: str,
                comp_excluded: Set[str],
                comp_restricted: Set[str],
                comp_anc: Set[str],
                partner_var: str,
                partner_port_key: str,
                partner_restricted: Set[str],
                partner_anc: Set[str],
                has_setter_binding: bool = False,
            ) -> Optional[Tuple[str, Dict[str, Any]]]:
                if comp_var == partner_var:
                    return ("complement_same_variable", {"v_name": comp_var, "port": comp_port_key, "partner_port": partner_port_key})
                if comp_restricted and partner_restricted and (comp_restricted & partner_restricted):
                    return (
                        "complement_restricted_overlap",
                        {"v_name": comp_var, "port": comp_port_key, "partner_port": partner_port_key, "keys": sorted(comp_restricted & partner_restricted)},
                    )
                if partner_restricted and not comp_restricted:
                    if has_setter_binding or (comp_var in partner_anc) or (comp_anc & partner_anc):
                        if not (partner_restricted <= comp_excluded):
                            return (
                                "complement_unexcluded_container",
                                {
                                    "v_name": comp_var,
                                    "port": comp_port_key,
                                    "partner_port": partner_port_key,
                                    "missing_excluded": sorted(partner_restricted - comp_excluded),
                                },
                            )
                return None

            disj_ref = getattr(port, "disjoint_from_port", None)
            if disj_ref:
                rej = _check_disjoint(disj_ref)
                if rej is not None:
                    return rej

            for other_name, other_port in (getattr(cell, "inputs", {}) or {}).items():
                if other_name == p_name or other_port is port:
                    continue
                other_disj = getattr(other_port, "disjoint_from_port", None)
                if other_disj:
                    resolved = _resolve_cell_port(cell, other_disj, prefer="inputs")
                    if resolved is port or getattr(resolved, "name", None) == p_name or other_disj == p_name:
                        rej = _check_disjoint(other_name)
                        if rej is not None:
                            return rej

            comp_ref = getattr(port, "complement_of_port", None)
            if comp_ref:
                hit = _bound_var_for(comp_ref)
                if hit:
                    partner_key, partner_var = hit
                    eff_p_keys, has_sf = _partner_container_keys(v_name, partner_var)
                    rej = _check_complement_pair(
                        comp_var=v_name,
                        comp_port_key=p_name,
                        comp_excluded=v_excluded,
                        comp_restricted=v_restricted,
                        comp_anc=v_anc,
                        partner_var=partner_var,
                        partner_port_key=partner_key,
                        partner_restricted=eff_p_keys,
                        partner_anc=_var_ancestors(partner_var, ctx),
                        has_setter_binding=has_sf,
                    )
                    if rej is not None:
                        return rej

            for other_name, other_port in (getattr(cell, "inputs", {}) or {}).items():
                if other_name == p_name or other_port is port:
                    continue
                other_comp = getattr(other_port, "complement_of_port", None)
                if other_comp:
                    resolved = _resolve_cell_port(cell, other_comp, prefer="inputs")
                    if resolved is port or getattr(resolved, "name", None) == p_name or other_comp == p_name:
                        hit = _bound_var_for(other_name)
                        if hit:
                            other_key, other_var = hit
                            eff_v_keys, has_sf = _partner_container_keys(other_var, v_name)
                            rej = _check_complement_pair(
                                comp_var=other_var,
                                comp_port_key=other_key,
                                comp_excluded=set((getattr(ctx, "var_excluded_origins", None) or {}).get(other_var, ()) or ()),
                                comp_restricted=_var_restricted_keys(other_var, ctx),
                                comp_anc=_var_ancestors(other_var, ctx),
                                partner_var=v_name,
                                partner_port_key=p_name,
                                partner_restricted=eff_v_keys,
                                partner_anc=v_anc,
                                has_setter_binding=has_sf,
                            )
                            if rej is not None:
                                return rej

            if str(getattr(cell, "optic_role", "") or "").strip().lower() == "getter" and v_excluded:
                cont_p_rej = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs") or getattr(cell, "primary_input", None)
                key_p_rej = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs")
                if cont_p_rej is not None and key_p_rej is not None and (p_name == getattr(cont_p_rej, "name", None) or cont_p_rej is port):
                    bound_k_val = bindings.get(getattr(key_p_rej, "name", None))
                    if bound_k_val is not None:
                        req_keys_rej = _extract_keys_from_val(bound_k_val)
                        if req_keys_rej and (req_keys_rej & v_excluded):
                            return (
                                "getter_key_excluded_in_container",
                                {"v_name": v_name, "port": p_name, "excluded": sorted(req_keys_rej & v_excluded)},
                            )

            # Enforce sample-axis partition consistency across co-arguments that declare
            # a shared symbolic leading dimension in their schema shape_contract/layout
            # (e.g., X:[n_samples, n_features] and y:[n_samples] cannot mix split/unsplit
            # or train/test partitions).
            v_parts = getattr(ctx, "var_partitions", None) or {}
            if v_parts:
                d_p = _port_lead_sym_dim(port)
                if d_p:
                    vp = v_parts.get(v_name)
                    for other_name, other_var in bindings.items():
                        if other_name == p_name or not isinstance(other_var, str) or other_var not in ctx.variables:
                            continue
                        other_port = (getattr(cell, "inputs", {}) or {}).get(other_name)
                        if other_port is None:
                            continue
                        d_o = _port_lead_sym_dim(other_port)
                        if not d_o or d_o != d_p:
                            continue
                        op = v_parts.get(other_var)
                        if vp is None and op is not None:
                            return (
                                "co_argument_partition_mismatch",
                                {"v_name": v_name, "port": p_name, "other_port": other_name, "other_var": other_var, "partition_v": None, "partition_other": op},
                            )
                        if vp is not None and op is None:
                            return (
                                "co_argument_partition_mismatch",
                                {"v_name": v_name, "port": p_name, "other_port": other_name, "other_var": other_var, "partition_v": vp, "partition_other": None},
                            )
                        if vp is not None and op is not None and vp[0] == op[0] and vp[1] != op[1]:
                            return (
                                "co_argument_partition_mismatch",
                                {"v_name": v_name, "port": p_name, "other_port": other_name, "other_var": other_var, "partition_v": vp, "partition_other": op},
                            )

            # Enforce non-leading feature-axis schema exclusion consistency between a fitted handle
            # (e.g. estimator fitted after dropping target columns from a container lineage) and any
            # candidate tensor descending from that same lineage on predict/transform.
            h_excls = getattr(ctx, "handle_dim_exclusions", None) or {}
            if h_excls:
                tail_dims = _port_nonlead_sym_dims(port)
                if tail_dims:
                    for other_name, other_var in bindings.items():
                        if other_name == p_name or not isinstance(other_var, str):
                            continue
                        for sd in tail_dims:
                            entry = h_excls.get((other_var, sd))
                            if not entry:
                                continue
                            req_excl, fit_anc = entry
                            if req_excl and (v_anc & fit_anc) and not (req_excl <= v_excluded):
                                return (
                                    "handle_feature_exclusion_mismatch",
                                    {
                                        "v_name": v_name,
                                        "port": p_name,
                                        "handle": other_var,
                                        "dim": sd,
                                        "missing_excluded": sorted(req_excl - v_excluded),
                                    },
                                )
                elif str(getattr(port, "structure_kind", "") or "").lower() == "handle":
                    for other_name, other_var in bindings.items():
                        if other_name == p_name or not isinstance(other_var, str) or other_var not in ctx.variables:
                            continue
                        other_port = (getattr(cell, "inputs", {}) or {}).get(other_name)
                        if other_port is None:
                            continue
                        other_tails = _port_nonlead_sym_dims(other_port)
                        if not other_tails:
                            continue
                        other_excl = set((getattr(ctx, "var_excluded_origins", None) or {}).get(other_var, ()) or ())
                        other_anc = _var_ancestors(other_var, ctx)
                        for sd in other_tails:
                            entry = h_excls.get((v_name, sd))
                            if not entry:
                                continue
                            req_excl, fit_anc = entry
                            if req_excl and (other_anc & fit_anc) and not (req_excl <= other_excl):
                                return (
                                    "handle_feature_exclusion_mismatch",
                                    {
                                        "v_name": other_var,
                                        "port": other_name,
                                        "handle": v_name,
                                        "dim": sd,
                                        "missing_excluded": sorted(req_excl - other_excl),
                                    },
                                )

    return None


def _update_variable_provenance(
    ctx: ExecutionContext,
    out_var: str,
    cell: Any,
    cell_bindings: Dict[str, Any],
    out_port_name: Optional[str] = None,
) -> None:
    """
    Updates variable provenance records using typed binding records and declared properties (R5-7).
    Tracks container ancestry, restricted key subsets (getter isolation), excluded keys
    (complement provenance), and upstream role-bound keys without domain hardcodes.
    """
    ctx.var_parents.setdefault(out_var, set())
    ctx.var_origins.setdefault(out_var, set())
    if not hasattr(ctx, "var_restricted_origins") or ctx.var_restricted_origins is None:
        ctx.var_restricted_origins = {}
    if not hasattr(ctx, "var_excluded_origins") or ctx.var_excluded_origins is None:
        ctx.var_excluded_origins = {}
    if not hasattr(ctx, "role_bound_keys") or ctx.role_bound_keys is None:
        ctx.role_bound_keys = {}
    ctx.var_restricted_origins.setdefault(out_var, set())
    ctx.var_excluded_origins.setdefault(out_var, set())
    try:
        from .lattice import TypeRegistry
        _prov_reg = TypeRegistry.get_instance()
    except Exception:
        try:
            from lattice import TypeRegistry
            _prov_reg = TypeRegistry.get_instance()
        except Exception:
            _prov_reg = None

    p_out = (
        getattr(cell, "outputs", {}).get(out_port_name)
        if out_port_name and getattr(cell, "outputs", None) and out_port_name in cell.outputs
        else (getattr(cell, "primary_output", None) or (next(iter(cell.outputs.values())) if getattr(cell, "outputs", None) else None))
    )
    out_sig = getattr(p_out, "signature", p_out) if p_out else None
    out_tn = str(getattr(getattr(out_sig, "signature", out_sig), "type_name", "") or getattr(out_sig, "type_name", "")).lower()
    is_out_table = bool(_prov_reg and (_prov_reg.is_composite_carrier(out_tn) or _prov_reg.is_abstract_carrier(out_tn)))

    # Record bound keys by declared port_role / binds for key_from_upstream_role resolution
    cell_inputs = getattr(cell, "inputs", {}) or {}
    df_roles = (
        _prov_reg.get_roles_for_category("data")
        | _prov_reg.get_data_bearing_roles()
        | _prov_reg.get_roles_for_category("strict")
        | _prov_reg.get_roles_for_category("estimator")
    )
    for p_n, p_val in cell_bindings.items():
        if p_n not in cell_inputs or not p_val or p_val is UNRESOLVED_PORT:
            continue
        p_sig_in = cell_inputs[p_n]
        if isinstance(p_val, str) and p_val in getattr(ctx, "variables", {}):
            continue
        if _lattice_is_path_port(p_sig_in):
            continue
        p_r_in = str(getattr(p_sig_in, "port_role", "") or "").strip().lower()
        p_b_in = str(getattr(p_sig_in, "binds", "") or "").strip().lower()
        keys_extracted = _extract_keys_from_val(p_val)
        if not keys_extracted:
            continue
        for role_tag in (p_r_in, p_b_in):
            if role_tag and role_tag not in df_roles:
                lst = ctx.role_bound_keys.setdefault(role_tag, [])
                for k_it in sorted(keys_extracted):
                    if k_it in lst:
                        lst.remove(k_it)
                    lst.append(k_it)

    optic_role = str(getattr(cell, "optic_role", "") or "").strip().lower()
    cont_port_sig = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs")
    key_port_sig = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs")
    focus_port_sig = _resolve_cell_port(cell, getattr(cell, "focus_port", None), prefer="inputs")

    def _port_bound_val(p_sig_obj: Any, fallback_ref: Optional[str] = None) -> Any:
        if p_sig_obj is not None:
            pn = getattr(p_sig_obj, "name", None)
            if pn and pn in cell_bindings:
                return cell_bindings[pn]
        if fallback_ref and fallback_ref in cell_bindings:
            return cell_bindings[fallback_ref]
        return None

    if optic_role == "getter":
        cont_val = _port_bound_val(cont_port_sig, getattr(cell, "container_port", None))
        if isinstance(cont_val, str) and cont_val in getattr(ctx, "variables", {}) and cont_val != out_var:
            ctx.var_parents[out_var].add(cont_val)
        key_val = _port_bound_val(key_port_sig, getattr(cell, "key_port", None))
        keys = _extract_keys_from_val(key_val)
        # Strictly isolate getter fiber from container's unrelated origins and exclusions
        ctx.var_origins[out_var].update(keys)
        ctx.var_restricted_origins[out_var].update(keys)
    elif optic_role == "complement":
        if cont_port_sig is None:
            cont_port_sig = getattr(cell, "primary_input", None)
        if key_port_sig is None:
            key_port_sig = next((p for p in cell_inputs.values() if getattr(p, "complement_of_port", None)), None)
        cont_val = _port_bound_val(cont_port_sig, getattr(cell, "container_port", None))
        key_val = _port_bound_val(key_port_sig, getattr(cell, "key_port", None))
        dropped_keys = _extract_keys_from_val(key_val)
        if isinstance(cont_val, str) and cont_val in getattr(ctx, "variables", {}):
            if cont_val != out_var:
                ctx.var_parents[out_var].add(cont_val)
            ctx.var_excluded_origins[out_var].update(ctx.var_excluded_origins.get(cont_val, set()))
            ctx.var_excluded_origins[out_var].update(dropped_keys)
            ctx.var_origins[out_var].update(ctx.var_origins.get(cont_val, set()) - dropped_keys)
            cont_restr = set(ctx.var_restricted_origins.get(cont_val, set()))
            if cont_restr:
                ctx.var_restricted_origins[out_var].update(cont_restr - dropped_keys)
    elif optic_role == "setter":
        if cont_port_sig is None:
            cont_port_sig = getattr(cell, "primary_input", None)
        cont_val = _port_bound_val(cont_port_sig, getattr(cell, "container_port", None))
        key_val = _port_bound_val(key_port_sig, getattr(cell, "key_port", None))
        focus_val = _port_bound_val(focus_port_sig, getattr(cell, "focus_port", None))
        written_keys = _extract_keys_from_val(key_val)
        if isinstance(cont_val, str) and cont_val in getattr(ctx, "variables", {}):
            cont_origs = set(ctx.var_origins.get(cont_val, set()))
            cont_excl = set(ctx.var_excluded_origins.get(cont_val, set()))
            cont_restr = set(ctx.var_restricted_origins.get(cont_val, set()))
            if cont_val != out_var:
                ctx.var_parents[out_var].add(cont_val)
                ctx.var_origins[out_var].update(cont_origs | written_keys)
                ctx.var_excluded_origins[out_var].update(cont_excl - written_keys)
                if cont_restr:
                    ctx.var_restricted_origins[out_var].update(cont_restr | written_keys)
            else:
                ctx.var_origins[out_var] = cont_origs | written_keys
                ctx.var_excluded_origins[out_var] = cont_excl - written_keys
                if cont_restr:
                    ctx.var_restricted_origins[out_var] = cont_restr | written_keys
        if isinstance(focus_val, str) and focus_val in getattr(ctx, "variables", {}) and focus_val != out_var:
            ctx.var_parents[out_var].add(focus_val)
    else:
        allowed_in_ports = _match_multi_output_inputs(out_port_name, p_out, cell, cell_bindings, ctx)
        var_sigs_curr = {v: sig_tuple[0] for v, sig_tuple in getattr(ctx, "variables", {}).items()}
        carrier_parents: List[str] = []
        rec_roles = (
            _prov_reg.get_roles_for_category("receiver")
            | _prov_reg.get_roles_for_category("instance_sink")
            | _prov_reg.get_roles_for_category("owner_sink")
        )
        for p_n, p_val in cell_bindings.items():
            if not p_val or p_val is UNRESOLVED_PORT:
                continue
            if p_n == "output_var" or p_n in (getattr(cell, "outputs", {}) or {}):
                continue
            if allowed_in_ports is not None and p_n not in allowed_in_ports:
                continue
            p_sig_n = cell_inputs.get(p_n) or getattr(cell, "outputs", {}).get(p_n)
            rec = (getattr(cell, "typed_bindings", {}) or {}).get(p_n)
            if rec is None:
                rec = resolve_typed_binding_record(
                    cell=cell,
                    port_name=p_n,
                    bound_val=p_val,
                    port_sig=getattr(p_sig_n, "signature", p_sig_n) if p_sig_n else None,
                    var_signatures=var_sigs_curr,
                )
            _is_projection = bool(rec.kind == "projection" and rec.projection and rec.projection.get("key"))
            if rec.root_var and rec.root_var in ctx.variables:
                if rec.root_var != out_var:
                    ctx.var_parents[out_var].add(rec.root_var)
                p_role_in = str(getattr(p_sig_n, "port_role", "") or "").lower()
                if p_n in cell_inputs and p_role_in not in rec_roles and rec.root_var not in carrier_parents:
                    carrier_parents.append(rec.root_var)
                if not _is_projection:
                    root_sig_tuple = ctx.variables.get(rec.root_var)
                    root_sig = root_sig_tuple[0] if root_sig_tuple else None
                    root_tn = str(getattr(getattr(root_sig, "signature", root_sig), "type_name", "") or getattr(root_sig, "type_name", "")).lower()
                    is_root_table = bool(_prov_reg and (_prov_reg.is_composite_carrier(root_tn) or _prov_reg.is_abstract_carrier(root_tn)))
                    if not (is_root_table and not is_out_table):
                        ctx.var_origins[out_var].update(ctx.var_origins.get(rec.root_var, set()))
            if p_val and str(p_val) in ctx.var_origins:
                ctx.var_origins[out_var].update(ctx.var_origins.get(str(p_val), set()))
            if rec.kind == "projection" and rec.projection:
                p_key = rec.projection.get("key")
                if p_key:
                    if isinstance(p_key, list):
                        for k_item in p_key:
                            ctx.var_origins[out_var].add(str(k_item).strip("'\""))
                    else:
                        ctx.var_origins[out_var].add(str(p_key).strip("'\""))
            elif rec.value_expr:
                p_binds = getattr(p_sig_n, "binds", None)
                p_role = str(getattr(p_sig_n, "port_role", "") or "").lower()
                bind_keys = _prov_reg.get_roles_for_category("key") | _prov_reg.get_roles_for_category("assigned_value")
                proj_roles = (
                    _prov_reg.get_roles_for_category("projection")
                    | set(_prov_reg.get_projection_tokens() or ())
                    | _prov_reg.get_roles_for_category("key")
                )
                proj_names = _prov_reg.get_roles_for_category("projection") | set(_prov_reg.get_projection_tokens() or ())
                if (
                    p_binds in bind_keys
                    or p_role in proj_roles
                    or p_n.lower() in proj_names
                ):
                    for cl_it in _extract_keys_from_val(rec.value_expr):
                        ctx.var_origins[out_var].add(cl_it)

        if len(carrier_parents) == 1:
            p0 = carrier_parents[0]
            ctx.var_restricted_origins[out_var].update(ctx.var_restricted_origins.get(p0, set()))
            ctx.var_excluded_origins[out_var].update(ctx.var_excluded_origins.get(p0, set()))
        elif len(carrier_parents) > 1:
            excl_sets = [set(ctx.var_excluded_origins.get(pv, set())) for pv in carrier_parents]
            ctx.var_excluded_origins[out_var].update(set.intersection(*excl_sets))
            restr_sets = [set(ctx.var_restricted_origins.get(pv, set())) for pv in carrier_parents]
            if all(restr_sets):
                ctx.var_restricted_origins[out_var].update(set().union(*restr_sets))

    _read_optics = _prov_reg.get_roles_for_category("optic") - {"setter"}
    if optic_role not in _read_optics and not ctx.var_origins[out_var]:
        c_lits = getattr(cell, "clause_literals", None)
        if c_lits:
            for cl_lit in c_lits:
                clean_lit = str(cl_lit).strip("'\"")
                if clean_lit:
                    ctx.var_origins[out_var].add(clean_lit)

    if hasattr(ctx, "multi_fiber_vars") and isinstance(ctx.multi_fiber_vars, set):
        out_origins = ctx.var_origins.get(out_var, set())
        if len(out_origins) > 1:
            ctx.multi_fiber_vars.add(out_var)
    # Track feature-axis schema exclusions committed onto a handle (e.g. fitted estimator)
    if str(getattr(p_out, "structure_kind", "") or "").lower() == "handle":
        if not hasattr(ctx, "handle_dim_exclusions") or ctx.handle_dim_exclusions is None:
            ctx.handle_dim_exclusions = {}
        for in_pn, in_po in cell_inputs.items():
            in_val = cell_bindings.get(in_pn)
            if not isinstance(in_val, str) or in_val not in getattr(ctx, "variables", {}):
                continue
            tail_dims = _port_nonlead_sym_dims(in_po)
            if not tail_dims:
                # Propagate upstream handle dim constraints through stateful mutators/chaining
                if str(getattr(in_po, "structure_kind", "") or "").lower() == "handle":
                    for (hv, sd), entry in list(ctx.handle_dim_exclusions.items()):
                        if hv == in_val and (out_var, sd) not in ctx.handle_dim_exclusions:
                            ctx.handle_dim_exclusions[(out_var, sd)] = (set(entry[0]), set(entry[1]))
                continue
            in_excl = set(ctx.var_excluded_origins.get(in_val, ()) or ())
            if in_excl:
                in_anc = _var_ancestors(in_val, ctx) | {in_val}
                for sd in tail_dims:
                    ctx.handle_dim_exclusions[(out_var, sd)] = (set(in_excl), set(in_anc))
    # Row-set epoch: how many row-changing steps (tree-declared extent_mutation/dedup/schema_removal effects) this value has been through.
    _has_row_fx = cell_has_effect_kind(cell, EXTENT_EFFECT_KINDS)
    _parent_vars = set(ctx.var_parents.get(out_var, set()))
    for p_v in cell_bindings.values():
        if isinstance(p_v, str) and p_v in ctx.var_epochs:
            _parent_vars.add(p_v)
    _ep = max((ctx.var_epochs.get(v, 0) for v in _parent_vars), default=0)
    if _has_row_fx:
        _ep += 1
    ctx.var_epochs[out_var] = _ep

# Purged: Zero hardcoded English wordlists and syntactic hacks
_SENTENCE_CONNECTIVES_REMOVED = None


@functools.lru_cache(maxsize=2048)
def _extract_template_callables_cached(tpl: str) -> Tuple[str, ...]:
    """Cached tuple of root callable identifiers extracted from a code template."""
    callables: Set[str] = set()
    if not tpl:
        return ()
    clean_tpl = tpl
    for ph in extract_template_placeholders(tpl):
        clean_tpl = clean_tpl.replace(f"{{{ph}}}", "_arg_")
    parsed_tpl, _ = _cached_ast_parse_ro(clean_tpl, "exec")
    if parsed_tpl is not None:
        for n in ast.walk(parsed_tpl):
            if isinstance(n, ast.Call):
                func = n.func
                while isinstance(func, ast.Attribute):
                    func = func.value
                if isinstance(func, ast.Name):
                    callables.add(func.id)
    else:
        i = 0
        n_len = len(clean_tpl)
        while i < n_len:
            if clean_tpl[i] == "(":
                j = i - 1
                while j >= 0 and clean_tpl[j].isspace():
                    j -= 1
                end_id = j + 1
                while j >= 0 and (clean_tpl[j].isalnum() or clean_tpl[j] == "_"):
                    j -= 1
                start_id = j + 1
                cand = clean_tpl[start_id:end_id]
                if cand and cand.isidentifier():
                    callables.add(cand)
            i += 1
    return tuple(sorted(callables))


def _extract_template_callables(tpl: str) -> Set[str]:
    """Extracts root callable identifiers from a code template without regular expressions."""
    return set(_extract_template_callables_cached(tpl or ""))


class IdentifierGroup:
    """
    An enumerable set of bare referential identifiers sharing one syntactic
    role context (e.g. ``normalize X column, Y column and Z column`` groups
    X, Y, Z under the role noun ``column``).

    members:     [(char_pos, token), ...] in prompt order
    role_tokens: stemmed context tokens naming the shared role ({"column"});
                 empty when an identifier stands alone with no content context.
    """

    __slots__ = ("members", "role_tokens")

    def __init__(self, members: List[Tuple[int, str]], role_tokens: Union[Set[str], Tuple[str, ...]]):
        self.members = members
        self.role_tokens = role_tokens

    def __repr__(self) -> str:
        return f"IdentifierGroup(members={[m[1] for m in self.members]}, role={set(self.role_tokens)})"


def is_top_symbol(raw: str, registry: Optional[Any] = None) -> bool:
    s = str(raw).strip()
    if s in ("⊤", "*"):
        return True
    if registry is None:
        registry = TypeRegistry.get_instance()
    return registry.is_declared_top(s)


def _normalize_quals(quals: Any) -> Tuple[Tuple[str, ...], ...]:
    if not quals:
        return ()
    res: Set[Tuple[str, ...]] = set()
    for q in quals:
        if isinstance(q, (list, tuple, set)):
            res.add(tuple(str(x) for x in q))
        else:
            res.add((str(q),))
    return tuple(sorted(res))


def _get_p_field(p: Any, field_name: str, default: Any = None) -> Any:
    if p is None:
        return default
    if isinstance(p, dict):
        return p.get(field_name, default)
    sig = getattr(p, "signature", None)
    if sig is not None and not isinstance(p, (str, int, float, bool)):
        val = getattr(sig, field_name, None)
        if val is not None:
            return val
    return getattr(p, field_name, default)


def _enrich_port_signature_qualifiers(port_sig: Any, extra_quals: Set[str]) -> Any:
    """Enriches a PortSignature or AlgebraicSignature with additional propagated qualifiers."""
    if not extra_quals or port_sig is None:
        return port_sig
    old_sig = getattr(port_sig, "signature", port_sig)
    raw_old = _get_p_field(port_sig, "qualifiers", None) or getattr(old_sig, "qualifiers", None) or ()
    norm_old = {
        tuple(str(el).lower() for el in q)
        for q in _normalize_qualifiers(raw_old)
        if q
    }
    norm_extra = {
        tuple(str(el).lower() for el in q) if isinstance(q, (tuple, list, set)) else (str(q).lower(),)
        for q in extra_quals
        if q is not None and str(q).strip()
    }
    new_norm = tuple(sorted(norm_old | norm_extra))

    new_alg_sig = AlgebraicSignature(
        type_name=getattr(old_sig, "type_name", "any"),
        state=getattr(old_sig, "state", "any"),
        qualifiers=new_norm,
        abstract_type=getattr(old_sig, "abstract_type", ""),
        accepted_states=getattr(old_sig, "accepted_states", ()),
        parent_state=getattr(old_sig, "parent_state", None),
        domain=getattr(old_sig, "domain", ""),
    )
    if isinstance(port_sig, PortSignature):
        return port_sig.with_signature(new_alg_sig)
    return new_alg_sig


def _enrich_port_signature_layout(
    port_sig: Any,
    key_layout: Optional[Dict[str, Any]],
    registry: Optional[TypeRegistry] = None,
) -> Any:
    """Enrich erased/polymorphic dtype and rank slots on `port_sig` from a container key's recorded layout."""
    if not isinstance(port_sig, PortSignature) or not isinstance(key_layout, dict):
        return port_sig
    reg = registry or TypeRegistry.get_instance()

    def _is_erased_dtype(dt: Any) -> bool:
        if dt is None:
            return True
        dt_s = str(dt).strip()
        dt_l = dt_s.lower()
        return (
            dt_l in ("", "any", "composite_record", "opaque", "object")
            or reg.is_type_variable(dt_s)
            or dt_l in reg.get_types_by_kind("opaque")
        )

    cur_lay = dict(port_sig.layout) if isinstance(port_sig.layout, dict) else {}
    new_lay = dict(cur_lay)
    if _is_erased_dtype(cur_lay.get("dtype")) and not _is_erased_dtype(key_layout.get("dtype")):
        new_lay["dtype"] = key_layout["dtype"]
    if (
        cur_lay.get("rank") is None
        or isinstance(cur_lay.get("rank"), str)
        or bool(cur_lay.get("rank_polymorphic"))
    ) and isinstance(key_layout.get("rank"), int) and not isinstance(key_layout.get("rank"), bool):
        new_lay["rank"] = key_layout["rank"]
        new_lay["rank_polymorphic"] = False
    if new_lay == cur_lay:
        return port_sig
    res = copy.copy(port_sig)
    res.layout = new_lay
    return res


def _lookup_consistent_key_layout(
    ctx: Any,
    cont_var: Optional[str],
    keys: Any,
) -> Optional[Dict[str, Any]]:
    """Return a shared layout dict if all requested `keys` on `cont_var` agree on `(dtype, rank)`."""
    if not isinstance(cont_var, str) or not keys:
        return None
    ckl_map = getattr(ctx, "container_key_layouts", None) or {}
    if not ckl_map:
        return None
    matched = [
        ckl_map[(cont_var, str(k).strip().lower())]
        for k in keys
        if str(k).strip() and (cont_var, str(k).strip().lower()) in ckl_map
    ]
    if not matched:
        return None
    if len({l.get("dtype") for l in matched}) == 1 and len({l.get("rank") for l in matched}) == 1:
        return matched[0]
    return None


def _propagate_container_key_layouts(
    ctx: Any,
    src_vars: Any,
    out_cont_var: Optional[str],
    excluded_keys: Any = (),
    overwrite: bool = False,
) -> None:
    """Propagate untouched per-key layouts from upstream carrier variables to `out_cont_var`."""
    if not isinstance(out_cont_var, str) or not out_cont_var:
        return
    ckl_map = getattr(ctx, "container_key_layouts", None)
    if not isinstance(ckl_map, dict) or not ckl_map:
        return
    src_set = {v for v in (src_vars or ()) if isinstance(v, str)}
    excl_set = {str(k).strip().lower() for k in (excluded_keys or ()) if str(k).strip()}
    for (kv_var, k_key), kl_val in list(ckl_map.items()):
        if kv_var in src_set and k_key not in excl_set and isinstance(kl_val, dict):
            if overwrite or (out_cont_var, k_key) not in ckl_map:
                ckl_map[(out_cont_var, k_key)] = dict(kl_val)


def _record_setter_key_layout(
    ctx: Any,
    cont_var: Optional[str],
    out_cont_var: Optional[str],
    focus_var: Optional[str],
    written_keys: Any,
) -> None:
    """Propagate untouched container key layouts and record written key layout from `focus_var`."""
    if not isinstance(out_cont_var, str) or not out_cont_var:
        return
    if not hasattr(ctx, "container_key_layouts") or ctx.container_key_layouts is None:
        ctx.container_key_layouts = {}
    wk_low = {str(wk).strip().lower() for wk in (written_keys or ()) if str(wk).strip()}
    _propagate_container_key_layouts(ctx, (cont_var,), out_cont_var, excluded_keys=wk_low, overwrite=True)
    f_lay = None
    if isinstance(focus_var, str) and focus_var in getattr(ctx, "variables", {}):
        f_sig_v = ctx.variables[focus_var][0]
        raw_lay = getattr(f_sig_v, "layout", None)
        if isinstance(raw_lay, dict):
            f_lay = dict(raw_lay)
    for wk in wk_low:
        if f_lay is not None:
            ctx.container_key_layouts[(out_cont_var, wk)] = dict(f_lay)
        else:
            ctx.container_key_layouts.pop((out_cont_var, wk), None)



def _collect_propagated_qualifiers(cell: Any, bindings: Dict[str, Any], var_sigs: Dict[str, Any]) -> Set[str]:
    """
    Functorial qualifier propagation:
    Collects monadic qualifiers (e.g. 'complex', 'tainted') from incoming bound variables
    across container mutators, element-wise transforms, and container extractions,
    unless the cell declares an explicit elimination/bridge for that qualifier.
    """
    bound_in_quals: Set[str] = set()
    for inp_name, inp_sig in getattr(cell, "inputs", {}).items():
        v_in = bindings.get(inp_name)
        if isinstance(v_in, str) and v_in in var_sigs:
            v_sig = var_sigs[v_in]
            q_set = _get_p_field(v_sig, "qualifiers", None) or getattr(getattr(v_sig, "signature", None), "qualifiers", None) or ()
            for q in q_set:
                if isinstance(q, (tuple, list, set)):
                    for q_el in q:
                        bound_in_quals.add(str(q_el).lower())
                else:
                    bound_in_quals.add(str(q).lower())

    if not bound_in_quals:
        return set()

    eliminated_quals: Set[str] = set()
    node_role = getattr(cell, "node_role", "")
    prim_out = getattr(cell, "primary_output", None)
    out_quals = set()
    if prim_out:
        q_raw = _get_p_field(prim_out, "qualifiers", None) or getattr(getattr(prim_out, "signature", None), "qualifiers", None) or ()
        for q in q_raw:
            if isinstance(q, (tuple, list, set)):
                out_quals.update(str(x).lower() for x in q)
            else:
                out_quals.add(str(q).lower())

    # Carrier conversion detection:
    # A morphism F: C1 -> C2 between distinct container carrier categories (e.g. Table -> Tensor)
    # is a carrier functor that preserves value-level monadic qualifiers.
    prim_in = getattr(cell, "primary_input", None) or (next(iter(cell.inputs.values())) if getattr(cell, "inputs", None) else None)
    in_carrier = str(getattr(getattr(prim_in, "signature", prim_in), "type_name", "") or getattr(prim_in, "type_name", "")).lower() if prim_in else ""
    out_carrier = str(getattr(getattr(prim_out, "signature", prim_out), "type_name", "") or getattr(prim_out, "type_name", "")).lower() if prim_out else ""
    
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
    except Exception:
        reg = None

    def _is_carrier_type(t_name: str) -> bool:
        if not t_name or reg is None:
            return False
        t_low = str(t_name).strip().lower()
        if hasattr(reg, "is_carrier"):
            return reg.is_carrier(t_low)
        return (
            reg.is_composite_carrier(t_low)
            or reg.is_abstract_carrier(t_low)
            or t_low in reg.get_top_level_carriers()
            or bool(reg.get_abstract_carrier(t_low))
        )

    is_in_carrier = _is_carrier_type(in_carrier)
    is_out_carrier = _is_carrier_type(out_carrier)
    is_carrier_bridge = bool(is_in_carrier and is_out_carrier and in_carrier != out_carrier)
    in_ports = list(getattr(cell, "inputs", {}).values())

    for q in bound_in_quals:
        if is_carrier_bridge:
            # Functorial preservation across carrier changes: F(Carrier1[q]) = Carrier2[q]
            continue
        is_eliminated = False
        # 1. Postcondition check: cell explicitly asserts that property is_{q} or has_{q} or {q} is False
        for post in (getattr(cell, "postconditions", None) or []):
            prop = str(post.get("property", "") if isinstance(post, dict) else getattr(post, "property", "")).lower()
            val = post.get("value") if isinstance(post, dict) else getattr(post, "value", None)
            if (prop in (f"is_{q}", f"has_{q}", q)) and val is False:
                is_eliminated = True
                break
        # 2. Effects check: cell explicitly declares an elimination effect
        if not is_eliminated:
            for eff in (getattr(cell, "effects", None) or []):
                eff_str = str(eff).lower()
                if eff_str in (f"eliminates:{q}", f"removes:{q}"):
                    is_eliminated = True
                    break
        # 3. Explicit qualifier-eliminating input state/qualifier transition:
        # input port primary state or qualifiers explicitly require q, and output does not carry q
        if not is_eliminated and q not in out_quals:
            for p_in in in_ports:
                p_in_quals = [str(x).lower() for x in (getattr(p_in, "qualifiers", None) or ())]
                p_in_state = str(getattr(getattr(p_in, "signature", p_in), "state", "") or "").lower()
                p_in_toks = set(p_in_quals) | set(CellTokenizer.tokenize_identifier(p_in_state))
                if str(q).lower() in p_in_toks:
                    is_eliminated = True
                    break
        if is_eliminated:
            eliminated_quals.add(q)

    return bound_in_quals - eliminated_quals


def _port_declares_acceptance(p_sig: Any, wanted_quals: Set[str]) -> bool:
    """True when port declared state/accepted_states/qualifiers vocab mentions any wanted qualifier."""
    declared: List[str] = []
    for q_item in (getattr(p_sig, "qualifiers", None) or ()):
        if isinstance(q_item, (tuple, list)):
            declared.extend(str(x) for x in q_item if x is not None)
        elif q_item is not None:
            declared.append(str(q_item))
    sig_inner = getattr(p_sig, "signature", None) or p_sig
    declared.append(str(getattr(sig_inner, "state", "") or ""))
    declared.extend(str(s) for s in (getattr(sig_inner, "accepted_states", None) or ()))
    declared_toks: Set[str] = set()
    for d in declared:
        declared_toks.update(CellTokenizer.tokenize_identifier(str(d).lower()))
    return any(str(q).lower() in declared_toks for q in wanted_quals)


def _extract_rejected_qualifiers(p_in: Any, consumer_cell: Optional[Any] = None) -> Set[str]:
    """Extracts qualifiers rejected by input port `p_in` from both its `rejected_qualifiers`
    declaration and any negating preconditions (`is_<q> == False`) on `consumer_cell` or `p_in`."""
    if p_in is None:
        return set()
    in_sig_o = getattr(p_in, "signature", p_in)
    if consumer_cell is None:
        consumer_cell = _get_p_field(p_in, "owner_cell") or getattr(in_sig_o, "owner_cell", None)
    rej_raw = (
        _get_p_field(p_in, "rejected_qualifiers", None)
        or getattr(in_sig_o, "rejected_qualifiers", None)
        or ()
    )
    cell_pre = _get_p_field(consumer_cell, "preconditions", None) if consumer_cell is not None else None
    port_pre = _get_p_field(p_in, "preconditions", None)
    if not rej_raw and not cell_pre and not port_pre:
        return set()

    flat_rejs: Set[str] = set()
    for r in rej_raw:
        if isinstance(r, (tuple, list, set)):
            flat_rejs.update(str(x).lower() for x in r if str(x).strip())
        elif str(r).strip():
            flat_rejs.add(str(r).strip().lower())

    if cell_pre or port_pre:
        p_name = _get_p_field(p_in, "name", None)
        preconds = list(cell_pre or ())
        if port_pre:
            preconds.extend(port_pre)
        for pre in preconds:
            if not isinstance(pre, dict):
                continue
            target = pre.get("target")
            if target and p_name and str(target) != str(p_name):
                continue
            prop = str(pre.get("property") or "").strip().lower()
            op = str(pre.get("operator", "==") or "==").strip()
            val = pre.get("value")
            if (op == "==" and val is False) or (op == "!=" and val is True):
                if prop.startswith("is_") and len(prop) > 3:
                    flat_rejs.add(prop[3:])
                elif prop.startswith("has_") and len(prop) > 4:
                    flat_rejs.add(prop[4:])
                elif prop:
                    flat_rejs.add(prop)
    return flat_rejs


def _instantiate_bridge_output_signature(
    b_cell: Any,
    b_out: Any,
    v_sig: Any,
    b_in_name: str = "x",
    v_name: str = "_v",
) -> Any:
    """
    Instantiates the concrete output PortSignature of an eliminator bridge `b_cell`
    applied to input `v_sig`:
      1. Propagates non-eliminated qualifiers from `v_sig` via `_collect_propagated_qualifiers`.
      2. If `b_out.shape_contract` is None (rank-polymorphic bridge), inherits `shape_contract`
         and `rank` (`canonical_rank`) from `v_sig`.
    """
    if b_out is None:
        return None
    prop_quals = _collect_propagated_qualifiers(b_cell, {b_in_name: v_name}, {v_name: v_sig})
    out_sig = _enrich_port_signature_qualifiers(b_out, prop_quals)
    if isinstance(out_sig, PortSignature):
        v_sc = _get_p_field(v_sig, "shape_contract", None)
        v_rank = getattr(v_sig, "canonical_rank", None)
        if v_rank is None:
            v_rank = _get_p_field(v_sig, "rank", None)
        if out_sig.shape_contract is None and (v_sc is not None or v_rank is not None):
            out_sig = copy.copy(out_sig)
            if v_sc is not None:
                out_sig.shape_contract = copy.deepcopy(v_sc) if isinstance(v_sc, dict) else v_sc
            elif v_rank is not None:
                out_sig.shape_contract = {"ndim": v_rank, "rank": v_rank}
            if v_rank is not None:
                out_sig.rank = v_rank
            out_sig._cached_ndim = None
            if hasattr(out_sig, "_cached_rank_cache"):
                out_sig._cached_rank_cache = None
    if isinstance(getattr(b_cell, "polymorphic_propagation", None), dict) and v_sig is not None:
        out_sig = apply_polymorphic_propagation(out_sig, v_sig, b_cell.polymorphic_propagation)
    return out_sig


def _find_bridge_for_qualifier_conflict(
    orch: Any,
    v_sig: Any,
    p_sig: Any,
    conflict: Set[str],
    rej_quals: Set[str],
    consumer_cell: Optional[Any] = None,
) -> Optional[Cell]:
    """
    Universal lattice lookup for an eliminator bridge:
    Searches loaded cells for a transformer/bridge whose input accepts the conflicting
    monadic qualifier(s) and whose output does NOT carry any rejected qualifier,
    unifying from the input carrier v_sig and unifying into the downstream target port p_sig.
    """
    if orch is None:
        try:
            from lattice import LatticeOrchestrator
            orch = LatticeOrchestrator.get_active_instance()
        except Exception:
            orch = None
    if orch is None or not hasattr(orch, "loaded_cells"):
        return None

    try:
        reg_br = TypeRegistry.get_instance()
        _br_roles = (
            reg_br.get_bridge_roles_set()
            | reg_br.get_roles_for_category("clean")
            | reg_br.get_roles_for_category("transform")
        )
    except Exception:
        _br_roles = set()
    for b_cand in orch.loaded_cells.values():
        cand_role = str(getattr(b_cand, "node_role", "") or getattr(b_cand, "role", "") or "").lower()
        if cand_role not in _br_roles:
            continue
        b_out = getattr(b_cand, "primary_output", None) or (next(iter(b_cand.outputs.values())) if getattr(b_cand, "outputs", None) else None)
        if b_out is None:
            continue

        if not any(_port_declares_acceptance(p_in, conflict) for p_in in b_cand.inputs.values()):
            continue

        b_in = getattr(b_cand, "primary_input", None) or (next(iter(b_cand.inputs.values())) if getattr(b_cand, "inputs", None) else None)
        if b_in is not None and not _shape_compatible(v_sig, b_in, consumer_cell=b_cand):
            continue

        b_in_name = getattr(b_in, "name", "x") if b_in is not None else "x"
        b_out_concrete = _instantiate_bridge_output_signature(b_cand, b_out, v_sig, b_in_name=b_in_name)

        b_out_quals = set()
        raw_out_q = _get_p_field(b_out_concrete, "qualifiers", None) or getattr(getattr(b_out_concrete, "signature", None), "qualifiers", None) or ()
        for q in raw_out_q:
            if isinstance(q, (tuple, list, set)):
                b_out_quals.update(str(x).lower() for x in q)
            else:
                b_out_quals.add(str(q).lower())
        if b_out_quals & (rej_quals | conflict):
            continue

        if not _shape_compatible(b_out_concrete, p_sig, producer_cell=b_cand, consumer_cell=consumer_cell):
            continue

        return b_cand

    return None


def _extract_ndim(sc: Any) -> Optional[int]:
    if sc is None:
        return None
    if hasattr(sc, "rank_polymorphic") and getattr(sc, "rank_polymorphic"):
        return None
    if hasattr(sc, "ndim") and getattr(sc, "ndim") is not None:
        return getattr(sc, "ndim")
    if isinstance(sc, dict):
        if sc.get("rank_polymorphic"):
            return None
        val = sc.get("ndim")
        if isinstance(val, int):
            return val
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None
    elif isinstance(sc, str):
        s = sc.strip()
        if not s or s.lower() in ("none", "null", "any", "*"):
            return None
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            inner = s[1:-1].strip()
            if not inner:
                return 0
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if any(p in ("*", "...", "any") for p in parts):
                return None
            return len(parts)
    return None


_ROLES_CONFLICT_CACHE: Dict[Tuple[int, int, str, str], bool] = {}


def _roles_conflict(p_role: str, v_role: str, reg: Optional[Any] = None) -> bool:
    """True iff consumer port role p_role and producer/variable role v_role conflict
    according to declared strict_roles and mutually_exclusive_role_pairs."""
    if not p_role or not v_role:
        return False
    if reg is None:
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            return False

    r_port = str(p_role).strip().lower()
    r_var = str(v_role).strip().lower()
    cache_k = (id(reg), getattr(reg, "_generation", 0), r_port, r_var)
    hit = _ROLES_CONFLICT_CACHE.get(cache_k)
    if hit is not None:
        return hit

    # Universal strict role gating: if either role is declared strict,
    # reject mismatched bindings unless one is subtype of the other or both are strict.
    strict_roles = reg.get_roles_for_category("strict") | reg.get_roles_for_category("estimator")
    res = False
    if r_port in strict_roles:
        if r_var != r_port and not reg.is_subtype(r_var, r_port) and r_var not in strict_roles:
            res = True
    if not res and r_var in strict_roles:
        if r_port != r_var and not reg.is_subtype(r_port, r_var) and r_port not in strict_roles:
            res = True

    if not res and r_port != "standard" and r_var != "standard":
        for group in reg.get_exclusive_roles():
            if not group or len(group) < 2:
                continue
            grp = [str(g).strip().lower() for g in group if str(g).strip()]
            port_idx = next((i for i, g in enumerate(grp) if r_port == g or reg.is_subtype(r_port, g)), None)
            var_idx = next((i for i, g in enumerate(grp) if r_var == g or reg.is_subtype(r_var, g)), None)
            if port_idx is not None and var_idx is not None and port_idx != var_idx:
                res = True
                break

    if len(_ROLES_CONFLICT_CACHE) > 8192:
        _ROLES_CONFLICT_CACHE.clear()
    _ROLES_CONFLICT_CACHE[cache_k] = res
    return res


def _flat_quals_set(raw_quals: Any) -> Set[str]:
    flat: Set[str] = set()
    if not raw_quals:
        return flat
    for q in raw_quals:
        if isinstance(q, (tuple, list, set)):
            for q_el in q:
                flat.add(str(q_el).lower())
        else:
            flat.add(str(q).lower())
    return flat


def is_edge_valid(
    p_out: Any,
    p_in: Any,
    producer_cell: Optional[Any] = None,
    consumer_cell: Optional[Any] = None,
) -> bool:
    """
    Universal mathematical edge-validity predicate: e = (u -> v).
    Validates whether producer port/signature `p_out` can soundly satisfy consumer
    port/signature `p_in` across all dual-layer typing contracts:
      tau = <rho, Phi>, where Phi = <S, Q, P, E>
    Validates:
      1. Roles: prediction vs target, feature/target/model gating.
      2. Abstract carrier: composite/abstract vs scalar/primitive.
      3. Rejected qualifiers: producer must not emit any qualifier rejected by consumer.
      4. Qualifiers: consumer required qualifiers must be satisfied.
      5. Rank & shape contract: ndim, channel counts, shapes, and canonical ranks must match.
      6. State compatibility: typestate transition in the state poset.
      7. Preconditions: consumer preconditions on the target port satisfied by producer properties.
      8. Structural type subtyping / Robinson unification: tau_out <= tau_in.
    """
    if p_out is None or p_in is None:
        return False

    is_out_ps = isinstance(p_out, PortSignature)
    is_in_ps = isinstance(p_in, PortSignature)
    out_sig_o = p_out.signature if is_out_ps else getattr(p_out, "signature", p_out)
    in_sig_o = p_in.signature if is_in_ps else getattr(p_in, "signature", p_in)

    if isinstance(out_sig_o, dict):
        out_tn = str(out_sig_o.get("type_name") or "any")
        out_st = str(out_sig_o.get("state") or "any")
        out_abs_t = str(out_sig_o.get("abstract_type") or "")
        out_acc = out_sig_o.get("accepted_states") or []
        out_parent_st = out_sig_o.get("parent_state")
        out_quals = out_sig_o.get("qualifiers") or []
        out_sig_o = AlgebraicSignature(
            type_name=out_tn,
            state=out_st,
            qualifiers=_normalize_quals(out_quals),
            abstract_type=out_abs_t,
            accepted_states=tuple(sorted({str(x).strip().lower() for x in out_acc if str(x).strip()})),
            parent_state=out_parent_st,
        )

    if isinstance(in_sig_o, dict):
        in_tn = str(in_sig_o.get("type_name") or "any")
        in_st = str(in_sig_o.get("state") or "any")
        in_abs_t = str(in_sig_o.get("abstract_type") or "")
        in_acc = in_sig_o.get("accepted_states") or []
        in_parent_st = in_sig_o.get("parent_state")
        in_quals = in_sig_o.get("qualifiers") or []
        in_sig_o = AlgebraicSignature(
            type_name=in_tn,
            state=in_st,
            qualifiers=_normalize_quals(in_quals),
            abstract_type=in_abs_t,
            accepted_states=tuple(sorted({str(x).strip().lower() for x in in_acc if str(x).strip()})),
            parent_state=in_parent_st,
        )

    reg = TypeRegistry.get_instance()
    is_out_alg = isinstance(out_sig_o, AlgebraicSignature)
    is_in_alg = isinstance(in_sig_o, AlgebraicSignature)

    # Fast structural & state unification pre-filter (O(1) cached lookup on AlgebraicSignature pair)
    already_unified = False
    if is_out_alg and is_in_alg:
        if out_sig_o.unifies_with(in_sig_o, reg):
            already_unified = True
        else:
            out_tn = out_sig_o.type_name
            if "[" not in out_tn or not out_tn.endswith("]"):
                return False
            ctor = out_tn.split("[", 1)[0].strip().lower()
            if not reg.is_product_constructor(ctor):
                return False
            if _lattice_is_path_port(p_out, reg) and _lattice_is_path_port(p_in, reg):
                return False

    if producer_cell is None:
        producer_cell = p_out.owner_cell if is_out_ps else (_get_p_field(p_out, "owner_cell") or getattr(out_sig_o, "owner_cell", None))
    if consumer_cell is None:
        consumer_cell = p_in.owner_cell if is_in_ps else (_get_p_field(p_in, "owner_cell") or getattr(in_sig_o, "owner_cell", None))

    # 1. Semantic Port Roles & Directional Gating
    if is_out_ps:
        out_role = p_out.port_role or ""
    else:
        out_role = str(_get_p_field(p_out, "port_role") or getattr(out_sig_o, "port_role", None) or getattr(out_sig_o, "derived_role", "") or "").lower()
    if is_in_ps:
        in_role = p_in.port_role or ""
    else:
        in_role = str(_get_p_field(p_in, "port_role") or getattr(in_sig_o, "port_role", None) or getattr(in_sig_o, "derived_role", "") or "").lower()

    pred_in, pred_out, pred_states, term_invalid_states, protocols = reg._get_edge_sem_bundle()

    out_state = (out_sig_o.state or "").lower() if is_out_alg else str(getattr(out_sig_o, "state", "") or _get_p_field(p_out, "state", "") or "").lower()
    in_state = (in_sig_o.state or "").lower() if is_in_alg else str(getattr(in_sig_o, "state", "") or _get_p_field(p_in, "state", "") or "").lower()

    props = reg._get_state_properties_ref(out_state)
    is_pred_state = bool(props.get("is_prediction")) or out_state in pred_states

    if in_role in pred_in and not (is_pred_state or out_role in pred_out):
        return False

    if is_pred_state and in_role and _roles_conflict(in_role, "prediction_output", reg=reg):
        return False

    if in_role and out_role and _roles_conflict(in_role, out_role, reg=reg):
        return False

    # 2. Structural Kind and Abstract Carrier Gating
    out_sk = (p_out.structure_kind or "").lower() if is_out_ps else str(_get_p_field(p_out, "structure_kind", "") or "").lower()
    in_sk = (p_in.structure_kind or "").lower() if is_in_ps else str(_get_p_field(p_in, "structure_kind", "") or "").lower()
    out_abs = (p_out.abstract_type or (out_sig_o.abstract_type if is_out_alg else "")).lower() if is_out_ps else str(_get_p_field(p_out, "abstract_type", "") or getattr(out_sig_o, "abstract_type", "") or "").lower()
    in_abs = (p_in.abstract_type or (in_sig_o.abstract_type if is_in_alg else "")).lower() if is_in_ps else str(_get_p_field(p_in, "abstract_type", "") or getattr(in_sig_o, "abstract_type", "") or "").lower()

    if not reg.are_structure_kinds_compatible(out_sk, in_sk, out_abs, in_abs):
        out_tn_v = str(getattr(out_sig_o, "type_name", "") or "").strip()
        in_tn_v = str(getattr(in_sig_o, "type_name", "") or "").strip()
        explicit_handle_subtype = bool(
            already_unified
            and out_sk == "handle"
            and in_sk == "scalar"
            and out_tn_v
            and in_tn_v
            and not reg.is_declared_top(in_tn_v)
            and not reg.is_type_variable(in_tn_v)
            and reg.is_subtype(out_tn_v, in_tn_v)
        )
        if not explicit_handle_subtype:
            return False

    # 3. Rejected Qualifiers
    out_quals = out_sig_o.qualifiers if is_out_alg else (
        _get_p_field(p_out, "qualifiers", None)
        or getattr(out_sig_o, "qualifiers", None)
        or ()
    )
    flat_out_quals: Optional[Set[str]] = None
    if is_in_ps:
        cell_pre_q = getattr(consumer_cell, "preconditions", None) if consumer_cell is not None else None
        if not p_in.rejected_qualifiers and not cell_pre_q:
            rej_quals: Any = ()
        else:
            rej_quals = _extract_rejected_qualifiers(p_in, consumer_cell=consumer_cell)
    else:
        rej_quals = _extract_rejected_qualifiers(p_in, consumer_cell=consumer_cell)

    if rej_quals and out_quals:
        flat_out_quals = _flat_quals_set(out_quals)
        if flat_out_quals & rej_quals:
            return False

    # 4. Shape Contract & Rank Compatibility
    out_sc = p_out.shape_contract if is_out_ps else (_get_p_field(p_out, "shape_contract", None) or getattr(out_sig_o, "shape_contract", None))
    in_sc = p_in.shape_contract if is_in_ps else (_get_p_field(p_in, "shape_contract", None) or getattr(in_sig_o, "shape_contract", None))

    out_poly = False
    in_poly = False
    if out_sc is not None:
        if isinstance(out_sc, dict):
            out_poly = bool(out_sc.get("rank_polymorphic"))
        elif getattr(out_sc, "rank_polymorphic", False):
            out_poly = True
    if in_sc is not None:
        if isinstance(in_sc, dict):
            in_poly = bool(in_sc.get("rank_polymorphic"))
        elif getattr(in_sc, "rank_polymorphic", False):
            in_poly = True

    if out_sc and in_sc and not (out_poly or in_poly):
        o_ndim = _extract_ndim(out_sc)
        i_ndim = _extract_ndim(in_sc)
        if o_ndim is not None and i_ndim is not None and o_ndim != i_ndim:
            return False
        o_shp = out_sc.get("shape") if isinstance(out_sc, dict) else getattr(out_sc, "shape", None)
        i_shp = in_sc.get("shape") if isinstance(in_sc, dict) else getattr(in_sc, "shape", None)
        if o_shp is not None and i_shp is not None and isinstance(o_shp, (list, tuple)) and isinstance(i_shp, (list, tuple)):
            if len(o_shp) == len(i_shp):
                for d1, d2 in zip(o_shp, i_shp):
                    if isinstance(d1, int) and isinstance(d2, int) and d1 != d2:
                        return False
        o_dict = out_sc if isinstance(out_sc, dict) else getattr(out_sc, "__dict__", {})
        i_dict = in_sc if isinstance(in_sc, dict) else getattr(in_sc, "__dict__", {})
        for k, v_in in i_dict.items():
            if k in ("rank_polymorphic", "shape", "ndim"):
                continue
            if k in o_dict and o_dict[k] is not None and v_in is not None:
                if o_dict[k] != v_in:
                    return False

    is_poly_out = reg.is_declared_top_state(out_state)
    is_poly_in = reg.is_declared_top_state(in_state)
    if is_in_ps and not is_poly_in:
        in_sem_v = str(getattr(p_in, "semantics", "") or "").strip().lower()
        in_lay_v = getattr(p_in, "layout", None)
        if in_sem_v == "any" and (not isinstance(in_lay_v, dict) or in_lay_v.get("rank") is None or in_lay_v.get("rank_polymorphic")):
            is_poly_in = True

    r_out = p_out.canonical_rank if is_out_ps else (_get_p_field(p_out, "canonical_rank", None) or getattr(out_sig_o, "canonical_rank", None) or _get_p_field(p_out, "rank", None))
    r_in = p_in.canonical_rank if is_in_ps else (_get_p_field(p_in, "canonical_rank", None) or getattr(in_sig_o, "canonical_rank", None) or _get_p_field(p_in, "rank", None))
    has_r_out = True

    if in_sc is not None and not in_poly and not out_poly and not is_poly_in:
        i_ndim = _extract_ndim(in_sc)
        o_ndim = _extract_ndim(out_sc) if out_sc is not None else None
        o_eff = o_ndim if o_ndim is not None else r_out
        if i_ndim is not None and o_eff is not None and o_eff != i_ndim:
            return False

    acc_in = (p_in.accepted_states or (in_sig_o.accepted_states if is_in_alg else ())) if is_in_ps else (_get_p_field(p_in, "accepted_states", None) or getattr(in_sig_o, "accepted_states", None) or ())
    if not is_poly_in and out_state and acc_in:
        if out_state in acc_in or reg.state_ancestry_reaches(out_state, acc_in if isinstance(acc_in, (set, tuple)) else tuple(str(a).strip().lower() for a in acc_in)):
            is_poly_in = True

    if not (is_poly_out or is_poly_in or out_poly or in_poly):
        if r_out is not None and r_in is not None and r_out != r_in:
            return False

    # 5. State Poset Transition Compatibility
    is_out_path = _lattice_is_path_port(p_out)
    is_in_path = _lattice_is_path_port(p_in)
    both_path = is_out_path and is_in_path
    is_path_compatible = both_path and (out_state not in term_invalid_states)

    if not is_path_compatible:
        can_reuse_state = (
            already_unified
            and not both_path
            and is_out_ps
            and is_in_ps
            and p_out.accepted_states == out_sig_o.accepted_states
            and p_in.accepted_states == in_sig_o.accepted_states
            and p_out.parent_state == out_sig_o.parent_state
            and p_in.parent_state == in_sig_o.parent_state
        )
        if not can_reuse_state:
            producer_accepted = p_out.accepted_states if is_out_ps else (_get_p_field(p_out, "accepted_states", None) or getattr(out_sig_o, "accepted_states", None) or ())
            consumer_accepted = acc_in
            consumer_parent = p_in.parent_state if is_in_ps else (getattr(in_sig_o, "parent_state", None) or _get_p_field(p_in, "parent_state", None))
            producer_parent = p_out.parent_state if is_out_ps else (getattr(out_sig_o, "parent_state", None) or _get_p_field(p_out, "parent_state", None))
            p_dom = str(getattr(producer_cell, "domain_name", "") or (p_out.domain if is_out_ps else _get_p_field(p_out, "domain", "")) or "").strip().lower() or None
            c_dom = str(getattr(consumer_cell, "domain_name", "") or (p_in.domain if is_in_ps else _get_p_field(p_in, "domain", "")) or "").strip().lower() or None
            if not reg.is_state_compatible(
                producer_state=out_state,
                consumer_state=in_state,
                producer_accepted=producer_accepted,
                consumer_accepted=consumer_accepted,
                consumer_parent=consumer_parent,
                producer_parent=producer_parent,
                producer_domain=p_dom,
                consumer_domain=c_dom,
            ):
                return False

    # 5b. Nominal Class / Protocol Identity Gate (e.g. Lasso.fit -> Ridge.predict mismatch)
    p_nom = str((p_out.nominal_type if is_out_ps else _get_p_field(p_out, "nominal_type", None)) or "").strip().lower()
    c_nom = str((p_in.nominal_type if is_in_ps else _get_p_field(p_in, "nominal_type", None)) or "").strip().lower()
    if p_nom and c_nom and p_nom != c_nom and not reg.is_subtype(p_nom, c_nom):
        return False

    # 6. Precondition Verification Against Producer Properties and Postconditions
    c_pre = (getattr(consumer_cell, "preconditions", None) or ()) if consumer_cell is not None else ()
    p_pre = () if is_in_ps else (_get_p_field(p_in, "preconditions", None) or ())
    if c_pre or p_pre:
        preconds = list(c_pre)
        if p_pre:
            preconds.extend(p_pre)
        p_name = p_in.name if is_in_ps else _get_p_field(p_in, "name", None)

        if not has_r_out:
            r_out = p_out.canonical_rank if is_out_ps else (_get_p_field(p_out, "canonical_rank", None) or getattr(out_sig_o, "canonical_rank", None) or _get_p_field(p_out, "rank", None))
        if flat_out_quals is None:
            flat_out_quals = _flat_quals_set(out_quals)

        producer_props: Dict[str, Any] = {}
        if out_sc:
            o_dict = out_sc if isinstance(out_sc, dict) else getattr(out_sc, "__dict__", {})
            for pk, pv in o_dict.items():
                if pk != "rank_polymorphic":
                    producer_props[pk] = pv
            o_ndim = _extract_ndim(out_sc)
            if o_ndim is not None:
                producer_props["ndim"] = o_ndim
            o_shp = out_sc.get("shape") if isinstance(out_sc, dict) else getattr(out_sc, "shape", None)
            if o_shp is not None:
                producer_props["shape"] = o_shp
        if r_out is not None and "ndim" not in producer_props:
            producer_props["ndim"] = r_out
        if out_state and out_state not in ("any", "*"):
            producer_props["state"] = out_state
        for q in flat_out_quals:
            producer_props.setdefault(f"is_{q}", True)
            producer_props.setdefault(f"has_{q}", True)
            producer_props.setdefault(q, True)

        out_p_name = p_out.name if is_out_ps else _get_p_field(p_out, "name", None)
        prim_out_name = getattr(producer_cell, "primary_out", None) if producer_cell is not None else None
        is_single_out = bool(producer_cell is not None and len(getattr(producer_cell, "outputs", {}) or {}) == 1)
        if producer_cell is not None:
            for post in (getattr(producer_cell, "postconditions", None) or ()):
                if not isinstance(post, dict):
                    continue
                post_target = post.get("target")
                if (
                    post_target is None
                    or post_target == out_p_name
                    or ((is_single_out or not out_p_name) and prim_out_name and post_target == prim_out_name)
                    or (post_target == "output_var" and is_single_out)
                ):
                    p_prop = post.get("property")
                    if p_prop:
                        producer_props[p_prop] = post.get("value")

        for pre in preconds:
            if not isinstance(pre, dict):
                continue
            target = pre.get("target")
            if target and p_name and target != p_name:
                continue
            prop = pre.get("property")
            op = pre.get("operator", "==")
            want = pre.get("value")

            if prop in producer_props:
                have = producer_props[prop]
                if op == "==" and have != want:
                    return False
                elif op == "!=" and have == want:
                    return False
                elif op == ">=" and not (have >= want):
                    return False
                elif op == "<=" and not (have <= want):
                    return False
                elif op == ">" and not (have > want):
                    return False
                elif op == "<" and not (have < want):
                    return False

    # 7. Base Structural Type Subtyping and Robinson Unification
    if both_path:
        raw_in_fmts = p_in.supported_formats if is_in_ps else (_get_p_field(p_in, "supported_formats", None) or ())
        raw_out_fmts = p_out.supported_formats if is_out_ps else (_get_p_field(p_out, "supported_formats", None) or ())
        if raw_in_fmts and raw_out_fmts:
            in_fmts = {str(f).lower().lstrip(".") for f in raw_in_fmts if f != "*"} - protocols
            out_fmts = {str(f).lower().lstrip(".") for f in raw_out_fmts if f != "*"} - protocols
            if in_fmts and out_fmts and not (in_fmts & out_fmts):
                return False
        if not already_unified and hasattr(out_sig_o, "unifies_with") and not out_sig_o.unifies_with(in_sig_o):
            return False
    elif not already_unified:
        if hasattr(out_sig_o, "unifies_with"):
            if not out_sig_o.unifies_with(in_sig_o):
                out_tn = str(getattr(out_sig_o, "type_name", "") or "")
                if "[" in out_tn and out_tn.endswith("]"):
                    ctor = out_tn.split("[", 1)[0].strip().lower()
                    if reg.is_product_constructor(ctor):
                        inner = out_tn[out_tn.index("[") + 1 : -1]
                        members = list(split_bracketed_args(inner))
                        matched_member = False
                        for m_str in members:
                            try:
                                m_term = TypeTerm.from_string(m_str)
                                in_term = in_sig_o if isinstance(in_sig_o, TypeTerm) else _to_type_term(in_sig_o)
                                if unify(m_term, in_term) is not None:
                                    matched_member = True
                                    break
                            except Exception:
                                continue
                        if not matched_member:
                            return False
                    else:
                        return False
                else:
                    return False
        elif hasattr(in_sig_o, "unifies_with"):
            if not in_sig_o.unifies_with(out_sig_o):
                return False
        else:
            u = unify(out_sig_o, in_sig_o)
            if u is None:
                return False

    if is_out_ps and is_in_ps:
        res = compute_port_residual(
            p_out, p_in, registry=reg, producer_cell=producer_cell, consumer_cell=consumer_cell
        )
        if res.nominal_conflict or res.layout_delta or res.semantics_delta or res.invariants_delta:
            return False

    return True


def _shape_compatible(p_out: Any, p_in: Any, producer_cell: Optional[Any] = None, consumer_cell: Optional[Any] = None) -> bool:
    return is_edge_valid(p_out, p_in, producer_cell=producer_cell, consumer_cell=consumer_cell)


@functools.lru_cache(maxsize=4096)
@functools.lru_cache(maxsize=2048)
def _extract_lhs_template_placeholders_cached(tpl: str) -> Tuple[str, ...]:
    """Cached tuple of placeholder names bound on the LHS of assignments in code_template."""
    if not tpl or "=" not in tpl:
        return ()
    phs = extract_template_placeholders(tpl)
    if not phs:
        return ()
    ast_ready = tpl
    for ph in phs:
        ast_ready = ast_ready.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
    parsed, _ = _cached_ast_parse_ro(ast_ready, "exec")
    if parsed is not None:
        lhs_phs: Set[str] = set()
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for t_node in ast.walk(target):
                        if isinstance(t_node, ast.Name) and t_node.id.startswith("_nstl_ph_"):
                            lhs_phs.add(t_node.id[len("_nstl_ph_"):])
        if lhs_phs:
            return tuple(sorted(lhs_phs))

    depth = 0
    in_quote = None
    eq_idx = None
    for i, ch in enumerate(tpl):
        if ch in ("'", '"'):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
        elif in_quote is None:
            if ch in ("(", "[", "{"):
                depth += 1
            elif ch in (")", "]", "}"):
                depth -= 1
            elif ch == "=" and depth == 0:
                prev_ch = tpl[i - 1] if i > 0 else ""
                next_ch = tpl[i + 1] if i + 1 < len(tpl) else ""
                if prev_ch not in ("=", "!", "<", ">") and next_ch != "=":
                    eq_idx = i
                    break
    if eq_idx is not None:
        lhs = tpl[:eq_idx]
        return tuple(sorted(set(extract_template_placeholders(lhs))))
    return ()


def _extract_lhs_template_placeholders(tpl: str) -> Set[str]:
    """Extracts placeholder names bound on the LHS of assignments in code_template."""
    return set(_extract_lhs_template_placeholders_cached(tpl or ""))


def _has_data_bearing_output_assignment(cell: Any) -> bool:
    """True iff cell's code_template assigns to a data-bearing output variable on LHS of '='."""
    if cell is None:
        return False
    outputs = getattr(cell, "outputs", None)
    if not outputs:
        return False

    try:
        reg = TypeRegistry.get_instance()
        mat_states = (
            set(reg.get_terminal_states() or ())
            | reg.get_states_by_property("is_materialized", True)
            | reg.get_states_by_property("is_terminal", True)
        )
    except Exception:
        reg = None
        mat_states = set()

    candidate_ports = []
    prim_out = getattr(cell, "primary_output", None)
    if prim_out is not None:
        candidate_ports.append(prim_out)
    if isinstance(outputs, dict):
        candidate_ports.extend(outputs.values())
    elif isinstance(outputs, (list, tuple, set)):
        candidate_ports.extend(outputs)

    data_bearing_names: Set[str] = set()
    void_types = reg.get_void_types_set() if reg else set()
    for p in candidate_ports:
        if p is None:
            continue
        p_name = getattr(p, "name", None)
        sig = getattr(p, "signature", p)
        t_name = str(getattr(sig, "type_name", "") or "").strip().lower()
        state = str(getattr(sig, "state", "") or "").strip().lower()
        if not t_name or t_name in void_types or (reg and reg.is_declared_bottom(t_name)):
            continue
        if state in mat_states:
            continue
        if p_name:
            data_bearing_names.add(p_name)

    if not data_bearing_names:
        return False

    tpl = getattr(cell, "code_template", "") or ""
    lhs_phs = _extract_lhs_template_placeholders(tpl)
    if not lhs_phs:
        return False

    return bool(lhs_phs & (data_bearing_names | {"output_var"}))


_SINK_CELL_CACHE: Dict[Tuple[str, int], bool] = {}
_SINK_CELL_LOCK = threading.Lock()
_MAX_SINK_CACHE_SIZE = 4096


def _is_sink_cell(cell: Any) -> bool:
    """Returns True if cell represents a void/materialization sink morphism with no outgoing dataflow."""
    if cell is None:
        return False
    try:
        from lattice import LatticeOrchestrator
        orch = LatticeOrchestrator.get_active_instance()
        version = getattr(orch, "_version", 0) if orch else 0
    except Exception:
        version = 0
    c_id = getattr(cell, "cell_id", None) or str(id(cell))
    cache_key = (c_id, version)
    with _SINK_CELL_LOCK:
        if cache_key in _SINK_CELL_CACHE:
            return _SINK_CELL_CACHE[cache_key]

    # A cell whose code_template assigns to {output_var} or {<primary_output.name>}
    # on the LHS of '=' and whose output type is a concrete data carrier is a
    # value-producing morphism (even if is_endable is True), NOT a sink.
    if _has_data_bearing_output_assignment(cell):
        res = False
    elif getattr(cell, "is_endable", False):
        res = True
    else:
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        term_roles = TypeRegistry.get_instance().get_terminal_roles_set()
        if role in term_roles:
            res = True
        else:
            outputs = getattr(cell, "outputs", None)
            res = not bool(outputs)

    with _SINK_CELL_LOCK:
        if len(_SINK_CELL_CACHE) >= _MAX_SINK_CACHE_SIZE:
            _SINK_CELL_CACHE.pop(next(iter(_SINK_CELL_CACHE)), None)
        _SINK_CELL_CACHE[cache_key] = res
    return res



def _is_string_carrier(port_sig: Any, concrete_sig: Any = None) -> bool:
    """Structural test: does this port carry a textual literal (str ⊑, path port, or enum)?
    Callable/statement-block slots are excluded even when stored under str type.
    Purely poset/port-driven; no type-name vocabulary."""
    reg = TypeRegistry.get_instance()
    for s in (concrete_sig, port_sig):
        if s is None:
            continue
        sig = getattr(s, "signature", s)
        s_abs = str(getattr(s, "abstract_type", "") or getattr(sig, "abstract_type", "") or "").strip().lower()
        if s_abs == "callable":
            return False
    for s in (concrete_sig, port_sig):
        if s is None:
            continue
        sig = getattr(s, "signature", s)
        t = str(getattr(sig, "type_name", "") or "").strip()
        if t and reg.is_subtype(t, "str"):
            return True
        if getattr(s, "enum_values", None):
            return True
        if _lattice_is_path_port(s):
            return True
    return False


def _pointed_default(port_sig: Any, cell: Any = None) -> Optional[str]:
    """Pointed monoidal default eta_tau : 1 -> tau for a semi-closed scalar parameter.

    Strictly eligible ONLY for optional / default-declared rank-0 scalar/enum
    configuration ports; required ports without a declared `default_value`,
    dataflow carriers, composites, collections/mappings, paths, key/projection ports,
    and constrained/optic input ports are never fabricated."""
    if getattr(port_sig, "required", True) and getattr(port_sig, "default_value", None) is None:
        return None
    reg = TypeRegistry.get_instance()
    sig = getattr(port_sig, "signature", port_sig)
    t = str(getattr(sig, "type_name", "") or "").strip()
    t_l = t.lower()
    if not t or "[" in t or reg.is_type_variable(t) or reg.is_declared_top(t_l) or reg.is_void_type(t_l):
        return None
    if reg.is_declared_bottom(t_l):
        return None
    sk = str(getattr(port_sig, "structure_kind", "") or "").strip().lower()
    rank = getattr(port_sig, "canonical_rank", None)
    if rank is None:
        rank = getattr(port_sig, "rank", None)
    if sk and sk != "scalar":
        return None
    if rank not in (None, 0):
        return None
    if reg.is_composite_carrier(t_l) or reg.is_abstract_carrier(t_l) or _lattice_is_path_port(port_sig):
        return None
    _coll_roots = reg.get_composite_carrier_roots()
    if any(reg.is_subtype(t_l, root) for root in _coll_roots) or t_l in _coll_roots:
        return None
    if (
        getattr(port_sig, "complement_of_port", None)
        or getattr(port_sig, "disjoint_from_port", None)
        or getattr(port_sig, "key_from_upstream_role", None)
    ):
        return None
    if cell is not None and getattr(cell, "optic_role", None):
        for attr_ref in ("container_port", "key_port", "focus_port"):
            if _port_matches_declared_ref(cell, attr_ref, port_sig):
                return None

    p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "role", "") or "").strip().lower()
    p_binds = str(getattr(port_sig, "binds", "") or "").strip().lower()
    p_kind = str(getattr(port_sig, "param_kind", "") or "").strip().lower()
    p_state = str(getattr(sig, "state", "") or "").strip().lower()

    cfg_roles = reg.get_config_roles_set()
    non_cfg_roles = (
        reg.get_roles_for_category("data")
        | reg.get_data_bearing_roles()
        | reg.get_roles_for_category("strict")
        | reg.get_roles_for_category("estimator")
        | reg.get_roles_for_category("projection")
        | set(reg.get_projection_tokens() or ())
        | reg.get_roles_for_category("feature")
        | reg.get_roles_for_category("target")
        | reg.get_roles_for_category("key")
        | reg.get_roles_for_category("assigned_value")
        | reg.get_roles_for_category("receiver")
        | reg.get_terminal_roles_set()
        | reg.get_roles_for_category("destination")
        | reg.get_ingress_roles_set()
        | reg.get_roles_for_category("path")
        | reg.get_roles_for_category("instance_sink")
        | reg.get_roles_for_category("owner_sink")
        | (reg.get_declared_role_carriers() - cfg_roles)
    )
    if p_role in non_cfg_roles or p_binds in non_cfg_roles:
        return None
    non_cfg_states = (
        reg.get_states_by_property("is_condition", True)
        | reg.get_states_by_property("is_projection", True)
        | reg.get_states_by_property("is_target_identifier", True)
        | reg.get_states_by_property("is_source_identifier", True)
        | reg.get_states_by_property("is_materialized", True)
        | reg.get_states_by_property("is_terminal", True)
        | set(reg.get_terminal_states() or ())
    )
    if p_state in non_cfg_states:
        return None

    enum_vals = getattr(port_sig, "enum_values", None)
    if enum_vals:
        return repr(list(enum_vals)[0])

    is_num_or_bool = reg.is_subtype(t_l, "numeric") or reg.is_subtype(t_l, "bool")
    is_str_sub = reg.is_subtype(t_l, "str")
    is_scalar_sub = reg.is_subtype(t_l, "scalar")
    if not (is_num_or_bool or is_str_sub or is_scalar_sub):
        return None
    if not is_num_or_bool and (p_role not in cfg_roles and p_kind not in cfg_roles):
        return None

    domain = getattr(cell, "domain_name", None) or getattr(port_sig, "domain", None)
    val = reg.get_type_default(t, domain)
    if val is None:
        return None
    vc = getattr(port_sig, "value_constraints", None)
    if isinstance(vc, dict) and vc:
        num_v: Optional[float] = None
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            num_v = float(val)
        elif isinstance(val, str) and not reg.is_subtype(t_l, "bool"):
            try:
                num_v = float(val.strip().strip("'\""))
            except (ValueError, TypeError):
                num_v = None
        if num_v is not None:
            for k_min in ("min", "minimum"):
                if vc.get(k_min) is not None:
                    try:
                        if num_v < float(vc[k_min]):
                            return None
                    except (ValueError, TypeError):
                        pass
            for k_emin in ("exclusive_min", "exclusiveMinimum", "gt"):
                if vc.get(k_emin) is not None:
                    try:
                        if num_v <= float(vc[k_emin]):
                            return None
                    except (ValueError, TypeError):
                        pass
            for k_max in ("max", "maximum"):
                if vc.get(k_max) is not None:
                    try:
                        if num_v > float(vc[k_max]):
                            return None
                    except (ValueError, TypeError):
                        pass
            for k_emax in ("exclusive_max", "exclusiveMaximum", "lt"):
                if vc.get(k_emax) is not None:
                    try:
                        if num_v >= float(vc[k_emax]):
                            return None
                    except (ValueError, TypeError):
                        pass
            if vc.get("non_zero") and num_v == 0.0:
                return None
    if isinstance(val, str):
        if val.strip("'\"") == "":
            return None
        return val
    return repr(val)


def resolve_synthetic_boundary_io(
    p_sig: Any,
    concrete_sig: Any = None,
    cell: Any = None,
    registry: Optional[TypeRegistry] = None,
    ctx: Optional["ExecutionContext"] = None,
) -> Optional[str]:
    """Schema-driven boundary I/O path resolution (Tier 2).

    When `strict_io` is False and a required filesystem path port has no explicit
    filename in the prompt, synthesizes a deterministic `input_n.<ext>` or
    `output_n.<ext>` literal from declared `supported_formats` (ignoring wildcards)
    or declared `asset_placeholders` / `output_asset_placeholders`, with direction
    derived solely from declared port/cell roles and typestates.
    """
    reg = registry or TypeRegistry.get_instance()
    if ctx is not None and getattr(ctx, "strict_io", None) is not None:
        strict_flag = bool(ctx.strict_io)
    else:
        strict_flag = True
        try:
            try:
                from .config import get_settings
            except (ImportError, ValueError):
                from config import get_settings
            strict_flag = bool(getattr(get_settings(), "strict_io", True))
        except Exception:
            strict_flag = True
    if strict_flag:
        return None

    if not (_lattice_is_path_port(p_sig, reg) or _lattice_is_path_port(concrete_sig, reg)):
        return None

    p_name = str(getattr(p_sig, "name", "") or getattr(concrete_sig, "name", "") or "path")
    cell_id = str(getattr(cell, "cell_id", "") or "cell")
    step_idx = getattr(ctx, "current_cell_step_idx", None) if ctx is not None else None
    cache_key = (f"{cell_id}@{step_idx}" if step_idx is not None else cell_id, p_name)
    if ctx is not None and cache_key in getattr(ctx, "synthetic_io_placeholders", {}):
        return ctx.synthetic_io_placeholders[cache_key]

    p_role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
    inner_sig = getattr(concrete_sig or p_sig, "signature", concrete_sig or p_sig)
    p_state = str(getattr(inner_sig, "state", "") or "").strip().lower()

    dest_roles = (
        reg.get_roles_for_category("destination")
        | reg.get_terminal_roles_set()
        | reg.get_roles_for_category("target")
    )
    src_roles = reg.get_ingress_roles_set()
    dest_states = (
        set(reg.get_terminal_states() or ())
        | reg.get_states_by_property("is_materialized", True)
        | reg.get_states_by_property("is_terminal", True)
        | reg.get_states_by_property("is_target_identifier", True)
    )
    src_states = reg.get_states_by_property("is_source_identifier", True)

    if p_role in src_roles or p_state in src_states:
        is_output = False
    elif p_role in dest_roles or p_state in dest_states:
        is_output = True
    else:
        is_output = bool(cell is not None and _is_sink_cell(cell))

    wildcards = reg.get_declared_top_states() | {"*", "any", "all", ""}
    ext: Optional[str] = None
    for sig_obj in (p_sig, concrete_sig):
        if sig_obj is None:
            continue
        fmts = getattr(sig_obj, "supported_formats", None) or ()
        for f in fmts:
            clean = str(f).strip().lower().lstrip(".")
            if clean and clean not in wildcards and clean.replace("_", "").isalnum():
                ext = clean
                break
        if ext:
            break

    result_lit: Optional[str] = None
    if ext:
        prefix = "output" if is_output else "input"
        if ctx is not None:
            idx = ctx._synthetic_io_counts.get(prefix, 0) + 1
            ctx._synthetic_io_counts[prefix] = idx
        else:
            idx = 1
        result_lit = repr(f"{prefix}_{idx}.{ext}")
    else:
        domain = str(getattr(cell, "domain_name", "") or getattr(p_sig, "domain", "") or "").strip().lower()
        ph = (
            reg.get_output_asset_placeholder(domain)
            if is_output
            else reg.get_asset_placeholder(domain)
        )
        if ph and str(ph).strip().lower() not in wildcards:
            result_lit = repr(str(ph).strip())

    if result_lit and ctx is not None:
        ctx.synthetic_io_placeholders[cache_key] = result_lit
        ctx.trace_binding(
            "synthetic_boundary_io",
            cell=cell_id,
            port=p_name,
            value=result_lit,
            direction=("output" if is_output else "input"),
        )
    return result_lit


class _TracedBindings(dict):
    """cell_bindings that records WHICH code site bound each port (for --debug).
    Purely observational: behaves exactly like a dict, while maintaining cell.typed_bindings
    as the live, authoritative TypedBindingRecord source of truth."""
    def __init__(self, ctx: Any, cell: Any):
        super().__init__()
        self._ctx, self._cell = ctx, cell
        if not hasattr(self._cell, "typed_bindings") or self._cell.typed_bindings is None:
            self._cell.typed_bindings = {}

    def __setitem__(self, port, value):
        super().__setitem__(port, value)
        try:
            import sys as _sys
            f = _sys._getframe(1)
            self._ctx.trace_binding(
                "bind", cell=getattr(self._cell, "cell_id", "?"), port=str(port), value=str(value),
                site=f"{f.f_code.co_name}:{f.f_lineno}",
                clause=getattr(self._cell, "matched_clause_idx", None))
        except Exception as _exc:
            logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
        try:
            p_name = str(port)
            p_sig = None
            if hasattr(self._cell, "inputs") and self._cell.inputs and p_name in self._cell.inputs:
                p_sig = getattr(self._cell.inputs[p_name], "signature", self._cell.inputs[p_name])
            elif hasattr(self._cell, "outputs") and self._cell.outputs and p_name in self._cell.outputs:
                p_sig = getattr(self._cell.outputs[p_name], "signature", self._cell.outputs[p_name])
            elif p_name == "output_var" and getattr(self._cell, "primary_output", None):
                p_sig = getattr(self._cell.primary_output, "signature", self._cell.primary_output)

            var_sigs = {}
            if hasattr(self._ctx, "variables") and isinstance(self._ctx.variables, dict):
                var_sigs = {v: s[0] if isinstance(s, tuple) else s for v, s in self._ctx.variables.items()}

            rec = resolve_typed_binding_record(
                cell=self._cell,
                port_name=p_name,
                bound_val=value,
                port_sig=p_sig,
                var_signatures=var_sigs,
            )
            if not hasattr(self._cell, "typed_bindings") or self._cell.typed_bindings is None:
                self._cell.typed_bindings = {}
            self._cell.typed_bindings[p_name] = rec
        except Exception as _tb_exc:
            logger.debug("[UNIFICATION] Live typed binding recording failed for %s: %s", port, _tb_exc)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *args):
        p_name = str(key)
        if hasattr(self._cell, "typed_bindings") and isinstance(self._cell.typed_bindings, dict):
            self._cell.typed_bindings.pop(p_name, None)
        return super().pop(key, *args)

    def __delitem__(self, key):
        p_name = str(key)
        if hasattr(self._cell, "typed_bindings") and isinstance(self._cell.typed_bindings, dict):
            self._cell.typed_bindings.pop(p_name, None)
        super().__delitem__(key)

class _DynamicTopTypeSet(collections.abc.Set):
    """Dynamic top type set backed by TypeRegistry declarations."""
    _from_iterable = staticmethod(set)
    def __contains__(self, item):
        return TypeRegistry.get_instance().is_declared_top(str(item))
    def __iter__(self):
        return iter(TypeRegistry.get_instance()._declared_top)
    def __len__(self):
        return len(TypeRegistry.get_instance()._declared_top)

TOP_TYPE_SET = _DynamicTopTypeSet()


# =====================================================================
# 1. Type Terms and Substitutions
# =====================================================================

class TypeTerm(ABC):
    """Abstract base class for formal type terms."""
    @property
    @abstractmethod
    def base(self) -> str:
        """Return lowercase base constructor/type name."""
        pass

    @abstractmethod
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        pass

    @classmethod
    def from_string(cls, s: str) -> 'TypeTerm':
        return parse_type_term(s)


@dataclass(frozen=True, slots=True)
class TopType(TypeTerm):
    """Universal Top type (wildcard) unifying with any type term."""
    @property
    def base(self) -> str:
        return "any"

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return "Top"


TOP = TopType()


@dataclass(frozen=True, slots=True)
class AtomicType(TypeTerm):
    """Ground type constant."""
    name: str

    @property
    def base(self) -> str:
        return self.name.strip().lower()

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class GenericTypeTerm(TypeTerm):
    """
    Parametric / Generic type constructor: C[T_1, ..., T_n].
    Conforms to Category of Generic Monads and Functors.
    """
    constructor: str
    args: Tuple[TypeTerm, ...]

    @property
    def base(self) -> str:
        return self.constructor.strip().lower()

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        resolved_args = tuple(a.apply_substitution(sigma, visited) for a in self.args)
        return GenericTypeTerm(self.constructor, resolved_args)

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.constructor}[{args_str}]"


@dataclass(frozen=True, slots=True)
class UnionTypeTerm(TypeTerm):
    """
    Coproduct / Sum / Union type term: T_1 | ... | T_n.
    Conforms to Category of Coproducts with canonical injection morphisms.
    """
    terms: Tuple[TypeTerm, ...]

    @property
    def base(self) -> str:
        return self.terms[0].base if self.terms else "any"

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        resolved_terms = tuple(t.apply_substitution(sigma, visited) for t in self.terms)
        return UnionTypeTerm(resolved_terms)

    def __repr__(self) -> str:
        return " | ".join(str(t) for t in self.terms)


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    """Type variable alpha, beta... subject to substitution."""
    var_name: str

    @property
    def base(self) -> str:
        return self.var_name.strip().lower()

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        if visited and self.var_name in visited:
            return self
        if self.var_name in sigma.mappings:
            target = sigma.mappings[self.var_name]
            new_visited = (visited or ()) + (self.var_name,)
            if isinstance(target, TypeTerm):
                return target.apply_substitution(sigma, new_visited)
            target_str = str(target).strip()
            if target_str == self.var_name or target_str == f"?{self.var_name}":
                return self
            parsed = TypeTerm.from_string(target_str)
            if isinstance(parsed, TypeVariable) and parsed.var_name == self.var_name:
                return self
            return parsed.apply_substitution(sigma, new_visited)
        return self

    def __repr__(self) -> str:
        return f"?{self.var_name}"


@functools.lru_cache(maxsize=4096)
def split_bracketed_args(inner: str) -> Tuple[str, ...]:
    """Splits comma-separated arguments respecting nested square brackets."""
    args = []
    depth = 0
    curr = []
    for ch in inner:
        if ch == "[":
            depth += 1
            curr.append(ch)
        elif ch == "]":
            depth -= 1
            curr.append(ch)
        elif ch == "," and depth == 0:
            arg = "".join(curr).strip()
            if arg:
                args.append(arg)
            curr = []
        else:
            curr.append(ch)
    if curr:
        arg = "".join(curr).strip()
        if arg:
            args.append(arg)
    return tuple(args)


def _split_top_level_union(s: str) -> List[str]:
    """Splits a type expression on `|` ONLY at bracket depth 0 across ()[]{}.

    Handles both spaced (`A | B`) and unspaced (`int|float`) unions without
    splitting nested generic unions like `List[int | str]`.
    """
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            if depth > 0:
                depth -= 1
            cur.append(ch)
        elif ch == "|" and depth == 0:
            seg = "".join(cur).strip()
            if seg:
                parts.append(seg)
            cur = []
        else:
            cur.append(ch)
    if cur:
        seg = "".join(cur).strip()
        if seg:
            parts.append(seg)
    return parts


@functools.lru_cache(maxsize=4096)
def _parse_type_term_cached(s_clean: str, gen: int) -> TypeTerm:
    """Generation-keyed cached structural TypeTerm parser."""
    registry = TypeRegistry.get_instance()
    if is_top_symbol(s_clean, registry):
        return TOP
    if s_clean.startswith("?"):
        return TypeVariable(s_clean[1:])
    # Top-level Coproduct / Sum / Union types (A | B or int|float) respecting bracket depth
    union_parts = _split_top_level_union(s_clean)
    if len(union_parts) > 1:
        return UnionTypeTerm(tuple(_parse_type_term_cached(p, gen) for p in union_parts))
    for _u_ctor in get_active_language_backend().union_type_constructors:
        _u_pfx = f"{_u_ctor}["
        if s_clean.startswith(_u_pfx) and s_clean.endswith("]"):
            inner = s_clean[len(_u_pfx):-1].strip()
            args = split_bracketed_args(inner)
            return UnionTypeTerm(tuple(_parse_type_term_cached(a, gen) for a in args if a))
    if registry.is_type_variable(s_clean):
        return TypeVariable(s_clean)
    if "[" in s_clean and s_clean.endswith("]"):
        bracket_idx = s_clean.index("[")
        constructor = s_clean[:bracket_idx].strip()
        inner = s_clean[bracket_idx + 1 : -1].strip()
        args = split_bracketed_args(inner)
        parsed_args = tuple(_parse_type_term_cached(a, gen) for a in args if a)
        if constructor == "":
            return GenericTypeTerm("", parsed_args)
        if parsed_args:
            return GenericTypeTerm(constructor, parsed_args)
    return AtomicType(s_clean)


def parse_type_term(type_str: str) -> TypeTerm:
    """Parses a type string into a structured TypeTerm, invalidated per TypeRegistry._generation."""
    s_clean = str(type_str).strip()
    gen = TypeRegistry.get_instance()._generation
    return _parse_type_term_cached(s_clean, gen)


@dataclass(frozen=True, slots=True)
class TypestateTerm(TypeTerm):
    """Typestate compound term: tau = (type_name, state, qualifiers, abstract_type, accepted_states, parent_state, domain)."""
    type_name: str
    state: str = "any"
    qualifiers: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    abstract_type: Optional[str] = None
    accepted_states: Tuple[str, ...] = field(default_factory=tuple)
    parent_state: Optional[str] = None
    domain: str = ""

    @property
    def base(self) -> str:
        return self.type_name.strip().lower()

    def __post_init__(self):
        raw = self.abstract_type
        if raw and str(raw).strip().lower() not in ("none", "null", ""):
            object.__setattr__(self, "abstract_type", str(raw).strip())
        else:
            object.__setattr__(self, "abstract_type", None)
        if self.qualifiers is not None:
            norm_q = _normalize_quals(self.qualifiers)
            if norm_q != self.qualifiers:
                object.__setattr__(self, "qualifiers", norm_q)
        else:
            object.__setattr__(self, "qualifiers", ())
        if self.accepted_states:
            acc_norm = tuple(sorted({str(s).strip().lower() for s in self.accepted_states if str(s).strip()}))
            if acc_norm != self.accepted_states:
                object.__setattr__(self, "accepted_states", acc_norm)
        elif not isinstance(self.accepted_states, tuple):
            object.__setattr__(self, "accepted_states", ())
        if self.parent_state:
            object.__setattr__(self, "parent_state", str(self.parent_state).strip().lower())
        dom_norm = str(self.domain or "").strip().lower()
        if dom_norm != self.domain:
            object.__setattr__(self, "domain", dom_norm)

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        if not sigma or not sigma.mappings:
            return self
        if visited and self.type_name in visited:
            return self
        if "[" in self.type_name or "|" in self.type_name:
            inner = parse_type_term(self.type_name)
            sub_inner = inner.apply_substitution(sigma, visited)
            if sub_inner != inner:
                return TypestateTerm(
                    type_name=str(sub_inner),
                    state=self.state,
                    qualifiers=self.qualifiers,
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state,
                    domain=self.domain,
                )
            return self
        if self.type_name not in sigma.mappings:
            return self
        if not TypeRegistry.get_instance().is_type_variable(self.type_name):
            return self
        val = sigma.mappings[self.type_name]
        if isinstance(val, AtomicType):
            t_resolved = val.name
        elif isinstance(val, TypeTerm):
            new_visited = (visited or ()) + (self.type_name,)
            t_resolved = str(val.apply_substitution(sigma, new_visited))
        else:
            t_resolved = str(val)
        return TypestateTerm(
            type_name=t_resolved,
            state=self.state,
            qualifiers=self.qualifiers,
            abstract_type=self.abstract_type,
            accepted_states=self.accepted_states,
            parent_state=self.parent_state,
            domain=self.domain,
        )

    def __repr__(self) -> str:
        abs_str = f", abs={self.abstract_type}" if self.abstract_type else ""
        acc_str = f", acc={list(self.accepted_states)}" if self.accepted_states else ""
        p_str = f", parent={self.parent_state}" if self.parent_state else ""
        return f"{self.type_name}[{self.state}{abs_str}{acc_str}{p_str}]"


class Substitution:
    """Mapping of type-variable identifiers to resolved TypeTerms (idempotent MGU)."""
    def __init__(self, mappings: Optional[Dict[str, Any]] = None):
        self.mappings: Dict[str, Any] = {}
        if mappings:
            for k, v in mappings.items():
                self.bind(k, v)

    def bind(self, var: str, value: Any):
        if isinstance(value, TypeTerm) and self.mappings:
            value = value.apply_substitution(self)
        if isinstance(value, TypeTerm):
            delta = Substitution.__new__(Substitution)
            delta.mappings = {var: value}
            for k, existing in list(self.mappings.items()):
                if k != var and isinstance(existing, TypeTerm):
                    self.mappings[k] = existing.apply_substitution(delta)
        self.mappings[var] = value

    def get(self, var: str, default: Any = None) -> Any:
        return self.mappings.get(var, default)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose substitutions: (sigma1 . sigma2)(t) = sigma1(sigma2(t))."""
        new_map: Dict[str, Any] = {}
        for k, v in other.mappings.items():
            if isinstance(v, TypeTerm):
                new_map[k] = v.apply_substitution(self)
            elif isinstance(v, str) and v.strip():
                resolved = TypeTerm.from_string(v).apply_substitution(self)
                new_map[k] = resolved if not isinstance(resolved, AtomicType) else str(resolved)
            else:
                new_map[k] = v
        for k, v in self.mappings.items():
            if k not in new_map:
                new_map[k] = v
        return Substitution(new_map)

    def copy(self) -> 'Substitution':
        return Substitution(dict(self.mappings))

    def __repr__(self) -> str:
        return f"σ({self.mappings})"


# =====================================================================
# 2. Robinson's First-Order Unification Algorithm
# =====================================================================

def occurs_check(
    var_name: str,
    term: Any,
    sigma: Optional['Substitution'] = None,
    visited: Optional[Tuple[str, ...]] = None
) -> bool:
    """
    Returns True if type variable `var_name` occurs within `term`.
    Occurs check is fundamental to Robinson's first-order unification to prevent
    infinite / cyclic terms (e.g. T = Sequence[T]).
    """
    if visited and var_name in visited:
        return False

    if isinstance(term, TypeVariable):
        if term.var_name == var_name:
            return True
        if sigma and term.var_name in sigma.mappings:
            new_visited = (visited or ()) + (term.var_name,)
            return occurs_check(var_name, sigma.mappings[term.var_name], sigma, new_visited)
        return False
    elif isinstance(term, GenericTypeTerm):
        return any(occurs_check(var_name, a, sigma, visited) for a in term.args)
    elif isinstance(term, UnionTypeTerm):
        return any(occurs_check(var_name, t, sigma, visited) for t in term.terms)
    elif isinstance(term, TypestateTerm):
        if "[" in term.type_name or "|" in term.type_name:
            if occurs_check(var_name, parse_type_term(term.type_name), sigma, visited):
                return True
        if TypeRegistry.get_instance().is_type_variable(term.type_name):
            if term.type_name == var_name:
                return True
            if sigma and term.type_name in sigma.mappings:
                new_visited = (visited or ()) + (term.type_name,)
                return occurs_check(var_name, sigma.mappings[term.type_name], sigma, new_visited)
        return False
    elif isinstance(term, AtomicType):
        return False
    elif isinstance(term, AlgebraicSignature):
        return occurs_check(var_name, term.type_name, sigma, visited)
    elif isinstance(term, str):
        if "[" in term or "|" in term:
            return occurs_check(var_name, parse_type_term(term), sigma, visited)
        if var_name == term and TypeRegistry.get_instance().is_type_variable(term):
            return True
        if sigma and term in sigma.mappings and (not visited or term not in visited):
            new_visited = (visited or ()) + (term,)
            return occurs_check(var_name, sigma.mappings[term], sigma, new_visited)
        return False
    return False


_UNIFY_CACHE_MAX = 32768
_UNIFY_CACHE_LOCK = threading.Lock()
_UNIFY_BASE_CACHE: collections.OrderedDict[Tuple[int, TypeTerm, TypeTerm], Optional[Substitution]] = collections.OrderedDict()


def _unify_cache_put(key: Tuple[int, TypeTerm, TypeTerm], value: Optional[Substitution]) -> None:
    """Thread-safe, bounded write to _UNIFY_BASE_CACHE."""
    with _UNIFY_CACHE_LOCK:
        _UNIFY_BASE_CACHE[key] = value
        if len(_UNIFY_BASE_CACHE) > _UNIFY_CACHE_MAX:
            _UNIFY_BASE_CACHE.popitem(last=False)


def _bare_producer_satisfies_typestate(t2: TypestateTerm, registry: TypeRegistry) -> bool:
    """True iff an unconstrained bare producer meets consumer typestate & non-advisory qualifiers."""
    if not registry.is_state_compatible(
        producer_state="any",
        consumer_state=t2.state,
        consumer_accepted=t2.accepted_states,
        consumer_parent=t2.parent_state,
        consumer_domain=t2.domain or None,
    ):
        return False
    if t2.qualifiers:
        req = set(t2.qualifiers) - registry.get_advisory_qualifiers()
        if req:
            return False
    return True


def _unify_arg_list(
    args1: Tuple[TypeTerm, ...],
    args2: Tuple[TypeTerm, ...],
    var_spec: Any,
    cur_sub: Substitution,
    idx: int = 0,
) -> Optional[Substitution]:
    """Recursive backtracking parameter unification enforcing constructor variance."""
    if idx >= len(args1):
        return cur_sub
    a1 = args1[idx].apply_substitution(cur_sub)
    a2 = args2[idx].apply_substitution(cur_sub)

    if isinstance(a2, UnionTypeTerm):
        for alt in a2.terms:
            s_alt = _unify_arg_step(a1, alt, var_spec, idx, cur_sub)
            if s_alt is not None:
                res = _unify_arg_list(args1, args2, var_spec, s_alt, idx + 1)
                if res is not None:
                    return res
        return None

    s_step = _unify_arg_step(a1, a2, var_spec, idx, cur_sub)
    if s_step is None:
        return None
    return _unify_arg_list(args1, args2, var_spec, s_step, idx + 1)


def _unify_arg_step(
    a1: TypeTerm,
    a2: TypeTerm,
    var_spec: Any,
    idx: int,
    cur_sub: Substitution,
) -> Optional[Substitution]:
    pos_var = (
        var_spec[idx]
        if isinstance(var_spec, tuple) and idx < len(var_spec)
        else (var_spec if isinstance(var_spec, str) else "covariant")
    )
    if pos_var == "invariant":
        s_fwd = unify(a1, a2, cur_sub)
        if s_fwd is None:
            return None
        if not (isinstance(a1, TypeVariable) or isinstance(a2, TypeVariable) or isinstance(a2, TopType)):
            if unify(a2, a1, cur_sub) is None:
                return None
        return s_fwd
    if pos_var == "contravariant":
        return unify(a2, a1, cur_sub)
    return unify(a1, a2, cur_sub)


def unify(
    term1: Union[TypeTerm, AlgebraicSignature, str],
    term2: Union[TypeTerm, AlgebraicSignature, str],
    sigma: Optional[Substitution] = None
) -> Optional[Substitution]:
    """
    Computes Most General Unifier (mgu) of term1 and term2.
    Returns updated Substitution sigma if unification succeeds, or None (bottom) on failure.
    """
    # Normalize AlgebraicSignature to TypestateTerm
    t1 = _to_type_term(term1)
    t2 = _to_type_term(term2)
    registry = TypeRegistry.get_instance()
    reg_gen = registry._generation

    is_ground_query = (sigma is None or not sigma.mappings)
    cache_key = (reg_gen, t1, t2)
    if is_ground_query:
        with _UNIFY_CACHE_LOCK:
            if cache_key in _UNIFY_BASE_CACHE:
                cached = _UNIFY_BASE_CACHE[cache_key]
                return Substitution(cached.mappings) if cached is not None else None

    sub = Substitution(sigma.mappings if sigma else {})

    if not is_ground_query:
        t1 = t1.apply_substitution(sub)
        t2 = t2.apply_substitution(sub)

    # 0. Structural typestate wrappers (generic/union carrier + state) meeting a
    #    bare structural term: compare structural carriers ONLY if state & qualifiers hold.
    def _structural_inner(t: TypeTerm) -> Optional[TypeTerm]:
        if isinstance(t, TypestateTerm) and ("[" in t.type_name or "|" in t.type_name):
            return parse_type_term(t.type_name)
        return None

    if isinstance(t1, TypestateTerm) != isinstance(t2, TypestateTerm):
        in1, in2 = _structural_inner(t1), _structural_inner(t2)
        if in1 is not None and not isinstance(t2, (TopType, TypeVariable)):
            return unify(in1, t2, sub)
        if in2 is not None and isinstance(t2, TypestateTerm) and not isinstance(t1, TypeVariable):
            if not _bare_producer_satisfies_typestate(t2, registry):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None
            return unify(t1, in2, sub)

    # 1. Identity, or consumer Top (consumer accepts any producer).
    #    An untyped Top PRODUCER carries no proof and cannot satisfy a concrete consumer.
    if t1 == t2 or isinstance(t2, TopType):
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub
    if isinstance(t1, TopType) and not isinstance(t2, TypeVariable):
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # 2. Variable binding (Robinson first-order unification with occurs check)
    if isinstance(t1, TypeVariable):
        if isinstance(t2, TypeVariable) and t1.var_name == t2.var_name:
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if occurs_check(t1.var_name, t2, sub):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # Occurs check failure -> bottom
        sub.bind(t1.var_name, t2)
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    if isinstance(t2, TypeVariable):
        if occurs_check(t2.var_name, t1, sub):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # Occurs check failure -> bottom
        sub.bind(t2.var_name, t1)
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 2.3. Coproduct / Union unification.
    #   Producer A | B flows into C iff EVERY branch flows into C, taking lattice join
    #   for any type variable bound differently across branches.
    if isinstance(t1, UnionTypeTerm):
        branch_subs: List[Optional[Substitution]] = [unify(alt, t2, sub) for alt in t1.terms]
        if any(bs is None for bs in branch_subs):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None
        merged = sub.copy()
        all_vars: Set[str] = set()
        for bs in branch_subs:
            if bs is not None:
                all_vars.update(bs.mappings.keys())
        for v in sorted(all_vars):
            vals: List[TypeTerm] = []
            for bs in branch_subs:
                if bs is not None and v in bs.mappings:
                    bv = bs.mappings[v]
                    bv_term = bv if isinstance(bv, TypeTerm) else parse_type_term(str(bv))
                    if bv_term not in vals:
                        vals.append(bv_term)
                elif v in sub.mappings:
                    sv = sub.mappings[v]
                    sv_term = sv if isinstance(sv, TypeTerm) else parse_type_term(str(sv))
                    if sv_term not in vals:
                        vals.append(sv_term)
            if not vals:
                continue
            if len(vals) == 1:
                chosen = vals[0]
            else:
                acc_lt = LatticeType.from_str(str(vals[0]))
                for nxt in vals[1:]:
                    acc_lt = acc_lt.join(LatticeType.from_str(str(nxt)), registry)
                chosen = parse_type_term(str(acc_lt))
            if occurs_check(v, chosen, merged):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None
            merged.bind(v, chosen)
        if is_ground_query:
            _unify_cache_put(cache_key, merged)
        return merged

    if isinstance(t2, UnionTypeTerm):
        for alt in t2.terms:
            sub_alt = unify(t1, alt, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _unify_cache_put(cache_key, sub_alt)
                return sub_alt
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # 2.5. Generic container unification (variance-aware + backtracking)
    if isinstance(t1, GenericTypeTerm) and isinstance(t2, GenericTypeTerm):
        c1 = t1.constructor.lower()
        c2 = t2.constructor.lower()
        compat = (c1 == c2) or bool(c1 and c2 and registry.is_subtype(c1, c2))
        if compat and len(t1.args) == len(t2.args):
            if c1 == "callable" and c2 == "callable" and len(t1.args) == 2:
                # Callable[[Dom1], Cod1] <= Callable[[Dom2], Cod2] iff Dom2 <= Dom1 and Cod1 <= Cod2
                a1_arg, a2_arg = t1.args[0], t2.args[0]
                if isinstance(a1_arg, AtomicType) and a1_arg.name in ("...", "Ellipsis"):
                    s_args = sub
                elif isinstance(a2_arg, AtomicType) and a2_arg.name in ("...", "Ellipsis"):
                    s_args = None
                else:
                    s_args = unify(a2_arg, a1_arg, sub)
                if s_args is None:
                    if is_ground_query:
                        _unify_cache_put(cache_key, None)
                    return None
                s_res = unify(t1.args[1], t2.args[1], s_args)
                if is_ground_query:
                    _unify_cache_put(cache_key, s_res)
                return s_res
            var_spec = _CONSTRUCTOR_VARIANCE.get(c2, "covariant")
            res_sub = _unify_arg_list(t1.args, t2.args, var_spec, sub, idx=0)
            if is_ground_query:
                _unify_cache_put(cache_key, res_sub)
            return res_sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # Raw type fallback: List[Contour] satisfies unparameterized List IFF state/qualifiers hold
    if isinstance(t1, GenericTypeTerm) and (isinstance(t2, AtomicType) or isinstance(t2, TypestateTerm)):
        if isinstance(t2, TypestateTerm):
            if not _bare_producer_satisfies_typestate(t2, registry):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None
            target_name = t2.type_name
        else:
            target_name = t2.name
        if registry.is_subtype(t1.constructor, target_name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub

    # Sound parameter demand: untyped list satisfies GenericTypeTerm ONLY IF parameters are type variables
    if isinstance(t2, GenericTypeTerm) and (isinstance(t1, AtomicType) or isinstance(t1, TypestateTerm)):
        source_name = t1.type_name if isinstance(t1, TypestateTerm) else t1.name
        exact_generic_sub = registry.is_subtype(source_name, str(t2))
        if exact_generic_sub or registry.is_subtype(source_name, t2.constructor):
            if exact_generic_sub or all(isinstance(arg, (TypeVariable, TopType)) for arg in t2.args):
                for arg in t2.args:
                    if isinstance(arg, TypeVariable) and arg.var_name not in sub.mappings:
                        sub.bind(arg.var_name, TOP)
                if is_ground_query:
                    _unify_cache_put(cache_key, sub)
                return sub
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None

    # 3. Typestate term unification
    if isinstance(t1, TypestateTerm) and isinstance(t2, TypestateTerm):
        tn1 = t1.type_name
        tn2 = t2.type_name
        has_structural_syntax = (
            ("[" in tn1) or ("[" in tn2) or ("|" in tn1) or ("|" in tn2)
            or (len(tn1) == 1 and tn1.isupper()) or (len(tn2) == 1 and tn2.isupper())
        )
        # Fast-reject concrete base-type mismatches (O(1) lookup) before walking state ancestry/qualifiers
        if not has_structural_syntax and not registry.is_subtype(tn1, tn2):
            t2_tn = (tn2 or "").strip().lower()
            t2_abs = (t2.abstract_type or "").strip().lower()
            is_consumer_abstract = (t2_tn in ABSTRACT_CARRIERS or t2_tn == t2_abs)
            if not (
                is_consumer_abstract
                and t1.abstract_type
                and t2.abstract_type
                and registry.is_subtype(t1.abstract_type, t2.abstract_type)
            ):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None  # Type mismatch -> bottom

        # State compatibility check: delegate to TypeRegistry.is_state_compatible
        if not registry.is_state_compatible(
            producer_state=t1.state,
            consumer_state=t2.state,
            producer_accepted=t1.accepted_states,
            consumer_accepted=t2.accepted_states,
            consumer_parent=t2.parent_state,
            producer_parent=t1.parent_state,
            producer_domain=t1.domain or None,
            consumer_domain=t2.domain or None,
        ):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # State mismatch -> bottom

        # Qualifier subset & mutual-exclusion check
        t1_quals = set(t1.qualifiers or ())
        t2_quals = set(t2.qualifiers or ())
        if registry.has_conflicting_qualifiers(t1_quals, t2_quals):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None
        if t2_quals and not t2_quals.issubset(t1_quals):
            ignorable = registry.get_advisory_qualifiers()
            req = {q for q in t2_quals if q not in ignorable}
            if req and not req.issubset(t1_quals):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None

        # Check if type_names contain generic / union definitions
        if has_structural_syntax:
            inner1 = parse_type_term(tn1)
            inner2 = parse_type_term(tn2)
            return unify(inner1, inner2, sub)

        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 4. Atomic type unification
    if isinstance(t1, AtomicType) and isinstance(t2, AtomicType):
        if registry.is_subtype(t1.name, t2.name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # 5. Mixed Atomic and Typestate unification: normalize atomic terms to unconstrained typestates
    if isinstance(t1, AtomicType) and isinstance(t2, TypestateTerm):
        return unify(TypestateTerm(type_name=t1.name, state="any"), t2, sub)

    if isinstance(t1, TypestateTerm) and isinstance(t2, AtomicType):
        return unify(t1, TypestateTerm(type_name=t2.name, state="any"), sub)

    if is_ground_query:
        _unify_cache_put(cache_key, None)
    return None


_ALGEBRAIC_SIG_TERM_CACHE: collections.OrderedDict[Tuple[int, AlgebraicSignature], TypeTerm] = collections.OrderedDict()
_TERM_CACHE_LOCK = threading.RLock()


def _to_type_term(item: Any) -> TypeTerm:
    if isinstance(item, TypeTerm):
        return item
    registry = TypeRegistry.get_instance()
    reg_gen = registry._generation
    if isinstance(item, PortSignature):
        cached = getattr(item, "_cached_term", None)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == reg_gen:
            return cached[1]
        res = _to_type_term(item.signature)
        try:
            item._cached_term = (reg_gen, res)
        except (AttributeError, TypeError):
            pass
        return res
    if isinstance(item, AlgebraicSignature):
        sig_key = (reg_gen, item)
        with _TERM_CACHE_LOCK:
            cached = _ALGEBRAIC_SIG_TERM_CACHE.get(sig_key)
            if cached is not None:
                return cached
        t_clean = item.type_name.strip()
        has_state_constraint = (
            (item.state is not None and not registry.is_declared_top_state(item.state))
            or bool(item.qualifiers)
            or bool(getattr(item, "accepted_states", ()))
        )
        if (item.is_top() or is_top_symbol(t_clean, registry)) and not has_state_constraint:
            res = TOP
        elif t_clean.startswith("?") or registry.is_type_variable(t_clean):
            res = TypeTerm.from_string(t_clean)
        elif "|" in t_clean or ("[" in t_clean and t_clean.endswith("]")):
            # Structural carrier: keep typestate/qualifiers when they constrain the term
            if registry.is_declared_top_state(item.state) and not item.qualifiers and not item.accepted_states:
                res = TypeTerm.from_string(t_clean)
            else:
                res = TypestateTerm(
                    type_name=t_clean,
                    state=item.state,
                    qualifiers=item.qualifiers,
                    abstract_type=getattr(item, "abstract_type", None),
                    accepted_states=getattr(item, "accepted_states", ()),
                    parent_state=getattr(item, "parent_state", None),
                    domain=getattr(item, "domain", "") or "",
                )
        else:
            canonical = registry.canonical_name(t_clean)
            if is_top_symbol(canonical, registry) and not has_state_constraint:
                res = TOP
            else:
                res = TypestateTerm(
                    type_name=canonical,
                    state=item.state,
                    qualifiers=item.qualifiers,
                    abstract_type=getattr(item, "abstract_type", None),
                    accepted_states=getattr(item, "accepted_states", ()),
                    parent_state=getattr(item, "parent_state", None),
                    domain=getattr(item, "domain", "") or "",
                )
        with _TERM_CACHE_LOCK:
            _ALGEBRAIC_SIG_TERM_CACHE[sig_key] = res
            if len(_ALGEBRAIC_SIG_TERM_CACHE) > _UNIFY_CACHE_MAX:
                _ALGEBRAIC_SIG_TERM_CACHE.popitem(last=False)
        return res
    if isinstance(item, str):
        if is_top_symbol(item, registry):
            return TOP
        return TypeTerm.from_string(item)
    return TOP


# =====================================================================
# 2.6. Generic Substitution and Topology Verification Gates
# =====================================================================

def substitute_generics(
    target: Any,
    sigma: Union[Substitution, Dict[str, Any]]
) -> Any:
    """
    Substitutes generic type variables in schemas, signatures, or type strings using substitution sigma.
    e.g. substitute_generics('Sequence[T]', {'T': 'MatLike'}) -> 'Sequence[MatLike]'
    e.g. substitute_generics('T', {'T': 'MatLike'}) -> 'MatLike'
    e.g. substitute_generics(PortSchema(type_name='T'), {'T': 'MatLike'}) -> PortSchema(type_name='MatLike')
    """
    if sigma is None:
        return target
    # Accept any substitution-like object (duck-typed) so that alternate module
    # instances can never corrupt the substitution application step.
    mappings = sigma.mappings if hasattr(sigma, "mappings") else dict(sigma)
    if not mappings:
        return target

    sub = Substitution({k: v for k, v in mappings.items()})

    if isinstance(target, str):
        term = TypeTerm.from_string(target)
        res_term = term.apply_substitution(sub)
        if isinstance(res_term, TypeVariable):
            return res_term.var_name
        return str(res_term)

    if isinstance(target, PortSignature):
        new_sig = substitute_generics(target.signature, sub)
        return target.with_signature(new_sig)

    if isinstance(target, AlgebraicSignature):
        new_type = substitute_generics(target.type_name, sub)
        return AlgebraicSignature(
            type_name=str(new_type),
            state=target.state,
            qualifiers=target.qualifiers,
            abstract_type=target.abstract_type,
            accepted_states=getattr(target, "accepted_states", ()),
            parent_state=getattr(target, "parent_state", None),
            domain=getattr(target, "domain", ""),
        )

    if hasattr(target, "type_name") and hasattr(target, "model_copy"):
        new_type = substitute_generics(target.type_name, sub)
        return target.model_copy(update={"type_name": str(new_type)})

    if isinstance(target, TypeTerm):
        return target.apply_substitution(sub)

    return target


def _five_axis_branch_compatible(
    then_port: Any,
    else_port: Any,
    join_port: Optional[Any] = None,
) -> bool:
    """Checks 5-axis compatibility for coproduct branch joins."""
    if not (isinstance(then_port, PortSignature) and isinstance(else_port, PortSignature)):
        return True
    reg = TypeRegistry.get_instance()
    if join_port is not None and isinstance(join_port, PortSignature):
        r1 = compute_port_residual(then_port, join_port, registry=reg)
        r2 = compute_port_residual(else_port, join_port, registry=reg)
        return not (
            r1.nominal_conflict
            or r2.nominal_conflict
            or r1.layout_delta
            or r2.layout_delta
            or r1.semantics_delta
            or r2.semantics_delta
            or r1.invariants_delta
            or r2.invariants_delta
        )
    r_fwd = compute_port_residual(then_port, else_port, registry=reg)
    r_rev = compute_port_residual(else_port, then_port, registry=reg)
    # Branch arms either coerce mutually or widen to a compatible supertype without layout/nominal clash
    if r_fwd.nominal_conflict or r_rev.nominal_conflict:
        return False
    if r_fwd.layout_delta and r_rev.layout_delta:
        return False
    return not (r_fwd.carrier_delta and r_rev.carrier_delta)


def verify_coproduct_branch(
    then_term: Union[TypeTerm, AlgebraicSignature, PortSignature, str],
    else_term: Union[TypeTerm, AlgebraicSignature, PortSignature, str],
    join_type: Optional[Union[TypeTerm, AlgebraicSignature, PortSignature, str]] = None,
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[TypeTerm], Optional[Substitution]]:
    """
    Verifies that coproduct True-path and False-path can unify to a common join type D:
      unify(tau_then, D) != bottom and unify(tau_else, D) != bottom
    Returns (is_valid, resolved_join_type, updated_sigma).
    """
    if not _five_axis_branch_compatible(then_term, else_term, join_type):
        return False, None, None

    sub = Substitution(sigma.mappings if sigma else {})
    t_then = _to_type_term(then_term)
    t_else = _to_type_term(else_term)

    if join_type is not None:
        target_D = _to_type_term(join_type)
        s1 = unify(t_then, target_D, sub)
        if s1 is None:
            return False, None, None
        s2 = unify(t_else, target_D, s1)
        if s2 is None:
            return False, None, None
        return True, target_D, s2

    # If no join_type given, check if then and else unify with each other
    s_join = unify(t_then, t_else, sub)
    if s_join is not None:
        return True, t_then.apply_substitution(s_join), s_join

    # Check poset reachability in TypeRegistry
    registry = TypeRegistry.get_instance()
    n_then = getattr(t_then, "type_name", getattr(t_then, "name", str(t_then)))
    n_else = getattr(t_else, "type_name", getattr(t_else, "name", str(t_else)))
    if registry.is_subtype(n_then, n_else):
        return True, t_else, sub
    if registry.is_subtype(n_else, n_then):
        return True, t_then, sub

    # Widening join on 5-axis product lattice if both share abstract carrier / top object
    if isinstance(then_term, PortSignature) and isinstance(else_term, PortSignature):
        if then_term.abstract_type and then_term.abstract_type == else_term.abstract_type:
            return True, _to_type_term(then_term.abstract_type), sub

    return False, None, None


def verify_traced_loop_invariant(
    feedback_in: Union[TypeTerm, AlgebraicSignature, PortSignature, str],
    feedback_out: Union[TypeTerm, AlgebraicSignature, PortSignature, str],
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[Substitution]]:
    """
    Verifies the categorical Traced Feedback loop invariant:
      unify(tau_feedback_out, tau_feedback_in) != bottom
    and S_out <= S_in across all 5 axes (nominal, carrier, layout, semantics, typestate).
    Ensures that loop body updates preserve or are compatible with the accumulator state U.
    """
    if isinstance(feedback_in, PortSignature) and isinstance(feedback_out, PortSignature):
        reg = TypeRegistry.get_instance()
        res = compute_port_residual(feedback_out, feedback_in, registry=reg)
        if (
            res.nominal_conflict
            or res.carrier_delta
            or res.layout_delta
            or res.semantics_delta
            or res.typestate_delta
            or res.invariants_delta
        ):
            return False, None

    t_in = _to_type_term(feedback_in)
    t_out = _to_type_term(feedback_out)
    sub = Substitution(sigma.mappings if sigma else {})
    new_sub = unify(t_out, t_in, sub)
    if new_sub is None:
        return False, None
    return True, new_sub


# =====================================================================
# 3. Formal Type Monad M_T(A)
# =====================================================================

class MonadResult(Generic[T], ABC):
    """
    Formal Type Monad Result: M_T(A) = { (a, sigma) } U { bottom }.
    """
    @abstractmethod
    def is_bottom(self) -> bool:
        pass


@dataclass(frozen=True, slots=True)
class Success(MonadResult[T]):
    """Successful computation carrying value a and substitution sigma."""
    value: T
    sigma: Substitution

    def is_bottom(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Failure(MonadResult[T]):
    """Failure bottom (⊥)."""
    reason: str

    def is_bottom(self) -> bool:
        return True


def _is_sequence_or_tuple_type(t_name: str, reg: Optional[TypeRegistry] = None) -> bool:
    """Return True if `t_name` denotes a non-string sequence/tuple/list/collection type."""
    return _tp_is_sequence_or_tuple_type(t_name, registry=reg)


def _is_valid_slot_value(val: Any, sig: Optional[PortSignature] = None) -> bool:
    """Validates whether an extracted slot/hyperparameter value conforms to port schema."""
    try:
        reg = TypeRegistry.get_instance()
    except Exception:
        reg = None
    return _tp_validate_slot_value(val, sig, registry=reg, type_term_cls=TypeTerm)


_PY_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _extract_free_expr_names(node: ast.AST, bound: Optional[Set[str]] = None) -> Set[str]:
    """Extracts unbound variable names loaded in an AST expression (excluding lambda/comprehension params)."""
    active_bound = set(bound) if bound else set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return set() if node.id in active_bound else {node.id}
    if isinstance(node, ast.Lambda):
        lam_bound = set(active_bound)
        for a in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            lam_bound.add(a.arg)
        if node.args.vararg:
            lam_bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            lam_bound.add(node.args.kwarg.arg)
        res_names: Set[str] = set()
        for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
            res_names.update(_extract_free_expr_names(d, active_bound))
        res_names.update(_extract_free_expr_names(node.body, lam_bound))
        return res_names
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        comp_bound = set(active_bound)
        res_names = set()
        for gen in node.generators:
            res_names.update(_extract_free_expr_names(gen.iter, comp_bound))
            for t_node in ast.walk(gen.target):
                if isinstance(t_node, ast.Name):
                    comp_bound.add(t_node.id)
            for if_c in gen.ifs:
                res_names.update(_extract_free_expr_names(if_c, comp_bound))
        if isinstance(node, ast.DictComp):
            res_names.update(_extract_free_expr_names(node.key, comp_bound))
            res_names.update(_extract_free_expr_names(node.value, comp_bound))
        else:
            res_names.update(_extract_free_expr_names(node.elt, comp_bound))
        return res_names
    res_names = set()
    for child in ast.iter_child_nodes(node):
        res_names.update(_extract_free_expr_names(child, active_bound))
    return res_names


def _classify_template_identifier_port(cell: Any, port_name: str) -> Tuple[bool, bool]:
    """
    Detects from `cell.code_template` whether `{port_name}` sits in a Python declaration
    identifier, exception type, or parameter-list position
    (`def {fn}({args}):` / `class {cls}:` / `except {exc}:` / `lambda {args}:`).
    Returns `(is_identifier_port, is_param_list)`.
    """
    tmpl = str(getattr(cell, "code_template", "") or "")
    if not tmpl or not port_name:
        return False, False
    ph = "{" + port_name + "}"
    if ph not in tmpl:
        return False, False
    for line in tmpl.splitlines():
        stripped = line.lstrip()
        for kw in ("def ", "class ", "except "):
            if stripped.startswith(kw):
                after = stripped[len(kw):].lstrip()
                if after.startswith(ph):
                    rest = after[len(ph):]
                    if not rest or rest[0] in (" ", "\t", "(", ":", ",", ")"):
                        return True, False
        if stripped.startswith("def ") and "(" in stripped:
            open_p = stripped.find("(")
            close_p = stripped.find(")", open_p + 1)
            if close_p != -1 and ph in stripped[open_p + 1 : close_p]:
                return True, True
        lam_idx = stripped.find("lambda")
        if lam_idx != -1 and (lam_idx == 0 or not (stripped[lam_idx - 1].isalnum() or stripped[lam_idx - 1] == "_")):
            after_lam = stripped[lam_idx + 6 :]
            if not after_lam or after_lam[0] in (" ", "\t", ":"):
                colon_idx = after_lam.find(":")
                if colon_idx != -1 and ph in after_lam[:colon_idx]:
                    return True, True
    return False, False


def _is_macro_block_slot(cell: Any, port_name: str) -> bool:
    """
    Returns True only when `{port_name}` is a macro/combinator statement-block slot
    (i.e. declared in `cell.slots` with a non-empty slot-spec dict or sitting on its
    own indented line in `cell.code_template`), not scalar parameter defaults in `slots`.
    """
    if cell is None or not port_name:
        return False
    slots = getattr(cell, "slots", None)
    if not isinstance(slots, dict) or port_name not in slots:
        return False
    slot_spec = slots.get(port_name)
    if isinstance(slot_spec, dict) and slot_spec:
        return True
    tmpl = str(getattr(cell, "code_template", "") or "")
    ph_slot = "{" + port_name + "}"
    return any(ln.strip() == ph_slot and len(ln) > len(ln.lstrip()) for ln in tmpl.splitlines())


def _normalize_identifier_port_binding(val: Any, is_param_list: bool = False) -> Optional[str]:
    """Normalizes and AST-verifies an unquoted Python identifier or parameter list binding."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (list, tuple)):
        parts = [str(v).strip().strip("'\"").strip() for v in val if str(v).strip()]
        raw = ", ".join(p for p in parts if p)
    else:
        raw = str(val).strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            parts = [p.strip().strip("'\"").strip() for p in inner.split(",") if p.strip()]
            raw = ", ".join(p for p in parts if p)
        else:
            raw = raw.strip("'\"").strip()
    if not raw:
        return None
    if not is_param_list:
        if not raw.isidentifier() or keyword.iskeyword(raw):
            return None
        parsed_fn, _ = _cached_ast_parse_ro(f"def {raw}(): pass", "exec")
        return raw if parsed_fn is not None else None
    parts = [p.strip().strip("'\"").strip() for p in raw.split(",")]
    if not parts or any((not p or not p.isidentifier() or keyword.iskeyword(p)) for p in parts):
        return None
    norm = ", ".join(parts)
    parsed_sig, _ = _cached_ast_parse_ro(f"def _f({norm}): pass", "exec")
    return norm if parsed_sig is not None else None


def _resolve_bound_slot_for_port(cell: Any, port_name: str, port_sig: Any) -> Optional[str]:
    """
    Returns a typing placeholder when `cell.bound_slots` already holds a planned sub-pipeline
    for this HOF / block-body input port (later substituted during code synthesis).
    """
    bound_slots = getattr(cell, "bound_slots", None)
    if not isinstance(bound_slots, dict) or not bound_slots:
        return None
    non_empty_slots = {k: v for k, v in bound_slots.items() if v}
    if not non_empty_slots:
        return None
    reg = TypeRegistry.get_instance()
    p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "role", "") or "").strip().lower()
    t_name = str(getattr(port_sig, "type_name", "") or "").strip().lower()
    is_hof = reg.is_hof_role(p_role) or t_name.startswith("callable")
    if not is_hof:
        return None
    if port_name in non_empty_slots:
        return "lambda *_: None"
    if any(
        port_name.startswith(f"{sk}_") or sk.startswith(f"{port_name}_")
        for sk in non_empty_slots
    ):
        return "lambda *_: None"
    tmpl = str(getattr(cell, "code_template", "") or "")
    ph_slot = "{" + port_name + "}"
    if tmpl and any(ln.strip() == ph_slot and len(ln) > len(ln.lstrip()) for ln in tmpl.splitlines()):
        return "lambda *_: None"
    if len(non_empty_slots) == 1:
        return "lambda *_: None"
    return None


def _is_valid_grounded_llm_expr(
    cand_val: Any,
    ctx: Any,
    cell: Any = None,
    port_name: Optional[str] = None,
) -> bool:
    """Validates an LLM-proposed candidate expression against active LanguageBackend & prompt literals."""
    if cell is not None and port_name:
        is_id_port, is_param_list = _classify_template_identifier_port(cell, port_name)
        if is_id_port:
            return _normalize_identifier_port_binding(cand_val, is_param_list=is_param_list) is not None
    cand_str = str(cand_val).strip()
    if not cand_str:
        return False
    be = get_active_language_backend()
    parsed_cand_body: Optional[ast.AST] = None
    if be.name == "python":
        parsed_cand_body = _parse_expr_ast(cand_str)
        var_names = _extract_free_expr_names(parsed_cand_body) if parsed_cand_body is not None else None
    else:
        var_names = be.extract_identifier_refs(cand_str)
    if var_names is None:
        return False
    if not var_names:
        if isinstance(cand_val, str) and not be.is_null_or_bool_literal(cand_val, case_insensitive=False):
            if isinstance(parsed_cand_body, ast.Lambda):
                return True
        valid_nums = {v for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("numeric", "int_literal", "float_literal")}
        if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
            val_str = str(int(cand_val)) if int(cand_val) == cand_val else str(cand_val)
            if val_str not in valid_nums:
                return False
        elif isinstance(cand_val, str) and not be.is_null_or_bool_literal(cand_val, case_insensitive=False):
            valid_strs = {v.lower() for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("quoted_str", "identifier", "file_asset")}
            if cand_val.lower() not in valid_strs:
                return False
    return all(
        (v in ctx.variables or (be.name == "python" and v in _PY_BUILTIN_NAMES))
        for v in var_names
    )


def _format_literal_value(val: Any, sig: Optional[PortSignature] = None) -> str:
    """Formats an extracted slot/hyperparameter value into an executable code literal."""
    try:
        reg = TypeRegistry.get_instance()
    except Exception:
        reg = None
    return _tp_format_literal_value(
        val,
        sig=sig,
        registry=reg,
        language_backend=get_active_language_backend(),
        parse_expr_ast_fn=_parse_expr_ast,
        type_term_cls=TypeTerm,
    )


def _consume_matching_literals(ctx: Any, slot_val: Any) -> None:
    """Marks prompt literals matching an SLM-bound value as consumed in `ctx.used_indices`."""
    _sb_consume_matching_literals(ctx, slot_val)



# =====================================================================
# 4. Domain-Agnostic Execution Context
# =====================================================================

class LiteralSpan(tuple):
    """3-tuple (pos, kind, val) that also retains exact span start and end positions."""
    def __new__(cls, pos: int, kind: str, val: str, end_pos: Optional[int] = None):
        inst = super().__new__(cls, (pos, kind, val))
        inst.end_pos = end_pos if end_pos is not None else (pos + len(val))
        return inst


def _split_dimension_token(word: str) -> List[Tuple[int, str]]:
    """
    Splits dimension notation such as "5x5", "800x600" or "3x3x3" into its numeric parts,
    returning [(offset_in_word, numeric_text), ...]; returns [] when `word` is not
    dimension notation. Lexical grammar only (no regex, no port/library knowledge): it lets
    "kernel size 5x5" bind exactly like "kernel size 5 by 5", because both then yield two
    adjacent numeric spans that the contiguous-tuple binder already understands.
    """
    seps = ("x", "X", "\u00d7")
    parts: List[Tuple[int, str]] = []
    start = 0
    for i, ch in enumerate(word + "x"):
        if i == len(word) or ch in seps:
            seg = word[start:i]
            if not seg or seg.count(".") > 1 or not seg.replace(".", "").isdigit() or seg.startswith(".") or seg.endswith("."):
                return []
            parts.append((start, seg))
            start = i + 1
    return parts if len(parts) >= 2 else []


def _hungarian_min_cost(cost_matrix: List[List[float]]) -> List[int]:
    """Exact O(n^3) minimum-cost square linear assignment solver (pure-Python fallback)."""
    n = len(cost_matrix)
    if n == 0:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    inf = float("inf")
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row_i0 = cost_matrix[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row_i0[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = [-1] * n
    for j in range(1, n + 1):
        if p[j] >= 1:
            ans[p[j] - 1] = j - 1
    return ans


def _min_cost_bipartite_assign(
    cost_matrix: List[List[float]],
    unassign_costs: List[float],
    forbidden: float = 1e9,
    _use_scipy: bool = True,
) -> Dict[int, int]:
    """Global min-cost bipartite port-to-literal assignment with explicit unassign columns."""
    m = len(cost_matrix)
    if m == 0:
        return {}
    k = len(cost_matrix[0]) if m else 0
    if k == 0:
        return {}
    cols_total = k + m
    aug: List[List[float]] = []
    for i in range(m):
        row = list(cost_matrix[i]) + [forbidden] * m
        row[k + i] = float(unassign_costs[i])
        aug.append(row)

    if _use_scipy:
        try:
            from scipy.optimize import linear_sum_assignment
            r_ind, c_ind = linear_sum_assignment(aug)
            return {
                int(r): int(c)
                for r, c in zip(r_ind, c_ind)
                if int(c) < k and cost_matrix[int(r)][int(c)] < forbidden
            }
        except Exception:
            pass

    sq_aug = list(aug)
    for _ in range(m, cols_total):
        sq_aug.append([0.0] * cols_total)
    assign = _hungarian_min_cost(sq_aug)
    return {
        r: assign[r]
        for r in range(m)
        if 0 <= assign[r] < k and cost_matrix[r][assign[r]] < forbidden
    }


class ExecutionContext:
    """
    Runtime execution scope holding bound variables and literal arguments.
    Operates strictly via algebraic typestates and substitutions with ZERO domain hardcodes.
    """

    @staticmethod
    def _is_path_string(val: Any) -> bool:
        if not val or not isinstance(val, str):
            return False
        s = val.strip()
        if len(s) < 2:
            return False
        if "/" in s or "\\" in s:
            return True
        if "." in s:
            base, _, ext = s.rpartition(".")
            if base and ext and len(ext) <= 8 and ext.isalnum():
                return True
        return False

    def __init__(self, prompt: str = "", scope: Optional[Dict[str, Any]] = None, initial_scope: Optional[Set[str]] = None, parameters: Optional[Dict[str, Any]] = None, strict_io: Optional[bool] = None):
        if strict_io is None:
            try:
                try:
                    from .config import get_settings
                except (ImportError, ValueError):
                    from config import get_settings
                self.strict_io: bool = bool(getattr(get_settings(), "strict_io", True))
            except Exception:
                self.strict_io = True
        else:
            self.strict_io = bool(strict_io)
        self._prompt = prompt or ""
        self.scope: Dict[str, Any] = dict(scope or {})
        if initial_scope is not None:
            self.initial_scope: Set[str] = set(initial_scope)
        else:
            self.initial_scope: Set[str] = set(scope.keys()) if (scope and isinstance(scope, dict)) else set()
        self.variables: Dict[str, Tuple[PortSignature, str]] = {}
        self.var_counter: int = 0
        self.pipeline_vars: Set[str] = set()
        self.var_sources: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = dict(parameters or {})
        self.used_indices: Set[int] = set()
        self.consumed_tokens: Set[str] = set()
        self.consumed_vars: Set[str] = set()
        # Tuple-member consumption tracking: (var_name, member_index) pairs already
        # bound by earlier cells. Later cells consuming the same heterogeneous
        # product prefer the remaining members — this is how allocation semantics
        # (train/verify/test partitions) emerge from consumption order.
        self.consumed_members: Set[Tuple[str, int]] = set()
        self.consumed_keys: Set[str] = set()
        self.unresolved_ports: List[Tuple[str, str]] = []
        self.typed_holes: List[Any] = []
        self.value_env: Dict[str, str] = {}
        self.unbindable_count: int = 0
        # Declared runtime egress aliases: maps a prompt-declared sink
        # identifier (e.g. "store the results into Z" -> "Z") to the pipeline
        # variable that carries the terminal value. Applied at the RUNTIME
        # namespace level by the sandbox — the emitted source is never
        # rewritten to fake an assignment.
        self.runtime_aliases: Dict[str, str] = {}
        self.ordered_literals: List[Tuple[int, str, str]] = self._extract_universal_literals(self._prompt)
        # D8: Map each literal to its prompt clause index
        self.literal_clause_map: Dict[int, int] = self._build_literal_clause_map(self._prompt, self.ordered_literals)
        self.target_literals: Set[str] = set()
        t_sink = (
            getattr(self, "target_key", None)
            or self.parameters.get("target_sink")
            or self.parameters.get("target_key")
            or self._extract_target_sink(self._prompt)
        )
        self.target_key: Optional[str] = str(t_sink).strip() if t_sink else None
        if self.target_key:
            self.target_literals.add(self.target_key)
        self.written_keys: Set[str] = set()
        self.current_cell_clause_idx: Optional[int] = None
        # Prompt clauses that at least one non-bridge pipeline cell serves (set by unify_pipeline).
        # None = unknown, in which case the clause-locality rule for REQUIRED numeric ports is not applied.
        self.pipeline_served_clauses: Optional[Set[int]] = None
        self.clause_spans: List[Tuple[int, int]] = []
        if self._prompt:
            try:
                from tokenizer import CellTokenizer
                clauses = CellTokenizer.split_prompt_clauses(self._prompt)
            except Exception:
                clauses = [self._prompt]
            cur_search = 0
            for cl in clauses:
                start = self._prompt.find(cl, cur_search)
                if start != -1:
                    end = start + len(cl)
                    self.clause_spans.append((start, end))
                    cur_search = end
                else:
                    start = self._prompt.find(cl)
                    if start != -1:
                        self.clause_spans.append((start, start + len(cl)))
        if not self.clause_spans and self._prompt:
            self.clause_spans = [(0, len(self._prompt))]
        # Bare-identifier role map: char position -> stemmed role context tokens
        # (e.g. pos(X) -> {"column"}). Drives role-conditioned identifier binding
        # and for-each multiplicity detection.
        self.identifier_roles: Dict[int, Set[str]] = self._build_identifier_role_map(self._prompt)
        # Provenance and monoidal DAG branch tracking
        self.var_origins: Dict[str, Set[str]] = {}
        self.var_restricted_origins: Dict[str, Set[str]] = {}
        self.var_excluded_origins: Dict[str, Set[str]] = {}
        self.role_bound_keys: Dict[str, List[str]] = {}
        self.setter_focus_keys: Dict[Tuple[str, str], Set[str]] = {}
        self.container_key_layouts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.optic_frames: List[Dict[str, Any]] = []
        self.multi_fiber_vars: Set[str] = set()
        self.current_pipeline: Optional[List[Any]] = None
        self.var_epochs: Dict[str, int] = {}   # row-set epoch per variable (tree-declared row_set_effects)
        self.var_partitions: Dict[str, Tuple[int, int, Optional[str]]] = {}
        self.partitioned_vars: Dict[str, int] = {}
        self.handle_dim_exclusions: Dict[Tuple[str, str], Tuple[Set[str], Set[str]]] = {}
        self.var_clause_tokens: Dict[str, Set[str]] = {}
        self.synthetic_io_placeholders: Dict[Tuple[str, str], str] = {}
        self._synthetic_io_counts: Dict[str, int] = {}
        self.binding_trace: List[Dict[str, Any]] = []
        self.superseded_vars: Set[str] = set()
        self.var_parents: Dict[str, Set[str]] = {}
        self.scope_variables: Dict[str, Any] = {}
        self.llm_slots: Dict[str, Dict[str, Any]] = {}
        self.step_llm_slots: Dict[int, Dict[str, Any]] = {}
        self._cell_slots_extracted: bool = False
        self.current_cell_step_idx: Optional[int] = None
        self._bipartite_literal_plan: Dict[Tuple[int, str], Tuple[int, str]] = {}
        self._explicit_keys: List[str] = []
        self.hyperparameters: Dict[str, Any] = {}
        self.composed_bridge_cells: List[Any] = []
        self.diagnostics: DiagnosticTelemetry = DiagnosticTelemetry()
        try:
            from dag_plan import SemanticBlackboard
            self.blackboard = SemanticBlackboard()
        except Exception:
            try:
                from .dag_plan import SemanticBlackboard
                self.blackboard = SemanticBlackboard()
            except Exception:
                self.blackboard = None
        if self.scope:
            for k, v in self.scope.items():
                self.declare_variable(k, v, k)

    @staticmethod
    def _extract_target_sink(prompt: str) -> Optional[str]:
        """Target sink extraction: in symbolic profiles, returns None; resolved by LLM/intent in neural profiles."""
        return None

    def declare_runtime_alias(self, alias_name: str, source_var: str) -> None:
        """Declares that a prompt-declared sink identifier aliases a pipeline variable."""
        name = str(alias_name or "").strip()
        source = str(source_var or "").strip()
        if name and source and name != source:
            self.runtime_aliases[name] = source

    def reset(self):
        """Resets transient pipeline variables and consumed literal indices, preserving prompt, parameters, and initial scope."""
        self.variables = {}
        self.var_sources = {}
        self.var_origins = {}
        self.var_restricted_origins = {}
        self.var_excluded_origins = {}
        self.role_bound_keys = {}
        self.setter_focus_keys = {}
        self.container_key_layouts = {}
        self.optic_frames = []
        self.multi_fiber_vars = set()
        self.var_partitions = {}
        self.partitioned_vars = {}
        self.handle_dim_exclusions = {}
        self.superseded_vars = set()
        self.var_parents = {}
        self.scope_variables = {}
        self.used_indices = set()
        self.consumed_tokens = set()
        self.consumed_vars = set()
        self.consumed_members = set()
        self.consumed_keys = set()
        self.target_literals = set()
        t_sink = (
            getattr(self, "target_key", None)
            or self.parameters.get("target_sink")
            or self.parameters.get("target_key")
            or self._extract_target_sink(self._prompt)
        )
        if t_sink:
            self.target_key = str(t_sink).strip()
            self.target_literals.add(self.target_key)
            self.target_literals.add(self.target_key.lower())
        self.current_cell_clause_idx = None
        self.current_cell_step_idx = None
        self.var_clause_tokens = {}
        self.synthetic_io_placeholders = {}
        self._synthetic_io_counts = {}
        self.unresolved_ports = []
        self.typed_holes = []
        self.value_env = {}
        self._bipartite_literal_plan = {}
        self.unbindable_count = 0
        self.var_counter = 0
        self.runtime_aliases = dict(getattr(self, "runtime_aliases", {}) or {})
        if hasattr(self, "scope") and self.scope:
            for k, v in self.scope.items():
                self.declare_variable(k, v, k)

    def trace_binding(self, event: str, **fields: Any) -> None:
        """Layer-3 binding telemetry for --debug. Never affects synthesis."""
        self.binding_trace.append({"event": event, **fields})

    def clone(self) -> "ExecutionContext":
        new_ctx = ExecutionContext(prompt=self._prompt, scope=self.scope, initial_scope=self.initial_scope)
        new_ctx.binding_trace = self.binding_trace  # shared by reference: one trace per synthesis
        new_ctx.variables = dict(self.variables)
        new_ctx.var_sources = dict(self.var_sources)
        new_ctx.var_origins = {k: set(v) for k, v in self.var_origins.items()}
        new_ctx.var_restricted_origins = {k: set(v) for k, v in getattr(self, "var_restricted_origins", {}).items()}
        new_ctx.var_excluded_origins = {k: set(v) for k, v in getattr(self, "var_excluded_origins", {}).items()}
        new_ctx.role_bound_keys = {k: list(v) for k, v in getattr(self, "role_bound_keys", {}).items()}
        new_ctx.setter_focus_keys = {k: set(v) for k, v in getattr(self, "setter_focus_keys", {}).items()}
        new_ctx.container_key_layouts = {k: dict(v) for k, v in getattr(self, "container_key_layouts", {}).items() if isinstance(v, dict)}
        new_ctx.optic_frames = [dict(f) for f in getattr(self, "optic_frames", [])]
        new_ctx.multi_fiber_vars = set(getattr(self, "multi_fiber_vars", set()))
        new_ctx.current_pipeline = getattr(self, "current_pipeline", None)
        new_ctx.var_epochs = dict(self.var_epochs)
        new_ctx.var_partitions = dict(getattr(self, "var_partitions", {}))
        new_ctx.partitioned_vars = dict(getattr(self, "partitioned_vars", {}))
        new_ctx.handle_dim_exclusions = {k: (set(v[0]), set(v[1])) for k, v in getattr(self, "handle_dim_exclusions", {}).items()}
        new_ctx.superseded_vars = set(self.superseded_vars)
        new_ctx.var_parents = {k: set(v) for k, v in self.var_parents.items()}
        new_ctx.scope_variables = dict(self.scope_variables)
        new_ctx.parameters = dict(self.parameters)
        new_ctx.used_indices = set(self.used_indices)
        new_ctx.consumed_tokens = set(self.consumed_tokens)
        new_ctx.consumed_members = set(self.consumed_members)
        new_ctx.consumed_keys = set(self.consumed_keys)
        new_ctx.target_literals = set(self.target_literals)
        new_ctx.current_cell_clause_idx = self.current_cell_clause_idx
        new_ctx.pipeline_served_clauses = self.pipeline_served_clauses
        new_ctx.literal_clause_map = dict(self.literal_clause_map)
        new_ctx.ordered_literals = list(self.ordered_literals)
        new_ctx.identifier_roles = dict(self.identifier_roles)
        new_ctx.clause_spans = list(self.clause_spans)
        new_ctx.unresolved_ports = list(self.unresolved_ports)
        new_ctx.typed_holes = list(getattr(self, "typed_holes", []))
        new_ctx.value_env = dict(getattr(self, "value_env", {}))
        new_ctx._bipartite_literal_plan = dict(getattr(self, "_bipartite_literal_plan", {}))
        new_ctx.unbindable_count = self.unbindable_count
        new_ctx.var_counter = self.var_counter
        new_ctx.strict_io = self.strict_io
        new_ctx.var_clause_tokens = {k: set(v) for k, v in getattr(self, "var_clause_tokens", {}).items()}
        new_ctx.synthetic_io_placeholders = dict(getattr(self, "synthetic_io_placeholders", {}))
        new_ctx._synthetic_io_counts = dict(getattr(self, "_synthetic_io_counts", {}))
        new_ctx.llm_slots = {k: dict(v) if isinstance(v, dict) else v for k, v in self.llm_slots.items()}
        new_ctx.step_llm_slots = {k: dict(v) for k, v in getattr(self, "step_llm_slots", {}).items()}
        new_ctx._cell_slots_extracted = bool(getattr(self, "_cell_slots_extracted", False))
        new_ctx.current_cell_step_idx = getattr(self, "current_cell_step_idx", None)
        new_ctx.hyperparameters = dict(self.hyperparameters)
        new_ctx.runtime_aliases = dict(getattr(self, "runtime_aliases", {}) or {})
        new_ctx.intent_data = dict(getattr(self, "intent_data", {}) or {})
        new_ctx.target_key = getattr(self, "target_key", None)
        new_ctx.variables_spec = list(getattr(self, "variables_spec", []))
        new_ctx.operations = list(getattr(self, "operations", []))
        new_ctx.diagnostics = self.diagnostics
        if new_ctx.llm_slots:
            new_ctx.sync_llm_slots()
        return new_ctx

    @staticmethod
    def _build_literal_clause_map(prompt: str, literals: List[Tuple[int, str, str]]) -> Dict[int, int]:
        """Maps each literal index to its originating prompt clause index."""
        try:
            from .planner import _segment_prompt_clauses
        except (ImportError, ValueError):
            from planner import _segment_prompt_clauses
        clauses = _segment_prompt_clauses(prompt)
        raw_spans = []
        cur_search = 0
        for cl in clauses:
            start = prompt.find(cl, cur_search)
            if start != -1:
                end = start + len(cl)
                raw_spans.append((start, end))
                cur_search = end
            else:
                start = prompt.find(cl)
                if start != -1:
                    raw_spans.append((start, start + len(cl)))
                    cur_search = max(cur_search, start + len(cl))
                else:
                    # cl may be a composite clause joined by ", "
                    sub_pieces = [sp.strip() for sp in cl.split(", ") if sp.strip()]
                    piece_spans = []
                    search_pos = cur_search
                    for sp in sub_pieces:
                        p_start = prompt.find(sp, search_pos)
                        if p_start == -1:
                            p_start = prompt.find(sp)
                        if p_start != -1:
                            piece_spans.append((p_start, p_start + len(sp)))
                            search_pos = max(search_pos, p_start + len(sp))
                    if piece_spans:
                        min_start = min(s for s, e in piece_spans)
                        max_end = max(e for s, e in piece_spans)
                        raw_spans.append((min_start, max_end))
                        cur_search = max(cur_search, max_end)
                    else:
                        raw_spans.append((cur_search, cur_search))
        if not raw_spans:
            raw_spans = [(0, len(prompt))]

        lit_map = {}
        for idx, (pos, kind, val) in enumerate(literals):
            if not raw_spans:
                lit_map[idx] = 0
                continue
            if pos < raw_spans[0][0]:
                lit_map[idx] = 0
                continue
            if pos >= raw_spans[-1][0]:
                lit_map[idx] = len(raw_spans) - 1
                continue
            c_idx = 0
            for i in range(len(raw_spans) - 1):
                if raw_spans[i][0] <= pos < raw_spans[i + 1][0]:
                    c_idx = i
                    break
            lit_map[idx] = c_idx
        return lit_map

    @staticmethod
    def _asset_direction(prompt: str, pos: int) -> str:
        prev_chunk = (prompt or "")[:pos].rstrip()
        # Scope backward scan to the current clause delimiter so prior clauses do not leak
        for delim in (",", ";", "\n", ". "):
            d_idx = prev_chunk.rfind(delim)
            if d_idx >= 0:
                prev_chunk = prev_chunk[d_idx + len(delim):]
        words = prev_chunk.split()
        if not words:
            return ""

        reg = TypeRegistry.get_instance()
        dest_triggers = set(reg.get_egress_tokens() or ())

        for word in reversed(words[-8:]):
            w = word.strip(".,!?:;'\"").lower()
            if not w.isalpha():
                trailing = []
                for ch in reversed(w):
                    if ch.isalpha():
                        trailing.append(ch)
                    else:
                        break
                w = "".join(reversed(trailing)).lower()
            if not w:
                continue
            if "." in word or "/" in word or "\\" in word:
                continue
            if w in dest_triggers:
                return "dest"
        return ""

    @property
    def source_files(self) -> List[str]:
        if self.parameters.get("source_uris"):
            return list(self.parameters["source_uris"])
        file_candidates = [
            (pos, val) for pos, kind, val in self.ordered_literals
            if kind in ("file_asset", "quoted_str") and ("." in val or "/" in val or "\\" in val)
        ]
        if not file_candidates:
            return []
        srcs = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) == "src"]
        if srcs:
            return srcs
        unmarked = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) != "dest"]
        return [unmarked[0]] if unmarked else [file_candidates[0][1]]

    @property
    def dest_files(self) -> List[str]:
        if self.parameters.get("dest_uris"):
            return list(self.parameters["dest_uris"])
        file_candidates = [
            (pos, val) for pos, kind, val in self.ordered_literals
            if kind in ("file_asset", "quoted_str") and ("." in val or "/" in val or "\\" in val)
        ]
        if len(file_candidates) < 1:
            return []
        dests = [val for pos, val in file_candidates if self._asset_direction(self.prompt, pos) == "dest"]
        if dests:
            return dests
        reg = TypeRegistry.get_instance()
        prompt_words = CellTokenizer.tokenize_prompt(self.prompt or "")
        has_prompt_egress = bool(prompt_words & set(reg.get_egress_tokens() or ()))
        if not has_prompt_egress:
            return []
        src_set = set(self.source_files)
        return [val for pos, val in file_candidates if val not in src_set and self._asset_direction(self.prompt, pos) != "src"]

    @property
    def flags(self) -> Dict[str, Any]:
        """Declared boolean flags and their antonym inversions from configuration parameters."""
        res: Dict[str, Any] = {}
        for k, v in self.parameters.items():
            if isinstance(v, bool):
                res[k] = v
            elif isinstance(v, str) and v.lower() in ("true", "false"):
                res[k] = (v.lower() == "true")

        return res

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, val: str):
        self._prompt = val or ""
        self.used_indices = set()
        self.consumed_tokens = set()
        self.consumed_vars = set()
        self.consumed_members = set()
        self.consumed_keys = set()
        self.target_literals = set()
        t_sink = self._extract_target_sink(self._prompt)
        if t_sink:
            self.target_literals.add(t_sink)
        self.ordered_literals = self._extract_universal_literals(self._prompt)
        self.identifier_roles = self._build_identifier_role_map(self._prompt)
        self.sync_llm_slots()
        self.literal_clause_map = self._build_literal_clause_map(self._prompt, self.ordered_literals)

    @staticmethod
    def _extract_universal_literals(prompt: str) -> List[Tuple[int, str, str]]:
        if not prompt:
            return []

        try:
            from lattice import LatticeOrchestrator, TypeRegistry
            if LatticeOrchestrator.get_active_instance() is None:
                LatticeOrchestrator(auto_load=False)
        except Exception:
            from lattice import TypeRegistry

        spans: List[Tuple[int, str, str]] = []
        n = len(prompt)

        # 1. Quoted literals: '...' or "..." -> kind "quoted_str"
        i = 0
        while i < n:
            ch = prompt[i]
            if ch in ("'", '"'):
                quote_char = ch
                j = i + 1
                while j < n and prompt[j] != quote_char:
                    if prompt[j] == '\\' and j + 1 < n:
                        j += 1
                    j += 1
                if j < n and prompt[j] == quote_char:
                    val = prompt[i + 1:j]
                    parts = val.rsplit(".", 1) if ("." in val and not val.startswith(".") and not val.endswith(".")) else ()
                    ext = parts[1].lower() if parts else ""
                    reg_fmts = TypeRegistry.get_instance().get_registered_supported_formats()
                    is_file = "/" in val or "\\" in val or (bool(ext) and (ext in reg_fmts or (not reg_fmts and ext.isalnum() and not ext.isdigit() and len(ext) <= 8)))
                    kind = "file_asset" if is_file else "quoted_str"
                    spans.append(LiteralSpan(i, kind, val, end_pos=j + 1))
                    i = j + 1
                    continue
            i += 1

        # 2. Word tokens for file assets and numerics (zero unquoted prose identifiers)
        words_with_pos: List[Tuple[int, str]] = []
        cur_word: List[str] = []
        w_start = None
        for idx, ch in enumerate(prompt):
            if not ch.isspace():
                if w_start is None:
                    w_start = idx
                cur_word.append(ch)
            else:
                if cur_word:
                    words_with_pos.append((w_start, "".join(cur_word)))
                    cur_word = []
                    w_start = None
        if cur_word and w_start is not None:
            words_with_pos.append((w_start, "".join(cur_word)))

        def _quoted_range(pos: int, length: int) -> bool:
            return any(
                getattr(s, "pos", s[0]) <= pos and pos + length <= getattr(s, "end_pos", s[0] + len(s[2]))
                for s in spans
            )

        for pos, raw_w in words_with_pos:
            w = raw_w.rstrip(".,;:)")
            # Handle leading brackets and quotes
            leading_strip = 0
            while leading_strip < len(w) and w[leading_strip] in ("(", "[", "{", "\"", "'"):
                leading_strip += 1
            if leading_strip > 0:
                w = w[leading_strip:]
                pos = pos + leading_strip
            w = w.rstrip(")]}\"'.,;:)")
            if not w:
                continue

            if _quoted_range(pos, len(w)):
                continue

            # Path or filename token (domain-agnostic, zero hardcoded extensions)
            if "/" in w or "\\" in w:
                spans.append(LiteralSpan(pos, "file_asset", w, end_pos=pos + len(w)))
                continue
            if "." in w and not w.startswith(".") and not w.endswith("."):
                parts = w.rsplit(".", 1)
                ext = parts[1].lower()
                reg_fmts = TypeRegistry.get_instance().get_registered_supported_formats()
                if ext.isalnum() and not ext.isdigit() and (ext in reg_fmts or (not reg_fmts and len(ext) <= 8)):
                    spans.append(LiteralSpan(pos, "file_asset", w, end_pos=pos + len(w)))
                    continue

            # Dimension notation ("5x5", "800x600"): emit each extent as its own numeric span
            _dim_parts = _split_dimension_token(w)
            if _dim_parts:
                for _off, _txt in _dim_parts:
                    spans.append(LiteralSpan(pos + _off, "numeric", _txt, end_pos=pos + _off + len(_txt)))
                continue

            # Numeric tokens (including comma-separated integers like "1,000" or "10,000")
            num_candidate = w.replace(",", "") if ("," in w and all(part.isdigit() for part in w.split(","))) else w
            try:
                float(num_candidate)
                spans.append(LiteralSpan(pos, "numeric", num_candidate, end_pos=pos + len(num_candidate)))
                continue
            except ValueError:
                pass

            # (dimension notation is handled above; everything else is a single-unit numeric token)

        # 3. Predicate / filter comparison expressions (e.g. "value > 100", "score <= 50", "x == 1") without regex
        for op in (">=", "<=", "==", "!=", ">", "<"):
            idx = 0
            while True:
                op_pos = prompt.find(op, idx)
                if op_pos == -1:
                    break
                if op in ("<", ">") and op_pos + 1 < len(prompt) and prompt[op_pos + 1] == "=":
                    idx = op_pos + 1
                    continue
                lhs_part = prompt[:op_pos].rstrip()
                lhs_tokens = lhs_part.split()
                if lhs_tokens:
                    lhs = lhs_tokens[-1].strip("(),[]{}")
                    if lhs.isidentifier():
                        rhs_part = prompt[op_pos + len(op):].lstrip()
                        rhs_tokens = rhs_part.split()
                        if rhs_tokens:
                            rhs = rhs_tokens[0].strip("(),;:?[]{}")
                            if rhs and rhs not in ("=", ">", "<", "!", ":"):
                                expr_str = f"{lhs} {op} {rhs}"
                                start_pos = prompt.rfind(lhs, 0, op_pos)
                                if start_pos != -1:
                                    spans.append(LiteralSpan(start_pos, "expr", expr_str, end_pos=start_pos + len(expr_str)))
                idx = op_pos + len(op)

        # Order strictly by character position in prompt
        spans.sort(key=lambda x: x[0])

        # Deduplicate identical literals at the same character position
        seen = set()
        deduped_spans: List[Tuple[int, str, str]] = []
        for pos, kind, val in spans:
            key = (pos, str(val).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped_spans.append((pos, kind, val))
        return deduped_spans

    @staticmethod
    def _is_bare_identifier_token(w: str) -> bool:
        """Domain-agnostic check if token is syntactically a bare code identifier."""
        if not w:
            return False
        clean = str(w).strip(" \t\n\r\"'.,;:()[]{}")
        return bool(clean and clean.isidentifier() and not keyword.iskeyword(clean.lower()))

    @staticmethod
    def extract_identifier_groups(prompt: str) -> List[IdentifierGroup]:
        """Universal stub: zero prompt-based role grouping heuristics; resolved by LLM in neural profiles."""
        return []

    @staticmethod
    def _build_identifier_role_map(prompt: str) -> Dict[int, Set[str]]:
        """
        Universal stub: domain-agnostic role mapping.
        Pure symbolic profiles do not guess roles from prose;
        neural profiles inject formal roles via IntentCompiler slots in Layer 0.
        """
        return {}

    def sync_llm_slots(self) -> None:
        """
        Synchronizes semantic symbol and entity slots (extracted by neural profiles
        in Layer 0) into ordered_literals and identifier_roles.
        Preserves pure symbolic integrity: if llm_slots is empty (e.g. Profile 0 / A),
        no identifiers are injected.
        """
        if not getattr(self, "llm_slots", None) or not getattr(self, "_prompt", None):
            return

        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        proj_tokens = reg.get_projection_tokens()
        reg_types = set(reg.get_registered_types())

        # Only process formal variables extracted by the Intent Compiler
        raw_vars = self.llm_slots.get("variables") or []
        if isinstance(raw_vars, (str, bytes)):
            raw_vars = [raw_vars]

        candidate_symbols = set()
        for v in raw_vars:
            if isinstance(v, str) and v.strip():
                candidate_symbols.add(v.strip())

        if getattr(self, "target_key", None):
            candidate_symbols.add(str(self.target_key).strip())

        filtered_symbols = set()
        for sym in candidate_symbols:
            s_low = sym.lower()
            if keyword.iskeyword(s_low) or s_low in reg_types:
                continue
            if sym.isidentifier():
                filtered_symbols.add(sym)

        if not filtered_symbols:
            return

        words_with_pos: List[Tuple[int, str]] = []
        cur_word: List[str] = []
        w_start = None
        for idx, ch in enumerate(self._prompt):
            if not ch.isspace():
                if w_start is None:
                    w_start = idx
                cur_word.append(ch)
            else:
                if cur_word:
                    words_with_pos.append((w_start, "".join(cur_word)))
                    cur_word = []
                    w_start = None
        if cur_word and w_start is not None:
            words_with_pos.append((w_start, "".join(cur_word)))

        existing_spans = {
            (getattr(s, "pos", s[0]), str(getattr(s, "val", s[2])).strip().lower())
            for s in self.ordered_literals
        }

        new_spans = []
        for pos, raw_w in words_with_pos:
            w_clean = raw_w.strip(" '\".,;:()[]{}=:")
            if not w_clean:
                continue
            if w_clean in filtered_symbols:
                w_pos = self._prompt.find(w_clean, pos)
                if w_pos == -1:
                    w_pos = pos
                key = (w_pos, w_clean.lower())
                if key not in existing_spans:
                    existing_spans.add(key)
                    span = LiteralSpan(w_pos, "identifier", w_clean, end_pos=w_pos + len(w_clean))
                    new_spans.append(span)
                    self.identifier_roles[w_pos] = set(proj_tokens)
                    if getattr(self, "target_key", None) and w_clean.lower() == str(self.target_key).strip().lower():
                        self.target_literals.add(w_clean.lower())

        if new_spans:
            self.ordered_literals.extend(new_spans)
            self.ordered_literals.sort(key=lambda s: getattr(s, "pos", s[0]))
            self.literal_clause_map = self._build_literal_clause_map(self._prompt, self.ordered_literals)

    def _get_clause_idx_for_pos(self, pos: int) -> int:
        """Map a character position in the prompt to its enclosing clause index."""
        if not getattr(self, "clause_spans", None):
            return 0
        for c_i, (s, e) in enumerate(self.clause_spans):
            if s <= pos <= e or (pos >= s and c_i == len(self.clause_spans) - 1):
                return c_i
        return 0

    def _current_cell_provenance_tokens(self, cell: Optional[Any] = None) -> Set[str]:
        """Prompt clause + matched-subgoal tokens active for the given cell."""
        toks: Set[str] = set()
        if cell is not None:
            sub_toks = getattr(cell, "matched_subgoal_tokens", None)
            if sub_toks:
                toks.update(str(t).lower() for t in sub_toks if t)
        cl_idx = getattr(cell, "matched_clause_idx", None) if cell is not None else None
        if cl_idx is None:
            cl_idx = self.current_cell_clause_idx
        if cl_idx is not None and 0 <= cl_idx < len(self.clause_spans) and self._prompt:
            start, end = self.clause_spans[cl_idx]
            cl_txt = self._prompt[start:end]
            toks.update(t.lower() for t in CellTokenizer.tokenize_prompt(cl_txt))
        return toks

    def declare_variable(self, name: str, port_sig: Union[PortSignature, AlgebraicSignature, Any], expr: str = "", cell: Optional[Any] = None):
        if not name:
            return
        if isinstance(port_sig, AlgebraicSignature):
            port_sig = PortSignature(name=name, signature=port_sig)
        elif not isinstance(port_sig, PortSignature):
            port_sig = PortSignature(name=name, signature=AlgebraicSignature(str(port_sig), "any"))
        self.variables[name] = (port_sig, expr or name)
        self.scope[name] = port_sig
        if hasattr(self, "pipeline_vars"):
            self.pipeline_vars.add(name)
        if cell is not None:
            self.var_sources[name] = cell
        if cell is not None or self.current_cell_clause_idx is not None:
            prov_toks = self._current_cell_provenance_tokens(cell)
            if prov_toks:
                self.var_clause_tokens[name] = prov_toks

    def get_variable_name(self, port_sig: PortSignature) -> Optional[str]:
        """Finds in-scope variable that unifies with port_sig."""
        for v_name, (v_sig, _) in reversed(list(self.variables.items())):
            if v_sig.unifies_with(port_sig):
                return v_name
        return None

    def _project_semantic_slot(self, param_name: str = "", role_label: str = "") -> Optional[str]:
        """
        Semantic Slot Projection:
        Projects the port's DECLARED semantic role (or, absent a declared role,
        the port's parameter name) onto unconsumed prompt tokens via dense
        vector cosine similarity.
        Guard: candidate words morphologically related to the port's own name
        are EXCLUDED — a slot's label is not a proxy for its value's meaning
        (measured: the word "named" winning a `name` slot over the semantically
        correct X/Y/Z because of near-identical spelling).
        Contains ZERO hardcoded keyword tuples, ZERO token distance hacks.
        """
        if not self.prompt:
            return None

        # Universal grammatical function words derived from registry / corpus
        _FUNCTION_WORDS = TypeRegistry.get_instance().get_function_words()

        # Candidate word tokens from prompt, excluding self-referential collisions, function words, and file assets
        file_assets = {val.lower() for _, kind, val in self.ordered_literals if kind == "file_asset"}
        p_stem = normalize_token((param_name or "").lower()) if param_name else ""
        words = []
        for w in self.prompt.strip().split():
            clean_w = w.strip(" '\".,;:()[]{}=:")
            w_lower = clean_w.lower()
            if len(clean_w) < 2 or w_lower in self.consumed_tokens or w_lower in _FUNCTION_WORDS or w_lower in file_assets:
                continue
            if p_stem:
                w_stem = normalize_token(w_lower)
                if (
                    w_stem == p_stem
                    or w_lower.startswith(p_stem)
                    or p_stem.startswith(w_stem)
                ):
                    continue
            words.append(clean_w)

        if not words:
            return None

        target_label = role_label or param_name or "parameter"
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            import numpy as np

            mm = ModelManager.get_instance()
            if mm.profile is not None:
                e_param = np.array(mm.get_embedding(target_label), dtype=np.float32)
                p_norm = np.linalg.norm(e_param)
                if p_norm > 0:
                    e_param = e_param / p_norm
                    best_word = None
                    best_sim = -1.0
                    word_embs = mm.get_embeddings(words, mode="query")
                    for w, raw_e in zip(words, word_embs):
                        e_w = np.array(raw_e, dtype=np.float32)
                        w_norm = np.linalg.norm(e_w)
                        if w_norm > 0:
                            sim = float(np.dot(e_param, e_w / w_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_word = w
                    if best_word is not None and best_sim > 0.35:
                        self.consumed_tokens.add(best_word.lower())
                        return best_word
        except Exception as e:
            logger.debug(f"[UNIFICATION] Semantic slot projection error: {e}")

        return None

    def _resolve_enum_constant(self, domain_spec: str, port_state: str = "") -> Optional[str]:
        """
        Resolves an Enum constant dynamically by matching prompt intent against candidate flags.
        Domain-agnostic: uses domain_spec (e.g. 'cv2.COLOR_*') and module reflection.
        Contains ZERO library-specific keywords or hardcoded bonuses.
        Grounds selection in:
          1. Continuous vector embedding cosine similarity
          2. Directional transition alignment (X2Y matching prompt target intent)
          3. Semantic token overlap and Occam's razor parsimony
        """
        if not domain_spec or "*" not in domain_spec:
            return None

        clean_spec = domain_spec.replace("_*", "").replace("*", "")
        parts = clean_spec.rsplit(".", 1)
        if len(parts) != 2:
            return None
        mod_name, prefix = parts

        try:
            import importlib
            mod = importlib.import_module(mod_name)
        except Exception as e:
            return None

        candidates = [name for name in dir(mod) if name.startswith(f"{prefix}_")]
        if not candidates:
            return None

        sorted_candidates = sorted(candidates)

        # 1. Continuous vector embedding similarity if ModelManager is active (with cached constant matrix)
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            import numpy as np

            mm = ModelManager.get_instance()
            if mm.profile is not None and self.prompt:
                emb_id = _embedder_cache_identity(mm)
                text_key = (emb_id, "query", self.prompt)
                enum_key = (emb_id, domain_spec)
                cand_tuple = tuple(sorted_candidates)

                with _ENUM_EMBEDDING_LOCK:
                    e_prompt = _TEXT_EMBEDDING_CACHE.get(text_key)
                if e_prompt is None:
                    raw_p = np.asarray(mm.get_embedding(self.prompt), dtype=np.float32)
                    p_norm = float(np.linalg.norm(raw_p)) if raw_p.ndim == 1 and raw_p.size > 0 else 0.0
                    if p_norm <= 0.0:
                        return None
                    e_prompt = raw_p / p_norm
                    with _ENUM_EMBEDDING_LOCK:
                        if len(_TEXT_EMBEDDING_CACHE) >= 512:
                            try:
                                del _TEXT_EMBEDDING_CACHE[next(iter(_TEXT_EMBEDDING_CACHE))]
                            except (StopIteration, KeyError):
                                _TEXT_EMBEDDING_CACHE.clear()
                        _TEXT_EMBEDDING_CACHE[text_key] = e_prompt

                with _ENUM_EMBEDDING_LOCK:
                    cached_entry = _ENUM_EMBEDDING_CACHE.get(enum_key)
                if cached_entry is not None and cached_entry[0] == cand_tuple:
                    valid_cands, cand_matrix = cached_entry[1], cached_entry[2]
                else:
                    cand_texts = [cand.replace("_", " ").lower() for cand in cand_tuple]
                    cand_embs = mm.get_embeddings(cand_texts, mode="query")
                    valid_cands_l: List[str] = []
                    valid_vecs: List[Any] = []
                    for cand, raw_e in zip(cand_tuple, cand_embs):
                        vec = np.asarray(raw_e, dtype=np.float32)
                        if vec.ndim == 1 and vec.size > 0:
                            c_norm = float(np.linalg.norm(vec))
                            if c_norm > 0.0:
                                valid_cands_l.append(cand)
                                valid_vecs.append(vec / c_norm)
                    if not valid_vecs:
                        return None
                    valid_cands = tuple(valid_cands_l)
                    cand_matrix = np.stack(valid_vecs, axis=0).astype(np.float32, copy=False)
                    with _ENUM_EMBEDDING_LOCK:
                        if len(_ENUM_EMBEDDING_CACHE) >= 256:
                            try:
                                del _ENUM_EMBEDDING_CACHE[next(iter(_ENUM_EMBEDDING_CACHE))]
                            except (StopIteration, KeyError):
                                _ENUM_EMBEDDING_CACHE.clear()
                        _ENUM_EMBEDDING_CACHE[enum_key] = (cand_tuple, valid_cands, cand_matrix)

                if cand_matrix.shape[0] > 0 and cand_matrix.shape[1] == e_prompt.shape[0]:
                    sims = cand_matrix @ e_prompt
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim > 0.30:
                        return f"{mod_name}.{valid_cands[best_idx]}"
        except Exception as e:
            logger.debug("suppressed: %s", e, exc_info=False)

        return None

    def is_llm_slot_grounded(
        self,
        slot_val: Any,
        port_sig: Optional[PortSignature] = None,
        cell: Optional[Any] = None,
    ) -> bool:
        """
        Verify an LLM-supplied slot value does not silently shadow an unconsumed
        prompt literal of the same channel in the active clause.
        """
        return _sb_is_slot_grounded_in_clause(
            self,
            slot_val,
            port_sig=port_sig,
            cell=cell,
            registry=TypeRegistry.get_instance(),
        )

    def plan_bipartite_literal_assignments(
        self,
        cells: List[Cell],
        registry: Optional[TypeRegistry] = None,
    ) -> Dict[Tuple[int, str], Tuple[int, str]]:
        """
        Globally matches unbound literal-eligible ports across `cells` against
        `self.ordered_literals` using minimum-cost bipartite matching.
        Guarantees order-independent literal grounding regardless of port or
        literal iteration order when channel/local-context affinities discriminate.
        """
        ordered = getattr(self, "ordered_literals", None) or ()
        if not cells or not ordered:
            return {}

        reg = registry or TypeRegistry.get_instance()
        prompt_txt = self._prompt or ""
        prompt_low = prompt_txt.lower()
        clause_spans = getattr(self, "clause_spans", None) or ()
        num_prompt_clauses = len(clause_spans)
        lit_clause_map = getattr(self, "literal_clause_map", None) or {}
        served_clauses = getattr(self, "pipeline_served_clauses", None)
        clauses = CellTokenizer.split_prompt_clauses(prompt_txt) if prompt_txt else []
        all_clause_toks = [
            {t.lower() for t in CellTokenizer.tokenize_prompt(cl)}
            for cl in clauses if cl.strip()
        ]
        num_clauses_nonempty = len(all_clause_toks)

        role_carriers = reg.get_declared_role_carriers()
        proj_roles_set = reg.get_roles_for_category("projection") | set(reg.get_projection_tokens() or ())
        proj_states_set = reg.get_states_by_property("is_projection", True)
        key_roles_set = reg.get_roles_for_category("key")
        path_roles = reg.get_roles_for_category("path")
        cond_states = reg.get_states_by_property("is_condition", True)
        sink_roles = reg.get_terminal_roles_set() | reg.get_roles_for_category("destination") | {"dest_sink"}
        dest_states = (
            set(reg.get_terminal_states() or ())
            | reg.get_states_by_property("is_materialized", True)
            | reg.get_states_by_property("is_terminal", True)
            | reg.get_states_by_property("is_destination_identifier", True)
        )
        source_roles = reg.get_ingress_roles_set() | reg.get_roles_for_category("source") | {"source_data"}
        source_states = reg.get_states_by_property("is_source_identifier", True)
        dest_target_states = reg.get_states_by_property("is_target_identifier", True)
        proj_tokens = reg.get_projection_tokens()
        egress_tokens = reg.get_egress_tokens()
        reg_types = reg.get_registered_types()
        target_lits = {str(tl).lower() for tl in (getattr(self, "target_literals", None) or ())}
        preferred_dest_uris = {str(u).strip() for u in (self.parameters.get("dest_uris") or ())}
        preferred_src_uris = {str(u).strip() for u in (self.parameters.get("source_uris") or ())}

        def _port_own_tokens_bp(ps: Any) -> Set[str]:
            toks: Set[str] = set()
            pn = getattr(ps, "name", None)
            if pn:
                toks |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(pn))}
                toks.add(str(pn).lower())
            ps_st = str(getattr(ps, "state", "") or "").lower()
            if ps_st not in ("any", "default", ""):
                toks |= {t.lower() for t in CellTokenizer.tokenize_identifier(ps_st)}
            return toks

        lit_local_toks: List[Set[str]] = []
        lit_clause_toks: List[Set[str]] = []
        for idx, (lit_pos, kind, val) in enumerate(ordered):
            lit_len = len(str(val)) if val is not None else 0
            lit_end = lit_pos + lit_len
            cl_start_bound, cl_end_bound = 0, len(prompt_txt)
            for cs_s, cs_e in clause_spans:
                if cs_s <= lit_pos <= cs_e:
                    cl_start_bound, cl_end_bound = cs_s, cs_e
                    break
            cl_chunk = prompt_txt[cl_start_bound:cl_end_bound] if prompt_txt else ""
            lit_clause_toks.append({t.lower() for t in CellTokenizer.tokenize_prompt(cl_chunk)} if cl_chunk else set())

            prev_same_end: Optional[int] = None
            next_same_start: Optional[int] = None
            if kind == "numeric":
                for _o_pos, _o_kind, _o_val in ordered:
                    if _o_kind != "numeric" or not (cl_start_bound <= _o_pos <= cl_end_bound):
                        continue
                    if _o_pos < lit_pos:
                        _o_end = _o_pos + len(str(_o_val))
                        if _o_end <= lit_pos and (prev_same_end is None or _o_end > prev_same_end):
                            prev_same_end = _o_end
                    elif _o_pos > lit_pos:
                        if next_same_start is None or _o_pos < next_same_start:
                            next_same_start = _o_pos
            if prev_same_end is not None:
                raw_left = prompt_txt[prev_same_end:lit_pos].lower()
                left_words = [w for w in raw_left.split() if w]
                left_win = " ".join(left_words[1:]) if len(left_words) >= 2 else raw_left
            else:
                left_win = prompt_txt[max(cl_start_bound, lit_pos - 32):lit_pos].lower()
            right_win = ""
            if lit_len > 0 and lit_end < cl_end_bound:
                raw_right_end = (
                    next_same_start
                    if (next_same_start is not None and next_same_start > lit_end)
                    else min(cl_end_bound, lit_end + 24)
                )
                raw_right_full = prompt_txt[lit_end:raw_right_end].lower()
                cut_at = len(raw_right_full)
                for _r_i, _r_ch in enumerate(raw_right_full):
                    if _r_ch in (",", ";", "\n"):
                        cut_at = _r_i
                        break
                right_words = [w for w in raw_right_full[:cut_at].split() if w]
                if next_same_start is not None and right_words:
                    right_words = right_words[:-1]
                right_win = " ".join(right_words[:2])
            l_toks = set(CellTokenizer.tokenize_prompt(left_win))
            if right_win:
                l_toks |= set(CellTokenizer.tokenize_prompt(right_win))
            lit_local_toks.append(l_toks)

        forbidden = 1e9
        port_meta: List[Tuple[int, str, bool, bool]] = []
        cost_matrix: List[List[float]] = []
        unassign_costs: List[float] = []
        _bp_cell_id_counts: Dict[str, int] = {}
        for _bp_c in cells:
            _bp_cid_k = getattr(_bp_c, "cell_id", None)
            if _bp_cid_k:
                _bp_cell_id_counts[_bp_cid_k] = _bp_cell_id_counts.get(_bp_cid_k, 0) + 1

        for step_idx, cell in enumerate(cells):
            cell_stage = getattr(cell, "stage", None)
            cell_inputs = getattr(cell, "inputs", {}) or {}
            cell_tokens = getattr(cell, "token_set", None) or set()
            cell_cl = getattr(cell, "matched_clause_idx", None)
            is_formal_step = (
                bool(getattr(cell, "_formal_step", False))
                or (getattr(cell, "matched_clause_idx", None) is not None and not prompt_txt)
            )
            if is_formal_step:
                continue

            exp_bindings = getattr(cell, "explicit_port_bindings", None) or {}
            step_slots = (
                self.step_llm_slots.get(step_idx, {})
                if isinstance(getattr(self, "step_llm_slots", None), dict)
                else {}
            )
            _bp_cid = getattr(cell, "cell_id", None)
            cell_slots = (
                self.llm_slots.get(_bp_cid, {})
                if _bp_cid
                and _bp_cell_id_counts.get(_bp_cid, 0) <= 1
                and isinstance(getattr(self, "llm_slots", None), dict)
                else {}
            )
            if not isinstance(step_slots, dict):
                step_slots = {}
            if not isinstance(cell_slots, dict):
                cell_slots = {}

            cell_has_path = any(_lattice_is_path_port(p, reg) for p in cell_inputs.values())
            cell_mut = str(getattr(cell, "mutation_type", "") or "").strip().lower()
            cell_out_states = {
                str(getattr(getattr(o, "signature", o), "state", "") or "").strip().lower()
                for o in (getattr(cell, "outputs", None) or {}).values()
            }
            c_nr = getattr(cell, "node_role", "")
            is_sink_ctx = (
                (c_nr in sink_roles)
                or (cell_mut and reg.is_terminal_role(cell_mut))
                or bool(cell_out_states & dest_states)
                or (not getattr(cell, "outputs", None))
                or (cell_stage == 3)
            )
            is_source_ctx = (
                (c_nr in source_roles)
                or (cell_mut and reg.is_ingress_role(cell_mut))
                or (len(cell_inputs) == 0)
                or (cell_stage == 1)
            )

            other_port_toks: Dict[str, Set[str]] = {}
            for op_name, op_sig in cell_inputs.items():
                op_tn = str(getattr(getattr(op_sig, "signature", op_sig), "type_name", "") or "").strip().lower()
                op_sk = str(getattr(op_sig, "structure_kind", "") or "").strip().lower()
                if op_sk == "handle" or reg.is_composite_carrier(op_tn) or reg.is_abstract_carrier(op_tn):
                    continue
                other_port_toks[op_name] = _port_own_tokens_bp(op_sig)

            for p_name, p_sig in cell_inputs.items():
                if (
                    p_name in exp_bindings
                    or p_name in self.parameters
                    or (isinstance(getattr(self, "hyperparameters", None), dict) and p_name in self.hyperparameters)
                    or p_name in step_slots
                    or p_name in cell_slots
                ):
                    continue
                if _is_macro_block_slot(cell, p_name):
                    continue
                is_id_p, _ = _classify_template_identifier_port(cell, p_name)
                if is_id_p:
                    continue

                t_name = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or getattr(p_sig, "type_name", "") or "").strip().lower()
                p_role = str(getattr(p_sig, "port_role", None) or getattr(p_sig, "role", None) or getattr(p_sig, "derived_role", None) or "").strip().lower()
                p_state = str(getattr(p_sig, "state", "") or "").strip().lower()
                p_sk = str(getattr(p_sig, "structure_kind", "") or "").strip().lower()
                is_path = _lattice_is_path_port(p_sig, reg) or (bool(p_role) and p_role in path_roles)

                is_carrier_port = (
                    not is_path
                    and (
                        p_role in role_carriers
                        or p_sk == "handle"
                        or reg.is_composite_carrier(t_name)
                        or reg.is_abstract_carrier(t_name)
                    )
                )
                if is_carrier_port:
                    continue

                is_bool = reg.is_subtype(t_name, "bool")
                if is_bool:
                    continue

                _sc_d = getattr(p_sig, "shape_contract", None) or {}
                _t_len = None
                if isinstance(_sc_d, dict):
                    if isinstance(_sc_d.get("len"), int) and _sc_d["len"] >= 2:
                        _t_len = _sc_d["len"]
                    elif isinstance(_sc_d.get("shape"), (list, tuple)) and len(_sc_d["shape"]) == 1 and isinstance(_sc_d["shape"][0], int) and _sc_d["shape"][0] >= 2:
                        _t_len = _sc_d["shape"][0]
                if _t_len is not None or t_name == "tuple" or t_name.startswith("tuple["):
                    continue

                is_num = reg.is_subtype(t_name, "numeric") and not is_bool
                is_str = reg.is_subtype(t_name, "str") or reg.is_declared_top(t_name)
                is_column_port = (
                    getattr(p_sig, "binds", "") in key_roles_set
                    or p_state in proj_states_set
                    or p_role in proj_roles_set
                )
                port_sink_ctx = is_sink_ctx or (p_role in sink_roles) or (p_state in dest_states)
                port_source_ctx = is_source_ctx or (p_role in source_roles) or (p_state in source_states)
                port_op = not port_source_ctx and not port_sink_ctx

                allow_str_asset = (
                    not is_column_port
                    and (port_sink_ctx or port_source_ctx or (cell_stage in (1, 3)))
                    and is_str
                    and not getattr(p_sig, "enum_values", None)
                    and not cell_has_path
                    and (p_sig.required or port_sink_ctx)
                )
                is_path_or_asset = is_path or allow_str_asset
                is_cond_state = p_state in cond_states
                is_cond_expr = is_cond_state or (is_str and port_op and p_sig.required)
                is_op_str = port_op and is_str and not is_path_or_asset and not is_num

                if not (is_num or is_path_or_asset or is_cond_expr or is_op_str):
                    continue

                _int_only = is_num and reg.is_subtype(t_name, "int") and not reg.is_subtype("float", t_name)
                _prefers_float = is_num and reg.is_subtype(t_name, "float") and not _int_only
                own_toks = _port_own_tokens_bp(p_sig)
                enum_vals = {str(e).strip().lower() for e in p_sig.enum_values} if getattr(p_sig, "enum_values", None) else None
                is_cell_key_port = _port_matches_declared_ref(cell, "key_port", p_sig)
                is_proj_port = (
                    p_role in proj_roles_set
                    or p_role in key_roles_set
                    or p_role in reg.get_roles_for_category("feature")
                    or p_role in reg.get_roles_for_category("target")
                    or str(getattr(p_sig, "binds", "") or "").lower() in key_roles_set
                    or p_state in proj_states_set
                    or is_cell_key_port
                )
                is_setter_port = (
                    is_proj_port
                    or p_role in reg.get_roles_for_category("destination")
                    or p_state in dest_target_states
                    or getattr(p_sig, "param_kind", "") == "target"
                )

                opt_evidence: Set[str] = set()
                if not p_sig.required:
                    opt_evidence |= own_toks
                    _desc = str(getattr(p_sig, "description", "") or getattr(p_sig, "doc", "") or "")
                    if _desc:
                        opt_evidence |= {t.lower() for t in CellTokenizer.tokenize_prompt(_desc)}
                    if not is_num and cell_tokens:
                        opt_evidence |= {t.lower() for t in cell_tokens}

                pref_uris = preferred_dest_uris if port_sink_ctx else (preferred_src_uris if port_source_ctx else set())
                excl_uris = (
                    (preferred_src_uris - preferred_dest_uris) if port_sink_ctx
                    else ((preferred_dest_uris - preferred_src_uris) if port_source_ctx else set())
                )
                port_fmts = getattr(p_sig, "supported_formats", None)
                port_fmts_low = {str(f).strip().lower() for f in port_fmts if str(f).strip()} if port_fmts else set()
                has_wild_fmt = bool(port_fmts_low & {"*", "any"})

                row: List[float] = []
                for idx, (lit_pos, kind, val) in enumerate(ordered):
                    val_str = str(val).strip()
                    val_low = val_str.lower()
                    lit_cl = lit_clause_map.get(idx)

                    if num_prompt_clauses > 1 and cell_cl is not None and lit_cl is not None and lit_cl != cell_cl:
                        if not p_sig.required or (served_clauses is not None and lit_cl in served_clauses):
                            row.append(forbidden)
                            continue

                    feasible = False
                    channel_bonus = 0.0
                    if is_num:
                        if kind == "numeric":
                            ok_num = True
                            if _int_only:
                                try:
                                    f_v = float(val_str)
                                    if not f_v.is_integer() or "." in val_str:
                                        ok_num = False
                                except (ValueError, TypeError):
                                    ok_num = False
                            if ok_num and not p_sig.required:
                                cl_toks = lit_clause_toks[idx]
                                if enum_vals is not None and not (cl_toks & own_toks):
                                    ok_num = False
                                elif not (cl_toks & opt_evidence):
                                    ok_num = False
                                elif num_clauses_nonempty > 1:
                                    has_disc = False
                                    for tok in (cl_toks & opt_evidence):
                                        df = sum(1 for ct in all_clause_toks if tok in ct)
                                        if df == 1 or (tok in own_toks and df <= max(1, num_clauses_nonempty // 2)):
                                            has_disc = True
                                            break
                                    if not has_disc:
                                        ok_num = False
                            if ok_num:
                                feasible = True
                                if _prefers_float and "." in val_str:
                                    channel_bonus -= 0.5
                    elif is_path_or_asset:
                        is_fq = (kind == "file_asset") or (kind == "quoted_str" and len(val_str) > 1 and is_path)
                        matches_path = ExecutionContext._is_path_string(val_str)
                        if (is_fq or (is_path and matches_path)) and val_str not in excl_uris:
                            if enum_vals is not None and val_low not in enum_vals:
                                feasible = False
                            elif matches_path and "." in val_str and port_fmts_low and not has_wild_fmt:
                                ext = val_str.rsplit(".", 1)[1].lower()
                                feasible = (ext in port_fmts_low)
                            else:
                                feasible = True
                            if feasible and val_str in pref_uris:
                                channel_bonus -= 5.0
                    elif is_cond_expr and (kind == "expr" or (is_cond_state and kind == "quoted_str")):
                        feasible = True
                    elif is_op_str and kind in ("quoted_str", "identifier"):
                        is_path_s = ExecutionContext._is_path_string(val_str) or ("." in val_str and not val_str.replace(".", "").replace("-", "").isdigit())
                        if (
                            not is_path_s
                            and not (val_low in target_lits and not is_setter_port)
                            and not (is_proj_port and (keyword.iskeyword(val_low) or val_low in reg_types))
                            and (enum_vals is None or val_low in enum_vals)
                        ):
                            roles = self.identifier_roles.get(lit_pos, set())
                            is_var_sym = (
                                bool(roles & proj_tokens)
                                or (is_proj_port and kind == "quoted_str")
                                or (is_setter_port and bool(roles & (proj_tokens | reg.get_roles_for_category("destination") | egress_tokens)))
                            )
                            if (is_proj_port and is_var_sym) or (not is_proj_port and not is_var_sym):
                                if p_sig.required or enum_vals is not None or bool(opt_evidence & lit_clause_toks[idx]) or bool(own_toks & set(prompt_low.split())):
                                    feasible = True
                                    if is_proj_port and kind == "quoted_str":
                                        channel_bonus -= 0.2

                    if not feasible:
                        row.append(forbidden)
                        continue

                    l_toks = lit_local_toks[idx]
                    own_hits = len(l_toks & own_toks)
                    other_hits = 0
                    for op_n, op_toks in other_port_toks.items():
                        if op_n != p_name and op_toks and (l_toks & op_toks) and not (op_toks <= own_toks):
                            other_hits = max(other_hits, len(l_toks & op_toks))

                    if not p_sig.required and is_num and other_hits > own_hits and own_hits == 0:
                        row.append(forbidden)
                        continue

                    cost = 10.0 + channel_bonus
                    if cell_cl is not None and lit_cl == cell_cl:
                        cost -= 5.0
                    cost += -4.0 * min(own_hits, 3) + 3.0 * min(other_hits, 3)
                    if enum_vals is not None and val_low in enum_vals:
                        cost -= 4.0
                    if is_path_or_asset:
                        if port_sink_ctx:
                            cost += 0.01 * (len(ordered) - 1 - idx)
                        else:
                            cost += 0.01 * idx
                    else:
                        cost += 0.001 * idx
                    row.append(cost)

                port_meta.append((step_idx, p_name, is_path_or_asset, is_num))
                cost_matrix.append(row)
                unassign_costs.append(18.0 if p_sig.required else 7.0)

        if not cost_matrix:
            return {}

        assigned = _min_cost_bipartite_assign(cost_matrix, unassign_costs, forbidden=forbidden)
        plan: Dict[Tuple[int, str], Tuple[int, str]] = {}
        for row_i, col_j in assigned.items():
            step_idx, p_name, _, is_num = port_meta[row_i]
            _, kind, val = ordered[col_j]
            formatted_val = str(val) if kind == "numeric" or is_num else json.dumps(val)
            plan[(step_idx, p_name)] = (col_j, formatted_val)
        return plan

    def resolve_literal_for_port(
        self,
        port_sig: PortSignature,
        cell_stage: Optional[int] = None,
        cell_inputs: Optional[Dict[str, PortSignature]] = None,
        cell_tokens: Optional[Set[str]] = None,
        cell: Optional[Any] = None,
        excluded_keys: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Resolves a value for a port using parameters, declared defaults, or literals.
        Zero domain-specific keywords or hardcoded values.
        Categorically grounded in morphism stages:
          - Stage 1 (Initial / Ingestion morphism Env -> C): resolves environment assets
          - Stage 2 (Endomorphism C x P -> C): resolves operational parameters P
          - Stage 3 (Terminal / Egress morphism C -> Env): resolves output destination

        Literal-to-port assignment follows a strict type-channel discipline:
          file_asset literals bind ONLY to path-typed ports (or to plain str ports on
          cells that declare no path-typed port), quoted_str to str ports, numeric to
          numeric ports. `cell_inputs` enables the cross-port guard that prevents
          auxiliary str ports from stealing file assets on cells that own a path port.
        """
        if cell is not None:
            if cell_stage is None:
                cell_stage = getattr(cell, "stage", None)
            if cell_inputs is None:
                cell_inputs = getattr(cell, "inputs", None)
            if cell_tokens is None:
                cell_tokens = getattr(cell, "token_set", None)
        p_name = (port_sig.name or "").lower() if port_sig and getattr(port_sig, "name", None) else ""
        t_name = (port_sig.type_name or "").lower() if port_sig and getattr(port_sig, "type_name", None) else ""

        def _cell_has_path_port() -> bool:
            if not cell_inputs:
                return False
            return any(_lattice_is_path_port(p) for p in cell_inputs.values())

        registry = TypeRegistry.get_instance()
        is_bool = registry.is_subtype(t_name, "bool")
        is_num = registry.is_subtype(t_name, "numeric") and not is_bool
        is_str = registry.is_subtype(t_name, "str") or registry.is_declared_top(t_name)

        # 1. Parameter explicitly supplied in context or LLM slots
        is_scalar = _tp_is_true_scalar_port(port_sig, registry)

        def _format_asset_literal(raw_val: Any, fallback_json: bool = True) -> str:
            if t_name == "path":
                s_chk = str(raw_val).strip()
                if s_chk.startswith("Path(") and s_chk.endswith(")"):
                    return s_chk
                return _format_literal_value(raw_val, port_sig)
            if fallback_json:
                return json.dumps(raw_val)
            return json.dumps(raw_val) if isinstance(raw_val, str) else str(raw_val)

        if port_sig.name in self.parameters:
            val = self.parameters[port_sig.name]
            if not (is_scalar and isinstance(val, (list, tuple))):
                return _format_asset_literal(val, fallback_json=False)
        if hasattr(self, "hyperparameters") and isinstance(self.hyperparameters, dict) and port_sig.name in self.hyperparameters:
            cand_val = self.hyperparameters[port_sig.name]
            if not (is_scalar and isinstance(cand_val, (list, tuple))):
                is_req = getattr(port_sig, "required", True)
                if is_req:
                    return _format_literal_value(cand_val, port_sig)
                # Optional port: verify token provenance grounding before overriding declared default
                val_s = str(cand_val).strip("'\"")
                prompt_str = str(self.prompt or "")
                p_n_lower = port_sig.name.lower()
                c_toks = {t.lower() for t in cell_tokens} if cell_tokens else set()
                valid_nums = {v for _, k_lit, v in getattr(self, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}
                has_name = p_n_lower in c_toks or (len(p_n_lower) > 2 and p_n_lower in prompt_str.lower())
                if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
                    has_val = (val_s in valid_nums) or (val_s in prompt_str)
                else:
                    has_val = (val_s.lower() in c_toks) or (len(val_s) > 2 and val_s.lower() in prompt_str.lower())
                if has_name and has_val:
                    return _format_literal_value(cand_val, port_sig)
        _step_idx = getattr(self, "current_cell_step_idx", None)

        def _is_future_quoted_slot(sv: Any) -> bool:
            if not isinstance(sv, str):
                return False
            _c_cl = getattr(cell, "matched_clause_idx", None) if cell is not None else getattr(self, "current_cell_clause_idx", None)
            if _c_cl is None or not getattr(self, "ordered_literals", None) or not getattr(self, "literal_clause_map", None):
                return False
            _sv_l = sv.strip().strip("'\"").lower()
            if not _sv_l:
                return False
            _q_cls = {
                self.literal_clause_map.get(_qi)
                for _qi, (_, _qk, _qv) in enumerate(self.ordered_literals)
                if _qk == "quoted_str"
                and str(_qv).strip().lower() == _sv_l
                and self.literal_clause_map.get(_qi) is not None
            }
            if not _q_cls or min(_q_cls) <= _c_cl:
                return False
            _fut_map = getattr(self, "_pipeline_future_clauses_by_step", None)
            if isinstance(_fut_map, dict) and _step_idx is not None and _step_idx in _fut_map:
                return bool(_q_cls & _fut_map[_step_idx])
            return True

        if _step_idx is not None and isinstance(getattr(self, "step_llm_slots", None), dict):
            _step_s = self.step_llm_slots.get(_step_idx)
            if isinstance(_step_s, dict) and port_sig.name in _step_s:
                slot_val = _step_s[port_sig.name]
                if (
                    not (is_scalar and isinstance(slot_val, (list, tuple)))
                    and not _is_future_quoted_slot(slot_val)
                    and self.is_llm_slot_grounded(slot_val, port_sig, cell=cell)
                ):
                    return _format_literal_value(slot_val, port_sig)
        _cid_counts_rl = getattr(self, "_pipeline_cell_id_counts", None)
        if hasattr(self, "llm_slots") and isinstance(self.llm_slots, dict):
            _cid = getattr(cell, "cell_id", None) if cell is not None else None
            _cid_unique = (not isinstance(_cid_counts_rl, dict)) or (_cid_counts_rl.get(_cid, 0) <= 1)
            _cell_s = self.llm_slots.get(_cid) if (_cid and _cid_unique) else None
            if isinstance(_cell_s, dict) and port_sig.name in _cell_s:
                slot_val = _cell_s[port_sig.name]
                if (
                    not (is_scalar and isinstance(slot_val, (list, tuple)))
                    and not _is_future_quoted_slot(slot_val)
                    and self.is_llm_slot_grounded(slot_val, port_sig, cell=cell)
                ):
                    return _format_literal_value(slot_val, port_sig)

        # 1b. Global min-cost bipartite assignment plan (order-independent)
        _bp_plan = getattr(self, "_bipartite_literal_plan", None)
        _bp_key = (_step_idx, getattr(port_sig, "name", None)) if _step_idx is not None else None
        if _bp_plan and _bp_key is not None and _bp_key in _bp_plan:
            _bp_idx, _bp_val = _bp_plan[_bp_key]
            if _bp_idx not in self.used_indices:
                _bp_excl_ok = True
                if excluded_keys:
                    _bp_excl_low = {str(ek).strip().lower() for ek in excluded_keys if str(ek).strip()}
                    _bp_val_keys = {str(k).strip().lower() for k in (_extract_keys_from_val(_bp_val) or ())}
                    if _bp_val_keys & _bp_excl_low:
                        _bp_excl_ok = False
                if _bp_excl_ok:
                    self.used_indices.add(_bp_idx)
                    if t_name == "path":
                        return _format_asset_literal(_bp_val)
                    return _bp_val

        # 2. Dynamic Enum / Flag Constant Grounding via Domain Reflection & Vector Similarity
        if (t_name == "enum" or getattr(port_sig, "domain", None)) and self.prompt:
            domain_spec = getattr(port_sig, "domain", "") or ""
            resolved_enum = self._resolve_enum_constant(domain_spec, port_sig.state)
            if resolved_enum is not None:
                return resolved_enum

        # 3. Vector Polarity Projection for Boolean / Valuation Ports
        if is_bool and self.prompt:
            raw_quals = getattr(port_sig.signature, "qualifiers", [])
            qualifier_map = {}
            for q in (raw_quals or ()):
                if isinstance(q, (tuple, list)) and len(q) == 2:
                    qualifier_map[q[0]] = q[1]
                elif isinstance(q, str):
                    qualifier_map[q] = q
            pos_label = qualifier_map.get("positive", port_sig.name)
            neg_label = qualifier_map.get("negative", f"not {port_sig.name}")

            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                import numpy as np

                mm = ModelManager.get_instance()
                if mm.profile is not None:
                    prompt_tokens = CellTokenizer.tokenize_prompt(self.prompt)
                    if prompt_tokens:
                        e_pos = np.array(mm.get_embedding(pos_label), dtype=np.float32)
                        e_neg = np.array(mm.get_embedding(neg_label), dtype=np.float32)
                        norm_pos = np.linalg.norm(e_pos)
                        norm_neg = np.linalg.norm(e_neg)
                        if norm_pos > 0 and norm_neg > 0:
                            e_pos = e_pos / norm_pos
                            e_neg = e_neg / norm_neg

                            token_list = list(prompt_tokens)
                            batch_embs = mm.get_embeddings_batch(token_list, mode="query")
                            if len(batch_embs) > 0:
                                t_norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
                                nonzero = (t_norms.squeeze(-1) > 0)
                                if np.any(nonzero):
                                    normed_embs = batch_embs[nonzero] / t_norms[nonzero]
                                    pos_score = float(np.max(np.dot(normed_embs, e_pos)))
                                    neg_score = float(np.max(np.dot(normed_embs, e_neg)))
                                    token_match = any(w in self.prompt.lower() for w in (port_sig.name.lower(), pos_label.lower(), neg_label.lower()))
                                    if (token_match or max(pos_score, neg_score) > 0.65) and abs(pos_score - neg_score) > 0.05:
                                        return "True" if pos_score > neg_score else "False"
            except Exception as e:
                logger.debug(f"[UNIFICATION] Vector polarity projection fallback: {e}")

            # Profile 0: exact token match baseline
            prompt_words = {w.lower().strip(" '\".,;:()[]{}=:") for w in self.prompt.split()}
            if "true" in prompt_words and "false" not in prompt_words:
                return "True"
            if "false" in prompt_words and "true" not in prompt_words:
                return "False"
            p_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
            if p_ident_toks and p_ident_toks.issubset(prompt_words):
                return "True"

            if port_sig.required:
                if port_sig.default_value is not None:
                    return str(port_sig.default_value)
                return None
            return None

        # 4. Numeric literals & fixed-shape numeric tuple configuration ports.
        # A REQUIRED numeric port consumes the unconsumed numeric with strongest
        # local port-context evidence (preventing adjacent parameter literals from
        # being stolen out of order). An OPTIONAL numeric port consumes one
        # only with lexical EVIDENCE: the literal's prompt context must
        # intersect the consuming cell's declared vocabulary (identity tokens,
        # port name/state tokens).
        def _port_own_tokens(ps: Any) -> Set[str]:
            toks: Set[str] = set()
            pn = getattr(ps, "name", None)
            if pn:
                toks |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(pn))}
                toks.add(str(pn).lower())
            ps_st = str(getattr(ps, "state", "") or "").lower()
            if ps_st not in ("any", "default", ""):
                toks |= {t.lower() for t in CellTokenizer.tokenize_identifier(ps_st)}
            return toks

        def _literal_port_evidence(lit_pos: int, lit_val: Optional[str] = None) -> Tuple[int, int]:
            """Returns (own_local_hits, other_port_local_hits) in the Voronoi local window around lit_pos."""
            own_toks = _port_own_tokens(port_sig)
            p_text = self.prompt or ""
            lit_len = len(str(lit_val)) if lit_val is not None else 0
            lit_end = lit_pos + lit_len
            cl_start_bound, cl_end_bound = 0, len(p_text)
            for cs_s, cs_e in (getattr(self, "clause_spans", None) or ()):
                if cs_s <= lit_pos <= cs_e:
                    cl_start_bound, cl_end_bound = cs_s, cs_e
                    break
            prev_num_end: Optional[int] = None
            next_num_start: Optional[int] = None
            for _o_pos, _o_kind, _o_val in self.ordered_literals:
                if _o_kind != "numeric" or not (cl_start_bound <= _o_pos <= cl_end_bound):
                    continue
                if _o_pos < lit_pos:
                    _o_end = _o_pos + len(str(_o_val))
                    if _o_end <= lit_pos and (prev_num_end is None or _o_end > prev_num_end):
                        prev_num_end = _o_end
                elif _o_pos > lit_pos:
                    if next_num_start is None or _o_pos < next_num_start:
                        next_num_start = _o_pos
            if prev_num_end is not None:
                raw_left = p_text[prev_num_end:lit_pos].lower()
                left_words = [w for w in raw_left.split() if w]
                left_win = " ".join(left_words[1:]) if len(left_words) >= 2 else raw_left
            else:
                left_win = p_text[max(cl_start_bound, lit_pos - 32):lit_pos].lower()
            right_win = ""
            if lit_len > 0 and lit_end < cl_end_bound:
                raw_right_end = (
                    next_num_start
                    if (next_num_start is not None and next_num_start > lit_end)
                    else min(cl_end_bound, lit_end + 24)
                )
                raw_right_full = p_text[lit_end:raw_right_end].lower()
                cut_at = len(raw_right_full)
                for _r_i, _r_ch in enumerate(raw_right_full):
                    if _r_ch in (",", ";", "\n"):
                        cut_at = _r_i
                        break
                right_words = [w for w in raw_right_full[:cut_at].split() if w]
                if next_num_start is not None and right_words:
                    right_words = right_words[:-1]
                right_win = " ".join(right_words[:2])
            local_toks = set(CellTokenizer.tokenize_prompt(left_win))
            if right_win:
                local_toks |= set(CellTokenizer.tokenize_prompt(right_win))
            own_hits = len(local_toks & own_toks)
            other_hits = 0
            if cell_inputs:
                for op_name, op_sig in cell_inputs.items():
                    if op_name == getattr(port_sig, "name", None):
                        continue
                    op_tn = str(getattr(getattr(op_sig, "signature", op_sig), "type_name", "") or "").strip().lower()
                    op_sk = str(getattr(op_sig, "structure_kind", "") or "").strip().lower()
                    if is_num and (
                        op_sk == "handle"
                        or not (registry.is_subtype(op_tn, "numeric") or registry.is_subtype(op_tn, "sequence") or op_tn.startswith("tuple["))
                    ):
                        continue
                    op_toks = _port_own_tokens(op_sig)
                    if op_toks and (local_toks & op_toks) and not (op_toks <= own_toks):
                        other_hits = max(other_hits, len(local_toks & op_toks))
            return own_hits, other_hits

        if is_num:
            _int_only = (
                registry is not None
                and registry.is_subtype(t_name, "int")
                and not registry.is_subtype("float", t_name)
            )
            _prefers_float = (
                registry is not None
                and registry.is_subtype(t_name, "float")
                and not _int_only
            )
            lit_clause_map = getattr(self, "literal_clause_map", {}) or {}
            cell_cl = getattr(self, "current_cell_clause_idx", None)
            num_prompt_clauses = len(getattr(self, "clause_spans", ()) or ())
            clauses = CellTokenizer.split_prompt_clauses(self.prompt) if self.prompt else []
            scored_num_cands: List[Tuple[Tuple[int, int, int, int, int], int, str]] = []
            for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                if idx in self.used_indices or kind != "numeric":
                    continue
                if _int_only:
                    try:
                        f_v = float(val)
                        if not f_v.is_integer() or "." in str(val):
                            continue
                    except (ValueError, TypeError):
                        continue
                in_cur_clause = 1 if (
                    cell_cl is not None and idx in lit_clause_map and lit_clause_map[idx] == cell_cl
                ) else 0
                _served_cl = getattr(self, "pipeline_served_clauses", None)
                if (
                    cell_cl is not None
                    and num_prompt_clauses > 1
                    and idx in lit_clause_map
                    and lit_clause_map[idx] != cell_cl
                    and (
                        not port_sig.required
                        # A required port may only borrow a literal that belongs to a clause NO cell serves
                        # (an orphan literal). Otherwise it steals a sibling clause's argument, e.g. a blur
                        # sigma consuming the next clause's edge-detector threshold.
                        or (_served_cl is not None and lit_clause_map[idx] in _served_cl)
                    )
                ):
                    continue
                if not port_sig.required:
                    # Clause-bounded context: restrict evidence to the clause enclosing the literal
                    clause_chunk = self.prompt
                    cur_search = 0
                    for cl in clauses:
                        cl_start = self.prompt.find(cl, cur_search)
                        if cl_start != -1:
                            cl_end = cl_start + len(cl)
                            cur_search = cl_end
                            if cl_start <= lit_pos <= cl_end:
                                clause_chunk = cl
                                break

                    context_toks = {
                        t for t in CellTokenizer.tokenize_prompt(clause_chunk)
                    }

                    # A port with declared discrete enum_values is a categorical flag, not a continuous/count scalar.
                    # It must only bind if the port name itself is explicitly spoken in the context clause.
                    if getattr(port_sig, "enum_values", None):
                        port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                        if port_sig.name:
                            port_ident_toks.add(port_sig.name.lower())
                        if not (context_toks & port_ident_toks):
                            continue

                    evidence = set()
                    evidence |= {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                    if port_sig.name:
                        evidence.add(port_sig.name.lower())
                    _st = str(getattr(port_sig, "state", "") or "")
                    if _st.lower() not in ("any", "default", ""):
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_identifier(_st)}
                    _desc = str(getattr(port_sig, "description", "") or getattr(port_sig, "doc", "") or "")
                    if _desc:
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_prompt(_desc)}
                    if in_cur_clause and cell is not None:
                        _sem_tags = getattr(cell, "semantic_tags", None) or ()
                        for _st_tag in _sem_tags:
                            evidence |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(_st_tag))}

                    # Only fall back to cell-wide identity tokens if the port itself lacks specific descriptive vocabulary
                    if not evidence and cell_tokens:
                        evidence |= {t.lower() for t in cell_tokens}

                    intersecting = context_toks & evidence
                    if not intersecting:
                        continue

                    # Prompt-clause IDF weighting:
                    # A word that appears in the port's boilerplate description or in multiple prompt clauses
                    # is non-discriminative background vocabulary (e.g. 'contour' in a contour detection prompt).
                    # An optional numeric port must only bind if the intersection with context contains
                    # discriminative evidence (high clause IDF or specific port identifier match).
                    all_clause_toks = [
                        {t.lower() for t in CellTokenizer.tokenize_prompt(cl)}
                        for cl in clauses if cl.strip()
                    ]
                    num_clauses = len(all_clause_toks)
                    if num_clauses > 1:
                        port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)}
                        if port_sig.name:
                            port_ident_toks.add(port_sig.name.lower())

                        discriminative_mass = 0.0
                        has_discriminative_hit = False
                        for tok in intersecting:
                            df = sum(1 for c_toks in all_clause_toks if tok in c_toks)
                            idf = math.log(1.0 + (num_clauses + 1.0) / (df + 1.0))
                            is_ident = tok in port_ident_toks
                            # High-IDF token (local to 1 clause) or specific port identity token
                            if df == 1 or (is_ident and df <= max(1, num_clauses // 2)):
                                discriminative_mass += idf * (1.5 if is_ident else 1.0)
                                has_discriminative_hit = True

                        if not has_discriminative_hit or discriminative_mass < 0.5:
                            continue
                own_hits, other_hits = _literal_port_evidence(lit_pos, val) if self.prompt else (0, 0)
                if not port_sig.required and other_hits > own_hits and own_hits == 0:
                    continue
                _float_spec = 1 if (_prefers_float and "." in str(val)) else 0
                scored_num_cands.append(((in_cur_clause, 1 if own_hits > 0 else 0, 0 if other_hits > 0 else 1, _float_spec, -idx), idx, val))
            if scored_num_cands:
                scored_num_cands.sort(key=lambda item: item[0], reverse=True)
                _, chosen_idx, chosen_val = scored_num_cands[0]
                self.used_indices.add(chosen_idx)
                return chosen_val

        # 4a. Fixed-shape numeric tuple configuration ports (e.g. kernel_size=(k, k), size=(w, h), color=(r, g, b))
        _sc_dict = getattr(port_sig, "shape_contract", None) or {}
        _tuple_len = None
        if isinstance(_sc_dict, dict):
            _sc_len = _sc_dict.get("len")
            _sc_shape = _sc_dict.get("shape")
            if isinstance(_sc_len, int) and _sc_len >= 2:
                _tuple_len = _sc_len
            elif isinstance(_sc_shape, (list, tuple)) and len(_sc_shape) == 1 and isinstance(_sc_shape[0], int) and _sc_shape[0] >= 2:
                _tuple_len = _sc_shape[0]
        if _tuple_len is None and registry is not None:
            try:
                _raw_tn = str(getattr(port_sig, "type_name", "") or "").strip()
                if "[" in _raw_tn and "..." not in _raw_tn:
                    _parsed_t = TypeTerm.from_string(_raw_tn)
                    if (
                        isinstance(_parsed_t, GenericTypeTerm)
                        and _parsed_t.base.lower() == "tuple"
                        and 2 <= len(_parsed_t.args) <= 4
                        and all(registry.is_subtype(str(a).lower(), "numeric") for a in _parsed_t.args)
                    ):
                        _tuple_len = len(_parsed_t.args)
            except Exception:
                pass
        _p_role_low = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "role", "") or "").strip().lower()
        if (
            _tuple_len is not None
            and _tuple_len <= 4
            and (
                t_name == "tuple"
                or t_name.startswith("tuple[")
                or (registry is not None and registry.is_subtype(t_name, "sequence"))
            )
            and (_p_role_low in registry.get_config_roles_set() or port_sig.required)
        ):
            lit_clause_map = getattr(self, "literal_clause_map", {}) or {}
            cell_cl = getattr(self, "current_cell_clause_idx", None)
            clause_spans = getattr(self, "clause_spans", ()) or ()
            num_prompt_clauses = len(clause_spans)

            def _collect_tuple_cands(allowed_clauses: Optional[Set[int]] = None) -> List[Tuple[int, int, str, int, int]]:
                cands_out: List[Tuple[int, int, str, int, int]] = []
                for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                    if idx in self.used_indices or kind != "numeric":
                        continue
                    l_cl = lit_clause_map.get(idx)
                    if allowed_clauses is not None and l_cl is not None and l_cl not in allowed_clauses:
                        continue
                    own_hits, other_hits = _literal_port_evidence(lit_pos, val) if self.prompt else (0, 0)
                    if other_hits > own_hits and allowed_clauses is not None and cell_cl not in allowed_clauses:
                        continue
                    if other_hits > own_hits and allowed_clauses is None:
                        continue
                    cands_out.append((idx, lit_pos, val, own_hits, other_hits))
                return cands_out

            def _extract_contiguous_tuple(cands_in: List[Tuple[int, int, str, int, int]], allow_cross_clause: bool = False) -> Optional[str]:
                if len(cands_in) < _tuple_len:
                    return None
                for start_i in range(len(cands_in) - _tuple_len + 1):
                    slice_t = cands_in[start_i:start_i + _tuple_len]
                    if not allow_cross_clause and num_prompt_clauses > 1:
                        cl_ids = {lit_clause_map.get(sc[0]) for sc in slice_t}
                        if len(cl_ids) > 1:
                            continue
                    spans_clean = True
                    for k_i in range(len(slice_t) - 1):
                        between = (self.prompt or "")[slice_t[k_i][1]:slice_t[k_i + 1][1]]
                        if any(tok.isalpha() and tok.lower() not in ("x", "by", "and") for tok in between.split()):
                            spans_clean = False
                            break
                    if spans_clean:
                        for sc_item in slice_t:
                            self.used_indices.add(sc_item[0])
                        return "(" + ", ".join(str(sc_item[2]) for sc_item in slice_t) + ")"
                return None

            local_allowed = {cell_cl} if (cell_cl is not None and num_prompt_clauses > 1) else None
            t_cands = _collect_tuple_cands(local_allowed)
            if t_cands:
                # Prefer a locally-evidenced contiguous slice when available
                prioritized = sorted(
                    range(len(t_cands) - _tuple_len + 1),
                    key=lambda si: sum(t_cands[si + k][3] for k in range(_tuple_len)),
                    reverse=True,
                )
                for si in prioritized:
                    sub_res = _extract_contiguous_tuple(t_cands[si:si + _tuple_len])
                    if sub_res is not None:
                        return sub_res
                ev_cands = [tc for tc in t_cands if tc[3] > 0]
                if ev_cands and _tuple_len == 2 and port_sig.required:
                    chosen_t = ev_cands[0]
                    self.used_indices.add(chosen_t[0])
                    return f"({chosen_t[2]}, {chosen_t[2]})"

            # Cross-clause fallback for fixed-tuple configuration ports whose values
            # were introduced in an earlier clause sharing non-stopword descriptors or port tokens
            if (
                (port_sig.required or _p_role_low in registry.get_config_roles_set())
                and cell_cl is not None
                and num_prompt_clauses > 1
                and 0 <= cell_cl < num_prompt_clauses
                and self.prompt
            ):
                _s_a, _e_a = clause_spans[cell_cl]
                cur_cl_toks = {t.lower() for t in CellTokenizer.tokenize_prompt(self.prompt[_s_a:_e_a]) if len(t) > 2}
                port_state_toks = _port_own_tokens(port_sig) if callable(_port_own_tokens) else set()
                matching_clauses: List[Tuple[int, int]] = []
                for cj, (s_j, e_j) in enumerate(clause_spans):
                    if cj == cell_cl:
                        continue
                    cl_j_toks = {t.lower() for t in CellTokenizer.tokenize_prompt(self.prompt[s_j:e_j]) if len(t) > 2}
                    port_ov = len(port_state_toks & cl_j_toks)
                    if not port_sig.required and port_ov == 0:
                        continue
                    overlap_cnt = len(cur_cl_toks & cl_j_toks) + port_ov
                    if overlap_cnt >= (1 if port_ov > 0 else 2):
                        matching_clauses.append((overlap_cnt, cj))
                matching_clauses.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                for _, cj in matching_clauses:
                    cj_cands = _collect_tuple_cands({cj})
                    sub_res = _extract_contiguous_tuple(cj_cands)
                    if sub_res is not None:
                        return sub_res

        # Structural Morphism Context
        _reg_ctx = TypeRegistry.get_instance()
        port_role = str(getattr(port_sig, "port_role", None) or getattr(port_sig, "role", None) or "").strip().lower()
        is_path = _lattice_is_path_port(port_sig, _reg_ctx)
        path_roles = _reg_ctx.get_roles_for_category("path")
        is_path_role = is_path or (bool(port_role) and port_role in path_roles)
        sink_roles = _reg_ctx.get_terminal_roles_set() | _reg_ctx.get_roles_for_category("destination") | {"dest_sink"}
        dest_states = (
            set(_reg_ctx.get_terminal_states() or ())
            | _reg_ctx.get_states_by_property("is_materialized", True)
            | _reg_ctx.get_states_by_property("is_terminal", True)
            | _reg_ctx.get_states_by_property("is_destination_identifier", True)
        )
        cell_mut = str(getattr(cell, "mutation_type", "") or "").strip().lower() if cell is not None else ""
        cell_out_states = {
            str(getattr(getattr(o, "signature", o), "state", "") or "").strip().lower()
            for o in (getattr(cell, "outputs", None) or {}).values()
        } if cell is not None else set()
        term_stages = {3}
        is_sink_ctx = (
            (cell is not None and getattr(cell, "node_role", "") in sink_roles)
            or (cell_mut and _reg_ctx.is_terminal_role(cell_mut))
            or bool(cell_out_states & dest_states)
            or (cell is not None and not getattr(cell, "outputs", None))
            or (cell_stage is not None and cell_stage in term_stages)
            or port_role in sink_roles
            or str(getattr(port_sig, "state", "")).lower() in dest_states
        )
        source_roles = _reg_ctx.get_ingress_roles_set() | _reg_ctx.get_roles_for_category("source") | {"source_data"}
        source_states = _reg_ctx.get_states_by_property("is_source_identifier", True)
        ing_stages = {1}
        is_source_ctx = (
            (cell is not None and getattr(cell, "node_role", "") in source_roles)
            or (cell_mut and _reg_ctx.is_ingress_role(cell_mut))
            or (cell is not None and len(getattr(cell, "inputs", {})) == 0)
            or (cell_stage is not None and cell_stage in ing_stages)
            or port_role in source_roles
            or str(getattr(port_sig, "state", "")).lower() in source_states
        )
        is_operational = not is_source_ctx and not is_sink_ctx

        # 4b. Predicate / boolean expression arguments, grounded by the DECLARED
        # typestate role (state) of the port — never by the port's identifier string.
        # Fallback: an expression literal (comparison syntax) can only ever ground
        # a string carrier, so a REQUIRED str port of a transform consumes it by
        # type affinity alone — no identifier knowledge, no state declaration
        # needed in the tree.
        _port_state = str(getattr(port_sig, "state", "")).lower()
        _registry = _reg_ctx
        _is_str_port = _registry.is_subtype(str(getattr(port_sig, "type_name", "")).lower(), "str")
        if _port_state in _registry.get_states_by_property("is_condition", True):
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind in ("expr", "quoted_str"):
                    self.used_indices.add(idx)
                    return json.dumps(val)
        elif _is_str_port and is_operational and port_sig.required:
            for idx, (_, kind, val) in enumerate(self.ordered_literals):
                if idx not in self.used_indices and kind == "expr":
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 5. Environmental Asset Grounding (Ingress / Egress)
        # file_asset literals flow only into path-typed ports. Plain str ports may
        # receive assets only when the cell declares NO dedicated path port, so that
        # auxiliary string parameters can never steal file assets from the sink/source.
        # Source vs. destination roles come strictly from declared port_role,
        # mutation_type, and typestate transitions — never token-guessing wordlists.
        def _take_asset(prefer: str) -> Optional[str]:
            fallback: Optional[Tuple[int, str]] = None
            path_candidates: List[Tuple[int, str]] = []
            clause_match_candidate: Optional[Tuple[int, str]] = None
            current_c_idx = getattr(self, "current_cell_clause_idx", None)
            preferred_uris: Set[str] = set()
            excluded_uris: Set[str] = set()
            if prefer == "dest":
                preferred_uris = {str(u).strip() for u in (self.parameters.get("dest_uris") or ())}
                excluded_uris = {str(u).strip() for u in (self.parameters.get("source_uris") or ())} - preferred_uris
            elif prefer == "src":
                preferred_uris = {str(u).strip() for u in (self.parameters.get("source_uris") or ())}
                excluded_uris = {str(u).strip() for u in (self.parameters.get("dest_uris") or ())} - preferred_uris

            for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                if idx in self.used_indices:
                    continue
                if getattr(port_sig, "enum_values", None):
                    if str(val).lower() not in {str(e).lower() for e in port_sig.enum_values}:
                        continue
                is_file_or_quoted = kind in ("file_asset", "quoted_str")
                # Guard: a single-character quoted_str with no path separator
                # or extension is structurally a label/column name, never a
                # file path. Prevent path ports from stealing column literals.
                if kind == "quoted_str" and len(str(val)) <= 1:
                    is_file_or_quoted = False
                matches_path = ExecutionContext._is_path_string(str(val))
                if matches_path:
                    # Modality/format compatibility guard
                    val_str = str(val).strip()
                    if "." in val_str:
                        ext = val_str.rsplit(".", 1)[1].lower()
                        port_fmts = getattr(port_sig, "supported_formats", None)
                        port_fmts_low = {str(f).strip().lower() for f in port_fmts if str(f).strip()} if port_fmts else set()
                        has_wild_fmt = bool(port_fmts_low & {"*", "any"})
                        if port_fmts_low and not has_wild_fmt and ext not in port_fmts_low:
                            continue
                        reg = TypeRegistry.get_instance()
                        port_t = str(getattr(port_sig, "type_name", "")).strip()

                        # Decompose generic path signature: File[Carrier, Format] or File[Modality]
                        inner_carrier = None
                        declared_fmt = None
                        if "[" in port_t and port_t.endswith("]"):
                            ctor, _, inner = port_t[:-1].partition("[")
                            if ctor.strip().lower() in (set(reg.get_types_by_kind("path") or ()) | {"path", "file"}):
                                parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                                if len(parts) >= 2:
                                    inner_carrier = parts[0]
                                    declared_fmt = parts[1]
                                elif len(parts) == 1:
                                    p0 = parts[0]
                                    if not p0 or p0 == "generic" or reg.is_declared_top(p0):
                                        declared_fmt = p0
                                    elif reg.is_registered(f"{p0}_format") or any(p0 in fg for fg in reg.get_format_groups()):
                                        declared_fmt = p0
                                    else:
                                        inner_carrier = p0
                                        declared_fmt = p0

                        # Format verification: if port explicitly declares supported_formats, that is authoritative.
                        if port_fmts_low:
                            if not has_wild_fmt and ext not in port_fmts_low:
                                continue
                        elif declared_fmt and declared_fmt != "generic" and not reg.is_declared_top(declared_fmt):
                            fmt_compat = (
                                ext == declared_fmt
                                or declared_fmt in ext
                                or ext in declared_fmt
                                or reg.is_subtype(ext, declared_fmt)
                                or reg.is_subtype(declared_fmt, ext)
                                or any(reg.is_subtype(ext, fg) and (reg.is_subtype(declared_fmt, fg) or declared_fmt in fg)
                                       for fg in reg.get_format_groups())
                            )
                            if not fmt_compat and inner_carrier:
                                fmt_anc = f"{inner_carrier}_format"
                                if reg.is_registered(fmt_anc) and (reg.is_subtype(ext, fmt_anc) or reg.is_subtype(ext, inner_carrier)):
                                    fmt_compat = True
                            if not fmt_compat:
                                continue
                        else:
                            # Wildcard/unspecified format: ensure cross-modality carrier compatibility
                            target_carrier = inner_carrier or reg.get_abstract_carrier(port_t.lower()) or reg.get_abstract_carrier(str(getattr(port_sig, "domain", "")).lower())
                            _path_and_top = set(reg.get_types_by_kind("path") or ()) | {"path", "file"} | reg.get_top_types_set()
                            if target_carrier and target_carrier not in _path_and_top and not reg.is_declared_top(target_carrier):
                                ext_carrier = None
                                for abs_c in sorted(reg.get_abstract_carriers()):
                                    if abs_c in _path_and_top or reg.is_declared_top(abs_c):
                                        continue
                                    if reg.is_subtype(ext, f"{abs_c}_format") or reg.is_subtype(ext, abs_c):
                                        ext_carrier = abs_c
                                        break
                                if ext_carrier and ext_carrier != target_carrier:
                                    if not (reg.is_subtype(ext_carrier, target_carrier) or reg.is_subtype(target_carrier, ext_carrier)):
                                        continue
                if is_file_or_quoted or (is_path_role and matches_path):
                    val_s = str(val).strip()
                    lit_clause = self._get_clause_idx_for_pos(lit_pos)
                    in_current_clause = (current_c_idx is not None and lit_clause == current_c_idx)

                    if not is_path and not is_path_role and not matches_path:
                        if current_c_idx is not None and lit_clause is not None and lit_clause != current_c_idx:
                            continue

                    if val_s in preferred_uris:
                        self.used_indices.add(idx)
                        return _format_asset_literal(val)

                    if in_current_clause and val_s not in excluded_uris:
                        self.used_indices.add(idx)
                        return _format_asset_literal(val)

                    if in_current_clause and clause_match_candidate is None:
                        clause_match_candidate = (idx, val)

                    if (
                        fallback is None
                        and val_s not in excluded_uris
                        and not (
                            not matches_path
                            and current_c_idx is not None
                            and lit_clause is not None
                            and lit_clause > current_c_idx
                        )
                    ):
                        fallback = (idx, val)
                    if matches_path:
                        path_candidates.append((idx, val))

            if clause_match_candidate is not None:
                self.used_indices.add(clause_match_candidate[0])
                return _format_asset_literal(clause_match_candidate[1])

            if is_path_role and len(path_candidates) == 1:
                idx, val = path_candidates[0]
                self.used_indices.add(idx)
                return _format_asset_literal(val)

            if fallback is not None:
                self.used_indices.add(fallback[0])
                return _format_asset_literal(fallback[1])
            return None

        _proj_states_set = _registry.get_states_by_property("is_projection", True)
        _proj_roles_set = (
            _registry.get_roles_for_category("projection")
            | set(_registry.get_projection_tokens() or ())
        )
        _key_roles_set = _registry.get_roles_for_category("key")
        is_column_port = (
            getattr(port_sig, "binds", "") in _key_roles_set
            or str(getattr(port_sig, "state", "")).lower() in _proj_states_set
            or getattr(port_sig, "port_role", "") in _proj_roles_set
        )
        allow_str_asset = (
            not is_column_port
            and (is_sink_ctx or is_source_ctx or (cell_stage is not None and cell_stage in (1, 3)))
            and is_str
            and not getattr(port_sig, "enum_values", None)
            and not _cell_has_path_port()
            and (port_sig.required or is_sink_ctx)
        )
        if is_path or is_path_role or allow_str_asset:
            if is_sink_ctx:
                taken = _take_asset("dest")
            elif is_source_ctx:
                taken = _take_asset("src")
            else:
                taken = _take_asset("")
            if taken is not None:
                return taken
            # If source cell has a required path port and prompt omitted explicit file literal:
            if port_sig.required and is_source_ctx:
                if port_sig.default_value is not None:
                    val = str(port_sig.default_value)
                    _def_be = get_active_language_backend()
                    if not _def_be.is_null_literal(val):
                        return json.dumps(val) if not (val.startswith(("'", '"')) or _def_be.is_bool_literal(val)) else val

        # 6. Operational Parameter Extraction
        _cfg_param_roles = (
            _reg_ctx.get_roles_for_category("configuration")
            | _reg_ctx.get_roles_for_category("parameter")
        )
        _is_non_path_cfg_str = (not is_path and not is_path_role and port_role in _cfg_param_roles)
        if (is_operational or _is_non_path_cfg_str) and is_str:
            enum_vals = {str(e).strip().lower() for e in port_sig.enum_values} if getattr(port_sig, "enum_values", None) else None
            p_state = str(getattr(port_sig, "state", "") or "").lower()
            p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
            reg = TypeRegistry.get_instance()
            proj_tokens = reg.get_projection_tokens()
            reg_types = reg.get_registered_types()
            is_cell_key_port = _port_matches_declared_ref(cell, "key_port", port_sig)
            is_proj_port = (
                p_role in _proj_roles_set
                or p_role in _key_roles_set
                or p_role in reg.get_roles_for_category("feature")
                or p_role in reg.get_roles_for_category("target")
                or str(getattr(port_sig, "binds", "") or "").lower() in _key_roles_set
                or p_state in _proj_states_set
                or is_cell_key_port
            )

            # Check literals for operational parameter candidates (prioritizing current clause)
            cand_indices = []
            if self.current_cell_clause_idx is not None:
                # 1. Unconsumed literals in the current clause
                cand_indices.extend([i for i in range(len(self.ordered_literals))
                                     if i not in self.used_indices and self.literal_clause_map.get(i) == self.current_cell_clause_idx])
                # 2. Reusable identifiers from the same clause (e.g. select and set on the same key/symbol in a transform)
                if is_proj_port:
                    cand_indices.extend([i for i in range(len(self.ordered_literals))
                                         if i in self.used_indices and self.literal_clause_map.get(i) == self.current_cell_clause_idx
                                         and getattr(self.ordered_literals[i], "kind", self.ordered_literals[i][1]) in ("identifier", "quoted_str")])
                # 3. Unconsumed literals from prior clauses (or global)
                cand_indices.extend([i for i in range(len(self.ordered_literals))
                                     if i not in self.used_indices and self.literal_clause_map.get(i) is not None
                                     and self.literal_clause_map.get(i) < self.current_cell_clause_idx])
            else:
                cand_indices = [i for i in range(len(self.ordered_literals)) if i not in self.used_indices]

            for idx in cand_indices:
                lit_pos, kind, val = self.ordered_literals[idx]
                if idx in self.used_indices and not (is_proj_port and self.current_cell_clause_idx is not None and self.literal_clause_map.get(idx) == self.current_cell_clause_idx and kind in ("identifier", "quoted_str")):
                    continue
                if kind not in ("quoted_str", "identifier"):
                    continue
                val_str = str(val).strip()
                val_low = val_str.lower()
                # Target protection: operational parameters must not steal the target sink variable
                # UNLESS this cell/port is a setter morphism that introduces/writes the target symbol or key
                is_setter_port = (
                    p_role in _proj_roles_set
                    or p_role in reg.get_roles_for_category("feature")
                    or p_role in reg.get_roles_for_category("target")
                    or p_role in _key_roles_set
                    or p_role in reg.get_roles_for_category("destination")
                    or p_state in reg.get_states_by_property("is_target_identifier", True)
                    or getattr(port_sig, "param_kind", "") == "target"
                )
                if val_low in self.target_literals and not is_setter_port:
                    continue
                # Exclude file assets from operational parameters
                if ExecutionContext._is_path_string(val_str) or ("." in val_str and not val_str.replace(".", "").replace("-", "").isdigit()):
                    continue
                # Enforce enum constraints if declared
                if enum_vals is not None:
                    if val_low not in enum_vals:
                        continue
                # Exclude Python keywords and registered types from being bound as projection symbols
                if is_proj_port and (keyword.iskeyword(val_low) or val_low in reg_types):
                    continue
                # Variable symbol & target protection: non-projection parameter ports must not steal variable symbols
                roles = self.identifier_roles.get(lit_pos, set())
                is_var_symbol = (
                    bool(roles & proj_tokens)
                    or (is_proj_port and kind == "quoted_str" and not ExecutionContext._is_path_string(val_str))
                    or (is_setter_port and bool(roles & (proj_tokens | reg.get_roles_for_category("destination") | reg.get_egress_tokens())))
                )
                if is_var_symbol and not is_proj_port:
                    continue
                if is_proj_port and not is_var_symbol:
                    continue
                # Optional parameters without explicit enum match require lexical evidence
                if not port_sig.required and enum_vals is None:
                    port_ident_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(port_sig.name)} if port_sig.name else set()
                    _desc = str(getattr(port_sig, "description", "") or getattr(port_sig, "doc", "") or "")
                    evidence = set(port_ident_toks)
                    if _desc:
                        evidence |= {t.lower() for t in CellTokenizer.tokenize_prompt(_desc)}
                    if cell_tokens:
                        evidence |= {t.lower() for t in cell_tokens}
                    clause_toks = {t.lower() for t in CellTokenizer.tokenize_prompt(self.prompt)}
                    if not (evidence & clause_toks) and not (port_ident_toks and any(tok in self.prompt.lower() for tok in port_ident_toks)):
                        continue
                self.used_indices.add(idx)
                return json.dumps(val)

        # 6b. Referential identifier grounding (role-conditioned).
        if is_operational and port_sig.required and port_sig.default_value is None:
            _tn = t_name
            _is_collection = (
                getattr(port_sig, "structure_kind", None) == "collection"
                or (getattr(port_sig, "canonical_rank", None) is not None and getattr(port_sig, "canonical_rank", None) > 0)
                or getattr(port_sig, "abstract_type", None) == "collection"
                or str(getattr(port_sig, "state", "")).lower() in _proj_states_set
                or (registry is not None and (registry.is_subtype(_tn, "collection") or registry.is_subtype(_tn, "sequence")))
            )
            _is_strict_str = (registry.has_type_kind(_tn, "string") or registry.is_subtype(_tn, "scalar")) and _tn not in ("any", "*", "top", "") and not _is_collection
            _state_tokens: Set[str] = set()
            _raw_state = str(getattr(port_sig, "state", "") or "")
            if _raw_state.lower() not in ("any", "default", ""):
                _state_tokens = CellTokenizer.tokenize_identifier(_raw_state)
            if _is_strict_str and (_state_tokens or cell_tokens):
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens
                if self.current_cell_clause_idx is not None and self.literal_clause_map:
                    cand_indices = [
                        i for i in range(len(self.ordered_literals))
                        if self.literal_clause_map.get(i) is None
                        or self.literal_clause_map.get(i) <= self.current_cell_clause_idx
                    ]
                    cand_indices.sort(key=lambda i: 0 if self.literal_clause_map.get(i) == self.current_cell_clause_idx else 1)
                else:
                    cand_indices = list(range(len(self.ordered_literals)))
                _p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
                _is_setter = is_setter_port if "is_setter_port" in locals() else (
                    _p_role in _proj_roles_set
                    or _p_role in registry.get_roles_for_category("feature")
                    or _p_role in registry.get_roles_for_category("target")
                    or _p_role in _key_roles_set
                    or _p_role in registry.get_roles_for_category("destination")
                    or _raw_state in registry.get_states_by_property("is_target_identifier", True)
                    or getattr(port_sig, "param_kind", "") == "target"
                )
                for idx in cand_indices:
                    pos, kind, val = self.ordered_literals[idx]
                    if idx in self.used_indices or kind != "identifier":
                        continue
                    val_str = str(val).strip()
                    val_low = val_str.lower()
                    if (val_low in self.target_literals and not _is_setter) or keyword.iskeyword(val_low) or val_low in registry.get_registered_types():
                        continue
                    role_tokens = self.identifier_roles.get(pos, set())
                    if not role_tokens or not (role_tokens & _identity_scope):
                        continue
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 6c. Multi-identifier collection / list projection (e.g. list for selection or projection)
        _t_name_concrete = bool(t_name) and not registry.is_type_variable(t_name) and t_name in registry.get_registered_types()
        _sig_carrier = getattr(getattr(port_sig, "signature", None), "carrier", None) or getattr(port_sig, "abstract_type", "") or t_name
        _elem_type = str(getattr(port_sig, "element_type", "") or "").lower()
        _struct_kind = str(getattr(port_sig, "structure_kind", "") or "").lower()
        _sig_struct_kind = str(getattr(getattr(port_sig, "signature", None), "structure_kind", "") or "").lower()
        _tensor_sks = (registry.get_types_by_kind("array") | registry.get_types_by_kind("tensor")) if registry is not None else ()
        _is_numeric_tensor = (
            (registry is not None and registry.is_subtype(_elem_type, "number"))
            or _struct_kind in _tensor_sks
            or _sig_struct_kind in _tensor_sks
            or (registry is not None and (registry.is_subtype(t_name, "tensor") or registry.is_subtype(t_name, "matrix") or registry.is_subtype(t_name, "number")))
            or (registry is not None and (registry.is_subtype(str(_sig_carrier).lower(), "tensor") or registry.is_subtype(str(_sig_carrier).lower(), "matrix")))
        )
        is_collection = (not _is_numeric_tensor) and (
            getattr(port_sig, "structure_kind", None) == "collection"
            or (
                getattr(port_sig, "canonical_rank", None) is not None
                and getattr(port_sig, "canonical_rank", None) > 0
                and not (registry is not None and (registry.is_subtype(t_name, "tensor") or registry.is_subtype(t_name, "matrix") or registry.is_subtype(t_name, "number")))
            )
            or (registry is not None and (registry.is_subtype(t_name, "collection") or registry.is_subtype(t_name, "sequence")) and not registry.is_subtype(t_name, "tensor"))
            or (
                not _t_name_concrete
                and (
                    getattr(port_sig, "abstract_type", None) == "collection"
                    or str(getattr(port_sig, "state", "")).lower() in _proj_states_set
                )
            )
        )
        if is_operational and is_collection:
            _raw_state = str(getattr(port_sig, "state", "") or "")
            _def_raw = str(port_sig.default_value).strip("'\"") if port_sig.default_value is not None else None
            _is_empty_req_key = bool(getattr(port_sig, "required", False) and _def_raw in ("", "[]", "()") and _is_key_or_complement_port(cell, port_sig, registry))
            if port_sig.default_value is not None and not _is_empty_req_key:
                pass  # Fall through to Step 7 (port default value)
            else:
                _state_tokens = CellTokenizer.tokenize_identifier(_raw_state) if _raw_state.lower() not in ("any", "default", "") else set()
                _proj_tokens = TypeRegistry.get_instance().get_projection_tokens()
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens | _proj_tokens
                egress_tokens = TypeRegistry.get_instance().get_egress_tokens() | TypeRegistry.get_instance().get_roles_for_category("destination")
                registered_types = TypeRegistry.get_instance().get_registered_types()

                matched_indices = []
                matched_vals = []
                cell_clause = getattr(self, "current_cell_clause_idx", None)
                lit_clause_map = getattr(self, "literal_clause_map", {})
                target_lits = getattr(self, "target_literals", set())
                excluded_keys_low = {str(ek).strip("'\"").strip().lower() for ek in (excluded_keys or ()) if str(ek).strip()}

                all_available = [
                    idx for idx, (_, kind, val) in enumerate(self.ordered_literals)
                    if idx not in self.used_indices and kind in ("quoted_str", "identifier")
                    and str(val).lower() not in target_lits
                    and str(val).strip().lower() not in excluded_keys_low
                ]
                if cell_clause is not None and lit_clause_map:
                    clause_cands = [idx for idx in all_available if lit_clause_map.get(idx) == cell_clause]
                    cands_to_check = clause_cands if (clause_cands or _is_empty_req_key) else all_available
                else:
                    cands_to_check = [] if _is_empty_req_key else all_available

                for idx in cands_to_check:
                    pos, kind, val = self.ordered_literals[idx]
                    val_str = str(val).strip()
                    val_low = val_str.lower()
                    if val_low in target_lits or keyword.iskeyword(val_low) or val_low in registered_types:
                        continue
                    role_tokens = self.identifier_roles.get(pos, set())
                    if role_tokens and (role_tokens & egress_tokens):
                        continue

                    is_candidate = False
                    if role_tokens and (role_tokens & _identity_scope):
                        is_candidate = True
                    elif (
                        kind == "quoted_str"
                        and not ExecutionContext._is_path_string(val_str)
                        and _is_key_or_complement_port(cell, port_sig, registry)
                    ):
                        is_candidate = True

                    if is_candidate and val not in matched_vals:
                        matched_indices.append(idx)
                        matched_vals.append(val)

                if matched_vals:
                    for idx in matched_indices:
                        self.used_indices.add(idx)
                        self.consumed_tokens.add(str(self.ordered_literals[idx][2]).lower())
                    return json.dumps(matched_vals)

        # 7. Port default value declared in tree schema
        if port_sig.default_value is not None:
            def_str = str(port_sig.default_value).strip()
            key_binds = _key_roles_set
            proj_roles = _proj_roles_set | _key_roles_set
            is_cell_key_port = _port_matches_declared_ref(cell, "key_port", port_sig)
            is_key_req = getattr(port_sig, "required", False) and (
                getattr(port_sig, "binds", "") in key_binds
                or str(getattr(port_sig, "port_role", "") or "").lower() in proj_roles
                or is_cell_key_port
                or bool(getattr(port_sig, "complement_of_port", None))
                or bool(getattr(port_sig, "key_from_upstream_role", None))
            )
            _def_be = get_active_language_backend()
            _def_stripped = def_str.strip("'\"")
            if is_key_req and (not _def_stripped or _def_be.is_empty_collection_literal(_def_stripped)):
                pass
            else:
                if _def_be.is_null_or_bool_literal(def_str, case_insensitive=False):
                    return def_str
                if not is_str:
                    return def_str
                # For string ports, check if def_str is an unquoted module constant (module-scoped enum attribute)
                parts = def_str.split(".")
                if len(parts) > 1 and parts[0] in sys.modules:
                    return def_str
                return def_str if (def_str.startswith('"') or def_str.startswith("'")) else json.dumps(def_str)

        # 8. Pure Vector Semantic Slot Projection for unquoted string/identifier arguments.
        if is_operational and is_str and self.prompt:
            _raw_state = str(getattr(port_sig, "state", "") or "")
            role_label = _raw_state if _raw_state.lower() not in ("any", "default", "") else ""
            projected = self._project_semantic_slot(port_sig.name, role_label=role_label)
            if projected:
                return json.dumps(projected)

        return None


# =====================================================================
# 5. Type-Monadic Unification Gate
# =====================================================================

@dataclass
class VerificationContract:
    """
    Structured task verification contract generated during AST synthesis.
    Encapsulates Phase-1 postconditions and terminal node intent checks for GEVR sandbox execution.
    """
    cell_checks: List[Dict[str, Any]] = field(default_factory=list)
    terminal_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_checks": self.cell_checks,
            "terminal_checks": self.terminal_checks
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerificationContract":
        return cls(
            cell_checks=d.get("cell_checks", []),
            terminal_checks=d.get("terminal_checks", [])
        )

@functools.lru_cache(maxsize=512)
def _get_module_symbols(mod_name: str) -> Tuple[str, ...]:
    """Caches exported symbols of submodules for fast dependency resolution."""
    try:
        import sys, importlib
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        return tuple(w for w in dir(mod) if not w.startswith("_"))
    except Exception:
        return ()


class UnificationGate:
    """
    Formal Unification Gate verifying dataflow composition and emitting code.
    Contains ZERO hardcoded domain libraries or prompt-sniffing regexes.
    """
    def __init__(self, orchestrator: Optional[Any] = None):
        if orchestrator is None:
            try:
                from .lattice import LatticeOrchestrator
            except (ImportError, ValueError):
                try:
                    from lattice import LatticeOrchestrator
                except ImportError:
                    LatticeOrchestrator = None
            if LatticeOrchestrator is not None:
                orchestrator = LatticeOrchestrator.get_active_instance()
        self.orchestrator = orchestrator
        self.context = ExecutionContext()
        self.last_egress_paths: List[str] = []
        self.last_pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        self.last_pipeline_typed_bindings: List[Dict[str, TypedBindingRecord]] = []
        self.last_typed_proof: Optional[TypedPipelineProof] = None
        self.last_verification_contract: Optional[VerificationContract] = None
        self.last_runtime_aliases: Dict[str, str] = {}
        self._generic_port_cache: Optional[Tuple[Tuple[int, int], Set[str]]] = None
        try:
            from .unification_stages import (
                PortWiringStage,
                ScopeResolutionStage,
            )
        except (ImportError, ValueError):
            from unification_stages import (
                PortWiringStage,
                ScopeResolutionStage,
            )
        self.wiring_stage = PortWiringStage(self)
        self.scope_stage = ScopeResolutionStage(self)

    def _generic_port_tokens(self) -> Set[str]:
        """
        Measured generic-vocabulary stop set: tokens whose document frequency
        across the loaded lattice exceeds one standard deviation above the mean.
        Such tokens appear on too many cells (and ports) to discriminate between
        candidate wire bindings. Derived purely from the loaded corpus; no
        literal token list lives in the engine.
        """
        orch = getattr(self, "orchestrator", None)
        if orch is not None and getattr(orch, "_generic_port_cache", None) is not None:
            return orch._generic_port_cache
        lc = getattr(self.orchestrator, "loaded_cells", None) or {}
        stamp = (id(lc), len(lc))
        if self._generic_port_cache is not None and self._generic_port_cache[0] == stamp:
            return self._generic_port_cache[1]
        dfs: List[int] = []
        toks: Dict[str, int] = {}
        for c in lc.values():
            for t in getattr(c, "token_set", set()):
                if isinstance(t, str) and len(t) >= 2:
                    toks[t.lower()] = toks.get(t.lower(), 0) + 1
        dfs = list(toks.values())
        if not dfs:
            out: Set[str] = set()
            self._generic_port_cache = (stamp, out)
            if orch is not None:
                orch._generic_port_cache = out
            return out
        mean = sum(dfs) / len(dfs)
        var = sum((v - mean) ** 2 for v in dfs) / len(dfs)
        cut = mean + (var ** 0.5)
        out = {t for t, v in toks.items() if v > cut}
        self._generic_port_cache = (stamp, out)
        if orch is not None:
            orch._generic_port_cache = out
        return out

    def get_runtime_aliases(self) -> Dict[str, str]:
        """
        Returns the prompt-declared egress aliases produced by the most recent
        synthesis (e.g. {"Z": "var_3"} for "store the results into Z"). The
        sandbox applies these in the runtime namespace so the declared sink
        identifier resolves without fabricating source-level assignments.
        """
        return dict(self.last_runtime_aliases)

    @staticmethod
    def _refuse_non_callable_calls(code: str) -> None:
        """
        Host-language sanity gate at emission time (domain-agnostic): a call
        whose callee is a non-callable CONSTANT (string/number/bool literal)
        can never execute. That pattern is the fingerprint of an unfilled
        code-template placeholder reaching emission (e.g. ``var = 'X'(var_1)``)
        and must fail the synthesis loudly here rather than at runtime.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return  # syntax problems are reported by the linters downstream
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Constant):
                raise UnresolvedPlaceholderError(
                    f"Code synthesis refused: emitted code calls the non-callable "
                    f"constant {node.func.value!r} as a function (an unfilled "
                    f"template placeholder reached emission)."
                )

    @staticmethod
    def _optic_getter_ports(cell: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Returns (container_in_port, key_in_port, focus_out_port) for a getter cell."""
        c_port = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs") or getattr(cell, "primary_input", None)
        k_port = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs")
        f_ref = getattr(cell, "focus_port", None)
        outputs = getattr(cell, "outputs", None) or {}
        f_out = None
        if f_ref and isinstance(outputs, dict):
            f_out = outputs.get(f_ref) or next((p for p in outputs.values() if getattr(p, "name", None) == f_ref), None)
        if f_out is None:
            f_out = getattr(cell, "primary_output", None) or (next(iter(outputs.values())) if outputs else None)
        return c_port, k_port, f_out

    @staticmethod
    def _optic_setter_ports(cell: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        """Returns (container_in_port, key_in_port, focus_in_port, container_out_port) for a setter cell."""
        c_port = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs") or getattr(cell, "primary_input", None)
        k_port = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs")
        f_port = _resolve_cell_port(cell, getattr(cell, "focus_port", None), prefer="inputs")
        outputs = getattr(cell, "outputs", None) or {}
        c_out = getattr(cell, "primary_output", None) or (next(iter(outputs.values())) if outputs else None)
        return c_port, k_port, f_port, c_out

    def _get_loaded_cells_pool(self) -> List[Any]:
        orch = getattr(self, "orchestrator", None)
        if orch is None:
            try:
                from .lattice import LatticeOrchestrator
            except (ImportError, ValueError):
                from lattice import LatticeOrchestrator
            orch = LatticeOrchestrator.get_active_instance()
        if orch is None:
            return []
        loaded = getattr(orch, "loaded_cells", None)
        if isinstance(loaded, dict):
            return list(loaded.values())
        return list(getattr(orch, "cells", ()) or ())

    def _find_optic_setter_for_getter(self, getter_cell: Any, focus_sig: Optional[Any] = None) -> Optional[Any]:
        """Finds a domain-matching setter cell (optic_role=='setter') whose (container, key, focus)
        ports compose with `getter_cell`'s extracted fiber."""
        g_cont, g_key, g_focus = self._optic_getter_ports(getter_cell)
        if g_cont is None or g_focus is None:
            return None
        g_cont_sig = getattr(g_cont, "signature", g_cont)
        f_check = getattr(focus_sig, "signature", focus_sig) if focus_sig is not None else getattr(g_focus, "signature", g_focus)
        g_dom = str(getattr(getter_cell, "domain_name", "") or "").lower()
        scored: List[Tuple[Tuple[int, int, int], Any]] = []
        for cand in self._get_loaded_cells_pool():
            if str(getattr(cand, "optic_role", "") or "").strip().lower() != "setter":
                continue
            s_cont, s_key, s_focus, s_out = self._optic_setter_ports(cand)
            if s_cont is None or s_key is None or s_focus is None or s_out is None:
                continue
            s_cont_sig = getattr(s_cont, "signature", s_cont)
            if unify(g_cont_sig, s_cont_sig) is None and unify(s_cont_sig, g_cont_sig) is None:
                continue
            s_out_sig = getattr(s_out, "signature", s_out)
            if unify(s_out_sig, s_cont_sig) is None and unify(s_out_sig, g_cont_sig) is None:
                continue
            if g_key is not None:
                gk_sig = getattr(g_key, "signature", g_key)
                sk_sig = getattr(s_key, "signature", s_key)
                if unify(gk_sig, sk_sig) is None and unify(sk_sig, gk_sig) is None:
                    continue
            sf_sig = getattr(s_focus, "signature", s_focus)
            if f_check is not None and not getattr(f_check, "is_top", lambda: False)():
                if unify(f_check, sf_sig) is None:
                    continue
            same_dom = 1 if (g_dom and str(getattr(cand, "domain_name", "") or "").lower() == g_dom) else 0
            exact_cont = 1 if str(getattr(g_cont_sig, "type_name", "")).lower() == str(getattr(s_cont_sig, "type_name", "")).lower() else 0
            vis_tie = (1 if getattr(cand, "is_public", True) else 0, -str(getattr(cand, "cell_id", "")).count("."))
            scored.append(((same_dom, exact_cont, vis_tie), cand))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _find_optic_lifted_getter_for_port(
        self,
        cont_var: str,
        cont_sig: Any,
        consumer_port: Any,
        consumer_cell: Any,
        ctx: ExecutionContext,
        registry: TypeRegistry,
    ) -> Optional[Tuple[Any, Optional[Any], str, Any, Optional[Tuple[Any, Any, Any, Dict[str, Any]]]]]:
        """Finds `(getter_cell, setter_cell_or_none, key_expr, focus_out_sig, optional_bridge)` that projects
        `cont_var` (directly or via a 1-hop unary bridge morphism) into `consumer_port` with a
        prompt/context-grounded key (zero fabricated keys)."""
        c_sig_inner = getattr(cont_sig, "signature", cont_sig)
        c_tn = str(getattr(c_sig_inner, "type_name", "") or "").strip().lower()
        if not c_tn or registry.is_declared_top(c_tn) or getattr(c_sig_inner, "is_top", lambda: False)():
            return None
        p_sig_inner = getattr(consumer_port, "signature", consumer_port)
        p_tn_lower = str(getattr(p_sig_inner, "type_name", "") or "").strip().lower()
        c_dom = str(getattr(consumer_cell, "domain_name", "") or "").lower()
        c_clause = getattr(consumer_cell, "matched_clause_idx", None)
        saved_clause = getattr(ctx, "current_cell_clause_idx", None)

        # Target / complement-partner ports are permitted to read synthesized target keys
        cp_name = getattr(consumer_port, "name", None)
        cp_role = str(getattr(consumer_port, "port_role", "") or getattr(consumer_port, "role", "") or "").strip().lower()
        cp_state = str(getattr(p_sig_inner, "state", "") or "").strip().lower()
        target_roles_sem = registry.get_roles_for_category("target")
        target_states_sem = registry.get_states_by_property("is_target_identifier", True)
        has_comp_partner = bool(
            cp_name
            and any(
                getattr(ip, "complement_of_port", None) == cp_name
                for ip in (getattr(consumer_cell, "inputs", {}) or {}).values()
            )
        )
        allow_target_key = bool(
            cp_role in target_roles_sem
            or cp_state in target_states_sem
            or has_comp_partner
        )
        c_kq_map = getattr(ctx, "container_key_qualifiers", None) or {}
        qual_matched_keys: Set[str] = {
            str(k_l).strip().lower()
            for (kv, k_l), kq_s in c_kq_map.items()
            if kv == cont_var
            and kq_s
            and (
                _port_declares_acceptance(consumer_port, set(kq_s))
                or _port_declares_acceptance(p_sig_inner, set(kq_s))
            )
        }
        target_lits_low = {str(tl).strip().lower() for tl in (getattr(ctx, "target_literals", set()) or set()) if str(tl).strip()}
        if getattr(ctx, "target_key", None):
            target_lits_low.add(str(ctx.target_key).strip().lower())
        written_lits_low = {str(wk).strip().lower() for wk in (getattr(ctx, "written_keys", set()) or set()) if str(wk).strip()}
        pending_setter_low = {str(pk).strip().lower() for pk in (getattr(ctx, "_pending_setter_keys", set()) or set()) if str(pk).strip()} - written_lits_low
        all_target_low = target_lits_low | written_lits_low

        pool_cells = self._get_loaded_cells_pool()
        candidates: List[Tuple[Tuple[int, int, int, int, int], Any, Optional[Any], str, Any, Optional[Tuple[Any, Any, Any, Dict[str, Any]]]]] = []
        for getter in pool_cells:
            if str(getattr(getter, "optic_role", "") or "").strip().lower() != "getter":
                continue
            g_cont, g_key, g_focus = self._optic_getter_ports(getter)
            if g_cont is None or g_key is None or g_focus is None:
                continue
            if not _shape_compatible(cont_sig, g_cont, producer_cell=ctx.var_sources.get(cont_var), consumer_cell=getter):
                continue
            if unify(c_sig_inner, getattr(g_cont, "signature", g_cont)) is None:
                continue
            gf_sig = getattr(g_focus, "signature", g_focus)
            focus_bridge: Optional[Tuple[Any, Any, Any, Dict[str, Any]]] = None
            is_direct = 0
            if _shape_compatible(g_focus, consumer_port, producer_cell=getter, consumer_cell=consumer_cell) and unify(gf_sig, p_sig_inner) is not None:
                is_direct = 1
            else:
                # 1-hop unary carrier bridge (e.g. Series -> ndarray) before consumer/qualifier bridging
                best_br: Optional[Tuple[Tuple[int, int, int], Any, Any, Any, Dict[str, Any]]] = None
                gf_tn = str(getattr(gf_sig, "type_name", "") or "").lower()
                for b_cand in pool_cells:
                    if str(getattr(b_cand, "node_role", "") or "").strip().lower() != "bridge":
                        continue
                    b_ins = getattr(b_cand, "inputs", {}) or {}
                    b_outs = getattr(b_cand, "outputs", {}) or {}
                    if not b_ins or not b_outs:
                        continue
                    b_in = getattr(b_cand, "primary_input", None) or next(iter(b_ins.values()), None)
                    b_out = getattr(b_cand, "primary_output", None) or next(iter(b_outs.values()), None)
                    if b_in is None or b_out is None:
                        continue
                    if not _shape_compatible(g_focus, b_in, producer_cell=getter, consumer_cell=b_cand):
                        continue
                    if unify(gf_sig, getattr(b_in, "signature", b_in)) is None:
                        continue
                    b_out_concrete = _instantiate_bridge_output_signature(b_cand, b_out, g_focus, b_in_name=getattr(b_in, "name", "x"))
                    if not _shape_compatible(b_out_concrete, consumer_port, producer_cell=b_cand, consumer_cell=consumer_cell):
                        continue
                    if unify(getattr(b_out_concrete, "signature", b_out_concrete), p_sig_inner) is None:
                        continue
                    b_extra: Dict[str, Any] = {}
                    ok_b = True
                    for bp_n, bp_s in b_ins.items():
                        if bp_s is b_in or bp_n == getattr(b_in, "name", None):
                            continue
                        if getattr(bp_s, "required", False):
                            if getattr(bp_s, "default_value", None) is not None:
                                b_extra[bp_n] = str(bp_s.default_value)
                            else:
                                eta = _pointed_default(bp_s, b_cand)
                                if eta is not None:
                                    b_extra[bp_n] = eta
                                else:
                                    ok_b = False
                                    break
                        else:
                            b_extra[bp_n] = None
                    if not ok_b:
                        continue
                    exact_in_b = 1 if gf_tn == str(getattr(getattr(b_in, "signature", b_in), "type_name", "") or "").lower() else 0
                    exact_out_b = 1 if p_tn_lower == str(getattr(getattr(b_out, "signature", b_out), "type_name", "") or "").lower() else 0
                    vis_tie_b = (1 if getattr(b_cand, "is_public", True) else 0, -str(getattr(b_cand, "cell_id", "")).count("."))
                    sc_b = (exact_out_b, exact_in_b, vis_tie_b)
                    if best_br is None or sc_b > best_br[0]:
                        best_br = (sc_b, b_cand, b_in, b_out, b_extra)
                if best_br is None:
                    continue
                _, b_c, b_i, b_o, b_ex = best_br
                focus_bridge = (b_c, b_i, b_o, b_ex)

            # Check setter pair exists for this getter (prefer lenses with setter closure)
            setter = self._find_optic_setter_for_getter(getter, focus_sig=gf_sig)
            # Check if a key can be grounded without consuming indices until commit
            gk_role = str(getattr(g_key, "port_role", "") or getattr(g_key, "binds", "") or "").strip().lower()
            has_role_key = bool(gk_role and (getattr(ctx, "role_bound_keys", None) or {}).get(gk_role))
            cont_origins_low = {str(o).strip().lower(): str(o).strip() for o in (getattr(ctx, "var_origins", {}).get(cont_var, set()) or set())}
            has_target_origin_key = bool(
                allow_target_key
                and cont_origins_low
                and any(tk in cont_origins_low for tk in all_target_low)
            )
            has_qual_origin_key = bool(
                qual_matched_keys
                and cont_origins_low
                and any(qk in cont_origins_low for qk in qual_matched_keys)
            )
            can_lit_key = False
            # Peek without mutating ctx.used_indices by checking ordered_literals / params / defaults
            g_name = getattr(g_key, "name", "")
            if g_name in getattr(ctx, "parameters", {}):
                can_lit_key = True
            elif has_target_origin_key or has_qual_origin_key:
                can_lit_key = True
            elif any(
                i not in getattr(ctx, "used_indices", set())
                and kind in ("quoted_str", "identifier")
                and str(val).strip().lower() not in pending_setter_low
                and (allow_target_key or str(val).strip().lower() not in target_lits_low)
                and not ExecutionContext._is_path_string(str(val))
                for i, (_, kind, val) in enumerate(getattr(ctx, "ordered_literals", []))
            ):
                can_lit_key = True
            elif has_role_key:
                can_lit_key = True
            elif getattr(g_key, "default_value", None) is not None:
                _gk_def = str(g_key.default_value).strip("'\"")
                _gk_be = get_active_language_backend()
                if _gk_def and not _gk_be.is_null_literal(_gk_def) and not _gk_be.is_empty_collection_literal(_gk_def):
                    can_lit_key = True
            if not can_lit_key:
                continue
            g_dom = str(getattr(getter, "domain_name", "") or "").lower()
            same_dom = 1 if (c_dom and g_dom == c_dom) else 0
            exact_focus = 1 if str(getattr(gf_sig, "type_name", "")).lower() == p_tn_lower else 0
            has_setter = 1 if setter is not None else 0
            vis_tie = (1 if getattr(getter, "is_public", True) else 0, -str(getattr(getter, "cell_id", "")).count("."))
            candidates.append(((is_direct, has_setter, same_dom, exact_focus, vis_tie), getter, setter, "", g_focus, focus_bridge))

        if not candidates:
            return None
        candidates.sort(key=lambda it: it[0], reverse=True)
        for _, getter, setter, _, g_focus, focus_bridge in candidates:
            g_clone = getter.clone() if hasattr(getter, "clone") else copy.copy(getter)
            g_clone.matched_clause_idx = c_clause
            _, g_key, _ = self._optic_getter_ports(g_clone)
            key_expr = None
            cont_origins_low = {str(o).strip().lower(): str(o).strip() for o in (getattr(ctx, "var_origins", {}).get(cont_var, set()) or set())}
            if qual_matched_keys and cont_origins_low:
                for qk in sorted(qual_matched_keys):
                    if qk in cont_origins_low:
                        key_expr = json.dumps(cont_origins_low[qk])
                        break
            if not key_expr and allow_target_key and cont_origins_low:
                t_key = getattr(ctx, "target_key", None)
                if t_key and str(t_key).strip().lower() in cont_origins_low:
                    key_expr = json.dumps(cont_origins_low[str(t_key).strip().lower()])
                else:
                    for tl in sorted(all_target_low):
                        if tl in cont_origins_low:
                            key_expr = json.dumps(cont_origins_low[tl])
                            break
            if not key_expr:
                saved_step_idx = getattr(ctx, "current_cell_step_idx", None)
                ctx.current_cell_clause_idx = c_clause
                ctx.current_cell_step_idx = None
                try:
                    key_expr = ctx.resolve_literal_for_port(
                        g_key,
                        cell_stage=getattr(g_clone, "stage", None),
                        cell_inputs=getattr(g_clone, "inputs", None),
                        cell_tokens=getattr(g_clone, "token_set", set()),
                        cell=g_clone,
                    )
                finally:
                    ctx.current_cell_clause_idx = saved_clause
                    ctx.current_cell_step_idx = saved_step_idx
            if not key_expr:
                gk_role = str(getattr(g_key, "port_role", "") or getattr(g_key, "binds", "") or "").strip().lower()
                rb_list = list((getattr(ctx, "role_bound_keys", None) or {}).get(gk_role, []) or [])
                if rb_list:
                    key_expr = json.dumps(rb_list[-1])
            _k_be = get_active_language_backend()
            _k_stripped = str(key_expr).strip("'\"") if key_expr else ""
            if _k_stripped and not _k_be.is_null_literal(_k_stripped) and not _k_be.is_empty_collection_literal(_k_stripped):
                matched_lay = _lookup_consistent_key_layout(ctx, cont_var, _extract_keys_from_val(key_expr))
                if matched_lay is not None:
                    g_focus_eff = _enrich_port_signature_layout(g_focus, matched_lay, registry)
                    if focus_bridge is None:
                        if not _shape_compatible(g_focus_eff, consumer_port, producer_cell=g_clone, consumer_cell=consumer_cell):
                            continue
                    else:
                        b_c, b_i, b_o, _ = focus_bridge
                        b_out_eff = _instantiate_bridge_output_signature(b_c, b_o, g_focus_eff, b_in_name=getattr(b_i, "name", "x"))
                        if not _shape_compatible(b_out_eff, consumer_port, producer_cell=b_c, consumer_cell=consumer_cell):
                            continue
                return (g_clone, setter, key_expr, g_focus, focus_bridge)
        return None

    def _emit_optic_setter_closure(
        self,
        frame: Dict[str, Any],
        ctx: ExecutionContext,
        registry: TypeRegistry,
        accumulated_sigma: Substitution,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        var_counter_ref: List[int],
        cell_out_vars: Dict[Any, str],
        cell_port_vars: Dict[Tuple[Any, str], str],
        clause_idx: Optional[int] = None,
    ) -> Optional[str]:
        """Materializes an optic setter call `setter(container, key, focus) -> new_container`
        and marks the frame closed."""
        setter_template = frame.get("setter_cell")
        if setter_template is None or frame.get("closed"):
            return None
        cont_var = frame.get("container_var")
        focus_var = frame.get("focus_var")
        key_expr = frame.get("key_expr")
        if not cont_var or not focus_var or not key_expr:
            return None
        if cont_var not in ctx.variables or focus_var not in ctx.variables:
            return None

        s_cell = setter_template.clone() if hasattr(setter_template, "clone") else copy.copy(setter_template)
        if clause_idx is not None:
            s_cell.matched_clause_idx = clause_idx
        s_cont, s_key, s_focus, s_out = self._optic_setter_ports(s_cell)
        if s_cont is None or s_key is None or s_focus is None or s_out is None:
            return None

        s_bindings: Dict[str, Any] = _TracedBindings(ctx, s_cell)
        s_bindings[s_cont.name] = cont_var
        s_bindings[s_key.name] = key_expr
        s_bindings[s_focus.name] = focus_var

        # Ground any remaining required config ports on setter (or fail closing safely if impossible)
        for sp_name, sp_sig in (getattr(s_cell, "inputs", {}) or {}).items():
            if sp_name in s_bindings:
                continue
            if getattr(sp_sig, "required", False):
                if getattr(sp_sig, "default_value", None) is not None:
                    s_bindings[sp_name] = str(sp_sig.default_value)
                else:
                    eta = _pointed_default(sp_sig, s_cell)
                    if eta is not None:
                        s_bindings[sp_name] = eta
                    else:
                        return None
            else:
                s_bindings[sp_name] = None

        if getattr(s_cell, "mutation_type", "pure") == "in_place":
            new_cont_var = cont_var
        else:
            var_counter_ref[0] += 1
            ctx.var_counter = var_counter_ref[0]
            new_cont_var = _semantic_var_name(
                s_cell,
                var_counter_ref[0],
                port_name=getattr(s_out, "name", "output_var"),
                out_sig=s_out,
            )

        s_bindings["output_var"] = new_cont_var
        if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
            ctx.var_port_map = {}
        if getattr(s_out, "name", None):
            s_bindings[s_out.name] = new_cont_var
            cell_port_vars[(s_cell.cell_id, s_out.name)] = new_cont_var
            cell_port_vars[(id(s_cell), s_out.name)] = new_cont_var
            ctx.var_port_map[(id(s_cell), s_out.name)] = new_cont_var
        cell_out_vars[s_cell.cell_id] = new_cont_var
        cell_out_vars[id(s_cell)] = new_cont_var
        cell_port_vars[(s_cell.cell_id, "output_var")] = new_cont_var
        cell_port_vars[(id(s_cell), "output_var")] = new_cont_var
        ctx.var_port_map[(id(s_cell), "output_var")] = new_cont_var

        concrete_out = substitute_generics(s_out, accumulated_sigma)
        ctx.declare_variable(new_cont_var, concrete_out, new_cont_var, cell=s_cell)
        _update_variable_provenance(ctx, new_cont_var, s_cell, s_bindings, out_port_name=getattr(s_out, "name", None))
        _record_setter_key_layout(ctx, cont_var, new_cont_var, focus_var, _extract_keys_from_val(key_expr))
        if new_cont_var != cont_var and getattr(s_cell, "mutation_type", "pure") != "in_place":
            ctx.superseded_vars.add(cont_var)
        ctx.active_carrier_var = new_cont_var
        try:
            self.scope_stage.finalize_cell_scope(s_cell, s_bindings, ctx, accumulated_sigma)
        except Exception:
            pass
        pipeline_bindings.append((s_cell, s_bindings))
        frame["closed"] = True
        frame["closed_var"] = new_cont_var
        ctx.trace_binding("optic_setter_closed", cell=s_cell.cell_id, container=cont_var, key=key_expr, focus=focus_var, out_var=new_cont_var)
        return new_cont_var

    def _flush_optic_frames_for_consumer(
        self,
        consumer_cell: Any,
        ctx: ExecutionContext,
        registry: TypeRegistry,
        accumulated_sigma: Substitution,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        var_counter_ref: List[int],
        cell_out_vars: Dict[Any, str],
        cell_port_vars: Dict[Tuple[Any, str], str],
    ) -> Optional[str]:
        """Flushes dirty open optic frames when an incoming cell consumes the container carrier
        (rather than the extracted focus or acting itself as a setter)."""
        if str(getattr(consumer_cell, "optic_role", "") or "").strip().lower() == "setter":
            return None
        frames = getattr(ctx, "optic_frames", None) or []
        if not frames:
            return None
        cfg_roles = registry.get_config_roles_set()
        consumer_inputs = [
            p for p in (getattr(consumer_cell, "inputs", {}) or {}).values()
            if not (
                (str(getattr(p, "port_role", "") or getattr(p, "param_kind", "") or "").lower() in cfg_roles)
                and (getattr(p, "canonical_rank", None) == 0 or str(getattr(p, "structure_kind", "") or "").lower() == "scalar")
            )
        ]
        if not consumer_inputs:
            return None

        c_optic_r = str(getattr(consumer_cell, "optic_role", "") or "").strip().lower()
        last_closed_var: Optional[str] = None
        for fr in frames:
            if fr.get("closed") or not fr.get("dirty"):
                continue
            if not fr.get("lifted") and c_optic_r == "setter":
                continue
            setter = fr.get("setter_cell")
            if setter is None:
                continue
            _, _, _, s_out = self._optic_setter_ports(setter)
            if s_out is None:
                continue
            s_out_sig = getattr(s_out, "signature", s_out)
            focus_var = fr.get("focus_var")
            focus_entry = ctx.variables.get(focus_var) if focus_var else None
            focus_sig = focus_entry[0] if focus_entry else None
            focus_sig_inner = getattr(focus_sig, "signature", focus_sig) if focus_sig else None

            is_poly_sink = _is_sink_cell(consumer_cell) or getattr(consumer_cell, "stage", None) == 3
            accepts_container = any(
                (not getattr(getattr(ip, "signature", ip), "is_top", lambda: False)())
                and not registry.is_declared_top(str(getattr(getattr(ip, "signature", ip), "type_name", "")).lower())
                and _shape_compatible(s_out, ip, producer_cell=setter, consumer_cell=consumer_cell)
                and unify(s_out_sig, getattr(ip, "signature", ip), accumulated_sigma) is not None
                for ip in consumer_inputs
            )
            if not accepts_container and is_poly_sink:
                accepts_container = any(
                    getattr(getattr(ip, "signature", ip), "is_top", lambda: False)()
                    or registry.is_declared_top(str(getattr(getattr(ip, "signature", ip), "type_name", "")).lower())
                    for ip in consumer_inputs
                )
            if not accepts_container:
                continue
            accepts_focus = bool(
                focus_sig_inner is not None
                and any(
                    (not getattr(getattr(ip, "signature", ip), "is_top", lambda: False)())
                    and not registry.is_declared_top(str(getattr(getattr(ip, "signature", ip), "type_name", "")).lower())
                    and _shape_compatible(focus_sig, ip, producer_cell=ctx.var_sources.get(focus_var), consumer_cell=consumer_cell)
                    and unify(focus_sig_inner, getattr(ip, "signature", ip), accumulated_sigma) is not None
                    for ip in consumer_inputs
                )
            )
            if accepts_focus:
                continue

            closed_v = self._emit_optic_setter_closure(
                fr,
                ctx=ctx,
                registry=registry,
                accumulated_sigma=accumulated_sigma,
                pipeline_bindings=pipeline_bindings,
                var_counter_ref=var_counter_ref,
                cell_out_vars=cell_out_vars,
                cell_port_vars=cell_port_vars,
                clause_idx=getattr(consumer_cell, "matched_clause_idx", None),
            )
            if closed_v:
                last_closed_var = closed_v
        return last_closed_var

    def _try_lift_lens_for_cell(
        self,
        consumer_cell: Any,
        cell_bindings: Dict[str, Any],
        ctx: ExecutionContext,
        registry: TypeRegistry,
        accumulated_sigma: Substitution,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        var_counter_ref: List[int],
        cell_out_vars: Dict[Any, str],
        cell_port_vars: Dict[Tuple[Any, str], str],
        target_port_only: Optional[str] = None,
    ) -> Optional[Tuple[str, Substitution]]:
        """Attempts Profunctor Optic lifting on the active container when `consumer_cell` requires
        a focus carrier that is not directly in scope."""
        req_ports = [
            p for p_n, p in (getattr(consumer_cell, "inputs", {}) or {}).items()
            if getattr(p, "required", False)
            and p_n not in cell_bindings
            and (target_port_only is None or p_n == target_port_only)
        ]
        if not req_ports:
            return None
        # Candidate containers: active_carrier_var first, then recent non-superseded variables
        cont_cands: List[str] = []
        act_v = getattr(ctx, "active_carrier_var", None)
        if act_v and act_v in ctx.variables and act_v not in getattr(ctx, "superseded_vars", set()):
            cont_cands.append(act_v)
        for v_n in reversed(list(ctx.variables.keys())):
            if v_n not in cont_cands and v_n not in getattr(ctx, "superseded_vars", set()):
                cont_cands.append(v_n)

        cfg_roles = registry.get_config_roles_set()
        data_roles = (
            registry.get_roles_for_category("data")
            | registry.get_roles_for_category("projection")
            | set(registry.get_projection_tokens() or ())
            | registry.get_roles_for_category("feature")
            | registry.get_roles_for_category("target")
            | registry.get_roles_for_category("key")
            | registry.get_roles_for_category("strict")
            | registry.get_roles_for_category("estimator")
            | registry.get_declared_role_carriers()
            | registry.get_data_bearing_roles()
        )
        prim_in_ref = getattr(consumer_cell, "primary_input", None)
        prim_in_name = getattr(consumer_cell, "primary_in", None)

        def _emit_bridge_step(
            b_proto: Any,
            b_in_p: Any,
            b_out_p: Any,
            b_extra: Dict[str, Any],
            in_var: str,
            in_sig: Any,
        ) -> Tuple[str, Any]:
            b_clone = b_proto.clone() if hasattr(b_proto, "clone") else copy.copy(b_proto)
            b_clone.matched_clause_idx = getattr(consumer_cell, "matched_clause_idx", None)
            b_in_name = getattr(b_in_p, "name", None) or next(iter((getattr(b_clone, "inputs", {}) or {}).keys()))
            b_binds: Dict[str, Any] = _TracedBindings(ctx, b_clone)
            b_binds[b_in_name] = in_var
            for bk, bv in b_extra.items():
                if bk not in b_binds:
                    b_binds[bk] = bv
            b_out_concrete = _instantiate_bridge_output_signature(b_clone, b_out_p, in_sig, b_in_name=b_in_name, v_name=in_var)
            if b_out_concrete is not None and getattr(b_out_p, "name", None) and b_out_p.name in (getattr(b_clone, "outputs", {}) or {}):
                b_clone.outputs[b_out_p.name] = b_out_concrete
            var_counter_ref[0] += 1
            ctx.var_counter = var_counter_ref[0]
            out_sig_use = b_out_concrete if b_out_concrete is not None else b_out_p
            b_out_var = _semantic_var_name(
                b_clone,
                var_counter_ref[0],
                port_name=getattr(b_out_p, "name", "output_var"),
                out_sig=out_sig_use,
            )
            b_binds["output_var"] = b_out_var
            if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                ctx.var_port_map = {}
            if getattr(b_out_p, "name", None):
                b_binds[b_out_p.name] = b_out_var
                cell_port_vars[(b_clone.cell_id, b_out_p.name)] = b_out_var
                cell_port_vars[(id(b_clone), b_out_p.name)] = b_out_var
                ctx.var_port_map[(id(b_clone), b_out_p.name)] = b_out_var
            cell_out_vars[b_clone.cell_id] = b_out_var
            cell_out_vars[id(b_clone)] = b_out_var
            cell_port_vars[(b_clone.cell_id, "output_var")] = b_out_var
            cell_port_vars[(id(b_clone), "output_var")] = b_out_var
            ctx.var_port_map[(id(b_clone), "output_var")] = b_out_var
            ctx.declare_variable(b_out_var, out_sig_use, b_out_var, cell=b_clone)
            _update_variable_provenance(ctx, b_out_var, b_clone, b_binds, out_port_name=getattr(b_out_p, "name", None))
            try:
                self.scope_stage.finalize_cell_scope(b_clone, b_binds, ctx, accumulated_sigma)
            except Exception:
                pass
            pipeline_bindings.append((b_clone, b_binds))
            if hasattr(ctx, "consumed_vars"):
                ctx.consumed_vars.add(in_var)
            return b_out_var, out_sig_use

        for req_p in req_ports:
            p_tn = str(getattr(getattr(req_p, "signature", req_p), "type_name", "") or "").lower()
            req_role = str(getattr(req_p, "port_role", "") or getattr(req_p, "role", "") or getattr(req_p, "derived_role", "") or "").strip().lower()
            req_pk = str(getattr(req_p, "param_kind", "") or "").strip().lower()
            if _lattice_is_path_port(req_p) or _is_key_or_complement_port(consumer_cell, req_p, registry):
                continue
            if req_role in cfg_roles or req_pk in cfg_roles:
                continue
            if p_tn != str.__name__ and registry.is_builtin_scalar_primitive(p_tn) and p_tn != bytes.__name__:
                continue
            if p_tn == str.__name__ and not (
                req_p is prim_in_ref
                or getattr(req_p, "name", None) == prim_in_name
                or req_role in data_roles
            ):
                continue
            for cont_v in cont_cands:
                cont_sig, _ = ctx.variables[cont_v]
                if _is_product_sig(cont_sig):
                    continue
                if unify(getattr(cont_sig, "signature", cont_sig), getattr(req_p, "signature", req_p), accumulated_sigma) is not None:
                    continue
                lift = self._find_optic_lifted_getter_for_port(cont_v, cont_sig, req_p, consumer_cell, ctx, registry)
                if lift is None:
                    continue
                g_clone, setter, key_expr, g_focus, focus_bridge = lift
                g_cont, g_key, _ = self._optic_getter_ports(g_clone)
                g_bindings: Dict[str, Any] = _TracedBindings(ctx, g_clone)
                g_bindings[g_cont.name] = cont_v
                g_bindings[g_key.name] = key_expr
                for gp_n, gp_s in (getattr(g_clone, "inputs", {}) or {}).items():
                    if gp_n in g_bindings:
                        continue
                    g_bindings[gp_n] = str(gp_s.default_value) if getattr(gp_s, "default_value", None) is not None else None

                var_counter_ref[0] += 1
                ctx.var_counter = var_counter_ref[0]
                focus_var = _semantic_var_name(
                    g_clone,
                    var_counter_ref[0],
                    port_name=getattr(g_focus, "name", "output_var"),
                    out_sig=g_focus,
                )
                g_bindings["output_var"] = focus_var
                if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                    ctx.var_port_map = {}
                if getattr(g_focus, "name", None):
                    g_bindings[g_focus.name] = focus_var
                    cell_port_vars[(g_clone.cell_id, g_focus.name)] = focus_var
                    cell_port_vars[(id(g_clone), g_focus.name)] = focus_var
                    ctx.var_port_map[(id(g_clone), g_focus.name)] = focus_var
                cell_out_vars[g_clone.cell_id] = focus_var
                cell_out_vars[id(g_clone)] = focus_var
                cell_port_vars[(g_clone.cell_id, "output_var")] = focus_var
                cell_port_vars[(id(g_clone), "output_var")] = focus_var
                ctx.var_port_map[(id(g_clone), "output_var")] = focus_var

                lifted_keys = _extract_keys_from_val(key_expr)
                if not hasattr(ctx, "getter_read_keys") or ctx.getter_read_keys is None:
                    ctx.getter_read_keys = set()
                ctx.getter_read_keys.update(k.lower() for k in lifted_keys if str(k).strip())

                concrete_focus = substitute_generics(g_focus, accumulated_sigma)
                key_quals: Set[str] = set()
                c_kq_map = getattr(ctx, "container_key_qualifiers", None) or {}
                for lk in lifted_keys:
                    key_quals.update(c_kq_map.get((cont_v, str(lk).strip().lower()), set()) or set())
                if key_quals:
                    concrete_focus = _enrich_port_signature_qualifiers(concrete_focus, key_quals)
                matched_lay = _lookup_consistent_key_layout(ctx, cont_v, lifted_keys)
                if matched_lay is not None:
                    concrete_focus = _enrich_port_signature_layout(concrete_focus, matched_lay, registry)
                ctx.declare_variable(focus_var, concrete_focus, focus_var, cell=g_clone)
                _update_variable_provenance(ctx, focus_var, g_clone, g_bindings, out_port_name=getattr(g_focus, "name", None))
                try:
                    self.scope_stage.finalize_cell_scope(g_clone, g_bindings, ctx, accumulated_sigma)
                except Exception:
                    pass
                pipeline_bindings.append((g_clone, g_bindings))

                initial_focus_var = focus_var
                if focus_bridge is not None:
                    b_c, b_i, b_o, b_ex = focus_bridge
                    focus_var, concrete_focus = _emit_bridge_step(b_c, b_i, b_o, b_ex, focus_var, concrete_focus)

                # Eliminator bridge insertion if projected/bridged key carries a rejected qualifier (e.g. complex -> abs)
                rej_quals = (
                    _extract_rejected_qualifiers(req_p, consumer_cell=consumer_cell)
                    | _extract_rejected_qualifiers(getattr(req_p, "signature", req_p), consumer_cell=consumer_cell)
                )
                if rej_quals:
                    f_raw_q = _get_p_field(concrete_focus, "qualifiers", None) or getattr(getattr(concrete_focus, "signature", None), "qualifiers", None) or ()
                    flat_fq: Set[str] = set()
                    for q in f_raw_q:
                        if isinstance(q, (tuple, list, set)):
                            flat_fq.update(str(x).lower() for x in q)
                        else:
                            flat_fq.add(str(q).lower())
                    conflict = flat_fq & rej_quals
                    if conflict:
                        q_bridge = _find_bridge_for_qualifier_conflict(
                            getattr(self, "orchestrator", None),
                            concrete_focus,
                            getattr(req_p, "signature", req_p),
                            conflict,
                            rej_quals,
                            consumer_cell=consumer_cell,
                        )
                        if q_bridge is not None:
                            qb_in = getattr(q_bridge, "primary_input", None) or next(iter((getattr(q_bridge, "inputs", {}) or {}).values()))
                            qb_out = getattr(q_bridge, "primary_output", None) or next(iter((getattr(q_bridge, "outputs", {}) or {}).values()))
                            focus_var, concrete_focus = _emit_bridge_step(q_bridge, qb_in, qb_out, {}, focus_var, concrete_focus)
                            ctx.trace_binding("lifted_qualifier_bridge_inserted", consumer=consumer_cell.cell_id, bridge=q_bridge.cell_id, conflict=sorted(conflict), var=focus_var)

                frame = {
                    "container_var": cont_v,
                    "key_expr": key_expr,
                    "key_set": lifted_keys,
                    "focus_var": focus_var,
                    "initial_focus_var": initial_focus_var,
                    "getter_cell": g_clone,
                    "setter_cell": setter,
                    "dirty": False,
                    "lifted": True,
                    "closed": False,
                }
                ctx.optic_frames.append(frame)

                u_new = unify(getattr(concrete_focus, "signature", concrete_focus), getattr(req_p, "signature", req_p), accumulated_sigma)
                if u_new is not None:
                    accumulated_sigma = u_new
                cell_bindings[req_p.name] = focus_var
                if hasattr(ctx, "consumed_vars"):
                    ctx.consumed_vars.add(focus_var)
                ctx.trace_binding("optic_lens_lifted", consumer=consumer_cell.cell_id, getter=g_clone.cell_id, container=cont_v, key=key_expr, focus=focus_var)
                return (focus_var, accumulated_sigma)
        return None

    def _resolve_role_partitioned_projection(
        self,
        cell: Cell,
        p_name: str,
        p_sig: Any,
        concrete_sig: Any,
        cell_bindings: Dict[str, Any],
        ctx: ExecutionContext,
        accumulated_sigma: Substitution,
        registry: Optional[TypeRegistry] = None,
        pipeline_bindings: Optional[List[Tuple[Cell, Dict[str, str]]]] = None,
        var_counter_ref: Optional[List[int]] = None,
        cell_out_vars: Optional[Dict[Any, str]] = None,
        cell_port_vars: Optional[Dict[Tuple[Any, str], str]] = None,
    ) -> Optional[Tuple[str, Substitution]]:
        """Resolves declarative port relations without domain hardcodes:
        1. `key_from_upstream_role`: inherits key literals stored under upstream role tags.
        2. Open optic frame key recovery for explicit `setter` key ports.
        3. `complement_of_port` (Case A: key/collection exclusion, e.g. complement dropper).
        4. `complement_of_port` (Case B: container complementation, auto-inserting `optic_role=='complement'` cell)."""
        reg = registry if registry is not None else TypeRegistry.get_instance()
        sig_inner = getattr(concrete_sig, "signature", concrete_sig)
        t_name = str(getattr(sig_inner, "type_name", "") or "").strip().lower()
        is_coll = (
            getattr(p_sig, "structure_kind", None) == "collection"
            or reg.is_subtype(t_name, "collection")
            or reg.is_subtype(t_name, "sequence")
        )

        # 1. key_from_upstream_role (e.g. merge/join inheriting groupby key)
        kfur = getattr(p_sig, "key_from_upstream_role", None) or getattr(concrete_sig, "key_from_upstream_role", None)
        if kfur:
            role_keys = list((getattr(ctx, "role_bound_keys", None) or {}).get(str(kfur).strip().lower(), []) or [])
            if role_keys:
                val_expr = json.dumps(role_keys) if is_coll else json.dumps(role_keys[-1])
                ctx.trace_binding("resolved_key_from_upstream_role", cell=cell.cell_id, port=p_name, role=kfur, value=val_expr)
                return (val_expr, Substitution(accumulated_sigma.mappings))

        # 2. Explicit optic setter recovering key from open optic frame
        if str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter" and _port_matches_declared_ref(cell, "key_port", p_sig):
            s_cont, _, s_focus, _ = self._optic_setter_ports(cell)
            bound_cont = cell_bindings.get(s_cont.name) if s_cont else None
            bound_focus = cell_bindings.get(s_focus.name) if s_focus else None
            for fr in reversed(getattr(ctx, "optic_frames", []) or []):
                if fr.get("closed"):
                    continue
                if bound_cont and fr.get("container_var") != bound_cont:
                    continue
                if bound_focus and fr.get("focus_var") != bound_focus:
                    continue
                if fr.get("key_expr"):
                    ctx.trace_binding("resolved_setter_key_from_optic", cell=cell.cell_id, port=p_name, value=fr["key_expr"])
                    return (fr["key_expr"], Substitution(accumulated_sigma.mappings))

        # 3 & 4. complement_of_port
        comp_ref = getattr(p_sig, "complement_of_port", None) or getattr(concrete_sig, "complement_of_port", None)
        if not comp_ref:
            return None

        comp_port_obj = _resolve_cell_port(cell, comp_ref, prefer="inputs")
        is_key_port = _is_key_or_complement_port(cell, p_sig, reg) and not (
            reg.is_composite_carrier(t_name) or reg.is_abstract_carrier(t_name)
        )

        # Case A: key/collection exclusion list (e.g. PD_DROP_COLUMNS.columns with complement_of_port='target')
        if comp_port_obj is None or is_key_port:
            excl: List[str] = []
            seen_excl: Set[str] = set()

            def _add_k(k_str: Any) -> None:
                ks = str(k_str).strip("'\"")
                if ks and ks not in seen_excl:
                    seen_excl.add(ks)
                    excl.append(ks)

            if comp_port_obj is not None:
                partner_b = cell_bindings.get(comp_port_obj.name)
                if isinstance(partner_b, str) and partner_b in ctx.variables:
                    for k_it in sorted(_var_restricted_keys(partner_b, ctx)):
                        _add_k(k_it)
                elif partner_b is not None:
                    for k_it in sorted(_extract_keys_from_val(partner_b)):
                        _add_k(k_it)
            for k_it in (getattr(ctx, "role_bound_keys", None) or {}).get(str(comp_ref).strip().lower(), []) or []:
                _add_k(k_it)
            for k_it in sorted(getattr(ctx, "target_literals", set()) or ()):
                _add_k(k_it)
            # Also gather restricted slices derived from the bound container of this complement cell
            cont_port_obj = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs") or getattr(cell, "primary_input", None)
            bound_cont = cell_bindings.get(cont_port_obj.name) if cont_port_obj else None
            if isinstance(bound_cont, str) and bound_cont in ctx.variables:
                for vn in ctx.variables:
                    restr = _var_restricted_keys(vn, ctx)
                    if restr and (bound_cont in _var_ancestors(vn, ctx) or (_var_ancestors(bound_cont, ctx) & _var_ancestors(vn, ctx))):
                        for k_it in sorted(restr):
                            _add_k(k_it)
            if excl:
                val_expr = json.dumps(excl) if is_coll else json.dumps(excl[-1])
                ctx.trace_binding("resolved_complement_key_list", cell=cell.cell_id, port=p_name, value=val_expr)
                return (val_expr, Substitution(accumulated_sigma.mappings))
            return None

        # Case B: full container complementation (e.g. fit.X / train_test_split.X with complement_of_port='y')
        partner_var = cell_bindings.get(comp_port_obj.name)
        if not isinstance(partner_var, str) or partner_var not in ctx.variables:
            return None
        excl_keys = _var_restricted_keys(partner_var, ctx)
        sf_map = getattr(ctx, "setter_focus_keys", None) or {}
        has_sf_excl = any(c_p == partner_var and ks for (_, c_p), ks in sf_map.items())
        if not excl_keys and not has_sf_excl:
            return None

        # B1. Reuse an already-complemented in-scope carrier if compatible
        for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
            if v_name == partner_var or _is_product_sig(v_sig):
                continue
            v_excl = set((getattr(ctx, "var_excluded_origins", None) or {}).get(v_name, ()) or ())
            if not (excl_keys <= v_excl):
                continue
            if not _shape_compatible(v_sig, concrete_sig, producer_cell=ctx.var_sources.get(v_name), consumer_cell=cell):
                continue
            if _wire_rejection(cell, p_name, p_sig, v_name, ctx, reg, cell_bindings=cell_bindings) is not None:
                continue
            u_reuse = unify(getattr(v_sig, "signature", v_sig), sig_inner, accumulated_sigma)
            if u_reuse is not None:
                ctx.trace_binding("reused_complement_carrier", cell=cell.cell_id, port=p_name, var=v_name)
                return (v_name, u_reuse)

        if pipeline_bindings is None or var_counter_ref is None:
            return None

        def _find_unary_bridge(src_sig_obj: Any, src_cell_obj: Any) -> Optional[Tuple[Any, Any, Any, Dict[str, Any], Substitution]]:
            best_hit: Optional[Tuple[Tuple[int, int, int], Any, Any, Any, Dict[str, Any], Substitution]] = None
            src_inner = getattr(src_sig_obj, "signature", src_sig_obj)
            src_tn = str(getattr(src_inner, "type_name", "") or "").lower()
            dst_tn = t_name
            for b_cand in self._get_loaded_cells_pool():
                if str(getattr(b_cand, "node_role", "") or "").strip().lower() != "bridge":
                    continue
                b_inputs = getattr(b_cand, "inputs", {}) or {}
                b_outputs = getattr(b_cand, "outputs", {}) or {}
                if not b_inputs or not b_outputs:
                    continue
                b_in = getattr(b_cand, "primary_input", None) or next(iter(b_inputs.values()), None)
                b_out = getattr(b_cand, "primary_output", None) or next(iter(b_outputs.values()), None)
                if b_in is None or b_out is None:
                    continue
                if not _shape_compatible(src_sig_obj, b_in, producer_cell=src_cell_obj, consumer_cell=b_cand):
                    continue
                if unify(src_inner, getattr(b_in, "signature", b_in)) is None:
                    continue
                if not _shape_compatible(b_out, concrete_sig, producer_cell=b_cand, consumer_cell=cell):
                    continue
                u_b = unify(getattr(b_out, "signature", b_out), sig_inner, accumulated_sigma)
                if u_b is None:
                    continue
                extra_binds: Dict[str, Any] = {}
                ok_b = True
                for bp_n, bp_s in b_inputs.items():
                    if bp_s is b_in or bp_n == getattr(b_in, "name", None):
                        continue
                    if getattr(bp_s, "required", False):
                        if getattr(bp_s, "default_value", None) is not None:
                            extra_binds[bp_n] = str(bp_s.default_value)
                        else:
                            eta = _pointed_default(bp_s, b_cand)
                            if eta is not None:
                                extra_binds[bp_n] = eta
                            else:
                                ok_b = False
                                break
                    else:
                        extra_binds[bp_n] = None
                if not ok_b:
                    continue
                exact_in = 1 if src_tn == str(getattr(getattr(b_in, "signature", b_in), "type_name", "") or "").lower() else 0
                exact_out = 1 if dst_tn == str(getattr(getattr(b_out, "signature", b_out), "type_name", "") or "").lower() else 0
                vis_tie = (1 if getattr(b_cand, "is_public", True) else 0, -str(getattr(b_cand, "cell_id", "")).count("."))
                score = (exact_out, exact_in, vis_tie)
                if best_hit is None or score > best_hit[0]:
                    best_hit = (score, b_cand, b_in, b_out, extra_binds, u_b)
            if best_hit is None:
                return None
            _, b_cand, b_in, b_out, extra_binds, u_b = best_hit
            return (b_cand, b_in, b_out, extra_binds, u_b)

        def _emit_step_into_pipeline(step_cell: Any, step_bindings: Dict[str, Any], out_port_sig: Any) -> str:
            var_counter_ref[0] += 1
            ctx.var_counter = var_counter_ref[0]
            out_v = _semantic_var_name(
                step_cell,
                var_counter_ref[0],
                port_name=getattr(out_port_sig, "name", "output_var"),
                out_sig=out_port_sig,
            )
            step_bindings["output_var"] = out_v
            if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                ctx.var_port_map = {}
            _is_auto_bridge = bool(getattr(step_cell, "_auto_inserted_bridge", False))
            if getattr(out_port_sig, "name", None):
                step_bindings[out_port_sig.name] = out_v
                ctx.var_port_map[(id(step_cell), out_port_sig.name)] = out_v
                if cell_port_vars is not None:
                    cell_port_vars[(id(step_cell), out_port_sig.name)] = out_v
                    if _is_auto_bridge:
                        cell_port_vars.setdefault((step_cell.cell_id, out_port_sig.name), out_v)
                    else:
                        cell_port_vars[(step_cell.cell_id, out_port_sig.name)] = out_v
            if cell_out_vars is not None:
                cell_out_vars[id(step_cell)] = out_v
                if _is_auto_bridge:
                    cell_out_vars.setdefault(step_cell.cell_id, out_v)
                else:
                    cell_out_vars[step_cell.cell_id] = out_v
            ctx.var_port_map[(id(step_cell), "output_var")] = out_v
            if cell_port_vars is not None:
                cell_port_vars[(id(step_cell), "output_var")] = out_v
                if _is_auto_bridge:
                    cell_port_vars.setdefault((step_cell.cell_id, "output_var"), out_v)
                else:
                    cell_port_vars[(step_cell.cell_id, "output_var")] = out_v
            concrete_step_out = substitute_generics(out_port_sig, accumulated_sigma)
            ctx.declare_variable(out_v, concrete_step_out, out_v, cell=step_cell)
            _update_variable_provenance(ctx, out_v, step_cell, step_bindings, out_port_name=getattr(out_port_sig, "name", None))
            try:
                self.scope_stage.finalize_cell_scope(step_cell, step_bindings, ctx, accumulated_sigma)
            except Exception:
                pass
            pipeline_bindings.append((step_cell, step_bindings))
            return out_v

        # B1b. Reuse an already-complemented in-scope carrier via 1-hop unary bridge morphism
        sup_set = getattr(ctx, "superseded_vars", set()) or set()
        b1b_vars = [
            v_n for v_n in reversed(list(ctx.variables.keys()))
            if v_n != partner_var and not _is_product_sig(ctx.variables[v_n][0])
        ]
        b1b_vars.sort(key=lambda vn: 1 if vn not in sup_set else 0, reverse=True)
        for v_name in b1b_vars:
            v_sig, _ = ctx.variables[v_name]
            v_excl = set((getattr(ctx, "var_excluded_origins", None) or {}).get(v_name, ()) or ())
            if not (excl_keys <= v_excl):
                continue
            bridge_match = _find_unary_bridge(v_sig, ctx.var_sources.get(v_name))
            if bridge_match is None:
                continue
            b_cand, b_in, b_out, b_extra, u_b = bridge_match
            b_clone = b_cand.clone() if hasattr(b_cand, "clone") else copy.copy(b_cand)
            b_clone.matched_clause_idx = getattr(cell, "matched_clause_idx", None)
            b_binds: Dict[str, Any] = _TracedBindings(ctx, b_clone)
            b_in_name = getattr(b_in, "name", None) or next(iter((getattr(b_clone, "inputs", {}) or {}).keys()))
            b_binds[b_in_name] = v_name
            for bk, bv in b_extra.items():
                if bk not in b_binds:
                    b_binds[bk] = bv
            b_out_sig = getattr(b_clone, "primary_output", None) or next(iter((getattr(b_clone, "outputs", {}) or {}).values()))
            _snap_pb_len = len(pipeline_bindings)
            _snap_vc = var_counter_ref[0]
            _snap_sup = set(getattr(ctx, "superseded_vars", set()) or set())
            bridge_out_var = _emit_step_into_pipeline(b_clone, b_binds, b_out_sig)
            if _wire_rejection(cell, p_name, p_sig, bridge_out_var, ctx, reg, cell_bindings=cell_bindings) is not None:
                del pipeline_bindings[_snap_pb_len:]
                var_counter_ref[0] = _snap_vc
                ctx.var_counter = _snap_vc
                ctx.variables.pop(bridge_out_var, None)
                if hasattr(ctx, "superseded_vars"):
                    ctx.superseded_vars = _snap_sup
                continue
            ctx.trace_binding("reused_complement_carrier_via_bridge", cell=cell.cell_id, port=p_name, source_var=v_name, bridge=b_clone.cell_id, out_var=bridge_out_var)
            return (bridge_out_var, u_b)

        # B2. Auto-insert a domain complement morphism (`optic_role == 'complement'`), plus optional 1-hop bridge
        # Locate nearest ancestor container of `partner_var` in ctx.variables
        cont_candidates: List[str] = []
        visited_anc: Set[str] = {partner_var}
        queue_anc: List[str] = list((getattr(ctx, "var_parents", None) or {}).get(partner_var, ()) or ())
        while queue_anc:
            curr = queue_anc.pop(0)
            if curr in visited_anc:
                continue
            visited_anc.add(curr)
            if curr in ctx.variables and not _is_product_sig(ctx.variables[curr][0]):
                c_sig_cand = getattr(ctx.variables[curr][0], "signature", ctx.variables[curr][0])
                c_tn_cand = str(getattr(c_sig_cand, "type_name", "") or "").lower()
                if reg.is_composite_carrier(c_tn_cand) or reg.is_abstract_carrier(c_tn_cand) or unify(c_sig_cand, sig_inner, accumulated_sigma) is not None:
                    cont_candidates.append(curr)
            for nxt in (getattr(ctx, "var_parents", None) or {}).get(curr, ()) or ():
                if nxt not in visited_anc:
                    queue_anc.append(nxt)
        act_v = getattr(ctx, "active_carrier_var", None)
        if act_v and act_v in ctx.variables and act_v not in cont_candidates and act_v != partner_var:
            cont_candidates.append(act_v)
        for v_k in reversed(list(ctx.variables.keys())):
            if v_k == partner_var or v_k in cont_candidates or _is_product_sig(ctx.variables[v_k][0]):
                continue
            v_sig_k = getattr(ctx.variables[v_k][0], "signature", ctx.variables[v_k][0])
            v_tn_k = str(getattr(v_sig_k, "type_name", "") or "").lower()
            if reg.is_composite_carrier(v_tn_k) or reg.is_abstract_carrier(v_tn_k):
                cont_candidates.append(v_k)
        var_recency = {vn: i for i, vn in enumerate(ctx.variables.keys())}
        excl_map_case_b = getattr(ctx, "var_excluded_origins", {}) or {}
        cont_candidates.sort(
            key=lambda c: (
                1 if c not in sup_set else 0,
                1 if c == act_v else 0,
                1 if (c, partner_var) in sf_map else 0,
                1 if (sf_map.get((c, partner_var)) or excl_keys) <= set(getattr(ctx, "var_origins", {}).get(c, set()) or set()) else 0,
                0 if ((sf_map.get((c, partner_var)) or excl_keys) <= set(excl_map_case_b.get(c, set()) or set())) else 1,
                var_recency.get(c, -1),
            ),
            reverse=True,
        )

        for cont_var in cont_candidates:
            cont_sf_keys = sf_map.get((cont_var, partner_var))
            eff_excl_keys = set(cont_sf_keys) if cont_sf_keys else set(excl_keys)
            if not eff_excl_keys:
                continue
            cont_sig, _ = ctx.variables[cont_var]
            cont_src_cell = ctx.var_sources.get(cont_var)
            for cand_comp in self._get_loaded_cells_pool():
                if str(getattr(cand_comp, "optic_role", "") or "").strip().lower() != "complement":
                    continue
                cc_cont = _resolve_cell_port(cand_comp, getattr(cand_comp, "container_port", None), prefer="inputs") or getattr(cand_comp, "primary_input", None)
                cc_key = _resolve_cell_port(cand_comp, getattr(cand_comp, "key_port", None), prefer="inputs") or next(
                    (p for p in (getattr(cand_comp, "inputs", {}) or {}).values() if getattr(p, "complement_of_port", None)),
                    None,
                )
                cc_out = getattr(cand_comp, "primary_output", None) or next(iter((getattr(cand_comp, "outputs", {}) or {}).values()), None)
                if cc_cont is None or cc_key is None or cc_out is None:
                    continue
                if not _shape_compatible(cont_sig, cc_cont, producer_cell=cont_src_cell, consumer_cell=cand_comp):
                    continue
                if unify(getattr(cont_sig, "signature", cont_sig), getattr(cc_cont, "signature", cc_cont)) is None:
                    continue

                direct_u: Optional[Substitution] = None
                bridge_match = None
                if _shape_compatible(cc_out, concrete_sig, producer_cell=cand_comp, consumer_cell=cell):
                    direct_u = unify(getattr(cc_out, "signature", cc_out), sig_inner, accumulated_sigma)
                if direct_u is None:
                    bridge_match = _find_unary_bridge(cc_out, cand_comp)
                    if bridge_match is None:
                        continue

                extra_comp_binds: Dict[str, Any] = {}
                ok_req = True
                for cp_n, cp_s in (getattr(cand_comp, "inputs", {}) or {}).items():
                    if cp_n in (cc_cont.name, cc_key.name):
                        continue
                    if getattr(cp_s, "required", False):
                        if getattr(cp_s, "default_value", None) is not None:
                            extra_comp_binds[cp_n] = str(cp_s.default_value)
                        else:
                            eta = _pointed_default(cp_s, cand_comp)
                            if eta is not None:
                                extra_comp_binds[cp_n] = eta
                            else:
                                ok_req = False
                                break
                    else:
                        extra_comp_binds[cp_n] = None
                if not ok_req:
                    continue

                pb_snap_len = len(pipeline_bindings)
                vc_snap = var_counter_ref[0]
                vars_snap = dict(ctx.variables)
                sources_snap = dict(ctx.var_sources)
                origins_snap = {k: set(v) for k, v in getattr(ctx, "var_origins", {}).items()}
                restr_snap = {k: set(v) for k, v in getattr(ctx, "var_restricted_origins", {}).items()}
                excl_snap = {k: set(v) for k, v in getattr(ctx, "var_excluded_origins", {}).items()}
                epochs_snap = dict(getattr(ctx, "var_epochs", {}) or {})
                parts_snap = dict(getattr(ctx, "var_partitions", {}) or {})
                parted_snap = dict(getattr(ctx, "partitioned_vars", {}) or {})
                parents_snap = {k: set(v) for k, v in getattr(ctx, "var_parents", {}).items()}
                vpmap_snap = dict(getattr(ctx, "var_port_map", {}) or {})
                vctok_snap = {k: set(v) for k, v in getattr(ctx, "var_clause_tokens", {}).items()}
                pvars_snap = set(getattr(ctx, "pipeline_vars", set()) or ())
                scope_snap = dict(getattr(ctx, "scope", {}) or {})
                sup_snap = set(getattr(ctx, "superseded_vars", set()) or ())
                cons_vars_snap = set(getattr(ctx, "consumed_vars", set()) or ())
                act_snap = getattr(ctx, "active_carrier_var", None)
                cov_snap = dict(cell_out_vars)
                cpv_snap = dict(cell_port_vars)
                sf_snap = {k: set(v) for k, v in sf_map.items()} if isinstance(sf_map, dict) else None

                def _rollback_case_b2() -> None:
                    del pipeline_bindings[pb_snap_len:]
                    var_counter_ref[0] = vc_snap
                    ctx.var_counter = vc_snap
                    ctx.variables.clear()
                    ctx.variables.update(vars_snap)
                    ctx.var_sources.clear()
                    ctx.var_sources.update(sources_snap)
                    if hasattr(ctx, "var_origins") and isinstance(ctx.var_origins, dict):
                        ctx.var_origins.clear()
                        ctx.var_origins.update(origins_snap)
                    if hasattr(ctx, "var_restricted_origins") and isinstance(ctx.var_restricted_origins, dict):
                        ctx.var_restricted_origins.clear()
                        ctx.var_restricted_origins.update(restr_snap)
                    if hasattr(ctx, "var_excluded_origins") and isinstance(ctx.var_excluded_origins, dict):
                        ctx.var_excluded_origins.clear()
                        ctx.var_excluded_origins.update(excl_snap)
                    if hasattr(ctx, "var_epochs") and isinstance(ctx.var_epochs, dict):
                        ctx.var_epochs.clear()
                        ctx.var_epochs.update(epochs_snap)
                    if hasattr(ctx, "var_partitions") and isinstance(ctx.var_partitions, dict):
                        ctx.var_partitions.clear()
                        ctx.var_partitions.update(parts_snap)
                    if hasattr(ctx, "partitioned_vars") and isinstance(ctx.partitioned_vars, dict):
                        ctx.partitioned_vars.clear()
                        ctx.partitioned_vars.update(parted_snap)
                    if hasattr(ctx, "var_parents") and isinstance(ctx.var_parents, dict):
                        ctx.var_parents.clear()
                        ctx.var_parents.update(parents_snap)
                    if hasattr(ctx, "var_port_map") and isinstance(ctx.var_port_map, dict):
                        ctx.var_port_map.clear()
                        ctx.var_port_map.update(vpmap_snap)
                    if hasattr(ctx, "var_clause_tokens") and isinstance(ctx.var_clause_tokens, dict):
                        ctx.var_clause_tokens.clear()
                        ctx.var_clause_tokens.update(vctok_snap)
                    if hasattr(ctx, "pipeline_vars") and isinstance(ctx.pipeline_vars, set):
                        ctx.pipeline_vars.clear()
                        ctx.pipeline_vars.update(pvars_snap)
                    if hasattr(ctx, "scope") and isinstance(ctx.scope, dict):
                        ctx.scope.clear()
                        ctx.scope.update(scope_snap)
                    if hasattr(ctx, "superseded_vars") and isinstance(ctx.superseded_vars, set):
                        ctx.superseded_vars.clear()
                        ctx.superseded_vars.update(sup_snap)
                    if hasattr(ctx, "consumed_vars") and isinstance(ctx.consumed_vars, set):
                        ctx.consumed_vars.clear()
                        ctx.consumed_vars.update(cons_vars_snap)
                    ctx.active_carrier_var = act_snap
                    cell_out_vars.clear()
                    cell_out_vars.update(cov_snap)
                    cell_port_vars.clear()
                    cell_port_vars.update(cpv_snap)
                    if sf_snap is not None and isinstance(sf_map, dict):
                        sf_map.clear()
                        sf_map.update(sf_snap)

                comp_clone = cand_comp.clone() if hasattr(cand_comp, "clone") else copy.copy(cand_comp)
                comp_clone.matched_clause_idx = getattr(cell, "matched_clause_idx", None)
                comp_clone._auto_inserted_bridge = True
                c_binds: Dict[str, Any] = _TracedBindings(ctx, comp_clone)
                c_binds[cc_cont.name] = cont_var

                k_sig_inner = getattr(cc_key, "signature", cc_key)
                k_tn = str(getattr(k_sig_inner, "type_name", "") or "").lower()
                k_is_coll = (
                    getattr(cc_key, "structure_kind", None) == "collection"
                    or reg.is_subtype(k_tn, "collection")
                    or reg.is_subtype(k_tn, "sequence")
                )
                sorted_excl = sorted(eff_excl_keys)
                c_binds[cc_key.name] = json.dumps(sorted_excl) if k_is_coll else json.dumps(sorted_excl[-1])
                for ck, cv in extra_comp_binds.items():
                    if ck not in c_binds:
                        c_binds[ck] = cv

                cc_out_sig = getattr(comp_clone, "primary_output", None) or next(iter((getattr(comp_clone, "outputs", {}) or {}).values()))
                comp_out_var = _emit_step_into_pipeline(comp_clone, c_binds, cc_out_sig)
                if comp_out_var and sf_map:
                    for (sc_v, sf_v), s_ks in list(sf_map.items()):
                        if sc_v == cont_var and s_ks:
                            sf_map.setdefault((comp_out_var, sf_v), set()).update(s_ks)

                if direct_u is not None:
                    if _wire_rejection(cell, p_name, p_sig, comp_out_var, ctx, reg, cell_bindings=cell_bindings) is not None:
                        _rollback_case_b2()
                        continue
                    ctx.trace_binding("complement_cell_auto_inserted", consumer=cell.cell_id, port=p_name, complement_cell=comp_clone.cell_id, excluded=sorted_excl, out_var=comp_out_var)
                    return (comp_out_var, direct_u)

                assert bridge_match is not None
                b_cand, b_in, b_out, b_extra, u_b = bridge_match
                b_clone = b_cand.clone() if hasattr(b_cand, "clone") else copy.copy(b_cand)
                b_clone.matched_clause_idx = getattr(cell, "matched_clause_idx", None)
                b_clone._auto_inserted_bridge = True
                b_binds: Dict[str, Any] = _TracedBindings(ctx, b_clone)
                b_in_name = getattr(b_in, "name", None) or next(iter((getattr(b_clone, "inputs", {}) or {}).keys()))
                b_binds[b_in_name] = comp_out_var
                for bk, bv in b_extra.items():
                    if bk not in b_binds:
                        b_binds[bk] = bv
                b_out_sig = getattr(b_clone, "primary_output", None) or next(iter((getattr(b_clone, "outputs", {}) or {}).values()))
                bridge_out_var = _emit_step_into_pipeline(b_clone, b_binds, b_out_sig)
                if bridge_out_var and comp_out_var and sf_map:
                    for (sc_v, sf_v), s_ks in list(sf_map.items()):
                        if sc_v == comp_out_var and s_ks:
                            sf_map.setdefault((bridge_out_var, sf_v), set()).update(s_ks)
                if _wire_rejection(cell, p_name, p_sig, bridge_out_var, ctx, reg, cell_bindings=cell_bindings) is not None:
                    _rollback_case_b2()
                    continue
                ctx.trace_binding("complement_cell_auto_inserted_with_bridge", consumer=cell.cell_id, port=p_name, complement_cell=comp_clone.cell_id, bridge=b_clone.cell_id, excluded=sorted_excl, out_var=bridge_out_var)
                return (bridge_out_var, u_b)
        return None

    def get_egress_paths(self) -> List[str]:
        """
        Returns destination artifact paths derived from the most recent synthesis.
        Single source of truth for sandbox egress verification: paths are the values
        bound to path-typed ports of terminal (Stage 3 / sink) morphisms during the
        last unify_pipeline run — never re-parsed from the raw prompt.
        """
        return list(self.last_egress_paths)

    @staticmethod
    def _derive_egress_paths(pipeline_bindings: List[Tuple[Cell, Dict[str, str]]]) -> List[str]:
        """
        Extracts egress destinations from verified pipeline bindings.
        A path qualifies as an egress artifact iff it is bound as a quoted literal
        to a path-typed port of:
          - a Stage 3 (terminal/egress) morphism, or
          - a cell whose output typestate declares materialization, or
          - any non-source (Stage 2+) morphism — in-place endomorphisms such as
            writers with a destination port (e.g. tabular writers) are Stage 2
            composable morphisms whose path port is still a materialization site.
        Type- and stage-driven, domain-agnostic.
        """
        registry = TypeRegistry.get_instance()
        egress: List[str] = []

        def _quoted_literal(v: Any) -> Optional[str]:
            if not isinstance(v, str):
                return None
            s = v.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                inner = s[1:-1].strip()
                return inner or None
            return None

        reg_eg = TypeRegistry.get_instance()
        term_roles = reg_eg.get_terminal_roles_set()
        ing_roles = reg_eg.get_ingress_roles_set()
        term_states = (
            set(reg_eg.get_terminal_states() or ())
            | reg_eg.get_states_by_property("is_terminal", True)
            | reg_eg.get_states_by_property("is_materialized", True)
        )
        term_stages = {3}
        ing_stages = {1}

        for cell, bindings in pipeline_bindings:
            stage = getattr(cell, "stage", None)
            node_role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
            is_terminal = (stage in term_stages) or (node_role in term_roles)
            if not is_terminal:
                for out_p in getattr(cell, "outputs", {}).values():
                    if str(getattr(out_p, "state", "")).lower() in term_states:
                        is_terminal = True
                        break
            is_source = (stage in ing_stages) or (node_role in ing_roles)
            if not is_terminal and is_source:
                continue

            for p_name, p_sig in getattr(cell, "inputs", {}).items():
                t_name = str(getattr(p_sig, "type_name", ""))
                is_path_port = _lattice_is_path_port(p_sig)
                if not is_path_port:
                    continue
                bound_val = bindings.get(p_name)
                unquoted = _quoted_literal(bound_val)
                if unquoted and (is_terminal or not is_source):
                    egress.append(unquoted)

        return egress

    def unify_transition(
        self,
        producer: Cell,
        consumer: Cell,
        current_sigma: Substitution,
        context: Optional[ExecutionContext] = None
    ) -> MonadResult[Substitution]:
        """
        Verifies that producer's output can satisfy an input of consumer,
        or that consumer's required inputs are satisfiable by available wires.
        Supports multi-port monoidal matching (⊗, Δ).
        """
        out_sig = producer.primary_output

        # 1. Primary input direct unification
        if _shape_compatible(out_sig, consumer.primary_input):
            new_sigma = unify(out_sig.signature, consumer.primary_input.signature, current_sigma)
            if new_sigma is not None:
                return Success(new_sigma, new_sigma)

        # 2. Multi-Port Monoidal Matching: check if producer output unifies with ANY input of consumer
        for p_name, p_port in consumer.inputs.items():
            if not _shape_compatible(out_sig, p_port):
                continue
            new_sigma = unify(out_sig.signature, p_port.signature, current_sigma)
            if new_sigma is not None:
                return Success(new_sigma, new_sigma)

        # 3. Port-sharing delta check: if consumer's inputs are satisfiable by in-scope variables
        if context is not None:
            for v_name, (v_sig, _) in context.variables.items():
                v_u = unify(v_sig.signature, consumer.primary_input.signature, current_sigma)
                if v_u is not None:
                    return Success(v_u, v_u)
            for p_name, p_port in consumer.inputs.items():
                if not getattr(p_port, "required", False) and p_port.default_value is not None:
                    continue
                for v_name, (v_sig, _) in context.variables.items():
                    if _shape_compatible(v_sig, p_port):
                        v_u = unify(v_sig.signature, p_port.signature, current_sigma)
                        if v_u is not None:
                            return Success(v_u, v_u)

        return Failure(
            f"Typestate Unification Failed: {producer.cell_id} outputs {out_sig.signature} "
            f"which cannot satisfy any input of {consumer.cell_id}"
        )


    @staticmethod
    def _semantic_var_name(
        cell: Any,
        counter: int,
        port_name: Optional[str] = None,
        out_sig: Optional[Any] = None,
    ) -> str:
        """
        Constructs a type-grounded semantic variable name (e.g. img_gray_1, edges_2)
        using the cell output's type name, state, and qualifiers.
        """
        sig = out_sig
        if sig is None and hasattr(cell, "primary_output") and cell.primary_output is not None:
            sig = cell.primary_output

        def _clean_token(s: Any) -> str:
            res = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s).lower()).strip("_")
            while "__" in res:
                res = res.replace("__", "_")
            return res

        _reg_sv = TypeRegistry.get_instance()
        prefix = ""
        if sig is not None:
            raw_sig = getattr(sig, "signature", sig)
            t_name = getattr(raw_sig, "type_name", None) or getattr(sig, "type_name", None)
            state = getattr(raw_sig, "state", None) or getattr(sig, "state", None)

            if t_name:
                t_clean = _clean_token(t_name)
                if (
                    t_clean
                    and not _reg_sv.is_declared_top(t_clean)
                    and not _reg_sv.is_void_type(t_clean)
                    and not _reg_sv.is_type_variable(t_clean)
                    and not _reg_sv.is_type_variable(t_name)
                ):
                    prefix = t_clean

            if not prefix and state:
                s_clean = _clean_token(state)
                if s_clean and not _reg_sv.is_declared_top_state(s_clean) and not _reg_sv.is_void_type(s_clean):
                    prefix = s_clean

        if not prefix and port_name:
            p_clean = _clean_token(port_name)
            if p_clean and p_clean not in {"output_var"}:
                prefix = p_clean

        if not prefix and hasattr(cell, "name") and cell.name:
            c_clean = _clean_token(cell.name)
            if c_clean:
                prefix = c_clean

        if not prefix:
            prefix = "var"
        if not prefix.isidentifier() or keyword.iskeyword(prefix):
            prefix = f"v_{prefix}"
        return f"{prefix}_{counter}"


    def unify_pipeline(
        self,
        cells: List[Cell],
        context: Optional[ExecutionContext] = None,
        allow_llm: bool = True,
        extract_llm_slots: bool = True,
    ) -> MonadResult[List[Tuple[Cell, Dict[str, str]]]]:
        """Symbolic-first pipeline unification with lazy hole-driven LLM slot/binding resolution."""
        if not cells:
            return Failure("Empty cell pipeline")
        if not extract_llm_slots:
            allow_llm = False

        ctx_target = context or ExecutionContext()
        can_try_llm_slots = False
        can_try_llm_synth = False
        if allow_llm:
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm_u = ModelManager.get_instance()
                can_try_llm_slots = bool(mm_u.can_extract_slots())
                can_try_llm_synth = bool(mm_u.can_synthesize())
            except Exception:
                can_try_llm_slots = False
                can_try_llm_synth = False

        base_retry_ctx = (
            ctx_target.clone()
            if ((can_try_llm_slots or can_try_llm_synth) and hasattr(ctx_target, "clone"))
            else None
        )
        # Pass 1: strictly zero-LLM deterministic unification
        res = self._unify_pipeline_impl(
            cells,
            context=ctx_target,
            allow_llm=False,
            extract_llm_slots=False,
            run_pipeline_llm_synth=False,
        )

        def _unresolved_count(monad_res: Any, uctx: Any) -> int:
            if monad_res.is_bottom() or not getattr(monad_res, "value", None):
                return 10**6
            cnt = max(
                len(getattr(uctx, "unresolved_ports", None) or ()),
                len(getattr(uctx, "typed_holes", None) or ()),
            )
            for _, bnd in monad_res.value:
                for val in (bnd or {}).values():
                    if is_unresolved_value(val):
                        cnt += 1
            return cnt

        p1_unres = _unresolved_count(res, ctx_target)
        if p1_unres == 0:
            return res

        if allow_llm and base_retry_ctx is not None and (can_try_llm_slots or can_try_llm_synth):
            best_res = res
            best_ctx = ctx_target
            best_unres = p1_unres
            saved_bindings = getattr(self, "last_pipeline_bindings", None)
            saved_typed = getattr(self, "last_pipeline_typed_bindings", None)
            saved_proof = getattr(self, "last_typed_proof", None)

            # Pass 2a: Lazy hole-driven slot extraction only when Pass 1 left unresolved holes
            if (
                can_try_llm_slots
                and extract_llm_slots
                and not getattr(ctx_target, "_cell_slots_extracted", False)
            ):
                slot_ctx = base_retry_ctx.clone()
                res_slots = self._unify_pipeline_impl(
                    cells,
                    context=slot_ctx,
                    allow_llm=True,
                    extract_llm_slots=True,
                    run_pipeline_llm_synth=False,
                )
                ctx_target._cell_slots_extracted = True
                base_retry_ctx._cell_slots_extracted = True
                if getattr(slot_ctx, "llm_slots", None):
                    base_retry_ctx.llm_slots = {
                        k: dict(v) if isinstance(v, dict) else v
                        for k, v in slot_ctx.llm_slots.items()
                    }
                    if hasattr(base_retry_ctx, "sync_llm_slots"):
                        base_retry_ctx.sync_llm_slots()
                if getattr(slot_ctx, "step_llm_slots", None):
                    base_retry_ctx.step_llm_slots = {
                        k: dict(v) for k, v in slot_ctx.step_llm_slots.items()
                    }
                u_slots = _unresolved_count(res_slots, slot_ctx)
                if u_slots < best_unres:
                    best_res = res_slots
                    best_ctx = slot_ctx
                    best_unres = u_slots
                    saved_bindings = getattr(self, "last_pipeline_bindings", None)
                    saved_typed = getattr(self, "last_pipeline_typed_bindings", None)
                    saved_proof = getattr(self, "last_typed_proof", None)

            # Pass 2a-bis: Targeted batched completion for any remaining unresolved required literal ports
            if (
                best_unres > 0
                and can_try_llm_slots
                and isinstance(best_res, Success)
                and getattr(best_res, "value", None)
            ):
                comp_ctx = base_retry_ctx.clone()
                comp_ctx._cell_slots_extracted = True
                filled_missing = PipelineSlotBinder.complete_missing_ports(
                    cells,
                    best_res.value,
                    (getattr(comp_ctx, "prompt", "") or getattr(comp_ctx, "_prompt", "") or getattr(self.context, "_prompt", "") or ""),
                    comp_ctx,
                    TypeRegistry.get_instance(),
                    is_path_port_fn=_lattice_is_path_port,
                    is_macro_block_slot_fn=_is_macro_block_slot,
                    is_unresolved_fn=is_unresolved_value,
                    is_path_string_fn=ExecutionContext._is_path_string,
                    type_term_cls=TypeTerm,
                )
                if filled_missing:
                    if getattr(comp_ctx, "llm_slots", None):
                        base_retry_ctx.llm_slots = {
                            k: dict(v) if isinstance(v, dict) else v
                            for k, v in comp_ctx.llm_slots.items()
                        }
                        if hasattr(base_retry_ctx, "sync_llm_slots"):
                            base_retry_ctx.sync_llm_slots()
                    if getattr(comp_ctx, "step_llm_slots", None):
                        base_retry_ctx.step_llm_slots = {
                            k: dict(v) for k, v in comp_ctx.step_llm_slots.items()
                        }
                    res_comp = self._unify_pipeline_impl(
                        cells,
                        context=comp_ctx,
                        allow_llm=True,
                        extract_llm_slots=False,
                        run_pipeline_llm_synth=False,
                    )
                    u_comp = _unresolved_count(res_comp, comp_ctx)
                    if u_comp < best_unres:
                        best_res = res_comp
                        best_ctx = comp_ctx
                        best_unres = u_comp
                        saved_bindings = getattr(self, "last_pipeline_bindings", None)
                        saved_typed = getattr(self, "last_pipeline_typed_bindings", None)
                        saved_proof = getattr(self, "last_typed_proof", None)

            # Pass 2b: Pipeline-level LLM binding synthesis ONLY if holes remain after Pass 2a
            if best_unres > 0 and can_try_llm_synth:
                synth_ctx = base_retry_ctx.clone()
                synth_ctx._cell_slots_extracted = True
                res_synth = self._unify_pipeline_impl(
                    cells,
                    context=synth_ctx,
                    allow_llm=True,
                    extract_llm_slots=False,
                    run_pipeline_llm_synth=True,
                )
                u_synth = _unresolved_count(res_synth, synth_ctx)
                if u_synth < best_unres:
                    best_res = res_synth
                    best_ctx = synth_ctx
                    best_unres = u_synth
                    saved_bindings = getattr(self, "last_pipeline_bindings", None)
                    saved_typed = getattr(self, "last_pipeline_typed_bindings", None)
                    saved_proof = getattr(self, "last_typed_proof", None)

            self.last_pipeline_bindings = saved_bindings
            self.last_pipeline_typed_bindings = saved_typed
            self.last_typed_proof = saved_proof
            if best_ctx is not ctx_target:
                try:
                    ctx_target.__dict__.update(best_ctx.__dict__)
                except Exception:
                    pass
                return best_res
        return res

    def _unify_pipeline_impl(
        self,
        cells: List[Cell],
        context: Optional[ExecutionContext] = None,
        allow_llm: bool = True,
        extract_llm_slots: bool = True,
        run_pipeline_llm_synth: bool = False,
    ) -> MonadResult[List[Tuple[Cell, Dict[str, str]]]]:
        """
        Chains a sequence of cells [v_1, ..., v_n] through the Type Monad.
        Binds port placeholders to variables in each step using multi-port monoidal matching.

        Structural extensions:
        - Zero-ary constructor morphisms (node_type 'constructor' with no required
          data port) are insertable at any position: they consume no incoming wire.
        - Heterogeneous product outputs (tuple[A, B, ...]) are projected member-wise:
          a downstream port binds to ``var[i]`` (product elimination). Members already
          consumed by earlier cells are deprioritized, so partitions (train/verify/test)
          allocate distinct members to distinct consumers by consumption order and
          declared member-state affinity with the consuming clause.
        """
        if not cells:
            return Failure("Empty cell pipeline")
        if not extract_llm_slots:
            allow_llm = False

        # Always clone each morphism instance so repeated pipelines never mutate shared cached Cell objects
        cells = [
            c.clone() if hasattr(c, "clone") else copy.copy(c)
            for c in cells
        ]

        registry = TypeRegistry.get_instance()
        ctx = context or ExecutionContext()
        ctx.current_pipeline = cells
        if hasattr(ctx, "multi_fiber_vars") and isinstance(ctx.multi_fiber_vars, set):
            ctx.multi_fiber_vars.clear()
        else:
            ctx.multi_fiber_vars = set()

        # Reset transient per-pipeline binding state on ctx at entry
        if hasattr(ctx, "variables") and isinstance(ctx.variables, dict):
            ctx.variables.clear()
        if hasattr(ctx, "initial_scope") and hasattr(ctx, "scope") and isinstance(ctx.scope, dict):
            init_keys = set(ctx.initial_scope)
            ctx.scope = {k: v for k, v in ctx.scope.items() if k in init_keys}
            for k, v in ctx.scope.items():
                ctx.declare_variable(k, v, k)
        elif hasattr(ctx, "scope") and isinstance(ctx.scope, dict) and hasattr(ctx, "variables"):
            ctx_pvars = getattr(ctx, "pipeline_vars", set())
            ctx.scope = {
                k: v
                for k, v in ctx.scope.items()
                if k not in ctx_pvars
            }
            for k, v in ctx.scope.items():
                ctx.declare_variable(k, v, k)
        if hasattr(ctx, "pipeline_vars") and isinstance(ctx.pipeline_vars, set):
            ctx.pipeline_vars.clear()
        if hasattr(ctx, "var_epochs") and isinstance(ctx.var_epochs, dict):
            ctx.var_epochs.clear()
        if hasattr(ctx, "var_origins") and isinstance(ctx.var_origins, dict):
            ctx.var_origins.clear()
        if hasattr(ctx, "var_restricted_origins") and isinstance(ctx.var_restricted_origins, dict):
            ctx.var_restricted_origins.clear()
        else:
            ctx.var_restricted_origins = {}
        if hasattr(ctx, "var_excluded_origins") and isinstance(ctx.var_excluded_origins, dict):
            ctx.var_excluded_origins.clear()
        else:
            ctx.var_excluded_origins = {}
        if hasattr(ctx, "role_bound_keys") and isinstance(ctx.role_bound_keys, dict):
            ctx.role_bound_keys.clear()
        else:
            ctx.role_bound_keys = {}
        if hasattr(ctx, "setter_focus_keys") and isinstance(ctx.setter_focus_keys, dict):
            ctx.setter_focus_keys.clear()
        else:
            ctx.setter_focus_keys = {}
        if hasattr(ctx, "optic_frames") and isinstance(ctx.optic_frames, list):
            ctx.optic_frames.clear()
        else:
            ctx.optic_frames = []
        if hasattr(ctx, "container_key_qualifiers") and isinstance(ctx.container_key_qualifiers, dict):
            ctx.container_key_qualifiers.clear()
        else:
            ctx.container_key_qualifiers = {}
        if hasattr(ctx, "container_key_layouts") and isinstance(ctx.container_key_layouts, dict):
            ctx.container_key_layouts.clear()
        else:
            ctx.container_key_layouts = {}
        if hasattr(ctx, "getter_read_keys") and isinstance(ctx.getter_read_keys, set):
            ctx.getter_read_keys.clear()
        else:
            ctx.getter_read_keys = set()
        if hasattr(ctx, "var_port_map") and isinstance(ctx.var_port_map, dict):
            ctx.var_port_map.clear()
        else:
            ctx.var_port_map = {}
        if hasattr(ctx, "var_parents") and isinstance(ctx.var_parents, dict):
            ctx.var_parents.clear()
        if getattr(ctx, "active_carrier_var", None) not in getattr(ctx, "variables", {}):
            ctx.active_carrier_var = None
        if hasattr(ctx, "written_keys") and isinstance(ctx.written_keys, set):
            ctx.written_keys.clear()
        if hasattr(ctx, "consumed_literals"):
            try:
                ctx.consumed_literals.clear()
            except Exception:
                ctx.consumed_literals = set()
        if hasattr(ctx, "consumed_tokens") and isinstance(ctx.consumed_tokens, set):
            ctx.consumed_tokens.clear()
        if hasattr(ctx, "used_indices") and isinstance(ctx.used_indices, set):
            ctx.used_indices.clear()
        if hasattr(ctx, "consumed_members") and isinstance(ctx.consumed_members, set):
            ctx.consumed_members.clear()
        if hasattr(ctx, "binding_trace") and isinstance(ctx.binding_trace, list):
            ctx.binding_trace.clear()
        if hasattr(ctx, "superseded_vars") and isinstance(ctx.superseded_vars, set):
            ctx.superseded_vars.clear()
        if hasattr(ctx, "var_partitions") and isinstance(ctx.var_partitions, dict):
            ctx.var_partitions.clear()
        if hasattr(ctx, "partitioned_vars") and isinstance(ctx.partitioned_vars, dict):
            ctx.partitioned_vars.clear()
        if hasattr(ctx, "handle_dim_exclusions") and isinstance(ctx.handle_dim_exclusions, dict):
            ctx.handle_dim_exclusions.clear()
        if hasattr(ctx, "unresolved_ports") and isinstance(ctx.unresolved_ports, list):
            ctx.unresolved_ports.clear()
        else:
            ctx.unresolved_ports = []
        if hasattr(ctx, "typed_holes") and isinstance(ctx.typed_holes, list):
            ctx.typed_holes.clear()
        else:
            ctx.typed_holes = []
        ctx.unbindable_count = 0
        ctx.var_counter = 0
        if hasattr(ctx, "_var_counter"):
            ctx._var_counter = 0
        if hasattr(ctx, "_prompt") and ctx._prompt:
            ctx.ordered_literals = ctx._extract_universal_literals(ctx._prompt)
            if hasattr(ctx, "sync_llm_slots"):
                ctx.sync_llm_slots()
            ctx.literal_clause_map = ctx._build_literal_clause_map(ctx._prompt, ctx.ordered_literals)

        accumulated_sigma = Substitution()
        value_env: Dict[str, str] = {}
        ctx.value_env = value_env
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        var_counter = 0
        var_counter_ref = [0]
        producer_var: Optional[str] = None
        cell_out_vars: Dict[Any, str] = {}
        cell_port_vars: Dict[Tuple[Any, str], str] = {}

        # Clause decomposition of the prompt (connector-based fast path; shared with
        # the planner) — used solely for member-state affinity when projecting
        # heterogeneous products. Zero domain vocabulary.
        prompt_text = (
            getattr(ctx, "prompt", "")
            or getattr(ctx, "_prompt", "")
            or (getattr(self.context, "prompt", "") if hasattr(self, "context") and self.context else "")
            or (getattr(self.context, "_prompt", "") if hasattr(self, "context") and self.context else "")
            or ""
        )
        if prompt_text and cells:
            try:
                from .route_methods.base import RouteMethod
                RouteMethod.tag_cells_with_clause_indices(cells, prompt_text, orchestrator=getattr(self, "orchestrator", None), ctx=ctx)
            except Exception as e1:
                try:
                    from route_methods.base import RouteMethod
                    RouteMethod.tag_cells_with_clause_indices(cells, prompt_text, orchestrator=getattr(self, "orchestrator", None), ctx=ctx)
                except Exception as e2:
                    logger.warning("Failed to tag cells with clause indices: %s | %s", e1, e2)

        llm_pipeline_bindings: Dict[str, Dict[str, Any]] = {}
        if prompt_text and cells and allow_llm and (extract_llm_slots or run_pipeline_llm_synth):
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm = ModelManager.get_instance()
                if extract_llm_slots and mm.can_extract_slots() and not getattr(ctx, "_cell_slots_extracted", False):
                    self._extract_pipeline_slots_with_llm(cells, prompt_text, ctx)
                if run_pipeline_llm_synth and mm.can_synthesize():
                    llm_pipeline_bindings = self._synthesize_pipeline_bindings_with_llm(cells, prompt_text, ctx)
            except Exception as e:
                logger.debug(f"[UNIFICATION] LLM pipeline synthesis setup failed: {e}")
                llm_pipeline_bindings = {}

        clause_token_sets: List[Set[str]] = []
        if prompt_text:
            for cl in CellTokenizer.split_prompt_clauses(prompt_text):
                cl = cl.strip()
                if cl:
                    clause_token_sets.append(CellTokenizer.tokenize_prompt(cl))

        def _cell_clause_tokens(cell: Any) -> Set[str]:
            if not clause_token_sets:
                return set()
            c_toks = getattr(cell, "token_set", set())
            best_idx, best_ov = 0, -1
            for idx, cl_toks in enumerate(clause_token_sets):
                ov = len(cl_toks & c_toks)
                if ov > best_ov:
                    best_ov, best_idx = ov, idx
            return clause_token_sets[best_idx] if best_ov > 0 else set()

        _is_product = _is_product_sig

        def _member_candidates(v_sig: Any, v_name: str, port_sig: Any, cell_toks: Set[str], registry: Optional[Any] = None) -> List[Tuple[float, int, Substitution]]:
            """Product-elimination candidates: (score, member_index, substitution)."""
            raw = str(getattr(getattr(v_sig, "signature", None), "type_name", "") or "")
            if "[" not in raw or not raw.endswith("]"):
                return []
            if registry is None:
                registry = TypeRegistry.get_instance()
            constructor = raw.split("[", 1)[0].strip().lower()
            if not registry.is_product_constructor(constructor):
                return []
            inner = raw[raw.index("[") + 1 : -1]
            members = list(split_bracketed_args(inner))
            if len(members) < 2:
                return []

            consumed = getattr(ctx, "consumed_members", set())
            candidates: List[Tuple[Any, int, Substitution]] = []
            for i, m_str in enumerate(members):
                try:
                    member_term = TypeTerm.from_string(m_str)
                except Exception as e:
                    continue
                u = unify(member_term, port_sig.signature, accumulated_sigma)
                if u is None:
                    continue
                is_unconsumed = 1 if (v_name, i) not in consumed else 0
                state_part = m_str[m_str.index("[") + 1 : -1] if "[" in m_str and m_str.endswith("]") else ""
                token_overlap = 0
                if state_part and state_part.lower() != "any" and cell_toks:
                    st_toks = CellTokenizer.tokenize_identifier(state_part)
                    token_overlap = len(st_toks & cell_toks)
                rank = (is_unconsumed, token_overlap, -i)
                candidates.append((rank, i, u))
            return candidates

        # Static Pre-Unification Structural Macro Expansion (Phase 4 / T4.1)
        cells = self.wiring_stage.expand_macros(cells, ctx)
        ctx.pipeline_served_clauses = {
            getattr(_c_srv, "matched_clause_idx", None)
            for _c_srv in cells
            if not getattr(_c_srv, "_auto_inserted_bridge", False)
            and getattr(_c_srv, "matched_clause_idx", None) is not None
        }
        if hasattr(ctx, "plan_bipartite_literal_assignments"):
            ctx._bipartite_literal_plan = ctx.plan_bipartite_literal_assignments(cells, registry=registry)

        # Pre-scan declared optic setter keys introduced later without prior read reference
        _setter_clause_indices: Set[int] = {
            getattr(_c_scan, "matched_clause_idx", None)
            for _c_scan in cells
            if str(getattr(_c_scan, "optic_role", "") or "").strip().lower() == "setter"
            and not getattr(_c_scan, "_auto_inserted_bridge", False)
            and getattr(_c_scan, "matched_clause_idx", None) is not None
        }
        _seen_before_setter: Set[str] = set()
        _pending_setter_keys: Set[str] = set()
        for _c_scan in cells:
            _scan_or = str(getattr(_c_scan, "optic_role", "") or "").strip().lower()
            _scan_cl = getattr(_c_scan, "matched_clause_idx", None)
            _is_auto_set = bool(getattr(_c_scan, "_auto_inserted_bridge", False))
            _scan_lits = [
                str(l).strip("'\"").strip().lower()
                for l in (getattr(_c_scan, "clause_literals", None) or ())
                if str(l).strip() and not ExecutionContext._is_path_string(str(l))
            ]
            if _scan_or == "setter" and not _is_auto_set:
                for _sl in _scan_lits:
                    if _sl not in _seen_before_setter:
                        _pending_setter_keys.add(_sl)
            elif _scan_cl not in _setter_clause_indices:
                _seen_before_setter.update(_scan_lits)
        ctx._pending_setter_keys = _pending_setter_keys
        _cell_id_counts: Dict[str, int] = {}
        for _c_cnt in cells:
            _cell_id_counts[_c_cnt.cell_id] = _cell_id_counts.get(_c_cnt.cell_id, 0) + 1
        ctx._pipeline_cell_id_counts = _cell_id_counts
        ctx._pipeline_future_clauses_by_step = {
            _si: {
                getattr(_fc, "matched_clause_idx", None)
                for _fc in cells[_si + 1:]
                if getattr(_fc, "matched_clause_idx", None) is not None
            }
            for _si in range(len(cells))
        }

        # Process each cell in sequence
        for idx, cell in enumerate(cells):
            # Backward Rank Demand Propagation:
            # If a downstream cell demands a specific canonical rank from this cell's output,
            # attach that rank demand to this cell's primary input so input unification selects
            # an appropriate rank producer (e.g. 1D series instead of 2D matrix).
            if not _is_sink_cell(cell) and cell.primary_input and cell.primary_output:
                if cell.primary_output.canonical_rank is None:
                    req_rank = None
                    for down_cell in cells[idx + 1:]:
                        if _is_sink_cell(down_cell):
                            continue
                        for dp_name, dp_sig in down_cell.inputs.items():
                            dp_r = getattr(dp_sig, "canonical_rank", None)
                            if dp_r is not None:
                                if unify(cell.primary_output.signature, dp_sig.signature) is None:
                                    continue
                                out_abs = getattr(cell.primary_output.signature, "abstract_type", None)
                                in_abs = getattr(dp_sig.signature, "abstract_type", None)
                                if out_abs and in_abs and out_abs != in_abs:
                                    if not (registry.is_subtype(out_abs, in_abs) or registry.is_subtype(in_abs, out_abs)):
                                        continue
                                val_roles = (
                                    registry.get_roles_for_category("assigned_value")
                                    | registry.get_roles_for_category("projection")
                                    | set(registry.get_projection_tokens() or ())
                                    | registry.get_roles_for_category("feature")
                                    | registry.get_roles_for_category("target")
                                    | registry.get_roles_for_category("key")
                                )
                                if getattr(dp_sig, "binds", None) in val_roles or getattr(dp_sig, "port_role", None) in val_roles:
                                    req_rank = dp_r
                                    break
                                elif down_cell.primary_input and down_cell.primary_input.name == dp_name:
                                    req_rank = dp_r
                                    break
                        if req_rank is not None:
                            break
                    if req_rank is not None:
                        # Respect upstream producers on polymorphic cells: never overwrite
                        # input rank with a conflicting downstream rank when an upstream
                        # ancestor already produces a unifiable output at a different rank.
                        up_ranks = set()
                        for up_cell in cells[:idx]:
                            if _is_sink_cell(up_cell):
                                continue
                            for up_out in up_cell.outputs.values():
                                if unify(up_out.signature, cell.primary_input.signature) is not None:
                                    if is_edge_valid(up_out, cell.primary_input, producer_cell=up_cell, consumer_cell=cell):
                                        ur = getattr(up_out, "canonical_rank", None)
                                        if ur is not None:
                                            up_ranks.add(ur)
                        if not up_ranks or req_rank in up_ranks:
                            cell.primary_input.canonical_rank = req_rank
                            cell.primary_output.canonical_rank = req_rank

            ctx.current_cell_step_idx = idx
            ctx.current_cell_clause_idx = getattr(cell, "matched_clause_idx", None)
            cell_clause = ctx.current_cell_clause_idx
            # For-each replicas (multiplicity expansion): a replica RE-CONSUMES
            # its receiver from the environment's in-scope variables (e.g. the
            # source table) instead of the previous wire, and its reference
            # port binds the NEXT member of the identifier role group. It
            # consumes no incoming wire, exactly like a zero-ary constructor.
            is_replica = bool(getattr(cell, "replica_of", None))
            cell_bindings: Dict[str, str] = _TracedBindings(ctx, cell)
            current_out_var = None
            ctx.consumed_tokens.update(t.lower() for t in cell.token_set)

            def _is_zero_ary_generator(c: Cell) -> bool:
                if not c.outputs or getattr(c, "stage", None) == 3 or str(getattr(c, "node_role", "")).lower() in registry.get_terminal_roles_set():
                    return False
                if any(_lattice_is_path_port(p) for p in c.inputs.values()):
                    return False
                for p in c.inputs.values():
                    if p.required and p.default_value is None:
                        return False
                return True

            is_zero_ary = (
                getattr(cell, "node_type", "") == "constructor"
                or _is_zero_ary_generator(cell)
            )

            if not is_replica and not is_zero_ary and getattr(ctx, "optic_frames", None):
                flushed_v = self._flush_optic_frames_for_consumer(
                    consumer_cell=cell,
                    ctx=ctx,
                    registry=registry,
                    accumulated_sigma=accumulated_sigma,
                    pipeline_bindings=pipeline_bindings,
                    var_counter_ref=var_counter_ref,
                    cell_out_vars=cell_out_vars,
                    cell_port_vars=cell_port_vars,
                )
                var_counter = var_counter_ref[0]
                if flushed_v:
                    producer_var = flushed_v

            bound_producer = False
            is_formal_step = bool(getattr(cell, "formal_program_step", False))

            step_key = str(idx)
            step_llm_bindings = (
                llm_pipeline_bindings.get(step_key)
                or llm_pipeline_bindings.get(f"step_{idx}")
                or llm_pipeline_bindings.get(cell.cell_id)
                or {}
            )
            for p_name, p_sig in cell.inputs.items():
                if p_name in step_llm_bindings and p_name not in cell_bindings:
                    cand_val = step_llm_bindings[p_name]
                    if cand_val is not None:
                        cand_str = str(cand_val).strip()
                        if cand_str:
                            concrete_sig = substitute_generics(p_sig, accumulated_sigma)
                            try:
                                if _is_valid_grounded_llm_expr(cand_val, ctx):
                                    if cand_str in ctx.variables:
                                        v_sig, _ = ctx.variables[cand_str]
                                        if _shape_compatible(v_sig, concrete_sig):
                                            u_cand = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                                            if u_cand is not None:
                                                cell_bindings[p_name] = cand_str
                                                accumulated_sigma = u_cand
                                                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "")
                                                if p_role in (registry.get_roles_for_category("data") | registry.get_data_bearing_roles()):
                                                    bound_producer = True
                                    else:
                                        rec = resolve_typed_binding_record(
                                            cell=cell,
                                            port_name=p_name,
                                            bound_val=cand_str,
                                            port_sig=concrete_sig,
                                            var_signatures={v: s[0] if isinstance(s, tuple) else s for v, s in ctx.variables.items()},
                                            expected_sig=concrete_sig,
                                        )
                                        if rec and rec.resulting_signature:
                                            u_cand = unify(rec.resulting_signature, concrete_sig.signature, accumulated_sigma)
                                            if u_cand is not None:
                                                cell_bindings[p_name] = cand_str
                                                accumulated_sigma = u_cand
                                                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "")
                                                if p_role in (registry.get_roles_for_category("data") | registry.get_data_bearing_roles()):
                                                    bound_producer = True
                            except Exception as _exc:
                                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            # 1. Multi-Carrier DAG Scope Verification & Wire Resolution:
            # When idx > 0, verify that cell's inputs are satisfiable from the full
            # DAG ancestor frontier (available_cells = cells[:idx]).
            if idx > 0 and not is_zero_ary and not is_replica:
                def _defers_unpartitioned_wire(p_n: str, p_o: Any, v_n: str) -> bool:
                    v_parts = getattr(ctx, "var_partitions", None) or {}
                    if not v_parts or v_n in v_parts or p_o is None:
                        return False
                    d_p = _port_lead_sym_dim(p_o)
                    if not d_p:
                        return False
                    sup_vs = getattr(ctx, "superseded_vars", set()) or set()
                    for o_name, o_port in (getattr(cell, "inputs", {}) or {}).items():
                        if o_name == p_n or o_port is p_o or not getattr(o_port, "required", False):
                            continue
                        if o_name in cell_bindings:
                            continue
                        if _port_lead_sym_dim(o_port) != d_p:
                            continue
                        has_part_cand = False
                        has_unpart_cand = False
                        probe_bindings = {**cell_bindings, p_n: v_n}
                        for cand_v, (cand_sig, _) in getattr(ctx, "variables", {}).items():
                            if cand_v in sup_vs or _is_product(cand_sig):
                                continue
                            if not _shape_compatible(cand_sig, o_port, consumer_cell=cell):
                                continue
                            if unify(cand_sig.signature, o_port.signature, accumulated_sigma) is None:
                                continue
                            if cand_v in v_parts:
                                has_part_cand = True
                            else:
                                if _wire_rejection(cell, o_name, o_port, cand_v, ctx, registry, cell_bindings=probe_bindings) is None:
                                    has_unpart_cand = True
                                    break
                        if has_part_cand and not has_unpart_cand:
                            return True
                    return False

                # Honor planned_port_wires if provided by the planner DAG
                planned_wires = getattr(cell, "planned_port_wires", None)
                if planned_wires:
                    for p_name, wire_target in sorted(
                        planned_wires.items(),
                        key=lambda kv: (
                            0 if str(getattr(cell.inputs.get(kv[0]), "structure_kind", "") or "").lower() == "handle"
                            else (2 if getattr(cell.inputs.get(kv[0]), "complement_of_port", None) else 1)
                        ),
                    ):
                        if isinstance(wire_target, tuple) and len(wire_target) == 2:
                            prod_cell, out_port_name = wire_target
                        else:
                            prod_cell, out_port_name = wire_target, "output"
                        v_name = None
                        if prod_cell is not None:
                            p_cid = getattr(prod_cell, "cell_id", str(prod_cell))
                            v_name = (
                                cell_port_vars.get((id(prod_cell), out_port_name))
                                or cell_out_vars.get(id(prod_cell))
                                or cell_port_vars.get((p_cid, out_port_name))
                                or cell_out_vars.get(p_cid)
                            )
                        if not v_name and out_port_name and out_port_name in getattr(ctx, "variables", {}):
                            v_name = out_port_name
                        if v_name and p_name not in cell_bindings:
                            if v_name in getattr(ctx, "superseded_vars", set()):
                                continue
                            if _defers_unpartitioned_wire(p_name, cell.inputs.get(p_name), v_name):
                                ctx.trace_binding("planned_wire_deferred_partition", cell=cell.cell_id, port=p_name, var=v_name)
                                continue
                            _rej = _wire_rejection(cell, p_name, cell.inputs.get(p_name), v_name, ctx, registry, cell_bindings=cell_bindings)
                            if _rej is not None:
                                ctx.trace_binding(f"planned_wire_rejected_{_rej[0]}", cell=cell.cell_id, port=p_name, var=v_name, **{k: v for k, v in _rej[1].items() if k not in ("cell", "port", "var")})
                                continue
                            cell_bindings[p_name] = v_name
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(v_name)
                            bound_producer = True
                            if prod_cell is not None and hasattr(prod_cell, "outputs") and hasattr(cell, "inputs"):
                                out_port = prod_cell.outputs.get(out_port_name) or prod_cell.primary_output
                                in_port = cell.inputs.get(p_name) or cell.primary_input
                                if out_port and in_port:
                                    u_wire = unify(out_port.signature, in_port.signature, accumulated_sigma)
                                    if u_wire is not None:
                                        accumulated_sigma = u_wire

                valid_ancestors = [c for c in cells[:idx] if not _is_sink_cell(c)]
                scope_res = unify_cell_with_scope(cell, available_cells=valid_ancestors, sigma=accumulated_sigma, context=ctx)
                if scope_res is not None:
                    accumulated_sigma, dag_scope_bindings = scope_res
                    for p_name, (prod_cell, out_port_name) in sorted(
                        dag_scope_bindings.items(),
                        key=lambda kv: (
                            0 if str(getattr(cell.inputs.get(kv[0]), "structure_kind", "") or "").lower() == "handle"
                            else (2 if getattr(cell.inputs.get(kv[0]), "complement_of_port", None) else 1)
                        ),
                    ):
                        v_name = None
                        if prod_cell is not None:
                            v_name = (
                                cell_port_vars.get((id(prod_cell), out_port_name))
                                or cell_out_vars.get(id(prod_cell))
                                or cell_port_vars.get((prod_cell.cell_id, out_port_name))
                                or cell_out_vars.get(prod_cell.cell_id)
                            )
                        if not v_name and out_port_name and out_port_name in getattr(ctx, "variables", {}):
                            v_name = out_port_name
                        if v_name and p_name not in cell_bindings:
                            if v_name in getattr(ctx, "superseded_vars", set()):
                                continue
                            if _defers_unpartitioned_wire(p_name, cell.inputs.get(p_name), v_name):
                                ctx.trace_binding("dag_scope_deferred_partition", cell=cell.cell_id, port=p_name, var=v_name)
                                continue
                            # If `v_name` was already consumed (e.g. written into an optic container)
                            # and the active unconsumed container carries a key whose qualifiers
                            # satisfy this port's accepted states, defer to optic lifting from the container.
                            if (
                                not is_formal_step
                                and v_name in getattr(ctx, "consumed_vars", set())
                                and str(getattr(cell, "optic_role", "") or "").strip().lower() not in registry.get_roles_for_category("optic")
                            ):
                                _dag_act = getattr(ctx, "active_carrier_var", None)
                                _dag_kq = getattr(ctx, "container_key_qualifiers", None) or {}
                                _dag_p = cell.inputs.get(p_name)
                                if (
                                    _dag_act
                                    and _dag_act in ctx.variables
                                    and _dag_act not in getattr(ctx, "superseded_vars", set())
                                    and _dag_act not in getattr(ctx, "consumed_vars", set())
                                    and _dag_p is not None
                                    and any(
                                        kv == _dag_act and kq_s and _port_declares_acceptance(_dag_p, set(kq_s))
                                        for (kv, _), kq_s in _dag_kq.items()
                                    )
                                ):
                                    continue
                            # DAG-resolved wires used to bypass clause/provenance checks entirely
                            # (type-only unification). Apply the same gate every other binding
                            # site uses so a wire derived from a different column than the one this
                            # clause names is left unbound for the projection branches to resolve.
                            _rej = _wire_rejection(cell, p_name, cell.inputs.get(p_name), v_name, ctx, registry, cell_bindings=cell_bindings)
                            if _rej is not None:
                                ctx.trace_binding(f"dag_scope_rejected_{_rej[0]}", cell=cell.cell_id, port=p_name, var=v_name, **{k: v for k, v in _rej[1].items() if k not in ("cell", "port", "var")})
                                continue
                            cell_bindings[p_name] = v_name
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(v_name)
                            bound_producer = True
                else:
                    # Fallback backward search if whole-scope unification was not satisfied
                    transition_res = None
                    for k in range(idx - 1, -1, -1):
                        cand_producer = cells[k]
                        if _is_sink_cell(cand_producer):
                            continue
                        res = self.unify_transition(cand_producer, cell, accumulated_sigma, context=ctx)
                        if not res.is_bottom():
                            transition_res = res
                            break
                    if transition_res is not None and not transition_res.is_bottom():
                        assert isinstance(transition_res, Success)
                        accumulated_sigma = transition_res.sigma
                    else:
                        # Before failing, check if unsatisfied required ports have projective roles
                        # that can be satisfied from an active carrier in scope (e.g. feature_input, target_input)
                        req_ports = [p for p in cell.inputs.values() if p.required]
                        projective_roles = (
                            registry.get_roles_for_category("projection")
                            | set(registry.get_projection_tokens() or ())
                            | registry.get_roles_for_category("feature")
                            | registry.get_roles_for_category("target")
                            | registry.get_roles_for_category("key")
                        )
                        has_carrier = any(
                            registry.is_composite_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                            or registry.is_abstract_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                            for v_sig, _ in getattr(ctx, "variables", {}).values()
                        )
                        c_role = str(getattr(cell, "node_role", "") or "").lower()
                        c_type = str(getattr(cell, "node_type", "") or "").lower()
                        strict_roles = registry.get_roles_for_category("strict") | registry.get_roles_for_category("estimator")
                        is_model_cell = (
                            c_role in strict_roles
                            or c_type in strict_roles
                        )
                        has_proj = any(
                            (
                                getattr(p, "port_role", "") in projective_roles
                                or getattr(p, "binds", "") in registry.get_roles_for_category("key")
                                or bool(getattr(p, "complement_of_port", None))
                                or (is_model_cell and getattr(p, "derived_role", "") in projective_roles)
                            )
                            for p in req_ports
                        )
                        all_req_bound = all(p_name in cell_bindings for p_name, p in cell.inputs.items() if p.required)
                        has_no_req_inputs = not any(getattr(p, "required", False) for p in cell.inputs.values())
                        is_external_source = bool(req_ports) and all(
                            _lattice_is_path_port(p, registry) or registry.is_config_role(getattr(p, "port_role", ""))
                            for p in req_ports
                        )
                        if not (has_proj and has_carrier) and not (planned_wires and all_req_bound) and not has_no_req_inputs and not is_external_source:
                            lifted_any = False
                            while not is_zero_ary and not is_replica:
                                lift_step1 = self._try_lift_lens_for_cell(
                                    consumer_cell=cell,
                                    cell_bindings=cell_bindings,
                                    ctx=ctx,
                                    registry=registry,
                                    accumulated_sigma=accumulated_sigma,
                                    pipeline_bindings=pipeline_bindings,
                                    var_counter_ref=var_counter_ref,
                                    cell_out_vars=cell_out_vars,
                                    cell_port_vars=cell_port_vars,
                                )
                                if lift_step1 is None:
                                    break
                                _, accumulated_sigma = lift_step1
                                var_counter = var_counter_ref[0]
                                lifted_any = True
                            if not lifted_any:
                                return Failure(transition_res.reason if isinstance(transition_res, Failure) else f"Transition failed for cell {cell.cell_id} from ancestors")

            # 2. Multi-Port Monoidal Matching: Bind input ports across available wires using semantic roles
            def _is_matching_port(p_n: str, p_s: Any) -> bool:
                reg = TypeRegistry.get_instance()
                role = str(getattr(p_s, "port_role", None) or getattr(p_s, "role", None) or getattr(p_s, "derived_role", None) or "").lower()
                cfg_roles = reg.get_config_roles_set()
                if role in cfg_roles or str(getattr(p_s, "param_kind", "")).lower() in cfg_roles:
                    str_kind = str(getattr(p_s, "structure_kind", "") or "").lower()
                    c_rank = getattr(p_s, "canonical_rank", None) if getattr(p_s, "canonical_rank", None) is not None else getattr(p_s, "rank", None)
                    t_name = str(getattr(getattr(p_s, "signature", p_s), "type_name", "") or getattr(p_s, "type_name", "") or "").strip().lower()
                    is_scalar = (
                        c_rank == 0
                        or str_kind == "scalar"
                        or (reg is not None and reg.is_subtype(t_name, "scalar"))
                    )
                    if is_scalar:
                        has_handle_wire = False
                        for v_sig, _ in ctx.variables.values():
                            if _is_product(v_sig):
                                continue
                            v_sk = str(getattr(v_sig, "structure_kind", "") or "").strip().lower()
                            v_abs = str(getattr(getattr(v_sig, "signature", v_sig), "abstract_type", "") or getattr(v_sig, "abstract_type", "") or "").strip().lower()
                            if (v_sk == "handle" or v_abs == "handle") and _shape_compatible(v_sig, p_s, consumer_cell=cell):
                                if unify(getattr(v_sig, "signature", v_sig), p_s.signature, accumulated_sigma) is not None:
                                    has_handle_wire = True
                                    break
                        if not has_handle_wire:
                            return False
                        return True

                data_roles = (
                    reg.get_roles_for_category("data")
                    | reg.get_roles_for_category("projection")
                    | set(reg.get_projection_tokens() or ())
                    | reg.get_roles_for_category("feature")
                    | reg.get_roles_for_category("target")
                    | reg.get_roles_for_category("key")
                    | reg.get_roles_for_category("strict")
                    | reg.get_roles_for_category("estimator")
                    | reg.get_declared_role_carriers()
                    | reg.get_data_bearing_roles()
                )
                if role in data_roles:
                    return True

                out_abs = str(getattr(p_s, "abstract_type", "") or "").lower()
                if out_abs and (reg.is_composite_carrier(out_abs) or reg.is_abstract_carrier(out_abs)):
                    return True

                if getattr(cell, "primary_in", None) == p_n or (hasattr(cell, "primary_input") and getattr(cell.primary_input, "name", None) == p_n):
                    return True

                return False

            multi_ports = [(k, p) for k, p in cell.inputs.items() if _is_matching_port(k, p) and k not in cell_bindings]
            bound_producer = bool(bound_producer)

            if len(multi_ports) > 1 and len(ctx.variables) >= 2:
                reg = TypeRegistry.get_instance()
                # Deterministic O(P * V) role-based matching with zero itertools.permutations
                # Restricts mutually exclusive roles declared by domain trees from cross-binding,
                # preventing reversed argument bugs like fit(y, X).
                avail_vars = [v for v in ctx.variables.keys() if not _is_product(ctx.variables[v][0])]

                # Strict / mutually exclusive semantic roles outrank producer recency
                # (preventing fit(y, X)); generic carrier roles (data_input / data_output)
                # must NOT outrank dataflow producer recency across cross-library boundaries.
                strict_excl_roles = (
                    set(reg.get_roles_for_category("strict"))
                    | set(reg.get_roles_for_category("estimator"))
                    | set(reg.get_roles_for_category("key"))
                    | set(reg.get_roles_for_category("projection"))
                )
                for _grp in reg.get_exclusive_roles():
                    strict_excl_roles.update(str(g).strip().lower() for g in _grp if str(g).strip())

                candidates = []
                cfg_roles_m = reg.get_config_roles_set()

                for p_name, p in multi_ports:
                    p_role = getattr(p, "port_role", None) or getattr(p, "derived_role", "standard")
                    p_str_k = str(getattr(p, "structure_kind", "") or "").strip().lower()
                    p_c_rank = getattr(p, "canonical_rank", None) if getattr(p, "canonical_rank", None) is not None else getattr(p, "rank", None)
                    p_t_n = str(getattr(getattr(p, "signature", p), "type_name", "") or getattr(p, "type_name", "") or "").strip().lower()
                    p_is_scalar = (p_c_rank == 0 or p_str_k == "scalar" or reg.is_subtype(p_t_n, "scalar"))
                    p_is_cfg_scalar = p_is_scalar and (
                        str(p_role).strip().lower() in cfg_roles_m
                        or str(getattr(p, "param_kind", "") or "").strip().lower() in cfg_roles_m
                    )
                    p_toks = CellTokenizer.tokenize_identifier((p_name or "").lower())
                    p_desc = str(getattr(p, "doc", "") or getattr(p, "description", "") or "")
                    if p_desc:
                        p_toks.update(CellTokenizer.tokenize_identifier(p_desc))
                    p_sig_inner = getattr(p, "signature", p)
                    p_state = (getattr(p_sig_inner, "state", "") or "").lower()
                    # Declared state vocabulary participates in token overlap
                    # (state names are declared data).
                    if p_state:
                        p_toks.update(p_state.split("_"))

                    for v_name in avail_vars:
                        v_sig, _ = ctx.variables[v_name]
                        if p_is_cfg_scalar:
                            v_sk = str(getattr(v_sig, "structure_kind", "") or "").strip().lower()
                            v_abs = str(getattr(getattr(v_sig, "signature", v_sig), "abstract_type", "") or getattr(v_sig, "abstract_type", "") or "").strip().lower()
                            if v_sk != "handle" and v_abs != "handle":
                                continue
                        v_role = getattr(v_sig, "port_role", None) or getattr(v_sig, "derived_role", "standard")

                        # Preserve declared callable defaults (e.g. try/except fallback handler)
                        # instead of overwriting a functional_operator port with a generic data wire.
                        if (
                            str(p_role).lower() == "functional_operator"
                            and getattr(p, "default_value", None) is not None
                            and str(v_role).lower() != "functional_operator"
                        ):
                            continue

                        # Conflict-based role gating via declared mutually exclusive role pairs
                        if _roles_conflict(p_role, v_role):
                            continue

                        # Signature unification check
                        if not _shape_compatible(v_sig, p, producer_cell=ctx.var_sources.get(v_name), consumer_cell=cell):
                            continue
                        u_p = unify(v_sig.signature, p.signature, accumulated_sigma)
                        if u_p is None:
                            continue

                        # Compute scale-free lexicographic rank for (p, v_name)
                        not_superseded = 1 if v_name not in getattr(ctx, "superseded_vars", set()) else 0
                        p_role_low = str(p_role).strip().lower()
                        v_role_low = str(v_role).strip().lower()
                        role_tier = 1
                        if p_role_low == v_role_low and p_role_low != "standard":
                            role_tier = 3
                        elif p_role_low != "standard" and v_role_low != "standard" and reg.is_subtype(v_role_low, p_role_low):
                            role_tier = 2
                        spec_role_tier = (
                            role_tier
                            if (p_role_low in strict_excl_roles or v_role_low in strict_excl_roles)
                            else 1
                        )

                        # Exact state match priority
                        v_sig_inner = getattr(v_sig, "signature", v_sig)
                        v_state = (getattr(v_sig_inner, "state", "") or "").lower()
                        state_match = 1 if (p_state and v_state and p_state == v_state) else 0

                        is_prod = 1 if (producer_var is not None and v_name == producer_var) else 0

                        # Token overlap
                        v_toks = set(CellTokenizer.tokenize_identifier((getattr(v_sig, "name", "") or "").lower()))
                        if v_name:
                            v_toks.update(CellTokenizer.tokenize_identifier(str(v_name).lower()))
                        if v_state:
                            v_toks.update(v_state.split("_"))
                        src_cell = getattr(ctx, "var_sources", {}).get(v_name)
                        if src_cell:
                            v_toks.update(src_cell.token_set)

                        # Generic port vocabulary is measured, not listed: tokens
                        # whose document frequency across the lattice exceeds one
                        # standard deviation above the mean cannot discriminate
                        # between candidate wires (same statistic the planner's
                        # LatticeVocabulary.frequent_tokens uses).
                        _GENERIC_PORT_STOP = self._generic_port_tokens()
                        overlap = len((p_toks & v_toks) - _GENERIC_PORT_STOP)

                        cell_prov_toks = ctx._current_cell_provenance_tokens(cell)
                        var_prov_toks = getattr(ctx, "var_clause_tokens", {}).get(v_name, set())
                        stop_prov = _GENERIC_PORT_STOP | set(reg.get_function_words())
                        clause_prov_overlap = len((cell_prov_toks & var_prov_toks) - stop_prov)

                        recency_idx = avail_vars.index(v_name)
                        no_comp_prio = 0 if getattr(p, "complement_of_port", None) else 1
                        is_handle_prio = 1 if str(getattr(p, "structure_kind", "") or "").lower() == "handle" else 0
                        rank = (no_comp_prio, is_handle_prio, not_superseded, spec_role_tier, is_prod, clause_prov_overlap, recency_idx, state_match, role_tier, overlap)
                        candidates.append((rank, p_name, p, v_name))

                # Deterministic greedy assignment with recency tie-breaking (active wire in frontier)
                candidates.sort(key=lambda item: item[0], reverse=True)
                assigned_ports = set()
                assigned_vars = set()
                assigned_origins = set()
                best_assign = {}
                test_sub = accumulated_sigma
                is_optic_setter = str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter"

                for score, p_name, p, v_name in candidates:
                    if p_name in assigned_ports or v_name in assigned_vars:
                        continue
                    comp_partner_n = getattr(p, "complement_of_port", None)
                    if comp_partner_n and comp_partner_n not in cell_bindings and comp_partner_n not in best_assign:
                        if not (getattr(ctx, "var_excluded_origins", None) or {}).get(v_name):
                            continue
                    v_origins = getattr(ctx, "var_origins", {}).get(v_name, set())
                    # Clause-scoped admissibility (column provenance + feature columns), shared with the DAG site.
                    _rej = _wire_rejection(cell, p_name, p, v_name, ctx, registry, cell_bindings={**cell_bindings, **best_assign})
                    if _rej is not None:
                        ctx.trace_binding(f"multiport_rejected_{_rej[0]}", cell=cell.cell_id, port=p_name, var=v_name, **{k: v for k, v in _rej[1].items() if k not in ("cell", "port", "var")})
                        continue
                    # Monoidal origin disjointness: avoid binding multiple symmetric join ports to the same origin wire
                    if not is_optic_setter and v_origins and (v_origins & assigned_origins):
                        continue
                    if v_name in getattr(ctx, "superseded_vars", set()):
                        has_active_alt = any(
                            alt_p == p_name
                            and alt_v not in assigned_vars
                            and alt_v not in getattr(ctx, "superseded_vars", set())
                            for _, alt_p, _, alt_v in candidates
                        )
                        if has_active_alt:
                            continue
                    v_sig, _ = ctx.variables[v_name]
                    u_curr = unify(v_sig.signature, p.signature, test_sub)
                    if u_curr is None:
                        continue
                    test_sub = u_curr
                    assigned_ports.add(p_name)
                    assigned_vars.add(v_name)
                    if v_origins:
                        assigned_origins.update(v_origins)
                    best_assign[p_name] = v_name

                if best_assign:
                    for p_name, v_name in best_assign.items():
                        cell_bindings[p_name] = v_name
                        if hasattr(ctx, "consumed_vars"):
                            ctx.consumed_vars.add(v_name)
                    accumulated_sigma = test_sub
                    if producer_var is not None and producer_var in assigned_vars:
                        bound_producer = True

            # If multi-port matching was not triggered or producer_var is not yet bound:
            # Replicas NEVER auto-bind the previous wire — their inputs come
            # from in-scope variables (the environment) or literals.
            col_mismatch = False
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica:
                prim_in = cell.primary_input
                target_port = prim_in if prim_in is not None else (cell.inputs.get("port_0") if hasattr(cell, "inputs") else None)
                _rej = _wire_rejection(cell, target_port.name if target_port else "port_0", target_port, producer_var, ctx, registry, cell_bindings=cell_bindings)
                col_mismatch = (_rej is not None)

                prim_in = cell.primary_input
                prim_comp_n = getattr(prim_in, "complement_of_port", None) if prim_in is not None else None
                prim_comp_pending = bool(
                    prim_comp_n
                    and prim_comp_n not in cell_bindings
                    and not (getattr(ctx, "var_excluded_origins", None) or {}).get(producer_var)
                )
                if prim_in is not None and prim_in.name in cell.inputs and not prim_comp_pending:
                    prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                    if prod_sig is not None and not _is_product(prod_sig):
                        if not col_mismatch and _shape_compatible(prod_sig, prim_in, producer_cell=ctx.var_sources.get(producer_var), consumer_cell=cell):
                            u_sub = unify(prod_sig.signature, prim_in.signature, accumulated_sigma)
                            if u_sub is not None:
                                cell_bindings[prim_in.name] = producer_var
                                if hasattr(ctx, "consumed_vars"):
                                    ctx.consumed_vars.add(producer_var)
                                accumulated_sigma = u_sub
                                bound_producer = True

            # If producer_var did not bind to primary input, check other compatible input ports (required ports take precedence)
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica:
                prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                if prod_sig is not None and not _is_product(prod_sig):
                    req_unbound = [(k, v) for k, v in cell.inputs.items() if v.required and k not in cell_bindings]
                    cand_ports = req_unbound if req_unbound else [(k, v) for k, v in cell.inputs.items() if k not in cell_bindings]
                    cand_ports = sorted(cand_ports, key=lambda kv: 1 if getattr(kv[1], "complement_of_port", None) else 0)
                    for p_name, p_sig in cand_ports:
                        comp_n = getattr(p_sig, "complement_of_port", None)
                        if comp_n and comp_n not in cell_bindings and not (getattr(ctx, "var_excluded_origins", None) or {}).get(producer_var):
                            continue
                        if not is_formal_step and str(getattr(cell, "optic_role", "") or "").strip().lower() not in registry.get_roles_for_category("optic"):
                            _p_r_t = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
                            _p_s_t = str(getattr(getattr(p_sig, "signature", p_sig), "state", "") or "").strip().lower()
                            _t_r_s = registry.get_roles_for_category("target")
                            _t_s_s = registry.get_states_by_property("is_target_identifier", True)
                            _is_tp = bool(
                                _p_r_t in _t_r_s
                                or _p_s_t in _t_s_s
                                or any(getattr(ip, "complement_of_port", None) == p_name for ip in cell.inputs.values())
                            )
                            if _is_tp:
                                _tgt_k_set = {str(tl).strip().lower() for tl in (getattr(ctx, "target_literals", set()) or set()) if str(tl).strip()}
                                if getattr(ctx, "target_key", None):
                                    _tgt_k_set.add(str(ctx.target_key).strip().lower())
                                _tgt_k_set.update(str(wk).strip().lower() for wk in (getattr(ctx, "written_keys", set()) or set()) if str(wk).strip())
                                _av_s2 = getattr(ctx, "active_carrier_var", None)
                                _av_o2 = {str(o).strip().lower() for o in (getattr(ctx, "var_origins", {}).get(_av_s2, set()) or set())} if _av_s2 else set()
                                _pv_o2 = {str(r).strip().lower() for r in _var_restricted_keys(producer_var, ctx)}
                                _unp_k2 = _tgt_k_set & _av_o2
                                if _unp_k2 and not (_pv_o2 & _unp_k2):
                                    continue
                        if _wire_rejection(cell, p_name, p_sig, producer_var, ctx, registry, cell_bindings=cell_bindings) is not None:
                            continue
                        if not _shape_compatible(prod_sig, p_sig, consumer_cell=cell):
                            continue
                        u_sub = unify(prod_sig.signature, p_sig.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[p_name] = producer_var
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(producer_var)
                            accumulated_sigma = u_sub
                            bound_producer = True
                            break

            # If producer_var did not bind (e.g. producer was an auxiliary/scalar output),
            # check the active carrier variable from lineage
            carrier_candidate = getattr(ctx, "active_carrier_var", None)
            if carrier_candidate is not None and carrier_candidate != producer_var and not bound_producer and not is_zero_ary and not is_replica:
                c_sig, _ = ctx.variables.get(carrier_candidate, (None, None))
                if c_sig is not None and not _is_product(c_sig):
                    prim_in = cell.primary_input
                    carr_comp_n = getattr(prim_in, "complement_of_port", None) if prim_in is not None else None
                    carr_comp_pending = bool(
                        carr_comp_n
                        and carr_comp_n not in cell_bindings
                        and not (getattr(ctx, "var_excluded_origins", None) or {}).get(carrier_candidate)
                    )
                    if (
                        prim_in is not None
                        and prim_in.name in cell.inputs
                        and not carr_comp_pending
                        and _wire_rejection(cell, prim_in.name, prim_in, carrier_candidate, ctx, registry, cell_bindings=cell_bindings) is None
                        and _shape_compatible(c_sig, prim_in, consumer_cell=cell)
                    ):
                        u_sub = unify(c_sig.signature, prim_in.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[prim_in.name] = carrier_candidate
                            accumulated_sigma = u_sub
                            bound_producer = True
                    if not bound_producer:
                        req_unbound = [(k, v) for k, v in cell.inputs.items() if v.required and k not in cell_bindings]
                        cand_ports = req_unbound if req_unbound else [(k, v) for k, v in cell.inputs.items() if k not in cell_bindings]
                        cand_ports = sorted(cand_ports, key=lambda kv: 1 if getattr(kv[1], "complement_of_port", None) else 0)
                        for p_name, p_sig in cand_ports:
                            comp_n = getattr(p_sig, "complement_of_port", None)
                            if comp_n and comp_n not in cell_bindings and not (getattr(ctx, "var_excluded_origins", None) or {}).get(carrier_candidate):
                                continue
                            if _wire_rejection(cell, p_name, p_sig, carrier_candidate, ctx, registry, cell_bindings=cell_bindings) is not None:
                                continue
                            if not _shape_compatible(c_sig, p_sig, consumer_cell=cell):
                                continue
                            u_sub = unify(c_sig.signature, p_sig.signature, accumulated_sigma)
                            if u_sub is not None:
                                cell_bindings[p_name] = carrier_candidate
                                accumulated_sigma = u_sub
                                bound_producer = True
                                break

            explicit_port_bindings = getattr(cell, "explicit_port_bindings", None)
            if explicit_port_bindings:
                for p_name, p_sig in cell.inputs.items():
                    if p_name in explicit_port_bindings and p_name not in cell_bindings:
                        exp_val = explicit_port_bindings[p_name]
                        concrete_sig = substitute_generics(p_sig, accumulated_sigma)
                        if exp_val is None:
                            formatted_exp = "None"
                        elif isinstance(exp_val, bool):
                            formatted_exp = "True" if exp_val else "False"
                        elif isinstance(exp_val, (int, float)):
                            formatted_exp = str(exp_val)
                        elif isinstance(exp_val, (list, tuple, dict, set)):
                            formatted_exp = repr(exp_val)
                        else:
                            formatted_exp = _format_literal_value(str(exp_val), concrete_sig)
                        cell_bindings[p_name] = formatted_exp
                        value_env[p_name] = formatted_exp
                        if _lattice_is_path_port(concrete_sig) and isinstance(exp_val, str):
                            ctx.target_literals.add(exp_val)

            # 3. Resolve auxiliary input ports (variable reuse / port sharing / parameters / literals).
            # REQUIRED ports are processed before optional ones so data ports claim
            # the prompt's shared literal channels (expressions, paths) first;
            # optional configuration knobs never steal them.
            cell_toks_for_projection = _cell_clause_tokens(cell)
            _pre_step_slots = (
                getattr(ctx, "step_llm_slots", {}).get(idx, {})
                if hasattr(ctx, "step_llm_slots") and isinstance(ctx.step_llm_slots, dict)
                else {}
            )
            _pre_cell_slots = (
                ctx.llm_slots.get(cell.cell_id, {})
                if _cell_id_counts.get(cell.cell_id, 0) <= 1 and hasattr(ctx, "llm_slots") and isinstance(ctx.llm_slots, dict)
                else {}
            )
            if not isinstance(_pre_step_slots, dict):
                _pre_step_slots = {}
            if not isinstance(_pre_cell_slots, dict):
                _pre_cell_slots = {}
            for p_name, p_sig in sorted(
                cell.inputs.items(),
                key=lambda kv: (
                    not kv[1].required,
                    0 if (kv[0] in _pre_step_slots or kv[0] in _pre_cell_slots) else 1,
                    1 if getattr(kv[1], "complement_of_port", None) else 0,
                ),
            ):
                if p_name in cell_bindings:
                    continue  # Already bound

                # Substitute generics if type variable in p_sig
                concrete_sig = substitute_generics(p_sig, accumulated_sigma)
                is_id_port, is_id_param_list = _classify_template_identifier_port(cell, p_name)

                # Formal Program_0 steps do not allow fuzzy raw-prompt literal guessing
                # for omitted non-carrier ports; optional ports use declared defaults only.
                if is_formal_step and not p_sig.required:
                    is_opt_carrier = (
                        getattr(p_sig, "port_role", None) in TypeRegistry.get_instance().get_declared_role_carriers()
                        or getattr(p_sig, "derived_role", "standard") in TypeRegistry.get_instance().get_declared_role_carriers()
                        or registry.is_composite_carrier(str(getattr(concrete_sig, "type_name", "")).lower())
                        or registry.is_abstract_carrier(str(getattr(concrete_sig, "type_name", "")).lower())
                    )
                    if not is_opt_carrier:
                        if p_sig.default_value is not None and not get_active_language_backend().is_null_literal(str(p_sig.default_value)):
                            val = _format_literal_value(p_sig.default_value, concrete_sig)
                            cell_bindings[p_name] = val
                            value_env[p_name] = val
                        else:
                            cell_bindings[p_name] = None
                        continue

                # Optional data carriers check in-scope variables first
                _tn_lower = str(getattr(concrete_sig, "type_name", "")).lower()
                ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
                is_data_carrier = (
                    getattr(p_sig, "port_role", None) in ROLE_CARRIERS
                    or getattr(p_sig, "derived_role", "standard") in ROLE_CARRIERS
                    or registry.is_composite_carrier(_tn_lower)
                    or registry.is_abstract_carrier(_tn_lower)
                )
                if not p_sig.required and is_data_carrier:
                    scoped_var = None
                    planned_parents = getattr(cell, "bound_parent_ids", None) or set()
                    candidate_vars = []
                    for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                        if _is_product(v_sig) or v_name in cell_bindings.values():
                            continue
                        if not _shape_compatible(v_sig, concrete_sig, consumer_cell=cell):
                            continue
                        if _wire_rejection(cell, p_name, p_sig, v_name, ctx, registry, cell_bindings=cell_bindings) is not None:
                            continue
                        u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                        if u_v is not None:
                            prod_cell = ctx.var_sources.get(v_name)
                            is_immediate_pred = bool(idx > 0 and prod_cell is cells[idx - 1])
                            is_superseded = v_name in getattr(ctx, "superseded_vars", set())
                            is_planned_name = bool(prod_cell and prod_cell.cell_id in planned_parents)
                            is_planned_score = 2 if is_immediate_pred else (1 if (is_planned_name and not is_superseded) else 0)
                            not_superseded_score = 0 if is_superseded else 1
                            candidate_vars.append(((is_planned_score, not_superseded_score), v_name, u_v))
                    if candidate_vars:
                        candidate_vars.sort(key=lambda x: x[0], reverse=True)
                        _, scoped_var, accumulated_sigma = candidate_vars[0]
                    if scoped_var is not None:
                        cell_bindings[p_name] = scoped_var
                        continue


                # Check LLM slot filling / hyperparameters before heuristic fallback
                llm_slot_val = None
                step_slots = getattr(ctx, "step_llm_slots", {}).get(idx, {}) if hasattr(ctx, "step_llm_slots") and isinstance(ctx.step_llm_slots, dict) else {}
                if p_name in step_slots:
                    llm_slot_val = step_slots[p_name]
                if llm_slot_val is None and _cell_id_counts.get(cell.cell_id, 0) <= 1 and hasattr(ctx, "llm_slots") and isinstance(ctx.llm_slots, dict):
                    cell_slots = ctx.llm_slots.get(cell.cell_id, {})
                    if isinstance(cell_slots, dict) and p_name in cell_slots:
                        llm_slot_val = cell_slots[p_name]
                if llm_slot_val is None and hasattr(ctx, "hyperparameters") and isinstance(ctx.hyperparameters, dict):
                    if p_name in ctx.hyperparameters:
                        cand_val = ctx.hyperparameters[p_name]
                        if p_sig.required:
                            llm_slot_val = cand_val
                        else:
                            # Optional port: check token provenance before overriding declared default
                            prompt_str = str(getattr(ctx, "prompt", "") or prompt_text or "")
                            val_str = str(cand_val).strip("'\"")
                            clause_toks = {t.lower() for t in _cell_clause_tokens(cell)} if callable(locals().get("_cell_clause_tokens")) else set()
                            c_toks = {t.lower() for t in (getattr(cell, "token_set", None) or set())}
                            p_lower = p_name.lower()
                            v_lower = val_str.lower()

                            valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}

                            has_name_evidence = (
                                p_lower in c_toks
                                or p_lower in clause_toks
                                or (len(p_name) > 2 and p_lower in prompt_str.lower())
                            )

                            if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
                                has_val_evidence = (val_str in valid_nums) or (val_str in prompt_str)
                            else:
                                has_val_evidence = (
                                    v_lower in c_toks
                                    or v_lower in clause_toks
                                    or (len(val_str) > 2 and v_lower in prompt_str.lower())
                                )

                            if has_name_evidence and has_val_evidence:
                                llm_slot_val = cand_val

                if llm_slot_val is not None and hasattr(ctx, "is_llm_slot_grounded") and not ctx.is_llm_slot_grounded(llm_slot_val, concrete_sig, cell=cell):
                    llm_slot_val = None
                if llm_slot_val is not None and not _is_valid_slot_value(llm_slot_val, concrete_sig):
                    llm_slot_val = None

                if llm_slot_val is not None and not p_sig.required:
                    # Optional port: if numeric or numeric sequence, verify provenance in user prompt
                    if isinstance(llm_slot_val, (int, float)) and not isinstance(llm_slot_val, bool):
                        prompt_str = str(getattr(ctx, "prompt", "") or prompt_text or "")
                        valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}
                        val_str = str(int(llm_slot_val)) if int(llm_slot_val) == llm_slot_val else str(llm_slot_val)
                        if val_str not in valid_nums and val_str not in prompt_str:
                            llm_slot_val = None
                    elif isinstance(llm_slot_val, (list, tuple)) and llm_slot_val and all(
                        isinstance(el, (int, float)) and not isinstance(el, bool)
                        for el in llm_slot_val
                    ):
                        prompt_str = str(getattr(ctx, "prompt", "") or prompt_text or "")
                        valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}
                        el_strs = [str(int(el)) if int(el) == el else str(el) for el in llm_slot_val]
                        if not any(es in valid_nums or es in prompt_str for es in el_strs):
                            llm_slot_val = None

                # Optional role-partitioned projection (e.g. join key_from_upstream_role) before omitting default
                if not p_sig.required and (
                    getattr(p_sig, "key_from_upstream_role", None)
                    or getattr(concrete_sig, "key_from_upstream_role", None)
                    or getattr(p_sig, "complement_of_port", None)
                    or getattr(concrete_sig, "complement_of_port", None)
                ):
                    opt_proj = self._resolve_role_partitioned_projection(
                        cell=cell,
                        p_name=p_name,
                        p_sig=p_sig,
                        concrete_sig=concrete_sig,
                        cell_bindings=cell_bindings,
                        ctx=ctx,
                        accumulated_sigma=accumulated_sigma,
                        registry=registry,
                        pipeline_bindings=pipeline_bindings,
                        var_counter_ref=var_counter_ref,
                        cell_out_vars=cell_out_vars,
                        cell_port_vars=cell_port_vars,
                    )
                    if opt_proj is not None:
                        opt_val, accumulated_sigma = opt_proj
                        var_counter = var_counter_ref[0]
                        cell_bindings[p_name] = opt_val
                        if isinstance(opt_val, str) and opt_val in getattr(ctx, "variables", {}):
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(opt_val)
                        else:
                            value_env[p_name] = opt_val
                        continue

                # Optional parameters with declared default: use prompt literal or omit/default
                if not p_sig.required and p_sig.default_value is not None:
                    is_macro_block_slot = _is_macro_block_slot(cell, p_name)
                    if not is_macro_block_slot and llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        _consume_matching_literals(ctx, llm_slot_val)
                        cell_bindings[p_name] = formatted_val
                        value_env[p_name] = formatted_val
                        continue

                    _bound_excl: Set[str] = set()
                    _v_excl_map = getattr(ctx, "var_excluded_origins", None) or {}
                    _h_excl_map = getattr(ctx, "handle_dim_exclusions", None) or {}
                    for _bv in cell_bindings.values():
                        if isinstance(_bv, str):
                            _bound_excl.update(str(k) for k in (_v_excl_map.get(_bv, ()) or ()))
                            for (_hv, _), (_he, _) in _h_excl_map.items():
                                if _hv == _bv:
                                    _bound_excl.update(str(k) for k in (_he or ()))
                    resolved_literal = (
                        None
                        if is_macro_block_slot
                        else ctx.resolve_literal_for_port(
                            concrete_sig,
                            cell_stage=cell.stage,
                            cell_inputs=cell.inputs,
                            cell_tokens=getattr(cell, "token_set", set()),
                            cell=cell,
                            excluded_keys=_bound_excl or None,
                        )
                    )
                    if resolved_literal is not None:
                        _is_id_dl, _is_param_dl = _classify_template_identifier_port(cell, p_name)
                        if _is_id_dl:
                            _norm_dl = _normalize_identifier_port_binding(resolved_literal, is_param_list=_is_param_dl)
                            if _norm_dl is not None:
                                resolved_literal = _norm_dl
                            else:
                                _def_norm = _normalize_identifier_port_binding(p_sig.default_value, is_param_list=_is_param_dl)
                                resolved_literal = _def_norm if _def_norm is not None else _unique_var(cell, p_name)
                        cell_bindings[p_name] = resolved_literal
                        value_env[p_name] = resolved_literal
                    elif get_active_language_backend().is_null_literal(str(p_sig.default_value)):
                        # Port default is None: omit from the emitted call
                        cell_bindings[p_name] = None
                    else:
                        val = str(p_sig.default_value)
                        c_tname = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        c_is_str = (
                            _is_string_carrier(p_sig, concrete_sig)
                            and not is_macro_block_slot
                            and not _classify_template_identifier_port(cell, p_name)[0]
                        )
                        if c_is_str and not (val.startswith(("'", '"')) or get_active_language_backend().is_null_or_bool_literal(val, case_insensitive=False)):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        value_env[p_name] = val
                    continue

                # Optional parameters WITHOUT a default are omitted from the
                # emitted call unless the prompt itself supplies a literal through
                # the type-affinity channels (a destination path, a numeric or
                # boolean qualifier).
                if not p_sig.required:
                    if llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        _consume_matching_literals(ctx, llm_slot_val)
                        cell_bindings[p_name] = formatted_val
                        value_env[p_name] = formatted_val
                        continue

                    _lit = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()), cell=cell)
                    if _lit is not None:
                        cell_bindings[p_name] = _lit
                        value_env[p_name] = _lit
                    else:
                        cell_bindings[p_name] = None
                    continue

                # A0. Target / complement-partner optic lifting:
                # If this required port is a supervised target or complement partner (e.g. X has complement_of_port=y)
                # and the active composite container holds a target/setter-written key that is NOT yet isolated
                # in any unconsumed compatible carrier, project that key (with carrier/qualifier bridges) first.
                if p_sig.required and not is_formal_step and str(getattr(cell, "optic_role", "") or "").strip().lower() not in registry.get_roles_for_category("optic"):
                    _p_role_tgt = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
                    _p_state_tgt = str(getattr(concrete_sig, "state", "") or "").strip().lower()
                    _tgt_roles_s = registry.get_roles_for_category("target")
                    _tgt_states_s = registry.get_states_by_property("is_target_identifier", True)
                    _is_tgt_or_partner = bool(
                        _p_role_tgt in _tgt_roles_s
                        or _p_state_tgt in _tgt_states_s
                        or any(getattr(ip, "complement_of_port", None) == p_name for ip in cell.inputs.values())
                    )
                    if _is_tgt_or_partner:
                        _all_tgt_lits = {str(tl).strip().lower() for tl in (getattr(ctx, "target_literals", set()) or set()) if str(tl).strip()}
                        if getattr(ctx, "target_key", None):
                            _all_tgt_lits.add(str(ctx.target_key).strip().lower())
                        _all_tgt_lits.update(str(wk).strip().lower() for wk in (getattr(ctx, "written_keys", set()) or set()) if str(wk).strip())
                        _act_v_t = getattr(ctx, "active_carrier_var", None)
                        _act_origs_t = {str(o).strip().lower() for o in (getattr(ctx, "var_origins", {}).get(_act_v_t, set()) or set())} if _act_v_t else set()
                        _unproj_tgt_keys = _all_tgt_lits & _act_origs_t
                        if _unproj_tgt_keys:
                            _has_isolated_match = any(
                                vn not in getattr(ctx, "consumed_vars", set())
                                and not _is_product(vsig)
                                and bool({str(r).strip().lower() for r in _var_restricted_keys(vn, ctx)} & _unproj_tgt_keys)
                                and _shape_compatible(vsig, concrete_sig, producer_cell=ctx.var_sources.get(vn), consumer_cell=cell)
                                and unify(vsig.signature, concrete_sig.signature, accumulated_sigma) is not None
                                for vn, (vsig, _) in ctx.variables.items()
                            )
                            if not _has_isolated_match:
                                _tgt_lift = self._try_lift_lens_for_cell(
                                    consumer_cell=cell,
                                    cell_bindings=cell_bindings,
                                    ctx=ctx,
                                    registry=registry,
                                    accumulated_sigma=accumulated_sigma,
                                    pipeline_bindings=pipeline_bindings,
                                    var_counter_ref=var_counter_ref,
                                    cell_out_vars=cell_out_vars,
                                    cell_port_vars=cell_port_vars,
                                    target_port_only=p_name,
                                )
                                if _tgt_lift is not None:
                                    _, accumulated_sigma = _tgt_lift
                                    var_counter = var_counter_ref[0]
                                    if p_name in cell_bindings:
                                        continue

                # A. Check in-scope variables first (environment / predecessor variables matching typestate)
                # One wire feeds ONE port per cell: a variable already bound to
                # this cell is skipped here. Silent duplication (the same
                # table into both concat inputs) is a planning artifact,
                # not a composition the prompt requested — a cell that genuinely
                # re-consumes a wire declares it (bound_slots / replicas).
                already_bound_vars = {
                    v for v in cell_bindings.values() if isinstance(v, str)
                }
                scoped_var = None
                planned_parents = getattr(cell, "bound_parent_ids", None) or set()
                candidate_vars = []
                p_role_in = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                val_roles = registry.get_roles_for_category("assigned_value")
                is_assigned_val = (
                    str(getattr(p_sig, "binds", "") or "").lower() in val_roles
                    or str(getattr(p_sig, "port_role", "") or "").lower() in val_roles
                    or str(p_name).lower() in val_roles
                )

                target_lits = getattr(ctx, "target_literals", set())
                var_origins_map = getattr(ctx, "var_origins", {})
                any_var_has_target = bool(target_lits and any(
                    any(str(orig).lower() in target_lits for orig in var_origins_map.get(vn, set()))
                    for vn in ctx.variables.keys()
                ))

                in_t_low = str(getattr(concrete_sig, "type_name", "") or "").strip().lower()
                in_t_base = in_t_low.split("[", 1)[0].strip()
                in_is_callable = (
                    str(p_role_in or "").strip().lower() == "functional_operator"
                    or registry.has_type_kind(in_t_low, "callable")
                    or registry.has_type_kind(in_t_base, "callable")
                    or registry.is_subtype(in_t_base, "callable")
                )

                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    # Heterogeneous products project member-wise; the whole container
                    # is never wired into a single data port.
                    if _is_product(v_sig):
                        continue
                    if v_name in already_bound_vars:
                        continue
                    v_t_low = str(getattr(v_sig.signature, "type_name", "") or "").strip().lower()
                    v_t_base = v_t_low.split("[", 1)[0].strip()
                    v_is_callable = (
                        registry.has_type_kind(v_t_low, "callable")
                        or registry.has_type_kind(v_t_base, "callable")
                        or registry.is_subtype(v_t_base, "callable")
                    )
                    if in_is_callable and not v_is_callable:
                        continue
                    if (not in_is_callable) and registry.is_type_variable(v_t_low) and (not registry.is_type_variable(in_t_low)) and (not registry.is_declared_top(in_t_low)):
                        continue
                    prod_cell_cand = ctx.var_sources.get(v_name) if hasattr(ctx, "var_sources") else None
                    if not _shape_compatible(v_sig, concrete_sig, producer_cell=prod_cell_cand, consumer_cell=cell):
                        continue

                    # Role-gating in scope resolution: conflict-based role rules
                    v_role = getattr(v_sig, "port_role", None) or getattr(v_sig, "derived_role", "")
                    if _roles_conflict(p_role_in, v_role, reg=registry):
                        continue

                    # Admissibility gating: provenance compatibility and single-key setter multi-fiber rejection
                    _rej = _wire_rejection(cell, p_name, p_sig, v_name, ctx, registry, cell_bindings=cell_bindings)
                    if _rej is not None:
                        ctx.trace_binding(f"in_scope_rejected_{_rej[0]}", cell=cell.cell_id, port=p_name, var=v_name, **{k: v for k, v in _rej[1].items() if k not in ("cell", "port", "var")})
                        continue

                    u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                    if u_v is not None:
                        prod_cell = ctx.var_sources.get(v_name)
                        is_immediate_pred = bool(idx > 0 and prod_cell is cells[idx - 1])
                        is_superseded = v_name in getattr(ctx, "superseded_vars", set())
                        is_planned_name = bool(prod_cell and prod_cell.cell_id in planned_parents)
                        is_planned_score = 2 if is_immediate_pred else (1 if (is_planned_name and not is_superseded) else 0)
                        not_superseded_score = 0 if is_superseded else 1

                        c_clause = getattr(cell, "matched_clause_idx", None)
                        prod_clause = getattr(prod_cell, "matched_clause_idx", None) if prod_cell else None

                        clause_causality = 0.0
                        if c_clause is not None and prod_clause is not None:
                            if prod_clause == c_clause:
                                clause_causality = 2.0
                            elif prod_clause < c_clause:
                                dist = c_clause - prod_clause
                                clause_causality = 1.0 / (1.0 + dist)
                            else:
                                clause_causality = -1.0

                        recency_idx = list(ctx.variables.keys()).index(v_name) if v_name in ctx.variables else 0
                        var_origins = getattr(ctx, "var_origins", {}).get(v_name, set())
                        target_lits = getattr(ctx, "target_literals", set())
                        has_target_origin = any(str(orig).lower() in target_lits for orig in var_origins)
                        port_state = str(getattr(concrete_sig, "state", "") or "").lower()
                        port_role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").lower()
                        reg = TypeRegistry.get_instance()
                        target_roles = reg.get_roles_for_category("target")
                        sink_roles = reg.get_terminal_roles_set() | reg.get_roles_for_category("destination")
                        target_states = (
                            reg.get_states_by_property("is_target_identifier", True)
                            | set(reg.get_terminal_states() or ())
                            | reg.get_states_by_property("is_terminal", True)
                            | reg.get_states_by_property("is_materialized", True)
                        )
                        p_role_toks = set(CellTokenizer.tokenize_identifier(port_role)) if port_role else set()
                        p_state_toks = set(CellTokenizer.tokenize_identifier(port_state)) if port_state else set()
                        is_target_port = bool(
                            port_role in target_roles
                            or port_role in sink_roles
                            or port_state in target_states
                            or (p_role_toks & (target_roles | sink_roles))
                            or (p_state_toks & target_states)
                        )

                        target_affinity = 0.0
                        if has_target_origin:
                            target_affinity = 1.0 if is_target_port else -1.0
                        elif is_target_port:
                            target_affinity = -0.5 if any_var_has_target else 0.0

                        is_unconsumed = 1 if v_name not in getattr(ctx, "consumed_vars", set()) else 0
                        causal_dir = 1 if clause_causality >= 0.0 else -1
                        cell_prov_toks = ctx._current_cell_provenance_tokens(cell)
                        var_prov_toks = getattr(ctx, "var_clause_tokens", {}).get(v_name, set())
                        stop_prov = self._generic_port_tokens() | set(reg.get_function_words())
                        clause_prov_overlap = len((cell_prov_toks & var_prov_toks) - stop_prov)

                        prio = (is_planned_score, not_superseded_score, target_affinity, is_unconsumed, causal_dir, clause_prov_overlap, clause_causality, recency_idx)
                        candidate_vars.append((prio, v_name, u_v))

                if candidate_vars:
                    # D6 Fix: Planned parent variables & clause adjacency take absolute priority over arbitrary recency
                    candidate_vars.sort(key=lambda x: x[0], reverse=True)
                    ctx.trace_binding(
                        "scoped_candidates", cell=cell.cell_id, port=p_name, clause=cell_clause,
                        candidates=[(v, str(pr), sorted(getattr(ctx, "var_origins", {}).get(v, set()))) for pr, v, _ in candidate_vars[:4]])
                    has_unconsumed_match = bool(candidate_vars[0][0][3] > 0)
                    if (
                        not has_unconsumed_match
                        and p_sig.required
                        and not is_formal_step
                        and str(getattr(cell, "optic_role", "") or "").strip().lower() not in registry.get_roles_for_category("optic")
                    ):
                        _act_v_c = getattr(ctx, "active_carrier_var", None)
                        _c_kq_m = getattr(ctx, "container_key_qualifiers", None) or {}
                        _has_qual_lift = bool(
                            _act_v_c
                            and _act_v_c in ctx.variables
                            and _act_v_c not in getattr(ctx, "superseded_vars", set())
                            and _act_v_c not in getattr(ctx, "consumed_vars", set())
                            and any(
                                kv == _act_v_c
                                and kq_s
                                and (_port_declares_acceptance(p_sig, set(kq_s)) or _port_declares_acceptance(concrete_sig, set(kq_s)))
                                for (kv, _), kq_s in _c_kq_m.items()
                            )
                        )
                        if _has_qual_lift:
                            _q_lift = self._try_lift_lens_for_cell(
                                consumer_cell=cell,
                                cell_bindings=cell_bindings,
                                ctx=ctx,
                                registry=registry,
                                accumulated_sigma=accumulated_sigma,
                                pipeline_bindings=pipeline_bindings,
                                var_counter_ref=var_counter_ref,
                                cell_out_vars=cell_out_vars,
                                cell_port_vars=cell_port_vars,
                                target_port_only=p_name,
                            )
                            if _q_lift is not None:
                                _, accumulated_sigma = _q_lift
                                var_counter = var_counter_ref[0]
                                if p_name in cell_bindings:
                                    continue
                    if candidate_vars[0][0][1] > 0:
                        _, scoped_var, accumulated_sigma = candidate_vars[0]

                if scoped_var is not None:
                    cell_bindings[p_name] = scoped_var
                    if hasattr(ctx, "consumed_vars"):
                        ctx.consumed_vars.add(scoped_var)
                    continue

                if scoped_var is None and candidate_vars and candidate_vars[0][0][1] >= 0:
                    _, scoped_var, accumulated_sigma = candidate_vars[0]
                    ctx.trace_binding("fallback_bound", cell=cell.cell_id, port=p_name, var=scoped_var)
                    cell_bindings[p_name] = scoped_var
                    if hasattr(ctx, "consumed_vars"):
                        ctx.consumed_vars.add(scoped_var)
                    continue

                # A2. Product-member projection: bind {port} to var[i] when the port's
                # declared signature unifies with a declared member carrier. Preference:
                # unconsumed members, then member-state affinity with the consuming clause.
                member_bound = False
                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    if not _is_product(v_sig):
                        continue
                    cands = _member_candidates(v_sig, v_name, concrete_sig, cell_toks_for_projection)
                    if not cands:
                        continue
                    cands.sort(key=lambda x: x[0], reverse=True)
                    _, best_i, best_u = cands[0]
                    cell_bindings[p_name] = f"{v_name}[{best_i}]"
                    accumulated_sigma = best_u
                    consumed_members = getattr(ctx, "consumed_members", None)
                    if consumed_members is not None:
                        consumed_members.add((v_name, best_i))
                    member_bound = True
                    break

                if member_bound:
                    continue

                # A3. Declarative role-partitioned projection / upstream role keys / optic setter keys
                # If an explicit step-scoped LLM slot exists for a setter's key_port, honor it first
                # (e.g., creating a new column 'Z' rather than overwriting the read column 'Y').
                if (
                    not is_formal_step
                    and str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter"
                    and _port_matches_declared_ref(cell, "key_port", p_sig)
                    and llm_slot_val is not None
                    and _is_valid_slot_value(llm_slot_val, concrete_sig)
                ):
                    formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                    _consume_matching_literals(ctx, llm_slot_val)
                    cell_bindings[p_name] = formatted_val
                    value_env[p_name] = formatted_val
                    continue

                has_rel_decl = bool(
                    getattr(p_sig, "key_from_upstream_role", None)
                    or getattr(concrete_sig, "key_from_upstream_role", None)
                    or getattr(p_sig, "complement_of_port", None)
                    or getattr(concrete_sig, "complement_of_port", None)
                    or (
                        str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter"
                        and _port_matches_declared_ref(cell, "key_port", p_sig)
                    )
                )
                if has_rel_decl:
                    proj_res = self._resolve_role_partitioned_projection(
                        cell=cell,
                        p_name=p_name,
                        p_sig=p_sig,
                        concrete_sig=concrete_sig,
                        cell_bindings=cell_bindings,
                        ctx=ctx,
                        accumulated_sigma=accumulated_sigma,
                        registry=registry,
                        pipeline_bindings=pipeline_bindings,
                        var_counter_ref=var_counter_ref,
                        cell_out_vars=cell_out_vars,
                        cell_port_vars=cell_port_vars,
                    )
                    if proj_res is not None:
                        proj_val, accumulated_sigma = proj_res
                        var_counter = var_counter_ref[0]
                        cell_bindings[p_name] = proj_val
                        if isinstance(proj_val, str) and proj_val in getattr(ctx, "variables", {}):
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(proj_val)
                        else:
                            value_env[p_name] = proj_val
                        continue

                # B. Check typestate-driven literal resolution from prompt or LLM slots
                if not is_formal_step and not _is_macro_block_slot(cell, p_name):
                    if is_id_port and llm_slot_val is not None:
                        norm_id = _normalize_identifier_port_binding(llm_slot_val, is_param_list=is_id_param_list)
                        if norm_id is not None:
                            _consume_matching_literals(ctx, llm_slot_val)
                            cell_bindings[p_name] = norm_id
                            value_env[p_name] = norm_id
                            continue
                    if (
                        isinstance(llm_slot_val, str)
                        and getattr(cell, "matched_clause_idx", None) is not None
                        and getattr(ctx, "ordered_literals", None)
                        and getattr(ctx, "literal_clause_map", None)
                    ):
                        _cand_l = llm_slot_val.strip().strip("'\"").lower()
                        _q_clauses = {
                            ctx.literal_clause_map.get(_qi)
                            for _qi, (_, _qk, _qv) in enumerate(ctx.ordered_literals)
                            if _qk == "quoted_str"
                            and str(_qv).strip().lower() == _cand_l
                            and ctx.literal_clause_map.get(_qi) is not None
                        }
                        if (
                            _q_clauses
                            and min(_q_clauses) > cell.matched_clause_idx
                            and any(getattr(_fc, "matched_clause_idx", None) in _q_clauses for _fc in cells[idx + 1:])
                        ):
                            llm_slot_val = None
                    if llm_slot_val is not None and _is_valid_slot_value(llm_slot_val, concrete_sig):
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        _consume_matching_literals(ctx, llm_slot_val)
                        cell_bindings[p_name] = formatted_val
                        value_env[p_name] = formatted_val
                        continue

                    _bound_excl: Set[str] = set()
                    _v_excl_map = getattr(ctx, "var_excluded_origins", None) or {}
                    _h_excl_map = getattr(ctx, "handle_dim_exclusions", None) or {}
                    for _bv in cell_bindings.values():
                        if isinstance(_bv, str):
                            _bound_excl.update(str(k) for k in (_v_excl_map.get(_bv, ()) or ()))
                            for (_hv, _), (_he, _) in _h_excl_map.items():
                                if _hv == _bv:
                                    _bound_excl.update(str(k) for k in (_he or ()))
                    resolved_literal = ctx.resolve_literal_for_port(
                        concrete_sig,
                        cell_stage=cell.stage,
                        cell_inputs=cell.inputs,
                        cell_tokens=getattr(cell, "token_set", set()),
                        cell=cell,
                        excluded_keys=_bound_excl or None,
                    )
                    if resolved_literal is not None:
                        if is_id_port:
                            norm_id = _normalize_identifier_port_binding(resolved_literal, is_param_list=is_id_param_list)
                            if norm_id is not None:
                                resolved_literal = norm_id
                        cell_bindings[p_name] = resolved_literal
                        value_env[p_name] = resolved_literal
                        continue

                # C. Check default value declared in tree
                if p_sig.default_value is not None:
                    val = str(p_sig.default_value)
                    _d_be = get_active_language_backend()
                    _val_stripped = val.strip("'\"")
                    is_key_req = p_sig.required and _is_key_or_complement_port(cell, p_sig, registry)
                    if is_key_req and (not _val_stripped or _d_be.is_empty_collection_literal(_val_stripped)):
                        # A required key/projection/complement port cannot be grounded by an empty default
                        pass
                    else:
                        c_tname = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        is_macro_block_slot = _is_macro_block_slot(cell, p_name) or is_id_port
                        c_is_str = _is_string_carrier(p_sig, concrete_sig) and not is_macro_block_slot
                        if is_id_port:
                            norm_id = _normalize_identifier_port_binding(val, is_param_list=is_id_param_list)
                            if norm_id is not None:
                                val = norm_id
                        elif c_is_str and not (val.startswith(("'", '"')) or _d_be.is_null_or_bool_literal(val, case_insensitive=False)):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        value_env[p_name] = val
                        continue

                # C2. Pointed monoidal default η_τ : 1 → τ (semi-closed literal parameters only)
                if not is_formal_step and p_sig.required and p_sig.default_value is None:
                    eta_val = _pointed_default(p_sig, cell)
                    if eta_val is not None:
                        cell_bindings[p_name] = eta_val
                        value_env[p_name] = eta_val
                        continue

                # D. Declared-domain enum grounding (reflection-driven, zero domain hardcodes)
                p_domain = getattr(concrete_sig, "domain", "") or ""
                if not is_formal_step and p_domain and getattr(ctx, "prompt", ""):
                    enum_val = ctx._resolve_enum_constant(p_domain, concrete_sig.state)
                    if enum_val is not None:
                        cell_bindings[p_name] = enum_val
                        value_env[p_name] = enum_val
                        continue

                # E. Unresolved port fallback: check grounded LLM pipeline bindings or targeted LLM synthesis
                if p_name not in cell_bindings or cell_bindings[p_name] in (None, UNRESOLVED_PORT):
                    if p_name in step_llm_bindings:
                        cand_val = step_llm_bindings[p_name]
                        if cand_val is not None:
                            cand_str = str(cand_val).strip()
                            if cand_str:
                                try:
                                    if _is_valid_grounded_llm_expr(cand_val, ctx, cell=cell, port_name=p_name):
                                        if is_id_port:
                                            norm_id = _normalize_identifier_port_binding(cand_val, is_param_list=is_id_param_list)
                                            if norm_id is not None:
                                                cell_bindings[p_name] = norm_id
                                                value_env[p_name] = norm_id
                                                continue
                                        if cand_str in ctx.variables:
                                            v_sig, _ = ctx.variables[cand_str]
                                            if _shape_compatible(v_sig, concrete_sig):
                                                u_cand = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                                                if u_cand is not None:
                                                    cell_bindings[p_name] = cand_str
                                                    accumulated_sigma = u_cand
                                                    continue
                                        else:
                                            rec = resolve_typed_binding_record(
                                                cell=cell,
                                                port_name=p_name,
                                                bound_val=cand_str,
                                                port_sig=concrete_sig,
                                                var_signatures={v: s[0] if isinstance(s, tuple) else s for v, s in ctx.variables.items()},
                                                expected_sig=concrete_sig,
                                            )
                                            if rec and rec.resulting_signature:
                                                u_cand = unify(rec.resulting_signature, concrete_sig.signature, accumulated_sigma)
                                                if u_cand is not None:
                                                    cell_bindings[p_name] = cand_str
                                                    accumulated_sigma = u_cand
                                                    continue
                                except Exception as _exc:
                                    logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

                # E.1.8 Dynamic Pre-Unification Qualifier Bridge Insertion
                # If a required port is still ungrounded, query the lattice for an eliminator bridge
                # that adapts an in-scope variable carrying conflicting qualifiers (e.g. complex -> real).
                if p_sig.required and p_name not in cell_bindings:
                    flat_p_rejs = (
                        _extract_rejected_qualifiers(p_sig, consumer_cell=cell)
                        | _extract_rejected_qualifiers(concrete_sig, consumer_cell=cell)
                    )

                    if flat_p_rejs:
                        bridged_bound = False
                        for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                            if _is_product(v_sig) or v_name in already_bound_vars:
                                continue
                            v_quals = _get_p_field(v_sig, "qualifiers", None) or getattr(getattr(v_sig, "signature", None), "qualifiers", None) or ()
                            flat_v_quals = set()
                            for q in v_quals:
                                if isinstance(q, (tuple, list, set)):
                                    flat_v_quals.update(str(x).lower() for x in q)
                                else:
                                    flat_v_quals.add(str(q).lower())

                            conflict = flat_v_quals & flat_p_rejs
                            if not conflict:
                                continue

                            b_cand = _find_bridge_for_qualifier_conflict(
                                getattr(self, "orchestrator", None),
                                v_sig,
                                concrete_sig,
                                conflict,
                                flat_p_rejs,
                                consumer_cell=cell,
                            )
                            if b_cand is not None:
                                b_cell = b_cand.clone() if hasattr(b_cand, "clone") else copy.copy(b_cand)
                                b_cell.matched_clause_idx = getattr(cell, "matched_clause_idx", None)
                                b_in = getattr(b_cell, "primary_input", None) or next(iter(b_cell.inputs.values()))
                                b_out = getattr(b_cell, "primary_output", None) or (next(iter(b_cell.outputs.values())) if getattr(b_cell, "outputs", None) else None)
                                b_in_name = b_in.name if b_in is not None else "x"
                                b_out_concrete = _instantiate_bridge_output_signature(b_cell, b_out, v_sig, b_in_name=b_in_name, v_name=v_name)
                                var_counter_ref[0] += 1
                                var_counter = var_counter_ref[0]
                                ctx.var_counter = var_counter
                                bridged_var = _semantic_var_name(b_cell, var_counter, port_name=b_out.name if b_out else "output_var", out_sig=b_out_concrete or b_out)
                                b_bindings = {b_in.name: v_name}
                                if b_out:
                                    b_bindings[b_out.name] = bridged_var
                                if "output_var" not in b_bindings:
                                    b_bindings["output_var"] = bridged_var
                                if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                                    ctx.var_port_map = {}
                                if b_out:
                                    ctx.var_port_map[(id(b_cell), b_out.name)] = bridged_var
                                ctx.var_port_map[(id(b_cell), "output_var")] = bridged_var

                                if b_out_concrete is not None and b_out is not None:
                                    b_cell.outputs[b_out.name] = b_out_concrete
                                    ctx.declare_variable(bridged_var, b_out_concrete, bridged_var, cell=b_cell)
                                    _update_variable_provenance(ctx, bridged_var, b_cell, b_bindings, out_port_name=b_out.name)
                                elif b_out:
                                    ctx.declare_variable(bridged_var, b_out, bridged_var, cell=b_cell)
                                    _update_variable_provenance(ctx, bridged_var, b_cell, b_bindings, out_port_name=b_out.name)
                                try:
                                    self.scope_stage.finalize_cell_scope(b_cell, b_bindings, ctx, accumulated_sigma)
                                except Exception:
                                    pass
                                pipeline_bindings.append((b_cell, b_bindings))
                                cell_bindings[p_name] = bridged_var
                                already_bound_vars.add(v_name)
                                already_bound_vars.add(bridged_var)
                                if hasattr(ctx, "consumed_vars"):
                                    ctx.consumed_vars.add(v_name)
                                    ctx.consumed_vars.add(bridged_var)
                                ctx.trace_binding("qualifier_bridge_inserted", cell=cell.cell_id, port=p_name, bridge=b_cell.cell_id, var=bridged_var, source_var=v_name)
                                bridged_bound = True
                                break
                        if bridged_bound:
                            continue

                # E.1.9 Profunctor Optic Lens Lifting before declaring unresolved required port
                if p_sig.required and p_name not in cell_bindings:
                    lift_res = self._try_lift_lens_for_cell(
                        consumer_cell=cell,
                        cell_bindings=cell_bindings,
                        ctx=ctx,
                        registry=registry,
                        accumulated_sigma=accumulated_sigma,
                        pipeline_bindings=pipeline_bindings,
                        var_counter_ref=var_counter_ref,
                        cell_out_vars=cell_out_vars,
                        cell_port_vars=cell_port_vars,
                    )
                    if lift_res is not None:
                        _, accumulated_sigma = lift_res
                        var_counter = var_counter_ref[0]
                        if p_name in cell_bindings:
                            continue

                # E.1.9.5 Macro sub-lattice bound slot satisfaction for HOF/block-body ports
                bound_slot_placeholder = _resolve_bound_slot_for_port(cell, p_name, concrete_sig)
                if bound_slot_placeholder is not None and p_name not in cell_bindings:
                    cell_bindings[p_name] = bound_slot_placeholder
                    value_env[p_name] = bound_slot_placeholder
                    continue

                # E.2 Deterministically resolve structural identifier / combinator identity ports before LLM fallback
                _is_combinator_cell = is_formal_step or (
                    str(getattr(cell, "node_role", "") or "").strip().lower() == "combinator"
                )
                if p_sig.required:
                    _is_id_p, _is_param_p = _classify_template_identifier_port(cell, p_name)
                    if _is_id_p:
                        if _is_param_p:
                            cell_bindings[p_name] = "x"
                        else:
                            cell_bindings[p_name] = _unique_var(cell, p_name)
                        value_env[p_name] = cell_bindings[p_name]
                        continue
                    _c_st_l = str(getattr(concrete_sig, "state", "") or "").strip().lower()
                    _c_tn_l = str(getattr(concrete_sig, "type_name", "") or "").strip().lower()
                    _p_rl_l = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
                    _c_tpl_s = str(getattr(cell, "code_template", "") or "")
                    if _c_st_l == "loop_condition" and _c_tn_l == "bool":
                        cell_bindings[p_name] = "False"
                        value_env[p_name] = "False"
                        continue
                    if _is_combinator_cell and _c_tn_l == "bool" and getattr(p_sig, "default_value", None) is None:
                        cell_bindings[p_name] = "True"
                        value_env[p_name] = "True"
                        continue
                    if (
                        (registry.is_hof_role(_p_rl_l) or _c_tn_l.startswith("callable"))
                        and "while " in _c_tpl_s
                        and "bool]" in _c_tn_l
                    ):
                        cell_bindings[p_name] = "lambda s: False"
                        value_env[p_name] = "lambda s: False"
                        continue
                    if _is_combinator_cell and (registry.is_hof_role(_p_rl_l) or _c_tn_l.startswith("callable")):
                        id_hof = "lambda *args: args[0] if len(args) == 1 else (args if args else None)"
                        cell_bindings[p_name] = id_hof
                        value_env[p_name] = id_hof
                        continue

                # E.1.10 Lazy per-port LLM fallback ONLY if deterministic unification + qualifier bridge + optic lifting all failed
                if run_pipeline_llm_synth and p_sig.required and p_name not in cell_bindings:
                    try:
                        from .inference import ModelManager
                        can_synth = ModelManager.get_instance().can_synthesize()
                    except Exception:
                        try:
                            from inference import ModelManager
                            can_synth = ModelManager.get_instance().can_synthesize()
                        except Exception:
                            can_synth = False
                    if can_synth:
                        fallback_expr = self._synthesize_single_port_binding_with_llm(cell, p_name, p_sig, concrete_sig, ctx)
                        if fallback_expr is not None:
                            cell_bindings[p_name] = fallback_expr
                            value_env[p_name] = fallback_expr
                            continue

                # E.2 Unresolved REQUIRED port: try schema-driven boundary I/O placeholder if permitted, else emit TypedHole & fail loudly
                if p_sig.required:
                    synth_io = resolve_synthetic_boundary_io(
                        p_sig,
                        concrete_sig=concrete_sig,
                        cell=cell,
                        registry=registry,
                        ctx=ctx,
                    )
                    if synth_io is not None:
                        cell_bindings[p_name] = synth_io
                        continue
                    ctx.unresolved_ports.append((cell.cell_id, p_name))
                    try:
                        from .dag_plan import TypedHole as _TypedHoleCls
                    except (ImportError, ValueError):
                        from dag_plan import TypedHole as _TypedHoleCls
                    if not hasattr(ctx, "typed_holes") or ctx.typed_holes is None:
                        ctx.typed_holes = []
                    ctx.typed_holes.append(
                        _TypedHoleCls(
                            hole_id=f"{cell.cell_id}.{p_name}",
                            task_description=f"Unresolved required port '{p_name}' on cell '{cell.cell_id}'",
                            input_ports={},
                            output_ports={p_name: getattr(concrete_sig, "signature", concrete_sig)},
                        )
                    )
                    ctx.unbindable_count += 1
                    cell_bindings[p_name] = UNRESOLVED_PORT
                else:
                    cell_bindings[p_name] = None

            # Optic getter provenance check: once `key_port` is resolved, ensure the bound
            # `container_port` carrier does not already exclude the requested key(s)
            # (e.g., an optic complement dropped that key downstream of the full carrier).
            if str(getattr(cell, "optic_role", "") or "").strip().lower() == "getter":
                g_cont_p, g_key_p, _ = self._optic_getter_ports(cell)
                if g_cont_p is not None and g_key_p is not None:
                    bound_cont_v = cell_bindings.get(g_cont_p.name)
                    bound_key_val = cell_bindings.get(g_key_p.name)
                    req_keys = _extract_keys_from_val(bound_key_val) if bound_key_val is not None else set()
                    cont_excl = set((getattr(ctx, "var_excluded_origins", None) or {}).get(bound_cont_v, ()) or ()) if isinstance(bound_cont_v, str) else set()
                    if isinstance(bound_cont_v, str) and bound_cont_v in ctx.variables and req_keys and (req_keys & cont_excl):
                        anc_order: List[str] = []
                        visited_g: Set[str] = {bound_cont_v}
                        q_g: List[str] = list((getattr(ctx, "var_parents", None) or {}).get(bound_cont_v, ()) or ())
                        while q_g:
                            cur_g = q_g.pop(0)
                            if cur_g in visited_g:
                                continue
                            visited_g.add(cur_g)
                            anc_order.append(cur_g)
                            for nxt_g in (getattr(ctx, "var_parents", None) or {}).get(cur_g, ()) or ():
                                if nxt_g not in visited_g:
                                    q_g.append(nxt_g)
                        direct_anc_set = set(anc_order)
                        for vn_g in reversed(list(ctx.variables.keys())):
                            if vn_g not in anc_order and vn_g != bound_cont_v:
                                anc_order.append(vn_g)

                        other_bound = {bv for bp, bv in cell_bindings.items() if bp != g_cont_p.name and isinstance(bv, str)}
                        sup_g = getattr(ctx, "superseded_vars", set()) or set()
                        anc_rank_map = {vn: idx_a for idx_a, vn in enumerate(anc_order)}
                        rebind_cands = []
                        concrete_cont_sig = substitute_generics(g_cont_p, accumulated_sigma)
                        for vn_g in anc_order:
                            if vn_g not in ctx.variables or vn_g in other_bound:
                                continue
                            v_sig_g, _ = ctx.variables[vn_g]
                            if _is_product(v_sig_g):
                                continue
                            v_excl_g = set((getattr(ctx, "var_excluded_origins", None) or {}).get(vn_g, ()) or ())
                            if req_keys & v_excl_g:
                                continue
                            if _wire_rejection(cell, g_cont_p.name, g_cont_p, vn_g, ctx, registry, cell_bindings=cell_bindings) is not None:
                                continue
                            if not _shape_compatible(v_sig_g, concrete_cont_sig, producer_cell=ctx.var_sources.get(vn_g), consumer_cell=cell):
                                continue
                            u_reb = unify(v_sig_g.signature, concrete_cont_sig.signature, accumulated_sigma)
                            if u_reb is not None:
                                is_direct_anc = 1 if vn_g in direct_anc_set else 0
                                not_sup = 1 if vn_g not in sup_g else 0
                                rebind_cands.append(((is_direct_anc, not_sup, -anc_rank_map[vn_g]), vn_g, u_reb))
                        if rebind_cands:
                            rebind_cands.sort(key=lambda x: x[0], reverse=True)
                            _, alt_cont_v, u_reb = rebind_cands[0]
                            cell_bindings[g_cont_p.name] = alt_cont_v
                            accumulated_sigma = u_reb
                            if hasattr(ctx, "consumed_vars"):
                                ctx.consumed_vars.add(alt_cont_v)
                            ctx.trace_binding(
                                "getter_rebound_pre_exclusion_carrier",
                                cell=cell.cell_id,
                                port=g_cont_p.name,
                                old_var=bound_cont_v,
                                new_var=alt_cont_v,
                                keys=sorted(req_keys),
                            )

            # Bound input variables for this cell, computed ONCE after all input
            # bindings are final and before any output branch runs (R6-1). Only
            # strings that are variables in scope count (matches the semantics the
            # pre-R5 refactor code had at both of its former definition sites).
            bound_in_vars = [v for v in cell_bindings.values() if isinstance(v, str) and v in ctx.variables]
            c_optic_role_pre = str(getattr(cell, "optic_role", "") or "").strip().lower()
            if c_optic_role_pre == "setter":
                # Setter focus qualifiers are tracked per-key on container_key_qualifiers
                # rather than tainting the entire composite container unconditionally.
                s_cont_pre, _, _, _ = self._optic_setter_ports(cell)
                cont_only_binds = {s_cont_pre.name: cell_bindings[s_cont_pre.name]} if (s_cont_pre and s_cont_pre.name in cell_bindings) else {}
                prop_quals = _collect_propagated_qualifiers(cell, cont_only_binds, {k: v[0] for k, v in ctx.variables.items()})
            else:
                prop_quals = _collect_propagated_qualifiers(cell, cell_bindings, {k: v[0] for k, v in ctx.variables.items()})
                # Propagate per-key qualifiers onto flat (non-composite) projections/conversions
                c_kq_pre = getattr(ctx, "container_key_qualifiers", None) or {}
                if c_kq_pre and c_optic_role_pre not in (registry.get_roles_for_category("optic") - {"setter"}):
                    out_p_pre = getattr(cell, "primary_output", None) or (next(iter(cell.outputs.values())) if getattr(cell, "outputs", None) else None)
                    out_tn_pre = str(getattr(getattr(out_p_pre, "signature", out_p_pre), "type_name", "") or "").lower()
                    if not (registry.is_composite_carrier(out_tn_pre) or registry.is_abstract_carrier(out_tn_pre)):
                        for bv_in in bound_in_vars:
                            for (kv_var, _), kq_set in c_kq_pre.items():
                                if kv_var == bv_in and kq_set:
                                    prop_quals |= set(kq_set)

            # Register output port(s) in context for future steps
            def _apply_cell_poly_prop(cell_obj: Any, out_port_sig: Any, bindings_map: Dict[str, Any], ctx_obj: Any, reg_obj: Any) -> Any:
                cand_pp = getattr(cell_obj, "polymorphic_propagation", None)
                if not isinstance(cand_pp, dict) or not (
                    cand_pp.get("preserves_semantics")
                    or cand_pp.get("preserves_layout")
                    or cand_pp.get("preserves_typestate")
                ):
                    return out_port_sig
                prim_in = getattr(cell_obj, "primary_input", None)
                src_pn = cand_pp.get("source_port") or (prim_in.name if prim_in is not None else None)
                src_var = bindings_map.get(src_pn) if src_pn else None
                if isinstance(src_var, str) and src_var in getattr(ctx_obj, "variables", {}):
                    src_entry = ctx_obj.variables[src_var]
                    src_sig_obj = src_entry[0] if isinstance(src_entry, tuple) else src_entry
                    return apply_polymorphic_propagation(out_port_sig, src_sig_obj, cand_pp, reg_obj)
                return out_port_sig

            if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                ctx.var_port_map = {}
            if len(cell.outputs) > 1:
                # Multi-output cell (e.g. train_test_split, load_wine, cv2.threshold, subplots)
                # Declare each individual output port in the execution context and bind in template
                first_out_var = None
                prim_out = cell.primary_output
                _cell_tpl = getattr(cell, "code_template", "") or ""
                if _is_sink_cell(cell):
                    producer_var = None
                    for out_name, out_sig in cell.outputs.items():
                        if f"{{{out_name}}}" in _cell_tpl or "{output_var}" in _cell_tpl:
                            var_counter_ref[0] += 1
                            ctx.var_counter = var_counter_ref[0]
                            _sink_v = _semantic_var_name(cell, var_counter_ref[0], port_name=out_name, out_sig=out_sig)
                            cell_bindings[out_name] = _sink_v
                            if producer_var is None:
                                producer_var = _sink_v
                        else:
                            cell_bindings[out_name] = None
                    if "{output_var}" in _cell_tpl and producer_var is not None and "output_var" not in cell_bindings:
                        cell_bindings["output_var"] = producer_var
                else:
                    prim_var = None
                    for idx_out, (out_name, out_sig) in enumerate(cell.outputs.items()):
                        var_counter_ref[0] += 1
                        var_counter = var_counter_ref[0]
                        ctx.var_counter = var_counter
                        out_var = _semantic_var_name(cell, var_counter, port_name=out_name, out_sig=out_sig)
                        if idx_out == 0:
                            first_out_var = out_var
                        cell_bindings[out_name] = out_var
                        cell_port_vars[(cell.cell_id, out_name)] = out_var
                        cell_port_vars[(id(cell), out_name)] = out_var
                        ctx.var_port_map[(id(cell), out_name)] = out_var

                        concrete_out = substitute_generics(out_sig, accumulated_sigma)
                        if prop_quals:
                            concrete_out = _enrich_port_signature_qualifiers(concrete_out, prop_quals)
                        concrete_out = _apply_cell_poly_prop(cell, concrete_out, cell_bindings, ctx, registry)
                        ctx.declare_variable(out_var, concrete_out, out_var, cell=cell)
                        _update_variable_provenance(ctx, out_var, cell, cell_bindings, out_port_name=out_name)
                        if prim_out and out_name == prim_out.name:
                            prim_var = out_var

                    producer_var = prim_var if prim_var is not None else (first_out_var or current_out_var)
                    if "output_var" not in cell.outputs:
                        cell_bindings["output_var"] = producer_var
                    cell_out_vars[cell.cell_id] = producer_var
                    cell_out_vars[id(cell)] = producer_var
                    cell_port_vars[(cell.cell_id, "output_var")] = producer_var
                    cell_port_vars[(id(cell), "output_var")] = producer_var
                    ctx.var_port_map[(id(cell), "output_var")] = producer_var

                    prim_sig_multi = getattr(prim_out, "signature", prim_out) if prim_out else None
                    out_t_multi = str(getattr(prim_sig_multi, "type_name", "") or "").lower()
                    out_abs_multi = str(getattr(prim_sig_multi, "abstract_type", "") or "").lower()
                    if registry.is_composite_carrier(out_t_multi) or registry.is_abstract_carrier(out_t_multi) or getattr(prim_sig_multi, "structure_kind", None) == "collection" or out_abs_multi == "collection" or getattr(ctx, "active_carrier_var", None) is None:
                        ctx.active_carrier_var = producer_var

                    # Record coproduct partitions and supersede partitioned unsplit inputs
                    # so downstream multi-argument cells (e.g. fit(X, y)) wire consistent
                    # partition branches instead of mixing pre-split and split carriers.
                    try:
                        from algebraic_effects import infer_cell_algebraic_effects
                    except (ImportError, ValueError):
                        from .algebraic_effects import infer_cell_algebraic_effects
                    _eff_multi = infer_cell_algebraic_effects(cell)
                    _part_outs = {
                        on: _eff_multi[f"out:{on}"].partition_index
                        for on in cell.outputs
                        if f"out:{on}" in _eff_multi and _eff_multi[f"out:{on}"].partition_index is not None
                    }
                    if len(set(_part_outs.values())) >= 2:
                        if not hasattr(ctx, "var_partitions") or ctx.var_partitions is None:
                            ctx.var_partitions = {}
                        if not hasattr(ctx, "partitioned_vars") or ctx.partitioned_vars is None:
                            ctx.partitioned_vars = {}
                        for on, pidx in _part_outs.items():
                            ov = cell_bindings.get(on)
                            if ov and ov in ctx.variables:
                                ctx.var_partitions[ov] = (id(cell), pidx, _port_lead_sym_dim(cell.outputs.get(on)))
                        for b_in_v in bound_in_vars:
                            ctx.partitioned_vars[b_in_v] = id(cell)
                            ctx.superseded_vars.add(b_in_v)
            elif len(cell.outputs) == 1:
                # Single output cell
                _cell_tpl_s = getattr(cell, "code_template", "") or ""
                if _is_sink_cell(cell):
                    producer_var = None
                    if cell.primary_output and cell.primary_output.name:
                        _p_name_s = cell.primary_output.name
                        if f"{{{_p_name_s}}}" in _cell_tpl_s or "{output_var}" in _cell_tpl_s:
                            var_counter_ref[0] += 1
                            ctx.var_counter = var_counter_ref[0]
                            _sink_v = _semantic_var_name(cell, var_counter_ref[0], port_name=_p_name_s, out_sig=cell.primary_output)
                            cell_bindings[_p_name_s] = _sink_v
                            if "{output_var}" in _cell_tpl_s and "output_var" not in cell.outputs:
                                cell_bindings["output_var"] = _sink_v
                        else:
                            cell_bindings[_p_name_s] = None
                else:
                    concrete_out = substitute_generics(cell.primary_output, accumulated_sigma)
                    if prop_quals:
                        concrete_out = _enrich_port_signature_qualifiers(concrete_out, prop_quals)
                    concrete_out = _apply_cell_poly_prop(cell, concrete_out, cell_bindings, ctx, registry)
                    # If the cell performs in-place mutation on a receiver, alias the output
                    # to the receiver: the receiver is the cell's declared receiver-role port,
                    # falling back to the primary input binding (declared-first, name-last).
                    if getattr(cell, "mutation_type", "pure") == "in_place":
                        receiver_var = None
                        rec_roles = registry.get_roles_for_category("receiver")
                        receiver_port = next(
                            (p for p in cell.inputs.values() if getattr(p, "port_role", None) in rec_roles),
                            None,
                        )
                        if receiver_port is not None:
                            receiver_var = cell_bindings.get(receiver_port.name)
                        if receiver_var is None and cell.primary_input is not None:
                            receiver_var = cell_bindings.get(cell.primary_input.name)
                        if receiver_var and receiver_var in ctx.variables:
                            current_out_var = receiver_var

                    if current_out_var is None:
                        var_counter_ref[0] += 1
                        var_counter = var_counter_ref[0]
                        ctx.var_counter = var_counter
                        current_out_var = _semantic_var_name(
                            cell,
                            var_counter,
                            port_name=(cell.primary_output.name if cell.primary_output else "output_var"),
                            out_sig=concrete_out,
                        )
                    cell_bindings["output_var"] = current_out_var

                    # Bind primary output port name if distinct from output_var
                    if cell.primary_output and cell.primary_output.name:
                        cell_bindings[cell.primary_output.name] = current_out_var
                        cell_port_vars[(cell.cell_id, cell.primary_output.name)] = current_out_var
                        cell_port_vars[(id(cell), cell.primary_output.name)] = current_out_var
                        ctx.var_port_map[(id(cell), cell.primary_output.name)] = current_out_var

                    if current_out_var is not None:
                        # Forward Rank Propagation:
                        # Preserve input rank/shape_contract when output omits shape_contract and either
                        # lacks canonical_rank or the primary input port itself is rank-polymorphic.
                        if cell.primary_input and _get_p_field(concrete_out, "shape_contract", None) is None:
                            bound_in_name = cell_bindings.get(cell.primary_input.name)
                            if bound_in_name and bound_in_name in ctx.variables:
                                in_sig_tuple = ctx.variables[bound_in_name]
                                in_port_sig = in_sig_tuple[0] if isinstance(in_sig_tuple, tuple) else in_sig_tuple
                                in_sc = _get_p_field(in_port_sig, "shape_contract", None)
                                in_ndim = _extract_ndim(in_sc)
                                in_r = in_ndim if in_ndim is not None else getattr(in_port_sig, "canonical_rank", None)
                                pi_poly = (
                                    _get_p_field(cell.primary_input, "shape_contract", None) is None
                                    and getattr(cell.primary_input, "rank", None) is None
                                )
                                if in_r is not None and (getattr(concrete_out, "canonical_rank", None) is None or pi_poly):
                                    concrete_out = copy.copy(concrete_out)
                                    concrete_out.rank = in_r
                                    if hasattr(concrete_out, "_cached_rank_cache"):
                                        concrete_out._cached_rank_cache = None
                                    if hasattr(concrete_out, "_cached_ndim"):
                                        concrete_out._cached_ndim = None
                                    if in_sc is not None:
                                        concrete_out.shape_contract = copy.deepcopy(in_sc) if isinstance(in_sc, dict) else in_sc
                                    else:
                                        concrete_out.shape_contract = {"ndim": in_r, "rank": in_r}

                        c_optic_role = str(getattr(cell, "optic_role", "") or "").strip().lower()
                        if c_optic_role == "getter":
                            g_cont_q, g_key_q, _ = self._optic_getter_ports(cell)
                            g_cont_vq = cell_bindings.get(g_cont_q.name) if g_cont_q else None
                            g_key_eq = cell_bindings.get(g_key_q.name) if g_key_q else None
                            g_keys_q = _extract_keys_from_val(g_key_eq) if g_key_eq else set()
                            if g_keys_q:
                                if not hasattr(ctx, "getter_read_keys") or ctx.getter_read_keys is None:
                                    ctx.getter_read_keys = set()
                                ctx.getter_read_keys.update(k.lower() for k in g_keys_q if str(k).strip())
                            if isinstance(g_cont_vq, str) and g_keys_q:
                                kq_map = getattr(ctx, "container_key_qualifiers", None) or {}
                                extra_kq: Set[str] = set()
                                for gk in g_keys_q:
                                    extra_kq.update(kq_map.get((g_cont_vq, str(gk).strip().lower()), set()) or set())
                                if extra_kq:
                                    concrete_out = _enrich_port_signature_qualifiers(concrete_out, extra_kq)
                                matched_lay = _lookup_consistent_key_layout(ctx, g_cont_vq, g_keys_q)
                                if matched_lay is not None:
                                    concrete_out = _enrich_port_signature_layout(concrete_out, matched_lay, registry)

                        ctx.declare_variable(current_out_var, concrete_out, current_out_var, cell=cell)
                        producer_var = current_out_var
                        cell_out_vars[cell.cell_id] = current_out_var
                        cell_out_vars[id(cell)] = current_out_var
                        cell_port_vars[(cell.cell_id, "output_var")] = current_out_var
                        cell_port_vars[(id(cell), "output_var")] = current_out_var
                        ctx.var_port_map[(id(cell), "output_var")] = current_out_var

                        concrete_sig = getattr(concrete_out, "signature", concrete_out)
                        out_t = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        out_abs = str(getattr(concrete_sig, "abstract_type", "") or "").lower()
                        is_carrier = (
                            registry.is_composite_carrier(out_t)
                            or registry.is_abstract_carrier(out_t)
                            or registry.is_composite_carrier(out_abs)
                            or getattr(concrete_sig, "structure_kind", None) == "collection"
                            or out_abs == "collection"
                        )
                        if is_carrier or getattr(ctx, "active_carrier_var", None) is None:
                            ctx.active_carrier_var = current_out_var

                        # Provenance and monoidal DAG branch tracking
                        ctx.var_parents.setdefault(current_out_var, set())
                        _update_variable_provenance(
                            ctx,
                            current_out_var,
                            cell,
                            cell_bindings,
                            out_port_name=(cell.primary_output.name if cell.primary_output else None),
                        )

                        # Profunctor Optic Frame Lifecycle
                        c_optic_role = str(getattr(cell, "optic_role", "") or "").strip().lower()
                        if c_optic_role == "getter":
                            g_cont, g_key, _ = self._optic_getter_ports(cell)
                            g_cont_v = cell_bindings.get(g_cont.name) if g_cont else None
                            g_key_e = cell_bindings.get(g_key.name) if g_key else None
                            if isinstance(g_cont_v, str) and g_cont_v in ctx.variables and g_key_e:
                                setter = self._find_optic_setter_for_getter(cell, focus_sig=concrete_out)
                                if not hasattr(ctx, "optic_frames") or ctx.optic_frames is None:
                                    ctx.optic_frames = []
                                ctx.optic_frames.append({
                                    "container_var": g_cont_v,
                                    "key_expr": g_key_e,
                                    "key_set": _extract_keys_from_val(g_key_e),
                                    "focus_var": current_out_var,
                                    "initial_focus_var": current_out_var,
                                    "getter_cell": cell,
                                    "setter_cell": setter,
                                    "dirty": False,
                                    "lifted": False,
                                    "closed": False,
                                })
                                if g_cont_v not in getattr(ctx, "superseded_vars", set()):
                                    ctx.active_carrier_var = g_cont_v
                        elif c_optic_role == "setter":
                            s_cont, s_key, s_focus, _ = self._optic_setter_ports(cell)
                            s_cont_v = cell_bindings.get(s_cont.name) if s_cont else None
                            s_key_e = cell_bindings.get(s_key.name) if s_key else None
                            s_focus_v = cell_bindings.get(s_focus.name) if s_focus else None
                            written_k_low = {
                                str(wk).strip().lower()
                                for wk in (_extract_keys_from_val(s_key_e) if s_key_e else ())
                                if str(wk).strip()
                            }
                            written_keys_raw = {
                                str(wk).strip()
                                for wk in (_extract_keys_from_val(s_key_e) if s_key_e else ())
                                if str(wk).strip()
                            }
                            if current_out_var and isinstance(s_cont_v, str):
                                sf_map = getattr(ctx, "setter_focus_keys", None)
                                if not isinstance(sf_map, dict):
                                    sf_map = {}
                                    ctx.setter_focus_keys = sf_map
                                for (sc_v, sf_v), s_ks in list(sf_map.items()):
                                    if sc_v == s_cont_v and s_ks:
                                        kept = {k for k in s_ks if k.lower() not in written_k_low or sf_v == s_focus_v}
                                        if kept:
                                            sf_map.setdefault((current_out_var, sf_v), set()).update(kept)
                                if written_keys_raw and isinstance(s_focus_v, str) and s_focus_v in ctx.variables:
                                    sf_map.setdefault((current_out_var, s_focus_v), set()).update(written_keys_raw)
                            if isinstance(s_focus_v, str):
                                for fr_s in getattr(ctx, "optic_frames", ()) or ():
                                    if not fr_s.get("closed") and fr_s.get("focus_var") == s_focus_v:
                                        fr_s["closed"] = True
                                        fr_s["dirty"] = False
                                        fr_s["closed_var"] = current_out_var
                            if written_k_low:
                                prev_reads = getattr(ctx, "getter_read_keys", None) or set()
                                new_written = written_k_low - prev_reads
                                if new_written:
                                    if not hasattr(ctx, "written_keys") or ctx.written_keys is None:
                                        ctx.written_keys = set()
                                    ctx.written_keys.update(new_written)
                            if not hasattr(ctx, "container_key_qualifiers") or ctx.container_key_qualifiers is None:
                                ctx.container_key_qualifiers = {}
                            if isinstance(s_cont_v, str):
                                for (kv_var, k_key), kq_val in list(ctx.container_key_qualifiers.items()):
                                    if kv_var == s_cont_v and k_key not in written_k_low and kq_val:
                                        ctx.container_key_qualifiers[(current_out_var, k_key)] = set(kq_val)
                            focus_q_set: Set[str] = set()
                            if isinstance(s_focus_v, str) and s_focus_v in ctx.variables:
                                f_sig_v = ctx.variables[s_focus_v][0]
                                f_raw_q = _get_p_field(f_sig_v, "qualifiers", None) or getattr(getattr(f_sig_v, "signature", None), "qualifiers", None) or ()
                                focus_q_set = _flat_quals_set(f_raw_q)
                            for wk in written_k_low:
                                if focus_q_set:
                                    ctx.container_key_qualifiers[(current_out_var, wk)] = set(focus_q_set)
                                else:
                                    ctx.container_key_qualifiers.pop((current_out_var, wk), None)
                            _record_setter_key_layout(ctx, s_cont_v, current_out_var, s_focus_v, written_k_low)
                            for fr in getattr(ctx, "optic_frames", ()) or ():
                                if fr.get("closed"):
                                    continue
                                fr_keys = {str(k).strip().lower() for k in (fr.get("key_set") or ()) if str(k).strip()}
                                matches_focus = bool(s_focus_v and fr.get("focus_var") == s_focus_v)
                                matches_cont_key = bool(
                                    s_cont_v
                                    and fr.get("container_var") == s_cont_v
                                    and (
                                        (written_k_low and fr_keys and bool(written_k_low & fr_keys))
                                        or (s_key_e and fr.get("key_expr") == s_key_e)
                                    )
                                )
                                if matches_focus or matches_cont_key:
                                    fr["closed"] = True
                                    fr["dirty"] = False
                                    fr["closed_var"] = current_out_var
                                elif s_cont_v and fr.get("container_var") == s_cont_v and current_out_var:
                                    fr["container_var"] = current_out_var
                            if isinstance(s_cont_v, str) and s_cont_v in ctx.variables and s_cont_v != current_out_var:
                                ctx.superseded_vars.add(s_cont_v)
                            ctx.active_carrier_var = current_out_var
                        elif c_optic_role == "complement":
                            comp_c_p = _resolve_cell_port(cell, getattr(cell, "container_port", None), prefer="inputs") or getattr(cell, "primary_input", None)
                            comp_k_p = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs") or next(
                                (p for p in (getattr(cell, "inputs", {}) or {}).values() if getattr(p, "complement_of_port", None)),
                                None,
                            )
                            comp_cv = cell_bindings.get(comp_c_p.name) if comp_c_p else None
                            comp_ke = cell_bindings.get(comp_k_p.name) if comp_k_p else None
                            dropped_k_low = {str(dk).strip().lower() for dk in (_extract_keys_from_val(comp_ke) if comp_ke else ()) if str(dk).strip()}
                            if isinstance(comp_cv, str) and hasattr(ctx, "container_key_qualifiers") and isinstance(ctx.container_key_qualifiers, dict):
                                for (kv_var, k_key), kq_val in list(ctx.container_key_qualifiers.items()):
                                    if kv_var == comp_cv and k_key not in dropped_k_low and kq_val:
                                        ctx.container_key_qualifiers[(current_out_var, k_key)] = set(kq_val)
                            _propagate_container_key_layouts(ctx, (comp_cv,), current_out_var, excluded_keys=dropped_k_low, overwrite=True)
                        else:
                            if is_carrier and hasattr(ctx, "container_key_qualifiers") and isinstance(ctx.container_key_qualifiers, dict):
                                for b_in_v in bound_in_vars:
                                    for (kv_var, k_key), kq_val in list(ctx.container_key_qualifiers.items()):
                                        if kv_var == b_in_v and kq_val and (current_out_var, k_key) not in ctx.container_key_qualifiers:
                                            ctx.container_key_qualifiers[(current_out_var, k_key)] = set(kq_val)
                            if is_carrier:
                                _propagate_container_key_layouts(ctx, bound_in_vars, current_out_var)
                            out_entry_meta = ctx.variables.get(current_out_var) if current_out_var else None
                            out_sig_meta = out_entry_meta[0] if isinstance(out_entry_meta, tuple) else out_entry_meta
                            out_str_k = str(getattr(out_sig_meta, "structure_kind", "") or "").lower()
                            out_rank_m = getattr(out_sig_meta, "canonical_rank", None)
                            if out_rank_m is None:
                                out_rank_m = getattr(out_sig_meta, "rank", None)
                            out_tn_m = str(
                                getattr(getattr(out_sig_meta, "signature", out_sig_meta), "type_name", "")
                                or getattr(out_sig_meta, "type_name", "")
                                or ""
                            ).strip().lower()
                            is_scalar_out = (
                                out_rank_m == 0
                                or out_str_k == "scalar"
                                or (bool(out_tn_m) and registry.is_subtype(out_tn_m, "scalar"))
                            )
                            c_nr_low = str(getattr(cell, "node_role", "") or "").strip().lower()
                            _optic_r = registry.get_roles_for_category("optic")
                            _est_r = registry.get_roles_for_category("estimator") | registry.get_roles_for_category("model")
                            is_non_mutating_consumer = (
                                is_scalar_out
                                or out_str_k == "handle"
                                or c_optic_role in _optic_r
                                or getattr(cell, "stage", None) == 3
                                or c_nr_low in (registry.get_terminal_roles_set() | _est_r)
                            )
                            if (
                                current_out_var
                                and not is_scalar_out
                                and out_str_k != "handle"
                                and c_optic_role not in _optic_r
                                and isinstance(getattr(ctx, "setter_focus_keys", None), dict)
                            ):
                                sf_map = ctx.setter_focus_keys
                                src_conts = set(bound_in_vars)
                                for biv in list(bound_in_vars):
                                    src_conts.update(getattr(ctx, "var_parents", {}).get(biv, ()) or ())
                                for (sc_v, sf_v), s_ks in list(sf_map.items()):
                                    if sc_v in src_conts and s_ks:
                                        sf_map.setdefault((current_out_var, sf_v), set()).update(s_ks)
                            for fr in getattr(ctx, "optic_frames", ()) or ():
                                if fr.get("closed"):
                                    continue
                                if fr.get("focus_var") in bound_in_vars:
                                    setter_c = fr.get("setter_cell")
                                    focus_compat = True
                                    if is_non_mutating_consumer or not current_out_var:
                                        focus_compat = False
                                    elif setter_c is not None and out_sig_meta is not None:
                                        _, _, sf_p, _ = self._optic_setter_ports(setter_c)
                                        if sf_p is not None and not _shape_compatible(out_sig_meta, sf_p, producer_cell=cell, consumer_cell=setter_c):
                                            focus_compat = False
                                    if focus_compat:
                                        fr["focus_var"] = current_out_var
                                        fr["dirty"] = True
                                        cont_v = fr.get("container_var")
                                        if cont_v in ctx.variables and cont_v not in getattr(ctx, "superseded_vars", set()):
                                            ctx.active_carrier_var = cont_v
                                    elif not fr.get("dirty"):
                                        fr["closed"] = True

                        if (
                            getattr(cell, "stage", None) == 2
                            and out_t
                            and c_optic_role not in (registry.get_roles_for_category("optic") - {"setter"})
                            and not getattr(cell, "projection", None)
                            and not schema_is_bridge_cell(cell, registry)
                        ):
                            canon_out = registry.canonical_name(out_t).lower()
                            open_optic_conts = {
                                fr.get("container_var")
                                for fr in (getattr(ctx, "optic_frames", ()) or ())
                                if not fr.get("closed")
                            }
                            for b_in_v in bound_in_vars:
                                if b_in_v in open_optic_conts:
                                    continue
                                b_in_sig = ctx.variables.get(b_in_v, (None, None))[0]
                                b_in_t = str(getattr(getattr(b_in_sig, "signature", b_in_sig), "type_name", "") or "").lower()
                                if b_in_t and registry.canonical_name(b_in_t).lower() == canon_out:
                                    ctx.superseded_vars.add(b_in_v)
                        if (
                            current_out_var
                            and current_out_var in ctx.variables
                            and not _is_sink_cell(cell)
                            and getattr(cell, "stage", None) == 2
                            and c_optic_role not in (registry.get_roles_for_category("optic") - {"setter"})
                            and hasattr(ctx, "var_partitions")
                            and isinstance(ctx.var_partitions, dict)
                            and current_out_var not in ctx.var_partitions
                        ):
                            _co_meta = ctx.variables[current_out_var][0]
                            _co_rk = getattr(_co_meta, "canonical_rank", None)
                            if _co_rk is None:
                                _co_rk = getattr(_co_meta, "rank", None)
                            _co_sk = str(getattr(_co_meta, "structure_kind", "") or "").lower()
                            _co_nr = str(getattr(cell, "node_role", "") or "").strip().lower()
                            if (
                                _co_sk not in ("scalar", "handle")
                                and (_co_rk is None or _co_rk >= 1)
                                and _co_nr not in registry.get_terminal_roles_set()
                            ):
                                _out_lead_dim = _port_lead_sym_dim(cell.primary_output)
                                _matched_part = None
                                if _out_lead_dim:
                                    for _in_pn, _in_port in (getattr(cell, "inputs", {}) or {}).items():
                                        if _port_lead_sym_dim(_in_port) == _out_lead_dim:
                                            _in_bv = cell_bindings.get(_in_pn)
                                            if isinstance(_in_bv, str) and _in_bv in ctx.var_partitions:
                                                _matched_part = ctx.var_partitions[_in_bv]
                                                break
                                if _matched_part is None:
                                    for b_in_v in bound_in_vars:
                                        if b_in_v in ctx.var_partitions:
                                            _matched_part = ctx.var_partitions[b_in_v]
                                            break
                                if _matched_part is not None:
                                    ctx.var_partitions[current_out_var] = _matched_part
            else:
                # Void output cell (outputs: {}) - e.g. .show(), .save() returning None.
                # Do not advance counter, do not declare phantom variables, and do not overwrite producer_var.
                pass

            cell_typed_bindings = self.scope_stage.finalize_cell_scope(cell, cell_bindings, ctx, accumulated_sigma)

            pipeline_bindings.append((cell, cell_bindings))

        # Close any dirty optic frames remaining at end-of-pipeline before terminal return
        for fr in getattr(ctx, "optic_frames", ()) or ():
            if not fr.get("closed") and fr.get("dirty"):
                closed_v = self._emit_optic_setter_closure(
                    fr,
                    ctx=ctx,
                    registry=registry,
                    accumulated_sigma=accumulated_sigma,
                    pipeline_bindings=pipeline_bindings,
                    var_counter_ref=var_counter_ref,
                    cell_out_vars=cell_out_vars,
                    cell_port_vars=cell_port_vars,
                )
                if closed_v:
                    var_counter = var_counter_ref[0]
                    producer_var = closed_v

        proof = build_typed_pipeline_proof(pipeline_bindings, ctx, accumulated_sigma)
        self.last_pipeline_bindings = pipeline_bindings
        self.last_pipeline_typed_bindings = [getattr(c, "typed_bindings", {}) for c, _ in pipeline_bindings]
        self.last_typed_proof = proof
        ctx.typed_proof = proof
        return Success(pipeline_bindings, accumulated_sigma)

    @staticmethod
    def _instantiate_ast_template(
        template: str,
        bindings: Dict[str, Any],
        inputs: Dict[str, Any],
        known_vars: Optional[Set[str]] = None,
    ) -> str:
        """
        Synthesizes executable Python code from an AST template, adhering to identity omission semantics:
        - Required positional parameters are instantiated with their bound values.
        - Actively bound optional configurations are emitted as keyword arguments (key=val).
        - Unbound optional parameters with defaults are omitted, letting runtime defaults apply.
        - Enforces param_kind calling conventions (positional_only, keyword_only, var_positional, var_keyword).
        """
        if not template or not template.strip():
            return ""

        active_vars = set(known_vars or ())
        for k_b, v in bindings.items():
            if isinstance(v, str):
                v_clean = v.strip().strip("'\"")
                if k_b == "output_var" and v_clean.isidentifier():
                    active_vars.add(v_clean)

        try:
            try:
                from .language_backend import get_active_language_backend
            except (ImportError, ValueError):
                from language_backend import get_active_language_backend
            if not get_active_language_backend().is_python_ast:
                res = template
                for k, v in bindings.items():
                    if v is not None:
                        res = res.replace(f"{{{k}}}", str(v))
                return res
        except Exception:
            pass

        ph_map: Dict[str, str] = {}
        known_placeholders = set(inputs.keys()) | set(bindings.keys()) | {"output_var"}
        ast_ready = template
        for name in extract_template_placeholders(template):
            if known_placeholders and name not in known_placeholders:
                continue
            ph_id = f"_nstl_ph_{name}"
            ph_map[ph_id] = name
            ast_ready = ast_ready.replace(f"{{{name}}}", ph_id)
        try:
            parsed = ast.parse(ast_ready)
        except SyntaxError as exc:
            raise EmissionLivenessError(f"Invalid Python template syntax: {exc}") from exc

        def _build_value_ast(orig_name: str, val: Any, p_sig: Optional[Any], in_store_ctx: bool = False) -> ast.AST:
            if is_unresolved_value(val):
                raise UnresolvedPlaceholderError(
                    f"Cannot emit code with unresolved port value for placeholder '{orig_name}' in template: {template}"
                )

            if in_store_ctx:
                var_name = str(val).strip().strip("'\"") if val is not None else ""
                if not var_name.isidentifier():
                    var_name = "_sink_out"
                return ast.Name(id=var_name, ctx=ast.Store())

            if val is None:
                return ast.Constant(value=None)

            if isinstance(val, ast.AST):
                return val

            if orig_name == "output_var":
                var_name = str(val).strip().strip("'\"")
                return ast.Name(id=var_name, ctx=ast.Load())

            if isinstance(val, bool):
                return ast.Constant(value=val)
            if isinstance(val, (int, float)):
                return ast.Constant(value=val)
            if isinstance(val, (list, tuple, dict)):
                try:
                    return ast.parse(repr(val), mode="eval").body
                except Exception as _exc:
                    logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            s = str(val).strip()
            _be_ast = get_active_language_backend()
            if _be_ast.is_bool_literal(s, case_insensitive=False):
                return ast.Constant(value=(s == _be_ast.bool_literals[0]))
            if s == _be_ast.null_literal:
                return ast.Constant(value=None)

            try:
                num_i = int(s)
                return ast.Constant(value=num_i)
            except ValueError:
                try:
                    num_f = float(s)
                    return ast.Constant(value=num_f)
                except ValueError:
                    pass

            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                try:
                    parsed_str_lit = ast.parse(s, mode="eval").body
                    if isinstance(parsed_str_lit, ast.Constant) and isinstance(parsed_str_lit.value, str):
                        return ast.Constant(value=parsed_str_lit.value)
                except SyntaxError:
                    pass

            # Check if value is or references in-scope variables or expressions
            unquoted = s
            if (unquoted.startswith("'") and unquoted.endswith("'")) or (unquoted.startswith('"') and unquoted.endswith('"')):
                unquoted = unquoted[1:-1].strip()

            if unquoted in active_vars:
                return ast.Name(id=unquoted, ctx=ast.Load())

            # Dynamic module & builtin reflection: resolves any valid Python package, builtin,
            # or registered lattice alias without requiring a static hardcoded module whitelist.
            def _is_resolvable_module_or_builtin(ident: str) -> bool:
                if not ident or not ident.isidentifier():
                    return False
                if ident in builtins.__dict__ or ident in sys.modules:
                    return True
                try:
                    from .lattice import TypeRegistry
                except (ImportError, ValueError):
                    try:
                        from lattice import TypeRegistry
                    except (ImportError, ValueError):
                        TypeRegistry = None
                if TypeRegistry is not None:
                    reg = TypeRegistry.get_instance()
                    if ident in reg.get_all_aliases():
                        return True
                try:
                    if importlib.util.find_spec(ident) is not None:
                        return True
                except Exception as _exc:
                    logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
                return False

            # Try parsing s as an evaluated Python expression (handles array projections, bridges, lambdas, indexed lookups)
            try:
                parsed_expr = ast.parse(s, mode="eval").body
                free_names = _extract_free_expr_names(parsed_expr)
                if free_names and all(
                    (v in active_vars or _is_resolvable_module_or_builtin(v))
                    for v in free_names
                ):
                    return parsed_expr
                if not free_names and (
                    isinstance(parsed_expr, (ast.Lambda, ast.List, ast.Dict, ast.Tuple, ast.Set, ast.Constant))
                ):
                    return parsed_expr
            except Exception as _exc:
                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            # String literal / parameter identifier (e.g. column name 'X', strategy 'mean', how 'any', filename 'input.csv')
            return ast.Constant(value=unquoted)

        def _build_unbound_required_ast(orig_name: str, p_sig: Optional[Any]) -> ast.AST:
            if p_sig is not None and getattr(p_sig, "default_value", None) is not None:
                return _build_value_ast(orig_name, p_sig.default_value, p_sig)
            raise UnresolvedPlaceholderError(
                f"Cannot emit code: required placeholder '{orig_name}' has no bound value or default in template: {template}"
            )

        class CallOptimizer(ast.NodeTransformer):
            def visit_Call(self, node):
                new_args = []
                new_keywords = []
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in ph_map:
                        orig_name = ph_map[arg.id]
                        p_sig = inputs.get(orig_name)
                        is_req = getattr(p_sig, "required", True) if p_sig else True
                        p_kind = getattr(p_sig, "param_kind", "standard")
                        val = bindings.get(orig_name)
                        if val is not None:
                            val_node = _build_value_ast(orig_name, val, p_sig)
                            if p_kind == "keyword_only":
                                new_keywords.append(ast.keyword(arg=orig_name, value=val_node))
                            elif p_kind == "var_keyword":
                                new_keywords.append(ast.keyword(arg=None, value=val_node))
                            else:
                                new_args.append(val_node)
                        elif is_req:
                            val_node = _build_unbound_required_ast(orig_name, p_sig)
                            new_args.append(val_node)
                        # If optional and val is None: omit from call
                    else:
                        new_args.append(self.visit(arg))

                for kw in node.keywords:
                    val_node = kw.value
                    if isinstance(val_node, ast.Name) and val_node.id in ph_map:
                        orig_name = ph_map[val_node.id]
                        p_sig = inputs.get(orig_name)
                        is_req = getattr(p_sig, "required", True) if p_sig else True
                        val = bindings.get(orig_name)
                        if val is not None:
                            kw_val_node = _build_value_ast(orig_name, val, p_sig)
                            new_keywords.append(ast.keyword(arg=kw.arg, value=kw_val_node))
                        elif is_req:
                            kw_val_node = _build_unbound_required_ast(orig_name, p_sig)
                            new_keywords.append(ast.keyword(arg=kw.arg, value=kw_val_node))
                        # If optional and val is None: omit keyword from call
                    else:
                        new_keywords.append(ast.keyword(arg=kw.arg, value=self.visit(kw.value)))

                node.func = self.visit(node.func)
                node.args = new_args
                node.keywords = new_keywords
                return node

            def visit_Subscript(self, node):
                node.value = self.visit(node.value)
                if isinstance(node.slice, ast.Name) and node.slice.id in ph_map:
                    orig_name = ph_map[node.slice.id]
                    p_sig = inputs.get(orig_name)
                    is_req = getattr(p_sig, "required", True) if p_sig else True
                    val = bindings.get(orig_name)
                    if val is not None:
                        node.slice = _build_value_ast(orig_name, val, p_sig)
                    elif is_req:
                        node.slice = _build_unbound_required_ast(orig_name, p_sig)
                else:
                    node.slice = self.visit(node.slice)
                return node

            def visit_Name(self, node):
                if node.id in ph_map:
                    orig = ph_map[node.id]
                    p_sig = inputs.get(orig)
                    if isinstance(node.ctx, ast.Store):
                        val = bindings.get(orig)
                        return _build_value_ast(orig, val, p_sig, in_store_ctx=True)
                    if orig in bindings and bindings[orig] is not None:
                        return _build_value_ast(orig, bindings[orig], p_sig, in_store_ctx=False)
                    is_req = getattr(p_sig, "required", True) if p_sig else True
                    if is_req:
                        return _build_unbound_required_ast(orig, p_sig)
                return node

            def _rewrite_fn_def(self, node):
                if node.name in ph_map:
                    orig = ph_map[node.name]
                    if orig in bindings and bindings[orig] is not None:
                        val = str(bindings[orig]).strip().strip("'\"")
                        if val.isidentifier():
                            node.name = val
                            active_vars.add(val)
                elif isinstance(node.name, str) and node.name.isidentifier():
                    active_vars.add(node.name)
                all_fn_args = (
                    list(getattr(node.args, "posonlyargs", []) or ())
                    + list(getattr(node.args, "args", []) or ())
                    + list(getattr(node.args, "kwonlyargs", []) or ())
                )
                if getattr(node.args, "vararg", None) is not None:
                    all_fn_args.append(node.args.vararg)
                if getattr(node.args, "kwarg", None) is not None:
                    all_fn_args.append(node.args.kwarg)
                for a in all_fn_args:
                    if a.arg in ph_map:
                        orig = ph_map[a.arg]
                        if orig in bindings and bindings[orig] is not None:
                            val = str(bindings[orig]).strip().strip("'\"")
                            if val.isidentifier():
                                a.arg = val
                                active_vars.add(val)
                    elif getattr(a, "arg", None) and a.arg.isidentifier():
                        active_vars.add(a.arg)
                self.generic_visit(node)
                return node

            def visit_FunctionDef(self, node):
                return self._rewrite_fn_def(node)

            def visit_AsyncFunctionDef(self, node):
                return self._rewrite_fn_def(node)

            def visit_ClassDef(self, node):
                if node.name in ph_map:
                    orig = ph_map[node.name]
                    if orig in bindings and bindings[orig] is not None:
                        val = str(bindings[orig]).strip().strip("'\"")
                        if val.isidentifier():
                            node.name = val
                            active_vars.add(val)
                elif isinstance(node.name, str) and node.name.isidentifier():
                    active_vars.add(node.name)
                self.generic_visit(node)
                return node

            def visit_ExceptHandler(self, node):
                if isinstance(node.type, ast.Name) and node.type.id in ph_map:
                    orig = ph_map[node.type.id]
                    if orig in bindings and bindings[orig] is not None:
                        val = str(bindings[orig]).strip().strip("'\"")
                        if val.isidentifier():
                            node.type = ast.Name(id=val, ctx=ast.Load())
                self.generic_visit(node)
                return node

        optimized = CallOptimizer().visit(parsed)

        try:
            return ast.unparse(optimized)
        except Exception as e:
            raise EmissionLivenessError(f"Failed to unparse AST template: {e}") from e

    @staticmethod
    def _prune_unconsumed_transform_outputs(
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]]
    ) -> List[Tuple[Cell, Dict[str, str]]]:
        """
        Removes non-sink, non-egress transform steps whose declared output
        variable(s) are never referenced by any later step's bindings in the
        ACTUAL wired pipeline. This is deliberately independent of the
        planner's search-time connectivity heuristic (type-level "could this
        feed something") -- it checks the binder's own ground truth instead,
        so it catches exactly the cases where the planner and the binder
        silently disagreed about which variable actually gets used.

        Stage-3 cells and cells with node_role == "sink" are always kept:
        their purpose is the side effect (write, plot, save), not a return
        value some later step consumes, so an "unused" return value there is
        normal and correct, not dead code.
        """
        if not pipeline_bindings:
            return pipeline_bindings

        current = list(pipeline_bindings)
        changed = True
        while changed:
            changed = False
            for idx, (cell, bnd) in enumerate(current):
                stage = getattr(cell, "stage", None)
                role = str(getattr(cell, "node_role", "") or "").lower()
                outputs = getattr(cell, "outputs", {}) or {}
                # The final cell produces the terminal pipeline result and must never be pruned
                reg_pr = TypeRegistry.get_instance()
                term_roles = reg_pr.get_terminal_roles_set()
                if idx == len(current) - 1 or role in term_roles or not outputs:
                    continue
                # Protect explicit calculations/metrics (e.g. mean, metrics, aggregations) that witness a clause,
                # but pure intermediate projections/slices (like PD_SELECT_COLUMN) whose output is never used
                # are dead extraction bridges and must be pruned to avoid anti-causal or unused variable errors.
                if getattr(cell, "is_goal", False):
                    continue
                proj_roles = reg_pr.get_roles_for_category("projection") | set(reg_pr.get_projection_tokens() or ())
                is_pure_projection_or_bridge = (
                    getattr(cell, "projection", None) is not None
                    or (
                        getattr(cell, "mutation_type", "") == "pure"
                        and (role in proj_roles or schema_is_bridge_cell(cell, reg_pr))
                    )
                )
                if not is_pure_projection_or_bridge and getattr(cell, "matched_clause_idx", None) is not None:
                    continue
                out_vars = {v for k, v in bnd.items() if k in outputs and isinstance(v, str)}
                if not out_vars:
                    continue
                is_py_ast = True
                try:
                    is_py_ast = get_active_language_backend(reg_pr).is_python_ast
                except Exception:
                    is_py_ast = True
                referenced_later: Set[str] = set()
                for _, other_bnd in current[idx + 1:]:
                    for val in other_bnd.values():
                        if isinstance(val, str):
                            if is_py_ast:
                                try:
                                    expr_ast = ast.parse(val.strip(), mode="eval")
                                    referenced_later.update(
                                        n.id for n in ast.walk(expr_ast)
                                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                                    )
                                    continue
                                except SyntaxError:
                                    pass
                            referenced_later.update(tokenize_alphanumeric(val))
                if not (out_vars & referenced_later):
                    current.pop(idx)
                    changed = True
                    break
        return current

    def _synthesize_pipeline_bindings_with_llm(
        self,
        cells: List[Cell],
        prompt: str,
        ctx: ExecutionContext
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synthesizes complete dataflow variable bindings, column projections, and parameters
        using the LLM (in supported profiles C, D, E, S) for the candidate cell pipeline.
        Returns a mapping from step key ('0', '1', ... or cell_id) to {port_name: expression}.
        """
        if not prompt or not cells:
            return {}
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if not mm.can_synthesize():
                return {}
        except Exception:
            return {}

        pipeline_steps_desc = []
        var_counter = 1  # the emitter's first variable is var_1 and it advances once per non-sink CELL
        for idx, c in enumerate(cells):
            in_desc = []
            for p_name, p_sig in (getattr(c, "inputs", {}) or {}).items():
                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or ""
                t_name = getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or ""
                req = "required" if getattr(p_sig, "required", False) else "optional"
                in_desc.append(f"{p_name} ({t_name}, role={p_role}, {req})")

            out_names = []
            if not _is_sink_cell(c):
                c_outputs = getattr(c, "outputs", {}) or {}
                if len(c_outputs) > 1:
                    for o_name, o_sig in c_outputs.items():
                        var_name = _semantic_var_name(c, var_counter, port_name=o_name, out_sig=o_sig)
                        var_counter += 1
                        o_type = getattr(getattr(o_sig, "signature", o_sig), "type_name", "") or ""
                        out_names.append(f"{var_name} ({o_type})")
                elif len(c_outputs) == 1:
                    prim_o = getattr(c, "primary_output", None)
                    o_name, o_sig = next(iter(c_outputs.items()))
                    var_name = _semantic_var_name(
                        c,
                        var_counter,
                        port_name=(getattr(prim_o, "name", None) or o_name),
                        out_sig=(prim_o or o_sig),
                    )
                    var_counter += 1
                    o_type = getattr(getattr(o_sig, "signature", o_sig), "type_name", "") or ""
                    out_names.append(f"{var_name} ({o_type})")
                else:
                    var_name = _semantic_var_name(c, var_counter)
                    var_counter += 1
                    out_names.append(var_name)

            pipeline_steps_desc.append(
                f"Step {idx} [cell_id: {c.cell_id}]:\n"
                f"  Inputs needed: {', '.join(in_desc) if in_desc else 'None'}\n"
                f"  Template: {getattr(c, 'code_template', '')}\n"
                f"  Outputs produced: {', '.join(out_names) if out_names else 'None (side-effect/sink)'}"
            )

        steps_text = "\n".join(pipeline_steps_desc)
        schema = {
            "type": "object",
            "properties": {
                "step_bindings": {
                    "type": "object",
                    "description": "Mapping from step index (e.g. '0', '1', '2') or cell_id to port bindings dict",
                    "additionalProperties": {
                        "type": "object",
                        "description": "Mapping of port_name to variable name (e.g. 'var_0'), member projection or subscript (e.g. \"var_0['member']\", \"var_0[0]\"), or literal",
                        "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                    }
                }
            },
            "required": ["step_bindings"]
        }

        query = (
            f"User Prompt: {prompt}\n\n"
            f"Ordered Pipeline Steps:\n{steps_text}\n\n"
            "Task: For each step, determine which variable (e.g. var_0, var_1), member projection or subscript (e.g. var_0['member'], var_0[0]), "
            "or literal value should be bound to each input port.\n"
            "Return valid JSON only matching schema with 'step_bindings'."
        )

        try:
            raw = mm.generate_text(
                query,
                max_tokens=1024,
                schema=schema,
                system_prompt="You are a dataflow compiler. Return exact variable wiring, member projections, and arguments for each pipeline step in valid JSON."
            )
            if not raw or not raw.strip():
                return {}
            text = raw.strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(text)
            bindings = parsed.get("step_bindings", {})
            return bindings if isinstance(bindings, dict) else {}
        except Exception as e:
            logger.debug(f"[UNIFICATION] LLM pipeline binding synthesis skipped or failed: {e}")
            return {}

    def _synthesize_single_port_binding_with_llm(
        self,
        cell: Cell,
        port_name: str,
        port_sig: Any,
        concrete_sig: Any,
        ctx: ExecutionContext
    ) -> Optional[str]:
        """
        Emergency fallback: queries LLM to bind an ungrounded required port to an in-scope variable or expression.
        """
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if not mm.can_synthesize():
                return None
        except Exception:
            return None

        prompt = getattr(ctx, "prompt", "") or getattr(ctx, "_prompt", "") or ""
        in_scope_vars = []
        for v_name, (v_sig, _) in ctx.variables.items():
            t_name = getattr(getattr(v_sig, "signature", v_sig), "type_name", "") or ""
            r_name = getattr(v_sig, "port_role", "") or getattr(v_sig, "derived_role", "") or ""
            in_scope_vars.append(f"- {v_name}: type={t_name}, role={r_name}")

        is_id_port, is_param_list = _classify_template_identifier_port(cell, port_name)
        p_role_str = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "role", "") or "").strip().lower()
        t_name_str = str(getattr(concrete_sig, "type_name", "") or "").strip().lower()
        is_callable_port = TypeRegistry.get_instance().is_hof_role(p_role_str) or t_name_str.startswith("callable")
        if is_id_port:
            binding_hint = (
                "Specify ONLY an unquoted Python parameter list (e.g. x or x, y)."
                if is_param_list
                else "Specify ONLY a single valid unquoted Python identifier name (e.g. my_fn)."
            )
        elif is_callable_port:
            binding_hint = (
                "Specify ONLY a valid Python lambda expression (e.g. lambda s: s < 5) or callable matching the prompt condition."
            )
        else:
            binding_hint = (
                f"Specify ONLY the exact variable name (e.g. var_0), projection or subscript (e.g. var_0['member'], var_0[0]), or literal that satisfies port '{port_name}'."
            )

        scope_desc = "\n".join(in_scope_vars) if in_scope_vars else "(none yet)"
        query = (
            f"User Prompt: {prompt}\n\n"
            f"Current Cell: {cell.cell_id}\n"
            f"Template: {getattr(cell, 'code_template', '')}\n"
            f"Required Unbound Port: '{port_name}' (expected type: {getattr(concrete_sig, 'type_name', '')}, role: {getattr(port_sig, 'port_role', '')})\n"
            f"In-scope available variables:\n{scope_desc}\n\n"
            f"{binding_hint}\n"
            "Respond in JSON: {\"binding\": \"expression\"}"
        )
        schema = {
            "type": "object",
            "properties": {
                "binding": {"type": "string", "description": "The exact variable name, identifier, projection, lambda, or literal to bind"}
            },
            "required": ["binding"]
        }
        try:
            raw = mm.generate_text(
                query,
                max_tokens=128,
                schema=schema,
                system_prompt="You are a dataflow compiler. Provide the exact variable, identifier, callable, or expression to satisfy the port."
            )
            if raw and raw.strip():
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(text)
                cand = parsed.get("binding")
                if cand:
                    cand_str = str(cand).strip()
                    if is_id_port:
                        norm_id = _normalize_identifier_port_binding(cand, is_param_list=is_param_list)
                        if norm_id is not None:
                            return norm_id
                        return None
                    # Validate candidate against scope, free AST names, and typing
                    try:
                        parsed_ast = ast.parse(cand_str, mode='eval')
                        free_names = _extract_free_expr_names(parsed_ast.body)
                        if all(
                            (v in ctx.variables or v in _PY_BUILTIN_NAMES) and not keyword.iskeyword(v)
                            for v in free_names
                        ):
                            # Check typing
                            rec = resolve_typed_binding_record(
                                cell=cell,
                                port_name=port_name,
                                bound_val=cand_str,
                                port_sig=concrete_sig,
                                var_signatures={v: s[0] if isinstance(s, tuple) else s for v, s in ctx.variables.items()},
                                expected_sig=concrete_sig,
                            )
                            if rec and rec.resulting_signature:
                                rec_sig = getattr(rec.resulting_signature, "signature", rec.resulting_signature)
                                u = unify(rec_sig, concrete_sig.signature, Substitution())
                                if u is not None:
                                    return cand_str
                    except Exception as _exc:
                        logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
        except Exception as e:
            logger.debug(f"[UNIFICATION] Single-port LLM synthesis failed: {e}")
        return None

    def _extract_pipeline_slots_with_llm(
        self,
        cells: List[Cell],
        prompt: str,
        ctx: ExecutionContext
    ) -> None:
        """
        Grounded parameter and variable binding via LLM slot extraction.
        Delegates to `PipelineSlotBinder.extract` using canonical `type_predicates`.
        """
        PipelineSlotBinder.extract(
            cells,
            prompt,
            ctx,
            TypeRegistry.get_instance(),
            is_path_port_fn=_lattice_is_path_port,
            is_macro_block_slot_fn=_is_macro_block_slot,
            schema_is_bridge_fn=schema_is_bridge_cell,
            resolve_cell_port_fn=_resolve_cell_port,
            is_path_string_fn=ExecutionContext._is_path_string,
            type_term_cls=TypeTerm,
        )

    def _resolve_missing_ports_with_llm(
        self,
        cell_objs: List[Cell],
        active_ctx: ExecutionContext,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        accum_sigma: Substitution,
        extract_llm_slots: bool,
    ) -> Tuple[List[Tuple[Cell, Dict[str, str]]], Substitution]:
        """Autonomous missing parameter resolution under --llm-feedback.

        Strictly disabled in symbolic/retrieval profiles (0, A, D).
        """
        allow_feedback = (
            (os.environ.get("NSTL_LLM_FEEDBACK") == "1")
            or bool(getattr(active_ctx, "llm_feedback", False))
        )
        if not (getattr(active_ctx, "unresolved_ports", None) and allow_feedback):
            return pipeline_bindings, accum_sigma

        try:
            from inference import ModelManager
        except (ImportError, ValueError):
            from .inference import ModelManager
        mm = ModelManager.get_instance()
        if not (mm.profile and mm.can_synthesize()):
            return pipeline_bindings, accum_sigma

        prompt_txt = getattr(active_ctx, "_prompt", "") or getattr(self.context, "_prompt", "")
        reg = TypeRegistry.get_instance()
        resolved_any = False
        rem_unres = []
        orig_params = dict(active_ctx.parameters)
        orig_lits = set(active_ctx.target_literals)
        attempted_ports = []
        for cid, p_name in list(active_ctx.unresolved_ports):
            cell_obj = next((c for c in cell_objs if c.cell_id == cid), None)
            p_sig = cell_obj.inputs.get(p_name) if cell_obj else None
            p_role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
            p_type = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or "").strip().lower()
            is_hof_or_callable = bool(
                (p_role and reg.is_hof_role(p_role))
                or "callable" in p_type
                or p_name in (getattr(cell_obj, "slots", None) or {})
            )
            if is_hof_or_callable:
                rem_unres.append((cid, p_name))
                continue
            p_desc = getattr(p_sig, "description", "") or getattr(p_sig, "type_name", "") or p_name
            query = (
                f"User Prompt: {prompt_txt}\n"
                f"Missing Parameter Required:\n"
                f"Node: {cid}\n"
                f"Parameter Port: {p_name} ({p_desc})\n"
                f"Extract the exact argument value (e.g. identifier, threshold, asset path, configuration mode) from the prompt for this port.\n"
                f'Return JSON: {{"value": <extracted_value>}}'
            )
            try:
                resp = mm.generate_text(query, max_tokens=64, system_prompt="You are a precise argument extractor.")
                if resp and "value" in resp:
                    import json as _json
                    cleaned = resp.strip()
                    if "```json" in cleaned:
                        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in cleaned:
                        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
                    parsed = _json.loads(cleaned)
                    val = parsed.get("value")
                    if val is not None and str(val).strip() and not is_unresolved_value(val):
                        active_ctx.parameters[p_name] = val
                        active_ctx.target_literals.add(str(val))
                        attempted_ports.append((cid, p_name))
                        resolved_any = True
                        continue
            except Exception as _ex:
                logger.debug(f"[UNIFICATION] LLM missing parameter extraction failed for {cid}.{p_name}: {_ex}")
            rem_unres.append((cid, p_name))

        if resolved_any:
            prev_unres = list(active_ctx.unresolved_ports)
            prev_holes = list(getattr(active_ctx, "typed_holes", ()) or ())
            active_ctx.unresolved_ports = rem_unres
            res = self.unify_pipeline(cell_objs, active_ctx, allow_llm=False)
            if not res.is_bottom() and isinstance(res, Success):
                syntax_ok = not any(
                    is_unresolved_value(v)
                    for _, bnd in res.value
                    for v in (bnd or {}).values()
                )
                if syntax_ok:
                    try:
                        try:
                            from .synthesis import render_cell as _probe_render_cell
                            has_rc = True
                        except (ImportError, ValueError):
                            try:
                                from synthesis import render_cell as _probe_render_cell
                                has_rc = True
                            except Exception:
                                _probe_render_cell = None
                                has_rc = False
                        rendered_parts: List[str] = []
                        for c_b, bnd in res.value:
                            if has_rc and _probe_render_cell is not None and (
                                bool(getattr(c_b, "bound_slots", None)) or getattr(c_b, "node_type", "") == "macro"
                            ):
                                r_part = _probe_render_cell(c_b, bnd or {}, indent_level=0, context=active_ctx, accumulated_sigma=res.sigma)
                                if r_part:
                                    rendered_parts.append(r_part)
                            elif getattr(c_b, "code_template", None):
                                rendered_parts.append(
                                    self._instantiate_ast_template(
                                        c_b.code_template.strip(),
                                        bnd or {},
                                        getattr(c_b, "inputs", {}),
                                    )
                                )
                        test_code = "\n".join(rendered_parts)
                        if test_code.strip():
                            ast.parse(test_code)
                    except Exception:
                        syntax_ok = False
                if syntax_ok:
                    pipeline_bindings = res.value
                    accum_sigma = res.sigma
                    remaining_hole_ids = {
                        f"{cid}.{pn}" for cid, pn in (getattr(active_ctx, "unresolved_ports", ()) or ())
                    }
                    if hasattr(active_ctx, "typed_holes") and isinstance(active_ctx.typed_holes, list):
                        active_ctx.typed_holes = [
                            h for h in active_ctx.typed_holes
                            if getattr(h, "hole_id", None) in remaining_hole_ids
                        ]
                    self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
                    self.last_runtime_aliases = dict(getattr(active_ctx, "runtime_aliases", {}) or {})
                    return pipeline_bindings, accum_sigma
            # Roll back if LLM feedback produced bottom or syntactically invalid bindings
            active_ctx.parameters = orig_params
            active_ctx.target_literals = orig_lits
            active_ctx.unresolved_ports = prev_unres
            active_ctx.typed_holes = prev_holes

        return pipeline_bindings, accum_sigma

    def emit_code(
        self,
        cells: Union[List[Cell], List[Tuple[Cell, Dict[str, str]]]],
        context: Optional[ExecutionContext] = None,
        extract_llm_slots: Optional[bool] = None,
        accum_sigma: Optional[Substitution] = None,
    ) -> str:
        """
        Emits clean, fully instantiated code from a verified cell pipeline.
        Replaces port placeholders strictly from unified variable bindings.
        """
        ctx = context or self.context
        if extract_llm_slots is None:
            _can_slot = False
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                _mm = ModelManager.get_instance()
                _can_slot = bool(getattr(_mm, "profile", None) and _mm.can_extract_slots())
            except Exception:
                _can_slot = False
            extract_llm_slots = (
                _can_slot
                or (os.environ.get("NSTL_LLM_FEEDBACK") == "1")
                or bool(getattr(ctx, "llm_feedback", False))
                or bool(getattr(self.context, "llm_feedback", False))
            )

        if accum_sigma is None:
            accum_sigma = Substitution()
        if hasattr(cells, "value"):
            cells = cells.value
        if not cells:
            return ""
        if isinstance(cells[0], tuple):
            active_ctx = ctx
            cell_objs = [c for c, _ in cells]
            pipeline_bindings = cells
        else:
            ctx_run = ctx.clone() if hasattr(ctx, "clone") else ctx
            active_ctx = ctx_run
            cell_objs = cells
            if hasattr(ctx_run, "reset"):
                ctx_run.reset()
            res = self.unify_pipeline(cells, ctx_run, allow_llm=extract_llm_slots)
            if res.is_bottom():
                reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
                raise ValueError(f"Unification Failed: {reason}")
            assert isinstance(res, Success)
            pipeline_bindings = res.value
            accum_sigma = res.sigma
            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
            self.last_runtime_aliases = dict(getattr(ctx_run, "runtime_aliases", {}) or {})

        pipeline_bindings, accum_sigma = self._resolve_missing_ports_with_llm(
            cell_objs=cell_objs,
            active_ctx=active_ctx,
            pipeline_bindings=pipeline_bindings,
            accum_sigma=accum_sigma,
            extract_llm_slots=bool(extract_llm_slots),
        )
        if getattr(active_ctx, "unresolved_ports", None):
            ports_str = ", ".join(f"{cid}.{p}" for cid, p in active_ctx.unresolved_ports)
            raise UnresolvedPlaceholderError(
                f"Code synthesis refused: pipeline contains unresolved ports: [{ports_str}]",
                ports=[(str(cid), str(p)) for cid, p in active_ctx.unresolved_ports],
                typed_holes=list(getattr(active_ctx, "typed_holes", []) or []),
            )

        # Ensure no binding contains an unresolved sentinel
        for cell, bnd in pipeline_bindings:
            for p_name, val in bnd.items():
                if is_unresolved_value(val):
                    raise UnresolvedPlaceholderError(
                        f"Code synthesis refused: port '{p_name}' of cell '{cell.cell_id}' has unresolved value '{val}'",
                        ports=[(str(cell.cell_id), str(p_name))],
                        typed_holes=list(getattr(active_ctx, "typed_holes", []) or []),
                    )

        # Ground-truth dead-branch elimination. The planner's search-time
        # dead-output penalty only checks whether a downstream cell's
        # declared input TYPE could accept a given output (see
        # LatticePlanner._cells_connect) -- never whether this binder
        # actually wired that specific variable anywhere. This binder makes
        # its own independent, greedy choice of which in-scope variable to
        # use for each port, so a step the planner scored as "consumed" can
        # still end up computing a value nothing in the emitted code ever
        # reads. Prune those branches here, against the binder's own real
        # bindings, before anything is turned into source text.
        pipeline_bindings = self._prune_unconsumed_transform_outputs(pipeline_bindings)

        self.last_pipeline_bindings = pipeline_bindings
        self.last_verification_contract = self.build_verification_contract(pipeline_bindings, active_ctx, getattr(active_ctx, "prompt", ""))

        # Collect dependencies recursively using the active LanguageBackend
        deps: List[str] = []
        reg = TypeRegistry.get_instance()
        reg_aliases = reg.get_all_aliases()
        active_backend = get_active_language_backend(reg)

        def collect_deps(c: Cell):
            all_deps = list(getattr(c, "dependencies", [])) + list(getattr(c, "imports", []))
            for dep in all_deps:
                dep_clean = dep.strip()
                if not dep_clean:
                    continue
                if active_backend.is_dependency_statement(dep_clean):
                    if dep_clean not in deps:
                        deps.append(dep_clean)
                    continue
                if not active_backend.is_python_ast:
                    stmt = active_backend.format_dependency_statement(dep_clean)
                    if stmt and stmt not in deps:
                        deps.append(stmt)
                    continue

                base_mod = dep_clean.split(".")[0]
                leaf_mod = dep_clean.rsplit(".", 1)[-1] if "." in dep_clean else ""
                parent_mod = dep_clean.rsplit(".", 1)[0] if "." in dep_clean else ""
                tpl = getattr(c, "code_template", "") or ""
                has_attr_call = f"{base_mod}." in tpl or f"{dep_clean}." in tpl
                found_alias = None
                for alias, target in reg_aliases.items():
                    if target == dep_clean or target == base_mod:
                        if f"{alias}." in tpl:
                            has_attr_call = True
                            found_alias = alias
                            break

                loaded_roots: Set[str] = set()
                if active_backend.is_python_ast and tpl:
                    sanitized_tpl = tpl
                    for _ph in set(extract_template_placeholders(tpl)):
                        sanitized_tpl = sanitized_tpl.replace("{" + _ph + "}", f"_nstl_ph_{_ph}")
                    _tpl_tree, _ = _cached_ast_parse_ro(sanitized_tpl, "exec")
                    if _tpl_tree is not None:
                        for _tn in ast.walk(_tpl_tree):
                            if isinstance(_tn, ast.Name) and isinstance(_tn.ctx, ast.Load) and not _tn.id.startswith("_nstl_ph_"):
                                loaded_roots.add(_tn.id)
                    else:
                        loaded_roots = {w for w in tokenize_alphanumeric(tpl) if not f".{w}" in tpl}
                else:
                    loaded_roots = set(tokenize_alphanumeric(tpl))

                if has_attr_call:
                    if found_alias and found_alias != dep_clean:
                        stmt = f"import {dep_clean} as {found_alias}"
                    else:
                        stmt = f"import {dep_clean}"
                    if stmt not in deps:
                        deps.append(stmt)
                elif leaf_mod and parent_mod and leaf_mod in loaded_roots and f"{leaf_mod}." in tpl:
                    # Submodule referenced by leaf name (e.g. "PIL.Image" used as Image.fromarray)
                    stmt = f"from {parent_mod} import {leaf_mod}"
                    if stmt not in deps:
                        deps.append(stmt)
                else:
                    # Template references bare loaded symbols directly (e.g. LogisticRegression(...) or train_test_split(...))
                    symbols = _get_module_symbols(dep_clean)
                    sym_set = set(symbols)
                    matched = {r for r in loaded_roots if r in sym_set}
                    if not matched and not active_backend.is_python_ast:
                        sym_map = {s.lower(): s for s in symbols}
                        words = set(tokenize_alphanumeric(tpl))
                        matched = {sym_map[w] for w in words if w in sym_map}

                    if matched:
                        for w in sorted(matched):
                            stmt = f"from {dep_clean} import {w}"
                            if stmt not in deps:
                                deps.append(stmt)
                    else:
                        stmt = f"import {dep_clean}"
                        if stmt not in deps:
                            deps.append(stmt)

            for sub_list in getattr(c, "bound_slots", {}).values():
                if isinstance(sub_list, list):
                    for sc in sub_list:
                        if hasattr(sc, "dependencies"):
                            collect_deps(sc)

        for cell, _ in pipeline_bindings:
            collect_deps(cell)

        if active_ctx is not None and hasattr(active_ctx, "composed_bridge_cells"):
            for bcell in active_ctx.composed_bridge_cells:
                collect_deps(bcell)

        # Ensure standard data science aliases referenced in any template or binding are imported
        all_rendered_text = " ".join([getattr(c, "code_template", "") or "" for c, _ in pipeline_bindings])
        for _, b in pipeline_bindings:
            for v in b.values():
                if isinstance(v, str):
                    all_rendered_text += " " + v

        for alias, target in reg_aliases.items():
            pattern = f"{alias}."
            if pattern in all_rendered_text:
                import_stmt = f"import {target} as {alias}"
                if not any(import_stmt in d or f"import {target}" in d for d in deps):
                    deps.append(import_stmt)

        code_lines: List[str] = []
        if deps:
            code_lines.extend(deps)
            code_lines.append("")

        try:
            from .synthesis import render_cell
            has_render_cell = True
        except (ImportError, ValueError):
            try:
                from synthesis import render_cell
                has_render_cell = True
            except ImportError:
                has_render_cell = False

        known_vars: Set[str] = set()
        if deps and active_backend.is_python_ast:
            _dep_ast, _ = _cached_ast_parse_ro("\n".join(deps), "exec")
            if _dep_ast is not None:
                for _dn in ast.walk(_dep_ast):
                    if isinstance(_dn, ast.Import):
                        for _al in _dn.names:
                            known_vars.add(_al.asname or _al.name.split(".")[0])
                    elif isinstance(_dn, ast.ImportFrom):
                        for _al in _dn.names:
                            if _al.name != "*":
                                known_vars.add(_al.asname or _al.name)
        if active_ctx is not None:
            if hasattr(active_ctx, "variables"):
                known_vars.update(active_ctx.variables.keys())
            if hasattr(active_ctx, "scope"):
                known_vars.update(active_ctx.scope.keys())
            if hasattr(active_ctx, "var_sources"):
                known_vars.update(active_ctx.var_sources.keys())
            if hasattr(active_ctx, "variables") and isinstance(active_ctx.variables, dict):
                known_vars.update(active_ctx.variables.keys())
            if hasattr(active_ctx, "pipeline_vars") and isinstance(active_ctx.pipeline_vars, set):
                known_vars.update(active_ctx.pipeline_vars)

        for cell, bindings in pipeline_bindings:
            for out_k in (getattr(cell, "outputs", None) or {}).keys():
                p_out_v = bindings.get(out_k)
                if p_out_v and isinstance(p_out_v, str):
                    known_vars.add(p_out_v.strip().strip("'\""))
            out_v = bindings.get("output_var")
            if out_v and isinstance(out_v, str):
                known_vars.add(out_v.strip().strip("'\""))
            if getattr(cell, "primary_output", None):
                p_out = bindings.get(cell.primary_output.name)
                if p_out and isinstance(p_out, str):
                    known_vars.add(p_out.strip().strip("'\""))

            if getattr(cell, "is_typed_hole", False):
                raise UnificationFailure(
                    f"Unfilled typed hole '{cell.cell_id}' cannot be emitted via raw unconstrained LLM synthesis; "
                    "every morphism must resolve to verified lattice or control-flow cells."
                )

            if has_render_cell and (bool(getattr(cell, "bound_slots", None)) or getattr(cell, "node_type", "") == "macro"):
                rendered = render_cell(cell, bindings, indent_level=0, context=active_ctx, accumulated_sigma=accum_sigma)
                if rendered:
                    code_lines.append(rendered)
            else:
                template = cell.code_template.strip()
                if not template:
                    continue

                instantiated = self._instantiate_ast_template(template, bindings, cell.inputs, known_vars=known_vars)
                code_lines.append(instantiated)

        final_code = "\n".join(code_lines).strip()
        final_code = self._reconcile_imports(final_code)
        prior_vars: Set[str] = set()
        if active_ctx is not None:
            if hasattr(active_ctx, "initial_scope") and isinstance(active_ctx.initial_scope, set):
                prior_vars.update(active_ctx.initial_scope)
            else:
                if hasattr(active_ctx, "scope") and isinstance(active_ctx.scope, dict):
                    prior_vars.update(active_ctx.scope.keys())
                if hasattr(active_ctx, "scope_variables") and isinstance(active_ctx.scope_variables, dict):
                    prior_vars.update(active_ctx.scope_variables.keys())
                if hasattr(active_ctx, "variables") and isinstance(active_ctx.variables, dict):
                    prior_vars.update(active_ctx.variables.keys())

            # Strictly exclude all pipeline-declared/produced variables from prior_vars
            pipeline_vars: Set[str] = set()
            for cell, bindings in pipeline_bindings:
                if isinstance(bindings, dict):
                    outputs = getattr(cell, "outputs", {})
                    if isinstance(outputs, dict):
                        for out_name in outputs.keys():
                            if out_name in bindings and isinstance(bindings[out_name], str):
                                pipeline_vars.add(bindings[out_name])
                    if "output_var" in bindings and isinstance(bindings["output_var"], str):
                        pipeline_vars.add(bindings["output_var"])
                    prim_out = getattr(cell, "primary_output", None)
                    if prim_out and getattr(prim_out, "name", None) in bindings:
                        p_val = bindings[prim_out.name]
                        if isinstance(p_val, str):
                            pipeline_vars.add(p_val)
                    for b_val in bindings.values():
                        if isinstance(b_val, str) and b_val.isidentifier() and not keyword.iskeyword(b_val):
                            if (
                                b_val in active_ctx.variables
                                or b_val in getattr(active_ctx, "pipeline_vars", set())
                            ):
                                pipeline_vars.add(b_val)

            if hasattr(active_ctx, "variables") and isinstance(active_ctx.variables, dict):
                for v_name in active_ctx.variables.keys():
                    if isinstance(v_name, str):
                        pipeline_vars.add(v_name)

            if hasattr(active_ctx, "scope_variables") and isinstance(active_ctx.scope_variables, dict):
                for v_name in active_ctx.scope_variables.keys():
                    if isinstance(v_name, str):
                        pipeline_vars.add(v_name)

            if hasattr(active_ctx, "runtime_aliases") and isinstance(active_ctx.runtime_aliases, dict):
                pipeline_vars.update(active_ctx.runtime_aliases.keys())
                pipeline_vars.update(active_ctx.runtime_aliases.values())

            prior_vars.difference_update(pipeline_vars)
            ctx_pipe_vars = getattr(active_ctx, "pipeline_vars", set()) if active_ctx else set()
            prior_vars = {v for v in prior_vars if not (isinstance(v, str) and (v in pipeline_vars or v in ctx_pipe_vars))}

        # Seed initial scope with root callable/operation names extracted from cell templates
        template_callables: Set[str] = set()
        source_cells = pipeline_bindings if "pipeline_bindings" in locals() and pipeline_bindings else (cells or [])
        for c in source_cells:
            cell_obj = c[0] if isinstance(c, tuple) else c
            tpl = getattr(cell_obj, "code_template", "") or ""
            if tpl:
                template_callables.update(_extract_template_callables(tpl))

        prior_vars.update(template_callables)
        self._verify_emitted_ast_liveness(final_code, initial_scope=prior_vars, cells=source_cells)
        self._refuse_non_callable_calls(final_code)
        return final_code

    @staticmethod
    def _reconcile_imports(code: str) -> str:
        """AST-level alias reconciliation: every Name root used as module/attribute
        must be bound by an import or local variable in the script; inject missing ones."""
        if not code or not code.strip():
            return code
        if not get_active_language_backend().is_python_ast:
            return code
        tree, _ = _cached_ast_parse_ro(code, "exec")
        if tree is None:
            return code

        import builtins
        bound: Set[str] = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    bound.add(a.asname or a.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.AnnAssign):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.AugAssign):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.NamedExpr):
                if node.target:
                    for sub in ast.walk(node.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for sub in ast.walk(node.target):
                    if isinstance(sub, ast.Name):
                        bound.add(sub.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars:
                        for sub in ast.walk(item.optional_vars):
                            if isinstance(sub, ast.Name):
                                bound.add(sub.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                if hasattr(node, "args"):
                    for arg in node.args.args + getattr(node.args, "kwonlyargs", []) + getattr(node.args, "posonlyargs", []):
                        bound.add(arg.arg)
                    if node.args.vararg:
                        bound.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        bound.add(node.args.kwarg.arg)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in node.generators:
                    for sub in ast.walk(gen.target):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    bound.add(node.name)

        used_roots: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used_roots.add(node.value.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
                used_roots.add("Path")

        unbound_roots = sorted(used_roots - bound)
        if not unbound_roots:
            return code

        registered_aliases = TypeRegistry.get_instance().get_all_aliases()
        import importlib.util
        injected: List[str] = []
        for root in unbound_roots:
            root_clean = root.strip()
            root_lower = root_clean.lower()
            if root_clean == "Path":
                stmt = "from pathlib import Path"
                if stmt not in injected:
                    injected.append(stmt)
                continue
            if root_lower in registered_aliases:
                target_mod = registered_aliases[root_lower]
                if target_mod == root_clean:
                    stmt = f"import {root_clean}"
                else:
                    stmt = f"import {target_mod} as {root_clean}"
                if stmt not in injected:
                    injected.append(stmt)
            else:
                try:
                    spec = importlib.util.find_spec(root_clean)
                    if spec is not None:
                        stmt = f"import {root_clean}"
                        if stmt not in injected:
                            injected.append(stmt)
                except Exception as e:
                    logger.debug("suppressed: %s", e, exc_info=False)

        if not injected:
            return code

        header = "\n".join(injected)
        return header + "\n" + code

    @staticmethod
    def _verify_emitted_ast_liveness(code: str, initial_scope: Optional[Set[str]] = None, cells: Optional[List[Any]] = None) -> None:
        """AST-level dataflow liveness validation: every variable loaded in the script
        must have been previously assigned, imported, or present in builtins.
        Catches undefined variables (e.g. var_7, var_3) before code emission."""
        if not code or not code.strip():
            return
        active_backend = get_active_language_backend()
        if not active_backend.is_python_ast:
            ok, err = active_backend.validate_syntax(code, allow_placeholders=False)
            if not ok:
                raise EmissionLivenessError(f"Emitted code has invalid syntax: {err}")
            return
        tree, err_msg = _cached_ast_parse_ro(code, "exec")
        if tree is None:
            raise EmissionLivenessError(f"Emitted code has invalid syntax: {err_msg}")

        import builtins
        global_scope: Set[str] = set(dir(builtins))
        global_scope.update({
            "__file__", "__name__", "__doc__", "__package__",
            "__spec__", "__path__", "__loader__", "__annotations__"
        })
        if initial_scope:
            global_scope.update(initial_scope)

        class ScopeChecker(ast.NodeVisitor):
            def __init__(self):
                self.scopes: List[Set[str]] = [set(global_scope)]
                self.undefined: List[Tuple[str, int]] = []

            def _is_bound(self, name: str) -> bool:
                return any(name in scope for scope in reversed(self.scopes))

            def _bind(self, name: str) -> None:
                self.scopes[-1].add(name)

            def _extract_targets(self, target_node, target_set: Set[str]):
                for n in ast.walk(target_node):
                    if isinstance(n, ast.Name):
                        if isinstance(n.ctx, ast.Store):
                            target_set.add(n.id)
                        elif isinstance(n.ctx, ast.Load):
                            self.visit(n)

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    self._bind(alias.asname or alias.name.split(".")[0])

            def visit_ImportFrom(self, node: ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        self._bind(alias.asname or alias.name)

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._bind(node.name)
                for dec in node.decorator_list:
                    self.visit(dec)
                for d in getattr(node.args, "defaults", []) or ():
                    self.visit(d)
                for d in getattr(node.args, "kw_defaults", []) or ():
                    if d is not None:
                        self.visit(d)
                func_scope: Set[str] = set()
                all_args = getattr(node.args, "posonlyargs", []) + node.args.args + getattr(node.args, "kwonlyargs", [])
                for arg in all_args:
                    func_scope.add(arg.arg)
                if node.args.vararg:
                    func_scope.add(node.args.vararg.arg)
                if node.args.kwarg:
                    func_scope.add(node.args.kwarg.arg)
                self.scopes.append(func_scope)
                for stmt in node.body:
                    self.visit(stmt)
                self.scopes.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node: ast.ClassDef):
                self._bind(node.name)
                for dec in node.decorator_list:
                    self.visit(dec)
                for base in node.bases:
                    self.visit(base)
                for kw in node.keywords:
                    self.visit(kw)
                class_scope: Set[str] = set()
                self.scopes.append(class_scope)
                for stmt in node.body:
                    self.visit(stmt)
                self.scopes.pop()

            def visit_Lambda(self, node: ast.Lambda):
                for d in getattr(node.args, "defaults", []) or ():
                    self.visit(d)
                for d in getattr(node.args, "kw_defaults", []) or ():
                    if d is not None:
                        self.visit(d)
                lam_scope: Set[str] = set()
                all_args = getattr(node.args, "posonlyargs", []) + node.args.args + getattr(node.args, "kwonlyargs", [])
                for arg in all_args:
                    lam_scope.add(arg.arg)
                if node.args.vararg:
                    lam_scope.add(node.args.vararg.arg)
                if node.args.kwarg:
                    lam_scope.add(node.args.kwarg.arg)
                self.scopes.append(lam_scope)
                self.visit(node.body)
                self.scopes.pop()

            def visit_ListComp(self, node):
                self._visit_comp(node)

            def visit_SetComp(self, node):
                self._visit_comp(node)

            def visit_DictComp(self, node):
                self._visit_comp(node)

            def visit_GeneratorExp(self, node):
                self._visit_comp(node)

            def _visit_comp(self, node):
                comp_scope: Set[str] = set()
                self.scopes.append(comp_scope)
                for gen in node.generators:
                    self.visit(gen.iter)
                    self._extract_targets(gen.target, comp_scope)
                    for if_expr in gen.ifs:
                        self.visit(if_expr)
                if isinstance(node, ast.DictComp):
                    self.visit(node.key)
                    self.visit(node.value)
                else:
                    self.visit(node.elt)
                self.scopes.pop()

            def visit_Assign(self, node: ast.Assign):
                self.visit(node.value)
                for target in node.targets:
                    self._extract_targets(target, self.scopes[-1])

            def visit_AnnAssign(self, node: ast.AnnAssign):
                if node.value:
                    self.visit(node.value)
                if node.target:
                    self._extract_targets(node.target, self.scopes[-1])

            def visit_AugAssign(self, node: ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    if not self._is_bound(node.target.id):
                        self.undefined.append((node.target.id, getattr(node, "lineno", 0)))
                else:
                    self.visit(node.target)
                self.visit(node.value)

            def visit_NamedExpr(self, node: ast.NamedExpr):
                self.visit(node.value)
                self._extract_targets(node.target, self.scopes[-1])

            def visit_For(self, node: ast.For):
                self.visit(node.iter)
                self._extract_targets(node.target, self.scopes[-1])
                for stmt in node.body:
                    self.visit(stmt)
                for stmt in node.orelse:
                    self.visit(stmt)

            def visit_AsyncFor(self, node: ast.AsyncFor):
                self.visit_For(node)

            def visit_With(self, node: ast.With):
                for item in node.items:
                    self.visit(item.context_expr)
                    if item.optional_vars:
                        self._extract_targets(item.optional_vars, self.scopes[-1])
                for stmt in node.body:
                    self.visit(stmt)

            def visit_AsyncWith(self, node: ast.AsyncWith):
                self.visit_With(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler):
                if node.type:
                    self.visit(node.type)
                if node.name:
                    self._bind(node.name)
                for stmt in node.body:
                    self.visit(stmt)

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    if not self._is_bound(node.id):
                        self.undefined.append((node.id, getattr(node, "lineno", 0)))
                elif isinstance(node.ctx, ast.Store):
                    self._bind(node.id)

        checker = ScopeChecker()
        checker.visit(tree)
        if checker.undefined:
            details = [f"'{name}' (line {line})" for name, line in checker.undefined]
            raise EmissionLivenessError(
                f"AST liveness check failed: undefined variables loaded before definition: {', '.join(details)}"
            )

    def unify_and_emit(
        self,
        cells: List[Cell],
        prompt: str = "",
        intent_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Main synthesis entrypoint."""
        self.context = ExecutionContext(prompt=prompt)
        if (intent_data is None or not intent_data) and prompt:
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm = ModelManager.get_instance()
                if mm.has_semantic_compiler():
                    intent_data = mm.compile_semantic_intent(prompt)
            except Exception:
                intent_data = None

        if intent_data and isinstance(intent_data, dict):
            self.context.intent_data = dict(intent_data)
            if hasattr(self.context, "blackboard") and self.context.blackboard is not None:
                self.context.blackboard.lock_from_intent(intent_data)
                if self.context.blackboard.locked_by_neural:
                    if self.context.blackboard.target_sink:
                        self.context.parameters["target_sink"] = self.context.blackboard.target_sink
            if intent_data.get("operations") and isinstance(intent_data["operations"], list):
                self.context.operations = [str(op).replace("_", " ").strip() for op in intent_data["operations"] if str(op).strip()]
            if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
                self.context.hyperparameters = dict(intent_data["hyperparameters"])
            if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
                self.context.llm_slots = dict(intent_data["slots"])
            if intent_data.get("variables") and isinstance(intent_data["variables"], list):
                self.context.variables_spec = [str(v).strip() for v in intent_data["variables"] if str(v).strip()]
                self.context.llm_slots.setdefault("variables", self.context.variables_spec)
            if hasattr(self.context, "sync_llm_slots"):
                self.context.sync_llm_slots()
            tgt = intent_data.get("target_sink") or intent_data.get("target_key") or intent_data.get("deliverable")
            if tgt:
                tgt_clean = str(tgt).strip()
                if tgt_clean:
                    self.context.target_key = tgt_clean
                    self.context.target_literals.add(tgt_clean.lower())
                    self.context.target_literals.add(tgt_clean)
                    self.context.parameters["target_sink"] = tgt_clean
            # Profile S (Semantic Compiler): literals extracted by the typed
            # IR compiler become declared context parameters — deterministic
            # binding without re-scanning the prompt.
            if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
                for p_key, p_val in intent_data["parameters"].items():
                    if p_key:
                        self.context.parameters.setdefault(str(p_key), p_val)
            if intent_data.get("source_files"):
                self.context.parameters["source_uris"] = list(intent_data["source_files"])
            if intent_data.get("dest_files"):
                self.context.parameters["dest_uris"] = list(intent_data["dest_files"])
        if not self.context.parameters.get("source_uris") and self.context.source_files:
            self.context.parameters["source_uris"] = list(self.context.source_files)
        if not self.context.parameters.get("dest_uris") and self.context.dest_files:
            self.context.parameters["dest_uris"] = list(self.context.dest_files)
        _can_slot = False
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            _mm_e = ModelManager.get_instance()
            _can_slot = bool(getattr(_mm_e, "profile", None) and _mm_e.can_extract_slots())
        except Exception:
            _can_slot = False
        extract_llm = _can_slot or (os.environ.get("NSTL_LLM_FEEDBACK") == "1") or bool(getattr(self.context, "llm_feedback", False))
        return self.emit_code(cells, self.context, extract_llm_slots=extract_llm)

    def build_verification_contract(
        self,
        pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
        ctx: Optional[ExecutionContext] = None,
        prompt: str = ""
    ) -> VerificationContract:
        """
        Builds a structured VerificationContract capturing Phase-1 postconditions
        and terminal node intent rules for GEVR sandbox execution.
        """
        try:
            try:
                from algebraic_effects import build_universal_verification_contract
            except (ImportError, ValueError):
                from .algebraic_effects import build_universal_verification_contract
            contract = build_universal_verification_contract(pipeline_bindings, VerificationContract)
        except Exception as _alg_exc:
            logger.debug("[UNIFICATION] Algebraic contract build fallback: %s", _alg_exc)
            contract = VerificationContract()

        # Phase-1 Cell Postconditions from schema declarations
        for cell, bindings in pipeline_bindings:
            cell_inputs = getattr(cell, "inputs", {}) or {}
            bound_literals: Dict[str, Any] = {}
            for in_pn in cell_inputs:
                b_val = bindings.get(in_pn)
                if b_val is None or is_unresolved_value(b_val):
                    continue
                if isinstance(b_val, (int, float, bool, tuple, list)):
                    bound_literals[in_pn] = b_val
                elif isinstance(b_val, str):
                    b_s = b_val.strip()
                    if b_s:
                        try:
                            parsed_arg = ast.literal_eval(b_s)
                            if isinstance(parsed_arg, (int, float, bool, str, tuple, list, type(None))):
                                bound_literals[in_pn] = parsed_arg
                        except Exception:
                            pass

            for post in (getattr(cell, "postconditions", []) or []):
                p_obj = post
                if not hasattr(p_obj, "property") and not hasattr(p_obj, "expression"):
                    try:
                        from .schema import ConditionPredicate
                        p_obj = ConditionPredicate.from_any(post)
                    except Exception:
                        try:
                            from schema import ConditionPredicate
                            p_obj = ConditionPredicate.from_any(post)
                        except Exception:
                            p_obj = post

                target_name = getattr(p_obj, "target", None) or "output_var"
                target_var = bindings.get(target_name)
                if not target_var:
                    target_var = bindings.get("output_var") or (bindings.get(cell.primary_output.name) if cell.primary_output else None)
                if target_var:
                    raw_val = getattr(p_obj, "value", None)
                    expected_ref_var: Optional[str] = None
                    if isinstance(raw_val, str) and raw_val in cell_inputs and raw_val in bindings:
                        bound_arg = bindings[raw_val]
                        if isinstance(bound_arg, str):
                            ba_s = bound_arg.strip()
                            try:
                                raw_val = ast.literal_eval(ba_s)
                            except Exception:
                                if ba_s.isidentifier():
                                    expected_ref_var = ba_s
                        else:
                            raw_val = bound_arg

                    check_entry: Dict[str, Any] = {
                        "cell_id": cell.cell_id,
                        "target_var": target_var,
                        "target_port": target_name,
                        "property": getattr(p_obj, "property", None),
                        "operator": getattr(p_obj, "operator", "=="),
                        "value": raw_val,
                        "expression": getattr(p_obj, "expression", None),
                        "description": getattr(p_obj, "description", None),
                    }
                    if expected_ref_var:
                        check_entry["expected_ref_var"] = expected_ref_var
                    if bound_literals:
                        check_entry["context_params"] = dict(bound_literals)
                    if check_entry not in contract.cell_checks:
                        contract.cell_checks.append(check_entry)

        return contract

    @classmethod
    def unify_cell(cls, *args, **kwargs) -> str:
        """Single-cell unification helper for backwards-compatibility with tests."""
        gate = cls()
        context = None
        cell = None
        for a in args:
            if isinstance(a, ExecutionContext):
                context = a
            elif isinstance(a, Cell):
                cell = a
            elif isinstance(a, str) and context is None:
                context = ExecutionContext(a)
        if kwargs:
            context = kwargs.get("ctx", kwargs.get("context", context))
            cell = kwargs.get("cell", cell)
        if cell is None:
            # Fallback for positional if neither matched isinstance
            if len(args) == 2:
                if isinstance(args[1], Cell):
                    context, cell = args[0], args[1]
                else:
                    cell, context = args[0], args[1]
            else:
                raise ValueError("unify_cell requires a Cell")
        ctx = context if isinstance(context, ExecutionContext) else ExecutionContext(str(context or ""))
        res = gate.unify_pipeline([cell], ctx)
        if res.is_bottom():
            reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
            raise ValueError(f"Unification Failed: {reason}")
        pipeline_bindings = res.value
        return gate.emit_code(pipeline_bindings, ctx)

    @classmethod
    def resolve_imports(cls, code_text: str, context: Any = None, chain_nodes: Any = None) -> Union[str, List[str]]:
        """Collects declared dependencies strictly from chain_nodes without domain hardcodes."""
        active_backend = get_active_language_backend()
        imports = set()
        if chain_nodes:
            for node in chain_nodes:
                for dep in getattr(node, "dependencies", []):
                    dep_str = dep.strip()
                    if dep_str:
                        stmt = active_backend.format_dependency_statement(dep_str)
                        if stmt:
                            imports.add(stmt)
        import_block = "\n".join(sorted(list(imports)))
        if code_text:
            if import_block:
                return f"{import_block}\n\n{code_text}".strip()
            return code_text.strip()
        return sorted(list(imports))


_semantic_var_name = UnificationGate._semantic_var_name

# =====================================================================
# Compatibility Helpers and Error Classes
# =====================================================================

class UnificationFailure(Exception):
    """Raised when monadic unification fails to find a valid substitution."""
    pass


class UnresolvedPlaceholderError(UnificationFailure):
    """Raised when a placeholder cannot be resolved. `ports` lists (cell_id, port_name) pairs when known."""

    def __init__(
        self,
        message: str = "",
        ports: Optional[List[Tuple[str, str]]] = None,
        typed_holes: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.ports: List[Tuple[str, str]] = list(ports or [])
        self.typed_holes: List[Any] = list(typed_holes or [])


class EmissionLivenessError(UnificationFailure):
    """
    Raised when emitted code is not well-formed (undefined names, invalid syntax).
    This is an ENGINE DEFECT in the emitter, not a theoretical limit of the lattice, so reports must not
    file it under "architectural"; `is_engine_defect` lets callers classify it honestly.
    """
    is_engine_defect = True


def types_unify(tau_expected: str, tau_actual: str) -> bool:
    """Verifies whether two types unify under the STRATA poset type system."""
    # unify(producer, consumer): the actual (produced) type must flow into the expected one
    producer = TypeTerm.from_string(tau_actual)
    consumer = TypeTerm.from_string(tau_expected)
    return unify(producer, consumer) is not None


def assert_placeholders_resolved(template: str, bindings: Optional[Dict[str, Any]] = None) -> None:
    """Asserts that all {placeholder} slots in a template are bound without regex."""
    if bindings:
        for k, v in bindings.items():
            template = template.replace(f"{{{k}}}", str(v))
    remaining = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == '{':
            j = template.find('}', i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier():
                    remaining.append(inner)
                i = j + 1
                continue
        i += 1
    if remaining:
        raise UnresolvedPlaceholderError(f"Unbound placeholders remaining: {remaining}")


def _adjacency_rank(cand_clause: Optional[int], producer_clause: Optional[int]) -> Tuple[int, int]:
    """Clause adjacency as an ordinal (R5-7): same clause > previous clause > earlier clauses (closer is
    better) > unknown > a producer from a LATER clause (backward flow). Replaces additive 35/30/25-5d/-50."""
    if cand_clause is None or producer_clause is None:
        return (1, -5)
    d = cand_clause - producer_clause
    if d == 0:
        return (3, 0)
    if d == 1:
        return (2, 0)
    if d > 1:
        return (1, -min(d, 5))
    return (-1, 0)


def _wire_rank(p_type: str, p_state: str, in_role: Any, w_type: str, w_state: str, w_role: Any,
               cand_clause: Optional[int], producer_clause: Optional[int], recency: int, aligned: int = 0) -> tuple:
    """Lexicographic preference between candidate wires into one port. Priority order:
    clause adjacency, exact carrier type, recency (fresher producer), then exact state, role, and alignment.
    A fresher producer is never outweighed by an exact-state bonus."""
    return (
        _adjacency_rank(cand_clause, producer_clause),
        1 if (p_type and w_type and p_type == w_type) else 0,
        recency,
        1 if (p_state and w_state and p_state == w_state) else 0,
        1 if (in_role and w_role and in_role == w_role) else 0,
        aligned,
    )


def unify_cell_with_scope(
    cand: Cell,
    available_cells: List[Cell],
    sigma: Optional[Substitution] = None,
    context: Optional[ExecutionContext] = None
) -> Optional[Tuple[Substitution, Dict[str, Tuple[Cell, str]]]]:
    """
    Scope-Level Robinson Unification for Dataflow DAGs.
    Verifies that all required input ports of `cand` are satisfiable from:
      1. Available output ports of cells in `available_cells`
      2. In-scope variables in `context` (if provided)
      3. Default values / prompt literals
    Returns (updated_substitution, port_bindings) where port_bindings maps
    port_name -> (producer_cell, producer_out_port_name), or None if unsatisfiable.
    """
    sub = Substitution(sigma.mappings if sigma else {})
    port_bindings: Dict[str, Tuple[Cell, str]] = {}
    bound_count = 0

    candidates = []
    # Collect candidate wire matches across all available DAG output ports
    for p_name, p_sig in cand.inputs.items():
        in_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
        p_sig_inner = getattr(p_sig, "signature", p_sig)
        p_type = str(getattr(p_sig_inner, "type_name", "") or "").lower()
        p_state = str(getattr(p_sig_inner, "state", "") or "").lower()

        # 1. Available cells in DAG history
        for idx_c, earlier_cell in enumerate(reversed(available_cells)):
            if _is_sink_cell(earlier_cell):
                continue
            cand_pres = getattr(cand, "preconditions", None) or []
            earlier_posts = getattr(earlier_cell, "postconditions", None) or []
            def _c_prop(c, key, default=None):
                if isinstance(c, dict):
                    return c.get(key, default)
                return getattr(c, key, default)

            _vpm_s1 = getattr(context, "var_port_map", None) if context is not None else None
            _ctx_vars_s1 = getattr(context, "variables", None) if context is not None else None
            _ctx_cons_s1 = (getattr(context, "consumed_vars", None) or set()) if context is not None else set()
            _ctx_sup_s1 = (getattr(context, "superseded_vars", None) or set()) if context is not None else set()
            for out_name, raw_out_sig in earlier_cell.outputs.items():
                _s1_vn = _vpm_s1.get((id(earlier_cell), out_name)) if isinstance(_vpm_s1, dict) else None
                out_sig = raw_out_sig
                if _s1_vn and isinstance(_ctx_vars_s1, dict) and _s1_vn in _ctx_vars_s1:
                    _live_entry = _ctx_vars_s1[_s1_vn]
                    if isinstance(_live_entry, tuple) and _live_entry[0] is not None:
                        out_sig = _live_entry[0]
                if cand_pres:
                    e_sig = getattr(out_sig, "signature", out_sig)
                    e_state = getattr(e_sig, "state", None) or getattr(out_sig, "state", None)
                    earlier_state_props = TypeRegistry.get_instance().get_state_properties(e_state) if e_state else {}
                    prop_conflict = False
                    for pre in cand_pres:
                        pre_target = _c_prop(pre, "target")
                        if pre_target and pre_target != p_name:
                            continue
                        p_prop = _c_prop(pre, "property")
                        p_op = _c_prop(pre, "operator")
                        p_val = _c_prop(pre, "value")
                        if p_prop is not None:
                            for q in earlier_posts:
                                if _c_prop(q, "operator") == "==" and _c_prop(q, "property") == p_prop:
                                    if p_op == "==" and _c_prop(q, "value") != p_val:
                                        prop_conflict = True
                                        break
                                    elif p_op == "!=" and _c_prop(q, "value") == p_val:
                                        prop_conflict = True
                                        break
                            if prop_conflict:
                                break
                            if p_prop in earlier_state_props:
                                known_val = earlier_state_props[p_prop]
                                if known_val is None and p_prop == "ndim":
                                    known_val = getattr(out_sig, "canonical_rank", None)
                                    if known_val is None:
                                        known_val = getattr(out_sig, "rank", None)
                                if known_val is not None:
                                    if p_op == "==" and known_val != p_val:
                                        prop_conflict = True
                                        break
                                    elif p_op == "!=" and known_val == p_val:
                                        prop_conflict = True
                                        break
                    if prop_conflict:
                        continue
                if not _shape_compatible(out_sig, p_sig, producer_cell=earlier_cell, consumer_cell=cand):
                    continue
                s_wire = unify(out_sig.signature, p_sig.signature, sub)
                if s_wire is not None:
                    out_sig_inner = getattr(out_sig, "signature", out_sig)
                    out_type = str(getattr(out_sig_inner, "type_name", "") or "").lower()
                    out_state = str(getattr(out_sig_inner, "state", "") or "").lower()
                    out_role = getattr(out_sig, "port_role", None) or getattr(out_sig, "derived_role", "")

                    cand_clause = getattr(cand, "matched_clause_idx", None)
                    earlier_clause = getattr(earlier_cell, "matched_clause_idx", None)

                    # Multi-input positional alignment: if the consumer has several inputs accepting this
                    # carrier type, align them with ancestor producers in topological order.
                    carrier_in_ports = [
                        k for k, p in cand.inputs.items()
                        if unify(getattr(p, "signature", p), getattr(p_sig, "signature", p_sig), sub) is not None
                    ]
                    matching_ancestors = [
                        c for c in available_cells
                        if any(unify(o.signature, p_sig.signature, sub) is not None for o in c.outputs.values())
                    ]
                    aligned = 0
                    if len(carrier_in_ports) > 1 and len(matching_ancestors) > 1:
                        port_idx = carrier_in_ports.index(p_name) if p_name in carrier_in_ports else -1
                        anc_idx = matching_ancestors.index(earlier_cell) if earlier_cell in matching_ancestors else -1
                        if port_idx >= 0 and port_idx == anc_idx:
                            aligned = 1

                    comp_prio = 0 if getattr(p_sig, "complement_of_port", None) else 1
                    score = _wire_rank(p_type, p_state, in_role, out_type, out_state, out_role,
                                       cand_clause, earlier_clause, -idx_c, aligned)

                    s1_not_superseded = 0 if (_s1_vn and _s1_vn in _ctx_sup_s1) else 1
                    s1_unconsumed = 0 if (_s1_vn and _s1_vn in _ctx_cons_s1) else 1
                    wire_key = (id(earlier_cell), out_name)
                    candidates.append(((comp_prio, s1_not_superseded, s1_unconsumed, score), p_name, earlier_cell, out_name, wire_key, s_wire, out_sig.signature))

        # 2. Context variables if available
        if context is not None:
            avail_cell_ids = {id(c) for c in available_cells}
            _vpm_s2 = getattr(context, "var_port_map", None)
            _vpm_mapped_vars = {vn for (cid_k, _), vn in _vpm_s2.items() if cid_k in avail_cell_ids} if isinstance(_vpm_s2, dict) else set()
            _consumed_s2 = getattr(context, "consumed_vars", set()) or set()
            _sup_s2 = getattr(context, "superseded_vars", set()) or set()
            for v_name, (v_sig, _) in context.variables.items():
                v_cell = getattr(context, "var_sources", {}).get(v_name)
                if v_cell is not None and id(v_cell) in avail_cell_ids and (not _vpm_mapped_vars or v_name in _vpm_mapped_vars):
                    continue
                _rej = _wire_rejection(cand, p_name, p_sig, v_name, context, TypeRegistry.get_instance())
                if _rej is not None:
                    continue
                if not _shape_compatible(v_sig, p_sig, producer_cell=v_cell, consumer_cell=cand):
                    continue
                s_wire = unify(v_sig.signature, p_sig.signature, sub)
                if s_wire is not None:
                    v_sig_inner = getattr(v_sig, "signature", v_sig)
                    v_type = str(getattr(v_sig_inner, "type_name", "") or "").lower()
                    v_state = str(getattr(v_sig_inner, "state", "") or "").lower()
                    v_role = getattr(v_sig, "port_role", None) or getattr(v_sig, "derived_role", "")

                    cand_clause = getattr(cand, "matched_clause_idx", None)
                    v_clause = getattr(v_cell, "matched_clause_idx", None) if v_cell else None
                    _order = list(context.variables.keys())
                    comp_prio = 0 if getattr(p_sig, "complement_of_port", None) else 1
                    s2_not_superseded = 0 if v_name in _sup_s2 else 1
                    s2_unconsumed = 0 if v_name in _consumed_s2 else 1
                    score = _wire_rank(p_type, p_state, in_role, v_type, v_state, v_role,
                                       cand_clause, v_clause, _order.index(v_name) - len(_order))

                    v_cell = getattr(context, "var_sources", {}).get(v_name)
                    wire_key = v_name
                    candidates.append(((comp_prio, s2_not_superseded, s2_unconsumed, score), p_name, v_cell, v_name, wire_key, s_wire, v_sig.signature))

    # Deterministic greedy assignment with disjoint wire tracking (non-complement ports bind first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_ports = set()
    assigned_wires = set()
    partial_var_bindings: Dict[str, Any] = {}
    bound_wire_keys_by_port: Dict[str, Any] = {}
    reg_scope = TypeRegistry.get_instance()

    for _rank_key, p_name, prod_cell, out_name, wire_key, _s_wire, src_sig in candidates:
        if p_name in assigned_ports or wire_key in assigned_wires:
            continue
        p_sig_curr = cand.inputs.get(p_name)

        # Enforce declarative disjoint_from_port symmetrically even without context
        disjoint_conflict = False
        if p_sig_curr is not None:
            d_ref = getattr(p_sig_curr, "disjoint_from_port", None)
            if d_ref:
                dp_obj = _resolve_cell_port(cand, d_ref, prefer="inputs")
                dp_key = getattr(dp_obj, "name", None) if dp_obj is not None else d_ref
                if dp_key in bound_wire_keys_by_port and bound_wire_keys_by_port[dp_key] == wire_key:
                    disjoint_conflict = True
            if not disjoint_conflict:
                for other_n, other_p in (cand.inputs or {}).items():
                    other_d = getattr(other_p, "disjoint_from_port", None)
                    if other_d:
                        resolved_d = _resolve_cell_port(cand, other_d, prefer="inputs")
                        if resolved_d is p_sig_curr or getattr(resolved_d, "name", None) == p_name or other_d == p_name:
                            if other_n in bound_wire_keys_by_port and bound_wire_keys_by_port[other_n] == wire_key:
                                disjoint_conflict = True
                                break
        if disjoint_conflict:
            continue

        bound_v: Optional[str] = None
        if context is not None:
            if isinstance(wire_key, str) and wire_key in getattr(context, "variables", {}):
                bound_v = wire_key
            elif prod_cell is not None:
                v_pmap = getattr(context, "var_port_map", None)
                if isinstance(v_pmap, dict):
                    bound_v = v_pmap.get((id(prod_cell), out_name))
                if bound_v is None:
                    matching_vs = [
                        vn for vn, vc in (getattr(context, "var_sources", None) or {}).items()
                        if vc is prod_cell
                    ]
                    if len(matching_vs) == 1:
                        bound_v = matching_vs[0]
                    elif len(matching_vs) > 1 and out_name:
                        out_l = str(out_name).lower()
                        bound_v = next((vn for vn in matching_vs if out_l in vn.lower()), None)
            if bound_v is not None:
                _rej_pair = _wire_rejection(
                    cand, p_name, p_sig_curr, bound_v, context, reg_scope,
                    cell_bindings=partial_var_bindings,
                )
                if _rej_pair is not None:
                    continue

        s_wire_curr = unify(src_sig, p_sig_curr.signature, sub) if p_sig_curr is not None else _s_wire
        if s_wire_curr is None:
            continue

        assigned_ports.add(p_name)
        assigned_wires.add(wire_key)
        bound_wire_keys_by_port[p_name] = wire_key
        if bound_v is not None:
            partial_var_bindings[p_name] = bound_v
        port_bindings[p_name] = (prod_cell, out_name)
        sub = s_wire_curr
        bound_count += 1

    # Check remaining unbound ports against defaults / literals / projective & complement carriers
    for p_name, p_sig in cand.inputs.items():
        if p_name in assigned_ports:
            continue
        is_required = getattr(p_sig, "required", False)
        default_val = getattr(p_sig, "default_value", None)
        if not is_required:
            continue
        registry = TypeRegistry.get_instance()
        is_key_or_comp = _is_key_or_complement_port(cand, p_sig, registry)
        if default_val is not None:
            if not (is_key_or_comp and str(default_val).strip("'\"") in ("", "[]", "()")):
                continue
        desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
        strict_roles = registry.get_roles_for_category("strict") | registry.get_roles_for_category("estimator")
        c_role = str(getattr(cand, "node_role", "") or "").lower()
        c_type = str(getattr(cand, "node_type", "") or "").lower()
        is_model_cand = (
            c_role in strict_roles
            or c_type in strict_roles
        )
        has_explicit_port_role = bool(getattr(p_sig, "port_role", None)) or bool(getattr(p_sig, "binds", None))
        p_role = str(getattr(p_sig, "port_role", None) or (getattr(p_sig, "derived_role", "") if is_model_cand else "") or "").lower()
        p_binds = str(getattr(p_sig, "binds", "") or "").lower()
        rec_names = registry.get_roles_for_category("receiver")
        carrier_binds = registry.get_roles_for_category("receiver") | registry.get_roles_for_category("data")
        projective_roles = (
            registry.get_roles_for_category("projection")
            | set(registry.get_projection_tokens() or ())
            | registry.get_roles_for_category("feature")
            | registry.get_roles_for_category("target")
            | registry.get_roles_for_category("key")
        )
        has_comp_decl = bool(getattr(p_sig, "complement_of_port", None))
        is_proj_port = bool(((is_model_cand or has_explicit_port_role) and p_role in projective_roles) or has_comp_decl)
        is_receiver = (
            not is_proj_port
            and (
                p_name in rec_names
                or p_role in strict_roles
                or p_binds in carrier_binds
                or any(tok in str(desc).lower() for tok in (carrier_binds | rec_names))
            )
        )
        satisfied = False
        if not is_receiver:
            t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
            is_carrier_type = bool(
                registry.is_composite_carrier(t_name)
                or registry.is_abstract_carrier(t_name)
                or registry.is_subtype(t_name, "tensor")
            )
            if _lattice_is_path_port(p_sig):
                satisfied = True
            elif (
                not is_key_or_comp
                and not is_proj_port
                and not is_carrier_type
                and (
                    registry.is_subtype(t_name, "str")
                    or registry.is_subtype(t_name, "int")
                    or registry.is_subtype(t_name, "float")
                    or registry.is_subtype(t_name, "bool")
                    or registry.is_subtype(t_name, "logical")
                    or registry.is_subtype(t_name, "scalar")
                )
            ):
                satisfied = True
            elif is_key_or_comp and not is_carrier_type:
                if context is None:
                    satisfied = True
                else:
                    has_lit = bool(
                        any(
                            k in ("quoted_str", "identifier")
                            for _, k, _ in (getattr(context, "ordered_literals", None) or ())
                        )
                        or (p_name in (getattr(context, "parameters", None) or {}))
                        or bool(getattr(context, "target_literals", None))
                        or bool(getattr(context, "role_bound_keys", None))
                        or any(bool(v) for v in (getattr(context, "var_restricted_origins", None) or {}).values())
                        or (
                            str(getattr(cand, "optic_role", "") or "").strip().lower() == "setter"
                            and any(
                                not fr.get("closed") and fr.get("key_expr")
                                for fr in (getattr(context, "optic_frames", None) or ())
                            )
                        )
                    )
                    if has_lit:
                        satisfied = True
            else:
                if is_proj_port:
                    can_project_type = bool(
                        is_carrier_type
                        or registry.is_subtype(t_name, "scalar")
                        or not t_name
                    )
                    if can_project_type:
                        has_carrier = any(
                            registry.is_composite_carrier(str(getattr(out_s.signature, "type_name", "")).lower())
                            or registry.is_abstract_carrier(str(getattr(out_s.signature, "type_name", "")).lower())
                            for prev_c in available_cells
                            for out_s in prev_c.outputs.values()
                        )
                        if not has_carrier and context is not None:
                            has_carrier = any(
                                registry.is_composite_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                                or registry.is_abstract_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                                for v_sig, _ in getattr(context, "variables", {}).values()
                            )
                        if has_carrier:
                            satisfied = True
                            bound_count += 1
        if not satisfied:
            return None

    is_macro_or_comb = (
        getattr(cand, "node_type", "") in ("constructor", "macro", "combinator")
        or str(getattr(cand, "node_role", "") or "").strip().lower() in ("macro", "combinator")
        or bool(getattr(cand, "slots", None))
    )
    if cand.inputs and bound_count == 0 and getattr(cand, "stage", None) != 1 and not is_macro_or_comb:
        return None

    return sub, port_bindings


# -----------------------------------------------------------------------------
# Static Value Property Contract Verification
# -----------------------------------------------------------------------------

Props = Dict[str, Tuple[Any, str, str]]   # property -> (value, producing cell id, description)


def _prop_const(node: ast.AST) -> Optional[Any]:
    return node.value if isinstance(node, ast.Constant) else None


@functools.lru_cache(maxsize=1024)
def _prop_parse_expr(expr_str: str) -> Optional[ast.AST]:
    tree, _ = _cached_ast_parse_ro(expr_str, "eval")
    return tree.body if tree is not None else None


def _prop_known(expr: Any, props: Dict[str, Props], member_props: Dict[Tuple[str, Any], Props]) -> Props:
    tree = _prop_parse_expr(str(expr))
    if tree is None:
        return {}
    return _prop_eval(tree, props, member_props)


def _prop_eval(node: ast.AST, props: Dict[str, Props], member_props: Dict[Tuple[str, Any], Props]) -> Props:
    if isinstance(node, ast.Name):
        return dict(props.get(node.id, {}))
    if isinstance(node, ast.Call):
        if len(node.args) == 1 and not node.keywords:
            arg_props = _prop_eval(node.args[0], props, member_props)
            if arg_props:
                return arg_props
        if isinstance(node.func, ast.Attribute):
            return _prop_eval(node.func.value, props, member_props)
    if isinstance(node, ast.Attribute):
        return _prop_eval(node.value, props, member_props)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        sl = node.slice
        keys: List[Any]
        if isinstance(sl, ast.List):
            keys = [_prop_const(e) for e in sl.elts]
        else:
            keys = [_prop_const(sl)]
        if not keys or any(c is None for c in keys):
            return {}
        per_mem = [member_props.get((node.value.id, c), {}) for c in keys]
        out: Props = {}
        for prop in set().union(*[set(p) for p in per_mem]):
            if all(prop in p for p in per_mem) and len({repr(p[prop][0]) for p in per_mem}) == 1:
                out[prop] = per_mem[0][prop]
        return out
    return {}


def _prop_cond_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Extracts condition field from either a dict or a ConditionPredicate/Pydantic object."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _extract_root_var_from_expr(expr_str: str) -> str:
    s = expr_str.strip()
    if s.isidentifier():
        return s
    try:
        node = ast.parse(s, mode="eval").body
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
    except Exception:
        pass
    return s


def analyze_property_contracts(
    pipeline_bindings: List[Tuple[Any, Dict[str, Any]]], orchestrator: Any = None
) -> List[Dict[str, Any]]:
    """
    Static propagation of tree-declared value properties and invariants.
    Propagates known properties along the bound dataflow across all output ports
    and verifies consumer preconditions via both exact matching and InvariantEntailer.
    """
    try:
        from .lattice import InvariantEntailer
        reg = TypeRegistry.get_instance()
    except Exception:
        try:
            from lattice import TypeRegistry, InvariantEntailer
            reg = TypeRegistry.get_instance()
        except Exception:
            return []
    props: Dict[str, Props] = {}
    member_props: Dict[Tuple[str, Any], Props] = {}
    var_invariants: Dict[str, List[Any]] = {}
    issues: List[Dict[str, Any]] = []

    supported_ops = ("==", "!=", "<", "<=", ">", ">=", "in", "not in")

    def _collect_vc_preds(vc_obj: Any) -> List[Any]:
        out_p: List[Any] = []
        if isinstance(vc_obj, dict):
            for vk, vv in vc_obj.items():
                if isinstance(vv, dict):
                    for inner_op, inner_val in vv.items():
                        out_p.append((str(vk), str(inner_op), inner_val))
                else:
                    out_p.append((str(vk), "==", vv))
        return out_p

    for cell, bindings in pipeline_bindings:
        out = bindings.get("output_var")
        typed_bindings = getattr(cell, "typed_bindings", {}) or {}
        c_inputs = getattr(cell, "inputs", {}) or {}
        bound_in_names = [n for n in c_inputs if n in bindings]
        prim_in_obj = getattr(cell, "primary_input", None)
        prim_in_name = getattr(prim_in_obj, "name", None) if prim_in_obj is not None else None
        if prim_in_name and prim_in_name in bindings:
            default_in_port = prim_in_name
        elif len(bound_in_names) == 1:
            default_in_port = bound_in_names[0]
        else:
            default_in_port = None

        # 1. Consumer preconditions & input port invariants/constraints
        consumer_reqs: List[Tuple[str, str, str, Any]] = []
        for pre in (getattr(cell, "preconditions", None) or []):
            if isinstance(pre, str):
                if default_in_port:
                    for p_prop, p_op, p_val in InvariantEntailer.parse_predicate(pre):
                        if p_op in supported_ops and p_prop:
                            consumer_reqs.append((default_in_port, p_prop, p_op, p_val))
                continue
            op = _prop_cond_field(pre, "operator")
            if op not in supported_ops:
                continue
            port = _prop_cond_field(pre, "target")
            if port in (None, "input_var") and default_in_port:
                port = default_in_port
            prop = _prop_cond_field(pre, "property")
            want = _prop_cond_field(pre, "value")
            if port in bindings and prop:
                consumer_reqs.append((str(port), str(prop), str(op), want))

        for in_p_name, in_p_obj in c_inputs.items():
            if in_p_name not in bindings or in_p_obj is None:
                continue
            for raw_inv in (getattr(in_p_obj, "invariants", None) or ()):
                for p_prop, p_op, p_val in InvariantEntailer.parse_predicate(raw_inv):
                    if p_op in supported_ops and p_prop:
                        consumer_reqs.append((in_p_name, p_prop, p_op, p_val))
            in_vc = getattr(in_p_obj, "value_constraints", None)
            if in_vc:
                for p_prop, p_op, p_val in _collect_vc_preds(in_vc):
                    if p_op in supported_ops and p_prop:
                        consumer_reqs.append((in_p_name, p_prop, p_op, p_val))
            in_pb = getattr(in_p_obj, "property_bundle", None)
            pb_entries = list(getattr(in_pb, "predicates", None) or ()) + list(getattr(in_pb, "invariants", None) or ())
            for raw_pb_inv in pb_entries:
                for p_prop, p_op, p_val in InvariantEntailer.parse_predicate(raw_pb_inv):
                    if p_op in supported_ops and p_prop:
                        consumer_reqs.append((in_p_name, p_prop, p_op, p_val))

        # Verify every collected consumer requirement against KNOWN properties + InvariantEntailer
        for port, prop, op, want in consumer_reqs:
            if port not in bindings or not prop:
                continue
            have = None
            known_preds: List[Any] = []
            rec = typed_bindings.get(port)
            bound_expr = bindings[port]
            target_root = None
            if rec is not None and rec.root_var:
                target_root = rec.root_var
                if rec.kind in ("wire", "variable"):
                    have = props.get(rec.root_var, {}).get(prop)
                    known_preds.extend(var_invariants.get(rec.root_var, ()))
                elif rec.kind == "projection" and rec.projection:
                    k = rec.projection.get("key")
                    if isinstance(k, (list, tuple, set)):
                        per_mem = [member_props.get((rec.root_var, c), {}) for c in k]
                        have_props: Props = {}
                        for p in set().union(*[set(cp) for cp in per_mem]):
                            if all(p in cp for cp in per_mem) and len({repr(cp[p][0]) for cp in per_mem}) == 1:
                                have_props[p] = per_mem[0][p]
                        have = have_props.get(prop)
                    elif k is not None:
                        have = member_props.get((rec.root_var, k), {}).get(prop)
            else:
                if isinstance(bound_expr, str) and bound_expr.strip():
                    target_root = _extract_root_var_from_expr(bound_expr)
                    known_preds.extend(var_invariants.get(target_root, ()))
            known_dict = _prop_known(bound_expr, props, member_props)
            if have is None:
                have = known_dict.get(prop)
            for kp_name, kp_tuple in known_dict.items():
                if kp_tuple and kp_tuple[0] is not None:
                    known_preds.append((str(kp_name), "==", kp_tuple[0]))
            if target_root and target_root in props:
                for kp_name, kp_tuple in props[target_root].items():
                    if kp_tuple and kp_tuple[0] is not None:
                        known_preds.append((str(kp_name), "==", kp_tuple[0]))

            violated = False
            known_val = have[0] if (have is not None and have[0] is not None) else None
            established_by = have[1] if have is not None else "invariant"
            established_note = have[2] if have is not None else "declared_invariant"

            if op in ("==", "!=") and have is not None and have[0] is not None:
                satisfied = (have[0] == want) if op == "==" else (have[0] != want)
                if not satisfied:
                    violated = True
            if not violated and known_preds:
                prop_low = str(prop).strip().lower()
                constrained_props = set()
                relevant_preds: List[Any] = []
                for raw_kp in known_preds:
                    parsed_kps = InvariantEntailer.parse_predicate(raw_kp)
                    for pk_name, pk_op, pk_val in parsed_kps:
                        constrained_props.add(pk_name)
                        if pk_name == prop_low:
                            relevant_preds.append((pk_name, pk_op, pk_val))
                if prop_low in constrained_props:
                    if not InvariantEntailer.entails(relevant_preds, [(prop, op, want)], reg):
                        violated = True
                        if known_val is None and relevant_preds:
                            known_val = f"{relevant_preds[0][1]} {relevant_preds[0][2]}"
            if violated:
                suggestions: List[str] = []
                _orch_ref = orchestrator
                if _orch_ref is None:
                    try:
                        try:
                            from .lattice import LatticeOrchestrator as _LO
                        except (ImportError, ValueError):
                            from lattice import LatticeOrchestrator as _LO
                        _orch_ref = _LO.get_active_instance()
                    except Exception:
                        _orch_ref = None
                if _orch_ref is not None:
                    for c in getattr(_orch_ref, "loaded_cells", {}).values():
                        if any(_prop_cond_field(q, "property") == prop and _prop_cond_field(q, "operator") == "=="
                               and _prop_cond_field(q, "value") == want and _prop_cond_field(q, "target") in ("output_var", None)
                               for q in (getattr(c, "postconditions", None) or [])):
                            suggestions.append(c.cell_id)
                issues.append({"cell": cell.cell_id, "port": port, "property": prop, "required": want,
                               "required_operator": op, "known": known_val, "bound": str(bound_expr),
                               "established_by": established_by, "established_note": established_note,
                               "bridge_candidates": sorted(set(suggestions))[:6]})

        # Collect ALL bound output variables for this step (including multi-output cells)
        c_outputs = getattr(cell, "outputs", {}) or {}
        out_port_vars: List[Tuple[Optional[str], Optional[Any], str]] = []
        seen_out_vars: Set[str] = set()
        for o_name, o_port_obj in c_outputs.items():
            ov = bindings.get(o_name)
            if isinstance(ov, str) and ov.strip():
                ov_s = ov.strip()
                out_port_vars.append((o_name, o_port_obj, ov_s))
                seen_out_vars.add(ov_s)
        if isinstance(out, str) and out.strip() and out.strip() not in seen_out_vars:
            prim_o = getattr(cell, "primary_output", None)
            if prim_o is None and c_outputs:
                prim_o = c_outputs.get("output_var") or next(iter(c_outputs.values()))
            out_port_vars.append((getattr(prim_o, "name", None) or "output_var", prim_o, out.strip()))
            seen_out_vars.add(out.strip())

        if not out_port_vars:
            continue

        def _seed_output_port_props(o_port: Any) -> Tuple[Props, List[Any]]:
            m_props: Props = {}
            m_invs: List[Any] = []
            if o_port is not None:
                o_sig = getattr(o_port, "signature", o_port)
                o_state = getattr(o_sig, "state", None) or getattr(o_port, "state", None)
                if o_state:
                    state_props = reg.get_state_properties(o_state)
                    for pk, pv in state_props.items():
                        if pv is not None:
                            m_props[pk] = (pv, cell.cell_id, f"typestate:{o_state}")
                            m_invs.append((str(pk), "==", pv))
                p_rank = getattr(o_port, "rank", None)
                if p_rank is not None and "ndim" not in m_props:
                    m_props["ndim"] = (p_rank, cell.cell_id, "port:rank")
                    m_invs.append(("ndim", "==", p_rank))
                sh_c = getattr(o_port, "shape_contract", None)
                if isinstance(sh_c, dict) and "ndim" in sh_c and "ndim" not in m_props:
                    m_props["ndim"] = (sh_c["ndim"], cell.cell_id, "port:shape_contract")
                    m_invs.append(("ndim", "==", sh_c["ndim"]))
                p_invs = getattr(o_port, "invariants", None) or ()
                for pinv in p_invs:
                    m_invs.append(pinv)
                p_vc = getattr(o_port, "value_constraints", None)
                if p_vc:
                    m_invs.extend(_collect_vc_preds(p_vc))
                o_pb = getattr(o_port, "property_bundle", None)
                pb_entries = list(getattr(o_pb, "predicates", None) or ()) + list(getattr(o_pb, "invariants", None) or ())
                for pb_inv in pb_entries:
                    m_invs.append(pb_inv)
            return m_props, m_invs

        preserves_props: Props = {}
        preserves_invs: List[Any] = []
        for eff in (getattr(cell, "effects", None) or []):
            if isinstance(eff, str) and eff.startswith("preserves:"):
                prop = eff.split(":", 1)[1].strip()
                for port_name in getattr(cell, "inputs", {}):
                    if port_name in bindings:
                        kn = None
                        rec = typed_bindings.get(port_name)
                        if rec is not None:
                            if rec.kind in ("wire", "variable") and rec.root_var:
                                kn = props.get(rec.root_var, {})
                                for r_inv in var_invariants.get(rec.root_var, ()):
                                    for pk_n, pk_op, pk_v in InvariantEntailer.parse_predicate(r_inv):
                                        if pk_n == prop.lower():
                                            preserves_invs.append((pk_n, pk_op, pk_v))
                            elif rec.kind == "projection" and rec.root_var and rec.projection:
                                k = rec.projection.get("key")
                                if isinstance(k, str):
                                    kn = member_props.get((rec.root_var, k), {})
                        if kn is None:
                            kn = _prop_known(bindings[port_name], props, member_props)
                        if prop in kn and prop not in preserves_props:
                            preserves_props[prop] = kn[prop]
                            break

        explicit_post_props: Set[str] = set()
        mine: Props = {}
        prim_out_var_name = out if (isinstance(out, str) and out.strip()) else out_port_vars[0][2]

        for o_name, o_port_obj, ov_name in out_port_vars:
            o_mine, o_invs = _seed_output_port_props(o_port_obj)
            is_prim_or_single = (ov_name == prim_out_var_name or len(out_port_vars) == 1)
            for post in (getattr(cell, "postconditions", None) or []):
                if isinstance(post, str):
                    if is_prim_or_single:
                        for p_prop, p_op, p_val in InvariantEntailer.parse_predicate(post):
                            if p_op in supported_ops and p_prop:
                                o_invs.append((str(p_prop), str(p_op), p_val))
                                if p_op == "==":
                                    o_mine[p_prop] = (p_val, cell.cell_id, "")
                                    explicit_post_props.add(p_prop)
                    continue
                p_op = _prop_cond_field(post, "operator")
                p_target = _prop_cond_field(post, "target")
                p_prop = _prop_cond_field(post, "property")
                p_val = _prop_cond_field(post, "value")
                p_desc = str(_prop_cond_field(post, "description", "") or "")
                matches_this_out = (
                    (p_target == o_name)
                    or (p_target in ("output_var", None) and is_prim_or_single)
                )
                if matches_this_out and p_prop:
                    if p_op == "==":
                        o_mine[p_prop] = (p_val, cell.cell_id, p_desc)
                        explicit_post_props.add(p_prop)
                    if p_op in supported_ops:
                        o_invs.append((str(p_prop), str(p_op), p_val))
            for pk, pv in preserves_props.items():
                if pk not in o_mine:
                    o_mine[pk] = pv
            o_invs.extend(preserves_invs)
            if o_mine:
                props[ov_name] = {**props.get(ov_name, {}), **o_mine}
                if ov_name == prim_out_var_name:
                    mine.update(o_mine)
            if o_invs:
                var_invariants.setdefault(ov_name, []).extend(o_invs)

        if not out:
            continue

        # 3. member writes
        reg_mw = TypeRegistry.get_instance()
        key_binds = reg_mw.get_roles_for_category("key")
        val_binds = reg_mw.get_roles_for_category("assigned_value")
        carrier_binds = reg_mw.get_roles_for_category("receiver") | reg_mw.get_roles_for_category("data")
        key_port = next((n for n, p in getattr(cell, "inputs", {}).items() if getattr(p, "binds", None) in key_binds), None)
        val_port = next((n for n, p in getattr(cell, "inputs", {}).items() if getattr(p, "binds", None) in val_binds), None)
        carrier_ports = [n for n, p in getattr(cell, "inputs", {}).items() if getattr(p, "binds", None) in carrier_binds]
        for cp in carrier_ports:
            cp_rec = typed_bindings.get(cp)
            cp_var = cp_rec.root_var if (cp_rec and cp_rec.root_var) else str(bindings.get(cp))
            for (v, c), pr in list(member_props.items()):
                if v == cp_var:
                    member_props[(out, c)] = dict(pr)
        if key_port and val_port and key_port in bindings and val_port in bindings:
            k_rec = typed_bindings.get(key_port)
            col = str(k_rec.value_expr).strip().strip("'\"") if (k_rec and k_rec.value_expr) else str(bindings[key_port]).strip().strip("'\"")
            vp = None
            v_rec = typed_bindings.get(val_port)
            if v_rec and v_rec.root_var and v_rec.kind in ("wire", "variable"):
                vp = props.get(v_rec.root_var)
            if not vp:
                vp = _prop_known(bindings[val_port], props, member_props)
            if vp:
                member_props[(out, col)] = vp
            else:
                member_props.pop((out, col), None)

        # 4. Composite carrier-to-carrier cells keep what is known about their component properties
        in_carriers = []
        for n, p in getattr(cell, "inputs", {}).items():
            p_tn = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
            if n in bindings and (reg.is_composite_carrier(p_tn) or reg.is_abstract_carrier(p_tn)):
                rec_t = typed_bindings.get(n)
                in_carriers.append(rec_t.root_var if (rec_t and rec_t.root_var) else str(bindings.get(n)))
        if in_carriers and not (key_port and val_port):
            optic_r = str(getattr(cell, "optic_role", "") or "").strip().lower()
            if optic_r == "getter":
                gk_name = key_port
                if not gk_name:
                    gk_p = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs")
                    gk_name = getattr(gk_p, "name", None) if gk_p is not None else None
                selected_c_low: Set[str] = set()
                if gk_name and gk_name in bindings:
                    selected_c_low = {str(sk).strip().lower() for sk in _extract_keys_from_val(bindings[gk_name]) if str(sk).strip()}
                matched_mem: List[Props] = []
                for (v, c), pr in list(member_props.items()):
                    if v in in_carriers:
                        if selected_c_low and str(c).strip().lower() not in selected_c_low:
                            continue
                        copied_pr = dict(pr)
                        for ep in explicit_post_props:
                            copied_pr[ep] = mine[ep]
                        if (out, c) not in member_props:
                            member_props[(out, c)] = copied_pr
                        matched_mem.append(copied_pr)
                if len(matched_mem) == 1:
                    for prop_name, prop_val in matched_mem[0].items():
                        if prop_name not in mine:
                            props[out] = {**props.get(out, {}), prop_name: prop_val}
                elif len(matched_mem) > 1:
                    for prop_name in set().union(*[set(p) for p in matched_mem]):
                        if prop_name in mine:
                            continue
                        if all(prop_name in p for p in matched_mem) and len({repr(p[prop_name][0]) for p in matched_mem}) == 1:
                            props[out] = {**props.get(out, {}), prop_name: matched_mem[0][prop_name]}
            else:
                dropped_c_low: Set[str] = set()
                if optic_r == "complement":
                    dk_p = _resolve_cell_port(cell, getattr(cell, "key_port", None), prefer="inputs") or next(
                        (p for p in (getattr(cell, "inputs", {}) or {}).values() if getattr(p, "complement_of_port", None)),
                        None,
                    )
                    dk_name = getattr(dk_p, "name", None) if dk_p is not None else None
                    if dk_name and dk_name in bindings:
                        dropped_c_low = {str(dk).strip().lower() for dk in _extract_keys_from_val(bindings[dk_name]) if str(dk).strip()}
                for (v, c), pr in list(member_props.items()):
                    if v in in_carriers:
                        if str(c).strip().lower() in dropped_c_low:
                            continue
                        copied_pr = dict(pr)
                        for ep in explicit_post_props:
                            copied_pr[ep] = mine[ep]
                        if (out, c) not in member_props:
                            member_props[(out, c)] = copied_pr
                        for prop_name, prop_val in copied_pr.items():
                            if prop_name in mine:
                                continue
                            if isinstance(prop_val, (tuple, list)) and prop_val[0] is True:
                                props[out] = {**props.get(out, {}), prop_name: prop_val}
    return issues


analyze = analyze_property_contracts


