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
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port, _normalize_qualifiers,
        compute_port_residual,
    )
    from .tokenizer import CellTokenizer, normalize_token
except (ImportError, ValueError):
    from lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port, _normalize_qualifiers,
        compute_port_residual,
    )
    from tokenizer import CellTokenizer, normalize_token

try:
    from .diagnostics import DiagnosticTelemetry, record_diagnostic_exception
except (ImportError, ValueError):
    from diagnostics import DiagnosticTelemetry, record_diagnostic_exception

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
    if value is None or value is UNRESOLVED_PORT:
        return True
    try:
        s = str(value).strip().lower()
    except Exception:
        return False
    return s in ("none", "null", "<unresolved>", "<unbound>", "<unresolved_port>")


@functools.lru_cache(maxsize=1024)
def _parse_expr_ast(expr_str: str) -> Optional[ast.AST]:
    try:
        return ast.parse(expr_str, mode="eval").body
    except Exception:
        return None


_cached_col_proj_sig: Optional[Any] = None
_cached_cols_proj_sig: Optional[Any] = None
_cached_attr_proj_sigs: Dict[str, Any] = {}
_PROJECTION_CACHE_LOCK = threading.RLock()


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

    if bound_val is None or _is_unbound(bound_val) or bound_val is UNRESOLVED_PORT or (
        isinstance(bound_val, str) and bound_val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")
    ):
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
            resulting_signature=expected_sig,
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
            resulting_signature=getattr(sig, "signature", sig) if sig else expected_sig,
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

        # (A) Column selection: var['X']
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            global _cached_col_proj_sig
            with _PROJECTION_CACHE_LOCK:
                col_sig = _cached_col_proj_sig
            if col_sig is None:
                try:
                    try:
                        from .lattice import LatticeOrchestrator
                    except (ImportError, ValueError):
                        from lattice import LatticeOrchestrator
                    orch = LatticeOrchestrator.get_active_instance()
                    if orch:
                        for c in orch.cells:
                            if getattr(c, "projection", None) in _declared_role_semantics("single_projection_kinds"):
                                p_out = getattr(c, "primary_output", None) or next(iter(c.outputs.values()), None)
                                if p_out:
                                    col_sig = getattr(p_out, "signature", p_out)
                                    with _PROJECTION_CACHE_LOCK:
                                        _cached_col_proj_sig = col_sig
                                    break
                except Exception as _exc:
                    logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            if col_sig is None:
                col_sig = expected_sig or (
                    getattr(base_sig, "signature", base_sig) if base_sig else None
                ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": "subscript", "key": sl.value},
                resulting_signature=(expected_sig or col_sig) if is_wrapped else col_sig,
            )

        # (B) Multi-key selection: var[['X', 'Y']]
        elif isinstance(sl, (ast.List, ast.Tuple)):
            global _cached_cols_proj_sig
            with _PROJECTION_CACHE_LOCK:
                cols_sig = _cached_cols_proj_sig
            if cols_sig is None:
                try:
                    try:
                        from .lattice import LatticeOrchestrator
                    except (ImportError, ValueError):
                        from lattice import LatticeOrchestrator
                    orch = LatticeOrchestrator.get_active_instance()
                    if orch:
                        for c in orch.cells:
                            if getattr(c, "projection", None) in _declared_role_semantics("multi_projection_kinds"):
                                p_out = getattr(c, "primary_output", None) or next(iter(c.outputs.values()), None)
                                if p_out:
                                    cols_sig = getattr(p_out, "signature", p_out)
                                    with _PROJECTION_CACHE_LOCK:
                                        _cached_cols_proj_sig = cols_sig
                                    break
                except Exception as _exc:
                    logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            if cols_sig is None:
                cols_sig = expected_sig or (
                    getattr(base_sig, "signature", base_sig) if base_sig else None
                ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
            keys = [elt.value for elt in sl.elts if isinstance(elt, ast.Constant)]
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": "subscript", "key": keys},
                resulting_signature=(expected_sig or cols_sig) if is_wrapped else cols_sig,
            )

        # (C) Index projection: var[0]
        elif isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            idx_val = sl.value
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
            return TypedBindingRecord(
                port=port_name,
                value_expr=val_str,
                kind="projection",
                root_var=root_v,
                projection={"kind": "index", "key": idx_val},
                resulting_signature=(expected_sig or idx_sig) if is_wrapped else idx_sig,
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

        if method_sig is None and expected_sig:
            method_sig = expected_sig
        if method_sig is None:
            method_sig = (
                getattr(base_sig, "signature", base_sig) if base_sig else None
            ) or AlgebraicSignature(type_name="any", state="any", abstract_type="any")
        return TypedBindingRecord(
            port=port_name,
            value_expr=val_str,
            kind="projection",
            root_var=root_v,
            projection={"kind": "method", "key": attr_name},
            resulting_signature=method_sig,
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
    key_binds = set(reg.get_verification_semantics("key_binds") or ())
    proj_roles = (
        set(reg.get_projection_tokens())
        | set(reg.get_verification_semantics("projection_port_roles") or ())
        | set(reg.get_verification_semantics("key_roles") or ())
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
    if not s or s in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>", "None", "True", "False", "[]", "()"):
        return keys
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return _extract_keys_from_val(parsed)
        except Exception:
            pass
    cl = s.strip("'\"").strip()
    if cl and cl not in ("None", "True", "False"):
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
        cfg_roles = (
            _declared_role_semantics("config_roles")
            | _declared_role_semantics("configuration_roles")
            | _declared_role_semantics("parameter_roles")
        )
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
                    and v_t_cfg == t_name
                    and (reg is None or not reg.is_builtin_scalar_primitive(t_name))
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
    df_roles = _declared_role_semantics("dataflow_roles") | _declared_role_semantics("strict_roles")
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
        rec_roles = _declared_role_semantics("receiver_roles") | _declared_role_semantics("instance_sink_roles")
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
                bind_keys = _declared_role_semantics("key_binds") | _declared_role_semantics("assigned_value_binds")
                proj_roles = _declared_role_semantics("projection_port_roles") | _declared_role_semantics("key_roles")
                proj_names = _declared_role_semantics("projection_port_names")
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

    if optic_role not in _declared_role_semantics("read_optic_roles"):
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
    # Row-set epoch: how many row-changing steps (tree-declared row_set_effects) this value has been through.
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        _row_fx = {
            str(e).lower()
            for key in ("extent_mutating_effects", "row_set_effects")
            for e in (reg.get_verification_semantics(key) or ())
        }
    except Exception:
        _row_fx = set()
    _parent_vars = set(ctx.var_parents.get(out_var, set()))
    for p_v in cell_bindings.values():
        if isinstance(p_v, str) and p_v in ctx.var_epochs:
            _parent_vars.add(p_v)
    _ep = max((ctx.var_epochs.get(v, 0) for v in _parent_vars), default=0)
    eff_names = cell.effect_names() if hasattr(cell, "effect_names") and callable(cell.effect_names) else set()
    if not eff_names and hasattr(cell, "effects"):
        try:
            from lattice import extract_effect_names
            eff_names = extract_effect_names(cell)
        except Exception:
            eff_names = {str(e).lower() for e in (getattr(cell, "effects", None) or [])}
    if _row_fx & eff_names:
        _ep += 1
    ctx.var_epochs[out_var] = _ep

# Purged: Zero hardcoded English wordlists and syntactic hacks
_SENTENCE_CONNECTIVES_REMOVED = None


def _extract_template_callables(tpl: str) -> Set[str]:
    """Extracts root callable identifiers from a code template without regular expressions."""
    callables: Set[str] = set()
    if not tpl:
        return callables
    clean_tpl = tpl
    for ph in extract_template_placeholders(tpl):
        clean_tpl = clean_tpl.replace(f"{{{ph}}}", "_arg_")
    try:
        parsed_tpl = ast.parse(clean_tpl)
        for n in ast.walk(parsed_tpl):
            if isinstance(n, ast.Call):
                func = n.func
                while isinstance(func, ast.Attribute):
                    func = func.value
                if isinstance(func, ast.Name):
                    callables.add(func.id)
    except Exception:
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
    return callables


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


_ROLE_SEMANTICS_CACHE: Dict[Tuple[int, int, str], Tuple[Any, ...]] = {}


def _declared_role_semantics(key: str) -> Set[Any]:
    """Declared role/effect/carrier vocabulary looked up by semantic key.
    The WORDS live in the trees' `verification_semantics` declarations; the
    engine never carries role-name literals of its own."""
    try:
        reg = TypeRegistry.get_instance()
        cache_k = (id(reg), reg._generation, str(key).strip().lower())
        hit = _ROLE_SEMANTICS_CACHE.get(cache_k)
        if hit is not None:
            return set(hit)
        items = []
        for r in reg.get_verification_semantics(key):
            if isinstance(r, (list, tuple)):
                items.append(tuple(str(x).strip().lower() for x in r if str(x).strip()))
            elif str(r).strip():
                items.append(str(r).strip().lower())
        res_t = tuple(dict.fromkeys(items))
        if len(_ROLE_SEMANTICS_CACHE) > 2048:
            _ROLE_SEMANTICS_CACHE.clear()
        _ROLE_SEMANTICS_CACHE[cache_k] = res_t
        return set(res_t)
    except Exception:
        return set()


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
    existing_quals = set()
    raw_old = _get_p_field(port_sig, "qualifiers", None) or getattr(old_sig, "qualifiers", None) or ()
    for q in raw_old:
        if isinstance(q, (tuple, list, set)):
            existing_quals.update(str(x).lower() for x in q)
        else:
            existing_quals.add(str(q).lower())

    combined = existing_quals | {str(x).lower() for x in extra_quals}
    new_norm = _normalize_qualifiers(combined)

    new_alg_sig = AlgebraicSignature(
        type_name=getattr(old_sig, "type_name", "any"),
        state=getattr(old_sig, "state", "any"),
        qualifiers=new_norm,
        abstract_type=getattr(old_sig, "abstract_type", ""),
        accepted_states=getattr(old_sig, "accepted_states", ()),
        parent_state=getattr(old_sig, "parent_state", None),
    )
    if isinstance(port_sig, PortSignature):
        return PortSignature(
            name=port_sig.name,
            signature=new_alg_sig,
            required=port_sig.required,
            default_value=port_sig.default_value,
            doc=port_sig.doc,
            domain=port_sig.domain,
            abstract_type=port_sig.abstract_type,
            enum_values=port_sig.enum_values,
            param_kind=port_sig.param_kind,
            value_constraints=port_sig.value_constraints,
            shape_contract=port_sig.shape_contract,
            rejected_qualifiers=port_sig.rejected_qualifiers,
            accepted_states=port_sig.accepted_states,
            parent_state=port_sig.parent_state,
            port_role=port_sig.port_role,
            binds=port_sig.binds,
            structure_kind=port_sig.structure_kind,
            rank=port_sig.rank,
            element_type=port_sig.element_type,
            supported_formats=port_sig.supported_formats,
            complement_of_port=getattr(port_sig, "complement_of_port", None),
            disjoint_from_port=getattr(port_sig, "disjoint_from_port", None),
            key_from_upstream_role=getattr(port_sig, "key_from_upstream_role", None),
            owner_cell=port_sig.owner_cell,
        )
    return new_alg_sig



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
        _br_roles = TypeRegistry.get_instance().get_bridge_roles_set() | _declared_role_semantics("clean_roles")
    except Exception:
        _br_roles = _declared_role_semantics("clean_roles")
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
    strict_roles = _declared_role_semantics("strict_roles")
    res = False
    if r_port in strict_roles:
        if r_var != r_port and not reg.is_subtype(r_var, r_port) and r_var not in strict_roles:
            res = True
    if not res and r_var in strict_roles:
        if r_port != r_var and not reg.is_subtype(r_port, r_var) and r_port not in strict_roles:
            res = True

    if not res and r_port != "standard" and r_var != "standard":
        exclusive_roles = _declared_role_semantics("mutually_exclusive_roles")
        if r_port in exclusive_roles and r_var in exclusive_roles and r_port != r_var:
            res = True

        if not res:
            exclusive_role_pairs = _declared_role_semantics("mutually_exclusive_role_pairs")
            for pair in exclusive_role_pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    p_a, p_b = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
                    if (r_port == p_a and r_var == p_b) or (r_port == p_b and r_var == p_a):
                        res = True
                        break
                    if (reg.is_subtype(r_port, p_a) and reg.is_subtype(r_var, p_b)) or (reg.is_subtype(r_port, p_b) and reg.is_subtype(r_var, p_a)):
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

    pred_in, pred_out, pred_states, term_invalid_states, out_targets, protocols = reg._get_edge_sem_bundle()

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

    if out_sk == "handle" and in_sk and in_sk != "handle":
        return False
    if in_sk == "handle" and out_sk and out_sk != "handle":
        return False

    _path_compat_sk = _declared_role_semantics("path_compat_structure_kinds")
    if out_sk == "path_spec" and in_sk and in_sk not in _path_compat_sk:
        return False
    if in_sk == "path_spec" and out_sk and out_sk not in _path_compat_sk:
        return False

    if out_sk or in_sk or out_abs or in_abs:
        _scalar_abs = _declared_role_semantics("scalar_abstract_types")
        _compound_sk = _declared_role_semantics("compound_structure_kinds")
        is_out_scalar = (out_sk == "scalar") or (out_abs in _scalar_abs)
        is_in_scalar = (in_sk == "scalar") or (in_abs in _scalar_abs)
        is_out_compound = (out_sk in _compound_sk) or (bool(out_abs) and (reg.is_composite_carrier(out_abs) or reg.is_abstract_carrier(out_abs)))
        is_in_compound = (in_sk in _compound_sk) or (bool(in_abs) and (reg.is_composite_carrier(in_abs) or reg.is_abstract_carrier(in_abs)))

        if is_out_scalar and is_in_compound:
            return False
        if is_in_scalar and is_out_compound:
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
        if producer_cell is not None:
            for post in (getattr(producer_cell, "postconditions", None) or ()):
                if not isinstance(post, dict):
                    continue
                post_target = post.get("target")
                if post_target is None or post_target == out_p_name or post_target in out_targets:
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
        if res.nominal_conflict or res.layout_delta or res.semantics_delta:
            return False

    return True


def _shape_compatible(p_out: Any, p_in: Any, producer_cell: Optional[Any] = None, consumer_cell: Optional[Any] = None) -> bool:
    return is_edge_valid(p_out, p_in, producer_cell=producer_cell, consumer_cell=consumer_cell)


@functools.lru_cache(maxsize=4096)
def _extract_lhs_template_placeholders(tpl: str) -> Set[str]:
    """Extracts placeholder names bound on the LHS of assignments in code_template."""
    if not tpl or "=" not in tpl:
        return set()
    phs = extract_template_placeholders(tpl)
    if not phs:
        return set()
    ast_ready = tpl
    for ph in phs:
        ast_ready = ast_ready.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
    try:
        parsed = ast.parse(ast_ready)
        lhs_phs = set()
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for t_node in ast.walk(target):
                        if isinstance(t_node, ast.Name) and t_node.id.startswith("_nstl_ph_"):
                            lhs_phs.add(t_node.id[len("_nstl_ph_"):])
        if lhs_phs:
            return set(lhs_phs)
    except Exception:
        pass

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
        return set(extract_template_placeholders(lhs))
    return set()


def _has_data_bearing_output_assignment(cell: Any) -> bool:
    """True iff cell's code_template assigns to a data-bearing output variable on LHS of '='."""
    if cell is None:
        return False
    outputs = getattr(cell, "outputs", None)
    if not outputs:
        return False

    try:
        reg = TypeRegistry.get_instance()
        mat_states = set(reg.get_materialization_states()) | set(_declared_role_semantics("materialization_states"))
    except Exception:
        reg = None
        mat_states = set(_declared_role_semantics("materialization_states"))

    candidate_ports = []
    prim_out = getattr(cell, "primary_output", None)
    if prim_out is not None:
        candidate_ports.append(prim_out)
    if isinstance(outputs, dict):
        candidate_ports.extend(outputs.values())
    elif isinstance(outputs, (list, tuple, set)):
        candidate_ports.extend(outputs)

    data_bearing_names: Set[str] = set()
    for p in candidate_ports:
        if p is None:
            continue
        p_name = getattr(p, "name", None)
        sig = getattr(p, "signature", p)
        t_name = str(getattr(sig, "type_name", "") or "").strip().lower()
        state = str(getattr(sig, "state", "") or "").strip().lower()
        void_types = _declared_role_semantics("void_types")
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

    return bool(lhs_phs & (data_bearing_names | _declared_role_semantics("output_targets")))


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
        term_roles = _declared_role_semantics("terminal_roles") | _declared_role_semantics("sink_roles")
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
    Purely poset/port-driven; no type-name vocabulary."""
    reg = TypeRegistry.get_instance()
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

    Strictly eligible ONLY for unconstrained rank-0 scalar/enum configuration ports:
    dataflow carriers, composites, collections/mappings, paths, key/projection ports,
    and constrained/optic input ports are never fabricated."""
    reg = TypeRegistry.get_instance()
    sig = getattr(port_sig, "signature", port_sig)
    t = str(getattr(sig, "type_name", "") or "").strip()
    t_l = t.lower()
    if not t or "[" in t or reg.is_type_variable(t) or reg.is_declared_top(t_l) or reg.is_void_type(t_l):
        return None
    if t_l in _declared_role_semantics("bottom_types"):
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
    _coll_roots = _declared_role_semantics("collection_types")
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

    cfg_roles = (
        _declared_role_semantics("config_roles")
        | _declared_role_semantics("configuration_roles")
        | _declared_role_semantics("parameter_roles")
    )
    non_cfg_roles = (
        _declared_role_semantics("dataflow_roles")
        | _declared_role_semantics("strict_roles")
        | _declared_role_semantics("projective_roles")
        | _declared_role_semantics("projection_port_roles")
        | _declared_role_semantics("key_roles")
        | _declared_role_semantics("key_binds")
        | _declared_role_semantics("assigned_value_roles")
        | _declared_role_semantics("assigned_value_binds")
        | _declared_role_semantics("receiver_roles")
        | _declared_role_semantics("sink_roles")
        | _declared_role_semantics("terminal_roles")
        | _declared_role_semantics("destination_roles")
        | _declared_role_semantics("source_roles")
        | _declared_role_semantics("ingress_roles")
        | _declared_role_semantics("path_roles")
        | _declared_role_semantics("instance_sink_roles")
        | _declared_role_semantics("carrier_binds")
        | (reg.get_declared_role_carriers() - cfg_roles)
    )
    if p_role in non_cfg_roles or p_binds in non_cfg_roles:
        return None
    non_cfg_states = (
        _declared_role_semantics("condition_states")
        | _declared_role_semantics("projection_port_states")
        | _declared_role_semantics("target_states")
        | _declared_role_semantics("source_states")
        | _declared_role_semantics("terminal_states")
        | _declared_role_semantics("materialization_states")
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
        strict_flag = False
        try:
            try:
                from .config import get_settings
            except (ImportError, ValueError):
                from config import get_settings
            strict_flag = bool(getattr(get_settings(), "strict_io", False))
        except Exception:
            strict_flag = False
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
        _declared_role_semantics("destination_roles")
        | _declared_role_semantics("sink_roles")
        | _declared_role_semantics("terminal_roles")
        | _declared_role_semantics("target_roles")
    )
    src_roles = _declared_role_semantics("source_roles") | _declared_role_semantics("ingress_roles")
    dest_states = (
        _declared_role_semantics("materialization_states")
        | _declared_role_semantics("terminal_states")
        | _declared_role_semantics("target_states")
    )
    src_states = _declared_role_semantics("source_states")

    if p_role in src_roles or p_state in src_states:
        is_output = False
    elif p_role in dest_roles or p_state in dest_states:
        is_output = True
    else:
        is_output = bool(cell is not None and _is_sink_cell(cell))

    wildcards = _declared_role_semantics("top_states") | {"*", "any", "all", ""}
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
    @abstractmethod
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        pass

    @classmethod
    def from_string(cls, s: str) -> 'TypeTerm':
        return parse_type_term(s)


@dataclass(frozen=True, slots=True)
class TopType(TypeTerm):
    """Universal Top type (wildcard) unifying with any type term."""
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return "Top"


TOP = TopType()


@dataclass(frozen=True, slots=True)
class AtomicType(TypeTerm):
    """Ground type constant."""
    name: str

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

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        resolved_terms = tuple(t.apply_substitution(sigma, visited) for t in self.terms)
        return UnionTypeTerm(resolved_terms)

    def __repr__(self) -> str:
        return " | ".join(str(t) for t in self.terms)


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    """Type variable alpha, beta... subject to substitution."""
    var_name: str

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


@functools.lru_cache(maxsize=4096)
def parse_type_term(type_str: str) -> TypeTerm:
    """Parses a type string into a structured TypeTerm, memoized with LRU cache."""
    s_clean = str(type_str).strip()
    registry = TypeRegistry.get_instance()
    if is_top_symbol(s_clean, registry):
        return TOP
    if s_clean.startswith("?"):
        return TypeVariable(s_clean[1:])
    # Coproduct / Sum / Union types (e.g. A | B or Union[A, B])
    if " | " in s_clean:
        parts = [p.strip() for p in s_clean.split(" | ")]
        return UnionTypeTerm(tuple(parse_type_term(p) for p in parts if p))
    if s_clean.startswith("Union[") and s_clean.endswith("]"):
        inner = s_clean[6:-1].strip()
        args = split_bracketed_args(inner)
        return UnionTypeTerm(tuple(parse_type_term(a) for a in args if a))
    if registry.is_type_variable(s_clean):
        return TypeVariable(s_clean)
    if "[" in s_clean and s_clean.endswith("]"):
        bracket_idx = s_clean.index("[")
        constructor = s_clean[:bracket_idx].strip()
        inner = s_clean[bracket_idx + 1 : -1].strip()
        args = split_bracketed_args(inner)
        parsed_args = tuple(parse_type_term(a) for a in args if a)
        if parsed_args:
            return GenericTypeTerm(constructor, parsed_args)
    return AtomicType(s_clean)


@dataclass(frozen=True, slots=True)
class TypestateTerm(TypeTerm):
    """Typestate compound term: tau = (type_name, state, qualifiers, abstract_type, accepted_states, parent_state)."""
    type_name: str
    state: str = "any"
    qualifiers: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    abstract_type: Optional[str] = None
    accepted_states: Tuple[str, ...] = field(default_factory=tuple)
    parent_state: Optional[str] = None

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

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[Tuple[str, ...]] = None) -> 'TypeTerm':
        t_resolved = self.type_name
        if self.type_name in sigma.mappings:
            if visited and self.type_name in visited:
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
        )

    def __repr__(self) -> str:
        abs_str = f", abs={self.abstract_type}" if self.abstract_type else ""
        acc_str = f", acc={list(self.accepted_states)}" if self.accepted_states else ""
        p_str = f", parent={self.parent_state}" if self.parent_state else ""
        return f"{self.type_name}[{self.state}{abs_str}{acc_str}{p_str}]"


class Substitution:
    """Mapping of variable identifiers to resolved type terms or values."""
    def __init__(self, mappings: Optional[Dict[str, Any]] = None):
        self.mappings: Dict[str, Any] = dict(mappings) if mappings else {}

    def bind(self, var: str, value: Any):
        self.mappings[var] = value

    def get(self, var: str, default: Any = None) -> Any:
        return self.mappings.get(var, default)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose substitutions: (sigma1 . sigma2)(t) = sigma1(sigma2(t))."""
        new_map = dict(self.mappings)
        for k, v in other.mappings.items():
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
        if term.type_name == var_name:
            return True
        if sigma and term.type_name in sigma.mappings:
            new_visited = (visited or ()) + (term.type_name,)
            return occurs_check(var_name, sigma.mappings[term.type_name], sigma, new_visited)
        return False
    elif isinstance(term, AtomicType):
        return term.name == var_name
    elif isinstance(term, AlgebraicSignature):
        return occurs_check(var_name, term.type_name, sigma, visited)
    elif isinstance(term, str):
        if var_name == term:
            return True
        tokens = tokenize_alphanumeric(term, min_len=1)
        if var_name in tokens:
            return True
        if sigma:
            for tok in tokens:
                if tok in sigma.mappings and (not visited or tok not in visited):
                    new_visited = (visited or ()) + (tok,)
                    if occurs_check(var_name, sigma.mappings[tok], sigma, new_visited):
                        return True
        return False
    return False


_UNIFY_CACHE_MAX = 4096
_UNIFY_CACHE_LOCK = threading.Lock()
_UNIFY_BASE_CACHE: collections.OrderedDict[Tuple[TypeTerm, TypeTerm], Optional[Substitution]] = collections.OrderedDict()


def _unify_cache_put(key: Tuple[TypeTerm, TypeTerm], value: Optional[Substitution]) -> None:
    """Thread-safe, bounded write to _UNIFY_BASE_CACHE."""
    with _UNIFY_CACHE_LOCK:
        _UNIFY_BASE_CACHE[key] = value
        if len(_UNIFY_BASE_CACHE) > _UNIFY_CACHE_MAX:
            _UNIFY_BASE_CACHE.popitem(last=False)


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

    is_ground_query = (sigma is None or not sigma.mappings)
    if is_ground_query:
        cache_key = (t1, t2)
        with _UNIFY_CACHE_LOCK:
            if cache_key in _UNIFY_BASE_CACHE:
                cached = _UNIFY_BASE_CACHE[cache_key]
                return Substitution(cached.mappings) if cached is not None else None

    sub = Substitution(sigma.mappings if sigma else {})

    t1 = t1.apply_substitution(sub)
    t2 = t2.apply_substitution(sub)

    # 0. Structural typestate wrappers (generic/union carrier + state) meeting a
    #    bare structural term: compare the structural carriers directly.
    def _structural_inner(t: TypeTerm) -> Optional[TypeTerm]:
        if isinstance(t, TypestateTerm) and ("[" in t.type_name or "|" in t.type_name):
            return TypeTerm.from_string(t.type_name)
        return None

    if isinstance(t1, TypestateTerm) != isinstance(t2, TypestateTerm):
        in1, in2 = _structural_inner(t1), _structural_inner(t2)
        if in1 is not None and not isinstance(t2, (TopType, TypeVariable)):
            return unify(in1, t2, sub)
        if in2 is not None and not isinstance(t1, TypeVariable):
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
    #   Producer A ⊕ B flows into C iff EVERY branch flows into C (copairing [f, g]);
    #   otherwise an explicit coproduct-branch cell is required.
    if isinstance(t1, UnionTypeTerm):
        acc_sub = sub
        for alt in t1.terms:
            acc_sub = unify(alt, t2, acc_sub)
            if acc_sub is None:
                break
        if is_ground_query:
            _unify_cache_put(cache_key, acc_sub)
        return acc_sub

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

    # 2.5. Generic container unification (with contravariant input / covariant output for Callable)
    if isinstance(t1, GenericTypeTerm) and isinstance(t2, GenericTypeTerm):
        registry = TypeRegistry.get_instance()
        c1 = t1.constructor.lower()
        c2 = t2.constructor.lower()
        compat = (c1 == c2) or registry.is_subtype(c1, c2)
        if compat and len(t1.args) == len(t2.args):
            if c1 == "callable" and c2 == "callable" and len(t1.args) == 2:
                # Callable[[Dom1], Cod1] <= Callable[[Dom2], Cod2] iff Dom2 <= Dom1 and Cod1 <= Cod2
                sub = unify(t2.args[0], t1.args[0], sub)
                if sub is None:
                    if is_ground_query:
                        _unify_cache_put(cache_key, None)
                    return None
                sub = unify(t1.args[1], t2.args[1], sub)
                if sub is None:
                    if is_ground_query:
                        _unify_cache_put(cache_key, None)
                    return None
                if is_ground_query:
                    _unify_cache_put(cache_key, sub)
                return sub
            for a1, a2 in zip(t1.args, t2.args):
                sub = unify(a1, a2, sub)
                if sub is None:
                    if is_ground_query:
                        _unify_cache_put(cache_key, None)
                    return None
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

    # Raw type fallback: List[Contour] satisfies unparameterized List
    if isinstance(t1, GenericTypeTerm) and (isinstance(t2, AtomicType) or isinstance(t2, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        target_name = t2.type_name if isinstance(t2, TypestateTerm) else t2.name
        if registry.is_subtype(t1.constructor, target_name):
            if is_ground_query:
                _unify_cache_put(cache_key, sub)
            return sub

    # Sound parameter demand: untyped list satisfies GenericTypeTerm ONLY IF parameters are type variables
    if isinstance(t2, GenericTypeTerm) and (isinstance(t1, AtomicType) or isinstance(t1, TypestateTerm)):
        registry = TypeRegistry.get_instance()
        source_name = t1.type_name if isinstance(t1, TypestateTerm) else t1.name
        if registry.is_subtype(source_name, t2.constructor):
            if all(isinstance(arg, (TypeVariable, TopType)) for arg in t2.args):
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
        # State compatibility check: delegate to TypeRegistry.is_state_compatible
        registry = TypeRegistry.get_instance()
        if not registry.is_state_compatible(
            producer_state=t1.state,
            consumer_state=t2.state,
            producer_accepted=t1.accepted_states,
            consumer_accepted=t2.accepted_states,
            consumer_parent=t2.parent_state,
            producer_parent=t1.parent_state,
        ):
            if is_ground_query:
                _unify_cache_put(cache_key, None)
            return None  # State mismatch -> bottom

        # Qualifier subset check
        t1_quals = set(t1.qualifiers or ())
        t2_quals = set(t2.qualifiers or ())
        if t2_quals and not t2_quals.issubset(t1_quals):
            ignorable = TypeRegistry.get_instance().get_advisory_qualifiers()
            req = {q for q in t2_quals if q not in ignorable}
            if req and not req.issubset(t1_quals):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None

        # Check if type_names contain generic / union definitions
        if ("[" in t1.type_name) or ("[" in t2.type_name) or ("|" in t1.type_name) or ("|" in t2.type_name) or (len(t1.type_name) == 1 and t1.type_name.isupper()) or (len(t2.type_name) == 1 and t2.type_name.isupper()):
            inner1 = TypeTerm.from_string(t1.type_name)
            inner2 = TypeTerm.from_string(t2.type_name)
            return unify(inner1, inner2, sub)

        # Type poset subtyping check: t1.type_name <= t2.type_name
        registry = TypeRegistry.get_instance()
        if not registry.is_subtype(t1.type_name, t2.type_name):
            t2_tn = (t2.type_name or "").strip().lower()
            t2_abs = (t2.abstract_type or "").strip().lower()
            is_consumer_abstract = (t2_tn in ABSTRACT_CARRIERS or t2_tn == t2_abs)
            if (
                is_consumer_abstract
                and t1.abstract_type
                and t2.abstract_type
                and registry.is_subtype(t1.abstract_type, t2.abstract_type)
            ):
                pass
            else:
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None  # Type mismatch -> bottom

        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

    # 4. Atomic type unification
    if isinstance(t1, AtomicType) and isinstance(t2, AtomicType):
        registry = TypeRegistry.get_instance()
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


_ALGEBRAIC_SIG_TERM_CACHE: Dict[AlgebraicSignature, TypeTerm] = {}
_TERM_CACHE_LOCK = threading.RLock()


def _to_type_term(item: Any) -> TypeTerm:
    if isinstance(item, TypeTerm):
        return item
    if isinstance(item, PortSignature):
        cached = getattr(item, "_cached_term", None)
        if cached is not None:
            return cached
        res = _to_type_term(item.signature)
        try:
            item._cached_term = res
        except (AttributeError, TypeError):
            pass
        return res
    if isinstance(item, AlgebraicSignature):
        with _TERM_CACHE_LOCK:
            cached = _ALGEBRAIC_SIG_TERM_CACHE.get(item)
            if cached is not None:
                return cached
        registry = TypeRegistry.get_instance()
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
                )
        with _TERM_CACHE_LOCK:
            _ALGEBRAIC_SIG_TERM_CACHE[item] = res
        return res
    registry = TypeRegistry.get_instance()
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
        return PortSignature(
            name=target.name,
            signature=new_sig,
            required=target.required,
            default_value=target.default_value,
            doc=target.doc,
            domain=target.domain,
            abstract_type=target.abstract_type,
            enum_values=target.enum_values,
            param_kind=target.param_kind,
            value_constraints=target.value_constraints,
            shape_contract=target.shape_contract,
            rejected_qualifiers=getattr(target, "rejected_qualifiers", ()),
            accepted_states=getattr(target, "accepted_states", ()),
            parent_state=getattr(target, "parent_state", None),
            port_role=getattr(target, "port_role", None),
            binds=getattr(target, "binds", None),
            structure_kind=getattr(target, "structure_kind", None),
            rank=getattr(target, "rank", None),
            element_type=getattr(target, "element_type", None),
            supported_formats=getattr(target, "supported_formats", None),
            complement_of_port=getattr(target, "complement_of_port", None),
            disjoint_from_port=getattr(target, "disjoint_from_port", None),
            key_from_upstream_role=getattr(target, "key_from_upstream_role", None),
            owner_cell=getattr(target, "owner_cell", None),
        )

    if isinstance(target, AlgebraicSignature):
        new_type = substitute_generics(target.type_name, sub)
        return AlgebraicSignature(
            type_name=str(new_type),
            state=target.state,
            qualifiers=target.qualifiers,
            abstract_type=target.abstract_type,
            accepted_states=getattr(target, "accepted_states", ()),
            parent_state=getattr(target, "parent_state", None)
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
        return not (r1.nominal_conflict or r2.nominal_conflict or r1.layout_delta or r2.layout_delta or r1.semantics_delta or r2.semantics_delta)
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
        if res.nominal_conflict or res.carrier_delta or res.layout_delta or res.semantics_delta or res.typestate_delta:
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


def unit(value: T, sigma: Optional[Substitution] = None) -> MonadResult[T]:
    """Monad unit: injects value a with initial substitution sigma into the monad."""
    return Success(value, sigma or Substitution())


def bind(
    m: MonadResult[T],
    k: Callable[[T, Substitution], MonadResult[U]]
) -> MonadResult[U]:
    """
    Monadic bind:
      bind(m, k) = k(a) with sigma_new if step succeeds; otherwise bottom.
    If m is Failure, short-circuits immediately to Failure.
    """
    if m.is_bottom():
        return Failure(m.reason if isinstance(m, Failure) else "Bottom")
    assert isinstance(m, Success)
    return k(m.value, m.sigma)


def _is_valid_slot_value(val: Any, sig: Optional[PortSignature] = None) -> bool:
    """Validates whether an extracted slot/hyperparameter value conforms to port schema."""
    if val is None or sig is None:
        return False
    val_s = str(val).strip()
    if not val_s:
        return False
    val_low = val_s.lower()
    if val_low in _declared_role_semantics("void_types"):
        return False

    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
    except Exception:
        reg = None

    c_rank = getattr(sig, "canonical_rank", None) if getattr(sig, "canonical_rank", None) is not None else getattr(sig, "rank", None)
    str_kind = str(getattr(sig, "structure_kind", "") or "").lower()
    t_name = str(getattr(sig, "type_name", "") or "").strip().lower()
    is_scalar = (
        c_rank == 0
        or str_kind == "scalar"
        or (reg is not None and reg.is_subtype(t_name, "scalar"))
    )
    if is_scalar and isinstance(val, (list, tuple)):
        return False

    declared_enums = getattr(sig, "enum_values", None)
    enum_set = {str(e).strip().lower() for e in declared_enums} if declared_enums else set()

    # Tier 1: Reject echoes of the port's own declared type or abstract type
    t_sig = str(getattr(sig, "type_name", "")).strip().lower()
    abs_sig = str(getattr(sig, "abstract_type", "")).strip().lower()
    if (val_low == t_sig or (abs_sig and val_low == abs_sig)) and val_low not in enum_set:
        return False

    # Tier 2: Dynamic Python Built-in Types (via runtime reflection, 0 hardcoded strings)
    import builtins
    if not hasattr(_is_valid_slot_value, "_builtin_type_names"):
        _is_valid_slot_value._builtin_type_names = {
            k.lower() for k, v in builtins.__dict__.items() if isinstance(v, type)
        }
    if val_low in _is_valid_slot_value._builtin_type_names and val_low not in enum_set:
        return False

    # Tier 3: Poset Ontology Types from TypeRegistry (ndarray, DataFrame, Series, etc.)
    if reg is not None:
        try:
            if reg.is_known_type_name(val_low) and val_low not in enum_set:
                return False
        except Exception as _exc:
            logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

    # If the port declares discrete enum_values, candidate MUST match one of them
    if declared_enums:
        return val_low in enum_set

    # Verify numeric convertibility using Poset subtyping if port is numeric
    if reg is not None:
        try:
            if reg.is_subtype(t_sig, "int"):
                try:
                    int(val_s)
                    return True
                except ValueError:
                    return False
            if reg.is_subtype(t_sig, "numeric"):
                try:
                    float(val_s)
                    return True
                except ValueError:
                    return False
        except Exception as _exc:
            logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

    return True


def _format_literal_value(val: Any, sig: Optional[PortSignature] = None) -> str:
    """Formats an extracted slot/hyperparameter value into an executable code literal."""
    if val is None:
        return "None"
    if sig is not None:
        c_rank = getattr(sig, "canonical_rank", None) if getattr(sig, "canonical_rank", None) is not None else getattr(sig, "rank", None)
        str_kind = str(getattr(sig, "structure_kind", "") or "").lower()
        t_name = str(getattr(sig, "type_name", "") or "").strip().lower()
        reg = TypeRegistry.get_instance()
        is_scalar = (
            c_rank == 0
            or str_kind == "scalar"
            or (reg is not None and reg.is_subtype(t_name, "scalar"))
        )
        if is_scalar and isinstance(val, (list, tuple)):
            if getattr(sig, "default_value", None) is not None:
                return str(sig.default_value)
            return "None"
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple, dict)):
        return repr(val)
    s = str(val).strip()
    if s in ("None", "True", "False"):
        return s

    t_name = str(getattr(sig, "type_name", "")).lower() if sig else ""
    str_types = _declared_role_semantics("string_like_types")
    try:
        registry = TypeRegistry.get_instance()
        is_str_like = (
            t_name in str_types
            or registry.is_subtype(t_name, "str")
            or registry.is_subtype(t_name, "string")
        )
    except Exception:
        is_str_like = t_name in str_types

    p_role = str(getattr(sig, "port_role", "") or getattr(sig, "role", "") or "").strip().lower()
    if t_name == "path":
        raw_p = s.strip("'\"").strip()
        if raw_p and not (raw_p.isidentifier() and "." not in raw_p and "/" not in raw_p and "\\" not in raw_p):
            return f"Path({repr(raw_p)})"
    if p_role == "data_input" and s.isidentifier() and not is_str_like and t_name != "path":
        return s

    declared_enums = getattr(sig, "enum_values", None)
    if declared_enums:
        enum_set = {str(e).strip("'\"").strip().lower() for e in declared_enums}
        raw_s = s.strip("'\"").strip()
        if raw_s.lower() in enum_set:
            return repr(raw_s) if is_str_like else raw_s
        elif s.lower() not in enum_set:
            if getattr(sig, "default_value", None) is not None:
                def_s = str(sig.default_value).strip("'\"").strip()
                return repr(def_s) if is_str_like else def_s
            return "None"

    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        inner = s[1:-1].strip()
        if (inner.startswith("'") and inner.endswith("'")) or (inner.startswith('"') and inner.endswith('"')):
            return repr(inner[1:-1].strip())
        return s
    try:
        float(s)
        return s
    except ValueError:
        pass

    reg_lit = TypeRegistry.get_instance()
    # Functional operators (lambda expressions / callable identifiers) must stay executable Python expressions,
    # never quoted as non-callable string constants.
    try:
        is_callable_type = bool(t_name) and TypeTerm.from_string(t_name).base == "callable"
    except Exception:
        is_callable_type = False
    if reg_lit.is_hof_role(p_role) or is_callable_type:
        unq = s[1:-1].strip() if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s
        try:
            node = ast.parse(unq, mode="eval").body
            if isinstance(node, (ast.Lambda, ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
                return unq
        except SyntaxError:
            pass
        return s

    lit_roles = set(reg_lit.get_verification_semantics("literal_roles") or ())
    is_lit_role = (p_role in lit_roles) or reg_lit.is_config_role(p_role)
    # If the value represents an identifier or pipeline variable expression and port is not string-like
    if not is_str_like and not is_lit_role:
        if s.isidentifier() or s.startswith("var_"):
            return s

    if is_str_like or not s.isidentifier() or is_lit_role:
        raw_s = s.strip("'\"").strip()
        return repr(raw_s)
    return s


def _consume_matching_literals(ctx: Any, slot_val: Any) -> None:
    """Marks prompt literals matching an SLM-bound value as consumed in `ctx.used_indices`."""
    ordered = getattr(ctx, "ordered_literals", None)
    used = getattr(ctx, "used_indices", None)
    if not ordered or used is None or slot_val is None:
        return
    if isinstance(slot_val, (list, tuple)):
        targets = [str(v).strip().strip("'\"").lower() for v in slot_val if v is not None]
    else:
        targets = [str(slot_val).strip().strip("'\"").lower()]
    targets = [t for t in targets if t]
    if not targets:
        return
    clause_idx = getattr(ctx, "current_cell_clause_idx", None)
    lit_clause_map = getattr(ctx, "literal_clause_map", {}) or {}
    for targ in targets:
        # Prefer unconsumed literal in current clause, then global unconsumed literal
        matched_i = None
        for idx, (_, _, lit_v) in enumerate(ordered):
            if idx in used:
                continue
            if str(lit_v).strip().strip("'\"").lower() == targ:
                if clause_idx is not None and lit_clause_map.get(idx) == clause_idx:
                    matched_i = idx
                    break
                if matched_i is None:
                    matched_i = idx
        if matched_i is not None:
            used.add(matched_i)



# =====================================================================
# 4. Domain-Agnostic Execution Context
# =====================================================================

class LiteralSpan(tuple):
    """3-tuple (pos, kind, val) that also retains exact span start and end positions."""
    def __new__(cls, pos: int, kind: str, val: str, end_pos: Optional[int] = None):
        inst = super().__new__(cls, (pos, kind, val))
        inst.end_pos = end_pos if end_pos is not None else (pos + len(val))
        return inst


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
                self.strict_io: bool = bool(getattr(get_settings(), "strict_io", False))
            except Exception:
                self.strict_io = False
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
        self.optic_frames: List[Dict[str, Any]] = []
        self.multi_fiber_vars: Set[str] = set()
        self.current_pipeline: Optional[List[Any]] = None
        self.var_epochs: Dict[str, int] = {}   # row-set epoch per variable (tree-declared row_set_effects)
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
        self.optic_frames = []
        self.multi_fiber_vars = set()
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
        new_ctx.optic_frames = [dict(f) for f in getattr(self, "optic_frames", [])]
        new_ctx.multi_fiber_vars = set(getattr(self, "multi_fiber_vars", set()))
        new_ctx.current_pipeline = getattr(self, "current_pipeline", None)
        new_ctx.var_epochs = dict(self.var_epochs)
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
        new_ctx.literal_clause_map = dict(self.literal_clause_map)
        new_ctx.ordered_literals = list(self.ordered_literals)
        new_ctx.identifier_roles = dict(self.identifier_roles)
        new_ctx.clause_spans = list(self.clause_spans)
        new_ctx.unresolved_ports = list(self.unresolved_ports)
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
        naming_connectives: Set[str] = set()
        dest_triggers = (reg.get_dest_port_tokens() | reg.get_egress_tokens()) - naming_connectives
        src_triggers = reg.get_source_port_tokens()

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
            if w in naming_connectives:
                continue
            if w in dest_triggers:
                return "dest"
            if w in src_triggers:
                return "src"
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
        prompt_words = set(str(self.prompt or "").lower().split())
        has_prompt_egress = bool(prompt_words & (reg.get_egress_tokens() | reg.get_dest_port_tokens()))
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

        # Invert any pairs declared in antonym_flag_pairs
        antonym_pairs = _declared_role_semantics("antonym_flag_pairs")
        for pair in antonym_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                f_a, f_b = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
                if f_a in res and f_b not in res:
                    res[f_b] = not res[f_a]
                elif f_b in res and f_a not in res:
                    res[f_a] = not res[f_b]
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

            # Numeric tokens (including comma-separated integers like "1,000" or "10,000")
            num_candidate = w.replace(",", "") if ("," in w and all(part.isdigit() for part in w.split(","))) else w
            try:
                float(num_candidate)
                spans.append(LiteralSpan(pos, "numeric", num_candidate, end_pos=pos + len(num_candidate)))
                continue
            except ValueError:
                pass

            # No heuristic dimension splitting; numeric tokens are extracted cleanly as units

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

        # 1. Continuous vector embedding similarity if ModelManager is active
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            import numpy as np

            mm = ModelManager.get_instance()
            if mm.profile is not None and self.prompt:
                e_prompt = np.array(mm.get_embedding(self.prompt), dtype=np.float32)
                p_norm = np.linalg.norm(e_prompt)
                if p_norm > 0:
                    e_prompt = e_prompt / p_norm
                    best_cand = None
                    best_sim = -1.0
                    cand_texts = [cand.replace("_", " ").lower() for cand in sorted_candidates]
                    cand_embs = mm.get_embeddings(cand_texts, mode="query")
                    for cand, raw_e in zip(sorted_candidates, cand_embs):
                        e_c = np.array(raw_e, dtype=np.float32)
                        c_norm = np.linalg.norm(e_c)
                        if c_norm > 0:
                            sim = float(np.dot(e_prompt, e_c / c_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_cand = cand
                    if best_cand and best_sim > 0.30:
                        return f"{mod_name}.{best_cand}"
        except Exception as e:
            logger.debug("suppressed: %s", e, exc_info=False)

        # 2. Token overlap and parsimony scoring fallback
        prompt_lower = (self.prompt or "").lower()
        p_tokens = CellTokenizer.tokenize_prompt(prompt_lower)

        scored = []
        for cand in sorted_candidates:
            parts = [p for p in CellTokenizer.tokenize_identifier(cand.lower()) if len(p) >= 2]
            if not parts:
                continue

            score = 0.0
            matched_parts = 0

            for idx, part in enumerate(parts):
                matched = False
                if part in p_tokens:
                    score += 2.0
                    matched = True
                elif any(t.startswith(part) or part.startswith(t) for t in p_tokens if len(t) >= 4 and len(part) >= 4):
                    score += 1.5
                    matched = True

                if matched:
                    matched_parts += 1
                    if idx == len(parts) - 1:
                        score += 2.0

            unmatched_parts = len(parts) - matched_parts
            score -= 0.5 * unmatched_parts
            scored.append((score, -len(cand), cand))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored and scored[0][0] > 0.0:
            return f"{mod_name}.{scored[0][2]}"

        return None

    def resolve_literal_for_port(
        self,
        port_sig: PortSignature,
        cell_stage: Optional[int] = None,
        cell_inputs: Optional[Dict[str, PortSignature]] = None,
        cell_tokens: Optional[Set[str]] = None,
        cell: Optional[Any] = None
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
        c_rank = getattr(port_sig, "canonical_rank", None) if getattr(port_sig, "canonical_rank", None) is not None else getattr(port_sig, "rank", None)
        str_kind = str(getattr(port_sig, "structure_kind", "") or "").lower()
        is_scalar = (c_rank == 0 or str_kind == "scalar" or registry.is_scalar_primitive(t_name) or registry.is_subtype(t_name, "scalar"))

        if port_sig.name in self.parameters:
            val = self.parameters[port_sig.name]
            if not (is_scalar and isinstance(val, (list, tuple))):
                return json.dumps(val) if isinstance(val, str) else str(val)
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
        if _step_idx is not None and isinstance(getattr(self, "step_llm_slots", None), dict):
            _step_s = self.step_llm_slots.get(_step_idx)
            if isinstance(_step_s, dict) and port_sig.name in _step_s:
                slot_val = _step_s[port_sig.name]
                if not (is_scalar and isinstance(slot_val, (list, tuple))):
                    return _format_literal_value(slot_val, port_sig)
        if hasattr(self, "llm_slots") and isinstance(self.llm_slots, dict):
            _cid = getattr(cell, "cell_id", None) if cell is not None else None
            _cell_s = self.llm_slots.get(_cid) if _cid else None
            if isinstance(_cell_s, dict) and port_sig.name in _cell_s:
                slot_val = _cell_s[port_sig.name]
                if not (is_scalar and isinstance(slot_val, (list, tuple))):
                    return _format_literal_value(slot_val, port_sig)

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
            # Inverted-pole fallback for DECLARED order-flag identifiers: a port
            # named "descending" means its positive pole is largest-first.
            # Token-wise so e.g. "description" never matches.
            if "negative" not in qualifier_map:
                neg_label = f"not {port_sig.name}"

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

        # 4. Numeric literals for numeric ports.
        # A REQUIRED numeric port may consume any unconsumed numeric (it must
        # bind or the pipeline fails). An OPTIONAL numeric port consumes one
        # only with lexical EVIDENCE: the literal's prompt context must
        # intersect the consuming cell's declared vocabulary (identity tokens,
        # port name/state tokens).
        if is_num:
            _int_only = (
                registry is not None
                and registry.is_subtype(t_name, "int")
                and not registry.is_subtype("float", t_name)
            )
            lit_clause_map = getattr(self, "literal_clause_map", {}) or {}
            cell_cl = getattr(self, "current_cell_clause_idx", None)
            num_prompt_clauses = len(getattr(self, "clause_spans", ()) or ())
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
                if (
                    not port_sig.required
                    and cell_cl is not None
                    and num_prompt_clauses > 1
                    and idx in lit_clause_map
                    and lit_clause_map[idx] != cell_cl
                ):
                    continue
                if not port_sig.required:
                    # Clause-bounded context: restrict evidence to the clause enclosing the literal
                    clause_chunk = self.prompt
                    clauses = CellTokenizer.split_prompt_clauses(self.prompt)
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
                self.used_indices.add(idx)
                return val

        # Structural Morphism Context
        port_role = str(getattr(port_sig, "port_role", None) or getattr(port_sig, "role", None) or "").strip().lower()
        is_path = _lattice_is_path_port(port_sig)
        path_roles = _declared_role_semantics("path_roles")
        is_path_role = (port_role in path_roles) if port_role else is_path
        sink_roles = _declared_role_semantics("sink_roles") | _declared_role_semantics("terminal_roles") | _declared_role_semantics("destination_roles")
        dest_states = _declared_role_semantics("materialization_states") | _declared_role_semantics("terminal_states")
        term_stages = {int(s) for s in _declared_role_semantics("terminal_stages") if str(s).isdigit()}
        is_sink_ctx = (
            (cell is not None and getattr(cell, "node_role", "") in sink_roles)
            or (cell is not None and not getattr(cell, "outputs", None))
            or (cell_stage is not None and cell_stage in term_stages)
            or port_role in sink_roles
            or str(getattr(port_sig, "state", "")).lower() in dest_states
        )
        source_roles = _declared_role_semantics("source_roles") | _declared_role_semantics("ingress_roles") | _declared_role_semantics("source_data_roles")
        source_states = _declared_role_semantics("source_states")
        ing_stages = {int(s) for s in _declared_role_semantics("ingress_stages") if str(s).isdigit()}
        is_source_ctx = (
            (cell is not None and getattr(cell, "node_role", "") in source_roles)
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
        _registry = TypeRegistry.get_instance()
        _is_str_port = _registry.is_subtype(str(getattr(port_sig, "type_name", "")).lower(), "str")
        if _port_state in _declared_role_semantics("condition_states"):
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
        # Direction-aware allocation: English marks asset roles with relational
        # prepositions ("FROM input.csv", "TO output.csv") — ingress (stage 1)
        # prefers source/unmarked assets, egress (stage 3) prefers
        # destination-marked assets, so a destination literal is never stolen
        # by the ingress when both directions exist in the prompt.
        def _asset_direction(pos: int) -> str:
            return ExecutionContext._asset_direction(self.prompt or "", pos)

        def _take_asset(prefer: str) -> Optional[str]:
            fallback: Optional[Tuple[int, str]] = None
            path_candidates: List[Tuple[int, str]] = []
            clause_match_candidate: Optional[Tuple[int, str]] = None
            current_c_idx = getattr(self, "current_cell_clause_idx", None)

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
                        if port_fmts and ext not in {f.lower() for f in port_fmts}:
                            continue
                        reg = TypeRegistry.get_instance()
                        port_t = str(getattr(port_sig, "type_name", "")).strip()

                        # Decompose generic path signature: File[Carrier, Format] or File[Modality]
                        inner_carrier = None
                        declared_fmt = None
                        if "[" in port_t and port_t.endswith("]"):
                            ctor, _, inner = port_t[:-1].partition("[")
                            if ctor.strip().lower() in _declared_role_semantics("path_type_names"):
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
                        if port_fmts:
                            if ext not in {f.lower() for f in port_fmts}:
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
                            _path_and_top = set(reg.get_path_port_tokens() or ()) | set(reg.get_verification_semantics("universal_top_types") or ())
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
                    direction = _asset_direction(lit_pos) if kind == "file_asset" else ""
                    lit_clause = self._get_clause_idx_for_pos(lit_pos)
                    in_current_clause = (current_c_idx is not None and lit_clause == current_c_idx)

                    if in_current_clause and (not prefer or direction == prefer or not direction):
                        self.used_indices.add(idx)
                        return json.dumps(val)

                    if in_current_clause and clause_match_candidate is None:
                        clause_match_candidate = (idx, val)

                    if prefer and direction == prefer:
                        self.used_indices.add(idx)
                        return json.dumps(val)
                    if fallback is None and direction != ("dest" if prefer == "src" else "src"):
                        fallback = (idx, val)
                    if matches_path:
                        path_candidates.append((idx, val))

            if clause_match_candidate is not None:
                self.used_indices.add(clause_match_candidate[0])
                return json.dumps(clause_match_candidate[1])

            if is_path_role and len(path_candidates) == 1:
                idx, val = path_candidates[0]
                self.used_indices.add(idx)
                return json.dumps(val)

            if fallback is not None:
                self.used_indices.add(fallback[0])
                return json.dumps(fallback[1])
            return None

        is_column_port = (
            getattr(port_sig, "binds", "") in _declared_role_semantics("key_binds")
            or str(getattr(port_sig, "state", "")).lower() in _declared_role_semantics("projection_port_states")
            or getattr(port_sig, "port_role", "") in _declared_role_semantics("projection_port_roles")
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
                    if val not in ("None", "none"):
                        return json.dumps(val) if not (val.startswith(("'", '"')) or val in ("True", "False")) else val

        # 6. Operational Parameter Extraction
        if is_operational and is_str:
            enum_vals = {str(e).strip().lower() for e in port_sig.enum_values} if getattr(port_sig, "enum_values", None) else None
            p_state = str(getattr(port_sig, "state", "") or "").lower()
            p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
            reg = TypeRegistry.get_instance()
            proj_tokens = reg.get_projection_tokens()
            reg_types = reg.get_registered_types()
            is_cell_key_port = _port_matches_declared_ref(cell, "key_port", port_sig)
            is_proj_port = (
                p_role in _declared_role_semantics("projection_port_roles")
                or p_role in _declared_role_semantics("projective_roles")
                or p_role in _declared_role_semantics("key_roles")
                or str(getattr(port_sig, "binds", "") or "").lower() in _declared_role_semantics("key_binds")
                or p_state in _declared_role_semantics("projection_port_states")
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
                    p_role in _declared_role_semantics("projective_roles")
                    or p_role in _declared_role_semantics("destination_roles")
                    or p_state in _declared_role_semantics("target_states")
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
                    or (kind == "quoted_str" and not ExecutionContext._is_path_string(val_str))
                    or (is_setter_port and bool(roles & (proj_tokens | reg.get_dest_port_tokens() | reg.get_egress_tokens())))
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
                or str(getattr(port_sig, "state", "")).lower() in _declared_role_semantics("projection_port_states")
                or (registry is not None and (registry.is_subtype(_tn, "collection") or registry.is_subtype(_tn, "sequence")))
            )
            _is_strict_str = (registry.is_subtype(_tn, "str") or registry.is_subtype(_tn, "scalar")) and _tn not in ("any", "*", "top", "") and not _is_collection
            _state_tokens: Set[str] = set()
            _raw_state = str(getattr(port_sig, "state", "") or "")
            if _raw_state.lower() not in ("any", "default", ""):
                _state_tokens = CellTokenizer.tokenize_identifier(_raw_state)
            if _is_strict_str and (_state_tokens or cell_tokens):
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens
                cand_indices = list(range(len(self.ordered_literals)))
                if self.current_cell_clause_idx is not None:
                    cand_indices.sort(key=lambda i: 0 if self.literal_clause_map.get(i) == self.current_cell_clause_idx else 1)
                _p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
                _is_setter = is_setter_port if "is_setter_port" in locals() else (
                    _p_role in _declared_role_semantics("projective_roles")
                    or _p_role in _declared_role_semantics("destination_roles")
                    or _raw_state in _declared_role_semantics("target_states")
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
        _tensor_sks = _declared_role_semantics("array_types")
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
                    or str(getattr(port_sig, "state", "")).lower() in _declared_role_semantics("projection_port_states")
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
                egress_tokens = TypeRegistry.get_instance().get_egress_tokens() | TypeRegistry.get_instance().get_dest_port_tokens()
                registered_types = TypeRegistry.get_instance().get_registered_types()

                matched_indices = []
                matched_vals = []
                cell_clause = getattr(self, "current_cell_clause_idx", None)
                lit_clause_map = getattr(self, "literal_clause_map", {})
                target_lits = getattr(self, "target_literals", set())

                all_available = [
                    idx for idx, (_, kind, val) in enumerate(self.ordered_literals)
                    if idx not in self.used_indices and kind in ("quoted_str", "identifier")
                    and str(val).lower() not in target_lits
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
            key_binds = set(registry.get_verification_semantics("key_binds") or ())
            proj_roles = (
                set(registry.get_projection_tokens())
                | set(registry.get_verification_semantics("projection_port_roles") or ())
                | set(registry.get_verification_semantics("key_roles") or ())
            )
            is_cell_key_port = _port_matches_declared_ref(cell, "key_port", port_sig)
            is_key_req = getattr(port_sig, "required", False) and (
                getattr(port_sig, "binds", "") in key_binds
                or str(getattr(port_sig, "port_role", "") or "").lower() in proj_roles
                or is_cell_key_port
                or bool(getattr(port_sig, "complement_of_port", None))
                or bool(getattr(port_sig, "key_from_upstream_role", None))
            )
            if is_key_req and def_str.strip("'\"") in ("", "[]", "()"):
                pass
            else:
                if def_str in ("True", "False", "None"):
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
        self.last_verification_contract: Optional[VerificationContract] = None
        self.last_runtime_aliases: Dict[str, str] = {}
        self._generic_port_cache: Optional[Tuple[Tuple[int, int], Set[str]]] = None
        try:
            from .unification_stages import (
                PortWiringStage,
                TypeCoercionStage,
                ScopeResolutionStage,
                TemplateInstantiationStage,
            )
        except (ImportError, ValueError):
            from unification_stages import (
                PortWiringStage,
                TypeCoercionStage,
                ScopeResolutionStage,
                TemplateInstantiationStage,
            )
        self.wiring_stage = PortWiringStage(self)
        self.coercion_stage = TypeCoercionStage(self)
        self.scope_stage = ScopeResolutionStage(self)
        self.template_stage = TemplateInstantiationStage(self)

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

    # ------------------------------------------------------------------
    # Declared bridge-morphism composition (Solution 3)
    # ------------------------------------------------------------------

    def _compose_declared_bridge_expr(
        self,
        src_var: str,
        src_type: str,
        dst_type: str,
        literal_bindings: Dict[str, str],
        dst_sig: Optional[Any] = None,
        ctx: Optional[ExecutionContext] = None,
        max_hops: int = 2,
    ) -> Optional[str]:
        """
        Composes an expression from DECLARED bridge morphisms connecting
        src_type to dst_type. Bridge morphisms are ordinary tree cells
        (node_role == 'bridge') whose single-statement template has the shape
        `{output_var} = <expression over its input ports>`.

        The engine contributes ONLY the composition mechanism: it reads the
        templates out of the loaded cells (which come from pluggable domain
        trees) and instantiates them generically. When no tree declares a
        morphism chain between the carriers, this returns None and the port
        remains unresolved — the engine never writes library-specific syntax
        (column projections, array conversions, ...) on its own.
        """
        orchestrator = getattr(self, "orchestrator", None)
        if orchestrator is None:
            try:
                from .lattice import LatticeOrchestrator
                orchestrator = LatticeOrchestrator.get_active_instance()
            except Exception:
                try:
                    from lattice import LatticeOrchestrator
                    orchestrator = LatticeOrchestrator.get_active_instance()
                except Exception:
                    orchestrator = None
        if orchestrator is None:
            return None
        registry = TypeRegistry.get_instance()

        bridge_cells = [
            c for c in (getattr(orchestrator, "loaded_cells", {}) or {}).values()
            if str(getattr(c, "node_role", "")).lower() == "bridge"
            or str(getattr(c, "node_type", "")).lower() == "tunnel"
            or (hasattr(orchestrator, "_is_valid_bridge_candidate") and orchestrator._is_valid_bridge_candidate(c))
        ]
        if not bridge_cells:
            return None

        def _accepts(port_sig: Any, have: str) -> bool:
            sig = getattr(port_sig, "signature", port_sig)
            pt = str(getattr(sig, "type_name", "") or getattr(port_sig, "type_name", "") or "").lower()
            have_l = (have or "").lower()
            if pt in ("any", "*", "top", ""):
                return True
            if not have_l:
                return False
            if registry.is_subtype(have_l, pt):
                return True
            ab = str(getattr(sig, "abstract_type", "") or getattr(port_sig, "abstract_type", "") or "").lower()
            if ab and ab not in ("any", "*", "top", ""):
                if registry.is_subtype(have_l, ab) or registry.is_subtype(ab, have_l) or ab == have_l:
                    return True
            if have_l in registry.get_abstract_carriers() and registry.is_subtype(pt, have_l):
                return True
            return False

        def _dst_accepts(have: str) -> bool:
            have_l = (have or "").lower()
            dst_l = str(dst_type or "").lower()
            if dst_l and not registry.is_declared_top(dst_l):
                if registry.is_subtype(have_l, dst_l):
                    return True
                if dst_l not in registry.get_abstract_carriers():
                    return False
            ab = str(getattr(dst_sig, "abstract_type", "") or "").lower() if dst_sig is not None else ""
            return bool(ab and ab not in ("any", "*") and registry.is_subtype(have_l, ab))

        def _template_rhs(cell: Any) -> Optional[Tuple[str, str, Dict[str, str]]]:
            tpl = (getattr(cell, "code_template", "") or "").strip()
            if not tpl or "\n" in tpl or tpl.count("=") != 1:
                return None
            lhs, rhs = tpl.split("=", 1)
            out_name = getattr(getattr(cell, "primary_output", None), "name", "") or "output_var"
            lhs_clean = lhs.strip().strip("{}")
            if lhs_clean != out_name and lhs_clean != "output_var" and lhs.strip() != "{" + out_name + "}":
                return None
            rhs = rhs.strip()
            placeholders = set(extract_template_placeholders(rhs))
            if not placeholders:
                return None
            out_decl = (getattr(cell, "outputs", {}) or {}).get(out_name)
            if out_decl is None:
                out_decl = getattr(cell, "primary_output", None)
                if out_decl is None and getattr(cell, "outputs", None):
                    out_decl = next(iter(cell.outputs.values()), None)
            out_type = ""
            if isinstance(out_decl, dict):
                out_type = str(out_decl.get("type_name", ""))
            elif out_decl is not None:
                out_type = str(getattr(out_decl, "type_name", ""))
            return rhs, out_name, {"output_type": out_type}

        def _literal_for(port_name: str, port_sig: Any) -> Tuple[Optional[str], Optional[str]]:
            # Match declared literal bindings by port name first, then by
            # tokenized overlap between the port's declared name/state and the
            # binding key (e.g. state "column_identifier" <-> key "column").
            if port_name in literal_bindings:
                return literal_bindings[port_name], port_name
            toks = CellTokenizer.tokenize_identifier(port_name)
            state = str(getattr(getattr(port_sig, "signature", port_sig), "state", "") or "")
            toks |= CellTokenizer.tokenize_identifier(state)
            proj_tokens = (
                set(registry.get_projection_tokens())
                | set(registry.get_verification_semantics("projection_port_names") or ())
            )
            for key, rendered in literal_bindings.items():
                if (CellTokenizer.tokenize_identifier(key) & toks) or (key in proj_tokens and (toks & proj_tokens)):
                    return rendered, key
            if not getattr(port_sig, "required", True):
                default_val = getattr(port_sig, "default_value", None)
                if default_val is not None:
                    return repr(default_val), None
                return "None", None
            return None, None

        frontier: List[Tuple[str, str, List[Any], Set[str]]] = [
            (str(src_type or "").lower(), str(src_var), [], set())
        ]
        seen_states: Set[Tuple[str, Tuple[str, ...]]] = {(str(src_type or "").lower(), ())}

        # 0-hop check: only valid if literal_bindings is empty.
        # If literal_bindings is specified, a bridge MUST consume the literals.
        if not literal_bindings and _dst_accepts(src_type):
            return str(src_var)

        proj_tokens = (
            set(registry.get_projection_tokens())
            | set(registry.get_verification_semantics("projection_port_names") or ())
        )
        has_proj_lit = bool(
            literal_bindings
            and any(
                CellTokenizer.tokenize_identifier(k) & proj_tokens
                for k in literal_bindings
            )
        )

        for _hop in range(max(1, max_hops)):
            next_frontier: List[Tuple[str, str, List[Any], Set[str]]] = []
            for cur_type, cur_expr, used_cells, consumed_keys in frontier:
                if used_cells and _dst_accepts(cur_type):
                    if not literal_bindings or (set(literal_bindings.keys()) <= consumed_keys):
                        if ctx is not None and hasattr(ctx, "composed_bridge_cells"):
                            ctx.composed_bridge_cells.extend(used_cells)
                        return cur_expr
                for cell in bridge_cells:
                    tpl = _template_rhs(cell)
                    if tpl is None:
                        continue
                    rhs, out_name, meta = tpl
                    placeholders = extract_template_placeholders(rhs)
                    out_type = (meta.get("output_type") or "").lower()
                    if not out_type:
                        continue
                    in_ports = list((getattr(cell, "inputs", {}) or {}).items())
                    if not in_ports:
                        continue

                    # Never use a 1-arg unprojected composite carrier bridge when
                    # literal_bindings specifies a projection key.
                    abstract_carriers = set(registry.get_abstract_carriers())
                    is_cur_composite = (
                        cur_type in abstract_carriers
                        or registry.is_abstract_carrier(cur_type)
                        or any(registry.is_subtype(cur_type, ac) for ac in abstract_carriers)
                    )
                    if has_proj_lit and is_cur_composite and len(in_ports) <= 1:
                        continue

                    # Try each input port as the chain carrier; every other
                    # required port must be satisfied from declared literals.
                    for carrier_name, carrier_sig in in_ports:
                        if not _accepts(carrier_sig, cur_type):
                            continue
                        subs: Dict[str, str] = {out_name: cur_expr, "output_var": cur_expr}
                        satisfiable = True
                        consumed_in_step: Set[str] = set()
                        for other_name, other_sig in in_ports:
                            if other_name == carrier_name:
                                continue
                            rendered, matched_k = _literal_for(other_name, other_sig)
                            if rendered is None:
                                satisfiable = False
                                break
                            subs[other_name] = rendered
                            if matched_k is not None:
                                consumed_in_step.add(matched_k)
                        if not satisfiable:
                            continue
                        new_consumed = consumed_keys | consumed_in_step
                        state_key = (out_type, tuple(sorted(new_consumed)))
                        if state_key in seen_states:
                            continue
                        subs[carrier_name] = cur_expr
                        for p in placeholders:
                            if p not in subs and p != out_name and p != "output_var":
                                subs[p] = cur_expr
                        composed = safe_substitute_template(rhs, subs)
                        leftover = set(extract_template_placeholders(composed))
                        if leftover:
                            continue
                        next_frontier.append((out_type, composed, used_cells + [cell], new_consumed))
                        seen_states.add(state_key)
                        break
            if not next_frontier:
                break
            frontier = next_frontier

        for cur_type, cur_expr, used_cells, consumed_keys in frontier:
            if used_cells and _dst_accepts(cur_type):
                if not literal_bindings or (set(literal_bindings.keys()) <= consumed_keys):
                    if ctx is not None and hasattr(ctx, "composed_bridge_cells"):
                        ctx.composed_bridge_cells.extend(used_cells)
                    return cur_expr
        return None

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
        target_roles_sem = set(registry.get_verification_semantics("target_roles") or ())
        target_states_sem = set(registry.get_verification_semantics("target_states") or ())
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
                and (allow_target_key or str(val).strip().lower() not in target_lits_low)
                and not ExecutionContext._is_path_string(str(val))
                for i, (_, kind, val) in enumerate(getattr(ctx, "ordered_literals", []))
            ):
                can_lit_key = True
            elif has_role_key:
                can_lit_key = True
            elif getattr(g_key, "default_value", None) is not None and str(g_key.default_value).strip("'\"") not in ("", "None", "none", "[]", "()"):
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
                ctx.current_cell_clause_idx = c_clause
                key_expr = ctx.resolve_literal_for_port(
                    g_key,
                    cell_stage=getattr(g_clone, "stage", None),
                    cell_inputs=getattr(g_clone, "inputs", None),
                    cell_tokens=getattr(g_clone, "token_set", set()),
                    cell=g_clone,
                )
                ctx.current_cell_clause_idx = saved_clause
            if not key_expr:
                gk_role = str(getattr(g_key, "port_role", "") or getattr(g_key, "binds", "") or "").strip().lower()
                rb_list = list((getattr(ctx, "role_bound_keys", None) or {}).get(gk_role, []) or [])
                if rb_list:
                    key_expr = json.dumps(rb_list[-1])
            if key_expr and str(key_expr).strip("'\"") not in ("", "None", "none", "[]", "()"):
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
        cfg_roles = (
            _declared_role_semantics("config_roles")
            | _declared_role_semantics("configuration_roles")
            | _declared_role_semantics("parameter_roles")
        )
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

            accepts_container = any(
                (not getattr(getattr(ip, "signature", ip), "is_top", lambda: False)())
                and not registry.is_declared_top(str(getattr(getattr(ip, "signature", ip), "type_name", "")).lower())
                and _shape_compatible(s_out, ip, producer_cell=setter, consumer_cell=consumer_cell)
                and unify(s_out_sig, getattr(ip, "signature", ip), accumulated_sigma) is not None
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

        cfg_roles = (
            _declared_role_semantics("config_roles")
            | _declared_role_semantics("configuration_roles")
            | _declared_role_semantics("parameter_roles")
        )
        data_roles = (
            _declared_role_semantics("dataflow_roles")
            | _declared_role_semantics("projective_roles")
            | _declared_role_semantics("strict_roles")
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
            if getattr(out_port_sig, "name", None):
                step_bindings[out_port_sig.name] = out_v
                ctx.var_port_map[(id(step_cell), out_port_sig.name)] = out_v
                if cell_port_vars is not None:
                    cell_port_vars[(step_cell.cell_id, out_port_sig.name)] = out_v
                    cell_port_vars[(id(step_cell), out_port_sig.name)] = out_v
            if cell_out_vars is not None:
                cell_out_vars[step_cell.cell_id] = out_v
                cell_out_vars[id(step_cell)] = out_v
            ctx.var_port_map[(id(step_cell), "output_var")] = out_v
            if cell_port_vars is not None:
                cell_port_vars[(step_cell.cell_id, "output_var")] = out_v
                cell_port_vars[(id(step_cell), "output_var")] = out_v
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

        term_roles = _declared_role_semantics("terminal_roles") | _declared_role_semantics("sink_roles")
        ing_roles = _declared_role_semantics("ingress_roles") | _declared_role_semantics("source_roles")
        term_states = _declared_role_semantics("terminal_states") | _declared_role_semantics("materialization_states")
        term_stages = {int(s) for s in _declared_role_semantics("terminal_stages") if str(s).isdigit()}
        ing_stages = {int(s) for s in _declared_role_semantics("ingress_stages") if str(s).isdigit()}

        for cell, bindings in pipeline_bindings:
            stage = getattr(cell, "stage", None)
            node_role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
            is_terminal = getattr(cell, "is_endable", False) or (stage in term_stages) or (node_role in term_roles)
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
            if p_clean and p_clean not in _declared_role_semantics("output_targets"):
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
            ctx.scope = {k: v for k, v in ctx.scope.items() if not (k.startswith("var_") or k in getattr(ctx, "pipeline_vars", set()))}
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
        ctx.var_counter = 0
        if hasattr(ctx, "_var_counter"):
            ctx._var_counter = 0
        if hasattr(ctx, "_prompt") and ctx._prompt:
            ctx.ordered_literals = ctx._extract_universal_literals(ctx._prompt)
            if hasattr(ctx, "sync_llm_slots"):
                ctx.sync_llm_slots()
            ctx.literal_clause_map = ctx._build_literal_clause_map(ctx._prompt, ctx.ordered_literals)

        accumulated_sigma = Substitution()
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
        if prompt_text and cells and allow_llm:
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm = ModelManager.get_instance()
                if mm.can_extract_slots() and not getattr(ctx, "_cell_slots_extracted", False):
                    self._extract_pipeline_slots_with_llm(cells, prompt_text, ctx)
                if mm.can_synthesize():
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
                                    _declared_role_semantics("assigned_value_binds")
                                    | _declared_role_semantics("assigned_value_roles")
                                    | _declared_role_semantics("projective_roles")
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
                if not c.outputs or getattr(c, "stage", None) == 3 or str(getattr(c, "node_role", "")).lower() in _declared_role_semantics("sink_roles"):
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
                                parsed_ast = ast.parse(cand_str, mode="eval")
                                var_names = {node.id for node in ast.walk(parsed_ast) if isinstance(node, ast.Name)}
                                if not var_names:
                                    # Constant literal proposed by LLM: must be grounded in prompt ordered literals
                                    valid_nums = {v for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("numeric", "int_literal", "float_literal")}
                                    if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
                                        val_str = str(int(cand_val)) if int(cand_val) == cand_val else str(cand_val)
                                        if val_str not in valid_nums:
                                            continue
                                    elif isinstance(cand_val, str) and cand_val not in ("True", "False", "None"):
                                        valid_strs = {v.lower() for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("quoted_str", "identifier", "file_asset")}
                                        if cand_val.lower() not in valid_strs:
                                            continue
                                if all(v in ctx.variables for v in var_names if not keyword.iskeyword(v)):
                                    if cand_str in ctx.variables:
                                        v_sig, _ = ctx.variables[cand_str]
                                        if _shape_compatible(v_sig, concrete_sig):
                                            u_cand = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                                            if u_cand is not None:
                                                cell_bindings[p_name] = cand_str
                                                accumulated_sigma = u_cand
                                                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "")
                                                if p_role in _declared_role_semantics("dataflow_roles"):
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
                                                if p_role in _declared_role_semantics("dataflow_roles"):
                                                    bound_producer = True
                            except Exception as _exc:
                                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

            # 1. Multi-Carrier DAG Scope Verification & Wire Resolution:
            # When idx > 0, verify that cell's inputs are satisfiable from the full
            # DAG ancestor frontier (available_cells = cells[:idx]).
            if idx > 0 and not is_zero_ary and not is_replica:
                # Honor planned_port_wires if provided by the planner DAG
                planned_wires = getattr(cell, "planned_port_wires", None)
                if planned_wires:
                    for p_name, wire_target in sorted(
                        planned_wires.items(),
                        key=lambda kv: 1 if getattr(cell.inputs.get(kv[0]), "complement_of_port", None) else 0,
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
                        key=lambda kv: 1 if getattr(cell.inputs.get(kv[0]), "complement_of_port", None) else 0,
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
                            # If `v_name` was already consumed (e.g. written into an optic container)
                            # and the active unconsumed container carries a key whose qualifiers
                            # satisfy this port's accepted states, defer to optic lifting from the container.
                            if (
                                not is_formal_step
                                and v_name in getattr(ctx, "consumed_vars", set())
                                and str(getattr(cell, "optic_role", "") or "").strip().lower() not in _declared_role_semantics("optic_roles")
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
                        projective_roles = _declared_role_semantics("projective_roles")
                        has_carrier = any(
                            registry.is_composite_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                            or registry.is_abstract_carrier(str(getattr(v_sig.signature, "type_name", "")).lower())
                            for v_sig, _ in getattr(ctx, "variables", {}).values()
                        )
                        c_role = str(getattr(cell, "node_role", "") or "").lower()
                        c_type = str(getattr(cell, "node_type", "") or "").lower()
                        strict_roles = _declared_role_semantics("strict_roles")
                        is_model_cell = (
                            c_role in strict_roles
                            or c_type in strict_roles
                        )
                        has_proj = any(
                            (
                                getattr(p, "port_role", "") in projective_roles
                                or getattr(p, "binds", "") in _declared_role_semantics("projective_binds")
                                or bool(getattr(p, "complement_of_port", None))
                                or (is_model_cell and getattr(p, "derived_role", "") in projective_roles)
                            )
                            for p in req_ports
                        )
                        all_req_bound = all(p_name in cell_bindings for p_name, p in cell.inputs.items() if p.required)
                        has_no_req_inputs = not any(getattr(p, "required", False) for p in cell.inputs.values())
                        if not (has_proj and has_carrier) and not (planned_wires and all_req_bound) and not has_no_req_inputs:
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
                cfg_roles = (
                    _declared_role_semantics("config_roles")
                    | _declared_role_semantics("configuration_roles")
                    | _declared_role_semantics("parameter_roles")
                )
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
                        return False

                data_roles = (
                    _declared_role_semantics("dataflow_roles")
                    | _declared_role_semantics("projective_roles")
                    | _declared_role_semantics("strict_roles")
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

                candidates = []

                for p_name, p in multi_ports:
                    p_role = getattr(p, "port_role", None) or getattr(p, "derived_role", "standard")
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
                        if not _shape_compatible(v_sig, p):
                            continue
                        u_p = unify(v_sig.signature, p.signature, accumulated_sigma)
                        if u_p is None:
                            continue

                        # Compute scale-free lexicographic rank for (p, v_name)
                        not_superseded = 1 if v_name not in getattr(ctx, "superseded_vars", set()) else 0
                        role_tier = 1
                        if p_role == v_role and p_role != "standard":
                            role_tier = 3
                        elif p_role != "standard" and v_role != "standard" and reg.is_subtype(v_role, p_role):
                            role_tier = 2

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
                        rank = (no_comp_prio, not_superseded, role_tier, is_prod, clause_prov_overlap, recency_idx, state_match, overlap)
                        candidates.append((rank, p_name, p, v_name))

                # Deterministic greedy assignment with recency tie-breaking (active wire in frontier)
                candidates.sort(key=lambda item: item[0], reverse=True)
                assigned_ports = set()
                assigned_vars = set()
                assigned_origins = set()
                best_assign = {}
                test_sub = accumulated_sigma

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
                    # Monoidal origin disjointness: avoid binding multiple join ports to the same origin wire
                    if v_origins and (v_origins & assigned_origins):
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
                        if not col_mismatch and _shape_compatible(prod_sig, prim_in, consumer_cell=cell):
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
                        if not is_formal_step and str(getattr(cell, "optic_role", "") or "").strip().lower() not in _declared_role_semantics("optic_roles"):
                            _p_r_t = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
                            _p_s_t = str(getattr(getattr(p_sig, "signature", p_sig), "state", "") or "").strip().lower()
                            _t_r_s = set(registry.get_verification_semantics("target_roles") or ())
                            _t_s_s = set(registry.get_verification_semantics("target_states") or ())
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
                        accumulated_sigma.bind(p_name, formatted_exp)
                        if _lattice_is_path_port(concrete_sig) and isinstance(exp_val, str):
                            ctx.target_literals.add(exp_val)

            # 3. Resolve auxiliary input ports (variable reuse / port sharing / parameters / literals).
            # REQUIRED ports are processed before optional ones so data ports claim
            # the prompt's shared literal channels (expressions, paths) first;
            # optional configuration knobs never steal them.
            cell_toks_for_projection = _cell_clause_tokens(cell)
            for p_name, p_sig in sorted(
                cell.inputs.items(),
                key=lambda kv: (not kv[1].required, 1 if getattr(kv[1], "complement_of_port", None) else 0),
            ):
                if p_name in cell_bindings:
                    continue  # Already bound

                # Substitute generics if type variable in p_sig
                concrete_sig = substitute_generics(p_sig, accumulated_sigma)

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
                        if p_sig.default_value is not None and str(p_sig.default_value) not in ("None", "none"):
                            val = _format_literal_value(p_sig.default_value, concrete_sig)
                            cell_bindings[p_name] = val
                            accumulated_sigma.bind(p_name, val)
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
                if llm_slot_val is None and hasattr(ctx, "llm_slots") and isinstance(ctx.llm_slots, dict):
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

                if llm_slot_val is not None and not _is_valid_slot_value(llm_slot_val, concrete_sig):
                    llm_slot_val = None

                if llm_slot_val is not None and not p_sig.required:
                    # Optional port: if numeric, verify it actually came from user prompt
                    if isinstance(llm_slot_val, (int, float)) and not isinstance(llm_slot_val, bool):
                        prompt_str = str(getattr(ctx, "prompt", "") or prompt_text or "")
                        valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}
                        val_str = str(int(llm_slot_val)) if int(llm_slot_val) == llm_slot_val else str(llm_slot_val)
                        if val_str not in valid_nums and val_str not in prompt_str:
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
                            accumulated_sigma.bind(p_name, opt_val)
                        continue

                # Optional parameters with declared default: use prompt literal or omit/default
                if not p_sig.required and p_sig.default_value is not None:
                    if llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        _consume_matching_literals(ctx, llm_slot_val)
                        cell_bindings[p_name] = formatted_val
                        accumulated_sigma.bind(p_name, formatted_val)
                        continue

                    resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()), cell=cell)
                    if resolved_literal is not None:
                        cell_bindings[p_name] = resolved_literal
                        accumulated_sigma.bind(p_name, resolved_literal)
                    elif str(p_sig.default_value) in ("None", "none"):
                        # Port default is None: omit from the emitted call
                        cell_bindings[p_name] = None
                    else:
                        val = str(p_sig.default_value)
                        c_tname = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        c_is_str = _is_string_carrier(p_sig, concrete_sig)
                        if c_is_str and not (val.startswith(("'", '"')) or val in ("None", "True", "False")):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        accumulated_sigma.bind(p_name, val)
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
                        accumulated_sigma.bind(p_name, formatted_val)
                        continue

                    _lit = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()), cell=cell)
                    if _lit is not None:
                        cell_bindings[p_name] = _lit
                        accumulated_sigma.bind(p_name, _lit)
                    else:
                        cell_bindings[p_name] = None
                    continue

                # A0. Target / complement-partner optic lifting:
                # If this required port is a supervised target or complement partner (e.g. X has complement_of_port=y)
                # and the active composite container holds a target/setter-written key that is NOT yet isolated
                # in any unconsumed compatible carrier, project that key (with carrier/qualifier bridges) first.
                if p_sig.required and not is_formal_step and str(getattr(cell, "optic_role", "") or "").strip().lower() not in _declared_role_semantics("optic_roles"):
                    _p_role_tgt = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
                    _p_state_tgt = str(getattr(concrete_sig, "state", "") or "").strip().lower()
                    _tgt_roles_s = set(registry.get_verification_semantics("target_roles") or ())
                    _tgt_states_s = set(registry.get_verification_semantics("target_states") or ())
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
                val_roles = set(registry.get_verification_semantics("assigned_value_roles") or ())
                val_port_names = set(registry.get_verification_semantics("assigned_value_port_names") or ())
                is_assigned_val = (
                    str(getattr(p_sig, "binds", "") or "").lower() in val_roles
                    or str(getattr(p_sig, "port_role", "") or "").lower() in val_roles
                    or str(p_name).lower() in val_port_names
                )

                target_lits = getattr(ctx, "target_literals", set())
                var_origins_map = getattr(ctx, "var_origins", {})
                any_var_has_target = bool(target_lits and any(
                    any(str(orig).lower() in target_lits for orig in var_origins_map.get(vn, set()))
                    for vn in ctx.variables.keys()
                ))

                in_t_low = str(getattr(concrete_sig, "type_name", "") or "").strip().lower()
                in_is_callable = (
                    str(p_role_in or "").strip().lower() == "functional_operator"
                    or in_t_low.startswith("callable")
                )

                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    # Heterogeneous products project member-wise; the whole container
                    # is never wired into a single data port.
                    if _is_product(v_sig):
                        continue
                    if v_name in already_bound_vars:
                        continue
                    v_t_low = str(getattr(v_sig.signature, "type_name", "") or "").strip().lower()
                    if in_is_callable and not v_t_low.startswith("callable"):
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
                        target_roles = set(reg.get_verification_semantics("target_roles") or ())
                        sink_roles = (
                            set(reg.get_verification_semantics("sink_roles") or ())
                            | set(reg.get_verification_semantics("destination_roles") or ())
                            | set(reg.get_verification_semantics("terminal_roles") or ())
                        )
                        target_states = (
                            set(reg.get_verification_semantics("target_states") or ())
                            | set(reg.get_verification_semantics("terminal_states") or ())
                            | set(reg.get_verification_semantics("materialization_states") or ())
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
                        and str(getattr(cell, "optic_role", "") or "").strip().lower() not in _declared_role_semantics("optic_roles")
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
                    if candidate_vars[0][0][1] >= 0:
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
                    accumulated_sigma.bind(p_name, formatted_val)
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
                            accumulated_sigma.bind(p_name, proj_val)
                        continue

                # B. Check typestate-driven literal resolution from prompt or LLM slots
                if not is_formal_step:
                    if llm_slot_val is not None and _is_valid_slot_value(llm_slot_val, concrete_sig):
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
                        _consume_matching_literals(ctx, llm_slot_val)
                        cell_bindings[p_name] = formatted_val
                        accumulated_sigma.bind(p_name, formatted_val)
                        continue

                    resolved_literal = ctx.resolve_literal_for_port(concrete_sig, cell_stage=cell.stage, cell_inputs=cell.inputs, cell_tokens=getattr(cell, "token_set", set()), cell=cell)
                    if resolved_literal is not None:
                        cell_bindings[p_name] = resolved_literal
                        accumulated_sigma.bind(p_name, resolved_literal)
                        continue

                # C. Check default value declared in tree
                if p_sig.default_value is not None:
                    val = str(p_sig.default_value)
                    is_key_req = p_sig.required and _is_key_or_complement_port(cell, p_sig, registry)
                    if is_key_req and val.strip("'\"") in ("", "[]", "()"):
                        # A required key/projection/complement port cannot be grounded by an empty default
                        pass
                    else:
                        c_tname = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        c_is_str = _is_string_carrier(p_sig, concrete_sig)
                        if c_is_str and not (val.startswith(("'", '"')) or val in ("None", "True", "False")):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        accumulated_sigma.bind(p_name, val)
                        continue

                # C2. Pointed monoidal default η_τ : 1 → τ (semi-closed literal parameters only)
                if not is_formal_step and p_sig.required and p_sig.default_value is None:
                    eta_val = _pointed_default(p_sig, cell)
                    if eta_val is not None:
                        cell_bindings[p_name] = eta_val
                        accumulated_sigma.bind(p_name, eta_val)
                        continue

                # D. Declared-domain enum grounding (reflection-driven, zero domain hardcodes)
                p_domain = getattr(concrete_sig, "domain", "") or ""
                if not is_formal_step and p_domain and getattr(ctx, "prompt", ""):
                    enum_val = ctx._resolve_enum_constant(p_domain, concrete_sig.state)
                    if enum_val is not None:
                        cell_bindings[p_name] = enum_val
                        accumulated_sigma.bind(p_name, enum_val)
                        continue

                # E. Unresolved port fallback: check grounded LLM pipeline bindings or targeted LLM synthesis
                if p_name not in cell_bindings or cell_bindings[p_name] in (None, UNRESOLVED_PORT):
                    if p_name in step_llm_bindings:
                        cand_val = step_llm_bindings[p_name]
                        if cand_val is not None:
                            cand_str = str(cand_val).strip()
                            if cand_str:
                                try:
                                    parsed_ast = ast.parse(cand_str, mode="eval")
                                    var_names = {node.id for node in ast.walk(parsed_ast) if isinstance(node, ast.Name)}
                                    valid_cand = True
                                    if not var_names:
                                        valid_nums = {v for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("numeric", "int_literal", "float_literal")}
                                        if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
                                            val_str = str(int(cand_val)) if int(cand_val) == cand_val else str(cand_val)
                                            if val_str not in valid_nums:
                                                valid_cand = False
                                        elif isinstance(cand_val, str) and cand_val not in ("True", "False", "None"):
                                            valid_strs = {v.lower() for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("quoted_str", "identifier", "file_asset")}
                                            if cand_val.lower() not in valid_strs:
                                                valid_cand = False
                                    if valid_cand and all(v in ctx.variables for v in var_names if not keyword.iskeyword(v)):
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

                    if p_sig.required:
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
                                accumulated_sigma.bind(p_name, fallback_expr)
                                continue

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

                # E.2 Unresolved REQUIRED port: try schema-driven boundary I/O placeholder if permitted, else fail loudly
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
                if c_kq_pre and c_optic_role_pre not in _declared_role_semantics("read_optic_roles"):
                    out_p_pre = getattr(cell, "primary_output", None) or (next(iter(cell.outputs.values())) if getattr(cell, "outputs", None) else None)
                    out_tn_pre = str(getattr(getattr(out_p_pre, "signature", out_p_pre), "type_name", "") or "").lower()
                    if not (registry.is_composite_carrier(out_tn_pre) or registry.is_abstract_carrier(out_tn_pre)):
                        for bv_in in bound_in_vars:
                            for (kv_var, _), kq_set in c_kq_pre.items():
                                if kv_var == bv_in and kq_set:
                                    prop_quals |= set(kq_set)

            # Register output port(s) in context for future steps
            if not hasattr(ctx, "var_port_map") or ctx.var_port_map is None:
                ctx.var_port_map = {}
            if len(cell.outputs) > 1:
                # Multi-output cell (e.g. train_test_split, load_wine, cv2.threshold, subplots)
                # Declare each individual output port in the execution context and bind in template
                first_out_var = None
                prim_out = cell.primary_output
                if _is_sink_cell(cell):
                    producer_var = None
                    for out_name in cell.outputs:
                        cell_bindings[out_name] = None
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
            elif len(cell.outputs) == 1:
                # Single output cell
                if _is_sink_cell(cell):
                    producer_var = None
                    if cell.primary_output and cell.primary_output.name:
                        cell_bindings[cell.primary_output.name] = None
                else:
                    concrete_out = substitute_generics(cell.primary_output, accumulated_sigma)
                    if prop_quals:
                        concrete_out = _enrich_port_signature_qualifiers(concrete_out, prop_quals)
                    # If the cell performs in-place mutation on a receiver, alias the output
                    # to the receiver: the receiver is the cell's declared receiver-role port,
                    # falling back to the primary input binding (declared-first, name-last).
                    if getattr(cell, "mutation_type", "pure") == "in_place":
                        receiver_var = None
                        rec_roles = _declared_role_semantics("receiver_roles")
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
                            for fr in getattr(ctx, "optic_frames", ()) or ():
                                if fr.get("closed"):
                                    continue
                                if (s_cont_v and fr.get("container_var") == s_cont_v) or (s_focus_v and fr.get("focus_var") == s_focus_v):
                                    fr["closed"] = True
                                    fr["closed_var"] = current_out_var
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
                        else:
                            if is_carrier and hasattr(ctx, "container_key_qualifiers") and isinstance(ctx.container_key_qualifiers, dict):
                                for b_in_v in bound_in_vars:
                                    for (kv_var, k_key), kq_val in list(ctx.container_key_qualifiers.items()):
                                        if kv_var == b_in_v and kq_val and (current_out_var, k_key) not in ctx.container_key_qualifiers:
                                            ctx.container_key_qualifiers[(current_out_var, k_key)] = set(kq_val)
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
                            is_non_mutating_consumer = (
                                is_scalar_out
                                or out_str_k == "handle"
                                or c_optic_role in _declared_role_semantics("optic_roles")
                                or getattr(cell, "stage", None) == 3
                                or c_nr_low in (_declared_role_semantics("sink_roles") | _declared_role_semantics("terminal_roles") | _declared_role_semantics("estimator_roles"))
                            )
                            if (
                                current_out_var
                                and not is_scalar_out
                                and out_str_k != "handle"
                                and c_optic_role not in _declared_role_semantics("accessor_optic_roles")
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

                        c_node_role = str(getattr(cell, "node_role", "") or "").strip().lower()
                        if (
                            getattr(cell, "stage", None) == 2
                            and out_t
                            and c_optic_role not in _declared_role_semantics("read_optic_roles")
                            and not getattr(cell, "projection", None)
                            and c_node_role != "bridge"
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

        self.last_pipeline_bindings = pipeline_bindings
        self.last_pipeline_typed_bindings = [getattr(c, "typed_bindings", {}) for c, _ in pipeline_bindings]
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
        for v in bindings.values():
            if isinstance(v, str):
                v_clean = v.strip().strip("'\"")
                if v_clean.startswith("var_") or v_clean.startswith("out_"):
                    active_vars.add(v_clean)

        ph_map: Dict[str, str] = {}
        known_placeholders = set(inputs.keys()) | set(bindings.keys()) | _declared_role_semantics("output_targets")
        ast_ready = template
        for name in extract_template_placeholders(template):
            if known_placeholders and name not in known_placeholders:
                continue
            ph_id = f"_nstl_ph_{name}"
            ph_map[ph_id] = name
            ast_ready = ast_ready.replace(f"{{{name}}}", ph_id)
        try:
            parsed = ast.parse(ast_ready)
        except SyntaxError:
            res = template
            for k, v in bindings.items():
                if v is not None:
                    res = res.replace(f"{{{k}}}", str(v))
            return res

        def _build_value_ast(orig_name: str, val: Any, p_sig: Optional[Any], in_store_ctx: bool = False) -> ast.AST:
            if val is UNRESOLVED_PORT or (isinstance(val, str) and val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
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
            if s in ("True", "False"):
                return ast.Constant(value=(s == "True"))
            if s == "None":
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

            # Check if value is or references in-scope variables or expressions
            unquoted = s
            if (unquoted.startswith("'") and unquoted.endswith("'")) or (unquoted.startswith('"') and unquoted.endswith('"')):
                unquoted = unquoted[1:-1].strip()

            if unquoted in active_vars or unquoted.startswith("var_") or unquoted.startswith("out_"):
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

            # Compute true unbound free names (excluding lambda parameters and comprehension targets)
            def _unbound_names(node: ast.AST, bound: Set[str]) -> Set[str]:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    return set() if node.id in bound else {node.id}
                if isinstance(node, ast.Lambda):
                    lam_bound = set(bound)
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
                        res_names.update(_unbound_names(d, bound))
                    res_names.update(_unbound_names(node.body, lam_bound))
                    return res_names
                if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    comp_bound = set(bound)
                    res_names = set()
                    for gen in node.generators:
                        res_names.update(_unbound_names(gen.iter, comp_bound))
                        for t_node in ast.walk(gen.target):
                            if isinstance(t_node, ast.Name):
                                comp_bound.add(t_node.id)
                        for if_c in gen.ifs:
                            res_names.update(_unbound_names(if_c, comp_bound))
                    if isinstance(node, ast.DictComp):
                        res_names.update(_unbound_names(node.key, comp_bound))
                        res_names.update(_unbound_names(node.value, comp_bound))
                    else:
                        res_names.update(_unbound_names(node.elt, comp_bound))
                    return res_names
                res_names = set()
                for child in ast.iter_child_nodes(node):
                    res_names.update(_unbound_names(child, bound))
                return res_names

            # Try parsing s as an evaluated Python expression (handles array projections, bridges, lambdas, indexed lookups)
            try:
                parsed_expr = ast.parse(s, mode="eval").body
                free_names = _unbound_names(parsed_expr, set())
                if free_names and all(
                    (v in active_vars or v.startswith("var_") or v.startswith("out_") or _is_resolvable_module_or_builtin(v))
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
                            val_node = _build_value_ast(orig_name, orig_name, p_sig)
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
                            kw_val_node = _build_value_ast(orig_name, orig_name, p_sig)
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
                    val = bindings.get(orig_name)
                    if val is not None:
                        node.slice = _build_value_ast(orig_name, val, p_sig)
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
                return node

            def visit_FunctionDef(self, node):
                if node.name in ph_map:
                    orig = ph_map[node.name]
                    if orig in bindings and bindings[orig] is not None:
                        val = str(bindings[orig]).strip().strip("'\"")
                        if val.isidentifier():
                            node.name = val
                            active_vars.add(val)
                elif isinstance(node.name, str) and node.name.isidentifier():
                    active_vars.add(node.name)
                for a in getattr(node.args, "args", []):
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
            res = template
            for k, v in bindings.items():
                if v is not None:
                    res = res.replace(f"{{{k}}}", str(v))
            return res

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
                term_roles = _declared_role_semantics("terminal_roles") | _declared_role_semantics("sink_roles")
                if idx == len(current) - 1 or role in term_roles or not outputs:
                    continue
                # Protect explicit calculations/metrics (e.g. mean, metrics, aggregations) that witness a clause,
                # but pure intermediate projections/slices (like PD_SELECT_COLUMN) whose output is never used
                # are dead extraction bridges and must be pruned to avoid anti-causal or unused variable errors.
                proj_roles = _declared_role_semantics("projection_roles")
                is_pure_projection = (
                    getattr(cell, "projection", None) is not None
                    or (getattr(cell, "mutation_type", "") == "pure" and role in proj_roles)
                )
                if not is_pure_projection:
                    if getattr(cell, "matched_clause_idx", None) is not None or getattr(cell, "is_goal", False):
                        continue
                out_vars = {v for k, v in bnd.items() if k in outputs and isinstance(v, str)}
                if not out_vars:
                    continue
                referenced_later: Set[str] = set()
                for _, other_bnd in current[idx + 1:]:
                    for val in other_bnd.values():
                        if isinstance(val, str):
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
                var_name = _semantic_var_name(c, var_counter)
                var_counter += 1
                for o_name, o_sig in (getattr(c, "outputs", {}) or {}).items():
                    o_type = getattr(getattr(o_sig, "signature", o_sig), "type_name", "") or ""
                    out_names.append(f"{var_name} ({o_type})")

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

        query = (
            f"User Prompt: {prompt}\n\n"
            f"Current Cell: {cell.cell_id}\n"
            f"Template: {getattr(cell, 'code_template', '')}\n"
            f"Required Unbound Port: '{port_name}' (expected type: {getattr(concrete_sig, 'type_name', '')}, role: {getattr(port_sig, 'port_role', '')})\n"
            f"In-scope available variables:\n" + "\n".join(in_scope_vars) + "\n\n"
            f"Specify ONLY the exact variable name (e.g. var_0), projection or subscript (e.g. var_0['member'], var_0[0]), or literal that satisfies port '{port_name}'.\n"
            "Respond in JSON: {\"binding\": \"expression\"}"
        )
        schema = {
            "type": "object",
            "properties": {
                "binding": {"type": "string", "description": "The exact variable name, projection, or expression to bind"}
            },
            "required": ["binding"]
        }
        try:
            raw = mm.generate_text(
                query,
                max_tokens=128,
                schema=schema,
                system_prompt="You are a dataflow compiler. Provide the exact variable or projection expression to satisfy the port."
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
                    # Validate candidate against scope and typing
                    try:
                        parsed_ast = ast.parse(cand_str, mode='eval')
                        var_names = {node.id for node in ast.walk(parsed_ast) if isinstance(node, ast.Name)}
                        if all(v in ctx.variables for v in var_names if not keyword.iskeyword(v)):
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
                                u = unify(rec.resulting_signature, concrete_sig.signature, Substitution())
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
        Replaces brittle regex scanning by extracting values for candidate non-carrier ports
        directly from the user prompt when an LLM is available.
        """
        if not prompt or not cells:
            return
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if not mm.can_extract_slots():
                return
        except Exception:
            return

        ctx._cell_slots_extracted = True
        registry = TypeRegistry.get_instance()
        try:
            clauses = CellTokenizer.split_prompt_clauses(prompt)
        except Exception:
            clauses = [prompt]

        step_candidates: Dict[str, Tuple[int, Cell, Dict[str, Dict[str, Any]]]] = {}
        key_proj_roles = _declared_role_semantics("projection_port_roles") | _declared_role_semantics("key_roles")
        key_binds = _declared_role_semantics("key_binds")
        data_carrier_roles = (
            _declared_role_semantics("dataflow_roles")
            | _declared_role_semantics("projective_roles")
            | _declared_role_semantics("strict_roles")
        )
        prompt_toks_low = set(CellTokenizer.tokenize_prompt(prompt)) if prompt else set()

        for step_idx, c in enumerate(cells):
            if not hasattr(c, "inputs") or not c.inputs:
                continue
            c_ports = {}
            for p_name, p_sig in c.inputs.items():
                p_role = str(getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or "").strip().lower()
                t_name = (getattr(p_sig, "type_name", "") or "").lower()
                is_path_p = _lattice_is_path_port(p_sig)
                is_key_proj = (p_role in key_proj_roles) or (str(getattr(p_sig, "binds", "") or "").lower() in key_binds)
                is_req = bool(getattr(p_sig, "required", False))
                if not (is_path_p or is_key_proj):
                    if p_role in data_carrier_roles:
                        continue
                    if registry.is_composite_carrier(t_name) or registry.is_abstract_carrier(t_name):
                        continue
                if not is_req:
                    # Only include optional ports when the prompt actually mentions the port name
                    p_toks = set(CellTokenizer.tokenize_identifier(p_name))
                    if not (p_toks and (p_toks & prompt_toks_low)):
                        continue
                sk = str(getattr(p_sig, "structure_kind", "") or "").lower()
                cr = getattr(p_sig, "canonical_rank", None)
                try:
                    t_base = TypeTerm.from_string(t_name).base if t_name else ""
                except Exception:
                    t_base = ""
                is_coll = (
                    sk != "scalar"
                    and (
                        sk == "collection"
                        or (cr is not None and cr >= 1 and not registry.is_subtype(t_name, "tensor"))
                        or (not registry.is_subtype(t_name, "str") and (registry.is_subtype(t_name, "collection") or registry.is_subtype(t_name, "sequence")))
                        or (bool(t_base) and not registry.is_subtype(t_base, "str") and (registry.is_subtype(t_base, "collection") or registry.is_subtype(t_base, "sequence")))
                    )
                )
                c_ports[p_name] = {
                    "type": t_name or "any",
                    "required": is_req,
                    "default": str(getattr(p_sig, "default_value", None)),
                    "enum_values": getattr(p_sig, "enum_values", None),
                    "is_collection": is_coll,
                    "is_path": is_path_p,
                    "port_role": p_role,
                    "sig": p_sig,
                }
            if c_ports:
                step_key = f"step_{step_idx}:{c.cell_id}"
                step_candidates[step_key] = (step_idx, c, c_ports)

        if not step_candidates:
            return

        # Build verbatim grounding vocabulary from prompt tokens and already-extracted literals
        prompt_token_case: Dict[str, str] = {}
        for raw_w in prompt.split():
            clean_w = raw_w.strip(".,;:()[]{}\"'`!?=")
            if clean_w:
                prompt_token_case.setdefault(clean_w.lower(), clean_w)
        ordered_lit_case: Dict[str, str] = {}
        for _, _, lit_v in getattr(ctx, "ordered_literals", []):
            lv = str(lit_v).strip()
            if lv:
                ordered_lit_case.setdefault(lv.lower(), lv)

        # Ordered distinct verbatim candidate tokens (preserving prompt order)
        all_verbatim_ordered: List[str] = []
        _seen_v_low: Set[str] = set()
        for lv in ordered_lit_case.values():
            if lv.lower() not in _seen_v_low:
                _seen_v_low.add(lv.lower())
                all_verbatim_ordered.append(lv)
        for pv in prompt_token_case.values():
            if pv.lower() not in _seen_v_low:
                _seen_v_low.add(pv.lower())
                all_verbatim_ordered.append(pv)

        # Collect operation/method tokens from non-optic cells so data identifiers rank above action verbs in clauses
        step_action_toks: Set[str] = set()
        for oc in cells:
            oc_optic = str(getattr(oc, "optic_role", "") or "").strip().lower()
            oc_role = str(getattr(oc, "node_role", "") or "").strip().lower()
            if oc_optic in _declared_role_semantics("optic_roles") or oc_role == "bridge":
                continue
            oc_id_toks = set(CellTokenizer.tokenize_identifier(oc.cell_id))
            oc_dom = str(getattr(oc, "domain_name", "") or "").strip().lower()
            if oc_dom:
                oc_id_toks -= set(CellTokenizer.tokenize_identifier(oc_dom))
            step_action_toks.update(t.lower() for t in oc_id_toks if len(t) > 1)

        def _port_enum_candidates(port_info: Dict[str, Any], clause_txt: str) -> List[str]:
            p_sig_o = port_info.get("sig")
            concrete_o = getattr(p_sig_o, "signature", p_sig_o) if p_sig_o else None
            is_path_port_o = bool(port_info.get("is_path"))
            clause_toks_o: List[str] = []
            if clause_txt:
                for rw in clause_txt.split():
                    cw = rw.strip(".,;:()[]{}\"'`!?=")
                    if cw:
                        clause_toks_o.append(cw.lower())
            clause_set_o = set(clause_toks_o)

            valid_cands: List[str] = []
            for tok in all_verbatim_ordered:
                if keyword.iskeyword(tok.lower()):
                    continue
                is_p_str = ExecutionContext._is_path_string(tok)
                if is_path_port_o and not is_p_str:
                    continue
                if not is_path_port_o and is_p_str:
                    continue
                if concrete_o and not _is_valid_slot_value(tok, concrete_o):
                    continue
                valid_cands.append(tok)
            if not valid_cands:
                return []
            # Order clause tokens first, preferring data identifiers over step operation words.
            # When the active clause already supplies data identifiers, keep only those so
            # unrelated prose words from earlier clauses (e.g. "named", "file") don't leak in.
            in_cl_data = [t for t in valid_cands if t.lower() in clause_set_o and t.lower() not in step_action_toks]
            in_cl_act = [t for t in valid_cands if t.lower() in clause_set_o and t.lower() in step_action_toks]
            out_cl = [t for t in valid_cands if t.lower() not in clause_set_o]
            if in_cl_data and not is_path_port_o:
                return in_cl_data[:48]
            return (in_cl_data + in_cl_act + out_cl)[:48]

        slot_props: Dict[str, Any] = {}
        cell_summary = []
        for step_key, (step_idx, c, ports) in step_candidates.items():
            cl_idx = getattr(c, "matched_clause_idx", None)
            cl_txt = clauses[cl_idx].strip() if (isinstance(cl_idx, int) and 0 <= cl_idx < len(clauses)) else ""
            port_props: Dict[str, Any] = {}
            req_ports_step: List[str] = []
            port_descs = []
            for p, info in ports.items():
                e_vals = info.get("enum_values")
                is_req_p = bool(info.get("required"))
                req_tag = "required" if is_req_p else "optional"
                if is_req_p:
                    req_ports_step.append(p)
                t_low = str(info.get("type") or "").lower()
                if e_vals:
                    base_types: Any = [str(ev) for ev in e_vals]
                    enum_schema: Dict[str, Any] = {"enum": base_types + ([None] if not is_req_p else [])}
                    port_props[p] = enum_schema
                    port_descs.append(f"{p} ({req_tag}, options={base_types})")
                elif info.get("is_collection"):
                    cands = _port_enum_candidates(info, cl_txt)
                    item_schema: Dict[str, Any] = {"enum": cands} if cands else {"type": "string"}
                    port_props[p] = (
                        {"type": "array", "items": item_schema}
                        if is_req_p
                        else {"type": ["array", "null"], "items": item_schema}
                    )
                    port_descs.append(f"{p} ({req_tag}, list)")
                else:
                    if registry.is_subtype(t_low, "bool"):
                        prim_t = "boolean"
                        port_props[p] = {"type": prim_t} if is_req_p else {"type": [prim_t, "null"]}
                        port_descs.append(f"{p} ({req_tag}, bool)")
                    elif registry.is_subtype(t_low, "numeric"):
                        prim_t = "number"
                        port_props[p] = {"type": prim_t} if is_req_p else {"type": [prim_t, "null"]}
                        port_descs.append(f"{p} ({req_tag}, number)")
                    else:
                        cands = _port_enum_candidates(info, cl_txt)
                        if cands:
                            port_props[p] = {"enum": cands + ([None] if not is_req_p else [])}
                            port_descs.append(f"{p} ({req_tag}, e.g. from clause)")
                        else:
                            port_props[p] = {"type": "string"} if is_req_p else {"type": ["string", "null"]}
                            port_descs.append(f"{p} ({req_tag})")
            step_obj: Dict[str, Any] = {
                "type": "object",
                "properties": port_props,
            }
            if req_ports_step:
                step_obj["required"] = req_ports_step
            slot_props[step_key] = step_obj
            cl_note = f' [clause: "{cl_txt}"]' if cl_txt else ""
            feeds_note = ""
            if step_idx + 1 < len(cells) and (
                str(getattr(c, "optic_role", "") or "").lower() == "getter"
                or str(getattr(c, "node_role", "") or "").lower() == "bridge"
            ):
                feeds_note = f" (feeds -> {cells[step_idx + 1].cell_id})"
            cell_summary.append(f"- {step_key}{cl_note}{feeds_note} -> {', '.join(port_descs)}")

        schema = {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "description": "Mapping from step key to port_name and extracted verbatim literal value",
                    "properties": slot_props,
                    "required": list(slot_props.keys()),
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "General hyperparameters or arguments extracted from prompt",
                    "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                }
            },
            "required": ["slots"]
        }

        query = (
            f"User Prompt: {prompt}\n\n"
            f"For each step below, select the exact identifier, column name, or file path referenced in that step's [clause]:\n"
            + "\n".join(cell_summary) + "\n\n"
            "Rules:\n"
            "1. Select the verbatim identifier/column/file name belonging to each step's [clause].\n"
            "2. Do NOT output type names, generic words like 'column', or operation/action words matching step names (select data identifiers instead; reuse the active data identifier when an extraction feeds a downstream transform).\n"
            "3. Only return null for optional ports not mentioned in the prompt.\n"
            "Return valid JSON only."
        )

        def _ground_str_token(val_s: str, sig_obj: Any) -> Optional[str]:
            clean = str(val_s).strip().strip("'\"")
            if not clean:
                return None
            c_low = clean.lower()
            declared_enums = getattr(sig_obj, "enum_values", None) if sig_obj else None
            if declared_enums:
                for ev in declared_enums:
                    if str(ev).strip().lower() == c_low:
                        return str(ev)
                return None
            if c_low in ordered_lit_case:
                return ordered_lit_case[c_low]
            if c_low in prompt_token_case:
                return prompt_token_case[c_low]
            return None

        try:
            raw = mm.generate_text(
                query,
                max_tokens=512,
                schema=schema,
                system_prompt="You are a precise dataflow parameter extractor. Copy exact parameter literals, column/variable identifiers, file paths, and options verbatim from the prompt."
            )
            if raw and raw.strip():
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    slots = parsed.get("slots", {})
                    valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("numeric", "int_literal", "float_literal")}
                    proj_roles = _declared_role_semantics("projection_port_roles") | _declared_role_semantics("key_roles")
                    proj_states = _declared_role_semantics("projection_port_states")
                    extracted_vars: List[str] = []

                    if isinstance(slots, dict):
                        cid_counts: Dict[str, int] = {}
                        for _, c_obj in enumerate(cells):
                            cid_counts[c_obj.cell_id] = cid_counts.get(c_obj.cell_id, 0) + 1

                        for k, v in slots.items():
                            if not isinstance(v, dict):
                                if v is not None:
                                    if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                                        ctx.hyperparameters = {}
                                    ctx.hyperparameters[k] = v
                                continue

                            step_info = step_candidates.get(k)
                            if step_info is not None:
                                step_idx, c_match, ports_info = step_info
                            else:
                                c_match = next((cl for cl in cells if cl.cell_id == k), None)
                                step_idx = next((si for si, cl in enumerate(cells) if cl.cell_id == k), None)
                                ports_info = {}

                            for p_name, val in v.items():
                                if val is None:
                                    continue
                                p_sig = c_match.inputs.get(p_name) if c_match and hasattr(c_match, "inputs") else None
                                concrete_sig = getattr(p_sig, "signature", p_sig) if p_sig else None
                                is_proj = False
                                if p_sig is not None:
                                    pr = str(getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or "").lower()
                                    ps = str(getattr(concrete_sig, "state", "") or "").lower()
                                    is_proj = (
                                        pr in proj_roles
                                        or ps in proj_states
                                        or getattr(p_sig, "binds", "") in _declared_role_semantics("key_binds")
                                    )

                                accepted_val: Any = None
                                if isinstance(val, list):
                                    grounded_items = []
                                    for item in val:
                                        if isinstance(item, str):
                                            g_it = _ground_str_token(item, concrete_sig)
                                            if g_it is not None:
                                                grounded_items.append(g_it)
                                                if is_proj and g_it.isidentifier():
                                                    extracted_vars.append(g_it)
                                    if grounded_items:
                                        accepted_val = grounded_items
                                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                                    val_str = str(int(val)) if int(val) == val else str(val)
                                    if val_str in valid_nums:
                                        accepted_val = val
                                elif isinstance(val, bool):
                                    accepted_val = val
                                elif isinstance(val, str):
                                    g_str = _ground_str_token(val, concrete_sig)
                                    if g_str is not None:
                                        accepted_val = g_str
                                        if is_proj and g_str.isidentifier():
                                            extracted_vars.append(g_str)

                                if accepted_val is None:
                                    continue
                                if concrete_sig and not _is_valid_slot_value(accepted_val, concrete_sig):
                                    continue

                                if step_idx is not None:
                                    ctx.step_llm_slots.setdefault(step_idx, {})[p_name] = accepted_val
                                if c_match is not None and (cid_counts.get(c_match.cell_id, 0) <= 1 or p_name not in ctx.llm_slots.get(c_match.cell_id, {})):
                                    ctx.llm_slots.setdefault(c_match.cell_id, {})[p_name] = accepted_val

                        # Causal read-before-define guard for optic getters:
                        # A getter at step i must not read a key whose first setter definition occurs
                        # at a later step j > i without appearing in step i's clause / prior steps.
                        setter_first: Dict[str, Tuple[int, Optional[int]]] = {}
                        for s_i, c_obj in enumerate(cells):
                            if str(getattr(c_obj, "optic_role", "") or "").strip().lower() != "setter":
                                continue
                            s_map = ctx.step_llm_slots.get(s_i, {})
                            if not s_map:
                                continue
                            k_p = _resolve_cell_port(c_obj, getattr(c_obj, "key_port", None), prefer="inputs")
                            k_pn = getattr(k_p, "name", None) if k_p is not None else None
                            k_val = s_map.get(k_pn) if k_pn else None
                            if isinstance(k_val, str) and k_val.strip():
                                kl = k_val.strip().lower()
                                if kl not in setter_first:
                                    setter_first[kl] = (s_i, getattr(c_obj, "matched_clause_idx", None))

                        defined_keys_before: Set[str] = set()
                        prior_getter_vals: List[Tuple[int, str]] = []
                        for s_i, c_obj in enumerate(cells):
                            orole = str(getattr(c_obj, "optic_role", "") or "").strip().lower()
                            s_map = ctx.step_llm_slots.get(s_i, {})
                            if not s_map:
                                continue
                            k_p = _resolve_cell_port(c_obj, getattr(c_obj, "key_port", None), prefer="inputs")
                            k_pn = getattr(k_p, "name", None) if k_p is not None else None
                            if not k_pn:
                                continue
                            k_val = s_map.get(k_pn)
                            if not isinstance(k_val, str) or not k_val.strip():
                                continue
                            kl = k_val.strip().lower()
                            if orole == "setter":
                                defined_keys_before.add(kl)
                                continue
                            if orole != "getter":
                                continue
                            setter_info = setter_first.get(kl)
                            if setter_info is not None and setter_info[0] > s_i and kl not in defined_keys_before:
                                g_cl_idx = getattr(c_obj, "matched_clause_idx", None)
                                s_cl_idx = setter_info[1]
                                g_cl_txt = clauses[g_cl_idx].strip() if (isinstance(g_cl_idx, int) and 0 <= g_cl_idx < len(clauses)) else ""
                                g_cl_words = [
                                    rw.strip(".,;:()[]{}\"'`!?=").lower()
                                    for rw in g_cl_txt.split()
                                    if rw.strip(".,;:()[]{}\"'`!?=")
                                ]
                                in_own_cl = kl in g_cl_words
                                # Allow getter+setter in the same clause (in-place column transform) only if
                                # that key literally appears in the getter's clause.
                                if (not in_own_cl) or (s_cl_idx is not None and g_cl_idx is not None and g_cl_idx < s_cl_idx):
                                    replacement_k: Optional[str] = None
                                    for _, p_gk in reversed(prior_getter_vals):
                                        if p_gk.lower() in g_cl_words and p_gk.lower() not in {sk for sk, sv in setter_first.items() if sv[0] > s_i and sk not in defined_keys_before}:
                                            replacement_k = p_gk
                                            break
                                    if replacement_k is None and g_cl_txt:
                                        p_info = step_candidates.get(f"step_{s_i}:{c_obj.cell_id}", (s_i, c_obj, {}))[2].get(k_pn)
                                        if p_info:
                                            for cand_t in _port_enum_candidates(p_info, g_cl_txt):
                                                ct_l = cand_t.lower()
                                                if ct_l in g_cl_words and ct_l not in step_action_toks and ct_l not in setter_first and ct_l in ordered_lit_case:
                                                    replacement_k = cand_t
                                                    break
                                    if replacement_k is None and prior_getter_vals:
                                        replacement_k = prior_getter_vals[-1][1]
                                    if replacement_k:
                                        s_map[k_pn] = replacement_k
                                        k_val = replacement_k
                                        kl = k_val.strip().lower()
                            defined_keys_before.add(kl)
                            prior_getter_vals.append((s_i, str(k_val).strip()))

                    if extracted_vars:
                        existing_vars = list(ctx.llm_slots.get("variables") or [])
                        for ev in extracted_vars:
                            if ev not in existing_vars:
                                existing_vars.append(ev)
                        ctx.llm_slots["variables"] = existing_vars
                        if hasattr(ctx, "sync_llm_slots"):
                            ctx.sync_llm_slots()

                    hparams = parsed.get("hyperparameters", {})
                    if isinstance(hparams, dict):
                        if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                            ctx.hyperparameters = {}
                        for hk, hv in hparams.items():
                            if hv is not None:
                                ctx.hyperparameters[hk] = hv
        except Exception as e:
            logger.debug(f"[UNIFICATION] LLM slot extraction skipped or failed: {e}")

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
        p_name_cur = getattr(mm, "current_profile_name", "")
        if not (mm.profile and p_name_cur not in ("0", "A", "D") and mm.can_synthesize()):
            return pipeline_bindings, accum_sigma

        prompt_txt = getattr(active_ctx, "_prompt", "") or getattr(self.context, "_prompt", "")
        resolved_any = False
        rem_unres = []
        for cid, p_name in list(active_ctx.unresolved_ports):
            cell_obj = next((c for c in cell_objs if c.cell_id == cid), None)
            p_sig = cell_obj.inputs.get(p_name) if cell_obj else None
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
                    if val is not None:
                        active_ctx.parameters[p_name] = val
                        active_ctx.target_literals.add(str(val))
                        resolved_any = True
                        continue
            except Exception as _ex:
                logger.debug(f"[UNIFICATION] LLM missing parameter extraction failed for {cid}.{p_name}: {_ex}")
            rem_unres.append((cid, p_name))

        if resolved_any:
            active_ctx.unresolved_ports = rem_unres
            res = self.unify_pipeline(cell_objs, active_ctx, allow_llm=False)
            if not res.is_bottom() and isinstance(res, Success):
                pipeline_bindings = res.value
                accum_sigma = res.sigma
                self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
                self.last_runtime_aliases = dict(getattr(active_ctx, "runtime_aliases", {}) or {})

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
                f"Code synthesis refused: pipeline contains unresolved ports: [{ports_str}]"
            )

        # Ensure no binding contains an unresolved sentinel
        for cell, bnd in pipeline_bindings:
            for p_name, val in bnd.items():
                if val is UNRESOLVED_PORT or (isinstance(val, str) and val in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                    raise UnresolvedPlaceholderError(
                        f"Code synthesis refused: port '{p_name}' of cell '{cell.cell_id}' has unresolved value '{val}'"
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

        # Collect dependencies recursively
        deps: List[str] = []
        reg_aliases = TypeRegistry.get_instance().get_all_aliases()

        def collect_deps(c: Cell):
            all_deps = list(getattr(c, "dependencies", [])) + list(getattr(c, "imports", []))
            for dep in all_deps:
                dep_clean = dep.strip()
                if not dep_clean:
                    continue
                if dep_clean.startswith(("import ", "from ")):
                    if dep_clean not in deps:
                        deps.append(dep_clean)
                    continue

                base_mod = dep_clean.split(".")[0]
                tpl = getattr(c, "code_template", "") or ""
                has_attr_call = f"{base_mod}." in tpl or f"{dep_clean}." in tpl
                found_alias = None
                for alias, target in reg_aliases.items():
                    if target == dep_clean or target == base_mod:
                        if f"{alias}." in tpl:
                            has_attr_call = True
                            found_alias = alias
                            break

                if has_attr_call:
                    if found_alias and found_alias != dep_clean:
                        stmt = f"import {dep_clean} as {found_alias}"
                    else:
                        stmt = f"import {dep_clean}"
                    if stmt not in deps:
                        deps.append(stmt)
                else:
                    # Template references symbols directly (e.g. LogisticRegression(...) or train_test_split(...))
                    symbols = _get_module_symbols(dep_clean)
                    sym_map = {s.lower(): s for s in symbols}
                    words = set(tokenize_alphanumeric(tpl))
                    matched = {sym_map[w] for w in words if w in sym_map}

                    # Structural fallback: if cell_id subcomponent appears in template
                    cid = getattr(c, "cell_id", "") or ""
                    if cid.startswith(dep_clean + "."):
                        sub_ident = cid[len(dep_clean) + 1:].split(".")[0]
                        if sub_ident and sub_ident.lower() in words:
                            matched.add(sub_ident)

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
        if deps:
            try:
                _dep_ast = ast.parse("\n".join(deps))
                for _dn in ast.walk(_dep_ast):
                    if isinstance(_dn, ast.Import):
                        for _al in _dn.names:
                            known_vars.add(_al.asname or _al.name.split(".")[0])
                    elif isinstance(_dn, ast.ImportFrom):
                        for _al in _dn.names:
                            if _al.name != "*":
                                known_vars.add(_al.asname or _al.name)
            except Exception:
                pass
        if active_ctx is not None:
            if hasattr(active_ctx, "variables"):
                known_vars.update(active_ctx.variables.keys())
            if hasattr(active_ctx, "scope"):
                known_vars.update(active_ctx.scope.keys())
            if hasattr(active_ctx, "var_sources"):
                known_vars.update(active_ctx.var_sources.keys())

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

        # Declared egress aliasing: when the prompt explicitly declares a
        # target sink ("store the results into Z"), the declaration is
        # honored at the RUNTIME namespace level (sandbox aliases Z to the
        # terminal pipeline variable). The emitted source is never rewritten
        # with a synthetic assignment — that would fabricate a sink morphism
        # that no declared cell provides, and would let verification observe
        # a binding the dataflow never performed.
        if active_ctx is not None and getattr(active_ctx, "prompt", None) and pipeline_bindings:
            target_ident = ExecutionContext._extract_target_sink(active_ctx.prompt)
            if target_ident:
                out_var = None
                def _is_var_assignment(line_str: str, var_name: str) -> bool:
                    if "=" not in line_str:
                        return False
                    lhs = line_str.split("=")[0].strip()
                    if lhs == var_name:
                        return True
                    parts = [p.strip() for p in lhs.split(",")]
                    return var_name in parts

                for c_prev, b_prev in reversed(pipeline_bindings):
                    cand_var = b_prev.get("output_var") or (b_prev.get(c_prev.primary_output.name) if getattr(c_prev, "primary_output", None) else None)
                    if cand_var and any(_is_var_assignment(line, str(cand_var)) for line in code_lines):
                        out_var = cand_var
                        break
                if out_var and str(out_var) != str(target_ident):
                    declare_alias = getattr(active_ctx, "declare_runtime_alias", None)
                    if callable(declare_alias):
                        declare_alias(target_ident, str(out_var))
                        self.last_runtime_aliases = dict(getattr(active_ctx, "runtime_aliases", {}) or {})
                        if ctx is not None and active_ctx is not ctx:
                            ctx.runtime_aliases = dict(self.last_runtime_aliases)

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
                            if b_val in active_ctx.variables or b_val.startswith("var_"):
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
            prior_vars = {v for v in prior_vars if not (isinstance(v, str) and (v in pipeline_vars or v in ctx_pipe_vars or v.startswith("var_")))}

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
        try:
            tree = ast.parse(code)
        except Exception as e:
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

        unbound_roots = sorted(used_roots - bound)
        if not unbound_roots:
            return code

        registered_aliases = TypeRegistry.get_instance().get_all_aliases()
        import importlib.util
        injected: List[str] = []
        for root in unbound_roots:
            root_clean = root.strip()
            root_lower = root_clean.lower()
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
        try:
            tree = ast.parse(code)
        except Exception as e:
            raise EmissionLivenessError(f"Emitted code has invalid syntax: {e}") from e

        import builtins
        global_scope: Set[str] = set(dir(builtins))
        global_scope.update({
            "__file__", "__name__", "__doc__", "__package__",
            "__spec__", "__path__", "__loader__", "__annotations__"
        })
        if initial_scope:
            global_scope.update(initial_scope)
        if cells:
            for c in cells:
                cell_obj = c[0] if isinstance(c, tuple) else c
                tpl = getattr(cell_obj, "code_template", "") or ""
                if tpl:
                    global_scope.update(_extract_template_callables(tpl))

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
                        target_set.add(n.id)

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
                    check_entry = {
                        "cell_id": cell.cell_id,
                        "target_var": target_var,
                        "target_port": target_name,
                        "property": getattr(p_obj, "property", None),
                        "operator": getattr(p_obj, "operator", "=="),
                        "value": getattr(p_obj, "value", None),
                        "expression": getattr(p_obj, "expression", None),
                        "description": getattr(p_obj, "description", None),
                    }
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
        imports = set()
        if chain_nodes:
            for node in chain_nodes:
                for dep in getattr(node, "dependencies", []):
                    dep_str = dep.strip()
                    if dep_str:
                        if not (dep_str.startswith("import ") or dep_str.startswith("from ")):
                            dep_str = f"import {dep_str}"
                        imports.add(dep_str)
        import_block = "\n".join(sorted(list(imports)))
        if code_text:
            if import_block:
                return f"{import_block}\n\n{code_text}".strip()
            return code_text.strip()
        return sorted(list(imports))

    @classmethod
    def validate_synthesis(cls, cell_dict: Dict[str, Any], expected_inputs: str, expected_outputs: str, trees_dir: str = "trees") -> bool:
        """Verifies whether a synthesized cell's inputs and outputs unify with required types."""
        in_spec = cell_dict.get("inputs", {})
        out_spec = cell_dict.get("outputs", {})
        actual_in = in_spec.get("type_name") if isinstance(in_spec, dict) else str(in_spec)
        actual_out = out_spec.get("type_name") if isinstance(out_spec, dict) else str(out_spec)
        return types_unify(expected_inputs, actual_in) and types_unify(expected_outputs, actual_out)


_semantic_var_name = UnificationGate._semantic_var_name

# =====================================================================
# Compatibility Helpers and Error Classes
# =====================================================================

class UnificationFailure(Exception):
    """Raised when monadic unification fails to find a valid substitution."""
    pass


class UnresolvedPlaceholderError(UnificationFailure):
    """Raised when a placeholder cannot be resolved."""
    pass


class EmissionLivenessError(UnificationFailure):
    """Raised when emitted code references variables that have not been defined or imported."""
    pass


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

            for out_name, out_sig in earlier_cell.outputs.items():
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

                    wire_key = (id(earlier_cell), out_name)
                    candidates.append(((comp_prio, score), p_name, earlier_cell, out_name, wire_key, s_wire, out_sig.signature))

        # 2. Context variables if available
        if context is not None:
            avail_cell_ids = {id(c) for c in available_cells}
            for v_name, (v_sig, _) in context.variables.items():
                v_cell = getattr(context, "var_sources", {}).get(v_name)
                if v_cell is not None and id(v_cell) in avail_cell_ids:
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
                    score = _wire_rank(p_type, p_state, in_role, v_type, v_state, v_role,
                                       cand_clause, v_clause, _order.index(v_name) - len(_order))

                    v_cell = getattr(context, "var_sources", {}).get(v_name)
                    wire_key = v_name
                    candidates.append(((comp_prio, score), p_name, v_cell, v_name, wire_key, s_wire, v_sig.signature))

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
        strict_roles = _declared_role_semantics("strict_roles")
        c_role = str(getattr(cand, "node_role", "") or "").lower()
        c_type = str(getattr(cand, "node_type", "") or "").lower()
        is_model_cand = (
            c_role in strict_roles
            or c_type in strict_roles
        )
        has_explicit_port_role = bool(getattr(p_sig, "port_role", None)) or bool(getattr(p_sig, "binds", None))
        p_role = str(getattr(p_sig, "port_role", None) or (getattr(p_sig, "derived_role", "") if is_model_cand else "") or "").lower()
        p_binds = str(getattr(p_sig, "binds", "") or "").lower()
        rec_names = _declared_role_semantics("receiver_port_names")
        carrier_binds = (
            _declared_role_semantics("carrier_binds")
            | _declared_role_semantics("receiver_roles")
        )
        projective_roles = _declared_role_semantics("projective_roles")
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
                and (
                    registry.is_subtype(t_name, "str")
                    or registry.is_subtype(t_name, "int")
                    or registry.is_subtype(t_name, "float")
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

    if cand.inputs and bound_count == 0 and getattr(cand, "stage", None) != 1 and getattr(cand, "node_type", "") != "constructor":
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
    try:
        return ast.parse(expr_str, mode="eval").body
    except SyntaxError:
        return None


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


def analyze_property_contracts(
    pipeline_bindings: List[Tuple[Any, Dict[str, Any]]], orchestrator: Any = None
) -> List[Dict[str, Any]]:
    """
    Static propagation of tree-declared value properties.
    Propagates known properties along the bound dataflow (reading cell.typed_bindings first
    for wire and projection lookups, and propagating cell.postconditions and preserves:<prop>
    effects across variables and keyed carrier assignments).
    """
    try:
        reg = TypeRegistry.get_instance()
    except Exception:
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
        except Exception:
            return []
    props: Dict[str, Props] = {}
    member_props: Dict[Tuple[str, Any], Props] = {}
    issues: List[Dict[str, Any]] = []

    for cell, bindings in pipeline_bindings:
        out = bindings.get("output_var")
        typed_bindings = getattr(cell, "typed_bindings", {}) or {}

        # 1. consumer preconditions against what is KNOWN
        for pre in (getattr(cell, "preconditions", None) or []):
            op = _prop_cond_field(pre, "operator")
            if op not in ("==", "!="):
                continue
            port = _prop_cond_field(pre, "target")
            prop = _prop_cond_field(pre, "property")
            want = _prop_cond_field(pre, "value")
            if port not in bindings or not prop:
                continue
            have = None
            rec = typed_bindings.get(port)
            if rec is not None:
                if rec.kind in ("wire", "variable") and rec.root_var:
                    have = props.get(rec.root_var, {}).get(prop)
                elif rec.kind == "projection" and rec.root_var and rec.projection:
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
            if have is None:
                have = _prop_known(bindings[port], props, member_props).get(prop)
            if have is None or have[0] is None:
                continue
            satisfied = (have[0] == want) if op == "==" else (have[0] != want)
            if not satisfied:
                suggestions: List[str] = []
                if orchestrator is not None:
                    for c in getattr(orchestrator, "loaded_cells", {}).values():
                        if any(_prop_cond_field(q, "property") == prop and _prop_cond_field(q, "operator") == "=="
                               and _prop_cond_field(q, "value") == want and _prop_cond_field(q, "target") in ("output_var", None)
                               for q in (getattr(c, "postconditions", None) or [])):
                            suggestions.append(c.cell_id)
                issues.append({"cell": cell.cell_id, "port": port, "property": prop, "required": want,
                               "required_operator": op, "known": have[0], "bound": str(bindings[port]),
                               "established_by": have[1], "established_note": have[2],
                               "bridge_candidates": sorted(set(suggestions))[:6]})

        if not out:
            continue

        # 2. producer postconditions
        mine: Props = {}
        # Seed known properties from output port typestate
        out_port = getattr(cell, "primary_output", None)
        if out_port is None and getattr(cell, "outputs", None):
            out_port = cell.outputs.get("output_var") or next(iter(cell.outputs.values()))
        if out_port is not None:
            out_sig = getattr(out_port, "signature", out_port)
            out_state = getattr(out_sig, "state", None) or getattr(out_port, "state", None)
            if out_state:
                state_props = reg.get_state_properties(out_state)
                for pk, pv in state_props.items():
                    if pv is not None:
                        mine[pk] = (pv, cell.cell_id, f"typestate:{out_state}")
            port_rank = getattr(out_port, "rank", None)
            if port_rank is not None and "ndim" not in mine:
                mine["ndim"] = (port_rank, cell.cell_id, "port:rank")
            shape_c = getattr(out_port, "shape_contract", None)
            if isinstance(shape_c, dict) and "ndim" in shape_c and "ndim" not in mine:
                mine["ndim"] = (shape_c["ndim"], cell.cell_id, "port:shape_contract")

        explicit_post_props: Set[str] = set()
        for post in (getattr(cell, "postconditions", None) or []):
            p_op = _prop_cond_field(post, "operator")
            p_target = _prop_cond_field(post, "target")
            p_prop = _prop_cond_field(post, "property")
            if p_op == "==" and p_target in ("output_var", None) and p_prop:
                mine[p_prop] = (_prop_cond_field(post, "value"), cell.cell_id, str(_prop_cond_field(post, "description", "") or ""))
                explicit_post_props.add(p_prop)

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
                            elif rec.kind == "projection" and rec.root_var and rec.projection:
                                k = rec.projection.get("key")
                                if isinstance(k, str):
                                    kn = member_props.get((rec.root_var, k), {})
                        if kn is None:
                            kn = _prop_known(bindings[port_name], props, member_props)
                        if prop in kn and prop not in mine:
                            mine[prop] = kn[prop]
                            break
        if mine:
            props[out] = {**props.get(out, {}), **mine}

        # 3. member writes
        key_binds = _declared_role_semantics("key_binds")
        val_binds = _declared_role_semantics("assigned_value_binds") | _declared_role_semantics("assigned_value_roles")
        carrier_binds = _declared_role_semantics("carrier_binds")
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
                        # Functorial Monoidal Property Lifting: constituent member properties transfer to container
                        # unless overridden by this cell's declared postconditions/typestate.
                        for prop_name, prop_val in copied_pr.items():
                            if prop_name in mine:
                                continue
                            if isinstance(prop_val, (tuple, list)) and prop_val[0] is True:
                                props[out] = {**props.get(out, {}), prop_name: prop_val}
    return issues


analyze = analyze_property_contracts


