"""
src/unification.py - STRATA (Neuro-Symbolic Topological Lattice)
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
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, Generic, TypeVar, FrozenSet

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
        is_path_port as _lattice_is_path_port,
    )
    from .tokenizer import CellTokenizer, normalize_token
except (ImportError, ValueError):
    from lattice import (
        AlgebraicSignature, PortSignature, Cell, MacroCell, TypeRegistry,
        ABSTRACT_CARRIERS, UNRESOLVED_PORT, GENERIC_TYPE_VARIABLE_NAMES,
        is_path_port as _lattice_is_path_port,
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


def _wire_rejection(cell: Any, p_name: str, port: Any, v_name: str, ctx: Any, registry: Any = None,
                    *args: Any, **kwargs: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Universal admissibility test for "may variable `v_name` feed port `p_name` of `cell`?".
    Checks universal dataflow availability and typestate compatibility without tabular assumptions.
    """
    if not ctx or not hasattr(ctx, "variables"):
        return None
    if v_name not in ctx.variables:
        return ("unknown_variable", {"v_name": v_name, "port": p_name})

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
            reg = registry if registry is not None else TypeRegistry.get_instance()
            is_scalar = (
                c_rank == 0
                or str_kind == "scalar"
                or (reg is not None and reg.is_subtype(t_name, "scalar"))
            )
            if is_scalar:
                return ("config_port_rejects_dataflow_wire", {"v_name": v_name, "port": p_name})

        v_entry = ctx.variables[v_name]
        v_sig = v_entry[0] if isinstance(v_entry, tuple) else v_entry
        v_actual_sig = getattr(v_sig, "signature", v_sig)
        p_actual_sig = getattr(port, "signature", port)
        if not _shape_compatible(v_actual_sig, port):
            return ("shape_incompatible", {"v_name": v_name, "port": p_name, "rejected_sig": str(v_actual_sig)})
        res = unify(v_actual_sig, p_actual_sig)
        if res is None:
            return ("type_incompatible", {"v_name": v_name, "port": p_name, "rejected_sig": str(v_actual_sig)})

    return None


def _cell_consumes_name_literal(cell: Any, ctx: Optional[Any] = None) -> bool:
    """True when the cell declares a port that receives a NAME/VALUE-to-store
    (binds key_binds / assigned_value_binds), or any cell in its clause does:
    its clause literals may be output names."""
    try:
        bind_keys = _declared_role_semantics("key_binds") | _declared_role_semantics("assigned_value_binds")
        if any(getattr(p, "binds", None) in bind_keys
               for p in getattr(cell, "inputs", {}).values()):
            return True
        if ctx is not None:
            pipeline = getattr(ctx, "current_pipeline", None) or getattr(ctx, "pipeline", None)
            c_clause = getattr(cell, "matched_clause_idx", None)
            if pipeline and c_clause is not None:
                for other in pipeline:
                    if getattr(other, "matched_clause_idx", None) == c_clause:
                        if any(getattr(p, "binds", None) in bind_keys
                               for p in getattr(other, "inputs", {}).values()):
                            return True
        return False
    except Exception:
        return True


def _update_variable_provenance(
    ctx: ExecutionContext,
    out_var: str,
    cell: Any,
    cell_bindings: Dict[str, Any],
) -> None:
    """
    Updates variable provenance records using typed binding records and declared properties (R5-7).
    Eliminates all substring/bracket parsing of bound strings.
    """
    ctx.var_parents.setdefault(out_var, set())
    ctx.var_origins.setdefault(out_var, set())
    try:
        from .lattice import TypeRegistry
        _prov_reg = TypeRegistry.get_instance()
    except Exception:
        try:
            from lattice import TypeRegistry
            _prov_reg = TypeRegistry.get_instance()
        except Exception:
            _prov_reg = None

    p_out = getattr(cell, "primary_output", None) or (next(iter(cell.outputs.values())) if getattr(cell, "outputs", None) else None)
    out_sig = getattr(p_out, "signature", p_out) if p_out else None
    out_tn = str(getattr(getattr(out_sig, "signature", out_sig), "type_name", "") or getattr(out_sig, "type_name", "")).lower()
    is_out_table = bool(_prov_reg and (_prov_reg.is_composite_carrier(out_tn) or _prov_reg.is_abstract_carrier(out_tn)))

    var_sigs_curr = {v: sig_tuple[0] for v, sig_tuple in getattr(ctx, "variables", {}).items()}
    for p_n, p_val in cell_bindings.items():
        if not p_val or p_val is UNRESOLVED_PORT:
            continue
        p_sig_n = getattr(cell, "inputs", {}).get(p_n) or getattr(cell, "outputs", {}).get(p_n)
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
            ctx.var_parents[out_var].add(rec.root_var)
            # A column PROJECTION carries only the projected column(s); inheriting the whole table's origins made
            # `var['X']` look derived from every column the table's lineage ever touched, so column-level
            # provenance gates could never tell X from Y.
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
            proj_roles = _declared_role_semantics("projection_port_roles")
            proj_names = _declared_role_semantics("projection_port_names")
            if (
                p_binds in bind_keys
                or p_role in proj_roles
                or p_n.lower() in proj_names
            ):
                val_raw = rec.value_expr
                if isinstance(val_raw, (list, tuple, set)):
                    for item in val_raw:
                        cl_it = str(item).strip("'\"")
                        if cl_it and not cl_it.isdigit():
                            ctx.var_origins[out_var].add(cl_it)
                elif isinstance(val_raw, str) and val_raw.strip().startswith("[") and val_raw.strip().endswith("]"):
                    try:
                        import ast as _ast
                        parsed_list = _ast.literal_eval(val_raw.strip())
                        if isinstance(parsed_list, (list, tuple, set)):
                            for item in parsed_list:
                                cl_it = str(item).strip("'\"")
                                if cl_it and not cl_it.isdigit():
                                    ctx.var_origins[out_var].add(cl_it)
                        else:
                            lit_clean = str(val_raw).strip("'\"")
                            if lit_clean and not lit_clean.isdigit():
                                ctx.var_origins[out_var].add(lit_clean)
                    except Exception:
                        lit_clean = str(val_raw).strip("'\"")
                        if lit_clean and not lit_clean.isdigit():
                            ctx.var_origins[out_var].add(lit_clean)
                else:
                    lit_clean = str(val_raw).strip("'\"")
                    if lit_clean and not lit_clean.isdigit():
                        ctx.var_origins[out_var].add(lit_clean)

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
_SENTENCE_CONNECTIVES: FrozenSet[str] = frozenset()
_ASCENDING_HINTS: FrozenSet[str] = frozenset()
_DESCENDING_HINTS: FrozenSet[str] = frozenset()
_ORDER_FLAG_NAMES: FrozenSet[str] = frozenset()
_ORDER_FLAG_STATES: FrozenSet[str] = frozenset()
_PREPOSITION_TRIGGERS: FrozenSet[str] = frozenset()
_CUE_WORDS: FrozenSet[str] = frozenset()


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

    def __init__(self, members: List[Tuple[int, str]], role_tokens: FrozenSet[str]):
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


def _declared_role_semantics(key: str) -> FrozenSet[Any]:
    """Declared role/effect/carrier vocabulary looked up by semantic key.
    The WORDS live in the trees' `verification_semantics` declarations; the
    engine never carries role-name literals of its own."""
    try:
        items = []
        for r in TypeRegistry.get_instance().get_verification_semantics(key):
            if isinstance(r, (list, tuple)):
                items.append(tuple(str(x).strip().lower() for x in r if str(x).strip()))
            elif str(r).strip():
                items.append(str(r).strip().lower())
        return frozenset(items)
    except Exception:
        return frozenset()


def _normalize_quals(quals: Any) -> FrozenSet[Tuple[str, ...]]:
    if not quals:
        return frozenset()
    res = []
    for q in quals:
        if isinstance(q, (list, tuple, set, frozenset)):
            res.append(tuple(str(x) for x in q))
        else:
            res.append((str(q),))
    return frozenset(res)


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

    # Universal strict role gating: if either role is declared strict,
    # reject mismatched bindings unless one is subtype of the other or both are strict.
    strict_roles = reg.get_verification_semantics("strict_roles")
    if r_port in strict_roles:
        if r_var != r_port and not reg.is_subtype(r_var, r_port) and r_var not in strict_roles:
            return True
    if r_var in strict_roles:
        if r_port != r_var and not reg.is_subtype(r_port, r_var) and r_port not in strict_roles:
            return True

    if r_port == "standard" or r_var == "standard":
        return False

    exclusive_roles = reg.get_verification_semantics("mutually_exclusive_roles")
    if r_port in exclusive_roles and r_var in exclusive_roles and r_port != r_var:
        return True

    exclusive_role_pairs = reg.get_verification_semantics("mutually_exclusive_role_pairs")
    for pair in exclusive_role_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            p_a, p_b = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
            if (r_port == p_a and r_var == p_b) or (r_port == p_b and r_var == p_a):
                return True
            if (reg.is_subtype(r_port, p_a) and reg.is_subtype(r_var, p_b)) or (reg.is_subtype(r_port, p_b) and reg.is_subtype(r_var, p_a)):
                return True
    return False


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

    out_sig_o = getattr(p_out, "signature", p_out)
    in_sig_o = getattr(p_in, "signature", p_in)

    if isinstance(out_sig_o, dict):
        out_tn = str(out_sig_o.get("type_name") or "any")
        out_st = str(out_sig_o.get("state") or "any")
        out_abs_t = str(out_sig_o.get("abstract_type") or "")
        out_acc = out_sig_o.get("accepted_states") or []
        out_parent_st = out_sig_o.get("parent_state")
        out_quals = out_sig_o.get("qualifiers") or []
        from lattice import AlgebraicSignature
        out_sig_o = AlgebraicSignature(
            type_name=out_tn,
            state=out_st,
            qualifiers=_normalize_quals(out_quals),
            abstract_type=out_abs_t,
            accepted_states=frozenset(str(x).strip().lower() for x in out_acc if str(x).strip()),
            parent_state=out_parent_st,
        )

    if isinstance(in_sig_o, dict):
        in_tn = str(in_sig_o.get("type_name") or "any")
        in_st = str(in_sig_o.get("state") or "any")
        in_abs_t = str(in_sig_o.get("abstract_type") or "")
        in_acc = in_sig_o.get("accepted_states") or []
        in_parent_st = in_sig_o.get("parent_state")
        in_quals = in_sig_o.get("qualifiers") or []
        from lattice import AlgebraicSignature
        in_sig_o = AlgebraicSignature(
            type_name=in_tn,
            state=in_st,
            qualifiers=_normalize_quals(in_quals),
            abstract_type=in_abs_t,
            accepted_states=frozenset(str(x).strip().lower() for x in in_acc if str(x).strip()),
            parent_state=in_parent_st,
        )

    if producer_cell is None:
        producer_cell = _get_p_field(p_out, "owner_cell") or getattr(out_sig_o, "owner_cell", None)
    if consumer_cell is None:
        consumer_cell = _get_p_field(p_in, "owner_cell") or getattr(in_sig_o, "owner_cell", None)

    reg = TypeRegistry.get_instance()

    # 1. Semantic Port Roles & Directional Gating
    out_role = str(_get_p_field(p_out, "port_role") or getattr(out_sig_o, "port_role", None) or getattr(out_sig_o, "derived_role", "") or "").lower()
    in_role = str(_get_p_field(p_in, "port_role") or getattr(in_sig_o, "port_role", None) or getattr(in_sig_o, "derived_role", "") or "").lower()

    pred_in = _declared_role_semantics("prediction_input_roles")
    pred_out = _declared_role_semantics("prediction_output_roles")

    out_state = str(getattr(out_sig_o, "state", "") or _get_p_field(p_out, "state", "") or "").lower()
    in_state = str(getattr(in_sig_o, "state", "") or _get_p_field(p_in, "state", "") or "").lower()

    props = reg.get_state_properties(out_state)
    is_pred_state = bool(props.get("is_prediction")) or out_state in set(reg.get_verification_semantics("prediction_states") or ())

    if in_role in pred_in and not (is_pred_state or out_role in pred_out):
        return False

    if is_pred_state and _roles_conflict(in_role, "prediction_output", reg=reg):
        return False

    if _roles_conflict(in_role, out_role, reg=reg):
        return False

    # 2. Structural Kind and Abstract Carrier Gating
    out_sk = str(_get_p_field(p_out, "structure_kind", "") or "").lower()
    in_sk = str(_get_p_field(p_in, "structure_kind", "") or "").lower()
    out_abs = str(_get_p_field(p_out, "abstract_type", "") or getattr(out_sig_o, "abstract_type", "") or "").lower()
    in_abs = str(_get_p_field(p_in, "abstract_type", "") or getattr(in_sig_o, "abstract_type", "") or "").lower()

    # Handles can only connect to handles
    if out_sk == "handle" and in_sk and in_sk != "handle":
        return False
    if in_sk == "handle" and out_sk and out_sk != "handle":
        return False

    # Path specs can only connect to path specs or path types
    if out_sk == "path_spec" and in_sk and in_sk not in ("path_spec", "scalar"):
        return False
    if in_sk == "path_spec" and out_sk and out_sk not in ("path_spec", "scalar"):
        return False

    # Scalars (rank 0 primitives) cannot connect to tensors or collections
    is_out_scalar = (out_sk == "scalar") or (out_abs in ("scalar", "primitive"))
    is_in_scalar = (in_sk == "scalar") or (in_abs in ("scalar", "primitive"))
    is_out_compound = (out_sk in ("tensor", "collection", "composite")) or (reg.is_composite_carrier(out_abs) or reg.is_abstract_carrier(out_abs))
    is_in_compound = (in_sk in ("tensor", "collection", "composite")) or (reg.is_composite_carrier(in_abs) or reg.is_abstract_carrier(in_abs))

    if is_out_scalar and is_in_compound:
        return False
    if is_in_scalar and is_out_compound:
        return False

    # 3. Rejected Qualifiers
    rej_quals = (
        _get_p_field(p_in, "rejected_qualifiers", None)
        or getattr(in_sig_o, "rejected_qualifiers", None)
        or frozenset()
    )
    if rej_quals:
        out_quals = (
            _get_p_field(p_out, "qualifiers", None)
            or getattr(out_sig_o, "qualifiers", None)
            or frozenset()
        )
        flat_out_quals = set()
        for q in out_quals:
            if isinstance(q, (tuple, list, set, frozenset)):
                for q_el in q:
                    flat_out_quals.add(str(q_el).lower())
            else:
                flat_out_quals.add(str(q).lower())
        if any(str(rq).lower() in flat_out_quals for rq in rej_quals):
            return False

    # 4. Shape Contract & Rank Compatibility
    out_sc = _get_p_field(p_out, "shape_contract", None) or getattr(out_sig_o, "shape_contract", None)
    in_sc = _get_p_field(p_in, "shape_contract", None) or getattr(in_sig_o, "shape_contract", None)

    out_poly = False
    in_poly = False
    if hasattr(out_sc, "rank_polymorphic") and getattr(out_sc, "rank_polymorphic"):
        out_poly = True
    elif isinstance(out_sc, dict) and out_sc.get("rank_polymorphic"):
        out_poly = True

    if hasattr(in_sc, "rank_polymorphic") and getattr(in_sc, "rank_polymorphic"):
        in_poly = True
    elif isinstance(in_sc, dict) and in_sc.get("rank_polymorphic"):
        in_poly = True

    if out_sc and in_sc and not (out_poly or in_poly):
        o_ndim = _extract_ndim(out_sc)
        i_ndim = _extract_ndim(in_sc)
        if o_ndim is not None and i_ndim is not None and o_ndim != i_ndim:
            return False
        o_shp = getattr(out_sc, "shape", None) if not isinstance(out_sc, dict) else out_sc.get("shape")
        i_shp = getattr(in_sc, "shape", None) if not isinstance(in_sc, dict) else in_sc.get("shape")
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

    acc_in = _get_p_field(p_in, "accepted_states", None) or getattr(in_sig_o, "accepted_states", None) or ()
    if out_state and acc_in and (out_state in acc_in or any(reg.state_ancestry_reaches(out_state, {acc}) for acc in acc_in)):
        is_poly_in = True

    r_out = _get_p_field(p_out, "canonical_rank", None) or getattr(out_sig_o, "canonical_rank", None) or _get_p_field(p_out, "rank", None)
    r_in = _get_p_field(p_in, "canonical_rank", None) or getattr(in_sig_o, "canonical_rank", None) or _get_p_field(p_in, "rank", None)
    if not (is_poly_out or is_poly_in or out_poly or in_poly):
        if r_out is not None and r_in is not None and r_out != r_in:
            return False

    # 5. State Poset Transition Compatibility
    producer_accepted = _get_p_field(p_out, "accepted_states", None) or getattr(out_sig_o, "accepted_states", None) or ()
    consumer_accepted = _get_p_field(p_in, "accepted_states", None) or getattr(in_sig_o, "accepted_states", None) or ()
    consumer_parent = getattr(in_sig_o, "parent_state", None) or _get_p_field(p_in, "parent_state", None)
    producer_parent = getattr(out_sig_o, "parent_state", None) or _get_p_field(p_out, "parent_state", None)

    from lattice import is_path_port
    is_out_path = is_path_port(p_out)
    is_in_path = is_path_port(p_in)
    is_path_compatible = False
    if is_out_path and is_in_path:
        terminal_invalid_states = _declared_role_semantics("terminal_invalid_states")
        if out_state not in terminal_invalid_states:
            is_path_compatible = True

    if not is_path_compatible and not reg.is_state_compatible(
        producer_state=out_state,
        consumer_state=in_state,
        producer_accepted=producer_accepted,
        consumer_accepted=consumer_accepted,
        consumer_parent=consumer_parent,
        producer_parent=producer_parent,
    ):
        return False

    # 6. Precondition Verification Against Producer Properties and Postconditions
    preconds = []
    if consumer_cell is not None:
        preconds.extend(_get_p_field(consumer_cell, "preconditions", []) or [])
    preconds.extend(_get_p_field(p_in, "preconditions", []) or [])

    p_name = _get_p_field(p_in, "name", None)

    producer_props: Dict[str, Any] = {}
    if out_sc:
        o_dict = out_sc if isinstance(out_sc, dict) else getattr(out_sc, "__dict__", {})
        for pk, pv in o_dict.items():
            if pk != "rank_polymorphic":
                producer_props[pk] = pv
        o_ndim = _extract_ndim(out_sc)
        if o_ndim is not None:
            producer_props["ndim"] = o_ndim
        o_shp = getattr(out_sc, "shape", None) if not isinstance(out_sc, dict) else out_sc.get("shape")
        if o_shp is not None:
            producer_props["shape"] = o_shp
    if r_out is not None and "ndim" not in producer_props:
        producer_props["ndim"] = r_out
    if out_state and out_state not in ("any", "*"):
        producer_props["state"] = out_state

    out_p_name = _get_p_field(p_out, "name", None)
    if producer_cell is not None:
        for post in _get_p_field(producer_cell, "postconditions", []) or []:
            if not isinstance(post, dict):
                continue
            post_target = post.get("target")
            if post_target is None or post_target == out_p_name or post_target in _declared_role_semantics("output_targets"):
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
    if is_out_path and is_in_path:
        # Both ports carry filesystem resource locators (path_spec).
        # Carrier equivalence satisfies Robinson unification across independently authored trees.
        protocols = _declared_role_semantics("protocols")
        in_fmts = {str(f).lower().lstrip(".") for f in (_get_p_field(p_in, "supported_formats", []) or []) if f != "*"} - protocols
        out_fmts = {str(f).lower().lstrip(".") for f in (_get_p_field(p_out, "supported_formats", []) or []) if f != "*"} - protocols
        if in_fmts and out_fmts and not (in_fmts & out_fmts):
            return False
        if hasattr(out_sig_o, "unifies_with") and not out_sig_o.unifies_with(in_sig_o):
            return False
    elif hasattr(out_sig_o, "unifies_with"):
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

    return True


def _shape_compatible(p_out: Any, p_in: Any) -> bool:
    return is_edge_valid(p_out, p_in)


@functools.lru_cache(maxsize=4096)
def _extract_lhs_template_placeholders(tpl: str) -> FrozenSet[str]:
    """Extracts placeholder names bound on the LHS of assignments in code_template."""
    if not tpl or "=" not in tpl:
        return frozenset()
    phs = extract_template_placeholders(tpl)
    if not phs:
        return frozenset()
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
            return frozenset(lhs_phs)
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
        return frozenset(extract_template_placeholders(lhs))
    return frozenset()


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

class _DynamicTopTypeSet(frozenset):
    """Dynamic top type set backed by TypeRegistry declarations."""
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
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        pass

    @classmethod
    def from_string(cls, s: str) -> 'TypeTerm':
        return parse_type_term(s)


@dataclass(frozen=True, slots=True)
class TopType(TypeTerm):
    """Universal Top type (wildcard) unifying with any type term."""
    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        return self

    def __repr__(self) -> str:
        return "Top"


TOP = TopType()


@dataclass(frozen=True, slots=True)
class AtomicType(TypeTerm):
    """Ground type constant."""
    name: str

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
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

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
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

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        resolved_terms = tuple(t.apply_substitution(sigma, visited) for t in self.terms)
        return UnionTypeTerm(resolved_terms)

    def __repr__(self) -> str:
        return " | ".join(str(t) for t in self.terms)


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    """Type variable alpha, beta... subject to substitution."""
    var_name: str

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        if visited and self.var_name in visited:
            return self
        if self.var_name in sigma.mappings:
            target = sigma.mappings[self.var_name]
            new_visited = (visited or frozenset()) | {self.var_name}
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
    qualifiers: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)
    abstract_type: Optional[str] = None
    accepted_states: FrozenSet[str] = field(default_factory=frozenset)
    parent_state: Optional[str] = None

    def __post_init__(self):
        raw = self.abstract_type
        if raw and str(raw).strip().lower() not in ("none", "null", ""):
            object.__setattr__(self, "abstract_type", str(raw).strip())
        else:
            object.__setattr__(self, "abstract_type", None)
        if self.accepted_states and not isinstance(self.accepted_states, frozenset):
            object.__setattr__(self, "accepted_states", frozenset(str(s).strip().lower() for s in self.accepted_states if str(s).strip()))
        if self.parent_state:
            object.__setattr__(self, "parent_state", str(self.parent_state).strip().lower())

    def apply_substitution(self, sigma: 'Substitution', visited: Optional[FrozenSet[str]] = None) -> 'TypeTerm':
        t_resolved = self.type_name
        if self.type_name in sigma.mappings:
            if visited and self.type_name in visited:
                return self
            val = sigma.mappings[self.type_name]
            if isinstance(val, AtomicType):
                t_resolved = val.name
            elif isinstance(val, TypeTerm):
                new_visited = (visited or frozenset()) | {self.type_name}
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
    visited: Optional[FrozenSet[str]] = None
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
            new_visited = (visited or frozenset()) | {term.var_name}
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
            new_visited = (visited or frozenset()) | {term.type_name}
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
                    new_visited = (visited or frozenset()) | {tok}
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

    # 1. Identity or Universal Top (Top unifies with any type term)
    if t1 == t2 or isinstance(t2, TopType) or isinstance(t1, TopType):
        if is_ground_query:
            _unify_cache_put(cache_key, sub)
        return sub

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

    # 2.3. Coproduct / Union unification (canonical injection)
    if isinstance(t1, UnionTypeTerm):
        for alt in t1.terms:
            sub_alt = unify(alt, t2, sub)
            if sub_alt is not None:
                if is_ground_query:
                    _unify_cache_put(cache_key, sub_alt)
                return sub_alt
        if is_ground_query:
            _unify_cache_put(cache_key, None)
        return None

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

    # 2.5. Generic container unification (covariant structural unification)
    if isinstance(t1, GenericTypeTerm) and isinstance(t2, GenericTypeTerm):
        registry = TypeRegistry.get_instance()
        c1 = t1.constructor.lower()
        c2 = t2.constructor.lower()
        compat = (c1 == c2) or registry.is_subtype(c1, c2)
        if compat and len(t1.args) == len(t2.args):
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
        if t2.qualifiers and not t2.qualifiers.issubset(t1.qualifiers):
            ignorable = TypeRegistry.get_instance().get_advisory_qualifiers()
            req = {q for q in t2.qualifiers if q not in ignorable}
            if req and not req.issubset(t1.qualifiers):
                if is_ground_query:
                    _unify_cache_put(cache_key, None)
                return None

        # Check if type_names contain generic definitions
        if ("[" in t1.type_name) or ("[" in t2.type_name) or (len(t1.type_name) == 1 and t1.type_name.isupper()) or (len(t2.type_name) == 1 and t2.type_name.isupper()):
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
        if item.is_top() or is_top_symbol(t_clean, registry):
            res = TOP
        elif t_clean.startswith("?") or "|" in t_clean or ("[" in t_clean and t_clean.endswith("]")) or registry.is_type_variable(t_clean):
            res = TypeTerm.from_string(t_clean)
        else:
            canonical = registry.canonical_name(t_clean)
            if is_top_symbol(canonical, registry):
                res = TOP
            else:
                res = TypestateTerm(
                    type_name=canonical,
                    state=item.state,
                    qualifiers=item.qualifiers,
                    abstract_type=getattr(item, "abstract_type", None),
                    accepted_states=getattr(item, "accepted_states", frozenset()),
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
            accepted_states=getattr(target, "accepted_states", frozenset()),
            parent_state=getattr(target, "parent_state", None),
            port_role=getattr(target, "port_role", None)
        )

    if isinstance(target, AlgebraicSignature):
        new_type = substitute_generics(target.type_name, sub)
        return AlgebraicSignature(
            type_name=str(new_type),
            state=target.state,
            qualifiers=target.qualifiers,
            abstract_type=target.abstract_type,
            accepted_states=getattr(target, "accepted_states", frozenset()),
            parent_state=getattr(target, "parent_state", None)
        )

    if hasattr(target, "type_name") and hasattr(target, "model_copy"):
        new_type = substitute_generics(target.type_name, sub)
        return target.model_copy(update={"type_name": str(new_type)})

    if isinstance(target, TypeTerm):
        return target.apply_substitution(sub)

    return target


def verify_coproduct_branch(
    then_term: Union[TypeTerm, AlgebraicSignature, str],
    else_term: Union[TypeTerm, AlgebraicSignature, str],
    join_type: Optional[Union[TypeTerm, AlgebraicSignature, str]] = None,
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[TypeTerm], Optional[Substitution]]:
    """
    Verifies that coproduct True-path and False-path can unify to a common join type D:
      unify(tau_then, D) != bottom and unify(tau_else, D) != bottom
    Returns (is_valid, resolved_join_type, updated_sigma).
    """
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

    return False, None, None


def verify_traced_loop_invariant(
    feedback_in: Union[TypeTerm, AlgebraicSignature, str],
    feedback_out: Union[TypeTerm, AlgebraicSignature, str],
    sigma: Optional[Substitution] = None
) -> Tuple[bool, Optional[Substitution]]:
    """
    Verifies the categorical Traced Feedback loop invariant:
      unify(tau_feedback_out, tau_feedback_in) != bottom
    Ensures that loop body updates preserve or are compatible with the accumulator state U.
    """
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

    p_role = str(getattr(sig, "port_role", "") or getattr(sig, "role", "") or "").strip().lower()
    if p_role == "data_input":
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

    # If the value represents an identifier or pipeline variable expression and port is not string-like
    if not is_str_like and p_role not in ("parameter", "literal_parameter", "literal", "config"):
        if s.isidentifier() or s.startswith("var_"):
            return s

    if is_str_like or not s.isidentifier() or p_role in ("parameter", "literal_parameter", "literal", "config"):
        raw_s = s.strip("'\"").strip()
        return repr(raw_s)
    return s



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

    def __init__(self, prompt: str = "", scope: Optional[Dict[str, Any]] = None, initial_scope: Optional[Set[str]] = None, parameters: Optional[Dict[str, Any]] = None):
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
        self.identifier_roles: Dict[int, FrozenSet[str]] = self._build_identifier_role_map(self._prompt)
        # Provenance and monoidal DAG branch tracking
        self.var_origins: Dict[str, Set[str]] = {}
        self.multi_fiber_vars: Set[str] = set()
        self.current_pipeline: Optional[List[Any]] = None
        self.var_epochs: Dict[str, int] = {}   # row-set epoch per variable (tree-declared row_set_effects)
        self.binding_trace: List[Dict[str, Any]] = []
        self.superseded_vars: Set[str] = set()
        self.var_parents: Dict[str, Set[str]] = {}
        self.scope_variables: Dict[str, Any] = {}
        self.llm_slots: Dict[str, Dict[str, Any]] = {}
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
        """Target sink extraction: in symbolic profiles, returns None; resolved by LLM in neural profiles."""
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
        self.current_cell_clause_idx = None
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
        new_ctx.llm_slots = {k: dict(v) if isinstance(v, dict) else v for k, v in self.llm_slots.items()}
        new_ctx.hyperparameters = dict(self.hyperparameters)
        new_ctx.runtime_aliases = dict(getattr(self, "runtime_aliases", {}) or {})
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
        words = prev_chunk.split()
        if not words:
            return ""

        reg = TypeRegistry.get_instance()
        naming_connectives: Set[str] = set()
        dest_triggers = (reg.get_dest_port_tokens() | reg.get_egress_tokens()) - naming_connectives
        src_triggers = reg.get_source_port_tokens()

        for word in reversed(words):
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
            # Naming connectives ("named", "called", etc.) introduce identifiers for
            # the preceding noun/verb; continue scanning backward to identify direction.
            if w in naming_connectives:
                continue
            if w in dest_triggers:
                return "dest"
            if w in src_triggers:
                return "src"
            # Conjunctions / coordinating words are DECLARED vocabulary; they
            # neither mark a direction nor end the backward scan.
            if w in reg.get_sentence_connectives():
                continue
            break
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
    def _build_identifier_role_map(prompt: str) -> Dict[int, FrozenSet[str]]:
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
        col_tokens = reg.get_column_projection_tokens()
        reg_types = set(reg.get_registered_types())
        reg_stopwords = set(reg.get_stopwords())

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
            if keyword.iskeyword(s_low) or s_low in reg_types or s_low in reg_stopwords:
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
                    self.identifier_roles[w_pos] = frozenset(col_tokens)
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
        is_str = registry.is_subtype(t_name, "str") or t_name in ("str", "any")

        # 1. Parameter explicitly supplied in context or LLM slots
        c_rank = getattr(port_sig, "canonical_rank", None) if getattr(port_sig, "canonical_rank", None) is not None else getattr(port_sig, "rank", None)
        str_kind = str(getattr(port_sig, "structure_kind", "") or "").lower()
        is_scalar = (c_rank == 0 or str_kind == "scalar" or t_name in ("str", "int", "float", "bool", "scalar") or registry.is_subtype(t_name, "scalar"))

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
                valid_nums = {v for _, k_lit, v in getattr(self, "ordered_literals", []) if k_lit in ("int_literal", "float_literal")}
                has_name = p_n_lower in c_toks or (len(p_n_lower) > 2 and p_n_lower in prompt_str.lower())
                if isinstance(cand_val, (int, float)) and not isinstance(cand_val, bool):
                    has_val = (val_s in valid_nums) or (val_s in prompt_str)
                else:
                    has_val = (val_s.lower() in c_toks) or (len(val_s) > 2 and val_s.lower() in prompt_str.lower())
                if has_name and has_val:
                    return _format_literal_value(cand_val, port_sig)
        if hasattr(self, "llm_slots") and isinstance(self.llm_slots, dict):
            for cell_s in self.llm_slots.values():
                if isinstance(cell_s, dict) and port_sig.name in cell_s:
                    slot_val = cell_s[port_sig.name]
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
            for idx, (lit_pos, kind, val) in enumerate(self.ordered_literals):
                if idx in self.used_indices or kind != "numeric":
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
                                    if p0 in ("any", "generic", "*", ""):
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
                        elif declared_fmt and declared_fmt not in ("any", "generic", "*", ""):
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
                            if target_carrier and target_carrier not in ("path", "object", "unknown", "top", "any", "*"):
                                ext_carrier = None
                                for abs_c in sorted(reg.get_abstract_carriers()):
                                    if abs_c in ("path", "object", "unknown", "top", "any", "*"):
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
                    return json.dumps(val) if not (val.startswith(("'", '"')) or val in ("None", "True", "False")) else val
                supp_fmts = getattr(port_sig, "supported_formats", None) or getattr(cell, "supported_formats", None) or ()
                if supp_fmts:
                    fmt = str(supp_fmts[0]).lstrip(".")
                    return json.dumps(f"input.{fmt}")
                t_str = str(getattr(port_sig, "type_name", "")).lower()
                dom = str(getattr(port_sig, "domain", "")).lower()
                registry = TypeRegistry.get_instance()
                modality = registry.get_abstract_carrier(t_str) or registry.get_abstract_carrier(dom)
                if not modality:
                    for cand_mod in sorted(registry.get_abstract_carriers()):
                        if registry.is_subtype(t_str, cand_mod) or registry.is_subtype(dom, cand_mod):
                            modality = cand_mod
                            break
                    if not modality:
                        modality = "default"
                placeholder = registry.get_asset_placeholder(modality)
                return json.dumps(placeholder)

            # If sink cell has a path port and prompt omitted explicit destination literal:
            if is_sink_ctx:
                supp_fmts = getattr(port_sig, "supported_formats", None) or getattr(cell, "supported_formats", None) or ()
                if supp_fmts:
                    fmt = str(supp_fmts[0]).lstrip(".")
                    p_name = getattr(port_sig, "name", "output")
                    return json.dumps(f"{p_name}.{fmt}")
                t_str = str(getattr(port_sig, "type_name", "")).lower()
                dom = str(getattr(port_sig, "domain", "")).lower()
                registry = TypeRegistry.get_instance()
                modality = registry.get_abstract_carrier(t_str) or registry.get_abstract_carrier(dom)
                if not modality:
                    for cand_mod in sorted(registry.get_abstract_carriers()):
                        if registry.is_subtype(t_str, cand_mod) or registry.is_subtype(dom, cand_mod):
                            modality = cand_mod
                            break
                    if not modality:
                        modality = "default"
                placeholder = registry.get_output_asset_placeholder(modality)
                return json.dumps(placeholder)

        # 6. Operational Parameter Extraction
        if is_operational and is_str:
            enum_vals = {str(e).strip().lower() for e in port_sig.enum_values} if getattr(port_sig, "enum_values", None) else None
            p_state = str(getattr(port_sig, "state", "") or "").lower()
            p_role = str(getattr(port_sig, "port_role", "") or getattr(port_sig, "derived_role", "") or "").lower()
            reg = TypeRegistry.get_instance()
            col_tokens = reg.get_column_projection_tokens()
            reg_types = reg.get_registered_types()
            is_col_port = (
                p_role in _declared_role_semantics("projection_port_roles")
                or p_role in _declared_role_semantics("projective_roles")
                or p_state in _declared_role_semantics("projection_port_states")
            )

            # Check literals for operational parameter candidates (prioritizing current clause)
            cand_indices = []
            if self.current_cell_clause_idx is not None:
                # 1. Unconsumed literals in the current clause
                cand_indices.extend([i for i in range(len(self.ordered_literals))
                                     if i not in self.used_indices and self.literal_clause_map.get(i) == self.current_cell_clause_idx])
                # 2. Reusable identifiers from the same clause (e.g. select and set on the same column in a transform)
                if is_col_port:
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
                if idx in self.used_indices and not (is_col_port and self.current_cell_clause_idx is not None and self.literal_clause_map.get(idx) == self.current_cell_clause_idx and kind in ("identifier", "quoted_str")):
                    continue
                if kind not in ("quoted_str", "identifier"):
                    continue
                val_str = str(val).strip()
                val_low = val_str.lower()
                # Target protection: operational parameters must not steal the target sink variable
                # UNLESS this cell/port is a setter morphism that introduces/writes the target column or key
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
                # Exclude Python keywords and registered types from being bound as column names
                if is_col_port and (keyword.iskeyword(val_low) or val_low in reg_types):
                    continue
                # Column & target protection: non-column parameter ports must not steal data columns
                roles = self.identifier_roles.get(lit_pos, frozenset())
                is_data_col = (
                    bool(roles & col_tokens)
                    or (kind == "quoted_str" and not ExecutionContext._is_path_string(val_str))
                    or (is_setter_port and bool(roles & (col_tokens | reg.get_dest_port_tokens() | reg.get_egress_tokens())))
                )
                if is_data_col and not is_col_port:
                    continue
                if is_col_port and not is_data_col:
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
                    role_tokens = self.identifier_roles.get(pos, frozenset())
                    if not role_tokens or not (role_tokens & _identity_scope):
                        continue
                    self.used_indices.add(idx)
                    return json.dumps(val)

        # 6c. Multi-identifier collection / list projection (e.g. list for selection or projection)
        _t_name_concrete = bool(t_name) and not registry.is_type_variable(t_name) and t_name in registry.get_registered_types()
        is_collection = (
            getattr(port_sig, "structure_kind", None) == "collection"
            or (getattr(port_sig, "canonical_rank", None) is not None and getattr(port_sig, "canonical_rank", None) > 0)
            or (registry is not None and (registry.is_subtype(t_name, "collection") or registry.is_subtype(t_name, "sequence")))
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
            if port_sig.default_value is not None:
                pass  # Fall through to Step 7 (port default value)
            else:
                _state_tokens = CellTokenizer.tokenize_identifier(_raw_state) if _raw_state.lower() not in ("any", "default", "") else set()
                _proj_tokens = TypeRegistry.get_instance().get_column_projection_tokens()
                _identity_scope: Set[str] = set(cell_tokens or set()) | _state_tokens | _proj_tokens
                egress_tokens = TypeRegistry.get_instance().get_egress_tokens() | TypeRegistry.get_instance().get_dest_port_tokens()
                registered_types = TypeRegistry.get_instance().get_registered_types()
                stopwords = TypeRegistry.get_instance().get_stopwords()

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
                    cands_to_check = clause_cands if clause_cands else all_available
                else:
                    cands_to_check = all_available

                for idx in cands_to_check:
                    pos, kind, val = self.ordered_literals[idx]
                    val_str = str(val).strip()
                    val_low = val_str.lower()
                    if val_low in target_lits or keyword.iskeyword(val_low) or val_low in registered_types or val_low in stopwords:
                        continue
                    role_tokens = self.identifier_roles.get(pos, frozenset())
                    if role_tokens and (role_tokens & egress_tokens):
                        continue

                    is_candidate = False
                    if role_tokens and (role_tokens & _identity_scope):
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
            proj_roles = set(registry.get_projection_tokens()) | set(registry.get_verification_semantics("projection_port_roles") or ())
            is_key_req = getattr(port_sig, "required", False) and (
                getattr(port_sig, "binds", "") in key_binds
                or str(getattr(port_sig, "port_role", "") or "").lower() in proj_roles
            )
            if is_key_req and def_str.strip("'\"") == "":
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
def _get_module_symbols(mod_name: str) -> FrozenSet[str]:
    """Caches exported symbols of submodules for fast dependency resolution."""
    try:
        import sys, importlib
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = importlib.import_module(mod_name)
        return frozenset(w for w in dir(mod) if not w.startswith("_"))
    except Exception as e:
        return frozenset()


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
        self._generic_port_cache: Optional[Tuple[Tuple[int, int], FrozenSet[str]]] = None
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

    def _generic_port_tokens(self) -> FrozenSet[str]:
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
            out = frozenset()
            self._generic_port_cache = (stamp, out)
            if orch is not None:
                orch._generic_port_cache = out
            return out
        mean = sum(dfs) / len(dfs)
        var = sum((v - mean) ** 2 for v in dfs) / len(dfs)
        cut = mean + (var ** 0.5)
        out = frozenset(t for t, v in toks.items() if v > cut)
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

    def _find_fibered_carrier_bridge(
        self,
        ctx: ExecutionContext,
        registry: TypeRegistry,
        is_multi_key: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Cell]]:
        """
        Discovers a Fibered Carrier from in-scope variables: an in-scope variable
        v_name whose type v_tn is accepted as primary input of a loaded bridge cell
        that possesses a key port.
        Returns:
            (var_name, var_type, key_port_name, bridge_cell)
        """
        orchestrator = getattr(self, "orchestrator", None) or getattr(ctx, "orchestrator", None)
        if orchestrator is None:
            try:
                from lattice import LatticeOrchestrator
                orchestrator = LatticeOrchestrator.get_active_instance()
            except Exception:
                orchestrator = None
        loaded = list(getattr(orchestrator, "loaded_cells", {}).values()) if orchestrator and hasattr(orchestrator, "loaded_cells") else []
        key_roles = set(registry.get_verification_semantics("key_roles"))
        key_port_names = set(registry.get_verification_semantics("projection_port_names"))

        candidate_bridges = []
        for c in loaded:
            c_role = str(getattr(c, "node_role", "") or "").lower()
            c_type = str(getattr(c, "node_type", "") or "").lower()
            if c_role != "bridge" and c_type != "tunnel" and getattr(c, "role", "") != "bridge":
                continue
            k_port = None
            for p_name, p_sig in c.inputs.items():
                bind = str(getattr(p_sig, "binds", "") or "").lower()
                role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").lower()
                if bind in key_roles or role in key_roles or p_name in key_port_names:
                    k_port = p_sig
                    break
            if not k_port or not c.primary_input:
                continue

            k_tn = str(getattr(getattr(k_port, "signature", k_port), "type_name", "")).lower()
            k_is_collection = (
                getattr(k_port, "structure_kind", None) == "collection"
                or (getattr(k_port, "canonical_rank", None) is not None and getattr(k_port, "canonical_rank", None) > 0)
                or getattr(k_port, "abstract_type", None) == "collection"
                or (registry is not None and (registry.is_subtype(k_tn, "collection") or registry.is_subtype(k_tn, "sequence")))
            )
            if is_multi_key and not k_is_collection:
                continue
            if not is_multi_key and k_is_collection:
                continue

            candidate_bridges.append((c, k_port.name))

        for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
            sig_obj = getattr(v_sig, "signature", v_sig)
            v_tn = str(getattr(sig_obj, "type_name", "") or "").lower()
            v_abs = str(getattr(sig_obj, "abstract_type", "") or "").lower()
            for b_cell, k_name in candidate_bridges:
                in_p = b_cell.primary_input
                in_tn = str(getattr(getattr(in_p, "signature", in_p), "type_name", "")).lower()
                in_abs = str(getattr(getattr(in_p, "signature", in_p), "abstract_type", "")).lower()
                accepts = (
                    in_tn in ("any", "*", "top", "")
                    or v_tn == in_tn
                    or registry.is_subtype(v_tn, in_tn)
                    or (v_abs and v_abs == in_abs)
                    or (v_abs and registry.is_subtype(v_abs, in_tn))
                    or (v_tn in registry.get_abstract_carriers() and registry.is_subtype(in_tn, v_tn))
                )
                if accepts:
                    return v_name, v_tn, k_name, b_cell

        return None, None, None, None

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
            col_proj_tokens = (
                set(registry.get_column_projection_tokens())
                | set(registry.get_verification_semantics("projection_port_names") or ())
            )
            for key, rendered in literal_bindings.items():
                if (CellTokenizer.tokenize_identifier(key) & toks) or (key in col_proj_tokens and (toks & col_proj_tokens)):
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
        seen_states: Set[Tuple[str, FrozenSet[str]]] = {(str(src_type or "").lower(), frozenset())}

        # 0-hop check: only valid if literal_bindings is empty.
        # If literal_bindings is specified, a bridge MUST consume the literals.
        if not literal_bindings and _dst_accepts(src_type):
            return str(src_var)

        col_proj_tokens = (
            set(registry.get_column_projection_tokens())
            | set(registry.get_verification_semantics("projection_port_names"))
        )
        has_proj_lit = bool(
            literal_bindings
            and any(
                CellTokenizer.tokenize_identifier(k) & col_proj_tokens
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
                        state_key = (out_type, frozenset(new_consumed))
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

    def _resolve_role_partitioned_projection(
        self,
        cell: Cell,
        p_name: str,
        p_sig: Any,
        concrete_sig: Any,
        cell_bindings: Dict[str, Any],
        ctx: ExecutionContext,
        accumulated_sigma: Substitution,
    ) -> Optional[Tuple[str, Substitution]]:
        return None

    def _project_complement_roles_from_carrier(
        self,
        cell: Cell,
        p_name: str,
        p_role: str,
        concrete_sig: Any,
        cell_bindings: Dict[str, Any],
        ctx: ExecutionContext,
        registry: TypeRegistry,
    ) -> Optional[str]:
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
                is_path_port = (
                    registry.is_subtype(t_name, "filepath")
                    or registry.is_subtype(t_name, "path")
                    or registry.is_subtype(t_name, "uri")
                )
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

        prefix = ""
        if sig is not None:
            raw_sig = getattr(sig, "signature", sig)
            t_name = getattr(raw_sig, "type_name", None) or getattr(sig, "type_name", None)
            state = getattr(raw_sig, "state", None) or getattr(sig, "state", None)

            if t_name:
                t_clean = _clean_token(t_name)
                if t_clean and t_clean not in ("any", "unknown", "none", "t", "v", "object"):
                    prefix = t_clean

            if not prefix and state:
                s_clean = _clean_token(state)
                if s_clean and s_clean not in ("none", "any", "unknown"):
                    prefix = s_clean

        if not prefix and port_name:
            p_clean = _clean_token(port_name)
            if p_clean and p_clean not in ("out", "output", "result", "res"):
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

        # Ensure distinct morphism instances: each step in the pipeline must have independent mutable state
        seen_cell_ids = set()
        cloned_cells = []
        for c in cells:
            if id(c) in seen_cell_ids:
                c = c.clone() if hasattr(c, "clone") else copy.copy(c)
            seen_cell_ids.add(id(c))
            cloned_cells.append(c)
        cells = cloned_cells

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
        if hasattr(ctx, "var_parents") and isinstance(ctx.var_parents, dict):
            ctx.var_parents.clear()
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
        llm_pipeline_bindings: Dict[str, Dict[str, Any]] = {}
        if prompt_text and cells and allow_llm:
            try:
                try:
                    from .inference import ModelManager
                except (ImportError, ValueError):
                    from inference import ModelManager
                mm = ModelManager.get_instance()
                if mm.can_synthesize():
                    if not getattr(ctx, "llm_slots", None):
                        self._extract_pipeline_slots_with_llm(cells, prompt_text, ctx)
                    llm_pipeline_bindings = self._synthesize_pipeline_bindings_with_llm(cells, prompt_text, ctx)
            except Exception as e:
                logger.debug(f"[UNIFICATION] LLM pipeline synthesis setup failed: {e}")
                llm_pipeline_bindings = {}

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
                        cell.primary_input.canonical_rank = req_rank
                        cell.primary_output.canonical_rank = req_rank

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
            if len(cell.outputs) == 1:
                var_counter += 1
                ctx.var_counter = var_counter
                current_out_var = _semantic_var_name(cell, var_counter)
                cell_bindings["output_var"] = current_out_var
            elif len(cell.outputs) > 1:
                # Multi-output variables will be allocated per-port
                pass
            # For len(cell.outputs) == 0 (void output like save/show): no output_var allocated
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

            bound_producer = False

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
                                    valid_nums = {v for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("int_literal", "float_literal")}
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
                    for p_name, wire_target in planned_wires.items():
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
                            _rej = _wire_rejection(cell, p_name, cell.inputs.get(p_name), v_name, ctx, registry)
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
                    for p_name, (prod_cell, out_port_name) in dag_scope_bindings.items():
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
                            # DAG-resolved wires used to bypass clause/provenance checks entirely
                            # (type-only unification). Apply the same gate every other binding
                            # site uses so a wire derived from a different column than the one this
                            # clause names is left unbound for the projection branches to resolve.
                            _rej = _wire_rejection(cell, p_name, cell.inputs.get(p_name), v_name, ctx, registry)
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
                                or (is_model_cell and getattr(p, "derived_role", "") in projective_roles)
                            )
                            for p in req_ports
                        )
                        all_req_bound = all(p_name in cell_bindings for p_name, p in cell.inputs.items() if p.required)
                        if not (has_proj and has_carrier) and not (planned_wires and all_req_bound):
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

                        recency_idx = avail_vars.index(v_name)
                        rank = (not_superseded, role_tier, is_prod, recency_idx, state_match, overlap)
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
                    v_origins = getattr(ctx, "var_origins", {}).get(v_name, set())
                    # Clause-scoped admissibility (column provenance + feature columns), shared with the DAG site.
                    _rej = _wire_rejection(cell, p_name, p, v_name, ctx, registry)
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
                _rej = _wire_rejection(cell, target_port.name if target_port else "port_0", target_port, producer_var, ctx, registry)
                col_mismatch = (_rej is not None)

                prim_in = cell.primary_input
                if prim_in is not None and prim_in.name in cell.inputs:
                    prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                    if prod_sig is not None and not _is_product(prod_sig):
                        if not col_mismatch and _shape_compatible(prod_sig, prim_in):
                            u_sub = unify(prod_sig.signature, prim_in.signature, accumulated_sigma)
                            if u_sub is not None:
                                cell_bindings[prim_in.name] = producer_var
                                if hasattr(ctx, "consumed_vars"):
                                    ctx.consumed_vars.add(producer_var)
                                accumulated_sigma = u_sub
                                bound_producer = True

            # If producer_var did not bind to primary input, check other compatible input ports (required ports take precedence)
            if producer_var is not None and not bound_producer and not is_zero_ary and not is_replica and not col_mismatch:
                prod_sig, _ = ctx.variables.get(producer_var, (None, None))
                if prod_sig is not None and not _is_product(prod_sig):
                    req_unbound = [(k, v) for k, v in cell.inputs.items() if v.required and k not in cell_bindings]
                    cand_ports = req_unbound if req_unbound else [(k, v) for k, v in cell.inputs.items() if k not in cell_bindings]
                    for p_name, p_sig in cand_ports:
                        if not _shape_compatible(prod_sig, p_sig):
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
                    if prim_in is not None and prim_in.name in cell.inputs and _shape_compatible(c_sig, prim_in):
                        u_sub = unify(c_sig.signature, prim_in.signature, accumulated_sigma)
                        if u_sub is not None:
                            cell_bindings[prim_in.name] = carrier_candidate
                            accumulated_sigma = u_sub
                            bound_producer = True
                    if not bound_producer:
                        req_unbound = [(k, v) for k, v in cell.inputs.items() if v.required and k not in cell_bindings]
                        cand_ports = req_unbound if req_unbound else [(k, v) for k, v in cell.inputs.items() if k not in cell_bindings]
                        for p_name, p_sig in cand_ports:
                            if not _shape_compatible(c_sig, p_sig):
                                continue
                            u_sub = unify(c_sig.signature, p_sig.signature, accumulated_sigma)
                            if u_sub is not None:
                                cell_bindings[p_name] = carrier_candidate
                                accumulated_sigma = u_sub
                                bound_producer = True
                                break

            # 3. Resolve auxiliary input ports (variable reuse / port sharing / parameters / literals).
            # REQUIRED ports are processed before optional ones so data ports claim
            # the prompt's shared literal channels (expressions, paths) first;
            # optional configuration knobs never steal them.
            cell_toks_for_projection = _cell_clause_tokens(cell)
            for p_name, p_sig in sorted(cell.inputs.items(), key=lambda kv: not kv[1].required):
                if p_name in cell_bindings:
                    continue  # Already bound

                # Substitute generics if type variable in p_sig
                concrete_sig = substitute_generics(p_sig, accumulated_sigma)

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
                        if not _shape_compatible(v_sig, concrete_sig):
                            continue
                        u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                        if u_v is not None:
                            prod_cell = ctx.var_sources.get(v_name)
                            is_planned = bool(prod_cell and prod_cell.cell_id in planned_parents)
                            candidate_vars.append((is_planned, v_name, u_v))
                    if candidate_vars:
                        candidate_vars.sort(key=lambda x: x[0], reverse=True)
                        _, scoped_var, accumulated_sigma = candidate_vars[0]
                    if scoped_var is not None:
                        cell_bindings[p_name] = scoped_var
                        continue


                # Check LLM slot filling / hyperparameters before heuristic fallback
                llm_slot_val = None
                if hasattr(ctx, "llm_slots") and isinstance(ctx.llm_slots, dict):
                    cell_slots = ctx.llm_slots.get(cell.cell_id, {})
                    if p_name in cell_slots:
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

                            valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("int_literal", "float_literal")}

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
                        valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("int_literal", "float_literal")}
                        val_str = str(int(llm_slot_val)) if int(llm_slot_val) == llm_slot_val else str(llm_slot_val)
                        if val_str not in valid_nums and val_str not in prompt_str:
                            llm_slot_val = None

                # Optional parameters with declared default: use prompt literal or omit/default
                if not p_sig.required and p_sig.default_value is not None:
                    if llm_slot_val is not None:
                        formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
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
                        c_is_str = c_tname in ("str", "enum", "column_identifier", "filepath", "path") or TypeRegistry.get_instance().is_subtype(c_tname, "str")
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

                for v_name, (v_sig, _) in reversed(list(ctx.variables.items())):
                    # Heterogeneous products project member-wise; the whole container
                    # is never wired into a single data port.
                    if _is_product(v_sig):
                        continue
                    if v_name in already_bound_vars:
                        continue
                    if not _shape_compatible(v_sig, concrete_sig):
                        continue

                    # Role-gating in scope resolution: conflict-based role rules
                    v_role = getattr(v_sig, "port_role", None) or getattr(v_sig, "derived_role", "")
                    if _roles_conflict(p_role_in, v_role, reg=registry):
                        continue

                    # Admissibility gating: provenance compatibility and single-key setter multi-fiber rejection
                    _rej = _wire_rejection(cell, p_name, p_sig, v_name, ctx, registry)
                    if _rej is not None:
                        ctx.trace_binding(f"in_scope_rejected_{_rej[0]}", cell=cell.cell_id, port=p_name, var=v_name, **{k: v for k, v in _rej[1].items() if k not in ("cell", "port", "var")})
                        continue

                    u_v = unify(v_sig.signature, concrete_sig.signature, accumulated_sigma)
                    if u_v is not None:
                        prod_cell = ctx.var_sources.get(v_name)
                        is_planned = bool(prod_cell and prod_cell.cell_id in planned_parents)
                        c_clause = getattr(cell, "matched_clause_idx", None)
                        prod_clause = getattr(prod_cell, "matched_clause_idx", None) if prod_cell else None

                        is_planned_score = 1 if is_planned else 0
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
                            target_affinity = -0.5

                        is_unconsumed = 1 if v_name not in getattr(ctx, "consumed_vars", set()) else 0

                        prio = (is_planned_score, target_affinity, is_unconsumed, clause_causality, recency_idx)
                        candidate_vars.append((prio, v_name, u_v))

                if candidate_vars:
                    # D6 Fix: Planned parent variables & clause adjacency take absolute priority over arbitrary recency
                    candidate_vars.sort(key=lambda x: x[0], reverse=True)
                    ctx.trace_binding(
                        "scoped_candidates", cell=cell.cell_id, port=p_name, clause=cell_clause,
                        candidates=[(v, str(pr), sorted(getattr(ctx, "var_origins", {}).get(v, set()))) for pr, v, _ in candidate_vars[:4]])
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

                # B. Check typestate-driven literal resolution from prompt or LLM slots
                if llm_slot_val is not None and _is_valid_slot_value(llm_slot_val, concrete_sig):
                    formatted_val = _format_literal_value(llm_slot_val, concrete_sig)
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
                    reg = TypeRegistry.get_instance()
                    key_binds = set(reg.get_verification_semantics("key_binds") or ())
                    proj_roles = set(reg.get_projection_tokens()) | set(reg.get_verification_semantics("projection_port_roles") or ())
                    is_key_req = p_sig.required and (
                        getattr(p_sig, "binds", "") in key_binds
                        or str(getattr(p_sig, "port_role", "") or "").lower() in proj_roles
                    )
                    if is_key_req and val.strip("'\"") == "":
                        # A required key/projection port cannot be grounded by an empty string default
                        pass
                    else:
                        c_tname = str(getattr(concrete_sig, "type_name", "") or "").lower()
                        c_is_str = c_tname in ("str", "enum", "column_identifier", "filepath", "path") or reg.is_subtype(c_tname, "str")
                        if c_is_str and not (val.startswith(("'", '"')) or val in ("None", "True", "False")):
                            val = repr(val)
                        cell_bindings[p_name] = val
                        accumulated_sigma.bind(p_name, val)
                        continue

                # D. Declared-domain enum grounding (reflection-driven, zero domain hardcodes)
                p_domain = getattr(concrete_sig, "domain", "") or ""
                if p_domain and getattr(ctx, "prompt", ""):
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
                                        valid_nums = {v for _, k, v in getattr(ctx, "ordered_literals", []) if k in ("int_literal", "float_literal")}
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

                # E.2 Unresolved REQUIRED port: fail loudly if still ungrounded
                if p_sig.required:
                    ctx.unresolved_ports.append((cell.cell_id, p_name))
                    ctx.unbindable_count += 1
                    cell_bindings[p_name] = UNRESOLVED_PORT
                else:
                    cell_bindings[p_name] = None

            # Bound input variables for this cell, computed ONCE after all input
            # bindings are final and before any output branch runs (R6-1). Only
            # strings that are variables in scope count (matches the semantics the
            # pre-R5 refactor code had at both of its former definition sites).
            bound_in_vars = [v for v in cell_bindings.values() if isinstance(v, str) and v in ctx.variables]

            # Register output port(s) in context for future steps
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
                        var_counter += 1
                        ctx.var_counter = var_counter
                        out_var = _semantic_var_name(cell, var_counter, port_name=out_name, out_sig=out_sig)
                        if idx_out == 0:
                            first_out_var = out_var
                        cell_bindings[out_name] = out_var
                        cell_port_vars[(cell.cell_id, out_name)] = out_var
                        cell_port_vars[(id(cell), out_name)] = out_var

                        concrete_out = substitute_generics(out_sig, accumulated_sigma)
                        ctx.declare_variable(out_var, concrete_out, out_var, cell=cell)
                        _update_variable_provenance(ctx, out_var, cell, cell_bindings)
                        if prim_out and out_name == prim_out.name:
                            prim_var = out_var

                    producer_var = prim_var if prim_var is not None else (first_out_var or current_out_var)
                    if "output_var" not in cell.outputs:
                        cell_bindings["output_var"] = producer_var
                    cell_out_vars[cell.cell_id] = producer_var
                    cell_out_vars[id(cell)] = producer_var
                    cell_port_vars[(cell.cell_id, "output_var")] = producer_var
                    cell_port_vars[(id(cell), "output_var")] = producer_var

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
                            cell_bindings["output_var"] = current_out_var

                    # Bind primary output port name if distinct from output_var
                    if cell.primary_output and cell.primary_output.name:
                        cell_bindings[cell.primary_output.name] = current_out_var
                        cell_port_vars[(cell.cell_id, cell.primary_output.name)] = current_out_var
                        cell_port_vars[(id(cell), cell.primary_output.name)] = current_out_var

                    if current_out_var is not None:
                        # Forward Rank Propagation:
                        if getattr(concrete_out, "canonical_rank", None) is None and cell.primary_input:
                            bound_in_name = cell_bindings.get(cell.primary_input.name)
                            if bound_in_name and bound_in_name in ctx.variables:
                                in_sig_tuple = ctx.variables[bound_in_name]
                                in_port_sig = in_sig_tuple[0] if isinstance(in_sig_tuple, tuple) else in_sig_tuple
                                in_r = getattr(in_port_sig, "canonical_rank", None)
                                if in_r is not None:
                                    concrete_out.canonical_rank = in_r
                        ctx.declare_variable(current_out_var, concrete_out, current_out_var, cell=cell)
                        producer_var = current_out_var
                        cell_out_vars[cell.cell_id] = current_out_var
                        cell_out_vars[id(cell)] = current_out_var
                        cell_port_vars[(cell.cell_id, "output_var")] = current_out_var
                        cell_port_vars[(id(cell), "output_var")] = current_out_var

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
                        _update_variable_provenance(ctx, current_out_var, cell, cell_bindings)

                        if getattr(cell, "stage", None) == 2 and out_t:
                            canon_out = registry.canonical_name(out_t).lower()
                            for b_in_v in bound_in_vars:
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

            # Try parsing s as an evaluated Python expression (handles array projections, bridges, indexed lookups)
            try:
                parsed_expr = ast.parse(s, mode="eval").body
                free_names = {node.id for node in ast.walk(parsed_expr) if isinstance(node, ast.Name)}
                if free_names and all(
                    (v in active_vars or v.startswith("var_") or v.startswith("out_") or _is_resolvable_module_or_builtin(v))
                    for v in free_names
                ):
                    return parsed_expr
                if isinstance(parsed_expr, (ast.List, ast.Dict, ast.Tuple, ast.Set)) and not any(
                    not (v in active_vars or v.startswith("var_") or v.startswith("out_") or _is_resolvable_module_or_builtin(v))
                    for v in free_names
                ):
                    return parsed_expr
                if not free_names and isinstance(parsed_expr, ast.Constant):
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
                for a in getattr(node.args, "args", []):
                    if a.arg in ph_map:
                        orig = ph_map[a.arg]
                        if orig in bindings and bindings[orig] is not None:
                            val = str(bindings[orig]).strip().strip("'\"")
                            if val.isidentifier():
                                a.arg = val
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
            if not mm.can_synthesize():
                return
        except Exception:
            return

        registry = TypeRegistry.get_instance()
        candidate_ports_by_cell = {}
        for c in cells:
            if not hasattr(c, "inputs") or not c.inputs:
                continue
            c_ports = {}
            for p_name, p_sig in c.inputs.items():
                p_role = getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or ""
                t_name = (getattr(p_sig, "type_name", "") or "").lower()
                if p_role in (
                    _declared_role_semantics("dataflow_roles")
                    | _declared_role_semantics("projective_roles")
                    | _declared_role_semantics("strict_roles")
                ):
                    continue
                if registry.is_composite_carrier(t_name) or registry.is_abstract_carrier(t_name):
                    continue
                c_ports[p_name] = {
                    "type": t_name or "any",
                    "required": getattr(p_sig, "required", False),
                    "default": str(getattr(p_sig, "default_value", None)),
                    "enum_values": getattr(p_sig, "enum_values", None)
                }
            if c_ports:
                candidate_ports_by_cell[c.cell_id] = c_ports

        if not candidate_ports_by_cell:
            return

        schema = {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "description": "Mapping from cell_id to port_name and extracted value",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                    }
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "General hyperparameters or arguments extracted from prompt",
                    "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                }
            }
        }

        cell_summary = []
        for cid, ports in candidate_ports_by_cell.items():
            port_descs = []
            for p, info in ports.items():
                e_vals = info.get("enum_values")
                e_str = f" [options: {e_vals}]" if e_vals else ""
                port_descs.append(f"{p}: {info['type']}{e_str}")
            cell_summary.append(f"- Cell '{cid}': {', '.join(port_descs)}")

        query = (
            f"User Prompt: {prompt}\n\n"
            f"Extract literal values (file paths, configurations, thresholds, hyperparameters) for the following pipeline nodes:\n"
            + "\n".join(cell_summary) + "\n\n"
            "Return valid JSON only matching schema with 'slots' ({cell_id: {port_name: value}}) and 'hyperparameters'."
        )

        try:
            raw = mm.generate_text(
                query,
                max_tokens=512,
                schema=schema,
                system_prompt="You are a precise dataflow parameter extractor. Extract exact parameter literals, configurations, and options mentioned in the prompt."
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
                    if isinstance(slots, dict):
                        for k, v in slots.items():
                            if isinstance(v, dict):
                                if k not in ctx.llm_slots:
                                    ctx.llm_slots[k] = {}
                                for p_name, val in v.items():
                                    if val is not None:
                                        c_match = next((cl for cl in cells if cl.cell_id == k), None)
                                        p_sig = c_match.inputs.get(p_name) if c_match and hasattr(c_match, "inputs") else None
                                        concrete_sig = getattr(p_sig, "signature", p_sig) if p_sig else None
                                        if concrete_sig and not _is_valid_slot_value(val, concrete_sig):
                                            continue
                                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                                            val_str = str(int(val)) if int(val) == val else str(val)
                                            valid_nums = {v for _, k_lit, v in getattr(ctx, "ordered_literals", []) if k_lit in ("int_literal", "float_literal")}
                                            if val_str not in valid_nums:
                                                continue
                                        ctx.llm_slots[k][p_name] = val
                            elif v is not None:
                                if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                                    ctx.hyperparameters = {}
                                ctx.hyperparameters[k] = v
                    hparams = parsed.get("hyperparameters", {})
                    if isinstance(hparams, dict):
                        if not hasattr(ctx, "hyperparameters") or ctx.hyperparameters is None:
                            ctx.hyperparameters = {}
                        for hk, hv in hparams.items():
                            if hv is not None:
                                ctx.hyperparameters[hk] = hv
        except Exception as e:
            logger.debug(f"[UNIFICATION] LLM slot extraction skipped or failed: {e}")

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
            extract_llm_slots = (
                (os.environ.get("NSTL_LLM_FEEDBACK") == "1")
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
            pipeline_bindings = cells
            # Autonomous Missing Parameter Resolution under --llm-feedback for pre-bound tuples
            # Strictly disabled in symbolic/retrieval profiles (0, A, D)
            if getattr(ctx, "unresolved_ports", None) and (extract_llm_slots or os.environ.get("NSTL_LLM_FEEDBACK") == "1"):
                try:
                    from inference import ModelManager
                except (ImportError, ValueError):
                    from .inference import ModelManager
                mm = ModelManager.get_instance()
                p_name_cur = getattr(mm, "current_profile_name", "")
                if mm.profile and p_name_cur not in ("0", "A", "D") and mm.can_synthesize():
                    prompt_txt = getattr(ctx, "_prompt", "") or getattr(self.context, "_prompt", "")
                    resolved_any = False
                    rem_unres = []
                    cell_objs = [c for c, _ in cells]
                    for cid, p_name in list(ctx.unresolved_ports):
                        cell_obj = next((c for c in cell_objs if c.cell_id == cid), None)
                        p_sig = cell_obj.inputs.get(p_name) if cell_obj else None
                        p_desc = getattr(p_sig, "description", "") or getattr(p_sig, "type_name", "") or p_name
                        query = (
                            f"User Prompt: {prompt_txt}\n"
                            f"Missing Parameter Required:\n"
                            f"Node: {cid}\n"
                            f"Parameter Port: {p_name} ({p_desc})\n"
                            f"Extract the exact argument value (e.g. identifier, threshold, asset path, configuration mode) from the prompt for this port.\n"
                            f"Return JSON: {{\"value\": <extracted_value>}}"
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
                                    ctx.parameters[p_name] = val
                                    ctx.target_literals.add(str(val))
                                    resolved_any = True
                                    continue
                        except Exception as _ex:
                            logger.debug(f"[UNIFICATION] LLM missing parameter extraction failed for {cid}.{p_name}: {_ex}")
                        rem_unres.append((cid, p_name))
                    if resolved_any:
                        ctx.unresolved_ports = rem_unres
                        res = self.unify_pipeline(cell_objs, ctx, allow_llm=False)
                        if not res.is_bottom() and isinstance(res, Success):
                            pipeline_bindings = res.value
                            accum_sigma = res.sigma
                            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
                            self.last_runtime_aliases = dict(getattr(ctx, "runtime_aliases", {}) or {})
            if getattr(ctx, "unresolved_ports", None):
                ports_str = ", ".join(f"{cid}.{p}" for cid, p in ctx.unresolved_ports)
                raise UnresolvedPlaceholderError(
                    f"Code synthesis refused: pipeline contains unresolved ports: [{ports_str}]"
                )
        else:
            ctx_run = ctx.clone() if hasattr(ctx, "clone") else ctx
            if hasattr(ctx_run, "reset"):
                ctx_run.reset()
            if extract_llm_slots:
                self._extract_pipeline_slots_with_llm(cells, getattr(ctx_run, "_prompt", "") or getattr(self.context, "_prompt", ""), ctx_run)
            res = self.unify_pipeline(cells, ctx_run, allow_llm=extract_llm_slots)
            if res.is_bottom():
                reason = res.reason if isinstance(res, Failure) else "Unknown unification failure"
                raise ValueError(f"Unification Failed: {reason}")
            assert isinstance(res, Success)
            pipeline_bindings = res.value
            accum_sigma = res.sigma
            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
            self.last_runtime_aliases = dict(getattr(ctx_run, "runtime_aliases", {}) or {})

            # Autonomous Missing Parameter Resolution under --llm-feedback
            # Strictly disabled in symbolic/retrieval profiles (0, A, D)
            if getattr(ctx_run, "unresolved_ports", None) and (extract_llm_slots or os.environ.get("NSTL_LLM_FEEDBACK") == "1"):
                try:
                    from inference import ModelManager
                except (ImportError, ValueError):
                    from .inference import ModelManager
                mm = ModelManager.get_instance()
                p_name_cur = getattr(mm, "current_profile_name", "")
                if mm.profile and p_name_cur not in ("0", "A", "D") and mm.can_synthesize():
                    prompt_txt = getattr(ctx_run, "_prompt", "") or getattr(self.context, "_prompt", "")
                    resolved_any = False
                    rem_unres = []
                    for cid, p_name in list(ctx_run.unresolved_ports):
                        cell_obj = next((c for c in cells if c.cell_id == cid), None)
                        p_sig = cell_obj.inputs.get(p_name) if cell_obj else None
                        p_desc = getattr(p_sig, "description", "") or getattr(p_sig, "type_name", "") or p_name
                        query = (
                            f"User Prompt: {prompt_txt}\n"
                            f"Missing Parameter Required:\n"
                            f"Node: {cid}\n"
                            f"Parameter Port: {p_name} ({p_desc})\n"
                            f"Extract the exact argument value (e.g. identifier, threshold, asset path, configuration mode) from the prompt for this port.\n"
                            f"Return JSON: {{\"value\": <extracted_value>}}"
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
                                    ctx_run.parameters[p_name] = val
                                    ctx_run.target_literals.add(str(val))
                                    resolved_any = True
                                    continue
                        except Exception as _ex:
                            logger.debug(f"[UNIFICATION] LLM missing parameter extraction failed for {cid}.{p_name}: {_ex}")
                        rem_unres.append((cid, p_name))
                    if resolved_any:
                        ctx_run.unresolved_ports = rem_unres
                        res = self.unify_pipeline(cells, ctx_run, allow_llm=False)
                        if not res.is_bottom() and isinstance(res, Success):
                            pipeline_bindings = res.value
                            accum_sigma = res.sigma
                            self.last_egress_paths = self._derive_egress_paths(pipeline_bindings)
                            self.last_runtime_aliases = dict(getattr(ctx_run, "runtime_aliases", {}) or {})

            if getattr(ctx_run, "unresolved_ports", None):
                ports_str = ", ".join(f"{cid}.{p}" for cid, p in ctx_run.unresolved_ports)
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
        self.last_verification_contract = self.build_verification_contract(pipeline_bindings, ctx, getattr(ctx, "prompt", ""))

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
                    words = frozenset(tokenize_alphanumeric(tpl))
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

        if ctx is not None and hasattr(ctx, "composed_bridge_cells"):
            for bcell in ctx.composed_bridge_cells:
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
        if ctx is not None:
            if hasattr(ctx, "variables"):
                known_vars.update(ctx.variables.keys())
            if hasattr(ctx, "scope"):
                known_vars.update(ctx.scope.keys())
            if hasattr(ctx, "var_sources"):
                known_vars.update(ctx.var_sources.keys())

        for cell, bindings in pipeline_bindings:
            out_v = bindings.get("output_var")
            if out_v and isinstance(out_v, str):
                known_vars.add(out_v.strip().strip("'\""))
            if getattr(cell, "primary_output", None):
                p_out = bindings.get(cell.primary_output.name)
                if p_out and isinstance(p_out, str):
                    known_vars.add(p_out.strip().strip("'\""))

            if getattr(cell, "is_typed_hole", False):
                try:
                    from synthesis import InductiveKernelGenerator
                except (ImportError, ValueError):
                    from .synthesis import InductiveKernelGenerator
                gen = InductiveKernelGenerator()
                in_sigs = {bnd_v: p.type_name for p_k, p in cell.inputs.items() if (bnd_v := bindings.get(p_k))}
                out_port = getattr(cell, "primary_output", None)
                out_port_name = getattr(out_port, "name", None) if out_port else (list(cell.outputs.keys())[0] if getattr(cell, "outputs", None) else None)
                out_v = (
                    bindings.get("output_var")
                    or (bindings.get(out_port_name) if out_port_name else None)
                    or next((bindings.get(k) for k in getattr(cell, "outputs", {}) if bindings.get(k)), None)
                    or "var_out"
                )
                out_sigs = {out_v: getattr(cell.primary_output, "type_name", "any")}
                code_lines.append(
                    gen.generate_kernel(
                        task_description=getattr(cell, "task_description", cell.cell_id),
                        state_inputs=in_sigs,
                        state_outputs=out_sigs,
                    )
                )
                continue

            if has_render_cell and (bool(getattr(cell, "bound_slots", None)) or getattr(cell, "node_type", "") == "macro"):
                rendered = render_cell(cell, bindings, indent_level=0, context=ctx, accumulated_sigma=accum_sigma)
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
        if ctx is not None and getattr(ctx, "prompt", None) and pipeline_bindings:
            target_ident = ExecutionContext._extract_target_sink(ctx.prompt)
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
                    declare_alias = getattr(ctx, "declare_runtime_alias", None)
                    if callable(declare_alias):
                        declare_alias(target_ident, str(out_var))

        final_code = "\n".join(code_lines).strip()
        final_code = self._reconcile_imports(final_code)
        prior_vars: Set[str] = set()
        if ctx is not None:
            if hasattr(ctx, "initial_scope") and isinstance(ctx.initial_scope, set):
                prior_vars.update(ctx.initial_scope)
            else:
                if hasattr(ctx, "scope") and isinstance(ctx.scope, dict):
                    prior_vars.update(ctx.scope.keys())
                if hasattr(ctx, "scope_variables") and isinstance(ctx.scope_variables, dict):
                    prior_vars.update(ctx.scope_variables.keys())
                if hasattr(ctx, "variables") and isinstance(ctx.variables, dict):
                    prior_vars.update(ctx.variables.keys())

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
                            if b_val in ctx.variables or b_val.startswith("var_"):
                                pipeline_vars.add(b_val)

            if hasattr(ctx, "variables") and isinstance(ctx.variables, dict):
                for v_name in ctx.variables.keys():
                    if isinstance(v_name, str):
                        pipeline_vars.add(v_name)

            if hasattr(ctx, "scope_variables") and isinstance(ctx.scope_variables, dict):
                for v_name in ctx.scope_variables.keys():
                    if isinstance(v_name, str):
                        pipeline_vars.add(v_name)

            if hasattr(ctx, "runtime_aliases") and isinstance(ctx.runtime_aliases, dict):
                pipeline_vars.update(ctx.runtime_aliases.keys())
                pipeline_vars.update(ctx.runtime_aliases.values())

            prior_vars.difference_update(pipeline_vars)
            ctx_pipe_vars = getattr(ctx, "pipeline_vars", set()) if ctx else set()
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
            if hasattr(self.context, "blackboard") and self.context.blackboard is not None:
                self.context.blackboard.lock_from_intent(intent_data)
                if self.context.blackboard.locked_by_neural:
                    if self.context.blackboard.target_sink:
                        self.context.parameters["target_sink"] = self.context.blackboard.target_sink
            if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
                self.context.hyperparameters = dict(intent_data["hyperparameters"])
            if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
                self.context.llm_slots = dict(intent_data["slots"])
                if hasattr(self.context, "sync_llm_slots"):
                    self.context.sync_llm_slots()
            tgt = intent_data.get("target_sink") or intent_data.get("target_key")
            if tgt:
                self.context.target_key = str(tgt).strip()
                self.context.target_literals.add(str(tgt).strip().lower())
            # Profile S (Semantic Compiler): literals extracted by the typed
            # IR compiler become declared context parameters — deterministic
            # binding without re-scanning the prompt.
            if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
                for p_key, p_val in intent_data["parameters"].items():
                    if p_key:
                        self.context.parameters.setdefault(str(p_key), p_val)
            if intent_data.get("by_column"):
                self.context.parameters["by_column"] = str(intent_data["by_column"]).strip()
            if intent_data.get("source_files"):
                self.context.parameters["source_uris"] = list(intent_data["source_files"])
            if intent_data.get("dest_files"):
                self.context.parameters["dest_uris"] = list(intent_data["dest_files"])
        if not self.context.parameters.get("source_uris") and self.context.source_files:
            self.context.parameters["source_uris"] = list(self.context.source_files)
        if not self.context.parameters.get("dest_uris") and self.context.dest_files:
            self.context.parameters["dest_uris"] = list(self.context.dest_files)
        extract_llm = (os.environ.get("NSTL_LLM_FEEDBACK") == "1") or bool(getattr(self.context, "llm_feedback", False))
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
        if ctx.unresolved_ports:
            ctx.unresolved_ports.clear()
            for c, bnd in pipeline_bindings:
                for k, v in bnd.items():
                    if v is UNRESOLVED_PORT or (isinstance(v, str) and v in ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")):
                        bnd[k] = k
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
    term1 = TypeTerm.from_string(tau_expected)
    term2 = TypeTerm.from_string(tau_actual)
    return unify(term1, term2) is not None


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


class DynamicPlaceholderResolver:
    """Compatibility resolver delegating to monadic ExecutionContext and UnificationGate."""
    def __init__(self):
        self.context = ExecutionContext()
        self.gate = UnificationGate()

    def assert_placeholders_resolved(self, code_str: str):
        assert_placeholders_resolved(code_str)

    def resolve_port(self, port_name: str, port_sig: Any, stage: int, ctx: Any, current_out_var: str) -> str:
        sig = getattr(port_sig, "signature", port_sig)
        t_name = str(getattr(sig, "type_name", "")).lower()
        s_name = str(getattr(sig, "state", "")).lower()
        p_lower = str(port_name).lower()
        port_role = str(getattr(port_sig, "port_role", "") or getattr(sig, "port_role", "") or getattr(port_sig, "role", "") or "").lower()
        abstract_t = str(getattr(port_sig, "abstract_type", "") or getattr(sig, "abstract_type", "") or "").lower()
        p_kind = str(getattr(port_sig, "param_kind", "") or getattr(sig, "param_kind", "") or "").lower()
        is_path = _lattice_is_path_port(port_sig)
        registry = TypeRegistry.get_instance()
        path_tokens = registry.get_path_port_tokens()
        dest_tokens = registry.get_dest_port_tokens()
        term_stages = {int(s) for s in _declared_role_semantics("terminal_stages") if str(s).isdigit()}
        ing_stages = {int(s) for s in _declared_role_semantics("ingress_stages") if str(s).isdigit()}
        dest_states = set(r.lower() for r in registry.get_verification_semantics("materialization_states")) | set(r.lower() for r in registry.get_verification_semantics("terminal_states"))
        source_states = set(r.lower() for r in registry.get_verification_semantics("source_states"))
        sink_roles = (
            set(r.lower() for r in registry.get_verification_semantics("sink_roles"))
            | set(r.lower() for r in registry.get_verification_semantics("terminal_roles"))
            | set(r.lower() for r in registry.get_verification_semantics("destination_roles"))
        )
        source_roles = (
            set(r.lower() for r in registry.get_verification_semantics("source_roles"))
            | set(r.lower() for r in registry.get_verification_semantics("ingress_roles"))
            | set(r.lower() for r in registry.get_verification_semantics("source_data_roles"))
        )

        # 1. Source files / paths
        is_source_port = (
            port_role in source_roles
            or s_name in source_states
            or (is_path and (stage in ing_stages or port_role in source_roles))
            or (stage in ing_stages and is_path)
            or (abstract_t == "path" and stage in ing_stages)
            or p_lower in path_tokens
            or p_lower in registry.get_source_port_tokens()
        )
        if is_source_port and hasattr(ctx, "source_files") and ctx.source_files:
            chosen = None
            if hasattr(ctx, "current_cell_clause_idx") and ctx.current_cell_clause_idx is not None and hasattr(ctx, "_get_clause_idx_for_pos"):
                for pos, kind, val in getattr(ctx, "ordered_literals", []):
                    if val in ctx.source_files and ctx._get_clause_idx_for_pos(pos) == ctx.current_cell_clause_idx:
                        if val not in getattr(ctx, "consumed_assets", set()):
                            chosen = val
                            break
            if not chosen:
                for sf in ctx.source_files:
                    if sf not in getattr(ctx, "consumed_assets", set()):
                        chosen = sf
                        break
            if not chosen:
                chosen = ctx.source_files[0]
            if not hasattr(ctx, "consumed_assets"):
                ctx.consumed_assets = set()
            ctx.consumed_assets.add(chosen)
            return f'"{chosen}"'

        # 2. Destination files / paths
        is_dest_port = (
            port_role in sink_roles
            or s_name in dest_states
            or (is_path and (stage in term_stages or port_role in sink_roles))
            or (stage in term_stages and is_path)
            or (abstract_t == "path" and stage in term_stages)
            or p_lower in dest_tokens
            or p_lower in registry.get_dest_port_tokens()
        )
        if is_dest_port and hasattr(ctx, "dest_files") and ctx.dest_files:
            chosen = None
            if hasattr(ctx, "current_cell_clause_idx") and ctx.current_cell_clause_idx is not None and hasattr(ctx, "_get_clause_idx_for_pos"):
                for pos, kind, val in getattr(ctx, "ordered_literals", []):
                    if val in ctx.dest_files and ctx._get_clause_idx_for_pos(pos) == ctx.current_cell_clause_idx:
                        if val not in getattr(ctx, "consumed_assets", set()):
                            chosen = val
                            break
            if not chosen:
                for df in ctx.dest_files:
                    if df not in getattr(ctx, "consumed_assets", set()):
                        chosen = df
                        break
            if not chosen:
                chosen = ctx.dest_files[-1]
            if not hasattr(ctx, "consumed_assets"):
                ctx.consumed_assets = set()
            ctx.consumed_assets.add(chosen)
            return f'"{chosen}"'

        # 3. Component / key identifiers
        proj_roles = (
            _declared_role_semantics("projection_port_roles")
            | _declared_role_semantics("key_roles")
        )
        is_key_port = (
            port_role in proj_roles
            or getattr(port_sig, "is_projection", False)
            or str(getattr(port_sig, "binds", "")).lower() in _declared_role_semantics("key_binds")
        )
        if is_key_port and hasattr(ctx, "ordered_literals"):
            chosen = None
            if hasattr(ctx, "current_cell_clause_idx") and ctx.current_cell_clause_idx is not None and hasattr(ctx, "_get_clause_idx_for_pos"):
                for pos, kind, val in ctx.ordered_literals:
                    if kind == "quoted_str" and not ExecutionContext._is_path_string(val) and ctx._get_clause_idx_for_pos(pos) == ctx.current_cell_clause_idx:
                        if val not in getattr(ctx, "consumed_keys", set()):
                            chosen = val
                            break
            if not chosen:
                for _, kind, val in ctx.ordered_literals:
                    if kind == "quoted_str" and not ExecutionContext._is_path_string(val) and val not in getattr(ctx, "consumed_keys", set()):
                        chosen = val
                        break
            if chosen is not None:
                if not hasattr(ctx, "consumed_keys"):
                    ctx.consumed_keys = set()
                ctx.consumed_keys.add(chosen)
                return f'"{chosen}"'

        # 4. Operational flags
        if hasattr(ctx, "flags") and isinstance(ctx.flags, dict) and port_name in ctx.flags:
            return str(ctx.flags[port_name])

        # 5. Direct context parameters
        if hasattr(ctx, "parameters") and port_name in ctx.parameters:
            return str(ctx.parameters[port_name])

        # 6. Default value (omit None default unless handles_none)
        if getattr(port_sig, "default_value", None) is not None:
            def_str = str(port_sig.default_value)
            if def_str in ("None", "none"):
                if getattr(port_sig, "handles_none", False):
                    return "None"
                return ""
            return def_str

        # 7. Type-compatible in-scope dataflow variable
        if hasattr(ctx, "scope_variables") and ctx.scope_variables:
            target_type = getattr(sig, "type_name", None)
            if target_type and target_type not in ("any", "*", "top"):
                for v_name, v_sig in reversed(list(ctx.scope_variables.items())):
                    v_type = getattr(getattr(v_sig, "signature", v_sig), "type_name", None)
                    if v_type == target_type:
                        return v_name
            return list(ctx.scope_variables.keys())[-1]

        return current_out_var


PlaceholderResolver = DynamicPlaceholderResolver


class DataflowLineageTracker(ast.NodeTransformer):
    """
    Tracks sequential variable transformations and re-links sink calls
    to the latest valid descendant in the lineage chain.
    """
    def __init__(self, target_cells=None):
        self.target_cells = target_cells or []
        self.lineage_tree: Dict[str, str] = {}
        self.latest_descendant: Dict[str, str] = {}
        self.assigned_vars: Set[str] = set()
        self._imported_names: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name.split('.')[0])
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name)
        if node.module:
            self._imported_names.add(node.module.split('.')[0])
        return node

    def visit_Assign(self, node: ast.Assign):
        self._rebind_sink_call(node.value)
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            self.assigned_vars.add(target_name)

            parent_var = None
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
                    candidate = node.value.func.value.id
                    if candidate not in self._imported_names:
                        parent_var = candidate
                elif isinstance(node.value.func, ast.Name):
                    for arg in node.value.args:
                        if isinstance(arg, ast.Name) and arg.id in self.assigned_vars:
                            parent_var = arg.id
                            break

            if parent_var:
                root = self._get_root(parent_var)
                self.lineage_tree[target_name] = parent_var
                self.latest_descendant[root] = target_name
                self.latest_descendant[parent_var] = target_name

        return node

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        self._rebind_sink_call(node.value)
        return node

    def _rebind_sink_call(self, call_node):
        if not isinstance(call_node, ast.Call):
            return

        method_name = ""
        callee_var = None
        if isinstance(call_node.func, ast.Attribute):
            method_name = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):
                callee_var = call_node.func.value.id
        elif isinstance(call_node.func, ast.Name):
            method_name = call_node.func.id

        # Sink-ness of the callee name is judged against the DECLARED egress /
        # destination vocabulary harvested from the trees (token match plus
        # declared-token substring for affixed forms).
        is_sink = False
        if method_name:
            try:
                _reg = TypeRegistry.get_instance()
                _sink_toks = set(_reg.get_egress_tokens()) | set(_reg.get_dest_port_tokens())
            except Exception:
                _sink_toks = set()
            m_lower = method_name.lower()
            m_toks = {t for t in CellTokenizer.tokenize_identifier(method_name)}
            if (m_toks & _sink_toks) or any(len(k) >= 3 and k in m_lower for k in _sink_toks):
                is_sink = True
        for cell in self.target_cells:
            if getattr(cell.outputs, "type_name", "") == "None" or getattr(cell, "metadata_tags", {}).get("is_sink", False):
                is_sink = True
                break

        if is_sink:
            if callee_var and callee_var in self.latest_descendant and callee_var not in self._imported_names:
                newest_var = self.latest_descendant[callee_var]
                if newest_var != callee_var:
                    call_node.func.value.id = newest_var

            for arg in call_node.args:
                if isinstance(arg, ast.Name) and arg.id in self.latest_descendant:
                    arg.id = self.latest_descendant[arg.id]

    def _get_root(self, var_name: str) -> str:
        curr = var_name
        while curr in self.lineage_tree:
            curr = self.lineage_tree[curr]
        return curr


def enforce_lineage_integrity(code: str, target_cells=None) -> str:
    """Parses generated code, traces lineage, and auto-corrects stale variable usages."""
    try:
        tree = ast.parse(code)
        transformer = DataflowLineageTracker(target_cells=target_cells)
        corrected_tree = transformer.visit(tree)
        ast.fix_missing_locations(corrected_tree)
        return ast.unparse(corrected_tree)
    except Exception:
        return code


@dataclass
class ExtractedSlots:
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    named_identifiers: List[str] = field(default_factory=list)
    numeric_literals: List[Union[int, float]] = field(default_factory=list)
    operational_flags: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_uris(self) -> List[str]:
        return self.inputs

    @source_uris.setter
    def source_uris(self, val: List[str]) -> None:
        self.inputs = list(val)

    @property
    def dest_uris(self) -> List[str]:
        return self.outputs

    @dest_uris.setter
    def dest_uris(self, val: List[str]) -> None:
        self.outputs = list(val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "named_identifiers": self.named_identifiers,
            "numeric_literals": self.numeric_literals,
            "operational_flags": self.operational_flags,
            "parameters": self.parameters,
            "source_uris": self.inputs,
            "dest_uris": self.outputs,
            "input_files": self.inputs,
            "output_files": self.outputs,
            "columns": self.named_identifiers,
            "by_column": self.named_identifiers[0] if self.named_identifiers else None,
        }


class ParameterExtractor:
    """Compatibility adapter over ExecutionContext."""

    @staticmethod
    def _declared_flag_patterns() -> List[Tuple[Any, str, Any, Dict[str, Any]]]:
        """
        Keyword->flag observation patterns are DECLARED DATA harvested from
        the loaded domain trees (tree key `semantic_flag_patterns`). The
        engine ships zero patterns of its own: e.g. a vision tree declares
        its colorspace observers and the primitives tree declares the
        language-level ordering observers (ascending/descending).
        """
        try:
            declared = TypeRegistry.get_instance().get_semantic_flag_patterns()
        except Exception:
            return []
        return [
            (p["keywords"], p["flag"], p["value"], dict(p.get("extras") or {}))
            for p in declared
        ]

    @staticmethod
    def extract_slots(prompt: str) -> ExtractedSlots:
        ctx = ExecutionContext(prompt=prompt)
        slots = ExtractedSlots()
        slots.source_uris = list(ctx.source_files)
        slots.dest_uris = list(ctx.dest_files)
        slots.numeric_literals = [
            float(val) if "." in val else int(val)
            for _, kind, val in ctx.ordered_literals if kind == "numeric"
        ]
        quoted_strings = [val for _, kind, val in ctx.ordered_literals if kind == "quoted_str" and "." not in val]
        if quoted_strings:
            slots.named_identifiers.extend(quoted_strings)

        flags = dict(ctx.flags)
        if prompt:
            p_low = prompt.lower()
            for keywords, flag_key, flag_val, extras in ParameterExtractor._declared_flag_patterns():
                if any(kw in p_low for kw in keywords):
                    flags[flag_key] = flag_val
                    flags.update(extras)

        slots.operational_flags = flags
        return slots

    @staticmethod
    def extract_parameters(prompt: str) -> Dict[str, Any]:
        return ParameterExtractor.extract_slots(prompt).to_dict()


def _enum_rule_for_placeholder(placeholder_lower: str) -> Optional[Dict[str, Any]]:
    """
    Finds a DECLARED enum-constant grounding rule (tree key
    `enum_constant_rules`) governing a placeholder name. Returns None when no
    tree declared a rule — the engine carries no placeholder vocabulary.
    """
    try:
        rules = TypeRegistry.get_instance().get_enum_constant_rules()
    except Exception:
        return None
    for rule in rules or []:
        if placeholder_lower in (rule.get("placeholder_names") or ()):  # declared frozenset
            return rule
    return None


def resolve_node_slots(template: str, extracted_params: Dict[str, Any]) -> Dict[str, str]:
    """Deterministically binds extracted prompt parameters to template slot placeholders.

    Placeholder classification (path / projection / polarity) is driven by the
    DECLARED token vocabulary harvested from the trees (path port tokens,
    destination and egress tokens, projection tokens, preposition triggers,
    semantic flag patterns). The engine carries no placeholder-name literals.
    """
    slots = {}
    placeholders = sorted(extract_template_placeholders(template))
    src_uris = extracted_params.get("source_uris", []) or extracted_params.get("input_files", []) or extracted_params.get("inputs", [])
    dst_uris = extracted_params.get("dest_uris", []) or extracted_params.get("output_files", []) or extracted_params.get("outputs", [])
    by_key = extracted_params.get("key") or extracted_params.get("by_column") or (extracted_params.get("named_identifiers", [None])[0] if extracted_params.get("named_identifiers") else None)
    flags = extracted_params.get("operational_flags", {})

    try:
        _reg = TypeRegistry.get_instance()
        _path_toks = {"filename", "pathname", "uri"} | set(_reg.get_path_port_tokens())
        _dest_toks = set(_reg.get_dest_port_tokens()) | set(_reg.get_egress_tokens())
        _proj_toks = set(_reg.get_column_projection_tokens()) | set(_reg.get_preposition_triggers())
        _flag_defaults: Dict[str, bool] = {}
        _flag_names = set()
        for _p in _reg.get_semantic_flag_patterns():
            _f = str(_p.get("flag", "")).lower()
            if not _f:
                continue
            _flag_names.add(_f)
            _v = _p.get("value")
            if isinstance(_v, bool):
                # A placeholder named for a declared flag defaults to the value
                # the tree's own pattern declares for that flag.
                _flag_defaults.setdefault(_f, _v)
    except Exception:
        _path_toks = set()
        _dest_toks = set()
        _proj_toks = set()
        _flag_defaults = {}
        _flag_names = set()

    def _tokenized(s: str) -> Set[str]:
        return {t for t in CellTokenizer.tokenize_identifier(s) if len(t) >= 2}

    for ph in placeholders:
        ph_l = ph.lower()
        ph_toks = _tokenized(ph_l)
        if (ph_toks & _path_toks) or any(k in ph_l for k in ("filename", "path", "uri")):
            if (ph_toks & _dest_toks) or any(
                len(k) >= 4 and k in ph_l for k in _dest_toks
            ):
                if dst_uris:
                    slots[ph] = f"'{dst_uris[-1]}'"
            else:
                if src_uris:
                    slots[ph] = f"'{src_uris[0]}'"
        elif ph_l in _proj_toks or (ph_toks and ph_toks <= _proj_toks):
            if by_key:
                slots[ph] = f"'{by_key}'"
        elif ph_l in _flag_names:
            default = _flag_defaults.get(ph_l, True)
            slots[ph] = str(flags.get(ph_l, default))
        elif _enum_rule_for_placeholder(ph_l) is not None:
            # Declared enum-constant grounding (Solution: domain data in trees):
            # a rule declares the placeholder names it governs, the module
            # attribute prefix to resolve (e.g. a colorspace-constant prefix),
            # and which operational flags select which format suffix. The
            # engine only performs generic module reflection — the vocabulary
            # itself lives entirely in the domain tree.
            rule = _enum_rule_for_placeholder(ph_l)
            ph_needle = f"{{{ph}}}"
            mod_name = None
            if ph_needle in template:
                prefix = template.split(ph_needle)[0]
                lparen_idx = prefix.rfind("(")
                if lparen_idx != -1:
                    caller = prefix[:lparen_idx].strip()
                    call_token = caller.split()[-1] if caller else ""
                    if "." in call_token:
                        candidate_mod = call_token.split(".")[0].lstrip("=,([+-*/% ")
                        if candidate_mod.isidentifier():
                            mod_name = candidate_mod
            if not mod_name:
                continue
            attr_prefix = str(rule.get("module_attribute_prefix", ""))
            target_fmt = str(rule.get("default_format", ""))
            format_flags = dict(rule.get("format_flags") or {})
            for flag_key, fmt in format_flags.items():
                if flags.get(flag_key):
                    target_fmt = str(fmt)
                    break
            if not target_fmt:
                continue
            resolved_code = None
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                cand = next(
                    (attr for attr in dir(mod) if attr_prefix and attr.startswith(attr_prefix) and attr.endswith(target_fmt)),
                    None,
                )
                if not cand:
                    cand = next(
                        (attr for attr in dir(mod) if attr_prefix and attr.startswith(attr_prefix.rstrip("_")) and attr.endswith(target_fmt)),
                        None,
                    )
                if cand:
                    resolved_code = f"{mod_name}.{cand}"
            except Exception as _exc:
                logger.debug("[UNIFICATION DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
            slots[ph] = resolved_code or (f"{mod_name}.{attr_prefix}{target_fmt}" if attr_prefix else f"{mod_name}.{target_fmt}")
        else:
            if ph in extracted_params:
                val = extracted_params[ph]
                slots[ph] = repr(val) if not (isinstance(val, str) and (val.startswith("'") or val.startswith('"'))) else val
            elif ph_l in extracted_params:
                val = extracted_params[ph_l]
                slots[ph] = repr(val) if not (isinstance(val, str) and (val.startswith("'") or val.startswith('"'))) else val
            else:
                hp = extracted_params.get("hyperparameters", {})
                llm_s = extracted_params.get("slots", {}) or extracted_params.get("llm_slots", {})
                if ph in hp:
                    slots[ph] = repr(hp[ph])
                elif ph_l in hp:
                    slots[ph] = repr(hp[ph_l])
                elif ph in llm_s:
                    slots[ph] = repr(llm_s[ph])
                elif ph_l in llm_s:
                    slots[ph] = repr(llm_s[ph_l])

    return slots


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
            earlier_out = getattr(earlier_cell, "primary_output", None)
            if earlier_out is None and getattr(earlier_cell, "outputs", None):
                earlier_out = earlier_cell.outputs.get("output_var") or next(iter(earlier_cell.outputs.values()))
            earlier_state_props = {}
            if earlier_out is not None:
                e_sig = getattr(earlier_out, "signature", earlier_out)
                e_state = getattr(e_sig, "state", None) or getattr(earlier_out, "state", None)
                if e_state:
                    earlier_state_props = TypeRegistry.get_instance().get_state_properties(e_state)

            prop_conflict = False
            if cand_pres:
                def _c_prop(c, key, default=None):
                    if isinstance(c, dict):
                        return c.get(key, default)
                    return getattr(c, key, default)
                for pre in cand_pres:
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
                            if known_val is None and earlier_out is not None and hasattr(earlier_out, "rank"):
                                known_val = getattr(earlier_out, "rank", None)
                            if known_val is not None:
                                if p_op == "==" and known_val != p_val:
                                    prop_conflict = True
                                    break
                                elif p_op == "!=" and known_val == p_val:
                                    prop_conflict = True
                                    break
            if prop_conflict:
                continue
            for out_name, out_sig in earlier_cell.outputs.items():
                if not _shape_compatible(out_sig, p_sig):
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

                    score = _wire_rank(p_type, p_state, in_role, out_type, out_state, out_role,
                                       cand_clause, earlier_clause, -idx_c, aligned)

                    wire_key = (id(earlier_cell), out_name)
                    candidates.append((score, p_name, earlier_cell, out_name, wire_key, s_wire))

        # 2. Context variables if available
        if context is not None:
            avail_cell_ids = {id(c) for c in available_cells}
            for v_name, (v_sig, _) in context.variables.items():
                v_cell = getattr(context, "var_sources", {}).get(v_name)
                if v_cell is not None and id(v_cell) in avail_cell_ids:
                    continue
                all_known_origins = set().union(*getattr(context, "var_origins", {}).values()) if getattr(context, "var_origins", None) else set()
                v_origins = getattr(context, "var_origins", {}).get(v_name, set())
                _rej = _wire_rejection(cand, p_name, p_sig, v_name, context, TypeRegistry.get_instance())
                if _rej is not None:
                    continue
                if not _shape_compatible(v_sig, p_sig):
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
                    score = _wire_rank(p_type, p_state, in_role, v_type, v_state, v_role,
                                       cand_clause, v_clause, _order.index(v_name) - len(_order))

                    v_cell = getattr(context, "var_sources", {}).get(v_name)
                    wire_key = v_name
                    candidates.append((score, p_name, v_cell, v_name, wire_key, s_wire))

    # Deterministic greedy assignment with disjoint wire tracking
    candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_ports = set()
    assigned_wires = set()

    for score, p_name, prod_cell, out_name, wire_key, s_wire in candidates:
        if p_name in assigned_ports or wire_key in assigned_wires:
            continue
        assigned_ports.add(p_name)
        assigned_wires.add(wire_key)
        port_bindings[p_name] = (prod_cell, out_name)
        sub = s_wire
        bound_count += 1

    # Check remaining unbound ports against defaults / literals / projective carriers
    for p_name, p_sig in cand.inputs.items():
        if p_name in assigned_ports:
            continue
        is_required = getattr(p_sig, "required", False)
        default_val = getattr(p_sig, "default_value", None)
        if not is_required or default_val is not None:
            continue
        desc = getattr(p_sig, "description", None) or getattr(p_sig, "doc", "") or ""
        registry = TypeRegistry.get_instance()
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
            | _declared_role_semantics("dataflow_roles")
        )
        is_receiver = (
            p_name in rec_names
            or p_role in strict_roles
            or p_binds in carrier_binds
            or any(tok in str(desc).lower() for tok in (carrier_binds | rec_names))
        )
        satisfied = False
        if not is_receiver:
            t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
            if registry.is_subtype(t_name, "str") or registry.is_subtype(t_name, "int") or registry.is_subtype(t_name, "float") or registry.is_subtype(t_name, "scalar"):
                satisfied = True
            else:
                projective_roles = _declared_role_semantics("projective_roles")
                if (is_model_cand or has_explicit_port_role) and p_role in projective_roles:
                    can_project_type = bool(
                        registry.is_composite_carrier(t_name)
                        or registry.is_abstract_carrier(t_name)
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

        for post in (getattr(cell, "postconditions", None) or []):
            p_op = _prop_cond_field(post, "operator")
            p_target = _prop_cond_field(post, "target")
            p_prop = _prop_cond_field(post, "property")
            if p_op == "==" and p_target in ("output_var", None) and p_prop:
                mine[p_prop] = (_prop_cond_field(post, "value"), cell.cell_id, str(_prop_cond_field(post, "description", "") or ""))

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
            for (v, c), pr in list(member_props.items()):
                if v in in_carriers:
                    if (out, c) not in member_props:
                        member_props[(out, c)] = dict(pr)
                    # Functorial Monoidal Property Lifting: constituent member properties transfer to container
                    for prop_name, prop_val in pr.items():
                        if isinstance(prop_val, (tuple, list)) and prop_val[0] is True:
                            props[out] = {**props.get(out, {}), prop_name: prop_val}
    return issues


analyze = analyze_property_contracts


