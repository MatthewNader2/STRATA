"""
src/type_predicates.py - Canonical type predicates, slot validation, and literal formatting.

Single source of truth shared across UnificationGate, PipelineSlotBinder,
PreflightLinter, and RouteMethod so scalar, sequence, handle, and literal
classifications never diverge between stages.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

from log_config import get_logger

try:
    from .lattice import TypeRegistry, is_path_port as _lattice_is_path_port
    from .language_backend import get_active_language_backend
except (ImportError, ValueError):
    from lattice import TypeRegistry, is_path_port as _lattice_is_path_port
    from language_backend import get_active_language_backend

logger = get_logger("type_predicates")

_BUILTIN_TYPE_NAMES: Set[str] = {
    k.lower() for k, v in builtins.__dict__.items() if isinstance(v, type)
}
_NON_LIT_STRUCTURAL_ROLES: Set[str] = {"receiver", "estimator", "container"}


@dataclass(frozen=True)
class NumericSeqSpec:
    """Parsed specification for a numeric tuple or list port."""

    base: str
    is_tuple: bool
    elem_is_int: bool
    arity: Optional[int]


def _get_registry(registry: Optional[Any] = None) -> Optional[Any]:
    if registry is not None:
        return registry
    try:
        return TypeRegistry.get_instance()
    except Exception:
        return None


def _parse_type_base_and_args(
    type_name: str,
    type_term_cls: Optional[Any] = None,
) -> tuple[str, List[str]]:
    """Extract normalized base type name and top-level type argument base strings."""
    t_raw = str(type_name or "").strip().lower()
    if not t_raw:
        return "", []
    if type_term_cls is not None:
        try:
            tt = type_term_cls.from_string(t_raw)
            t_base = str(getattr(tt, "base", "") or "").strip().lower()
            t_args = list(getattr(tt, "args", None) or ())
            arg_bases = [str(getattr(a, "base", "") or "").strip().lower() for a in t_args]
            return t_base, arg_bases
        except Exception:
            pass
    if "[" in t_raw and t_raw.endswith("]"):
        base_part, _, inner = t_raw[:-1].partition("[")
        parts: List[str] = []
        depth = 0
        cur: List[str] = []
        for ch in inner:
            if ch in ("[", "("):
                depth += 1
                cur.append(ch)
            elif ch in ("]", ")"):
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "," and depth == 0:
                seg = "".join(cur).strip()
                if seg:
                    parts.append(seg)
                cur.clear()
            else:
                cur.append(ch)
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
        return base_part.strip(), parts
    return t_raw, []


def is_sequence_or_tuple_type(type_name: str, registry: Optional[Any] = None) -> bool:
    """Return True if `type_name` denotes a non-string sequence/tuple/list/collection type."""
    t_l = str(type_name or "").strip().lower()
    if not t_l:
        return False
    if t_l in ("tuple", "list", "set", "sequence", "collection") or t_l.startswith(
        ("tuple[", "list[", "set[", "sequence[", "collection[")
    ):
        return True
    reg = _get_registry(registry)
    if reg is not None and not reg.is_subtype(t_l, "str"):
        if reg.is_subtype(t_l, "sequence") or reg.is_subtype(t_l, "collection"):
            return True
    return False


def parse_numeric_sequence_spec(
    port_or_sig: Any,
    registry: Optional[Any] = None,
    type_term_cls: Optional[Any] = None,
) -> Optional[NumericSeqSpec]:
    """Return a NumericSeqSpec when `port_or_sig` expects a numeric tuple or list."""
    if port_or_sig is None:
        return None
    reg = _get_registry(registry)
    if reg is None:
        return None
    inner = getattr(port_or_sig, "signature", port_or_sig)
    t_name = str(getattr(inner, "type_name", None) or getattr(port_or_sig, "type_name", None) or "").strip().lower()
    if not t_name:
        return None

    t_base, arg_bases = _parse_type_base_and_args(t_name, type_term_cls=type_term_cls)
    is_tuple_type = t_base == "tuple"
    has_ellipsis = "..." in arg_bases
    non_ellipsis = [ab for ab in arg_bases if ab and ab != "..."]

    if t_base in ("tuple", "list") and non_ellipsis and all(
        reg.is_subtype(ab, "numeric") and not reg.is_subtype(ab, "bool")
        for ab in non_ellipsis
    ):
        elem_is_int = all(
            reg.is_subtype(ab, "int") and not reg.is_subtype("float", ab)
            for ab in non_ellipsis
        )
        arity: Optional[int] = len(arg_bases) if (is_tuple_type and not has_ellipsis) else None
        if arity is None and is_tuple_type:
            sc_d = getattr(port_or_sig, "shape_contract", None)
            if isinstance(sc_d, dict):
                if isinstance(sc_d.get("len"), int) and sc_d["len"] >= 2:
                    arity = sc_d["len"]
                elif (
                    isinstance(sc_d.get("shape"), (list, tuple))
                    and len(sc_d["shape"]) == 1
                    and isinstance(sc_d["shape"][0], int)
                    and sc_d["shape"][0] >= 2
                ):
                    arity = sc_d["shape"][0]
        return NumericSeqSpec(
            base=t_base,
            is_tuple=is_tuple_type,
            elem_is_int=elem_is_int,
            arity=arity,
        )

    if is_tuple_type and not arg_bases:
        sc_d = getattr(port_or_sig, "shape_contract", None)
        arity_sc: Optional[int] = None
        if isinstance(sc_d, dict):
            if isinstance(sc_d.get("len"), int) and sc_d["len"] >= 2:
                arity_sc = sc_d["len"]
            elif (
                isinstance(sc_d.get("shape"), (list, tuple))
                and len(sc_d["shape"]) == 1
                and isinstance(sc_d["shape"][0], int)
                and sc_d["shape"][0] >= 2
            ):
                arity_sc = sc_d["shape"][0]
        if arity_sc is not None:
            return NumericSeqSpec(
                base=t_base,
                is_tuple=True,
                elem_is_int=False,
                arity=arity_sc,
            )
    return None


def is_true_scalar_port(port_or_sig: Any, registry: Optional[Any] = None) -> bool:
    """Return True only for atomic scalars; non-string sequence/tuple types never count as scalar."""
    if port_or_sig is None:
        return False
    reg = _get_registry(registry)
    inner = getattr(port_or_sig, "signature", port_or_sig)
    t_name = str(getattr(inner, "type_name", None) or getattr(port_or_sig, "type_name", None) or "").strip().lower()
    if is_sequence_or_tuple_type(t_name, reg):
        return False
    c_rank = getattr(port_or_sig, "canonical_rank", None)
    if c_rank is None:
        c_rank = getattr(port_or_sig, "rank", None)
    str_kind = str(getattr(port_or_sig, "structure_kind", "") or "").strip().lower()
    if str_kind == "handle":
        return False
    return bool(
        c_rank == 0
        or str_kind == "scalar"
        or (reg is not None and (reg.is_scalar_primitive(t_name) or reg.is_subtype(t_name, "scalar")))
    )


def _is_object_receiver_role(role: str, reg: Any) -> bool:
    if not role or reg is None:
        return False
    roles: Set[str] = set()
    for cat in ("receiver", "estimator", "owner_sink", "instance_sink"):
        getter = getattr(reg, "get_roles_for_category", None)
        if callable(getter):
            try:
                roles.update(str(r).strip().lower() for r in getter(cat))
            except Exception:
                continue
    return role in roles


def is_linear_handle_port(port_or_sig: Any, registry: Optional[Any] = None) -> bool:
    """Canonical category-theoretic linear handle predicate across planner, unifier, and linter."""
    if port_or_sig is None:
        return False
    inner = getattr(port_or_sig, "signature", None)
    sk = str(getattr(port_or_sig, "structure_kind", "") or getattr(inner, "structure_kind", "") or "").strip().lower()
    at = str(getattr(port_or_sig, "abstract_type", "") or getattr(inner, "abstract_type", "") or "").strip().lower()
    if sk == "handle" or at == "handle":
        return True
    pr = str(getattr(port_or_sig, "port_role", "") or getattr(port_or_sig, "derived_role", "") or "").strip().lower()
    tn = str(getattr(port_or_sig, "type_name", None) or getattr(inner, "type_name", "") or "").strip().lower()
    reg = _get_registry(registry)
    if reg is None:
        return False
    if _is_object_receiver_role(pr, reg):
        return True
    return bool(
        reg.is_subtype(pr, "handle")
        or reg.is_subtype(tn, "handle")
        or reg.is_subtype(pr, "model")
        or reg.is_subtype(tn, "model")
    )


def is_literal_parameter_port(
    cell: Any,
    port_name: str,
    port_sig: Any,
    registry: Optional[Any] = None,
    is_path_port_fn: Optional[Callable[..., bool]] = None,
    is_macro_block_slot_fn: Optional[Callable[..., bool]] = None,
    type_term_cls: Optional[Any] = None,
) -> bool:
    """Return True when `port_sig` accepts a prompt literal / configuration estimate (not a runtime carrier)."""
    if port_sig is None:
        return False
    if is_macro_block_slot_fn is not None and is_macro_block_slot_fn(cell, port_name):
        return False
    reg = _get_registry(registry)
    if reg is None:
        return False
    p_role = str(getattr(port_sig, "derived_role", "") or getattr(port_sig, "port_role", "") or "").strip().lower()
    inner = getattr(port_sig, "signature", port_sig)
    t_name = str(getattr(inner, "type_name", None) or getattr(port_sig, "type_name", "") or "").strip().lower()
    if is_linear_handle_port(port_sig, reg):
        return False
    if p_role in _NON_LIT_STRUCTURAL_ROLES or reg.is_abstract_carrier(t_name):
        return False

    path_fn = is_path_port_fn or _lattice_is_path_port
    if path_fn(port_sig, reg):
        return True
    key_proj_roles = (
        reg.get_roles_for_category("projection")
        | set(reg.get_projection_tokens() or ())
        | reg.get_roles_for_category("key")
    )
    key_binds = reg.get_roles_for_category("key")
    if p_role in key_proj_roles or str(getattr(port_sig, "binds", "") or "").strip().lower() in key_binds:
        return True
    if getattr(port_sig, "enum_values", None):
        return True
    if parse_numeric_sequence_spec(port_sig, reg, type_term_cls=type_term_cls) is not None:
        return True
    if reg.is_subtype(t_name, "str") or reg.is_subtype(t_name, "numeric") or reg.is_subtype(t_name, "bool"):
        return True
    return False


def validate_slot_value(
    val: Any,
    sig: Any,
    registry: Optional[Any] = None,
    type_term_cls: Optional[Any] = None,
) -> bool:
    """Validates whether an extracted or estimated slot/hyperparameter value conforms to port schema."""
    if val is None or sig is None:
        return False
    val_s = str(val).strip()
    if not val_s:
        return False
    val_low = val_s.lower()
    reg = _get_registry(registry)
    if reg is not None and reg.is_void_type(val_low):
        return False

    inner = getattr(sig, "signature", sig)
    t_name = str(getattr(inner, "type_name", None) or getattr(sig, "type_name", "") or "").strip().lower()
    if is_true_scalar_port(sig, reg) and isinstance(val, (list, tuple)):
        return False

    declared_enums = getattr(sig, "enum_values", None)
    enum_set = {str(e).strip().lower() for e in declared_enums} if declared_enums else set()

    abs_sig = str(getattr(sig, "abstract_type", "") or "").strip().lower()
    if (val_low == t_name or (abs_sig and val_low == abs_sig)) and val_low not in enum_set:
        return False
    if val_low in _BUILTIN_TYPE_NAMES and val_low not in enum_set:
        return False

    if reg is not None:
        try:
            if reg.is_known_type_name(val_low) and val_low not in enum_set:
                return False
        except Exception as _exc:
            logger.debug("[TYPE_PREDICATES] Suppressed is_known_type_name check: %s", _exc)

    if declared_enums:
        return val_low in enum_set

    elem_type = str(getattr(sig, "element_type", "") or "").strip().lower()
    str_kind = str(getattr(sig, "structure_kind", "") or "").strip().lower()
    sig_struct_kind = str(getattr(inner, "structure_kind", "") or "").strip().lower()
    is_numeric_elem = bool(reg is not None and elem_type and (reg.is_subtype(elem_type, "number") or reg.is_subtype(elem_type, "numeric")))
    tensor_kinds = (reg.get_types_by_kind("array") | reg.get_types_by_kind("tensor")) if reg is not None else set()
    is_tensor_or_matrix = (
        str_kind in tensor_kinds
        or sig_struct_kind in tensor_kinds
        or (
            reg is not None
            and any(
                reg.is_subtype(tn, root)
                for tn in (t_name, abs_sig)
                if tn
                for root in ("tensor", "matrix")
            )
        )
    )
    if reg is not None and not reg.is_declared_top(t_name) and not reg.is_subtype(t_name, "str") and not reg.is_subtype(t_name, "path"):
        if is_numeric_elem or is_tensor_or_matrix:
            def _all_numeric_leaves(x: Any) -> bool:
                if isinstance(x, (list, tuple)):
                    return bool(x) and all(_all_numeric_leaves(el) for el in x)
                if isinstance(x, bool):
                    return False
                if isinstance(x, (int, float)):
                    return True
                try:
                    float(str(x).strip())
                    return True
                except (ValueError, TypeError):
                    return False

            if not _all_numeric_leaves(val):
                return False

    if reg is not None:
        try:
            if reg.is_subtype(t_name, "bool"):
                return isinstance(val, bool) or val_low in ("true", "false")
            if reg.is_subtype(t_name, "int"):
                if isinstance(val, bool):
                    return False
                if isinstance(val, int):
                    return True
                if isinstance(val, float):
                    return (val == val) and abs(val) != float("inf") and val.is_integer()
                try:
                    int(val_s)
                    return True
                except ValueError:
                    return False
            if reg.is_subtype(t_name, "numeric"):
                if isinstance(val, bool):
                    return False
                try:
                    f_val = float(val_s)
                    return (f_val == f_val) and abs(f_val) != float("inf")
                except ValueError:
                    return False
        except Exception as _exc:
            logger.debug("[TYPE_PREDICATES] Suppressed numeric check: %s", _exc)

        t_base, arg_bases = _parse_type_base_and_args(t_name, type_term_cls=type_term_cls)
        has_ellipsis = "..." in arg_bases
        non_ellipsis = [ab for ab in arg_bases if ab and ab != "..."]
        if (
            t_base in ("tuple", "list")
            and non_ellipsis
            and all(reg.is_subtype(ab, "numeric") and not reg.is_subtype(ab, "bool") for ab in non_ellipsis)
        ):
            seq_val = val
            if isinstance(seq_val, str):
                try:
                    seq_val = ast.literal_eval(seq_val.strip())
                except Exception:
                    return False
            if not isinstance(seq_val, (list, tuple)) or len(seq_val) == 0:
                return False
            if t_base == "tuple" and not has_ellipsis and len(seq_val) != len(arg_bases):
                return False
            for i_el, el in enumerate(seq_val):
                if isinstance(el, bool):
                    return False
                if isinstance(el, (int, float)):
                    el_f = float(el)
                else:
                    try:
                        el_f = float(str(el).strip())
                    except (ValueError, TypeError):
                        return False
                if not (el_f == el_f) or abs(el_f) == float("inf"):
                    return False
                arg_b = (
                    arg_bases[i_el]
                    if (t_base == "tuple" and not has_ellipsis and i_el < len(arg_bases))
                    else non_ellipsis[0]
                )
                if reg.is_subtype(arg_b, "int") and not reg.is_subtype("float", arg_b):
                    if not el_f.is_integer():
                        return False

    return True


def format_literal_value(
    val: Any,
    sig: Any = None,
    registry: Optional[Any] = None,
    language_backend: Optional[Any] = None,
    parse_expr_ast_fn: Optional[Callable[[str], Optional[ast.AST]]] = None,
    type_term_cls: Optional[Any] = None,
) -> str:
    """Formats an extracted or estimated slot/hyperparameter value into an executable code literal."""
    be = language_backend or get_active_language_backend()
    if val is None:
        return be.null_literal
    reg = _get_registry(registry)
    if sig is not None and is_true_scalar_port(sig, reg) and isinstance(val, (list, tuple)):
        if getattr(sig, "default_value", None) is not None:
            return str(sig.default_value)
        return be.null_literal
    if isinstance(val, bool):
        return be.format_bool(val)
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list) and sig is not None:
        t_name_f = str(getattr(sig, "type_name", "") or "").strip().lower()
        if t_name_f == "tuple" or t_name_f.startswith("tuple["):
            return repr(tuple(val))
    if isinstance(val, (list, tuple, dict)):
        return repr(val)
    s = str(val).strip()
    if be.is_null_or_bool_literal(s, case_insensitive=False):
        return s

    t_name = str(getattr(sig, "type_name", "")).lower() if sig else ""
    is_str_like = False
    if reg is not None:
        try:
            str_types = reg.get_types_by_kind("string") | reg.get_types_by_kind("path")
            is_str_like = (
                t_name in str_types
                or reg.has_type_kind(t_name, "string")
                or reg.is_subtype(t_name, "string")
            )
        except Exception:
            is_str_like = False

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
        if s.lower() not in enum_set:
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

    t_base, _ = _parse_type_base_and_args(t_name, type_term_cls=type_term_cls)
    is_callable_type = t_base == "callable"
    if (reg is not None and reg.is_hof_role(p_role)) or is_callable_type:
        unq = s[1:-1].strip() if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s
        node: Optional[ast.AST] = None
        if parse_expr_ast_fn is not None:
            node = parse_expr_ast_fn(unq)
        else:
            try:
                node = ast.parse(unq, mode="eval").body
            except Exception:
                node = None
        if isinstance(node, (ast.Lambda, ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
            return unq
        return s

    lit_roles = reg.get_roles_for_category("literal") if reg is not None else set()
    is_lit_role = (p_role in lit_roles) or (reg is not None and reg.is_config_role(p_role))
    if not is_str_like and not is_lit_role and s.isidentifier():
        return s
    if is_str_like or not s.isidentifier() or is_lit_role:
        return repr(s.strip("'\"").strip())
    return s


__all__ = [
    "NumericSeqSpec",
    "format_literal_value",
    "is_linear_handle_port",
    "is_literal_parameter_port",
    "is_sequence_or_tuple_type",
    "is_true_scalar_port",
    "parse_numeric_sequence_spec",
    "validate_slot_value",
]
