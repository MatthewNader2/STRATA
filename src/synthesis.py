"""
src/synthesis.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Dynamic MicroCell Synthesizer & Deterministic Structural Cell Renderer.
"""

from __future__ import annotations
import ast
import keyword
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from log_config import get_logger

try:
    from .language_backend import get_active_language_backend
    from .lattice import TypeRegistry
    from .utils import (
        safe_substitute_template,
        extract_template_placeholders,
        unique_var as _unique_var,
    )
except (ImportError, ValueError):
    from language_backend import get_active_language_backend
    from lattice import TypeRegistry
    from utils import (
        safe_substitute_template,
        extract_template_placeholders,
        unique_var as _unique_var,
    )

logger = get_logger("synthesis")


# --------------------------------------------------------------------------- #
# Loop rendering helpers (all data-driven, zero hard-coded variable names)
# --------------------------------------------------------------------------- #

def _derive_loop_item_type(cell: Any, accumulated_sigma: Optional[Any]) -> Optional[Any]:
    """
    Derives the declared iteration carrier (item type) of a traced-loop macro.
    """
    try:
        from .unification import TypeTerm, GenericTypeTerm, substitute_generics
    except (ImportError, ValueError):
        from unification import TypeTerm, GenericTypeTerm, substitute_generics

    for p_sig in getattr(cell, "inputs", {}).values():
        raw_t = str(
            getattr(getattr(p_sig, "signature", None), "type_name", "")
            or getattr(p_sig, "type_name", "")
        )
        if not raw_t or "[" not in raw_t:
            continue
        concrete_t = substitute_generics(raw_t, accumulated_sigma) if accumulated_sigma is not None else raw_t
        try:
            term = TypeTerm.from_string(concrete_t)
        except Exception as e:
            logger.debug("[SYNTHESIS] TypeTerm parse skipped for '%s': %s", concrete_t, e)
            continue
        if isinstance(term, GenericTypeTerm) and term.args:
            return term.args[0]
    return None


def _declared_loop_kind(cell: Any) -> Optional[str]:
    """
    Reads the declared loop kind from cell metadata (never from template strings).
    Returns 'collect', 'extremum', 'fold', or None.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "loop_kind", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("loop_kind")

    try:
        from lattice import TypeRegistry
    except (ImportError, ValueError):
        from .lattice import TypeRegistry
    reg = TypeRegistry.get_instance()

    fold_kinds = reg.get_roles_for_category("fold")
    collect_kinds = reg.get_roles_for_category("collect")
    extremum_kinds = reg.get_roles_for_category("extremum")

    if declared:
        d_lower = str(declared).lower()
        if d_lower in fold_kinds:
            return "fold"
        if d_lower in collect_kinds:
            return "collect"
        if d_lower in extremum_kinds:
            return "extremum"

    topology = str(getattr(cell, "topology_type", "") or "").lower()
    if topology in collect_kinds:
        return "collect"
    if topology in extremum_kinds:
        return "extremum"
    if topology in fold_kinds:
        return "fold"
    return None


def _sanitize_template_for_ast(template: str) -> str:
    """
    Sanitizes template placeholders `{ph}` to valid Python identifiers `_nstl_ph_{ph}`
    so the template can be safely parsed by `ast.parse`.
    """
    if not template or not template.strip():
        return ""
    phs = extract_template_placeholders(template)
    sanitized = template
    for ph in phs:
        sanitized = sanitized.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
    return sanitized


def _unsanitize_ast_source(expr: str) -> str:
    """Restores `_nstl_ph_<name>` tokens back to `{<name>}` placeholders."""
    phs = []
    idx = 0
    marker = "_nstl_ph_"
    while True:
        pos = expr.find(marker, idx)
        if pos == -1:
            phs.append(expr[idx:])
            break
        phs.append(expr[idx:pos])
        end = pos + len(marker)
        while end < len(expr) and (expr[end].isalnum() or expr[end] == "_"):
            end += 1
        name = expr[pos + len(marker):end]
        phs.append(f"{{{name}}}" if name else marker)
        idx = end
    return "".join(phs)


def _template_assigns_var(rendered: str, var_name: str) -> bool:
    """AST/backend check: returns True iff `var_name` is assigned as a target in `rendered`."""
    if not rendered or not var_name:
        return False
    try:
        try:
            from .language_backend import get_active_language_backend
        except (ImportError, ValueError):
            from language_backend import get_active_language_backend
        if var_name in get_active_language_backend().extract_assignment_targets(rendered):
            return True
    except Exception:
        pass
    try:
        parsed = ast.parse(_sanitize_template_for_ast(rendered))
    except SyntaxError:
        return False
    for stmt in parsed.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == var_name and stmt.value is not None:
                return True
    return False


def _format_default_value(p_sig: Any, reg: Optional[Any] = None) -> str:
    """Formats a port's declared `default_value` using TypeRegistry subtype checks."""
    dv = getattr(p_sig, "default_value", None)
    if reg is None and "TypeRegistry" in globals():
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    p_tn = str(
        getattr(getattr(p_sig, "signature", None), "type_name", "")
        or getattr(p_sig, "type_name", "")
        or ""
    ).strip().lower()
    is_str_type = (
        reg.is_subtype(p_tn, "str")
        if (reg is not None and p_tn)
        else (p_tn == str.__name__)
    )
    if isinstance(dv, str) and is_str_type:
        try:
            evaled = ast.literal_eval(dv.strip())
            if isinstance(evaled, str):
                return dv.strip()
        except Exception:
            pass
        return repr(dv)
    return str(dv)


def _template_initializes_container(template: str, primary_out_name: Optional[str] = None) -> Optional[str]:
    """
    AST inspection: detect whether the template initializes the output
    container variable (`_nstl_ph_output_var` or `_nstl_ph_<primary_out_name>`)
    to an empty container and return its container kind.
    """
    if not template or not template.strip():
        return None
    try:
        try:
            from .language_backend import get_active_language_backend
        except (ImportError, ValueError):
            from language_backend import get_active_language_backend
        if not get_active_language_backend().is_python_ast:
            return None
    except Exception:
        pass
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    target_ids = {"_nstl_ph_output_var"}
    if primary_out_name:
        target_ids.add(f"_nstl_ph_{primary_out_name}")

    try:
        from .language_backend import get_active_language_backend
    except (ImportError, ValueError):
        from language_backend import get_active_language_backend
    constructors = get_active_language_backend().container_constructors
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_out = any(
                isinstance(t, ast.Name) and t.id in target_ids
                for t in targets
            )
            if not is_out:
                continue
            val = node.value
            if isinstance(val, ast.List):
                return "list"
            if isinstance(val, ast.Dict):
                return "dict"
            if isinstance(val, ast.Set):
                return "set"
            if isinstance(val, ast.Tuple):
                return "tuple"
            if isinstance(val, ast.Call):
                func_name = None
                if isinstance(val.func, ast.Name):
                    func_name = val.func.id
                elif isinstance(val.func, ast.Attribute):
                    func_name = val.func.attr
                if func_name and func_name in constructors:
                    return constructors[func_name]
    return None


def _detect_loop_item_var(template: str) -> Optional[str]:
    """
    AST inspection: scans the sanitized template AST for an `ast.For` or `ast.AsyncFor`
    node and extracts the loop variable name from node.target.
    """
    if not template or not template.strip():
        return None
    try:
        try:
            from .language_backend import get_active_language_backend
        except (ImportError, ValueError):
            from language_backend import get_active_language_backend
        be_loop = get_active_language_backend()
        if not be_loop.is_python_ast:
            return be_loop.extract_loop_item_var(template)
    except Exception:
        pass
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    for node in ast.walk(parsed):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            target = node.target
            target_name = None
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, (ast.Tuple, ast.List)) and target.elts:
                first = target.elts[0]
                if isinstance(first, ast.Name):
                    target_name = first.id
            if target_name:
                if target_name.startswith("_nstl_ph_"):
                    return target_name[len("_nstl_ph_"):]
                return target_name
    return None


def _resolve_item_var(cell: Any, template: str) -> str:
    """
    Resolves the loop item variable name from declared metadata, or detected
    from the template's `for ... in ...` header, or a cell-scoped fallback.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "item_var", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("item_var")
    if declared:
        return str(declared)
    detected = _detect_loop_item_var(template)
    if detected:
        return detected
    return _unique_var(cell, "item")


def _resolve_direction_is_max(cell: Any, bindings: Dict[str, Any]) -> bool:
    """
    Resolves the declared extremum direction from a bound port value. Only
    reads declared bindings (never prompt text). Returns True for max/argmax.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    direction_port = (
        getattr(cell, "direction_port", None)
        or (meta.get("direction_port") if isinstance(meta, dict) else None)
    )
    raw = bindings.get(direction_port) if direction_port else None
    if raw is None:
        try:
            try:
                from .lattice import TypeRegistry
            except (ImportError, ValueError):
                from lattice import TypeRegistry
            reg_ord = TypeRegistry.get_instance()
            order_ports = reg_ord.get_roles_for_category("direction") | reg_ord.get_roles_for_category("order")
        except Exception:
            order_ports = set()
        for k in tuple(order_ports):
            if k in bindings and bindings[k] is not None:
                raw = bindings[k]
                break
    if raw is None:
        return True
    try:
        from .language_backend import get_active_language_backend
    except (ImportError, ValueError):
        from language_backend import get_active_language_backend
    be_dir = get_active_language_backend()
    ok_lit, parsed = be_dir.parse_literal(str(raw).strip())
    if ok_lit and isinstance(parsed, (bool, int, float)):
        return bool(parsed)
    s = str(raw).strip().strip("'\"").lower()
    min_vals: Set[str] = set()
    try:
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            from lattice import TypeRegistry
        min_vals |= TypeRegistry.get_instance().get_roles_for_category("min_direction")
    except Exception:
        pass
    for v in (getattr(cell, "min_direction_values", None) or (meta.get("min_direction_values") if isinstance(meta, dict) else None) or ()):
        min_vals.add(str(v).strip().lower())
    return s not in min_vals


def render_cell(
    cell: Any,
    bindings: Dict[str, Any],
    indent_level: int = 0,
    context: Optional[Any] = None,
    accumulated_sigma: Optional[Any] = None,
) -> str:
    """
    Renders an executable block of Python code for a cell at the specified indentation level.
    Recursively renders child sub-pipelines in bound_slots at indent_level + 1.
    """
    try:
        from .unification import TypeTerm, GenericTypeTerm, UnificationGate, unify, Substitution
    except (ImportError, ValueError):
        from unification import TypeTerm, GenericTypeTerm, UnificationGate, unify, Substitution
    try:
        from .coalgebraic_unfolder import is_macro_combinator_cell
    except (ImportError, ValueError):
        from coalgebraic_unfolder import is_macro_combinator_cell

    reg_syn = TypeRegistry.get_instance() if "TypeRegistry" in globals() else None
    indent_prefix = "    " * indent_level
    template = cell.code_template.strip()
    if not template:
        return ""

    slots = getattr(cell, "slots", {}) or {}
    bound_slots = getattr(cell, "bound_slots", {}) or {}
    prim_out_obj = getattr(cell, "primary_output", None)
    prim_out_name_outer = getattr(prim_out_obj, "name", None) if prim_out_obj else None

    out_var = bindings.get(
        "output_var",
        f"_{getattr(prim_out_obj, 'state', None) or 'out'}",
    )

    loop_kind = _declared_loop_kind(cell)
    init_container = _template_initializes_container(template, primary_out_name=prim_out_name_outer)
    if loop_kind == "extremum":
        is_extremum_loop, is_collect_loop, is_fold_loop = True, False, False
    elif loop_kind == "collect":
        is_extremum_loop, is_collect_loop, is_fold_loop = False, True, False
    elif loop_kind == "fold":
        is_extremum_loop, is_collect_loop, is_fold_loop = False, False, True
    else:
        has_direction_port = "direction" in getattr(cell, "inputs", {})
        direction_bound = bindings.get("direction") is not None
        is_extremum_loop = has_direction_port and direction_bound
        is_collect_loop = (
            not is_extremum_loop and init_container is not None
        )
        is_fold_loop = False

    is_max = _resolve_direction_is_max(cell, bindings) if is_extremum_loop else False

    item_var = _resolve_item_var(cell, template)
    metric_var = _unique_var(cell, "max_metric" if is_max else "min_metric")

    item_type = (
        _derive_loop_item_type(cell, accumulated_sigma)
        if (is_collect_loop or is_extremum_loop or is_fold_loop)
        else None
    )

    slot_rendered: Dict[str, str] = {}
    slot_last_out: Dict[str, str] = {}
    cell_inputs = getattr(cell, "inputs", {}) or {}
    slots_items = slots.items() if isinstance(slots, dict) else [(s, {}) for s in slots]
    macro_slot_items = [
        (sn, sp)
        for sn, sp in slots_items
        if (str(sn) in bound_slots) or (str(sn) not in cell_inputs)
    ]
    for slot_name, slot_spec in macro_slot_items:
        child_nodes = bound_slots.get(slot_name, [])
        if not child_nodes:
            # Every macro slot must be grounded in verified lattice / control-flow cells;
            # raw unconstrained LLM code generation is never invoked here.
            slot_rendered[slot_name] = get_active_language_backend(reg_syn).noop_statement
            continue

        if isinstance(child_nodes, str):
            slot_rendered[slot_name] = child_nodes
            continue

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        child_lines: List[str] = []
        last_metric: Optional[str] = None
        enclosing_known_vars: Set[str] = set()
        # Track active typed outputs produced within this slot (and from parent context)
        # so multi-step slot sub-pipelines wire type-compatibly without hardcodes.
        slot_typed_vars: List[Tuple[Any, str]] = []
        if context is not None:
            for ctx_attr in ("variables", "var_types"):
                ctx_map = getattr(context, ctx_attr, None)
                if isinstance(ctx_map, dict):
                    if ctx_attr == "variables":
                        enclosing_known_vars.update(ctx_map.keys())
                    for v_name, v_entry in ctx_map.items():
                        v_port = v_entry[0] if isinstance(v_entry, tuple) and v_entry else v_entry
                        v_sig = getattr(v_port, "signature", v_port)
                        if v_sig is not None:
                            slot_typed_vars.append((v_sig, v_name))
            if hasattr(context, "active_variables") and isinstance(context.active_variables, (set, list)):
                enclosing_known_vars.update(context.active_variables)
        for p_in_name, p_in_sig in cell_inputs.items():
            if p_in_name in bindings and isinstance(bindings[p_in_name], str):
                b_clean = bindings[p_in_name].strip()
                if b_clean.isidentifier() and not keyword.iskeyword(b_clean):
                    in_sig = getattr(p_in_sig, "signature", p_in_sig)
                    slot_typed_vars.append((in_sig, b_clean))
        for v in bindings.values():
            if isinstance(v, str):
                v_clean = v.strip()
                if v_clean.isidentifier() and not keyword.iskeyword(v_clean):
                    enclosing_known_vars.add(v_clean)
        if item_var and item_var.isidentifier():
            enclosing_known_vars.add(item_var)
        if out_var and out_var.isidentifier():
            enclosing_known_vars.add(out_var)

        for child_idx, child in enumerate(child_nodes):
            child_bindings: Dict[str, Any] = {}
            for p_name, p_sig in (getattr(child, "inputs", {}) or {}).items():
                bound = False
                if item_type is not None:
                    child_sig = getattr(p_sig, "signature", None)
                    if child_sig is not None:
                        try:
                            if unify(item_type, child_sig, Substitution()) is not None:
                                child_bindings[p_name] = item_var
                                bound = True
                        except Exception as e:
                            logger.debug("[SYNTHESIS] Loop item unify check skipped: %s", e)
                            bound = False
                if bound:
                    continue
                if hasattr(child, "typed_bindings") and child.typed_bindings and p_name in child.typed_bindings:
                    rec = child.typed_bindings[p_name]
                    if rec and rec.value_expr:
                        child_bindings[p_name] = rec.value_expr
                        continue
                # Match compatible upstream output variable inside slot by type/carrier
                p_role = str(getattr(p_sig, "port_role", "") or "").lower()
                child_in_sig = getattr(p_sig, "signature", None)
                matched_slot_var = None
                literal_roles = (
                    (reg_syn.get_roles_for_category("literal") | reg_syn.get_roles_for_category("config"))
                    if reg_syn is not None else set()
                )
                is_literal_port = (
                    p_role in literal_roles
                    or (reg_syn.is_config_role(p_role) if reg_syn is not None and p_role else False)
                )
                if not is_literal_port and child_in_sig is not None:
                    in_t = str(getattr(child_in_sig, "type_name", "") or "").lower()
                    is_prim_or_top = (
                        not in_t
                        or (
                            (reg_syn.is_declared_top(in_t) or reg_syn.is_scalar_primitive(in_t))
                            if reg_syn is not None else False
                        )
                    )
                    if not is_prim_or_top:
                        for prev_sig, prev_var in reversed(slot_typed_vars):
                            try:
                                if unify(prev_sig, child_in_sig, Substitution()) is not None:
                                    matched_slot_var = prev_var
                                    break
                            except Exception:
                                continue
                enclosing_var_types: Dict[str, Any] = {}
                for _ev_sig, _ev_name in slot_typed_vars:
                    if _ev_name and _ev_name not in enclosing_var_types:
                        enclosing_var_types[_ev_name] = _ev_sig
                if item_var and item_type is not None:
                    enclosing_var_types.setdefault(item_var, item_type)

                def _enclosing_name_compatible(cand_key: str, cand_val: Any) -> bool:
                    if child_in_sig is None:
                        return True
                    c_tn = str(getattr(child_in_sig, "type_name", "") or "").strip().lower()
                    if not c_tn or (reg_syn is not None and (reg_syn.is_declared_top(c_tn) or reg_syn.is_type_variable(c_tn))):
                        return True
                    val_s = str(cand_val).strip() if isinstance(cand_val, str) else ""
                    enc_sig = enclosing_var_types.get(val_s) or enclosing_var_types.get(cand_key)
                    if enc_sig is None and cand_key in cell_inputs:
                        enc_sig = getattr(cell_inputs[cand_key], "signature", cell_inputs[cand_key])
                    if enc_sig is None:
                        return True
                    try:
                        return unify(enc_sig, child_in_sig, Substitution()) is not None
                    except Exception:
                        return True

                if matched_slot_var is not None:
                    child_bindings[p_name] = matched_slot_var
                elif p_name in bindings and _enclosing_name_compatible(p_name, bindings[p_name]):
                    child_bindings[p_name] = bindings[p_name]
                elif p_name in enclosing_known_vars and _enclosing_name_compatible(p_name, p_name):
                    child_bindings[p_name] = p_name
                else:
                    try:
                        try:
                            from .unification import (
                                _classify_template_identifier_port,
                                _normalize_identifier_port_binding,
                            )
                        except (ImportError, ValueError):
                            from unification import (
                                _classify_template_identifier_port,
                                _normalize_identifier_port_binding,
                            )
                        is_id_port, is_param_port = _classify_template_identifier_port(child, p_name)
                    except Exception:
                        is_id_port, is_param_port = False, False
                        _normalize_identifier_port_binding = None  # type: ignore[assignment]

                    def _maybe_norm_id(val_s: Any) -> Any:
                        if not is_id_port or _normalize_identifier_port_binding is None:
                            return val_s
                        norm_v = _normalize_identifier_port_binding(val_s, is_param_list=is_param_port)
                        return norm_v if norm_v is not None else ("x" if is_param_port else f"_{p_name}_{child_idx}")

                    lit_bound = None
                    if context is not None and hasattr(context, "resolve_literal_for_port"):
                        prev_cl = getattr(context, "current_cell_clause_idx", None)
                        ch_cl = getattr(child, "matched_clause_idx", None)
                        if ch_cl is not None:
                            context.current_cell_clause_idx = ch_cl
                        try:
                            lit_bound = context.resolve_literal_for_port(p_sig, cell=child)
                        except Exception:
                            lit_bound = None
                        finally:
                            context.current_cell_clause_idx = prev_cl
                    if lit_bound is not None:
                        child_bindings[p_name] = _maybe_norm_id(lit_bound)
                        continue
                    try:
                        try:
                            from .unification import resolve_synthetic_boundary_io
                        except (ImportError, ValueError):
                            from unification import resolve_synthetic_boundary_io
                        synth_io = resolve_synthetic_boundary_io(p_sig, cell=child, ctx=context)
                    except Exception:
                        synth_io = None
                    if synth_io is not None:
                        child_bindings[p_name] = _maybe_norm_id(synth_io)
                        continue
                    if getattr(p_sig, "default_value", None) is not None:
                        if is_id_port:
                            child_bindings[p_name] = _maybe_norm_id(getattr(p_sig, "default_value"))
                        else:
                            child_bindings[p_name] = _format_default_value(p_sig, reg=reg_syn)
                    else:
                        try:
                            try:
                                from .unification import _pointed_default
                            except (ImportError, ValueError):
                                from unification import _pointed_default
                            eta_v = _pointed_default(p_sig, child)
                            if eta_v is not None:
                                child_bindings[p_name] = _maybe_norm_id(eta_v)
                            elif is_id_port:
                                child_bindings[p_name] = "x" if is_param_port else f"_{p_name}_{child_idx}"
                        except Exception:
                            if is_id_port:
                                child_bindings[p_name] = "x" if is_param_port else f"_{p_name}_{child_idx}"

            child_out_var = f"{_unique_var(child, 'out')}_{child_idx}"
            child_bindings["output_var"] = child_out_var
            last_metric = child_out_var
            enclosing_known_vars.add(child_out_var)

            child_outputs = getattr(child, "outputs", {}) or {}
            prim_out = getattr(child, "primary_output", None)
            prim_out_name = getattr(prim_out, "name", None) if prim_out else None
            for out_p_name, out_p_sig in child_outputs.items():
                if len(child_outputs) == 1 or out_p_name == prim_out_name:
                    ov = child_out_var
                else:
                    ov = f"{_unique_var(child, out_p_name)}_{child_idx}"
                child_bindings[out_p_name] = ov
                enclosing_known_vars.add(ov)
                out_sig_obj = getattr(out_p_sig, "signature", out_p_sig)
                if out_sig_obj is not None:
                    slot_typed_vars.append((out_sig_obj, ov))

            if (
                bool(getattr(child, "bound_slots", None))
                or is_macro_combinator_cell(child)
            ):
                child_text = render_cell(
                    child,
                    child_bindings,
                    indent_level=0,
                    context=context,
                    accumulated_sigma=accumulated_sigma,
                )
            else:
                child_text = UnificationGate._instantiate_ast_template(
                    child.code_template,
                    child_bindings,
                    child.inputs,
                    known_vars=enclosing_known_vars,
                )
            if child_text:
                child_lines.append(child_text)

        if is_extremum_loop and last_metric is not None:
            op = ">" if is_max else "<"
            update_block = (
                f"if {metric_var} is None or {last_metric} {op} {metric_var}:\n"
                f"    {metric_var} = {last_metric}\n"
                f"    {out_var} = {item_var}"
            )
            child_lines.append(update_block)
        elif is_collect_loop and last_metric is not None:
            be_col = get_active_language_backend(reg_syn)
            tmpl_map = be_col.collection_append_templates
            container_type = init_container
            if container_type is None:
                out_sig = getattr(cell, "primary_output", None)
                t_name = str(getattr(out_sig, "type_name", "") or "").strip().lower()
                base_t = t_name
                try:
                    t_parsed = TypeTerm.from_string(t_name) if t_name else None
                    if isinstance(t_parsed, GenericTypeTerm):
                        base_t = str(t_parsed.base or t_name).strip().lower()
                except Exception:
                    base_t = t_name.split("[", 1)[0].strip() if "[" in t_name else t_name
                if reg_syn is not None and tmpl_map:
                    for cand_ct in tmpl_map:
                        if base_t == cand_ct or reg_syn.is_subtype(base_t, cand_ct):
                            container_type = cand_ct
                            break
                if container_type is None:
                    container_type = next(iter(tmpl_map), "list")

            append_stmt = be_col.format_collection_append(
                container_type, out_var, str(last_metric), key_expr=str(item_var)
            )
            if append_stmt:
                child_lines.append(append_stmt)
        elif is_fold_loop and last_metric is not None:
            meta = getattr(cell, "tree_meta", None) or {}
            reducer = getattr(cell, "reducer", None)
            if reducer is None and isinstance(meta, dict):
                reducer = meta.get("reducer") or meta.get("fold_op") or meta.get("accumulation_op")

            if reducer in ("+", "-", "*", "/", "%", "&", "|", "^"):
                child_lines.append(f"{out_var} = {out_var} {reducer} {last_metric}")
            elif reducer and str(reducer).isidentifier():
                child_lines.append(f"{out_var} = {reducer}({out_var}, {last_metric})")
            else:
                null_lit = get_active_language_backend(reg_syn).null_literal
                child_lines.append(f"{out_var} = {last_metric} if {out_var} is {null_lit} else ({out_var} + {last_metric})")

        be_loop_end = get_active_language_backend(reg_syn)
        slot_rendered[slot_name] = "\n".join(child_lines) if child_lines else be_loop_end.noop_statement
        if last_metric is not None:
            slot_last_out[slot_name] = last_metric

    rendered = template
    be_rend = get_active_language_backend(reg_syn)
    if is_extremum_loop:
        init_prefix: List[str] = []
        if not _template_assigns_var(rendered, metric_var):
            init_prefix.append(f"{metric_var} {be_rend.assignment_operator} {be_rend.null_literal}")
        if not _template_assigns_var(rendered, out_var):
            init_prefix.append(f"{out_var} {be_rend.assignment_operator} {be_rend.null_literal}")
        if init_prefix:
            rendered = "\n".join(init_prefix) + "\n" + rendered
    elif is_fold_loop and not _template_assigns_var(rendered, out_var):
        init_val = getattr(cell, "init_val", None)
        if init_val is None and isinstance(getattr(cell, "tree_meta", None), dict):
            init_val = cell.tree_meta.get("init_val") or cell.tree_meta.get("initial_value")
        init_expr = str(init_val) if init_val is not None else be_rend.null_literal
        rendered = f"{out_var} {be_rend.assignment_operator} {init_expr}\n" + rendered

    # Map macro slot names to non-predicate HOF ports when the template uses
    # functional call syntax (e.g. ({body_fn})(_state)) rather than {body}.
    action_op_ports: List[str] = []
    for p_name, p_sig in cell_inputs.items():
        p_role = str(getattr(p_sig, "port_role", "") or "").strip().lower()
        is_hof = (
            reg_syn.is_hof_role(p_role)
            if (reg_syn is not None and p_role)
            else False
        )
        if is_hof:
            t_str = str(getattr(getattr(p_sig, "signature", None), "type_name", "") or getattr(p_sig, "type_name", "") or "").strip()
            is_bool_predicate = False
            if t_str:
                try:
                    t_term = TypeTerm.from_string(t_str)
                    if isinstance(t_term, GenericTypeTerm) and t_term.args:
                        ret_term = t_term.args[-1]
                        ret_name = str(getattr(ret_term, "name", None) or getattr(ret_term, "base", None) or ret_term).strip().lower()
                        is_bool_predicate = (
                            reg_syn.is_subtype(ret_name, "bool")
                            if reg_syn is not None
                            else (ret_name == bool.__name__)
                        )
                except Exception:
                    is_bool_predicate = False
            if is_bool_predicate:
                continue
            action_op_ports.append(p_name)

    def _resolve_op_port_for_slot(s_name: str, s_idx: int) -> Optional[str]:
        s_low = s_name.lower()
        s_stem = s_low.split("_")[0]
        for op_p in action_op_ports:
            op_low = op_p.lower()
            if op_low == s_low or op_low.startswith(s_stem):
                return op_p
        if len(action_op_ports) == 1 and len(slot_rendered) == 1:
            return action_op_ports[0]
        if s_idx < len(action_op_ports):
            return action_op_ports[s_idx]
        return None

    for s_idx, (slot_name, slot_code) in enumerate(slot_rendered.items()):
        placeholder = f"{{{slot_name}}}"
        if any(ln.strip() == placeholder for ln in rendered.splitlines()):
            base_indent = "    "
            found_target = None
            for line in rendered.splitlines():
                stripped = line.lstrip(" \t")
                if stripped.startswith(placeholder):
                    base_indent = line[: len(line) - len(stripped)]
                    found_target = base_indent + placeholder
                    break

            indented_slot = "\n".join(
                (base_indent + line if line.strip() else line)
                for line in slot_code.splitlines()
            )
            if found_target and found_target in rendered:
                rendered = rendered.replace(found_target, indented_slot)
            else:
                rendered = rendered.replace(placeholder, indented_slot)
            continue

        # Fallback: template uses functional_operator call placeholder ({body_fn}) etc.
        if not bound_slots.get(slot_name):
            continue
        op_port = _resolve_op_port_for_slot(slot_name, s_idx)
        if not op_port:
            continue
        op_ph = f"{{{op_port}}}"
        if op_ph not in rendered:
            continue
        new_lines: List[str] = []
        replaced = False
        for line in rendered.splitlines():
            if not replaced and op_ph in line:
                stripped = line.lstrip(" \t")
                base_indent = line[: len(line) - len(stripped)] or "    "
                indented_slot = "\n".join(
                    (base_indent + sl if sl.strip() else sl)
                    for sl in slot_code.splitlines()
                )
                new_lines.append(indented_slot)
                slot_out_var = slot_last_out.get(slot_name)
                # Inspect the sanitized AST statement (when Python AST backend is active)
                # to preserve `return` or assignment LHS, else delegate to LanguageBackend.
                stmt_node = None
                be_syn = get_active_language_backend()
                if be_syn.is_python_ast:
                    try:
                        src_probe = _sanitize_template_for_ast(stripped)
                        if be_syn.block_suffix and stripped.endswith(be_syn.block_suffix):
                            src_probe = src_probe + f"\n    {be_syn.noop_statement}"
                        parsed_stmt = ast.parse(src_probe)
                        stmt_node = parsed_stmt.body[0] if parsed_stmt.body else None
                    except SyntaxError:
                        stmt_node = None
                if isinstance(stmt_node, ast.Return):
                    ret_expr = slot_out_var or bindings.get("output_var") or be_syn.null_literal
                    new_lines.append(f"{base_indent}{be_syn.return_keyword} {ret_expr}")
                elif isinstance(stmt_node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    raw_tgt = stmt_node.targets[0] if isinstance(stmt_node, ast.Assign) else stmt_node.target
                    lhs_target = _unsanitize_ast_source(ast.unparse(raw_tgt)).strip()
                    if lhs_target and slot_out_var and lhs_target != slot_out_var:
                        new_lines.append(f"{base_indent}{lhs_target} = {slot_out_var}")
                elif stmt_node is None:
                    lhs_targets = be_syn.extract_assignment_targets(stripped)
                    if lhs_targets and slot_out_var and lhs_targets[0] != slot_out_var:
                        term_s = be_syn.statement_terminator
                        op_s = be_syn.assignment_operator
                        decl_p = be_syn.declaration_prefixes[0] + " " if be_syn.declaration_prefixes else ""
                        new_lines.append(f"{base_indent}{decl_p}{lhs_targets[0]} {op_s} {slot_out_var}{term_s}")
                replaced = True
            else:
                new_lines.append(line)
        if replaced:
            rendered = "\n".join(new_lines)

    render_bindings = dict(bindings)
    for p_name, p_sig in cell_inputs.items():
        if p_name not in render_bindings or render_bindings[p_name] is None:
            if getattr(p_sig, "default_value", None) is not None:
                render_bindings[p_name] = _format_default_value(p_sig, reg=reg_syn)
    cell_outputs = getattr(cell, "outputs", {}) or {}
    prim_out = getattr(cell, "primary_output", None)
    prim_out_name = getattr(prim_out, "name", None) if prim_out else None
    for out_p_name in cell_outputs.keys():
        if out_p_name not in render_bindings or not render_bindings[out_p_name]:
            if len(cell_outputs) == 1 or out_p_name == prim_out_name:
                render_bindings[out_p_name] = out_var
            else:
                render_bindings[out_p_name] = _unique_var(cell, out_p_name)

    rendered = safe_substitute_template(rendered, render_bindings)

    if indent_level > 0:
        rendered = "\n".join(
            (indent_prefix + line if line.strip() else line)
            for line in rendered.splitlines()
        )

    return rendered


__all__ = [
    "render_cell",
]
