"""
src/port_binder.py — Language-agnostic algebraic port binder and emitter.

Binds data ports strictly from PlannedStep wiring and non-data ports strictly from
TypedIR parameters (with optional constrained slot-fill fallback), formatting code
and validating syntax purely through the active LanguageBackend.
"""
from __future__ import annotations

from dataclasses import dataclass
import keyword
import string
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    from .language_backend import LexicalLanguageBackend, get_active_language_backend
    from .schema import (
        align_node_params_to_ports,
        coerce_literal_for_port,
        derive_value_class,
        get_attr,
        is_egress_asset_port,
        is_value_compatible_with_primitive_carrier,
    )
    from .type_system import RefinementSet
    from .typed_ir import IRNode, TypedIR
    from .typed_planner import CandidatePlan, PlannedStep
except (ImportError, ValueError):
    from language_backend import LexicalLanguageBackend, get_active_language_backend
    from schema import (
        align_node_params_to_ports,
        coerce_literal_for_port,
        derive_value_class,
        get_attr,
        is_egress_asset_port,
        is_value_compatible_with_primitive_carrier,
    )
    from type_system import RefinementSet
    from typed_ir import IRNode, TypedIR
    from typed_planner import CandidatePlan, PlannedStep


@dataclass(frozen=True)
class BoundStep:
    cell_id: str
    ir_node_id: Optional[str]
    is_live_bridge: bool
    bindings: Dict[str, str]
    code: str
    dependencies: Tuple[str, ...]


@dataclass(frozen=True)
class BindEmitResult:
    ok: bool
    steps: Tuple[BoundStep, ...] = ()
    code: str = ""
    refusal_kind: Optional[str] = None
    error: Optional[str] = None
    unbound_ports: Tuple[str, ...] = ()
    input_assets: Tuple[str, ...] = ()
    egress_paths: Tuple[str, ...] = ()


def _clean_ir_condition(raw_cond: Optional[str]) -> str:
    """Resolve `$ident` IR variable references to bare target identifiers without regex."""
    s = str(raw_cond or "").strip()
    if "$" not in s:
        return s
    out: List[str] = []
    in_quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "$" and i + 1 < n and (s[i + 1].isalpha() or s[i + 1] == "_"):
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _align_template_slots(
    tmpl: str,
    ins_keys: List[str],
    outs_keys: List[str],
    backend: LexicalLanguageBackend,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    fmt = string.Formatter()
    phs: List[str] = []
    seen: Set[str] = set()
    try:
        for _, field_name, _, _ in fmt.parse(tmpl):
            if field_name and field_name.isidentifier() and field_name not in seen:
                seen.add(field_name)
                phs.append(field_name)
    except (ValueError, IndexError):
        return {}, {}

    if not phs:
        return {}, {}

    lhs_phs = backend.extract_lhs_placeholders(tmpl, phs)
    lhs_set = set(lhs_phs)
    rhs_phs = [ph for ph in phs if ph not in lhs_set]

    in_map: Dict[str, str] = {}
    out_map: Dict[str, str] = {}
    used_lhs: Set[str] = set()
    used_rhs: Set[str] = set()

    for ok in outs_keys:
        if ok in lhs_set:
            out_map[ok] = ok
            used_lhs.add(ok)
    rem_outs = [ok for ok in outs_keys if ok not in out_map]
    rem_lhs = [ph for ph in lhs_phs if ph not in used_lhs]
    for ok, ph in zip(rem_outs, rem_lhs):
        out_map[ok] = ph

    rhs_set = set(rhs_phs)
    for ik in ins_keys:
        if ik in rhs_set:
            in_map[ik] = ik
            used_rhs.add(ik)
    rem_ins = [ik for ik in ins_keys if ik not in in_map]
    rem_rhs = [ph for ph in rhs_phs if ph not in used_rhs]
    for ik, ph in zip(rem_ins, rem_rhs):
        in_map[ik] = ph

    return in_map, out_map


def _format_scalar_binding(
    val: Any,
    p_obj: Any,
    v_class: str,
    backend: LexicalLanguageBackend,
) -> str:
    role_low = str(get_attr(p_obj, "port_role", None) or "").strip().lower()
    if v_class == "identifier" and role_low == "source_identifier" and isinstance(val, str):
        return val.strip()
    return backend.format_literal(val)


def _extract_imported_bindings(dep_stmt: str) -> Set[str]:
    """Extract locally bound identifier names from formatted dependency statements without regex or ast."""
    bindings: Set[str] = set()
    s = str(dep_stmt or "").strip()
    if not s:
        return bindings
    parts = s.split()
    if parts[0] == "import":
        rest = s[len("import"):].strip()
        for item in rest.split(","):
            toks = item.strip().split()
            if not toks:
                continue
            if len(toks) >= 3 and toks[1] == "as":
                bindings.add(toks[2].strip())
            else:
                bindings.add(toks[0].split(".", 1)[0].strip())
    elif parts[0] == "from" and " import " in s:
        rhs = s.split(" import ", 1)[1].strip()
        for item in rhs.split(","):
            toks = item.strip().split()
            if not toks:
                continue
            if len(toks) >= 3 and toks[1] == "as":
                bindings.add(toks[2].strip())
            elif toks[0] != "*":
                bindings.add(toks[0].strip())
    return {b for b in bindings if b}


def _collect_plan_symbols(
    steps: Iterable[PlannedStep],
    backend: LexicalLanguageBackend,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Recursively gather imported bindings, defined output variable names, and all variable names in plan."""
    imported: Set[str] = set()
    defined_outs: Set[str] = set()
    all_vars: Set[str] = set()

    def _visit(seq: Iterable[PlannedStep]) -> None:
        for st in seq:
            if st.output_var:
                defined_outs.add(st.output_var)
                all_vars.add(st.output_var)
            for ov in (getattr(st, "output_vars", None) or {}).values():
                if ov:
                    defined_outs.add(str(ov))
                    all_vars.add(str(ov))
            for iv in (st.input_wiring or {}).values():
                if iv:
                    all_vars.add(str(iv))
            if st.cell is not None:
                for dep in (get_attr(st.cell, "dependencies", ()) or ()):
                    d_stmt = backend.format_dependency_statement(str(dep or ""))
                    if d_stmt:
                        imported.update(_extract_imported_bindings(d_stmt))
            if st.body_steps:
                _visit(st.body_steps)
            if st.else_steps:
                _visit(st.else_steps)

    _visit(steps)
    return imported, defined_outs, all_vars


def bind_and_emit(
    plan: CandidatePlan,
    ir: TypedIR,
    slot_fill_fn: Optional[Callable[[Any, List[str], Optional[IRNode]], Dict[str, Any]]] = None,
    language_backend: Optional[LexicalLanguageBackend] = None,
) -> BindEmitResult:
    """Bind all ports from plan wiring + TypedIR, emit structured code via LanguageBackend, and validate syntax."""
    backend = language_backend or get_active_language_backend()
    nodes_by_id: Dict[str, IRNode] = {n.id: n for n in ir.flatten_nodes()}
    grounded_assets = ir.grounded_assets()

    imported_bindings, defined_outs, all_vars = _collect_plan_symbols(plan.steps, backend)
    reserved_syms = imported_bindings | set(keyword.kwlist)
    var_remap: Dict[str, str] = {}
    used_safe = set(reserved_syms) | set(all_vars)
    for var_name in sorted(defined_outs):
        if var_name in reserved_syms:
            cand = f"{var_name}_val"
            suffix = 2
            while cand in used_safe:
                cand = f"{var_name}_val_{suffix}"
                suffix += 1
            var_remap[var_name] = cand
            used_safe.add(cand)

    def _safe_v(name: str) -> str:
        return var_remap.get(name, name)

    bound_steps: List[BoundStep] = []
    ordered_deps: List[str] = []
    seen_deps: Set[str] = set()
    input_assets_acc: List[str] = []
    egress_paths_acc: List[str] = []

    def _emit_step(step: PlannedStep, indent_level: int) -> Optional[BindEmitResult]:
        indent = "    " * indent_level
        if step.kind == "loop":
            cond_expr = _clean_ir_condition(step.condition)
            lines: List[str] = [f"{indent}{backend.format_loop_header(cond_expr)}"]
            if step.body_steps:
                for sub_step in step.body_steps:
                    sub_res = _emit_step(sub_step, indent_level + 1)
                    if sub_res is not None:
                        if not sub_res.ok:
                            return sub_res
                        if sub_res.code:
                            lines.append(sub_res.code)
            elif backend.noop_statement:
                lines.append(f"{indent}    {backend.noop_statement}")
            blk_end = backend.format_block_end()
            if blk_end:
                lines.append(f"{indent}{blk_end}")
            return BindEmitResult(ok=True, code="\n".join(lines))

        if step.kind == "branch":
            cond_expr = _clean_ir_condition(step.condition)
            lines = [f"{indent}{backend.format_branch_header(cond_expr)}"]
            if step.body_steps:
                for sub_step in step.body_steps:
                    sub_res = _emit_step(sub_step, indent_level + 1)
                    if sub_res is not None:
                        if not sub_res.ok:
                            return sub_res
                        if sub_res.code:
                            lines.append(sub_res.code)
            elif backend.noop_statement:
                lines.append(f"{indent}    {backend.noop_statement}")
            if step.else_steps:
                else_hdr = backend.format_else_header()
                blk_end = backend.format_block_end()
                if blk_end and not else_hdr.startswith(blk_end):
                    lines.append(f"{indent}{blk_end}")
                lines.append(f"{indent}{else_hdr}")
                for sub_step in step.else_steps:
                    sub_res = _emit_step(sub_step, indent_level + 1)
                    if sub_res is not None:
                        if not sub_res.ok:
                            return sub_res
                        if sub_res.code:
                            lines.append(sub_res.code)
            blk_end = backend.format_block_end()
            if blk_end:
                lines.append(f"{indent}{blk_end}")
            return BindEmitResult(ok=True, code="\n".join(lines))

        cell = step.cell
        cid = step.cell_id
        ir_node = nodes_by_id.get(step.ir_node_id) if step.ir_node_id else None
        bindings: Dict[str, str] = {}

        raw_outs = get_attr(cell, "outputs", {}) or {}
        raw_ins = get_attr(cell, "inputs", {}) or {}
        tmpl = str(backend.select_cell_template(cell) or "").strip()
        if not tmpl:
            return BindEmitResult(
                ok=False,
                refusal_kind="EMPTY_TEMPLATE",
                error=f"Cell '{cid}' has an empty code_template.",
            )

        out_keys = [str(k) for k in raw_outs.keys()] if isinstance(raw_outs, dict) else []
        in_keys = [str(k) for k in raw_ins.keys()] if isinstance(raw_ins, dict) else []
        in_port_to_slot, out_port_to_slot = _align_template_slots(tmpl, in_keys, out_keys, backend)

        if isinstance(raw_outs, dict) and raw_outs:
            step_out_vars = getattr(step, "output_vars", None) or {}
            for ok_str, ov_name in step_out_vars.items():
                ok_s = str(ok_str)
                if ok_s in raw_outs and ov_name:
                    safe_ov = _safe_v(str(ov_name))
                    bindings[ok_s] = safe_ov
                    slot_out = out_port_to_slot.get(ok_s)
                    if slot_out and slot_out not in bindings:
                        bindings[slot_out] = safe_ov

            prim_out = get_attr(cell, "primary_out", None)
            first_out = str(prim_out) if (prim_out and str(prim_out) in raw_outs) else str(next(iter(raw_outs)))
            safe_prim_out = _safe_v(step.output_var)
            if first_out not in bindings:
                bindings[first_out] = safe_prim_out
            if first_out in out_port_to_slot and out_port_to_slot[first_out] not in bindings:
                bindings[out_port_to_slot[first_out]] = bindings[first_out]
            for out_k in raw_outs.keys():
                ok_str = str(out_k)
                if ok_str not in bindings:
                    out_vname = _safe_v(f"{safe_prim_out}_{ok_str}")
                    bindings[ok_str] = out_vname
                    slot_out = out_port_to_slot.get(ok_str)
                    if slot_out and slot_out not in bindings:
                        bindings[slot_out] = out_vname

        if isinstance(raw_ins, dict):
            node_params = ir_node.params if ir_node is not None else {}
            aligned_param_keys = align_node_params_to_ports(
                node_params,
                raw_ins,
                in_port_to_slot,
                allow_single_fallback=(ir.source_mode == "slm_compiled"),
                raw_prompt=ir.raw_prompt,
            )
            grounded_node_assets = [
                pk for pk, pv in node_params.items()
                if pv.value_class == "external_asset" and pv.provenance == "prompt" and pv.value is not None
            ]
            used_node_assets: Set[str] = set()

            for pname, p_obj in raw_ins.items():
                p_str = str(pname)
                slot_name = in_port_to_slot.get(p_str)
                if p_str in step.input_wiring:
                    wired_var = _safe_v(step.input_wiring[p_str])
                    bindings[p_str] = wired_var
                    if slot_name:
                        bindings[slot_name] = wired_var
                    continue
                if slot_name and slot_name in step.input_wiring:
                    wired_var = _safe_v(step.input_wiring[slot_name])
                    bindings[p_str] = wired_var
                    bindings[slot_name] = wired_var
                    continue

                vc = derive_value_class(p_obj)
                is_req = bool(get_attr(p_obj, "required", True))
                raw_def_val = get_attr(p_obj, "default_value", None)
                def_val = backend.parse_schema_default(raw_def_val)

                def _bind_port_and_slot(rendered_val: str) -> None:
                    bindings[p_str] = rendered_val
                    if slot_name:
                        bindings[slot_name] = rendered_val

                if vc == "external_asset":
                    asset_val: Optional[str] = None
                    p_entry = node_params.get(p_str)
                    matched_key: Optional[str] = p_str
                    if p_entry is None and slot_name:
                        p_entry = node_params.get(slot_name)
                        matched_key = slot_name
                    if p_entry is not None and p_entry.value_class == "external_asset" and p_entry.provenance == "prompt" and p_entry.value is not None:
                        asset_val = str(p_entry.value)
                        if matched_key:
                            used_node_assets.add(matched_key)
                    else:
                        rem_assets = [k for k in grounded_node_assets if k not in used_node_assets]
                        if len(rem_assets) == 1:
                            asset_val = str(node_params[rem_assets[0]].value)
                            used_node_assets.add(rem_assets[0])
                        elif ir_node is not None:
                            for inp_ref in ir_node.inputs:
                                if inp_ref in grounded_assets:
                                    asset_val = str(grounded_assets[inp_ref])
                                    break

                    if asset_val is None:
                        if raw_def_val is not None and not is_req:
                            _bind_port_and_slot(backend.format_literal(def_val))
                            continue
                        return BindEmitResult(
                            ok=False,
                            refusal_kind="UNBOUND_EXTERNAL_ASSET",
                            error=f"Missing prompt-grounded external_asset '{cid}.{p_str}'",
                            unbound_ports=(f"{cid}.{p_str}",),
                        )
                    asset_clean = asset_val.strip()
                    if asset_clean:
                        if is_egress_asset_port(p_obj):
                            if asset_clean not in egress_paths_acc:
                                egress_paths_acc.append(asset_clean)
                        elif asset_clean not in input_assets_acc:
                            input_assets_acc.append(asset_clean)
                    _bind_port_and_slot(backend.format_literal(asset_val))
                    continue

                param_val = None
                has_param = False
                aligned_k = aligned_param_keys.get(p_str)
                p_entry = node_params.get(aligned_k) if aligned_k is not None else node_params.get(p_str)
                if p_entry is None and slot_name:
                    p_entry = node_params.get(slot_name)
                if p_entry is not None and p_entry.provenance in ("prompt", "estimated", "default") and p_entry.value is not None:
                    param_val = p_entry.value
                    has_param = True
                elif raw_def_val is not None:
                    param_val = def_val
                    has_param = True
                elif is_req and slot_fill_fn is not None:
                    filled = slot_fill_fn(cell, [p_str], ir_node) or {}
                    if p_str in filled and filled[p_str] is not None:
                        param_val = filled[p_str]
                        has_param = True
                    elif slot_name and slot_name in filled and filled[slot_name] is not None:
                        param_val = filled[slot_name]
                        has_param = True

                if not has_param:
                    if is_req:
                        return BindEmitResult(
                            ok=False,
                            refusal_kind="UNBOUND_PORT",
                            error=f"Missing required parameter '{cid}.{p_str}'",
                            unbound_ports=(f"{cid}.{p_str}",),
                        )
                    _bind_port_and_slot(backend.null_literal)
                    continue

                param_val = coerce_literal_for_port(param_val, p_obj)
                if param_val is not None and not is_value_compatible_with_primitive_carrier(param_val, p_obj):
                    return BindEmitResult(
                        ok=False,
                        refusal_kind="TYPE_MISMATCH",
                        error=(
                            f"Value {param_val!r} is incompatible with port '{cid}.{p_str}' "
                            f"carrier '{get_attr(p_obj, 'carrier', 'any')}'"
                        ),
                    )

                if vc == "enum":
                    enum_vals = get_attr(p_obj, "enum_values", None) or ()
                    if enum_vals:
                        unwrapped_strs = {str(backend.parse_schema_default(ev)) for ev in enum_vals}
                        if param_val not in enum_vals and str(param_val) not in unwrapped_strs:
                            return BindEmitResult(
                                ok=False,
                                refusal_kind="INVALID_ENUM_VALUE",
                                error=f"Value {param_val!r} for '{cid}.{p_str}' not in allowed enum {list(enum_vals)}",
                            )

                if param_val is not None:
                    ok_ref, ref_err = RefinementSet.from_invariants(
                        get_attr(p_obj, "invariants", ()) or ()
                    ).check_literal(param_val)
                    if not ok_ref:
                        return BindEmitResult(
                            ok=False,
                            refusal_kind="REFINEMENT_VIOLATION",
                            error=f"Value {param_val!r} for '{cid}.{p_str}' violates refinement invariant: {ref_err}",
                        )

                _bind_port_and_slot(_format_scalar_binding(param_val, p_obj, vc, backend))

        try:
            rendered = tmpl.format(**bindings)
        except Exception as exc:
            return BindEmitResult(
                ok=False,
                refusal_kind="TEMPLATE_FORMAT_ERROR",
                error=f"Failed rendering template for '{cid}': {exc}",
            )

        raw_deps = get_attr(cell, "dependencies", ()) or ()
        step_deps: List[str] = []
        for dep in raw_deps:
            d_str = backend.format_dependency_statement(str(dep or ""))
            if d_str:
                step_deps.append(d_str)
                if d_str not in seen_deps:
                    seen_deps.add(d_str)
                    ordered_deps.append(d_str)

        bound_steps.append(
            BoundStep(
                cell_id=cid,
                ir_node_id=step.ir_node_id,
                is_live_bridge=step.is_live_bridge,
                bindings=bindings,
                code=rendered,
                dependencies=tuple(step_deps),
            )
        )
        indented = "\n".join(
            f"{indent}{ln}" if ln.strip() else ln
            for ln in rendered.splitlines()
        )
        return BindEmitResult(ok=True, code=indented)

    code_lines: List[str] = []
    for step in plan.steps:
        res = _emit_step(step, 0)
        if res is not None:
            if not res.ok:
                return res
            if res.code:
                code_lines.append(res.code)

    body_code = "\n".join(code_lines)
    full_code = ("\n".join(ordered_deps) + "\n\n" + body_code) if ordered_deps else body_code

    ok_syn, syn_err = backend.validate_syntax(full_code, allow_placeholders=False)
    if not ok_syn:
        return BindEmitResult(
            ok=False,
            steps=tuple(bound_steps),
            code=full_code,
            refusal_kind="SYNTAX_ERROR",
            error=f"Syntax validation failed: {syn_err}",
        )

    input_set = set(input_assets_acc)
    return BindEmitResult(
        ok=True,
        steps=tuple(bound_steps),
        code=full_code,
        input_assets=tuple(input_assets_acc),
        egress_paths=tuple(p for p in egress_paths_acc if p not in input_set),
    )


__all__ = [
    "BoundStep",
    "BindEmitResult",
    "bind_and_emit",
]
