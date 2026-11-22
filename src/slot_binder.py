"""
src/slot_binder.py - Single modular LLM parameter extraction and missing-port completion engine.

Replaces scattered parameter-extraction helpers inside `src/unification.py` with
a cohesive, schema-constrained pipeline slot binder parameterized by `type_predicates`.
"""

from __future__ import annotations

import json
import keyword
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from log_config import get_logger

try:
    from .tokenizer import CellTokenizer
    from .type_predicates import (
        is_literal_parameter_port,
        parse_numeric_sequence_spec,
        validate_slot_value,
    )
except (ImportError, ValueError):
    from tokenizer import CellTokenizer
    from type_predicates import (
        is_literal_parameter_port,
        parse_numeric_sequence_spec,
        validate_slot_value,
    )

logger = get_logger("slot_binder")


def consume_matching_literals(ctx: Any, slot_val: Any) -> None:
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
        matched_i = None
        for idx, (_, _, lit_v) in enumerate(ordered):
            if idx in used:
                continue
            if str(lit_v).strip().strip("'\"").lower() == targ:
                lit_cl = lit_clause_map.get(idx)
                if clause_idx is not None and lit_cl == clause_idx:
                    matched_i = idx
                    break
                if matched_i is None and (clause_idx is None or lit_cl is None or lit_cl <= clause_idx):
                    matched_i = idx
        if matched_i is not None:
            used.add(matched_i)


def is_slot_grounded_in_clause(
    ctx: Any,
    slot_val: Any,
    port_sig: Optional[Any] = None,
    cell: Optional[Any] = None,
    registry: Optional[Any] = None,
) -> bool:
    """
    Verify an LLM-supplied slot value does not silently shadow an unconsumed
    prompt literal of the same channel in the active clause.
    """
    _c_cl = getattr(ctx, "current_cell_clause_idx", None)
    _lit_cl_map = getattr(ctx, "literal_clause_map", None) or {}
    _num_cl = len(getattr(ctx, "clause_spans", ()) or ())
    if _c_cl is None or _num_cl <= 1 or not _lit_cl_map:
        return True
    if port_sig is not None:
        reg = registry
        t_n = (getattr(port_sig, "type_name", None) or "").lower()
        is_b = bool(reg and reg.is_subtype(t_n, "bool"))
        is_s = bool(reg and (reg.is_subtype(t_n, "str") or reg.is_declared_top(t_n)))
        is_n = (bool(reg and reg.is_subtype(t_n, "numeric")) and not is_b) or (
            isinstance(slot_val, (int, float)) and not isinstance(slot_val, bool)
        )
        if is_s and not is_n:
            _kinds: Tuple[str, ...] = ("identifier", "quoted_str")
        elif is_n and not is_s:
            _kinds = ("numeric", "int_literal", "float_literal")
        else:
            _kinds = ("identifier", "quoted_str", "numeric", "int_literal", "float_literal")
    else:
        _kinds = ("identifier", "quoted_str", "numeric", "int_literal", "float_literal")

    _slot_items = (
        [str(el).strip().lower() for el in slot_val]
        if isinstance(slot_val, (list, tuple))
        else [str(slot_val).strip().lower()]
    )
    ordered_lits = getattr(ctx, "ordered_literals", ()) or ()
    if getattr(cell, "_auto_inserted_bridge", False):
        _all_prompt_lits = {
            str(v).strip().lower()
            for _, k_lit, v in ordered_lits
            if k_lit in _kinds
        }
        if _slot_items and all(si in _all_prompt_lits for si in _slot_items):
            return True
    _cl_lits = [
        (i, str(v).strip().lower())
        for i, (_, k_lit, v) in enumerate(ordered_lits)
        if _lit_cl_map.get(i) == _c_cl and k_lit in _kinds
    ]
    used_idx = getattr(ctx, "used_indices", set()) or set()
    _unconsumed_cl = [lv for i, lv in _cl_lits if i not in used_idx]
    return not (_cl_lits and _unconsumed_cl and not any(si in {lv for _, lv in _cl_lits} for si in _slot_items))


def _load_model_manager() -> Optional[Any]:
    try:
        from .inference import ModelManager
        return ModelManager.get_instance()
    except (ImportError, ValueError):
        try:
            from inference import ModelManager
            return ModelManager.get_instance()
        except Exception:
            return None
    except Exception:
        return None


def _extract_json_block(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else None


def _build_verbatim_vocab(
    prompt: str,
    ctx: Any,
) -> Tuple[str, Dict[str, str], Dict[str, str], List[str]]:
    prompt_low_full = prompt.lower()
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

    all_verbatim_ordered: List[str] = []
    seen_low: Set[str] = set()
    for lv in ordered_lit_case.values():
        if lv.lower() not in seen_low:
            seen_low.add(lv.lower())
            all_verbatim_ordered.append(lv)
    for pv in prompt_token_case.values():
        if pv.lower() not in seen_low:
            seen_low.add(pv.lower())
            all_verbatim_ordered.append(pv)
    return prompt_low_full, prompt_token_case, ordered_lit_case, all_verbatim_ordered


def _ground_str_token(
    val_s: str,
    sig_obj: Any,
    prompt: str,
    prompt_low_full: str,
    prompt_token_case: Dict[str, str],
    ordered_lit_case: Dict[str, str],
) -> Optional[str]:
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
    if len(clean) >= 2 and (" " in clean or ":" in clean or "/" in clean or "." in clean or "-" in clean):
        pos_m = prompt_low_full.find(c_low)
        if pos_m >= 0:
            return prompt[pos_m : pos_m + len(clean)]
    return None


def _build_port_json_schema(
    p_name: str,
    info: Dict[str, Any],
    registry: Any,
    clause_txt: str,
    cands_fn: Callable[[Dict[str, Any], str], List[str]],
) -> Tuple[Dict[str, Any], str]:
    e_vals = info.get("enum_values")
    is_req_p = bool(info.get("required"))
    req_tag = "required" if is_req_p else "optional"
    t_low = str(info.get("type") or "").lower()
    doc_s = str(info.get("doc") or "").strip()
    doc_note = f": {doc_s}" if doc_s else ""

    if e_vals:
        base_types: Any = [str(ev) for ev in e_vals]
        return {"enum": base_types + ([None] if not is_req_p else [])}, f"{p_name} ({req_tag}, options={base_types}{doc_note})"

    num_seq: Optional[Any] = info.get("num_seq")
    if num_seq is not None:
        item_t = "integer" if num_seq.elem_is_int else "number"
        arr_sch: Dict[str, Any] = {
            "type": "array" if is_req_p else ["array", "null"],
            "items": {"type": item_t},
        }
        if num_seq.arity:
            arr_sch["minItems"] = int(num_seq.arity)
            arr_sch["maxItems"] = int(num_seq.arity)
        len_note = f" len={num_seq.arity}" if num_seq.arity else ""
        return arr_sch, f"{p_name} ({req_tag}, {t_low}{len_note}{doc_note})"

    if info.get("is_collection"):
        cands = cands_fn(info, clause_txt) if info.get("is_key_proj") else []
        item_schema: Dict[str, Any] = {"enum": cands} if cands else {"type": "string"}
        coll_sch = (
            {"type": "array", "items": item_schema}
            if is_req_p
            else {"type": ["array", "null"], "items": item_schema}
        )
        return coll_sch, f"{p_name} ({req_tag}, list{doc_note})"

    if registry.is_subtype(t_low, "bool"):
        b_sch = {"type": "boolean"} if is_req_p else {"type": ["boolean", "null"]}
        return b_sch, f"{p_name} ({req_tag}, bool{doc_note})"

    if registry.is_subtype(t_low, "numeric"):
        n_sch = {"type": "number"} if is_req_p else {"type": ["number", "null"]}
        return n_sch, f"{p_name} ({req_tag}, number{doc_note})"

    if info.get("is_path"):
        path_cands = cands_fn(info, clause_txt)
        if path_cands:
            return {"enum": path_cands + [None]}, f"{p_name} (path from prompt{doc_note})"
        return {"type": ["string", "null"]}, f"{p_name} (verbatim path from prompt or null if absent{doc_note})"

    cands = cands_fn(info, clause_txt) if info.get("is_key_proj") else []
    if cands and info.get("is_key_proj"):
        k_sch = {"enum": cands + ([None] if not is_req_p else [])}
        return k_sch, f"{p_name} ({req_tag}, key from clause{doc_note})"
    s_sch = {"type": "string"} if is_req_p else {"type": ["string", "null"]}
    return s_sch, f"{p_name} ({req_tag}, str{doc_note})"


def _validate_and_normalize_slot_value(
    val: Any,
    p_info: Dict[str, Any],
    p_sig: Any,
    concrete_sig: Any,
    registry: Any,
    prompt: str,
    prompt_low_full: str,
    prompt_token_case: Dict[str, str],
    ordered_lit_case: Dict[str, str],
    valid_nums: Set[str],
    key_proj_roles: Set[str],
    proj_states: Set[str],
    key_binds: Set[str],
    extracted_vars: List[str],
    type_term_cls: Any,
) -> Any:
    is_req_p = bool(p_info.get("required")) if p_info else bool(getattr(p_sig, "required", False))
    t_low_p = str(p_info.get("type") or getattr(concrete_sig, "type_name", "") or "").lower()
    declared_enums_p = p_info.get("enum_values") if p_info else getattr(concrete_sig, "enum_values", None)
    is_path_p = bool(p_info.get("is_path"))
    is_path_str_fn = p_info.get("_is_path_str")
    is_proj = False
    if p_sig is not None:
        pr = str(getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or "").lower()
        ps = str(getattr(concrete_sig, "state", "") or "").lower()
        is_proj = (
            pr in key_proj_roles
            or ps in proj_states
            or getattr(p_sig, "binds", "") in key_binds
        )

    accepted_val: Any = None
    if isinstance(val, list):
        num_seq = p_info.get("num_seq")
        if num_seq is None and concrete_sig is not None:
            num_seq = parse_numeric_sequence_spec(concrete_sig, registry, type_term_cls=type_term_cls)
        if num_seq is not None:
            num_items: List[Union[int, float]] = []
            seq_ok = True
            for item in val:
                if isinstance(item, bool):
                    seq_ok = False
                    break
                try:
                    f_it = float(item)
                except (TypeError, ValueError):
                    seq_ok = False
                    break
                if not (f_it == f_it) or abs(f_it) == float("inf"):
                    seq_ok = False
                    break
                if num_seq.elem_is_int and f_it.is_integer():
                    num_items.append(int(f_it))
                elif isinstance(item, int):
                    num_items.append(int(item))
                else:
                    num_items.append(f_it)
            if seq_ok and num_items and (num_seq.arity is None or len(num_items) == int(num_seq.arity)):
                if is_req_p or any(str(el) in valid_nums or str(el) in prompt for el in num_items):
                    accepted_val = tuple(num_items) if num_seq.is_tuple else num_items
        else:
            grounded_items: List[str] = []
            for item in val:
                if isinstance(item, str):
                    g_it = _ground_str_token(
                        item, concrete_sig, prompt, prompt_low_full, prompt_token_case, ordered_lit_case
                    )
                    if is_path_p:
                        if g_it is not None and (is_path_str_fn is None or is_path_str_fn(g_it)):
                            grounded_items.append(g_it)
                    elif g_it is not None:
                        grounded_items.append(g_it)
                        if is_proj and g_it.isidentifier():
                            extracted_vars.append(g_it)
                    elif is_req_p and not is_proj and not is_path_p and not declared_enums_p:
                        clean_it = item.strip().strip("'\"")
                        if clean_it:
                            grounded_items.append(clean_it)
            if grounded_items:
                accepted_val = grounded_items
    elif isinstance(val, (int, float)) and not isinstance(val, bool):
        is_int_target = registry.is_subtype(t_low_p, "int") and not registry.is_subtype(t_low_p, "bool")
        norm_num: Union[int, float] = (
            int(val)
            if (is_int_target and isinstance(val, float) and val.is_integer())
            else (int(val) if isinstance(val, float) and int(val) == val else val)
        )
        val_str = str(norm_num)
        if val_str in valid_nums or str(val) in valid_nums or is_req_p:
            accepted_val = norm_num
    elif isinstance(val, bool):
        accepted_val = val
    elif isinstance(val, str):
        g_str = _ground_str_token(
            val, concrete_sig, prompt, prompt_low_full, prompt_token_case, ordered_lit_case
        )
        if is_path_p:
            if g_str is not None and (is_path_str_fn is None or is_path_str_fn(g_str)):
                accepted_val = g_str
        elif g_str is not None:
            accepted_val = g_str
            if is_proj and g_str.isidentifier():
                extracted_vars.append(g_str)
        elif is_req_p and not is_proj and not is_path_p and not declared_enums_p:
            clean_est = val.strip().strip("'\"")
            if clean_est:
                accepted_val = clean_est

    if accepted_val is None:
        return None
    if concrete_sig and not validate_slot_value(accepted_val, concrete_sig, registry=registry, type_term_cls=type_term_cls):
        return None
    return accepted_val


class PipelineSlotBinder:
    """Single modular slot extractor and missing-parameter completer backed by LLM + type predicates."""

    @classmethod
    def _port_descriptor(
        cls,
        cell: Any,
        p_name: str,
        p_sig: Any,
        registry: Any,
        *,
        key_proj_roles: Set[str],
        key_binds: Set[str],
        is_path_port_fn: Callable[..., bool],
        type_term_cls: Any,
        is_path_string_fn: Callable[[str], bool],
    ) -> Dict[str, Any]:
        p_role = str(getattr(p_sig, "derived_role", "") or getattr(p_sig, "port_role", "") or "").strip().lower()
        t_name = (getattr(p_sig, "type_name", "") or "").lower()
        is_path_p = is_path_port_fn(p_sig)
        is_key_proj = (p_role in key_proj_roles) or (str(getattr(p_sig, "binds", "") or "").lower() in key_binds)
        is_req = bool(getattr(p_sig, "required", False))
        num_seq = parse_numeric_sequence_spec(p_sig, registry, type_term_cls=type_term_cls)
        sk = str(getattr(p_sig, "structure_kind", "") or "").lower()
        cr = getattr(p_sig, "canonical_rank", None)
        is_coll = (
            num_seq is None
            and sk != "scalar"
            and (
                sk == "collection"
                or (cr is not None and cr >= 1 and not registry.is_subtype(t_name, "tensor"))
                or (not registry.is_subtype(t_name, "str") and (registry.is_subtype(t_name, "collection") or registry.is_subtype(t_name, "sequence")))
            )
        )
        return {
            "type": t_name or "any",
            "required": is_req,
            "default": str(getattr(p_sig, "default_value", None)),
            "enum_values": getattr(p_sig, "enum_values", None),
            "is_collection": is_coll,
            "num_seq": num_seq,
            "is_key_proj": is_key_proj,
            "is_path": is_path_p,
            "port_role": p_role,
            "doc": str(getattr(p_sig, "doc", "") or getattr(p_sig, "description", "") or "").strip()[:80],
            "sig": p_sig,
            "_is_path_str": is_path_string_fn,
        }

    @classmethod
    def extract(
        cls,
        cells: List[Any],
        prompt: str,
        ctx: Any,
        registry: Any,
        *,
        is_path_port_fn: Callable[..., bool],
        is_macro_block_slot_fn: Callable[..., bool],
        schema_is_bridge_fn: Callable[..., bool],
        resolve_cell_port_fn: Callable[..., Any],
        is_path_string_fn: Callable[[str], bool],
        type_term_cls: Any,
    ) -> None:
        if not prompt or not cells:
            return
        mm = _load_model_manager()
        if mm is None or not mm.can_extract_slots():
            return

        ctx._cell_slots_extracted = True
        try:
            clauses = CellTokenizer.split_prompt_clauses(prompt)
        except Exception:
            clauses = [prompt]

        key_proj_roles = (
            registry.get_roles_for_category("projection")
            | set(registry.get_projection_tokens() or ())
            | registry.get_roles_for_category("key")
        )
        key_binds = registry.get_roles_for_category("key")
        prompt_toks_low = set(CellTokenizer.tokenize_prompt(prompt)) if prompt else set()

        clauses_have_numeric: Set[int] = set()
        lit_cl_map = getattr(ctx, "literal_clause_map", None) or {}
        for li, (_, lk, _) in enumerate(getattr(ctx, "ordered_literals", []) or ()):
            if lk in ("numeric", "int_literal", "float_literal"):
                lcl = lit_cl_map.get(li)
                if lcl is not None:
                    clauses_have_numeric.add(lcl)
        for ci_s, cl_str in enumerate(clauses):
            if any(ch.isdigit() for ch in cl_str):
                clauses_have_numeric.add(ci_s)

        step_candidates: Dict[str, Tuple[int, Any, Dict[str, Dict[str, Any]]]] = {}
        for step_idx, c in enumerate(cells):
            if not hasattr(c, "inputs") or not c.inputs:
                continue
            has_req_num_port = any(
                bool(getattr(ps, "required", False))
                and not is_macro_block_slot_fn(c, pn)
                and (
                    (registry.is_subtype((getattr(ps, "type_name", "") or "").lower(), "numeric") and not registry.is_subtype((getattr(ps, "type_name", "") or "").lower(), "bool"))
                    or parse_numeric_sequence_spec(ps, registry, type_term_cls=type_term_cls) is not None
                )
                for pn, ps in c.inputs.items()
            )
            cl_idx_c = getattr(c, "matched_clause_idx", None)
            cl_has_num = (cl_idx_c in clauses_have_numeric) if cl_idx_c is not None else bool(clauses_have_numeric)

            c_ports: Dict[str, Dict[str, Any]] = {}
            for p_name, p_sig in c.inputs.items():
                if not is_literal_parameter_port(
                    c, p_name, p_sig, registry,
                    is_path_port_fn=is_path_port_fn,
                    is_macro_block_slot_fn=is_macro_block_slot_fn,
                    type_term_cls=type_term_cls,
                ):
                    continue
                is_req = bool(getattr(p_sig, "required", False))
                t_name = (getattr(p_sig, "type_name", "") or "").lower()
                num_seq = parse_numeric_sequence_spec(p_sig, registry, type_term_cls=type_term_cls)
                if not is_req:
                    p_toks = set(CellTokenizer.tokenize_identifier(p_name))
                    is_num_or_seq = (num_seq is not None) or (
                        registry.is_subtype(t_name, "numeric") and not registry.is_subtype(t_name, "bool")
                    )
                    if not (
                        (p_toks and (p_toks & prompt_toks_low))
                        or (is_num_or_seq and cl_has_num and not has_req_num_port)
                    ):
                        continue
                desc = cls._port_descriptor(
                    c, p_name, p_sig, registry,
                    key_proj_roles=key_proj_roles, key_binds=key_binds,
                    is_path_port_fn=is_path_port_fn, type_term_cls=type_term_cls,
                    is_path_string_fn=is_path_string_fn,
                )
                c_ports[p_name] = desc
            if c_ports:
                step_candidates[f"step_{step_idx}:{c.cell_id}"] = (step_idx, c, c_ports)

        if not step_candidates:
            return

        prompt_low_full, prompt_token_case, ordered_lit_case, all_verbatim_ordered = _build_verbatim_vocab(prompt, ctx)

        step_action_toks: Set[str] = set()
        optic_roles_slot = registry.get_roles_for_category("optic")
        for oc in cells:
            oc_optic = str(getattr(oc, "optic_role", "") or "").strip().lower()
            if oc_optic in optic_roles_slot or schema_is_bridge_fn(oc, registry):
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
            clause_set_o = {
                rw.strip(".,;:()[]{}\"'`!?=").lower()
                for rw in (clause_txt or "").split()
                if rw.strip(".,;:()[]{}\"'`!?=")
            }
            valid_cands: List[str] = []
            for tok in all_verbatim_ordered:
                if keyword.iskeyword(tok.lower()):
                    continue
                is_p_str = is_path_string_fn(tok)
                if is_path_port_o and not is_p_str:
                    continue
                if not is_path_port_o and is_p_str:
                    continue
                if concrete_o and not validate_slot_value(tok, concrete_o, registry=registry, type_term_cls=type_term_cls):
                    continue
                valid_cands.append(tok)
            if not valid_cands:
                return []
            in_cl_data = [t for t in valid_cands if t.lower() in clause_set_o and t.lower() not in step_action_toks]
            in_cl_act = [t for t in valid_cands if t.lower() in clause_set_o and t.lower() in step_action_toks]
            out_cl = [t for t in valid_cands if t.lower() not in clause_set_o]
            if in_cl_data and not is_path_port_o:
                return in_cl_data[:48]
            return (in_cl_data + in_cl_act + out_cl)[:48]

        slot_props: Dict[str, Any] = {}
        cell_summary: List[str] = []
        for step_key, (step_idx, c, ports) in step_candidates.items():
            cl_idx = getattr(c, "matched_clause_idx", None)
            cl_txt = clauses[cl_idx].strip() if (isinstance(cl_idx, int) and 0 <= cl_idx < len(clauses)) else ""
            port_props: Dict[str, Any] = {}
            req_ports_step: List[str] = []
            port_descs: List[str] = []
            for p_name, info in ports.items():
                if info.get("required"):
                    req_ports_step.append(p_name)
                p_sch, p_desc = _build_port_json_schema(p_name, info, registry, cl_txt, _port_enum_candidates)
                port_props[p_name] = p_sch
                port_descs.append(p_desc)
            step_obj: Dict[str, Any] = {"type": "object", "properties": port_props}
            if req_ports_step:
                step_obj["required"] = req_ports_step
            slot_props[step_key] = step_obj
            cl_note = f' [clause: "{cl_txt}"]' if cl_txt else ""
            feeds_note = (
                f" (feeds -> {cells[step_idx + 1].cell_id})"
                if step_idx + 1 < len(cells)
                and (
                    str(getattr(c, "optic_role", "") or "").lower() == "getter"
                    or str(getattr(c, "node_role", "") or "").lower() == "bridge"
                )
                else ""
            )
            cell_summary.append(f"- {step_key}{cl_note}{feeds_note} -> {', '.join(port_descs)}")

        schema = {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "description": "Mapping from step key to port_name and extracted or estimated literal argument value",
                    "properties": slot_props,
                    "required": list(slot_props.keys()),
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "General hyperparameters or arguments extracted from prompt",
                    "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
                },
            },
            "required": ["slots"],
        }
        query = (
            f"User Prompt: {prompt}\n\n"
            f"For each step below, populate every parameter port:\n"
            + "\n".join(cell_summary)
            + "\n\nRules:\n"
            "1. First extract exact values stated in the prompt (prefer each step's [clause], or relevant values from the full prompt for setup/prerequisite steps; represent tuples as fixed-length JSON number arrays and keep multi-word quoted text intact).\n"
            "2. For any REQUIRED configuration, numeric, tuple, boolean, or non-path string parameter NOT explicitly provided in the prompt, synthesize the best sensible, valid, standard estimate matching the port's type, constraints, and task context (never leave such a parameter empty or echo type names). NEVER invent or guess file paths/filenames or key/column names not present in the prompt (return null if a required path or key is absent).\n"
            "3. Do NOT output type names, generic words like 'column', or operation verbs matching step names (reuse the active data identifier when an extraction feeds a downstream transform).\n"
            "4. Return null for optional ports not mentioned in the prompt.\n"
            "Return valid JSON only."
        )

        slot_max_tokens = min(2048, max(512, 96 * len(step_candidates)))
        try:
            raw = mm.generate_text(
                query,
                max_tokens=slot_max_tokens,
                schema=schema,
                system_prompt=(
                    "You are a precise dataflow parameter extractor and estimator. "
                    "Copy explicit arguments verbatim from the prompt (preferring each step's clause), "
                    "and synthesize sensible standard estimates for required parameters omitted from the prompt."
                ),
            )
            parsed = _extract_json_block(raw)
            if not parsed:
                return

            slots = parsed.get("slots", {})
            valid_nums = {
                v for _, k_lit, v in getattr(ctx, "ordered_literals", [])
                if k_lit in ("numeric", "int_literal", "float_literal")
            }
            proj_states = registry.get_states_by_property("is_projection", True)
            extracted_vars: List[str] = []

            if isinstance(slots, dict):
                cid_counts: Dict[str, int] = {}
                for c_obj in cells:
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
                        ports_info = (
                            step_candidates.get(f"step_{step_idx}:{c_match.cell_id}", (None, None, {}))[2]
                            if (c_match is not None and step_idx is not None)
                            else {}
                        )

                    for p_name, val in v.items():
                        if val is None:
                            continue
                        p_sig = c_match.inputs.get(p_name) if c_match and hasattr(c_match, "inputs") else None
                        concrete_sig = getattr(p_sig, "signature", p_sig) if p_sig else None
                        p_info = ports_info.get(p_name, {}) if isinstance(ports_info, dict) else {}
                        accepted_val = _validate_and_normalize_slot_value(
                            val, p_info, p_sig, concrete_sig, registry, prompt,
                            prompt_low_full, prompt_token_case, ordered_lit_case,
                            valid_nums, key_proj_roles, proj_states, key_binds,
                            extracted_vars, type_term_cls,
                        )
                        if accepted_val is None:
                            continue
                        if step_idx is not None:
                            ctx.step_llm_slots.setdefault(step_idx, {})[p_name] = accepted_val
                        if c_match is not None and (step_idx is None or cid_counts.get(c_match.cell_id, 0) <= 1):
                            ctx.llm_slots.setdefault(c_match.cell_id, {})[p_name] = accepted_val

                cls._enforce_optic_getter_causality(
                    cells, clauses, step_candidates, _port_enum_candidates,
                    step_action_toks, ordered_lit_case, resolve_cell_port_fn, ctx,
                )

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
            logger.debug(f"[SLOT_BINDER] LLM slot extraction skipped or failed: {e}")

    @classmethod
    def _enforce_optic_getter_causality(
        cls,
        cells: List[Any],
        clauses: List[str],
        step_candidates: Dict[str, Tuple[int, Any, Dict[str, Dict[str, Any]]]],
        port_enum_candidates_fn: Callable[[Dict[str, Any], str], List[str]],
        step_action_toks: Set[str],
        ordered_lit_case: Dict[str, str],
        resolve_cell_port_fn: Callable[..., Any],
        ctx: Any,
    ) -> None:
        """Prevent optic getters from reading a key defined solely by a later setter."""
        setter_first: Dict[str, Tuple[int, Optional[int]]] = {}
        for s_i, c_obj in enumerate(cells):
            if str(getattr(c_obj, "optic_role", "") or "").strip().lower() != "setter":
                continue
            s_map = ctx.step_llm_slots.get(s_i, {})
            if not s_map:
                continue
            k_p = resolve_cell_port_fn(c_obj, getattr(c_obj, "key_port", None), prefer="inputs")
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
            k_p = resolve_cell_port_fn(c_obj, getattr(c_obj, "key_port", None), prefer="inputs")
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
                if (not in_own_cl) or (s_cl_idx is not None and g_cl_idx is not None and g_cl_idx < s_cl_idx):
                    replacement_k: Optional[str] = None
                    future_undef = {sk for sk, sv in setter_first.items() if sv[0] > s_i and sk not in defined_keys_before}
                    for _, p_gk in reversed(prior_getter_vals):
                        if p_gk.lower() in g_cl_words and p_gk.lower() not in future_undef:
                            replacement_k = p_gk
                            break
                    if replacement_k is None and g_cl_txt:
                        p_info = step_candidates.get(f"step_{s_i}:{c_obj.cell_id}", (s_i, c_obj, {}))[2].get(k_pn)
                        if p_info:
                            for cand_t in port_enum_candidates_fn(p_info, g_cl_txt):
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

    @classmethod
    def complete_missing_ports(
        cls,
        cells: List[Any],
        pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
        prompt: str,
        ctx: Any,
        registry: Any,
        *,
        is_path_port_fn: Callable[..., bool],
        is_macro_block_slot_fn: Callable[..., bool],
        is_unresolved_fn: Callable[[Any], bool],
        is_path_string_fn: Callable[[str], bool],
        type_term_cls: Any,
    ) -> bool:
        """
        Single targeted completion pass for remaining unresolved required literal
        parameter ports (strictly skipping dataflow/handle/receiver/macro ports).
        """
        if not prompt or not pipeline_bindings:
            return False
        mm = _load_model_manager()
        if mm is None or not mm.can_extract_slots():
            return False

        try:
            clauses = CellTokenizer.split_prompt_clauses(prompt)
        except Exception:
            clauses = [prompt]

        key_proj_roles = (
            registry.get_roles_for_category("projection")
            | set(registry.get_projection_tokens() or ())
            | registry.get_roles_for_category("key")
        )
        key_binds = registry.get_roles_for_category("key")
        cell_obj_to_step = {id(c): idx for idx, c in enumerate(cells)}
        cid_counts: Dict[str, int] = {}
        for c_obj in cells:
            cid_counts[c_obj.cell_id] = cid_counts.get(c_obj.cell_id, 0) + 1

        prompt_low_full, prompt_token_case, ordered_lit_case, all_verbatim_ordered = _build_verbatim_vocab(prompt, ctx)

        def _enum_cands_simple(port_info: Dict[str, Any], clause_txt: str) -> List[str]:
            p_sig_o = port_info.get("sig")
            concrete_o = getattr(p_sig_o, "signature", p_sig_o) if p_sig_o else None
            is_path_port_o = bool(port_info.get("is_path"))
            clause_set_o = {
                rw.strip(".,;:()[]{}\"'`!?=").lower()
                for rw in (clause_txt or "").split()
                if rw.strip(".,;:()[]{}\"'`!?=")
            }
            valid: List[str] = []
            for tok in all_verbatim_ordered:
                if keyword.iskeyword(tok.lower()):
                    continue
                is_p_str = is_path_string_fn(tok)
                if is_path_port_o and not is_p_str:
                    continue
                if not is_path_port_o and is_p_str:
                    continue
                if concrete_o and not validate_slot_value(tok, concrete_o, registry=registry, type_term_cls=type_term_cls):
                    continue
                valid.append(tok)
            in_cl = [t for t in valid if t.lower() in clause_set_o]
            out_cl = [t for t in valid if t.lower() not in clause_set_o]
            return (in_cl or out_cl)[:48]

        missing_by_step: Dict[str, Tuple[Optional[int], Any, Dict[str, Dict[str, Any]]]] = {}
        for c_obj, bnd in pipeline_bindings:
            if not isinstance(bnd, dict):
                continue
            c_inputs = getattr(c_obj, "inputs", None) or {}
            step_idx = cell_obj_to_step.get(id(c_obj))
            if step_idx is None:
                continue
            step_key = f"step_{step_idx}:{c_obj.cell_id}"
            for p_name, val in bnd.items():
                if not is_unresolved_fn(val):
                    continue
                p_sig = c_inputs.get(p_name)
                if p_sig is None or not bool(getattr(p_sig, "required", False)):
                    continue
                if is_path_port_fn(p_sig):
                    continue
                if not is_literal_parameter_port(
                    c_obj, p_name, p_sig, registry,
                    is_path_port_fn=is_path_port_fn,
                    is_macro_block_slot_fn=is_macro_block_slot_fn,
                    type_term_cls=type_term_cls,
                ):
                    continue
                info = cls._port_descriptor(
                    c_obj, p_name, p_sig, registry,
                    key_proj_roles=key_proj_roles, key_binds=key_binds,
                    is_path_port_fn=is_path_port_fn, type_term_cls=type_term_cls,
                    is_path_string_fn=is_path_string_fn,
                )
                entry = missing_by_step.setdefault(step_key, (step_idx, c_obj, {}))
                entry[2][p_name] = info

        if not missing_by_step:
            return False

        slot_props: Dict[str, Any] = {}
        summary_lines: List[str] = []
        total_missing = 0
        for step_key, (_, c_obj, ports) in missing_by_step.items():
            cl_idx = getattr(c_obj, "matched_clause_idx", None)
            cl_txt = clauses[cl_idx].strip() if (isinstance(cl_idx, int) and 0 <= cl_idx < len(clauses)) else ""
            port_props: Dict[str, Any] = {}
            port_descs: List[str] = []
            req_list: List[str] = []
            for p_name, info in ports.items():
                total_missing += 1
                req_list.append(p_name)
                p_sch, p_desc = _build_port_json_schema(p_name, info, registry, cl_txt, _enum_cands_simple)
                port_props[p_name] = p_sch
                port_descs.append(p_desc)
            slot_props[step_key] = {"type": "object", "properties": port_props, "required": req_list}
            cl_note = f' [clause: "{cl_txt}"]' if cl_txt else ""
            summary_lines.append(f"- {step_key}{cl_note} -> {', '.join(port_descs)}")

        schema = {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "object",
                    "properties": slot_props,
                    "required": list(slot_props.keys()),
                }
            },
            "required": ["slots"],
        }
        query = (
            f"User Prompt: {prompt}\n\n"
            "The following required parameter ports still need values:\n"
            + "\n".join(summary_lines)
            + "\n\nRules:\n"
            "1. Extract the exact verbatim value from the prompt (preferring each step's [clause]) when present.\n"
            "2. If a required configuration/numeric/tuple/boolean/non-path string value was omitted from the prompt, provide the best sensible, valid, standard estimate for the task. NEVER invent or guess file paths/filenames or key/column names not present in the prompt (return null if a required path or key is absent).\n"
            "Return valid JSON only."
        )
        max_tokens = min(1024, max(256, 128 * total_missing))
        filled_any = False
        try:
            raw = mm.generate_text(
                query,
                max_tokens=max_tokens,
                schema=schema,
                system_prompt=(
                    "You are a precise dataflow parameter completer. "
                    "Supply exact values from the prompt or sensible domain-standard estimates for omitted required parameters."
                ),
            )
            parsed = _extract_json_block(raw)
            if not parsed or not isinstance(parsed.get("slots"), dict):
                return False

            valid_nums = {
                v for _, k_lit, v in getattr(ctx, "ordered_literals", [])
                if k_lit in ("numeric", "int_literal", "float_literal")
            }
            proj_states = registry.get_states_by_property("is_projection", True)
            extracted_vars: List[str] = []

            for step_key, val_map in parsed["slots"].items():
                if not isinstance(val_map, dict) or step_key not in missing_by_step:
                    continue
                step_idx, c_match, ports_info = missing_by_step[step_key]
                for p_name, val in val_map.items():
                    if val is None or p_name not in ports_info:
                        continue
                    p_sig = (getattr(c_match, "inputs", None) or {}).get(p_name)
                    concrete_sig = getattr(p_sig, "signature", p_sig) if p_sig else None
                    accepted_val = _validate_and_normalize_slot_value(
                        val, ports_info[p_name], p_sig, concrete_sig, registry, prompt,
                        prompt_low_full, prompt_token_case, ordered_lit_case,
                        valid_nums, key_proj_roles, proj_states, key_binds,
                        extracted_vars, type_term_cls,
                    )
                    if accepted_val is None:
                        continue
                    if step_idx is not None:
                        ctx.step_llm_slots.setdefault(step_idx, {})[p_name] = accepted_val
                    if cid_counts.get(c_match.cell_id, 0) <= 1:
                        ctx.llm_slots.setdefault(c_match.cell_id, {})[p_name] = accepted_val
                    filled_any = True

            if extracted_vars:
                existing_vars = list(ctx.llm_slots.get("variables") or [])
                for ev in extracted_vars:
                    if ev not in existing_vars:
                        existing_vars.append(ev)
                ctx.llm_slots["variables"] = existing_vars
                if hasattr(ctx, "sync_llm_slots"):
                    ctx.sync_llm_slots()
        except Exception as e:
            logger.debug(f"[SLOT_BINDER] Missing-port completion skipped or failed: {e}")
        return filled_any


__all__ = [
    "PipelineSlotBinder",
    "consume_matching_literals",
    "is_slot_grounded_in_clause",
]
