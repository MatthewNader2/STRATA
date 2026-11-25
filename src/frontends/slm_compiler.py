"""
src/frontends/slm_compiler.py — Unified SLM TypedIR compiler frontend.

Converts free-form natural-language user specifications into TypedIR once,
strictly separating prompt-grounded external assets from SLM-estimated config defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    from ..typed_ir import IRNode, IRParamValue, TypedIR, infer_loop_var
    from ..utils import extract_json_object
except (ImportError, ValueError):
    from typed_ir import IRNode, IRParamValue, TypedIR, infer_loop_var
    from utils import extract_json_object


_SCALAR_OR_LIST_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
        {"type": "array", "maxItems": 8, "items": {"type": ["string", "number", "integer", "boolean"]}},
        {
            "type": "object",
            "properties": {
                "value": {},
                "provenance": {"type": "string", "enum": ["prompt", "estimated", "default", "missing"]},
                "value_class": {"type": "string", "enum": ["literal", "external_asset", "identifier", "enum"]},
            },
        },
    ]
}

_NODE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["id", "intent"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "kind": {"type": "string", "enum": ["call", "loop", "branch"]},
        "intent": {"type": "string"},
        "domain_hint": {"type": ["string", "null"]},
        "inputs": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "outputs": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "params": {"type": "object", "maxProperties": 12, "additionalProperties": _SCALAR_OR_LIST_SCHEMA},
        "condition": {"type": ["string", "null"]},
        "body": {"type": "array", "maxItems": 8, "items": {"type": "object"}},
        "else_body": {"type": "array", "maxItems": 8, "items": {"type": "object"}},
    },
}

SLM_IR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["nodes"],
    "additionalProperties": False,
    "properties": {
        "assets": {
            "type": "object",
            "maxProperties": 8,
            "additionalProperties": {"type": ["string", "null", "object"]},
        },
        "required_domains": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "nodes": {"type": "array", "minItems": 1, "maxItems": 16, "items": _NODE_JSON_SCHEMA},
        "deliverable": {"type": ["string", "null"]},
        "ambiguities": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
    },
}

SLM_SYSTEM_PROMPT_TEMPLATE = """\
Convert every action in the user request into an ordered JSON pipeline of atomic steps.
Loaded domains: {loaded_domains}

Output ONLY one valid JSON object with:
- "assets": object mapping short key to concrete file path or URI verbatim from the user text (ONLY paths with file extensions or / : separators; NEVER put plain IDs, tags, or generic nouns in assets; use {{}} if none).
- "required_domains": list of library/package names ONLY when the user explicitly writes "using X" or "in library X" (verbatim, even if NOT in Loaded domains); otherwise [].
- "nodes": list with ONE node per comma-separated action clause in the user text in order (keep each action together with all its quoted tags and "at"/"with"/"on"/"for"/"over"/"into" parameters in ONE node rather than splitting an action and its parameter into two nodes; NEVER merge two separate action verbs into one node and NEVER omit the final save/export/close/vent/park/quit step). Each node has:
  - "id": "n1", "n2", ...
  - "intent": short neutral description of that single operation
  - "domain_hint": matching name from Loaded domains (or explicitly requested library name if not loaded, or null if uncertain)
  - "inputs": list of ONLY earlier node "outputs" names or "assets" keys used by this step (every "assets" key MUST be referenced in "inputs")
  - "outputs": list of 1 short output variable name produced by this step (fresh name distinct from inputs)
  - "params": object containing EVERY explicitly stated number, quoted tag/string, or tuple from that clause (omit unit words like nm/mK/mW/GHz/ns; NEVER invent unstated parameters)
- "deliverable": final output variable name (or null)
- "ambiguities": list of ambiguity reasons (or [])"""

SLM_FEW_SHOT_MESSAGES: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": 'load "raw.dat", transform with factor 2.5 and offset 1, and compute metric',
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "assets": {"src": "raw.dat"},
                "required_domains": [],
                "nodes": [
                    {"id": "n1", "intent": "load data from file", "domain_hint": None, "inputs": ["src"], "outputs": ["d1"], "params": {}},
                    {"id": "n2", "intent": "transform values", "domain_hint": None, "inputs": ["d1"], "outputs": ["d2"], "params": {"factor": 2.5, "offset": 1}},
                    {"id": "n3", "intent": "compute metric", "domain_hint": None, "inputs": ["d2"], "outputs": ["m1"], "params": {}},
                ],
                "deliverable": "m1",
                "ambiguities": [],
            }
        ),
    },
    {
        "role": "user",
        "content": 'using mypkg, read "item.bin" tag "A1" at 4.2, convert format, and save to "out.bin"',
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "assets": {"src": "item.bin", "dst": "out.bin"},
                "required_domains": ["mypkg"],
                "nodes": [
                    {"id": "n1", "intent": "read item file", "domain_hint": "mypkg", "inputs": ["src"], "outputs": ["obj1"], "params": {"tag": "A1", "level": 4.2}},
                    {"id": "n2", "intent": "convert format", "domain_hint": "mypkg", "inputs": ["obj1"], "outputs": ["obj2"], "params": {}},
                    {"id": "n3", "intent": "save item to file", "domain_hint": "mypkg", "inputs": ["obj2", "dst"], "outputs": ["res1"], "params": {}},
                ],
                "deliverable": "res1",
                "ambiguities": [],
            }
        ),
    },
]


@dataclass
class FrontendCompileResult:
    ok: bool
    ir: Optional[TypedIR]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _is_grounded_in_prompt(value: Any, raw_prompt: str) -> bool:
    if value is None or not raw_prompt:
        return False
    prompt_low = raw_prompt.lower()
    if isinstance(value, bool):
        return ("true" if value else "false") in prompt_low
    if isinstance(value, (int, float)):
        return str(value).lower() in prompt_low
    if isinstance(value, str):
        v_str = value.strip()
        return bool(v_str) and (v_str.lower() in prompt_low)
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_grounded_in_prompt(item, raw_prompt) for item in value)
    return False


def _is_explicitly_quoted(value: str, raw_prompt: str) -> bool:
    v_low = value.strip().lower()
    p_low = raw_prompt.lower()
    return bool(v_low) and (f'"{v_low}"' in p_low or f"'{v_low}'" in p_low)


def _is_concrete_external_asset(value: Any, raw_prompt: str) -> bool:
    """Return True iff `value` is a concrete file path/URI with separator/extension grounded in `raw_prompt`."""
    if not isinstance(value, str) or not _is_grounded_in_prompt(value, raw_prompt):
        return False
    s = value.strip()
    if not s or s.replace(".", "", 1).replace("-", "", 1).isdigit():
        return False
    return any(sep in s for sep in (".", "/", "\\", "://", ":")) and not any(ch.isspace() for ch in s)


def _looks_like_untagged_path_param(val: Any, raw_prompt: str) -> bool:
    """Return True iff flat string `val` in node params is an unquoted/path-shaped external asset in `raw_prompt`."""
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s or any(ch.isspace() for ch in s):
        return False
    if s.replace(".", "", 1).replace("-", "", 1).isdigit():
        return False
    return any(sep in s for sep in (".", "/", "\\", "://")) and _is_grounded_in_prompt(s, raw_prompt)


def _ssa_rename_nodes(
    nodes: List[IRNode],
    initial_defined: Set[str],
    deliverable: Optional[str],
) -> Tuple[List[IRNode], Optional[str]]:
    """Alpha-rename shadowed sequential output variables so each IR symbol is strictly SSA-unique."""

    def _rename_block(
        node_list: List[IRNode],
        block_defined: Set[str],
        block_remap: Dict[str, str],
    ) -> Tuple[List[IRNode], Set[str], Dict[str, str]]:
        cur_defined = set(block_defined)
        cur_remap = dict(block_remap)
        out_nodes: List[IRNode] = []

        for node in node_list:
            new_inputs = [cur_remap.get(inp, inp) for inp in node.inputs]
            mapped_in_set = set(new_inputs)
            pre_defined = set(cur_defined)
            else_remap = dict(cur_remap)

            if node.kind in ("loop", "branch"):
                block_scope_def = pre_defined | {node.id}
                new_body = node.body
                if node.body:
                    body_def = set(block_scope_def)
                    if node.kind == "loop":
                        cond = str(node.condition or "").strip()
                        for sep in (" in ", " of "):
                            if sep in cond:
                                lhs = cond.split(sep, 1)[0].strip()
                                if lhs.startswith("for "):
                                    lhs = lhs[4:].strip()
                                if lhs.isidentifier():
                                    body_def.add(lhs)
                                break
                    new_body, _, _ = _rename_block(node.body, body_def, cur_remap)

                new_else = node.else_body
                if node.else_body:
                    new_else, _, _ = _rename_block(node.else_body, block_scope_def, else_remap)

                for out_sym in node.outputs:
                    cur_defined.add(out_sym)
                    cur_remap.setdefault(out_sym, out_sym)
                cur_defined.add(node.id)
                out_nodes.append(
                    node.model_copy(
                        update={
                            "inputs": new_inputs,
                            "body": new_body,
                            "else_body": new_else,
                        }
                    )
                )
                continue

            new_outputs: List[str] = []
            pending_remaps: Dict[str, str] = {}
            for out_sym in node.outputs:
                if out_sym in cur_defined or out_sym in mapped_in_set or out_sym == node.id:
                    cand = f"{out_sym}_{node.id}"
                    suffix = 2
                    while cand in cur_defined or cand in mapped_in_set or cand == node.id:
                        cand = f"{out_sym}_{node.id}_{suffix}"
                        suffix += 1
                    fresh = cand
                else:
                    fresh = out_sym
                new_outputs.append(fresh)
                cur_defined.add(fresh)
                pending_remaps[out_sym] = fresh

            cur_defined.add(node.id)
            cur_remap.update(pending_remaps)
            out_nodes.append(
                node.model_copy(
                    update={
                        "inputs": new_inputs,
                        "outputs": new_outputs,
                    }
                )
            )

        return out_nodes, cur_defined, cur_remap

    renamed_nodes, _, top_remap = _rename_block(nodes, set(initial_defined), {})
    new_deliv = top_remap.get(deliverable, deliverable) if deliverable is not None else None
    return renamed_nodes, new_deliv


def _normalize_param(
    raw_val: Any,
    raw_prompt: str,
    param_key: str,
    warnings: List[str],
    force_class: Optional[str] = None,
) -> IRParamValue:
    valid_classes = ("literal", "external_asset", "identifier", "enum")
    valid_provs = ("prompt", "estimated", "default", "missing")

    if isinstance(raw_val, dict) and any(k in raw_val for k in ("value", "provenance", "value_class")):
        raw_cls = str(raw_val.get("value_class") or force_class or "literal").strip().lower()
        v_cls = raw_cls if raw_cls in valid_classes else "literal"
        if force_class in valid_classes:
            v_cls = force_class
        raw_prov = str(raw_val.get("provenance") or "").strip().lower()
        val = raw_val.get("value")
    else:
        v_cls = force_class if (force_class in valid_classes) else "literal"
        raw_prov = ""
        val = raw_val

    if v_cls == "external_asset":
        if raw_prov == "missing" or val is None or (isinstance(val, str) and not val.strip()):
            return IRParamValue(value=None, provenance="missing", value_class="external_asset")
        if not _is_concrete_external_asset(val, raw_prompt):
            warnings.append(f"Downgraded ungrounded external_asset '{param_key}'={val!r} to missing")
            return IRParamValue(value=None, provenance="missing", value_class="external_asset")
        return IRParamValue(value=str(val).strip(), provenance="prompt", value_class="external_asset")

    if raw_prov == "missing" or val is None:
        return IRParamValue(value=None, provenance="missing", value_class=v_cls)

    if v_cls == "literal" and isinstance(val, str) and not _is_explicitly_quoted(val, raw_prompt):
        s_num = val.strip()
        sign_body = s_num[1:] if s_num[:1] in "+-" else s_num
        if sign_body and sign_body.isdigit():
            val = int(s_num)
        elif (
            sign_body.count(".") == 1
            and sign_body.replace(".", "", 1).isdigit()
            and any(ch.isdigit() for ch in sign_body)
        ):
            val = float(s_num)

    if raw_prov in valid_provs:
        prov = raw_prov
    else:
        prov = "prompt" if _is_grounded_in_prompt(val, raw_prompt) else "estimated"

    return IRParamValue(value=val, provenance=prov, value_class=v_cls)


def _sanitize_output_ident(raw_out: str, fallback: str) -> str:
    """Coerce an arbitrary string into a valid non-empty Python-safe identifier."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(raw_out or "")).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned if cleaned.isidentifier() else fallback


def _build_ir_node(
    raw_node: Dict[str, Any],
    idx_hint: str,
    raw_prompt: str,
    assets_out: Dict[str, IRParamValue],
    warnings: List[str],
    grounded_val_to_key: Optional[Dict[str, str]] = None,
    output_remap: Optional[Dict[str, str]] = None,
) -> IRNode:
    val_to_key = grounded_val_to_key if grounded_val_to_key is not None else {}
    node_id = str(raw_node.get("id") or f"n{idx_hint}").strip() or f"n{idx_hint}"
    raw_kind = str(raw_node.get("kind") or "call").strip().lower()
    kind = raw_kind if raw_kind in ("call", "loop", "branch") else "call"
    intent = str(raw_node.get("intent") or raw_node.get("op") or "").strip()

    domain_hint_raw = raw_node.get("domain_hint")
    domain_hint = str(domain_hint_raw).strip() if isinstance(domain_hint_raw, str) and domain_hint_raw.strip() else None

    raw_inputs = raw_node.get("inputs") or []
    inputs: List[str] = []
    if isinstance(raw_inputs, list):
        for x in raw_inputs:
            if not isinstance(x, str):
                continue
            xs = x.strip()
            if not xs:
                continue
            if xs not in assets_out and xs in val_to_key:
                xs = val_to_key[xs]
            elif output_remap and xs in output_remap:
                xs = output_remap[xs]
            inputs.append(xs)

    raw_outputs = raw_node.get("outputs") or []
    outputs: List[str] = []
    if isinstance(raw_outputs, list):
        for out_i, raw_out in enumerate(raw_outputs, start=1):
            if not isinstance(raw_out, str):
                continue
            o = raw_out.strip()
            if not o:
                continue
            fallback_sym = f"{node_id}_out" if out_i == 1 and f"{node_id}_out" not in outputs else f"{node_id}_out_{out_i}"
            if o in val_to_key:
                a_key = val_to_key[o]
                if a_key not in inputs:
                    inputs.append(a_key)
                safe_out = fallback_sym
                if output_remap is not None:
                    output_remap[o] = safe_out
                warnings.append(f"Replaced asset output name {o!r} on node '{node_id}' with '{safe_out}'.")
                outputs.append(safe_out)
            elif _looks_like_untagged_path_param(o, raw_prompt):
                asset_key = f"{node_id}.out_path" if out_i == 1 else f"{node_id}.out_path_{out_i}"
                assets_out[asset_key] = IRParamValue(value=o, provenance="prompt", value_class="external_asset")
                val_to_key[o] = asset_key
                if asset_key not in inputs:
                    inputs.append(asset_key)
                safe_out = fallback_sym
                if output_remap is not None:
                    output_remap[o] = safe_out
                warnings.append(f"Registered untagged path output {o!r} on node '{node_id}' as input asset '{asset_key}'.")
                outputs.append(safe_out)
            elif not o.isidentifier():
                safe_out = _sanitize_output_ident(o, fallback_sym)
                if output_remap is not None:
                    output_remap[o] = safe_out
                warnings.append(f"Sanitized invalid output identifier {o!r} on node '{node_id}' to '{safe_out}'.")
                outputs.append(safe_out)
            else:
                outputs.append(o)

    params: Dict[str, IRParamValue] = {}
    raw_params = raw_node.get("params") or {}
    if isinstance(raw_params, dict):
        if (
            ("provenance" in raw_params or "value_class" in raw_params)
            and set(raw_params.keys()) <= {"value", "provenance", "value_class"}
        ):
            warnings.append(f"Ignored unkeyed parameter envelope in node '{node_id}'.")
            raw_params = {}
        for pk, pv in raw_params.items():
            pk_str = str(pk).strip()
            if not pk_str:
                continue
            if isinstance(pv, str):
                pv_s = pv.strip()
                if pv_s in assets_out:
                    p_norm = assets_out[pv_s]
                elif pv_s in val_to_key or _looks_like_untagged_path_param(pv_s, raw_prompt):
                    p_norm = _normalize_param(pv_s, raw_prompt, pk_str, warnings, force_class="external_asset")
                else:
                    p_norm = _normalize_param(pv, raw_prompt, pk_str, warnings)
            else:
                p_norm = _normalize_param(pv, raw_prompt, pk_str, warnings)
            params[pk_str] = p_norm
            if p_norm.value_class == "external_asset":
                asset_key = f"{node_id}.{pk_str}"
                if p_norm.provenance == "prompt" and p_norm.value is not None:
                    v_clean = str(p_norm.value).strip()
                    if v_clean not in val_to_key and asset_key not in assets_out:
                        assets_out[asset_key] = p_norm
                        val_to_key[v_clean] = asset_key
                elif asset_key not in assets_out:
                    assets_out[asset_key] = p_norm

    cond_raw = raw_node.get("condition")
    condition = str(cond_raw).strip() if isinstance(cond_raw, str) and cond_raw.strip() else None

    body_nodes: List[IRNode] = []
    for bi, b_raw in enumerate(raw_node.get("body") or (), start=1):
        if isinstance(b_raw, dict):
            body_nodes.append(_build_ir_node(b_raw, f"{idx_hint}_b{bi}", raw_prompt, assets_out, warnings, val_to_key, output_remap))

    else_nodes: List[IRNode] = []
    for ei, e_raw in enumerate(raw_node.get("else_body") or (), start=1):
        if isinstance(e_raw, dict):
            else_nodes.append(_build_ir_node(e_raw, f"{idx_hint}_e{ei}", raw_prompt, assets_out, warnings, val_to_key, output_remap))

    return IRNode(
        id=node_id,
        kind=kind,
        intent=intent,
        domain_hint=domain_hint,
        inputs=inputs,
        outputs=outputs,
        params=params,
        condition=condition,
        body=body_nodes,
        else_body=else_nodes,
    )


def _demote_quoted_non_path_inputs(
    nodes: List[IRNode],
    demoted_literals: Dict[str, str],
    warnings: List[str],
) -> List[IRNode]:
    """Convert non-path quoted ID/tag keys referenced in node.inputs into prompt literal params."""
    if not demoted_literals:
        return nodes
    out: List[IRNode] = []
    for node in nodes:
        new_body = _demote_quoted_non_path_inputs(node.body, demoted_literals, warnings) if node.body else []
        new_else = _demote_quoted_non_path_inputs(node.else_body, demoted_literals, warnings) if node.else_body else []
        new_inputs: List[str] = []
        new_params = dict(node.params)
        existing_vals = {str(pv.value).strip() for pv in new_params.values() if pv.value is not None}
        changed = bool(node.body or node.else_body)
        for inp in node.inputs:
            if inp in demoted_literals:
                lit_val = demoted_literals[inp]
                if inp not in new_params and lit_val not in existing_vals:
                    new_params[inp] = IRParamValue(value=lit_val, provenance="prompt", value_class="literal")
                    existing_vals.add(lit_val)
                warnings.append(f"Demoted non-path quoted asset key '{inp}'={lit_val!r} on node '{node.id}' to literal.")
                changed = True
            else:
                new_inputs.append(inp)
        if changed:
            out.append(node.model_copy(update={"inputs": new_inputs, "params": new_params, "body": new_body, "else_body": new_else}))
        else:
            out.append(node)
    return out


def _collect_defined_symbols(nodes: List[IRNode], seed_syms: Set[str]) -> Set[str]:
    syms = set(seed_syms)
    for node in nodes:
        if node.id:
            syms.add(node.id)
        for o in node.outputs:
            if o:
                syms.add(o)
        if node.kind == "loop":
            lv = infer_loop_var(node)
            if lv:
                syms.add(lv)
        if node.body:
            syms.update(_collect_defined_symbols(node.body, set()))
        if node.else_body:
            syms.update(_collect_defined_symbols(node.else_body, set()))
    return syms


def _sanitize_undefined_slm_inputs(
    nodes: List[IRNode],
    all_defined: Set[str],
    warnings: List[str],
) -> List[IRNode]:
    """Drop SLM-hallucinated input names absent from all defined assets/nodes/outputs while keeping cycles intact."""
    out: List[IRNode] = []
    for node in nodes:
        new_body = _sanitize_undefined_slm_inputs(node.body, all_defined, warnings) if node.body else []
        new_else = _sanitize_undefined_slm_inputs(node.else_body, all_defined, warnings) if node.else_body else []
        new_inputs: List[str] = []
        changed = bool(node.body or node.else_body)
        for inp in node.inputs:
            if inp not in all_defined:
                warnings.append(f"Dropped undefined SLM input label '{inp}' on node '{node.id}'.")
                changed = True
            else:
                new_inputs.append(inp)
        if changed:
            out.append(node.model_copy(update={"inputs": new_inputs, "body": new_body, "else_body": new_else}))
        else:
            out.append(node)
    return out


def parse_slm_json_to_typed_ir(raw_text: str, raw_prompt: str) -> FrontendCompileResult:
    """Parse raw JSON output returned by an SLM into a validated TypedIR."""
    warnings: List[str] = []
    data = extract_json_object(str(raw_text or ""))
    if not isinstance(data, dict):
        return FrontendCompileResult(ok=False, ir=None, error="SLM output did not contain a valid JSON object.")

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list):
        return FrontendCompileResult(ok=False, ir=None, error="SLM JSON output missing required 'nodes' list.")

    assets_out: Dict[str, IRParamValue] = {}
    demoted_literals: Dict[str, str] = {}
    raw_assets = data.get("assets") or {}
    if isinstance(raw_assets, dict):
        for ak, av in raw_assets.items():
            ak_str = str(ak).strip()
            if not ak_str:
                continue
            raw_val = av.get("value") if isinstance(av, dict) else av
            if (
                isinstance(raw_val, str)
                and not _is_concrete_external_asset(raw_val, raw_prompt)
                and _is_grounded_in_prompt(raw_val, raw_prompt)
                and _is_explicitly_quoted(raw_val, raw_prompt)
            ):
                demoted_literals[ak_str] = raw_val.strip()
                continue
            assets_out[ak_str] = _normalize_param(
                av, raw_prompt, ak_str, warnings, force_class="external_asset"
            )

    grounded_val_to_key: Dict[str, str] = {}
    for ak, av in assets_out.items():
        if av.provenance == "prompt" and av.value is not None:
            grounded_val_to_key.setdefault(str(av.value).strip(), ak)

    output_remap: Dict[str, str] = {}
    nodes: List[IRNode] = []
    for idx, rn in enumerate(raw_nodes, start=1):
        if not isinstance(rn, dict):
            return FrontendCompileResult(ok=False, ir=None, error=f"Node entry at index {idx - 1} is not a JSON object.")
        node_obj = _build_ir_node(rn, str(idx), raw_prompt, assets_out, warnings, grounded_val_to_key, output_remap)
        if not node_obj.intent:
            return FrontendCompileResult(ok=False, ir=None, error=f"IRNode '{node_obj.id}' has empty intent.")
        nodes.append(node_obj)

    nodes = _demote_quoted_non_path_inputs(nodes, demoted_literals, warnings)

    raw_deliv = data.get("deliverable")
    deliverable = str(raw_deliv).strip() if isinstance(raw_deliv, str) and raw_deliv.strip() else None
    if deliverable is not None:
        deliverable = output_remap.get(deliverable, deliverable)

    nodes, deliverable = _ssa_rename_nodes(nodes, set(assets_out.keys()), deliverable)
    all_defined_syms = _collect_defined_symbols(nodes, set(assets_out.keys()))
    nodes = _sanitize_undefined_slm_inputs(nodes, all_defined_syms, warnings)

    raw_amb = data.get("ambiguities") or []
    ambiguities = [str(a).strip() for a in raw_amb if isinstance(a, str) and str(a).strip()] if isinstance(raw_amb, list) else []

    raw_req = data.get("required_domains") or []
    req_doms = (
        list(
            dict.fromkeys(
                str(r).strip().lower()
                for r in raw_req
                if isinstance(r, str) and str(r).strip()
            )
        )
        if isinstance(raw_req, list)
        else []
    )

    try:
        ir = TypedIR(
            source_mode="slm_compiled",
            raw_prompt=raw_prompt,
            nodes=nodes,
            assets=assets_out,
            required_domains=req_doms,
            deliverable=deliverable,
            ambiguities=ambiguities,
        )
    except Exception as exc:
        return FrontendCompileResult(ok=False, ir=None, error=f"TypedIR schema validation failed: {exc}", warnings=warnings)

    dag_errs = ir.validate_dag()
    if dag_errs:
        return FrontendCompileResult(ok=False, ir=None, error="; ".join(dag_errs), warnings=warnings)

    return FrontendCompileResult(ok=True, ir=ir, error=None, warnings=warnings)


def compile_slm_typed_ir(
    prompt: str,
    generate_fn: Callable[..., str],
    loaded_domains: Iterable[str] = (),
) -> FrontendCompileResult:
    """Invoke `generate_fn(prompt, schema, system_prompt)` and compile the response into TypedIR."""
    text = str(prompt or "").strip()
    if not text:
        return FrontendCompileResult(ok=False, ir=None, error="Empty prompt.")
    doms = sorted({str(d).strip() for d in loaded_domains if str(d).strip()})
    dom_str = ", ".join(doms) if doms else "(infer purely from neutral API operations)"
    sys_prompt = SLM_SYSTEM_PROMPT_TEMPLATE.format(loaded_domains=dom_str)
    accepts_few_shot = False
    try:
        sig = inspect.signature(generate_fn)
        accepts_few_shot = "few_shot" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_few_shot = False
    try:
        if accepts_few_shot:
            raw_resp = generate_fn(text, SLM_IR_JSON_SCHEMA, sys_prompt, few_shot=list(SLM_FEW_SHOT_MESSAGES))
        else:
            raw_resp = generate_fn(text, SLM_IR_JSON_SCHEMA, sys_prompt)
    except Exception as exc:
        return FrontendCompileResult(ok=False, ir=None, error=f"SLM generation failed: {exc}")
    return parse_slm_json_to_typed_ir(raw_resp, text)


__all__ = [
    "SLM_IR_JSON_SCHEMA",
    "SLM_SYSTEM_PROMPT_TEMPLATE",
    "SLM_FEW_SHOT_MESSAGES",
    "FrontendCompileResult",
    "parse_slm_json_to_typed_ir",
    "compile_slm_typed_ir",
]
