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
    from ..typed_ir import IRNode, IRParamValue, TypedIR
    from ..utils import extract_json_object
except (ImportError, ValueError):
    from typed_ir import IRNode, IRParamValue, TypedIR
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
        "nodes": {"type": "array", "minItems": 1, "maxItems": 16, "items": _NODE_JSON_SCHEMA},
        "deliverable": {"type": ["string", "null"]},
        "ambiguities": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
    },
}

SLM_SYSTEM_PROMPT_TEMPLATE = """\
Convert every action in the user request into an ordered JSON pipeline of atomic steps.
Loaded domains (use ONLY names from this list or null for domain_hint): {loaded_domains}

Output ONLY one valid JSON object with:
- "assets": object mapping short key to concrete file path or URI verbatim from the user text (include both input and output save paths; NEVER invent missing paths or put generic nouns like "file", "data", "media" or in-memory keywords in assets; use {{}} if no concrete path/URI appears).
- "nodes": list with ONE node per explicitly requested action in order (do NOT invent extra unrequested steps; keep a single clause like "run/read X into Y" as ONE atomic node whose intent includes both X and Y rather than splitting it into two steps). Each node has:
  - "id": "n1", "n2", ...
  - "intent": short neutral description of that single operation
  - "domain_hint": matching name from Loaded domains (or null if uncertain or cross-domain)
  - "inputs": list of earlier node "outputs" names or "assets" keys used by this step
  - "outputs": list of 1 short output variable name produced by this step (must be a fresh name distinct from inputs)
  - "params": object of ONLY explicitly stated parameter values from the user text (omit unstated parameters)
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


def _is_concrete_external_asset(value: Any, raw_prompt: str) -> bool:
    """Return True iff `value` is a concrete file path/URI/quoted token grounded in `raw_prompt`."""
    if not isinstance(value, str) or not _is_grounded_in_prompt(value, raw_prompt):
        return False
    s = value.strip()
    if not s or s.replace(".", "", 1).replace("-", "", 1).isdigit():
        return False
    prompt_low = raw_prompt.lower()
    s_low = s.lower()
    if f'"{s_low}"' in prompt_low or f"'{s_low}'" in prompt_low:
        return True
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

    if raw_prov in valid_provs:
        prov = raw_prov
    else:
        prov = "prompt" if _is_grounded_in_prompt(val, raw_prompt) else "estimated"

    return IRParamValue(value=val, provenance=prov, value_class=v_cls)


def _build_ir_node(
    raw_node: Dict[str, Any],
    idx_hint: str,
    raw_prompt: str,
    assets_out: Dict[str, IRParamValue],
    warnings: List[str],
    grounded_val_to_key: Optional[Dict[str, str]] = None,
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
            inputs.append(xs)

    raw_outputs = raw_node.get("outputs") or []
    outputs = [str(x).strip() for x in raw_outputs if isinstance(x, str) and str(x).strip()] if isinstance(raw_outputs, list) else []

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
            body_nodes.append(_build_ir_node(b_raw, f"{idx_hint}_b{bi}", raw_prompt, assets_out, warnings, val_to_key))

    else_nodes: List[IRNode] = []
    for ei, e_raw in enumerate(raw_node.get("else_body") or (), start=1):
        if isinstance(e_raw, dict):
            else_nodes.append(_build_ir_node(e_raw, f"{idx_hint}_e{ei}", raw_prompt, assets_out, warnings, val_to_key))

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
    raw_assets = data.get("assets") or {}
    if isinstance(raw_assets, dict):
        for ak, av in raw_assets.items():
            ak_str = str(ak).strip()
            if ak_str:
                assets_out[ak_str] = _normalize_param(
                    av, raw_prompt, ak_str, warnings, force_class="external_asset"
                )

    grounded_val_to_key: Dict[str, str] = {}
    for ak, av in assets_out.items():
        if av.provenance == "prompt" and av.value is not None:
            grounded_val_to_key.setdefault(str(av.value).strip(), ak)

    nodes: List[IRNode] = []
    for idx, rn in enumerate(raw_nodes, start=1):
        if not isinstance(rn, dict):
            return FrontendCompileResult(ok=False, ir=None, error=f"Node entry at index {idx - 1} is not a JSON object.")
        node_obj = _build_ir_node(rn, str(idx), raw_prompt, assets_out, warnings, grounded_val_to_key)
        if not node_obj.intent:
            return FrontendCompileResult(ok=False, ir=None, error=f"IRNode '{node_obj.id}' has empty intent.")
        nodes.append(node_obj)

    raw_deliv = data.get("deliverable")
    deliverable = str(raw_deliv).strip() if isinstance(raw_deliv, str) and raw_deliv.strip() else None

    nodes, deliverable = _ssa_rename_nodes(nodes, set(assets_out.keys()), deliverable)

    raw_amb = data.get("ambiguities") or []
    ambiguities = [str(a).strip() for a in raw_amb if isinstance(a, str) and str(a).strip()] if isinstance(raw_amb, list) else []

    try:
        ir = TypedIR(
            source_mode="slm_compiled",
            raw_prompt=raw_prompt,
            nodes=nodes,
            assets=assets_out,
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
