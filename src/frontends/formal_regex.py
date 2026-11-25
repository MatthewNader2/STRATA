"""
src/frontends/formal_regex.py — Strict formal-syntax regex frontend for model-free profiles.

Quarantined as the ONLY module under src/ allowed to use `re`. Accepts deterministic
formal pipeline or assignment-DAG syntax and cleanly rejects free-form natural language.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

try:
    from ..typed_ir import IRNode, IRParamValue, TypedIR
except (ImportError, ValueError):
    from typed_ir import IRNode, IRParamValue, TypedIR


_ASSIGN_RE = re.compile(
    r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*=\s*(?P<call>.+?)\s*$",
    re.DOTALL,
)
_CALL_RE = re.compile(
    r"^\s*(?:(?P<domain>[A-Za-z_][A-Za-z0-9_]*)::)?(?:(?P<quote>[\"'])(?P<quoted_op>.+?)(?P=quote)|(?P<op>[A-Za-z_][A-Za-z0-9_.]*))\s*\((?P<args>.*)\)\s*$",
    re.DOTALL,
)
_KV_RE = re.compile(r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<val>.+?)\s*$", re.DOTALL)
_VAR_REF_RE = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
_ASSET_QUOTED_RE = re.compile(r"^@([\"'])(?P<body>.*)\1$", re.DOTALL)
_QUOTED_STR_RE = re.compile(r"^([\"'])(?P<body>.*)\1$", re.DOTALL)
_BOOL_RE = re.compile(r"^(?:true|false)$", re.IGNORECASE)
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class FormalParseResult:
    ok: bool
    ir: Optional[TypedIR]
    error: Optional[str] = None


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_brack = 0
    quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(text)
    sep_len = len(sep)

    while i < n:
        ch = text[i]
        if quote is not None:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth_paren -= 1
            if depth_paren < 0:
                raise ValueError("Unbalanced closing parenthesis ')'.")
            buf.append(ch)
            i += 1
            continue
        if ch == "[":
            depth_brack += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            depth_brack -= 1
            if depth_brack < 0:
                raise ValueError("Unbalanced closing bracket ']'.")
            buf.append(ch)
            i += 1
            continue

        if depth_paren == 0 and depth_brack == 0 and text.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += sep_len
            continue

        buf.append(ch)
        i += 1

    if quote is not None:
        raise ValueError("Unterminated string literal.")
    if depth_paren != 0 or depth_brack != 0:
        raise ValueError("Unbalanced delimiter in formal expression.")
    parts.append("".join(buf))
    return parts


def _split_statements(prompt: str) -> List[str]:
    raw_lines: List[str] = []
    for semi_part in _split_top_level(prompt, ";"):
        for line_part in _split_top_level(semi_part, "\n"):
            stripped = line_part.strip()
            if stripped:
                raw_lines.append(stripped)
    return raw_lines


def _parse_scalar_literal(raw_val: str):
    v = raw_val.strip()
    m_q = _QUOTED_STR_RE.match(v)
    if m_q:
        return m_q.group("body"), "literal"
    if _BOOL_RE.match(v):
        return v.lower() == "true", "literal"
    if _INT_RE.match(v):
        return int(v), "literal"
    if _FLOAT_RE.match(v):
        return float(v), "literal"
    if v.startswith("(") and v.endswith(")"):
        inner = v[1:-1].strip()
        if not inner:
            return (), "literal"
        items = [_parse_scalar_literal(item)[0] for item in _split_top_level(inner, ",") if item.strip()]
        return tuple(items), "literal"
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return [], "literal"
        items = [_parse_scalar_literal(item)[0] for item in _split_top_level(inner, ",") if item.strip()]
        return list(items), "literal"
    if _IDENT_RE.match(v):
        return v, "identifier"
    raise ValueError(f"Unsupported formal argument value: {v!r}")


def _parse_call_args(
    args_str: str,
    node_id: str,
    assets: Dict[str, IRParamValue],
) -> Tuple[List[str], Dict[str, IRParamValue]]:
    inputs: List[str] = []
    params: Dict[str, IRParamValue] = {}
    if not args_str.strip():
        return inputs, params

    for item in _split_top_level(args_str, ","):
        item_s = item.strip()
        if not item_s:
            raise ValueError(f"Empty argument slot in node '{node_id}'.")
        m_kv = _KV_RE.match(item_s)
        if not m_kv:
            raise ValueError(f"Expected 'key=value' argument in node '{node_id}', got {item_s!r}.")
        key = m_kv.group("key")
        val = m_kv.group("val").strip()
        if key in params:
            raise ValueError(f"Duplicate argument '{key}' in node '{node_id}'.")

        m_var = _VAR_REF_RE.match(val)
        if m_var:
            vref = m_var.group("name")
            if vref not in inputs:
                inputs.append(vref)
            continue

        m_asset = _ASSET_QUOTED_RE.match(val)
        if m_asset:
            param_val = IRParamValue(
                value=m_asset.group("body"),
                provenance="prompt",
                value_class="external_asset",
            )
            params[key] = param_val
            assets[f"{node_id}.{key}"] = param_val
            continue

        lit_val, v_cls = _parse_scalar_literal(val)
        params[key] = IRParamValue(
            value=lit_val,
            provenance="prompt",
            value_class=v_cls,
        )

    return inputs, params


def parse_formal_prompt(prompt: str) -> FormalParseResult:
    """
    Parse structured pipe (`A(...) |> B(...)`) or assignment-DAG (`x = A(...); y = B(in=$x)`) syntax
    into TypedIR. Rejects unconstrained natural-language prose.
    """
    text = str(prompt or "").strip()
    if not text:
        return FormalParseResult(ok=False, ir=None, error="Empty formal specification.")

    try:
        pipe_segments = [s.strip() for s in _split_top_level(text, "|>")]
        nodes: List[IRNode] = []
        assets: Dict[str, IRParamValue] = {}

        if len(pipe_segments) > 1:
            if any(not seg for seg in pipe_segments):
                return FormalParseResult(ok=False, ir=None, error="Empty stage around '|>' pipe operator.")
            prev_out: Optional[str] = None
            for idx, seg in enumerate(pipe_segments, start=1):
                if _ASSIGN_RE.match(seg):
                    return FormalParseResult(
                        ok=False,
                        ir=None,
                        error="Mixed assignment '=' and pipe '|>' syntax is not supported.",
                    )
                m_call = _CALL_RE.match(seg)
                op_name = ((m_call.group("quoted_op") or m_call.group("op") or "").strip()) if m_call else ""
                if not m_call or not op_name:
                    return FormalParseResult(
                        ok=False,
                        ir=None,
                        error=f"Invalid formal call syntax at pipe stage {idx}: {seg!r}.",
                    )
                node_id = f"n{idx}"
                out_sym = f"v{idx}"
                inputs, params = _parse_call_args(m_call.group("args"), node_id, assets)
                if prev_out is not None and prev_out not in inputs:
                    inputs.insert(0, prev_out)
                nodes.append(
                    IRNode(
                        id=node_id,
                        kind="call",
                        intent=op_name,
                        domain_hint=m_call.group("domain"),
                        inputs=inputs,
                        outputs=[out_sym],
                        params=params,
                    )
                )
                prev_out = out_sym
        else:
            stmts = _split_statements(text)
            if not stmts:
                return FormalParseResult(ok=False, ir=None, error="No valid formal statements found.")
            seen_vars = set()
            for idx, stmt in enumerate(stmts, start=1):
                m_assign = _ASSIGN_RE.match(stmt)
                if m_assign:
                    lhs_vars = [s.strip() for s in m_assign.group("lhs").split(",") if s.strip()]
                    if not lhs_vars:
                        return FormalParseResult(
                            ok=False,
                            ir=None,
                            error=f"Empty LHS assignment in {stmt!r}.",
                        )
                    lhs_seen: set[str] = set()
                    for sym in lhs_vars:
                        if sym in lhs_seen or sym in seen_vars:
                            return FormalParseResult(
                                ok=False,
                                ir=None,
                                error=f"Duplicate variable assignment '{sym}'.",
                            )
                        lhs_seen.add(sym)
                    seen_vars.update(lhs_vars)
                    call_text = m_assign.group("call")
                    node_id = lhs_vars[0]
                else:
                    lhs_vars = [f"v{idx}"]
                    call_text = stmt
                    node_id = f"n{idx}"

                m_call = _CALL_RE.match(call_text)
                op_name = ((m_call.group("quoted_op") or m_call.group("op") or "").strip()) if m_call else ""
                if not m_call or not op_name:
                    return FormalParseResult(
                        ok=False,
                        ir=None,
                        error=f"Invalid formal statement syntax: {stmt!r}.",
                    )
                inputs, params = _parse_call_args(m_call.group("args"), node_id, assets)
                nodes.append(
                    IRNode(
                        id=node_id,
                        kind="call",
                        intent=op_name,
                        domain_hint=m_call.group("domain"),
                        inputs=inputs,
                        outputs=lhs_vars,
                        params=params,
                    )
                )

        deliverable = nodes[-1].outputs[0] if (nodes and nodes[-1].outputs) else None
        ir = TypedIR(
            source_mode="formal_syntax",
            raw_prompt=prompt,
            nodes=nodes,
            assets=assets,
            deliverable=deliverable,
        )
        dag_errs = ir.validate_dag()
        if dag_errs:
            return FormalParseResult(ok=False, ir=None, error="; ".join(dag_errs))
        return FormalParseResult(ok=True, ir=ir, error=None)
    except Exception as exc:
        return FormalParseResult(ok=False, ir=None, error=str(exc))


__all__ = ["FormalParseResult", "parse_formal_prompt"]
