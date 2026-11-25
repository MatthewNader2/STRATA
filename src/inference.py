"""
src/inference.py — STRATA Unified Multi-Profile Inference Runtime.

Manages lazy neural model discovery and execution (SentenceTransformer embedders,
GGUF Llama models, and first-token logprob choice scoring) driven strictly by
ProfileSpec and NSTLSettings, with zero hardcoded model names or legacy profile classes.
"""
from __future__ import annotations

import gc
import inspect
import json
import math
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

try:
    from .config import MODELS_DIR, ProfileSpec, resolve_profile_spec, settings
    from .hardware import HardwareProfiler
    from .log_config import get_logger
    from .utils import extract_json_object, is_split_gguf_shard
except (ImportError, ValueError):
    from config import MODELS_DIR, ProfileSpec, resolve_profile_spec, settings
    from hardware import HardwareProfiler
    from log_config import get_logger
    from utils import extract_json_object, is_split_gguf_shard

logger = get_logger("inference")

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch = None  # type: ignore[assignment]
TORCH_AVAILABLE: Optional[bool] = None


def _ensure_torch() -> bool:
    """Lazily import PyTorch once on first neural inference use."""
    global torch, TORCH_AVAILABLE
    if TORCH_AVAILABLE is not None:
        return TORCH_AVAILABLE
    try:
        import torch as _torch
        torch = _torch
        TORCH_AVAILABLE = True
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]
        TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


def _trim_host_memory() -> None:
    """Return freed glibc heap arenas back to the OS on Linux."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def _instantiate_llama(llama_cls: Any, **kwargs: Any) -> Any:
    """Instantiate Llama with flash_attn=True when supported, falling back cleanly."""
    if "flash_attn" not in kwargs:
        try:
            return llama_cls(**kwargs, flash_attn=True)
        except TypeError:
            pass
    return llama_cls(**kwargs)


def _validate_json_schema(val: Any, schema: Optional[Dict[str, Any]], _depth: int = 0) -> bool:
    """Lightweight recursive JSON Schema validator for fast unconstrained JSON verification."""
    if not isinstance(schema, dict) or _depth > 8:
        return True
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(_validate_json_schema(val, sub, _depth + 1) for sub in any_of if isinstance(sub, dict)):
            return False
    if "enum" in schema and val not in (schema.get("enum") or ()):
        return False
    expected_t = schema.get("type")
    if expected_t is not None:
        t_list = [expected_t] if isinstance(expected_t, str) else list(expected_t)
        ok_t = False
        for t_s in t_list:
            if t_s == "object" and isinstance(val, dict):
                ok_t = True
            elif t_s == "array" and isinstance(val, list):
                ok_t = True
            elif t_s == "string" and isinstance(val, str):
                ok_t = True
            elif t_s == "integer" and isinstance(val, int) and not isinstance(val, bool):
                ok_t = True
            elif t_s == "number" and isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                ok_t = True
            elif t_s == "boolean" and isinstance(val, bool):
                ok_t = True
            elif t_s == "null" and val is None:
                ok_t = True
        if not ok_t:
            return False
    if isinstance(val, dict):
        max_props = schema.get("maxProperties")
        if max_props is not None and len(val) > int(max_props):
            return False
        for req_k in schema.get("required") or ():
            if req_k not in val:
                return False
        props = schema.get("properties")
        if isinstance(props, dict):
            for pk, p_sch in props.items():
                if pk in val and not _validate_json_schema(val[pk], p_sch, _depth + 1):
                    return False
        add_p = schema.get("additionalProperties")
        if add_p is False and isinstance(props, dict):
            declared = set(props.keys())
            for vk in val.keys():
                if vk not in declared:
                    return False
        elif isinstance(add_p, dict):
            declared = set(props.keys()) if isinstance(props, dict) else set()
            for vk, vv in val.items():
                if vk not in declared and not _validate_json_schema(vv, add_p, _depth + 1):
                    return False
    elif isinstance(val, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(val) < int(min_items):
            return False
        max_items = schema.get("maxItems")
        if max_items is not None and len(val) > int(max_items):
            return False
        item_sch = schema.get("items")
        if isinstance(item_sch, dict):
            for elem in val:
                if not _validate_json_schema(elem, item_sch, _depth + 1):
                    return False
    return True


def _render_schema_skeleton(schema: Any, depth: int = 0) -> str:
    """Render a JSON Schema into a concise structural type-spec."""
    if not isinstance(schema, dict) or depth > 5:
        return "any"
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        rendered = [
            _render_schema_skeleton(sub, depth + 1)
            for sub in any_of[:4]
            if isinstance(sub, dict)
        ]
        rendered = [r for r in rendered if r]
        return "|".join(dict.fromkeys(rendered)) if rendered else "any"
    enums = schema.get("enum")
    if isinstance(enums, list) and enums:
        shown = [json.dumps(x) for x in enums[:12]]
        suffix = ", ..." if len(enums) > 12 else ""
        return "one of (" + "|".join(shown) + suffix + ")"
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t == "object" or ("properties" in schema and t is None):
        props = schema.get("properties")
        if isinstance(props, dict) and props:
            parts = []
            for k, v in list(props.items())[:24]:
                parts.append(f"{json.dumps(k)}: {_render_schema_skeleton(v, depth + 1)}")
            return "{" + ", ".join(parts) + "}"
        return "object"
    if t == "array" or ("items" in schema and t is None):
        items = schema.get("items")
        if isinstance(items, dict):
            return "list<" + _render_schema_skeleton(items, depth + 1) + ">"
        return "list<any>"
    if t in ("string", "number", "integer", "boolean"):
        return t
    return "any"


def _normalize_json_to_schema(val: Any, schema: Optional[Dict[str, Any]], _depth: int = 0) -> Any:
    """Losslessly coerce minor wrapper shape mismatches to satisfy JSON schema."""
    if not isinstance(schema, dict) or _depth > 8:
        return val
    expected_t = schema.get("type")
    t_list = [expected_t] if isinstance(expected_t, str) else (list(expected_t) if isinstance(expected_t, list) else [])
    allows_null = "null" in t_list
    primary_t = next((x for x in t_list if x != "null"), None)
    if primary_t == "string":
        if val is None and not allows_null:
            return ""
        if isinstance(val, (list, tuple)) and val and all(isinstance(x, (str, int, float, bool)) for x in val):
            return str(val[0]) if len(val) == 1 else ", ".join(str(x) for x in val)
        return val
    if primary_t == "array" or ("items" in schema and primary_t is None):
        if val is None and not allows_null:
            val = []
        if isinstance(val, list):
            item_sch = schema.get("items")
            if isinstance(item_sch, dict):
                return [_normalize_json_to_schema(elem, item_sch, _depth + 1) for elem in val]
        return val
    if primary_t == "object" or ("properties" in schema and primary_t is None):
        if val is None and not allows_null:
            val = {}
        if isinstance(val, dict):
            props = schema.get("properties")
            add_p = schema.get("additionalProperties")
            out = dict(val)
            if isinstance(props, dict):
                for pk, p_sch in props.items():
                    if pk in out:
                        out[pk] = _normalize_json_to_schema(out[pk], p_sch, _depth + 1)
            if isinstance(add_p, dict):
                declared = set(props.keys()) if isinstance(props, dict) else set()
                for vk in list(out.keys()):
                    if vk not in declared:
                        out[vk] = _normalize_json_to_schema(out[vk], add_p, _depth + 1)
            return out
    return val


def _strip_thinking_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning traces without regex."""
    if not text:
        return ""
    s = str(text)
    while True:
        low = s.lower()
        start = low.find("<think>")
        if start == -1:
            break
        end = low.find("</think>", start + 7)
        if end != -1:
            s = s[:start] + s[end + 8 :]
        else:
            s = s[:start]
            break
    low_rem = s.lower()
    tail_idx = low_rem.rfind("</think>")
    if tail_idx != -1:
        s = s[tail_idx + 8 :]
    return s.strip()


def _llm_supports_kwarg(llm: Any, method_name: str, kwarg_name: str) -> bool:
    fn = getattr(llm, method_name, None)
    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if kwarg_name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _get_no_think_chat_handler(llm: Any) -> Optional[Any]:
    """Build or return a cached llama_cpp chat handler with enable_thinking=False forced."""
    cached = getattr(llm, "_nstl_no_think_handler", None)
    if cached is not None:
        return cached
    meta = getattr(llm, "metadata", None)
    if not isinstance(meta, dict):
        return None
    tmpl = str(meta.get("tokenizer.chat_template", "") or "")
    if "enable_thinking" not in tmpl:
        return None
    try:
        import llama_cpp.llama_chat_format as cf

        def _decode_tok(tok_fn_name: str) -> Tuple[str, Optional[int]]:
            detok = getattr(llm, "detokenize", None)
            tok_fn = getattr(llm, tok_fn_name, None)
            if callable(detok) and callable(tok_fn):
                tok_id = tok_fn()
                try:
                    raw = detok([tok_id], special=True)
                except TypeError:
                    raw = detok([tok_id])
                txt = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw or "")
                parsed_id = tok_id if isinstance(tok_id, int) and not isinstance(tok_id, bool) and tok_id >= 0 else None
                return txt, parsed_id
            return "", None

        eos, eos_id = _decode_tok("token_eos")
        if not eos:
            return None
        bos, _ = _decode_tok("token_bos")

        class _NoThinkJinja2Formatter(cf.Jinja2ChatFormatter):
            def __call__(self, *args: Any, **kw: Any) -> Any:
                kw["enable_thinking"] = False
                return super().__call__(*args, **kw)

        stop_ids = [eos_id] if eos_id is not None else None
        try:
            formatter = _NoThinkJinja2Formatter(
                template=tmpl, eos_token=eos, bos_token=bos, stop_token_ids=stop_ids
            )
        except TypeError:
            formatter = _NoThinkJinja2Formatter(template=tmpl, eos_token=eos, bos_token=bos)
        handler = formatter.to_chat_handler()
        try:
            setattr(llm, "_nstl_no_think_handler", handler)
        except Exception:
            pass
        return handler
    except Exception:
        return None


def _call_llama_chat_no_think(llm: Any, lock: threading.Lock, call_kwargs: Dict[str, Any]) -> Any:
    """Invoke create_chat_completion with thinking suppressed."""
    kw = dict(call_kwargs)
    kw["messages"] = [dict(m) for m in (kw.get("messages") or [])]
    supports_ctk = _llm_supports_kwarg(llm, "create_chat_completion", "chat_template_kwargs")
    if supports_ctk:
        ctk = dict(kw.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        kw["chat_template_kwargs"] = ctk

    with lock:
        if not supports_ctk:
            nt_handler = _get_no_think_chat_handler(llm)
            if nt_handler is not None and hasattr(llm, "chat_handler"):
                prev_handler = llm.chat_handler
                llm.chat_handler = nt_handler
                try:
                    return llm.create_chat_completion(**kw)
                finally:
                    llm.chat_handler = prev_handler
        return llm.create_chat_completion(**kw)


def _count_fixed_leaves(sch: Any, depth: int) -> Optional[int]:
    if not isinstance(sch, dict) or depth > 2:
        return None
    if any(k in sch for k in ("anyOf", "oneOf", "allOf")):
        return None
    add_p = sch.get("additionalProperties")
    if isinstance(add_p, dict) or add_p is True:
        return None
    t = sch.get("type")
    t_list = [t] if isinstance(t, str) else (list(t) if isinstance(t, list) else [])
    prims = ("string", "integer", "number", "boolean", "null")
    if "enum" in sch and (not t_list or all(x in prims for x in t_list)):
        return 1
    if t_list and all(x in prims for x in t_list):
        return 1
    if "array" in t_list or ("items" in sch and not t_list):
        item_sch = sch.get("items")
        if not isinstance(item_sch, dict) or "properties" in item_sch or item_sch.get("type") in ("object", "array"):
            return None
        return 1 if _count_fixed_leaves(item_sch, depth + 1) == 1 else None
    if "object" in t_list or ("properties" in sch and not t_list):
        props = sch.get("properties")
        if not isinstance(props, dict) or not props:
            return None
        total = 0
        for p_sch in props.values():
            sub = _count_fixed_leaves(p_sch, depth + 1)
            if sub is None:
                return None
            total += sub
        return total
    return None


def _parse_and_validate_schema_output(raw_text: str, schema: Dict[str, Any]) -> Optional[str]:
    cleaned = _strip_thinking_blocks(raw_text)
    if not cleaned:
        return None
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = extract_json_object(cleaned)
    if isinstance(parsed, list) and isinstance(schema, dict):
        expected_t = schema.get("type")
        req_keys = list(schema.get("required") or ())
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if expected_t == "object" and len(req_keys) == 1:
            rk = req_keys[0]
            r_prop = props.get(rk)
            if isinstance(r_prop, dict) and r_prop.get("type") == "array":
                parsed = {rk: parsed}
    if isinstance(parsed, dict):
        parsed = _normalize_json_to_schema(parsed, schema)
        if _validate_json_schema(parsed, schema):
            return json.dumps(parsed)
    return None


def _logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _extract_first_token_top_logprobs(response: Any) -> Dict[str, float]:
    """Extract first generated token logprobs `{token: float}` from chat or completion output."""
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    lp_block = choices[0].get("logprobs")
    if not isinstance(lp_block, dict):
        return {}

    out: Dict[str, float] = {}

    def _record(tok: Any, lp: Any) -> None:
        if tok is None:
            return
        try:
            f = float(lp)
        except (TypeError, ValueError):
            return
        if not math.isfinite(f):
            return
        ts = str(tok)
        if ts not in out or f > out[ts]:
            out[ts] = f

    content_lps = lp_block.get("content")
    if isinstance(content_lps, list) and content_lps and isinstance(content_lps[0], dict):
        entry0 = content_lps[0]
        _record(entry0.get("token"), entry0.get("logprob"))
        tops = entry0.get("top_logprobs")
        if isinstance(tops, list):
            for item in tops:
                if isinstance(item, dict):
                    _record(item.get("token"), item.get("logprob"))
        elif isinstance(tops, dict):
            for tk, lv in tops.items():
                _record(tk, lv)
        if out:
            return out

    tokens = lp_block.get("tokens")
    tok_lps = lp_block.get("token_logprobs")
    if isinstance(tokens, list) and isinstance(tok_lps, list) and tokens and tok_lps:
        _record(tokens[0], tok_lps[0])
    raw_tops = lp_block.get("top_logprobs")
    if isinstance(raw_tops, list) and raw_tops and isinstance(raw_tops[0], dict):
        for tk, lv in raw_tops[0].items():
            _record(tk, lv)
    elif isinstance(raw_tops, dict):
        for tk, lv in raw_tops.items():
            _record(tk, lv)
    return out


def _map_label_logprobs_to_probs(token_lps: Dict[str, float], label_to_cid: Dict[str, str]) -> Dict[str, float]:
    """Map first-token option-label logprobs (A, B, ...) to normalized softmax probabilities."""
    if not token_lps or not label_to_cid:
        return {}
    strip_chars = " \t\r\n()[]*:\"'."
    label_lps: Dict[str, float] = {}
    for lbl in label_to_cid:
        exact_matches = [lv for tk, lv in token_lps.items() if tk.strip(strip_chars) == lbl]
        if exact_matches:
            label_lps[lbl] = _logsumexp(exact_matches)
            continue
        ci_matches = [lv for tk, lv in token_lps.items() if tk.strip(strip_chars).upper() == lbl]
        if ci_matches:
            label_lps[lbl] = _logsumexp(ci_matches)

    if not label_lps:
        return {}

    tail_floor = min(token_lps.values()) - 2.0
    full_lps = {lbl: label_lps.get(lbl, tail_floor) for lbl in label_to_cid}
    max_lp = max(full_lps.values())
    exps = {lbl: math.exp(lp - max_lp) for lbl, lp in full_lps.items()}
    z = sum(exps.values())
    if z <= 0.0 or not math.isfinite(z):
        return {}
    return {label_to_cid[lbl]: exps[lbl] / z for lbl in label_to_cid}


def _generate_llama_with_optional_schema(
    llm: Any,
    lock: threading.Lock,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    schema: Optional[dict],
) -> str:
    """Run llama_cpp chat completion with optional JSON Schema enforcement."""
    if not schema:
        call_kwargs = dict(kwargs)
        call_kwargs["messages"] = messages
        with lock:
            response = llm.create_chat_completion(**call_kwargs)
        return (response["choices"][0]["message"]["content"] or "").strip()

    leaves = _count_fixed_leaves(schema, 0)
    if leaves is not None and 1 <= leaves <= 6:
        gbnf_kwargs = dict(kwargs)
        gbnf_kwargs["messages"] = [dict(m) for m in messages]
        gbnf_kwargs["temperature"] = 0.0
        gbnf_kwargs["repeat_penalty"] = 1.0
        gbnf_kwargs["response_format"] = {"type": "json_object", "schema": schema}
        response = _call_llama_chat_no_think(llm, lock, gbnf_kwargs)
        raw = (response["choices"][0]["message"]["content"] or "").strip()
        validated = _parse_and_validate_schema_output(raw, schema)
        if validated is not None:
            return validated
        raise ValueError("Structured LLM output failed JSON schema validation.")

    try:
        fast_msgs = [dict(m) for m in messages]
        sys_txt = fast_msgs[0]["content"] if (fast_msgs and fast_msgs[0].get("role") == "system") else ""
        if (
            len(fast_msgs) <= 2
            and "Example format" not in sys_txt
            and "Available loaded domains" not in sys_txt
            and "Loaded domains" not in sys_txt
        ):
            spec_str = _render_schema_skeleton(schema)
            if len(spec_str) > 2000:
                spec_str = spec_str[:2000] + "...}"
            req_keys = list(schema.get("required") or ()) if isinstance(schema, dict) else []
            req_note = (
                f" MUST include all required keys ({', '.join(req_keys)} — use [] or \"\" if none)."
                if req_keys
                else ""
            )
            schema_hint = (
                "\nOutput ONLY ONE valid minified single-line JSON object adhering to this type specification:"
                f"{req_note}\nTypeSpec: {spec_str}"
            )
            if fast_msgs and fast_msgs[0].get("role") == "system":
                fast_msgs[0]["content"] = sys_txt + schema_hint
            else:
                fast_msgs.insert(0, {"role": "system", "content": schema_hint.strip()})
        fast_kwargs = dict(kwargs)
        fast_kwargs["messages"] = fast_msgs
        fast_kwargs["temperature"] = 0.0
        fast_kwargs["repeat_penalty"] = 1.0
        fast_resp = _call_llama_chat_no_think(llm, lock, fast_kwargs)
        fast_raw = (fast_resp["choices"][0]["message"]["content"] or "").strip()
        validated_fast = _parse_and_validate_schema_output(fast_raw, schema)
        if validated_fast is not None:
            return validated_fast
    except Exception as exc:
        logger.debug("[INFERENCE] Fast JSON path fell back to GBNF schema: %s", exc)

    gbnf_kwargs = dict(kwargs)
    gbnf_kwargs["messages"] = [dict(m) for m in messages]
    gbnf_kwargs["temperature"] = 0.0
    gbnf_kwargs["repeat_penalty"] = 1.0
    gbnf_kwargs["response_format"] = {"type": "json_object", "schema": schema}
    response = _call_llama_chat_no_think(llm, lock, gbnf_kwargs)
    raw = (response["choices"][0]["message"]["content"] or "").strip()
    validated = _parse_and_validate_schema_output(raw, schema)
    if validated is not None:
        return validated
    raise ValueError("Structured LLM output failed JSON schema validation.")


def _discover_local_embedders() -> List[str]:
    emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
    if not os.path.isdir(emb_base_dir):
        return []
    return sorted(
        d for d in os.listdir(emb_base_dir)
        if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
    )


def select_optimal_embedder(requested_name: str = "") -> str:
    """
    Resolve an embedding model strictly from explicit argument, configured setting,
    `settings.preferred_embedders`, or sorted local discovery — zero name hardcodes.
    """
    available = _discover_local_embedders()
    req = (requested_name or settings.embedder_model or "").strip()
    if not available:
        return req if req and req.lower() not in ("auto", "default") else "default"

    if req and req.lower() not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand
        raise FileNotFoundError(f"Requested embedder '{req}' not found among local models: {available}")

    for pref in (settings.preferred_embedders or ()):
        p_s = str(pref or "").strip()
        if not p_s:
            continue
        if p_s in available:
            return p_s
        for cand in available:
            if p_s.lower() in cand.lower():
                return cand

    return available[0]


def select_optimal_llm(requested_name: str = "") -> str:
    """Resolve a local GGUF LLM from MODELS_DIR/llms cleanly without hardcoded model names."""
    llm_base_dir = os.path.join(MODELS_DIR, "llms")
    if not os.path.exists(llm_base_dir):
        raise FileNotFoundError(f"LLM directory does not exist: {llm_base_dir}")

    available = sorted(
        d for d in os.listdir(llm_base_dir)
        if os.path.isdir(os.path.join(llm_base_dir, d))
        and any(f.endswith(".gguf") for f in os.listdir(os.path.join(llm_base_dir, d)))
    )
    if not available:
        raise FileNotFoundError(f"No GGUF LLM models found under {llm_base_dir}")

    req = (requested_name or settings.llm_model or "").strip()
    if req and req.lower() not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand
        raise FileNotFoundError(f"Requested LLM '{req}' not found among local models: {available}")

    for pref in (settings.preferred_llms or ()):
        p_s = str(pref or "").strip()
        if p_s in available:
            return p_s
    return available[0]


_GLOBAL_EMBED_CACHE: Dict[Tuple[Any, str, str], List[float]] = {}
_GLOBAL_EMBED_CACHE_LOCK = threading.RLock()


def clear_embedding_cache() -> None:
    """Clear the global in-memory embedding cache."""
    with _GLOBAL_EMBED_CACHE_LOCK:
        _GLOBAL_EMBED_CACHE.clear()


def _encode_with_modes(
    model: Any,
    texts: List[str],
    mode: str = "query",
    batch_size: int = 32,
    lock: Optional[threading.Lock] = None,
) -> List[List[float]]:
    """Encode texts with query/document mode handling and OOM micro-batch recovery."""
    if not texts:
        return []

    if not hasattr(model, "_model_cache_id"):
        try:
            import uuid
            model._model_cache_id = f"{type(model).__name__}_{id(model)}_{uuid.uuid4().hex}"
        except Exception:
            model._model_cache_id = f"{type(model).__name__}_{id(model)}"
    m_id = model._model_cache_id
    cached_map: Dict[int, List[float]] = {}
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    with _GLOBAL_EMBED_CACHE_LOCK:
        for idx, txt in enumerate(texts):
            key = (m_id, txt, mode)
            if key in _GLOBAL_EMBED_CACHE:
                cached_map[idx] = _GLOBAL_EMBED_CACHE[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(txt)

    if not missing_texts:
        return [cached_map[i] for i in range(len(texts))]

    prompt_name = "query" if mode == "query" else "document"
    has_torch = _ensure_torch()

    def _invoke(p_name: Optional[str], t_name: Optional[str], b_size: int, input_texts: List[str]):
        kwargs: Dict[str, Any] = {"convert_to_numpy": True, "batch_size": b_size, "show_progress_bar": False}
        if p_name:
            kwargs["prompt_name"] = p_name
        if t_name:
            kwargs["task"] = t_name
        if lock is not None:
            with lock:
                if has_torch:
                    with torch.inference_mode():
                        return model.encode(input_texts, **kwargs)
                return model.encode(input_texts, **kwargs)
        if has_torch:
            with torch.inference_mode():
                return model.encode(input_texts, **kwargs)
        return model.encode(input_texts, **kwargs)

    attempts = [(prompt_name, "retrieval"), (prompt_name, None), (None, "retrieval"), (None, None)]
    sig_cache = getattr(model, "_working_encode_sig", None)
    if isinstance(sig_cache, dict) and prompt_name in sig_cache and sig_cache[prompt_name] in attempts:
        pref = sig_cache[prompt_name]
        attempts = [pref] + [a for a in attempts if a != pref]

    last_sig_error = None
    computed: Optional[List[List[float]]] = None

    def _remember_sig(p_name: Optional[str], t_name: Optional[str]) -> None:
        try:
            sc = getattr(model, "_working_encode_sig", None)
            if not isinstance(sc, dict):
                sc = {}
                setattr(model, "_working_encode_sig", sc)
            sc[prompt_name] = (p_name, t_name)
        except Exception:
            pass

    for p_name, t_name in attempts:
        try:
            res = _invoke(p_name, t_name, batch_size, missing_texts)
            computed = res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
            _remember_sig(p_name, t_name)
            break
        except (TypeError, ValueError) as arg_err:
            last_sig_error = arg_err
            continue
        except Exception as err:
            is_oom = has_torch and isinstance(err, torch.cuda.OutOfMemoryError)
            if not is_oom and "out of memory" not in str(err).lower():
                raise err
            if has_torch and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            micro_bs = max(1, batch_size // 4)
            micro_results: List[List[float]] = []
            try:
                for i in range(0, len(missing_texts), micro_bs):
                    sub_res = _invoke(p_name, t_name, micro_bs, missing_texts[i : i + micro_bs])
                    micro_results.extend(sub_res.tolist() if hasattr(sub_res, "tolist") else [list(r) for r in sub_res])
                computed = micro_results
                _remember_sig(p_name, t_name)
                break
            except (TypeError, ValueError) as arg_err:
                last_sig_error = arg_err
                continue
            except Exception as sub_err:
                is_sub_oom = has_torch and isinstance(sub_err, torch.cuda.OutOfMemoryError)
                if is_sub_oom or "out of memory" in str(sub_err).lower():
                    if has_torch and torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                raise sub_err

    if computed is None:
        try:
            if lock is not None:
                with lock:
                    res = model.encode(missing_texts, convert_to_numpy=True)
            else:
                res = model.encode(missing_texts, convert_to_numpy=True)
            computed = res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
        except Exception as final_err:
            raise final_err from last_sig_error

    with _GLOBAL_EMBED_CACHE_LOCK:
        max_size = settings.max_embedding_cache_size
        for idx, txt, emb in zip(missing_indices, missing_texts, computed):
            _GLOBAL_EMBED_CACHE[(m_id, txt, mode)] = emb
            cached_map[idx] = emb
        while len(_GLOBAL_EMBED_CACHE) > max_size:
            try:
                del _GLOBAL_EMBED_CACHE[next(iter(_GLOBAL_EMBED_CACHE))]
            except (StopIteration, KeyError):
                break

    return [cached_map[i] for i in range(len(texts))]


def _resolve_embedder_task(emb_path: str) -> Optional[str]:
    """Read configured or model-declared default task metadata without name heuristics."""
    if getattr(settings, "embedder_task", None):
        return str(settings.embedder_task)
    if not emb_path or not os.path.isdir(emb_path):
        return None
    declared_tasks: List[str] = []
    explicit_default: Optional[str] = None
    for cfg_name in ("config.json", "config_sentence_transformers.json"):
        cfg_file = os.path.join(emb_path, cfg_name)
        if not os.path.isfile(cfg_file):
            continue
        try:
            with open(cfg_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not explicit_default and isinstance(data.get("default_task"), str) and data["default_task"].strip():
            explicit_default = data["default_task"].strip()
        for list_key in ("task_names", "lora_adaptors", "adapters"):
            vals = data.get(list_key)
            if isinstance(vals, list):
                declared_tasks.extend(str(v).strip() for v in vals if isinstance(v, str) and v.strip())
            elif isinstance(vals, dict):
                declared_tasks.extend(str(k).strip() for k in vals.keys() if isinstance(k, str) and k.strip())
        if isinstance(data.get("prompts"), dict) and "retrieval" in data["prompts"]:
            declared_tasks.append("retrieval")
    if explicit_default:
        return explicit_default
    if "retrieval" in declared_tasks:
        return "retrieval"
    return declared_tasks[0] if declared_tasks else None


def _build_embedder_model_kwargs(task_name: Optional[str], device: str) -> Dict[str, Any]:
    mk: Dict[str, Any] = {}
    if task_name:
        mk["default_task"] = task_name
    if "cuda" in str(device).lower() and _ensure_torch() and torch.cuda.is_available():
        mk["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mk["low_cpu_mem_usage"] = True
    return mk


def _cast_embedder_half_precision(model: Any, device: str) -> Any:
    if model is None or "cuda" not in str(device).lower() or not _ensure_torch():
        return model
    try:
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(target_dtype)
        torch.cuda.empty_cache()
    except Exception:
        pass
    return model


def resolve_embedding_dimension(model: Any) -> int:
    """Resolve embedding dimension of a loaded SentenceTransformer model."""
    modern = getattr(model, "get_embedding_dimension", None)
    if callable(modern):
        try:
            dim = int(modern())
            if dim > 0:
                return dim
        except Exception:
            pass
    legacy = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(legacy):
        try:
            dim = int(legacy())
            if dim > 0:
                return dim
        except Exception:
            pass
    vecs = _encode_with_modes(model, ["strata dimension probe"], mode="document")
    dim = len(vecs[0]) if vecs and vecs[0] else 0
    if dim > 0:
        return dim
    raise RuntimeError("Could not resolve embedding dimension from loaded model.")


def get_adaptive_batch_size(device: str = "cuda", min_batch: int = 8, max_batch: int = 128) -> int:
    """Compute batch size dynamically from free CUDA VRAM headroom."""
    if "cuda" not in str(device).lower() or not _ensure_torch() or not torch.cuda.is_available():
        return 32
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        power = max(3, min(7, math.floor(math.log2(max(1.0, free_gb * 8.0)))))
        return max(min_batch, min(max_batch, int(2 ** power)))
    except Exception:
        return 32


class ModelManager:
    """Capability-driven singleton managing loaded embedder and GGUF LLM weights."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self.active_spec: Optional[ProfileSpec] = None
        self.current_profile_name: Optional[str] = None
        self.embedder: Any = None
        self.embedder_name: str = ""
        self.llm: Any = None
        self.llm_name: str = ""
        self._llm_logits_all: bool = False
        self._dim: int = 0
        self._embedder_lock = threading.Lock()
        self._llm_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def profile(self) -> Optional[ProfileSpec]:
        return self.active_spec

    @property
    def active_profile(self) -> Optional[ProfileSpec]:
        return self.active_spec

    def _load_embedder(self, emb_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        emb_path = os.path.join(MODELS_DIR, "embeddings", emb_name)
        task_name = _resolve_embedder_task(emb_path)
        model_kwargs = _build_embedder_model_kwargs(task_name, device)
        try:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        except Exception:
            fallback_mk = {"default_task": task_name} if task_name else {}
            try:
                self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=fallback_mk)
            except Exception:
                self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True)
        self.embedder = _cast_embedder_half_precision(self.embedder, device)
        self.embedder_name = emb_name
        self._dim = resolve_embedding_dimension(self.embedder)

    def _load_llm(self, llm_name: str, device: str, *, logits_all: bool) -> None:
        from llama_cpp import Llama

        llm_dir = os.path.join(MODELS_DIR, "llms", llm_name)
        if not os.path.isdir(llm_dir):
            raise FileNotFoundError(f"LLM directory not found: {llm_dir}")
        ggufs = sorted(f for f in os.listdir(llm_dir) if f.endswith(".gguf"))
        if not ggufs:
            raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
        unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
        model_file = os.path.join(llm_dir, unified_ggufs[0] if unified_ggufs else ggufs[0])
        gpu_layers = -1 if device in ("cuda", "mps", "metal", "rocm") else 0
        n_threads = max(1, (os.cpu_count() or 4) - 1)
        self.llm = _instantiate_llama(
            Llama,
            model_path=model_file,
            n_ctx=settings.llm_context_length,
            n_gpu_layers=gpu_layers,
            n_threads=n_threads,
            logits_all=bool(logits_all),
            verbose=False,
        )
        self.llm_name = llm_name
        self._llm_logits_all = bool(logits_all)

    def cleanup(self, release_reranker: bool = True) -> None:
        """Release active embedder/LLM weights and reclaim GPU/host memory."""
        with self._lock:
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
                self.llm = None
            self.llm_name = ""
            self._llm_logits_all = False
            self.embedder = None
            self.embedder_name = ""
            self._dim = 0
            self.active_spec = None
            self.current_profile_name = None

            if release_reranker:
                try:
                    try:
                        from .reranker import LocalReranker
                    except (ImportError, ValueError):
                        from reranker import LocalReranker
                    LocalReranker.release_shared()
                except Exception:
                    pass

            gc.collect()
            if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            _trim_host_memory()

    def initialize_profile(self, profile_type: str, embedder_name: str = "", llm_name: str = "") -> None:
        with self._lock:
            spec = resolve_profile_spec(profile_type)
            device = HardwareProfiler.get_optimal_device()

            if not spec.uses_embedder:
                self.embedder = None
                self.embedder_name = ""
                self._dim = 0
            else:
                target_emb = select_optimal_embedder(embedder_name)
                if self.embedder is None or self.embedder_name != target_emb:
                    self.embedder = None
                    self._load_embedder(target_emb, device)

            if not spec.uses_llm:
                if self.llm is not None:
                    try:
                        self.llm.close()
                    except Exception:
                        pass
                self.llm = None
                self.llm_name = ""
                self._llm_logits_all = False
            else:
                target_llm = select_optimal_llm(llm_name)
                need_logits = bool(spec.uses_choice_scorer)
                can_reuse_llm = (
                    self.llm is not None
                    and self.llm_name == target_llm
                    and (not need_logits or self._llm_logits_all)
                )
                if not can_reuse_llm:
                    if self.llm is not None:
                        try:
                            self.llm.close()
                        except Exception:
                            pass
                        self.llm = None
                    self._llm_logits_all = False
                    self._load_llm(target_llm, device, logits_all=need_logits)

            if os.environ.get("NSTL_LOW_VRAM") == "1" and not spec.uses_reranker:
                try:
                    try:
                        from .reranker import LocalReranker
                    except (ImportError, ValueError):
                        from reranker import LocalReranker
                    LocalReranker.release_shared()
                except Exception:
                    pass

            self.active_spec = spec
            self.current_profile_name = spec.code
            gc.collect()
            _trim_host_memory()

    def is_pure_symbolic(self) -> bool:
        return self.current_profile_name == "0"

    def has_embedder(self) -> bool:
        return self.embedder is not None and self._dim > 0

    def has_llm(self) -> bool:
        return self.llm is not None

    def can_synthesize(self) -> bool:
        return self.has_llm()

    def uses_reranker(self) -> bool:
        return bool(self.active_spec.uses_reranker) if self.active_spec is not None else False

    def has_choice_scorer(self) -> bool:
        return self.llm is not None and bool(self.active_spec and self.active_spec.uses_choice_scorer)

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.embedder is None:
            return []
        device = getattr(self.embedder, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.embedder, texts, mode=mode, batch_size=batch_size, lock=self._embedder_lock)

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        schema: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded for the active profile.")
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for ex in few_shot or ():
            if isinstance(ex, dict) and ex.get("role") and ex.get("content") is not None:
                messages.append({"role": str(ex["role"]), "content": str(ex["content"])})
        messages.append({"role": "user", "content": prompt})
        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens if max_tokens is not None else settings.llm_max_tokens),
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "repeat_penalty": settings.llm_repeat_penalty,
        }
        if settings.llm_stop_tokens:
            kwargs["stop"] = list(settings.llm_stop_tokens)
        return _generate_llama_with_optional_schema(self.llm, self._llm_lock, messages, kwargs, schema)

    def score_choices(self, prompt: str, candidates: List[str], context_desc: str = "") -> Dict[str, float]:
        """Single-pass 1-token option-label logprob scorer over candidate operation IDs."""
        clean_cands: List[str] = []
        seen = set()
        for c in (candidates or ()):
            s = str(c).strip() if c is not None else ""
            if s and s not in seen:
                seen.add(s)
                clean_cands.append(s)
        if not clean_cands or self.llm is None:
            return {}
        capped = clean_cands[:12]
        if len(capped) == 1:
            return {capped[0]: 1.0}

        labels = [chr(ord("A") + i) for i in range(len(capped))]
        label_to_cid = dict(zip(labels, capped))
        ctx_line = f"\nContext: {context_desc}" if context_desc else ""
        opts_block = "\n".join(f"{lbl}) {cid}" for lbl, cid in zip(labels, capped))
        sys_p = (
            f"Select the single best next operation option ({labels[0]}-{labels[-1]}). "
            "Reply with ONLY the single capital letter."
        )
        usr_p = f"Goal: {prompt}{ctx_line}\nOptions:\n{opts_block}\nAnswer ({labels[0]}-{labels[-1]}):"
        call_kwargs: Dict[str, Any] = {
            "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": max(20, len(capped) * 2),
        }
        try:
            resp = _call_llama_chat_no_think(self.llm, self._llm_lock, call_kwargs)
            token_lps = _extract_first_token_top_logprobs(resp)
            return _map_label_logprobs_to_probs(token_lps, label_to_cid)
        except Exception as err:
            logger.debug("[INFERENCE] Constrained choice scoring skipped: %s", err)
            return {}
