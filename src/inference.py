"""
src/inference.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Multi-Profile Inference Engine with Dynamic Hardware Introspection,
Agnostic Semantic Compilation, and Thread-Safe Generation.
"""

from __future__ import annotations
import gc
import inspect
import json
import math
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union

try:
    from utils import is_split_gguf_shard
except ImportError:
    from .utils import is_split_gguf_shard

import numpy as np

# Configure PyTorch CUDA memory allocator before any lazy torch import
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Torch is imported lazily on first neural use so symbolic commands (Profile 0, compile, harvest)
# avoid ~200 MB of startup RAM and cold torch import latency.
torch = None  # type: ignore[assignment]
TORCH_AVAILABLE: Optional[bool] = None


def _ensure_torch() -> bool:
    """Lazily imports PyTorch once on first neural inference use."""
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
    """Returns freed glibc heap arenas (e.g. safetensors/torch load staging buffers) back to the OS."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def _instantiate_llama(llama_cls: Any, **kwargs: Any) -> Any:
    """Instantiates Llama with flash_attn=True when supported, falling back cleanly."""
    if "flash_attn" not in kwargs:
        try:
            return llama_cls(**kwargs, flash_attn=True)
        except TypeError:
            pass
    return llama_cls(**kwargs)

try:
    from log_config import get_logger
except ImportError:
    try:
        from .log_config import get_logger
    except ImportError:
        import logging
        get_logger = logging.getLogger

try:
    from .config import MODELS_DIR, settings
    from .utils import extract_code_from_llm_response
except (ImportError, ValueError):
    from config import MODELS_DIR, settings
    from utils import extract_code_from_llm_response

logger = get_logger("inference")


def _validate_json_schema(val: Any, schema: Optional[Dict[str, Any]], _depth: int = 0) -> bool:
    """Lightweight recursive JSON Schema validator for fast unconstrained JSON verification."""
    if not isinstance(schema, dict) or _depth > 8:
        return True
    if "enum" in schema:
        if val not in (schema.get("enum") or ()):
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
    """Renders a JSON Schema into a concise structural type-spec without literal '<string>' tokens."""
    if not isinstance(schema, dict) or depth > 5:
        return "any"
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
            req = set(schema.get("required") or ())
            parts = []
            for k, v in list(props.items())[:24]:
                mark = "*" if k in req else ""
                parts.append(f"{json.dumps(k)}{mark}: {_render_schema_skeleton(v, depth + 1)}")
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
    """Losslessly coerces minor wrapper shape mismatches (e.g. [['op']] -> ['op']) to satisfy JSON schema."""
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
    """Strips <think>...</think> reasoning traces (and unclosed leading <think>) without regex."""
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


def _is_thinking_model_llm(llm: Any) -> bool:
    """Detects whether a loaded LLM declares <think> reasoning tags or enable_thinking template support."""
    if llm is None:
        return False
    if bool(getattr(llm, "supports_thinking", False)):
        return True
    meta = getattr(llm, "metadata", None)
    if isinstance(meta, dict):
        tmpl = str(meta.get("tokenizer.chat_template", "") or "")
        if "enable_thinking" in tmpl or "<think>" in tmpl:
            return True
    return False


def _llm_supports_kwarg(llm: Any, method_name: str, kwarg_name: str) -> bool:
    """Checks via inspect.signature whether llm.<method_name> accepts kwarg_name or **kwargs."""
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
    """Builds or returns a cached llama_cpp chat handler with enable_thinking=False forced."""
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
                if isinstance(raw, bytes):
                    txt = raw.decode("utf-8", "ignore")
                else:
                    txt = str(raw or "")
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
            formatter = _NoThinkJinja2Formatter(
                template=tmpl, eos_token=eos, bos_token=bos
            )
        handler = formatter.to_chat_handler()
        try:
            setattr(llm, "_nstl_no_think_handler", handler)
        except Exception:
            pass
        return handler
    except Exception:
        return None


def _call_llama_chat_no_think(
    llm: Any,
    lock: threading.Lock,
    call_kwargs: Dict[str, Any],
) -> Any:
    """Invokes create_chat_completion with thinking suppressed via API kwargs or validated Jinja handler."""
    kw = dict(call_kwargs)
    msgs = [dict(m) for m in (kw.get("messages") or [])]
    kw["messages"] = msgs
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
    """Counts fixed primitive/list-of-primitive leaves in a bounded JSON Schema."""
    if not isinstance(sch, dict) or depth > 2:
        return None
    if any(k in sch for k in ("anyOf", "oneOf", "allOf")):
        return None
    add_p = sch.get("additionalProperties")
    if isinstance(add_p, dict) or add_p is True:
        return None

    t = sch.get("type")
    if isinstance(t, str):
        t_list = [t]
    elif isinstance(t, list):
        t_list = list(t)
    else:
        t_list = []

    prims = ("string", "integer", "number", "boolean", "null")
    if "enum" in sch and (not t_list or all(x in prims for x in t_list)):
        return 1
    if t_list and all(x in prims for x in t_list):
        return 1

    if "array" in t_list or ("items" in sch and not t_list):
        item_sch = sch.get("items")
        if not isinstance(item_sch, dict):
            return None
        if "properties" in item_sch or item_sch.get("type") in ("object", "array"):
            return None
        sub = _count_fixed_leaves(item_sch, depth + 1)
        return 1 if sub == 1 else None

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


def _is_small_fixed_schema(schema: Optional[Dict[str, Any]]) -> bool:
    """Returns True for small fixed JSON schemas suited for direct single-pass GBNF decoding."""
    leaves = _count_fixed_leaves(schema, 0)
    return leaves is not None and 1 <= leaves <= 6


def _parse_and_validate_schema_output(raw_text: str, schema: Dict[str, Any]) -> Optional[str]:
    """Strips thinking tags, parses JSON, normalizes wrappers, and strictly validates schema."""
    cleaned = _strip_thinking_blocks(raw_text)
    if not cleaned:
        return None
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            try:
                from .utils import extract_json_object
            except (ImportError, ValueError):
                from utils import extract_json_object
            parsed = extract_json_object(cleaned)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        parsed = _normalize_json_to_schema(parsed, schema)
        if _validate_json_schema(parsed, schema):
            return json.dumps(parsed)
    return None


def _logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _extract_first_token_top_logprobs(response: Any) -> Dict[str, float]:
    """Extracts first generated token's logprobs {token_str: float} from chat or completion output."""
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return {}
    lp_block = choice0.get("logprobs")
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
    if isinstance(content_lps, list) and content_lps:
        entry0 = content_lps[0]
        if isinstance(entry0, dict):
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
    if isinstance(raw_tops, list) and raw_tops:
        top0 = raw_tops[0]
        if isinstance(top0, dict):
            for tk, lv in top0.items():
                _record(tk, lv)
    elif isinstance(raw_tops, dict):
        for tk, lv in raw_tops.items():
            _record(tk, lv)
    return out


def _map_label_logprobs_to_probs(
    token_lps: Dict[str, float],
    label_to_cid: Dict[str, str],
) -> Dict[str, float]:
    """Maps first-token option-label logprobs (A, B, ...) to calibrated softmax probabilities."""
    if not token_lps or not label_to_cid:
        return {}
    strip_chars = " \t\r\n()[]*:\"'."
    label_lps: Dict[str, float] = {}
    for lbl in label_to_cid:
        exact_matches = [
            lv for tk, lv in token_lps.items()
            if tk.strip(strip_chars) == lbl
        ]
        if exact_matches:
            label_lps[lbl] = _logsumexp(exact_matches)
            continue
        ci_matches = [
            lv for tk, lv in token_lps.items()
            if tk.strip(strip_chars).upper() == lbl
        ]
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
    """
    Runs llama_cpp chat completion with optional JSON Schema enforcement.
    - Small fixed schemas use direct single-pass GBNF with thinking suppressed (1 LLM call).
    - Large/open schemas attempt fast unconstrained JSON (thinking suppressed) and fall back to GBNF only on failure.
    """
    if not schema:
        call_kwargs = dict(kwargs)
        call_kwargs["messages"] = messages
        with lock:
            response = llm.create_chat_completion(**call_kwargs)
        return (response["choices"][0]["message"]["content"] or "").strip()

    if _is_small_fixed_schema(schema):
        gbnf_kwargs = dict(kwargs)
        gbnf_kwargs["messages"] = [dict(m) for m in messages]
        gbnf_kwargs["temperature"] = 0.0
        gbnf_kwargs["response_format"] = {"type": "json_object", "schema": schema}
        response = _call_llama_chat_no_think(llm, lock, gbnf_kwargs)
        raw = (response["choices"][0]["message"]["content"] or "").strip()
        validated = _parse_and_validate_schema_output(raw, schema)
        if validated is not None:
            return validated
        raise ValueError("Structured LLM output failed JSON schema validation.")

    try:
        spec_str = _render_schema_skeleton(schema)
        if len(spec_str) > 2000:
            spec_str = spec_str[:2000] + "...}"
        req_keys = list(schema.get("required") or ()) if isinstance(schema, dict) else []
        req_note = f" MUST include all required keys ({', '.join(req_keys)}, marked * — use [] or \"\" if none)." if req_keys else ""
        schema_hint = (
            "\nOutput ONLY ONE valid minified single-line JSON object (no newlines, no indentation) adhering to this type specification "
            "(fill required fields with actual values derived from the user request; omit optional empty fields; match array vs object "
            f"types and enum strings verbatim; no markdown fences, no commentary):{req_note}\nTypeSpec: {spec_str}"
        )
        fast_msgs = [dict(m) for m in messages]
        if fast_msgs and fast_msgs[0].get("role") == "system":
            fast_msgs[0]["content"] = fast_msgs[0]["content"] + schema_hint
        else:
            fast_msgs.insert(0, {"role": "system", "content": schema_hint.strip()})
        fast_kwargs = dict(kwargs)
        fast_kwargs["messages"] = fast_msgs
        fast_kwargs["temperature"] = 0.0
        fast_resp = _call_llama_chat_no_think(llm, lock, fast_kwargs)
        fast_raw = (fast_resp["choices"][0]["message"]["content"] or "").strip()
        validated_fast = _parse_and_validate_schema_output(fast_raw, schema)
        if validated_fast is not None:
            return validated_fast
    except Exception as _fast_err:
        logger.debug(f"[INFERENCE] Fast JSON path fell back to GBNF schema: {_fast_err}")

    gbnf_kwargs = dict(kwargs)
    gbnf_kwargs["messages"] = [dict(m) for m in messages]
    gbnf_kwargs["temperature"] = 0.0
    gbnf_kwargs["response_format"] = {"type": "json_object", "schema": schema}
    response = _call_llama_chat_no_think(llm, lock, gbnf_kwargs)
    raw = (response["choices"][0]["message"]["content"] or "").strip()
    validated = _parse_and_validate_schema_output(raw, schema)
    if validated is not None:
        return validated
    raise ValueError("Structured LLM output failed JSON schema validation.")


def resolve_embedding_dimension(model: Any) -> int:
    """
    Resolves the embedding dimension of a loaded SentenceTransformer model
    across library versions. Fails loudly if the dimension cannot be inspected.
    """
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
        except TypeError:
            name = getattr(model, "_model_name", None) or getattr(model, "model_name_or_path", None)
            if isinstance(name, str) and name:
                try:
                    dim = int(legacy(name))
                    if dim > 0:
                        return dim
                except Exception:
                    pass
        except Exception:
            pass

    vecs = _encode_with_modes(model, ["nstl dimension probe"], mode="document")
    dim = len(vecs[0]) if vecs and vecs[0] else 0
    if dim > 0:
        return dim
    raise RuntimeError("Could not resolve embedding dimension from loaded model.")


def get_adaptive_batch_size(device: str = "cuda", min_batch: int = 8, max_batch: int = 128) -> int:
    """
    Computes an optimal batch size dynamically based on available device memory headroom.
    Fully hardware-agnostic: tracks continuous free VRAM headroom without hardcoded card thresholds.
    """
    if "cuda" not in str(device).lower() or not _ensure_torch() or not torch.cuda.is_available():
        return 32

    try:
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        # Continuous logarithmic scaling mapped to powers of 2 (approx. 8 items per 1 GB of free headroom)
        power = int(np.clip(np.floor(np.log2(max(1.0, free_gb * 8))), 3, 7))
        calculated_bs = int(2 ** power)
        return int(max(min_batch, min(max_batch, calculated_bs)))
    except Exception as e:
        logger.debug(f"[HARDWARE TELEMETRY] Adaptive batch size fallback: {e}")
        return 32


class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self, embedder_name: str, llm_name: str):
        pass

    @abstractmethod
    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        pass

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        return [self.get_embedding(t, mode=mode) for t in texts]

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def can_synthesize(self) -> bool:
        pass

    @abstractmethod
    def can_feedback_check(self) -> bool:
        pass

    def has_translator_pass(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return self.can_synthesize()

    def has_ir_compiler(self) -> bool:
        return False

    def has_coalgebraic_unfolder(self) -> bool:
        return False

    def has_constrained_choice_scorer(self) -> bool:
        return False

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        return {}

    def has_embedder(self) -> bool:
        return self.embedding_dimension > 0

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass

    @abstractmethod
    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        pass


def select_optimal_embedder(requested_name: str = "") -> str:
    """
    Selects the optimal local embedding model based on precomputed cache presence and footprint efficiency.
    Scoring is dynamically normalized across available candidate models without hardcoded architecture
    substrings, fixed cache naming prefixes, or arbitrary byte constants.
    """
    emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
    if not os.path.exists(emb_base_dir):
        return requested_name or "default"

    available = sorted([
        d for d in os.listdir(emb_base_dir)
        if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
    ])
    if not available:
        return requested_name or "default"

    req = (requested_name or "").strip()
    if req and req not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand
    else:
        # Default / auto selection prioritizes the nano embedder when present
        m_nano = next((m for m in available if "nano" in m.lower()), None)
        if m_nano is not None:
            return m_nano

    # Locate cache directories
    possible_cache_dirs = [
        os.path.join(os.path.dirname(MODELS_DIR), ".rag_cache"),
        os.path.join(MODELS_DIR, ".rag_cache"),
        os.path.abspath(".rag_cache")
    ]
    cache_dir = next((cd for cd in possible_cache_dirs if os.path.isdir(cd)), None)

    model_disk_sizes: Dict[str, int] = {}
    model_cache_sizes: Dict[str, int] = {}

    for cand in available:
        cand_path = os.path.join(emb_base_dir, cand)
        try:
            total_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(cand_path)
                for f in files
            )
        except Exception:
            total_size = 1
        model_disk_sizes[cand] = max(1, total_size)

        cand_cache_size = 0
        if cache_dir and os.path.isdir(cache_dir):
            cand_norm = cand.lower().replace("-", "_")
            try:
                for f in os.listdir(cache_dir):
                    f_lower = f.lower()
                    if (cand.lower() in f_lower or cand_norm in f_lower) and "cache" in f_lower:
                        cand_cache_size = max(cand_cache_size, os.path.getsize(os.path.join(cache_dir, f)))
            except Exception:
                pass
        model_cache_sizes[cand] = cand_cache_size

    min_disk = min(model_disk_sizes.values()) if model_disk_sizes else 1
    max_cache = max(model_cache_sizes.values()) if model_cache_sizes else 0

    best_model = available[0]
    best_score = -1.0

    for cand in available:
        efficiency = float(min_disk / model_disk_sizes[cand])
        coverage = float(model_cache_sizes[cand] / max_cache) if max_cache > 0 else 0.0

        score = 0.85 * coverage + 0.15 * efficiency
        logger.debug(f"[EMBEDDER EVAL] Model: {cand} | Coverage: {coverage:.2f} | Efficiency: {efficiency:.2f} | Score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_model = cand

    logger.info(f"[EMBEDDER SELECTION] Selected optimal model '{best_model}' (readiness score: {best_score:.3f})")
    return best_model


def select_optimal_llm(requested_name: str = "") -> str:
    """
    Selects a local GGUF LLM model from MODELS_DIR/llms (respecting explicit request,
    configured `settings.llm_model` / `settings.preferred_llms`, or deterministic sorted discovery).
    Fails loudly if no local GGUF models exist.
    """
    llm_base_dir = os.path.join(MODELS_DIR, "llms")
    if not os.path.exists(llm_base_dir):
        raise FileNotFoundError(f"LLM directory does not exist: {llm_base_dir}")

    available = sorted([
        d for d in os.listdir(llm_base_dir)
        if os.path.isdir(os.path.join(llm_base_dir, d)) and any(f.endswith(".gguf") for f in os.listdir(os.path.join(llm_base_dir, d)))
    ])
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

    prefs = [str(p).strip() for p in (getattr(settings, "preferred_llms", None) or ()) if str(p).strip()]
    for pref in prefs:
        if pref in available:
            return pref
    return available[0]


_GLOBAL_EMBED_CACHE: Dict[Tuple[Any, str, str], List[float]] = {}
_GLOBAL_EMBED_CACHE_LOCK = threading.RLock()


def clear_embedding_cache() -> None:
    """Clears the global in-memory embedding cache."""
    with _GLOBAL_EMBED_CACHE_LOCK:
        _GLOBAL_EMBED_CACHE.clear()


def _encode_with_modes(
    model: Any,
    texts: List[str],
    mode: str = "query",
    batch_size: int = 32,
    lock: Optional[threading.Lock] = None
) -> List[List[float]]:
    """
    Robust embedding encoder applying model-specific prompt modes (query vs document)
    with adaptive fallback across SentenceTransformer models.
    Properly isolates CUDA OOM handling from non-OOM runtime exceptions.
    Uses in-memory LRU caching to eliminate redundant forward passes across route methods.
    """
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
        else:
            if has_torch:
                with torch.inference_mode():
                    return model.encode(input_texts, **kwargs)
            return model.encode(input_texts, **kwargs)

    attempts = [
        (prompt_name, "retrieval"),
        (prompt_name, None),
        (None, "retrieval"),
        (None, None),
    ]
    sig_cache = getattr(model, "_working_encode_sig", None)
    if isinstance(sig_cache, dict) and prompt_name in sig_cache:
        pref = sig_cache[prompt_name]
        if pref in attempts:
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
                logger.error(f"[ENCODER] Fatal encoding error: {err}")
                raise err

            logger.warning(f"[ENCODER] CUDA OOM encountered at batch_size={batch_size}. Retrying with micro-batches.")
            if has_torch and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            micro_bs = max(1, batch_size // 4)
            micro_results: List[List[float]] = []
            try:
                for i in range(0, len(missing_texts), micro_bs):
                    sub_batch = missing_texts[i : i + micro_bs]
                    sub_res = _invoke(p_name, t_name, micro_bs, sub_batch)
                    if hasattr(sub_res, "tolist"):
                        micro_results.extend(sub_res.tolist())
                    else:
                        micro_results.extend([list(r) for r in sub_res])
                computed = micro_results
                _remember_sig(p_name, t_name)
                break
            except (TypeError, ValueError) as arg_err:
                last_sig_error = arg_err
                continue
            except Exception as sub_err:
                is_sub_oom = has_torch and isinstance(sub_err, torch.cuda.OutOfMemoryError)
                if is_sub_oom or "out of memory" in str(sub_err).lower():
                    logger.warning(f"[ENCODER] Micro-batch size {micro_bs} triggered OOM. Attempting next signature candidate.")
                    if has_torch and torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                logger.error(f"[ENCODER] Fatal error in micro-batch encoding: {sub_err}")
                raise sub_err

    if computed is None:
        # Final guarded fallback if model signature rejected prompt/task arguments
        try:
            if lock is not None:
                with lock:
                    if has_torch:
                        with torch.inference_mode():
                            res = model.encode(missing_texts, convert_to_numpy=True)
                    else:
                        res = model.encode(missing_texts, convert_to_numpy=True)
            else:
                if has_torch:
                    with torch.inference_mode():
                        res = model.encode(missing_texts, convert_to_numpy=True)
                else:
                    res = model.encode(missing_texts, convert_to_numpy=True)
            computed = res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
        except Exception as final_err:
            logger.error(f"[ENCODER] Fallback encoding failed: {final_err}")
            raise final_err from last_sig_error

    with _GLOBAL_EMBED_CACHE_LOCK:
        max_size = settings.max_embedding_cache_size
        for idx, txt, emb in zip(missing_indices, missing_texts, computed):
            key = (m_id, txt, mode)
            _GLOBAL_EMBED_CACHE[key] = emb
            cached_map[idx] = emb
        while len(_GLOBAL_EMBED_CACHE) > max_size:
            try:
                first_key = next(iter(_GLOBAL_EMBED_CACHE))
                del _GLOBAL_EMBED_CACHE[first_key]
            except (StopIteration, KeyError):
                break

    return [cached_map[i] for i in range(len(texts))]


def _resolve_embedder_task(emb_path: str) -> Optional[str]:
    """Reads configured or model-declared default task/adapter metadata without name heuristics."""
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
        if not explicit_default:
            dt = data.get("default_task")
            if isinstance(dt, str) and dt.strip():
                explicit_default = dt.strip()
        for list_key in ("task_names", "lora_adaptors", "adapters"):
            vals = data.get(list_key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v.strip():
                        declared_tasks.append(v.strip())
            elif isinstance(vals, dict):
                for k in vals.keys():
                    if isinstance(k, str) and k.strip():
                        declared_tasks.append(k.strip())
        prompts = data.get("prompts")
        if isinstance(prompts, dict) and "retrieval" in prompts:
            declared_tasks.append("retrieval")

    if explicit_default:
        return explicit_default
    if "retrieval" in declared_tasks:
        return "retrieval"
    if declared_tasks:
        return declared_tasks[0]
    return None


def _build_embedder_model_kwargs(task_name: Optional[str], device: str) -> Dict[str, Any]:
    """Builds SentenceTransformer model_kwargs from model/settings capabilities and CUDA precision."""
    mk: Dict[str, Any] = {}
    if task_name:
        mk["default_task"] = task_name
    if "cuda" in str(device).lower() and _ensure_torch() and torch.cuda.is_available():
        mk["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mk["low_cpu_mem_usage"] = True
    return mk


def _cast_embedder_half_precision(model: Any, device: str) -> Any:
    """Casts SentenceTransformer embedders on CUDA to BF16/FP16 to halve VRAM while keeping float fallback."""
    if model is None or "cuda" not in str(device).lower() or not _ensure_torch():
        return model
    try:
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(target_dtype)
        torch.cuda.empty_cache()
    except Exception:
        pass
    return model


class BenchmarkProfile_0(InferenceProfile):
    """
    Profile 0: Pure Symbolic Baseline.
    Pure lexical (BM25 / token intersection). Zero embeddings, zero LLM, zero English hacks.
    """
    def __init__(self):
        self.embedder_name = "none"
        self.llm_name = "none"
        self._dim = 0

    def load_models(self, embedder_name: str = "", llm_name: str = ""):
        pass

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        return []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        return []

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile 0 (Pure Symbolic) does not support neural text generation.")

    def can_synthesize(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return 0


class BenchmarkProfile_B(InferenceProfile):
    """Profile B: Bi-Encoder Dense Embeddings Only (no reranker, no LLM)."""
    def __init__(self):
        self.model = None
        self.embedder = None
        self.llm = None
        self._dim = 384
        self.embedder_name = "default"
        self.llm_name = "none"
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str = ""):
        from sentence_transformers import SentenceTransformer
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        self.embedder_name = select_optimal_embedder(embedder_name)
        self.llm_name = "none"
        self.llm = None
        if self.model is None:
            emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
            device = HardwareProfiler.get_optimal_device()
            task_name = _resolve_embedder_task(emb_path)
            model_kwargs = _build_embedder_model_kwargs(task_name, device)
            try:
                self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
            except Exception:
                fallback_mk = {"default_task": task_name} if task_name else {}
                try:
                    self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=fallback_mk)
                except Exception:
                    self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True)
            self.model = _cast_embedder_half_precision(self.model, device)
            self.embedder = self.model
            self._dim = resolve_embedding_dimension(self.model)
            logger.info(f"[PROFILE B] Loaded embedder '{self.embedder_name}' (dim={self._dim}, no LLM, no reranker) on {device.upper()}")
        else:
            self.embedder = self.model
            if not self._dim:
                self._dim = resolve_embedding_dimension(self.model)

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.model is None:
            return []
        device = getattr(self.model, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.model, texts, mode=mode, batch_size=batch_size, lock=self._lock)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile B (Dense Embeddings Only) does not support text generation.")

    def can_synthesize(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_A(BenchmarkProfile_B):
    """Profile A: Bi-Encoder Dense Embeddings + Cross-Encoder / Listwise Reranker (no LLM)."""

    def uses_reranker(self) -> bool:
        return True

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile A (Dense Embeddings + Reranker) does not support text generation.")


class BenchmarkProfile_C(InferenceProfile):
    """Profile C: Dedicated Embedder + GGUF LLM."""
    def __init__(self):
        self.embedder = None
        self.llm = None
        self._dim = 384
        self.embedder_name = "default"
        self.llm_name = "default"
        self._embedder_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._lock = self._llm_lock  # backward compatibility alias

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        device = HardwareProfiler.get_optimal_device()

        # 1. Load Embedder (skip if pre-attached via ModelManager weight reuse)
        self.embedder_name = select_optimal_embedder(embedder_name)
        if self.embedder is None:
            emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
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

            self._dim = resolve_embedding_dimension(self.embedder)

        # 2. Load LLM (skip if pre-attached via ModelManager weight reuse)
        self.llm_name = select_optimal_llm(llm_name)
        if self.llm is None:
            llm_base_dir = os.path.join(MODELS_DIR, "llms")
            llm_dir = os.path.join(llm_base_dir, self.llm_name)
            if not os.path.isdir(llm_dir):
                raise FileNotFoundError(f"LLM directory not found: {llm_dir}")

            ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
            if not ggufs:
                raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
            unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
            chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
            model_file = os.path.join(llm_dir, chosen_gguf)

            gpu_layers = -1 if device in ("cuda", "mps", "metal", "rocm") else 0
            n_threads = max(1, (os.cpu_count() or 4) - 1)
            self.llm = _instantiate_llama(
                Llama,
                model_path=model_file,
                n_ctx=settings.llm_context_length,
                n_gpu_layers=gpu_layers,
                n_threads=n_threads,
                logits_all=bool(self.has_constrained_choice_scorer()),
                verbose=False,
            )
            logger.info(f"[PROFILE C] Loaded Embedder '{self.embedder_name}' + LLM '{self.llm_name}' on {device.upper()} (gpu_layers={gpu_layers}, n_threads={n_threads})")

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

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "repeat_penalty": 1.15,
        }
        if settings.llm_stop_tokens:
            kwargs["stop"] = list(settings.llm_stop_tokens)
        return _generate_llama_with_optional_schema(
            self.llm, self._llm_lock, messages, kwargs, schema
        )

    def can_synthesize(self) -> bool:
        return True

    def can_extract_slots(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        from language_backend import get_active_language_backend
        lang_name = get_active_language_backend().name or "target"
        prompt = (
            f"Fix the runtime error in this code and return ONLY the complete, standalone corrected code inside ```{lang_name} ```. "
            "Preserve all required imports and ensure every referenced variable is defined before use:\n\n"
            f"ERROR:\n{traceback_error}\n\nCODE:\n```{lang_name}\n{failing_code}\n```"
        )
        try:
            raw = self.generate_text(prompt, max_tokens=1024)
            return extract_code_from_llm_response(raw)
        except Exception as e:
            logger.error(f"[FEEDBACK CHECK ERROR] {e}")
            return failing_code

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts structured semantic intent, parameters, and topological bindings using the LLM.
        Domain-agnostic with customizable schema and prompt contracts.
        """
        target_schema = schema or {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "domain": {"type": "string"},
                "deliverable": {
                    "type": "string",
                    "description": "The primary terminal entity, variable, model, or metric requested by the user."
                },
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit input file names, asset paths, or external sources mentioned in the specification."
                },
                "outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit output file names or destination paths mentioned in the specification (empty list if deliverable is in-memory)."
                },
                "operations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Strictly ordered sequence of atomic, irreducible computational morphisms. Each step must represent a single discrete transformation or calculation."
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All explicit variable, column, feature, channel, and symbol identifiers mentioned in the specification."
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Key-value mapping of execution parameters, numeric values, dimension/coordinate tuples "
                        "(as JSON arrays), booleans, thresholds, and configuration options extracted verbatim from the "
                        "prompt, or sensible domain-standard best estimates when an implied operation omits a required parameter."
                    ),
                },
                "slots": {
                    "type": "object",
                    "description": "Mapping of named entities, symbolic variables, field/feature identifiers, or target variables referenced in the specification."
                },
                "effective_prompt": {"type": "string"},
            },
            "required": ["task", "domain", "deliverable", "operations", "variables", "inputs"]
        }

        target_system_prompt = system_prompt or (
            "You are a formal semantic compiler for computational pipelines.\n"
            "Analyze the given natural language user specification and decompose it strictly into a JSON object adhering to the schema:\n"
            "- task: high-level intent.\n"
            "- domain: problem domain.\n"
            "- deliverable: the primary terminal entity, variable, model, or metric requested.\n"
            "- operations: ordered JSON array of SHORT strings (each a 2-5 word atomic step phrase, e.g. [\"load asset\", \"transform field\"]; NOT objects).\n"
            "- variables: JSON array of distinct column, feature, variable, and parameter strings mentioned verbatim in the prompt.\n"
            "- parameters: JSON object of EVERY explicit numeric value, dimension/coordinate tuple (as JSON arrays), boolean, threshold, or configuration parameter from the prompt; if an implied operation omits a required numeric/configuration parameter, supply a sensible domain-standard best estimate rather than omitting it or returning null.\n"
            "- inputs: JSON array of explicit input file path strings mentioned in the prompt (empty list [] if none).\n"
            "- outputs: JSON array of explicit output destination file path strings mentioned in the prompt (empty list [] if deliverable is in-memory).\n\n"
            "Guidelines:\n"
            "1. Exhaustive Coverage: operations must NEVER be truncated or omitted; every sequential clause in the prompt must have its corresponding concise string in operations.\n"
            "2. Extract ALL variables, identifiers, and files that are explicitly mentioned in the user specification without omitting any identifier or target entity.\n"
            "3. Preserve explicit variable/column/file identifiers verbatim (never invent output filenames unless saving/exporting is requested), while supplying sensible best estimates for omitted required execution parameters.\n"
            "4. Output compact single-line valid JSON only."
        )

        try:
            raw = self.generate_text(prompt, max_tokens=1024, schema=target_schema, system_prompt=target_system_prompt)
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                try:
                    from utils import extract_json_object
                    data = extract_json_object(raw)
                except Exception:
                    pass
            if isinstance(data, dict):
                # Strict literal grounding: destination files must be grounded in prompt text
                raw_outs = data.get("outputs") or []
                valid_outs = []
                prompt_lower = prompt.lower()
                for out_f in raw_outs:
                    if isinstance(out_f, str) and out_f.strip():
                        clean_out = out_f.strip()
                        if clean_out.lower() in prompt_lower:
                            valid_outs.append(clean_out)
                        else:
                            try:
                                from unification import ExecutionContext
                                if any(clean_out.lower() in lit_val.lower() for _, lit_k, lit_val in ExecutionContext._extract_universal_literals(prompt)):
                                    valid_outs.append(clean_out)
                            except Exception:
                                pass
                data["outputs"] = valid_outs
                data["dest_files"] = list(valid_outs)

                # Ground input source_files against prompt text/literals symmetrically
                # so speculative filenames from the small compiler never poison Layer 2 Step 0.
                raw_ins = data.get("source_files") if "source_files" in data else (data.get("inputs") or [])
                if not isinstance(raw_ins, list):
                    raw_ins = [raw_ins] if raw_ins else []
                valid_ins = []
                extracted_lits = None
                for in_f in raw_ins:
                    if isinstance(in_f, str) and in_f.strip():
                        clean_in = in_f.strip()
                        if clean_in.lower() in prompt_lower:
                            valid_ins.append(clean_in)
                        else:
                            try:
                                if extracted_lits is None:
                                    from unification import ExecutionContext
                                    extracted_lits = ExecutionContext._extract_universal_literals(prompt)
                                if any(clean_in.lower() in str(lit_val).lower() for _, _, lit_val in (extracted_lits or ())):
                                    valid_ins.append(clean_in)
                            except Exception:
                                pass
                data["inputs"] = list(valid_ins)
                data["source_files"] = list(valid_ins)
                if "parameters" in data and "hyperparameters" not in data:
                    data["hyperparameters"] = data["parameters"]
                if "variables" in data and isinstance(data["variables"], list):
                    if "slots" not in data or not isinstance(data["slots"], dict):
                        data["slots"] = {}
                    data["slots"]["variables"] = [str(v).strip() for v in data["variables"] if str(v).strip()]
                if "effective_prompt" not in data or not data["effective_prompt"]:
                    data["effective_prompt"] = prompt
                else:
                    try:
                        from utils import ensure_comprehensive_prompt
                        data["effective_prompt"] = ensure_comprehensive_prompt(prompt, data["effective_prompt"])
                    except Exception:
                        pass
                # When the prompt is a single unpunctuated compound sentence yet the compiler extracted
                # multiple atomic operations without losing informative prompt content, emit punctuated clauses.
                try:
                    try:
                        from .tokenizer import CellTokenizer
                        from .utils import ensure_comprehensive_prompt
                    except (ImportError, ValueError):
                        from tokenizer import CellTokenizer
                        from utils import ensure_comprehensive_prompt
                    ops_raw = data.get("operations")
                    if (
                        isinstance(ops_raw, list)
                        and len(CellTokenizer.split_prompt_clauses(data["effective_prompt"])) <= 1
                    ):
                        clean_ops = [str(op).strip(" .;,\t\n") for op in ops_raw if isinstance(op, str) and str(op).strip()]
                        if len(clean_ops) >= 2:
                            ops_joined = "; ".join(clean_ops)
                            if ensure_comprehensive_prompt(prompt, ops_joined) == ops_joined:
                                data["effective_prompt"] = ops_joined
                except Exception:
                    pass
                return data
        except Exception as e:
            logger.debug(f"[LLM] Semantic intent compilation fallback: {e}")

        raise RuntimeError(f"Semantic intent compilation failed for profile {getattr(self, 'llm_name', 'unknown')}.")

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_D(BenchmarkProfile_C):
    """Profile D: Embeddings + Schema-Constrained SLM Slot Extraction Only."""

    def can_extract_slots(self) -> bool:
        return True

    def can_synthesize(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False


class BenchmarkProfile_E(BenchmarkProfile_C):
    """Profile E: Translator Pass + Semantic Intent Compiler + Neuro-Symbolic Synthesis."""

    def has_semantic_compiler(self) -> bool:
        return True

    def has_translator_pass(self) -> bool:
        return True

    def has_ir_compiler(self) -> bool:
        return False


class BenchmarkProfile_S(BenchmarkProfile_E):
    """Profile S: Typed IR Compiler + Semantic Compiler + Translator + Neuro-Symbolic Synthesis."""

    def has_semantic_compiler(self) -> bool:
        return True

    def has_translator_pass(self) -> bool:
        return True

    def has_ir_compiler(self) -> bool:
        return True

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        return super().compile_semantic_intent(prompt, schema=schema, system_prompt=system_prompt)


class BenchmarkProfile_U(InferenceProfile):
    """
    Profile U: Single-Model Early-Exit Layer Tapping.
    Cuts VRAM footprint in half by loading only ONE LLM.
    Uses intermediate layer k (e.g. k = floor(N/2)) for vector embeddings/RAG,
    and all N layers for generation/unfolding.
    """

    def __init__(self, early_exit_layer: Optional[int] = None):
        self.llm = None
        self.model = None
        self.tokenizer = None
        self._dim = 896
        self.early_exit_layer = early_exit_layer
        self.k_layer = 0
        self.early_exit_active: bool = False
        self.embedder_name = "llm_early_exit"
        self.llm_name = "default"
        self.is_torch = False
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str):
        self.llm_name = select_optimal_llm(llm_name)
        llm_base_dir = os.path.join(MODELS_DIR, "llms")
        llm_dir = os.path.join(llm_base_dir, self.llm_name)

        device = "cpu"
        try:
            from .router import HardwareProfiler
            device = HardwareProfiler.get_optimal_device()
        except Exception:
            try:
                from router import HardwareProfiler
                device = HardwareProfiler.get_optimal_device()
            except Exception:
                pass

        loaded = False
        # 1. Attempt GGUF / llama_cpp loading with embedding=True
        if os.path.isdir(llm_dir):
            ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
            if ggufs:
                try:
                    import llama_cpp
                    from llama_cpp import Llama
                    unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
                    chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
                    gguf_path = os.path.join(llm_dir, chosen_gguf)
                    n_gpu = -1 if device in ("cuda", "mps", "metal", "rocm") else 0
                    n_threads = max(1, (os.cpu_count() or 4) - 1)
                    mean_pooling = getattr(llama_cpp, "LLAMA_POOLING_TYPE_MEAN", 1)
                    self.llm = _instantiate_llama(
                        Llama,
                        model_path=gguf_path,
                        n_ctx=settings.llm_context_length,
                        n_gpu_layers=n_gpu,
                        n_threads=n_threads,
                        embedding=True,  # Enable dual embedding + completion from single instance
                        pooling_type=mean_pooling,
                        verbose=False,
                    )
                    try:
                        test_emb = self.llm.create_embedding("probe")
                        probe_arr = np.asarray(test_emb["data"][0]["embedding"], dtype=np.float32)
                        self._dim = int(probe_arr.shape[-1]) if probe_arr.ndim >= 1 and probe_arr.size > 0 else 896
                    except Exception:
                        self._dim = getattr(self.llm, "n_embd", lambda: 896)() or 896
                    self.is_torch = False
                    self.early_exit_active = False
                    loaded = True
                    logger.info(
                        f"[PROFILE U] Loaded single GGUF LLM '{self.llm_name}' (dim={self._dim}, "
                        "embedding=True, early_exit_active=False full-layer pooling) on "
                        f"{device.upper()}"
                    )
                except Exception as e:
                    logger.debug(f"[PROFILE U] GGUF loading skipped: {e}")

        # 2. Attempt PyTorch / HuggingFace model loading
        if not loaded and _ensure_torch():
            try:
                if os.path.isdir(llm_dir) and any(os.path.exists(os.path.join(llm_dir, f)) for f in ("config.json", "model.safetensors")):
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(llm_dir, trust_remote_code=True).to(device)
                    self.model.eval()
                    total_layers = len(getattr(getattr(self.model, "model", self.model), "layers", []))
                    raw_k = self.early_exit_layer or max(1, total_layers // 2)
                    self.k_layer = max(1, min(int(raw_k), total_layers)) if total_layers > 0 else 0
                    self.early_exit_active = bool(total_layers > 0 and 0 < self.k_layer < total_layers)
                    self._dim = getattr(self.model.config, "hidden_size", 896)
                    self.is_torch = True
                    loaded = True
                    logger.info(
                        f"[PROFILE U] Loaded single PyTorch LLM '{self.llm_name}' "
                        f"(k={self.k_layer}/{total_layers}, early_exit_active={self.early_exit_active}, "
                        f"dim={self._dim}) on {device.upper()}"
                    )
            except Exception as e:
                logger.debug(f"[PROFILE U] PyTorch model loading skipped: {e}")

        if not loaded:
            raise FileNotFoundError(
                f"Profile U failed to load local GGUF or PyTorch LLM '{self.llm_name}' from '{llm_dir}'."
            )

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            # 1. PyTorch early-exit forward pass through layers[:k]
            if self.is_torch and self.model is not None and self.tokenizer is not None:
                _ensure_torch()
                device = getattr(self.model, "device", "cpu")
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    core_model = getattr(self.model, "model", self.model)
                    hidden = core_model.embed_tokens(inputs.input_ids)
                    seq_len = inputs.input_ids.shape[1]
                    batch_size = inputs.input_ids.shape[0]
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_len)
                    rotary_pos_emb = None
                    if hasattr(core_model, "rotary_emb"):
                        try:
                            rotary_pos_emb = core_model.rotary_emb(hidden, position_ids)
                        except Exception:
                            rotary_pos_emb = None
                    causal_mask = None
                    attn_2d = getattr(inputs, "attention_mask", None)
                    if attn_2d is not None:
                        min_val = torch.finfo(hidden.dtype).min if hidden.dtype.is_floating_point else -1e9
                        causal_base = torch.triu(
                            torch.full((seq_len, seq_len), min_val, dtype=hidden.dtype, device=device),
                            diagonal=1,
                        ).unsqueeze(0).unsqueeze(0)
                        pad_mask = (1.0 - attn_2d.to(dtype=hidden.dtype)).unsqueeze(1).unsqueeze(2) * min_val
                        causal_mask = causal_base + pad_mask

                    layers = getattr(core_model, "layers", [])
                    total_layers = len(layers)
                    k = max(1, min(int(self.k_layer or total_layers), total_layers)) if total_layers > 0 else 0
                    self.early_exit_active = bool(total_layers > 0 and 0 < k < total_layers)
                    for layer in layers[:k]:
                        layer_kwargs: Dict[str, Any] = {}
                        try:
                            sig = inspect.signature(layer.forward)
                            params = sig.parameters
                            if "position_ids" in params:
                                layer_kwargs["position_ids"] = position_ids
                            if "position_embeddings" in params and rotary_pos_emb is not None:
                                layer_kwargs["position_embeddings"] = rotary_pos_emb
                            if "attention_mask" in params and causal_mask is not None:
                                layer_kwargs["attention_mask"] = causal_mask
                        except (TypeError, ValueError):
                            layer_kwargs = {"position_ids": position_ids}
                        layer_out = layer(hidden, **layer_kwargs)
                        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    if hasattr(core_model, "norm"):
                        hidden = core_model.norm(hidden)
                    mask = inputs.attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_emb = torch.sum(hidden * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    pooled = sum_emb / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return normalized.detach().cpu().numpy().tolist()

            # 2. GGUF / llama_cpp native embedding pooling endpoint
            if self.llm is not None and hasattr(self.llm, "create_embedding"):
                embeddings = []
                for t in texts:
                    res = self.llm.create_embedding(t)
                    emb = res["data"][0]["embedding"]
                    arr = np.asarray(emb, dtype=np.float32)
                    if arr.ndim > 1:
                        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                    norm = float(np.linalg.norm(arr))
                    if norm > 1e-9:
                        arr = arr / norm
                    embeddings.append(arr.tolist())
                return embeddings

            raise RuntimeError("Profile U embeddings require a loaded PyTorch or GGUF LLM.")

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        if self.is_torch and self.model is not None and self.tokenizer is not None:
            with self._lock:
                device = getattr(self.model, "device", "cpu")
                full_p = f"{system_prompt}\n{prompt}" if system_prompt else prompt
                inputs = self.tokenizer(full_p, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=max_tokens)
                decoded = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return _strip_thinking_blocks(decoded) if schema is not None else decoded

        if self.llm is not None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "repeat_penalty": 1.15,
            }
            if settings.llm_stop_tokens:
                kwargs["stop"] = list(settings.llm_stop_tokens)
            return _generate_llama_with_optional_schema(
                self.llm, self._lock, messages, kwargs, schema
            )
        raise RuntimeError("Profile U LLM is not loaded.")

    def can_synthesize(self) -> bool:
        return True

    def can_extract_slots(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        from language_backend import get_active_language_backend
        lang_name = get_active_language_backend().name or "target"
        prompt = (
            f"Fix the runtime error in this code and return ONLY the complete, standalone corrected code inside ```{lang_name} ```. "
            "Preserve all required imports and ensure every referenced variable is defined before use:\n\n"
            f"ERROR:\n{traceback_error}\n\nCODE:\n```{lang_name}\n{failing_code}\n```"
        )
        try:
            raw = self.generate_text(prompt, max_tokens=1024)
            repaired = extract_code_from_llm_response(raw)
            return repaired if repaired else failing_code
        except Exception as e:
            logger.error(f"[PROFILE U FEEDBACK CHECK ERROR] {e}")
            return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_AE(InferenceProfile):
    """
    Profile AE: Self-Supervised Structural Transition Autoencoder Manifold.
    Projects discrete node signatures and typestates into a continuous transition manifold.
    Enables routing via fast continuous vector arithmetic: z_{tau_in} + t_c ≈ z_{tau_out}.
    """

    def __init__(self, dimension: int = 64):
        self.autoencoder: Optional[Any] = None
        self._dim = dimension
        self.embedder_name = "transition_autoencoder"
        self.llm_name = "none"
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str = "", llm_name: str = ""):
        try:
            from .latent_autoencoder import TransitionAutoencoder, train_autoencoder_from_orchestrator
        except (ImportError, ValueError):
            from latent_autoencoder import TransitionAutoencoder, train_autoencoder_from_orchestrator

        orch = None
        try:
            try:
                from .lattice import LatticeOrchestrator
            except (ImportError, ValueError):
                from lattice import LatticeOrchestrator
            orch = LatticeOrchestrator.get_active_instance()
        except Exception:
            orch = None

        if orch is None or not getattr(orch, "loaded_cells", None):
            try:
                from .main import get_orchestrator
                orch = get_orchestrator()
            except Exception:
                try:
                    from main import get_orchestrator
                    orch = get_orchestrator()
                except Exception:
                    orch = None

        if orch is not None and getattr(orch, "loaded_cells", None):
            self.autoencoder = train_autoencoder_from_orchestrator(orch, dimension=self._dim)
        else:
            raise RuntimeError("Profile AE requires loaded orchestrator cells to train the transition autoencoder.")

        logger.info(f"[PROFILE AE] Initialized Structural Transition Autoencoder (dim={self._dim})")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.autoencoder is None:
            return []
        with self._lock:
            out: List[List[float]] = []
            ts_map = getattr(self.autoencoder, "typestate_to_idx", {}) or {}
            txt_trained = bool(getattr(self.autoencoder, "_text_projection_trained", False))
            for t in texts:
                if t in ts_map:
                    out.append(self.autoencoder.encode_signature(t).tolist())
                elif txt_trained and hasattr(self.autoencoder, "encode_text"):
                    out.append(self.autoencoder.encode_text(t).tolist())
                else:
                    out.append(self.autoencoder.encode_signature(t).tolist())
            return out

    def score_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
        target_vec: Union[np.ndarray, str, Any],
    ) -> float:
        if self.autoencoder is None:
            return 0.0
        with self._lock:
            return self.autoencoder.score_transition(state_vec, cell_id, target_vec)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile AE is a continuous transition manifold and does not support generative text.")

    def can_synthesize(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_Omega(BenchmarkProfile_S):
    """
    Profile Omega (O / Ω): Full STRATA 4-Stage Formal Neuro-Symbolic Pipeline.
      - Stage 1: Recursive Coalgebraic Subgoal Unfolding (open -> semi-closed -> closed)
      - Stage 2: Calibrated Sigmoid Hybrid Retrieval + Backward Precondition Cone
      - Stage 3: Type-Directed Bidirectional A* Search over Symmetric Monoidal Category
                 with schema-constrained SLM ranking over E_valid(Gamma_t)
      - Stage 4: Exact Robinson Unification + Profunctor Lens Lifting + GBNF Slot Extraction
                 + Pointed Monoidal Defaults eta_tau : 1 -> tau
    """

    def uses_reranker(self) -> bool:
        return True

    def has_coalgebraic_unfolder(self) -> bool:
        return True

    def has_constrained_choice_scorer(self) -> bool:
        return True

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        """
        Single-pass 1-token option-label logprob choice scorer restricted strictly to E_valid(Gamma_t).
        Maps labels (A, B, ...) via log-sum-exp and numerically stable softmax into calibrated
        probabilities summing to 1.0 over `candidates[:12]`, or {} on empty/missing logprobs.
        """
        clean_cands: List[str] = []
        seen = set()
        for c in (candidates or []):
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
        usr_p = (
            f"Goal: {prompt}{ctx_line}\n"
            f"Options:\n{opts_block}\n"
            f"Answer ({labels[0]}-{labels[-1]}):"
        )
        call_kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
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
            logger.debug(f"[OMEGA] Constrained logprob choice scoring skipped: {err}")
            return {}


class ModelManager:
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self.active_profile: Optional[InferenceProfile] = None
        self.current_profile_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> ModelManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def profile(self) -> Optional[InferenceProfile]:
        with self._lock:
            return self.active_profile

    def cleanup(self, release_reranker: bool = True):
        """Forces complete teardown of active models, releasing all GPU VRAM and CPU RAM."""
        with self._lock:
            if self.active_profile is not None:
                if hasattr(self.active_profile, 'llm') and self.active_profile.llm is not None:
                    try:
                        self.active_profile.llm.close()
                    except Exception:
                        pass
                    self.active_profile.llm = None
                if hasattr(self.active_profile, 'embedder') and self.active_profile.embedder is not None:
                    self.active_profile.embedder = None
                if hasattr(self.active_profile, 'model') and self.active_profile.model is not None:
                    self.active_profile.model = None
                self.active_profile = None
            self.current_profile_name = None
            try:
                from cognitive_pathfinder import LatentVectorExtractor
                LatentVectorExtractor.reset_instance()
            except Exception:
                pass
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
            logger.info("[MODEL MANAGER] Completed VRAM cleanup and memory reclamation.")

    def initialize_profile(self, profile_type: str, embedder_name: str = "", llm_name: str = ""):
        with self._lock:
            from config import resolve_profile_spec
            p_type = resolve_profile_spec(profile_type).code
            profile_classes = {
                "0": BenchmarkProfile_0,
                "A": BenchmarkProfile_A,
                "B": BenchmarkProfile_B,
                "C": BenchmarkProfile_C,
                "D": BenchmarkProfile_D,
                "E": BenchmarkProfile_E,
                "S": BenchmarkProfile_S,
                "U": BenchmarkProfile_U,
                "AE": BenchmarkProfile_AE,
                "O": BenchmarkProfile_Omega,
            }
            prof = profile_classes[p_type]()

            # Transfer matching resident weights across compatible profile switches to avoid reload/VRAM churn
            prev = self.active_profile
            if prev is not None and isinstance(prof, (BenchmarkProfile_B, BenchmarkProfile_C)):
                if isinstance(prev, (BenchmarkProfile_B, BenchmarkProfile_C)) and getattr(prev, "embedder", None) is not None:
                    target_emb = select_optimal_embedder(embedder_name)
                    if getattr(prev, "embedder_name", None) == target_emb:
                        reused_emb = prev.embedder
                        reused_dim = getattr(prev, "_dim", 384)
                        prev.embedder = None
                        if hasattr(prev, "model"):
                            prev.model = None
                        prof.embedder = reused_emb
                        prof.embedder_name = target_emb
                        prof._dim = reused_dim
                        if isinstance(prof, BenchmarkProfile_B):
                            prof.model = reused_emb
                if isinstance(prof, BenchmarkProfile_C) and isinstance(prev, BenchmarkProfile_C) and getattr(prev, "llm", None) is not None:
                    target_llm = select_optimal_llm(llm_name)
                    if getattr(prev, "llm_name", None) == target_llm and (
                        (not prof.has_constrained_choice_scorer())
                        or bool(getattr(prev.llm, "_logits_all", False))
                    ):
                        prof.llm = prev.llm
                        prof.llm_name = target_llm
                        prev.llm = None

            # Clean up only remaining non-reused weights on previous profile
            needs_cleanup = prev is not None and (
                getattr(prev, "llm", None) is not None
                or getattr(prev, "embedder", None) is not None
                or getattr(prev, "model", None) is not None
                or getattr(prev, "autoencoder", None) is not None
            )
            low_vram = os.environ.get("NSTL_LOW_VRAM") == "1"
            release_rr = low_vram and not prof.uses_reranker()
            if needs_cleanup:
                self.cleanup(release_reranker=release_rr)
            else:
                self.active_profile = None
                self.current_profile_name = None
                if release_rr:
                    try:
                        try:
                            from .reranker import LocalReranker
                        except (ImportError, ValueError):
                            from reranker import LocalReranker
                        LocalReranker.release_shared()
                    except Exception:
                        pass

            try:
                prof.load_models(embedder_name, llm_name)
                self.active_profile = prof
                self.current_profile_name = p_type
                gc.collect()
                _trim_host_memory()
            except Exception as e:
                self.cleanup()
                raise e

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        with self._lock:
            return self.active_profile.get_embedding(text, mode=mode) if self.active_profile else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        with self._lock:
            return self.active_profile.get_embeddings(texts, mode=mode) if self.active_profile else []

    def get_embeddings_batch(self, texts: List[str], mode: str = "query") -> np.ndarray:
        with self._lock:
            if not self.active_profile or not texts:
                dim = getattr(self.active_profile, "_dim", 384) if self.active_profile else 384
                return np.empty((0, dim), dtype=np.float32)
            res = self.active_profile.get_embeddings(texts, mode=mode)
            return np.array(res, dtype=np.float32)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        with self._lock:
            if not self.active_profile:
                return ""
            return self.active_profile.generate_text(prompt, max_tokens, schema, system_prompt=system_prompt)

    def can_synthesize(self) -> bool:
        with self._lock:
            return self.active_profile.can_synthesize() if self.active_profile else False

    def can_extract_slots(self) -> bool:
        with self._lock:
            if not self.active_profile:
                return False
            if hasattr(self.active_profile, "can_extract_slots"):
                return self.active_profile.can_extract_slots()
            return self.active_profile.can_synthesize()

    def is_pure_symbolic(self) -> bool:
        with self._lock:
            return self.current_profile_name == "0"

    def uses_reranker(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "uses_reranker", lambda: False)() if self.active_profile else False

    def has_ir_compiler(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_ir_compiler", lambda: False)() if self.active_profile else False

    def has_coalgebraic_unfolder(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_coalgebraic_unfolder", lambda: False)() if self.active_profile else False

    def has_constrained_choice_scorer(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_constrained_choice_scorer", lambda: False)() if self.active_profile else False

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "score_choices"):
                return self.active_profile.score_choices(prompt, candidates, context_desc=context_desc)
            return {}

    def has_embedder(self) -> bool:
        with self._lock:
            if not self.active_profile:
                return False
            if hasattr(self.active_profile, "has_embedder"):
                return self.active_profile.has_embedder()
            return getattr(self.active_profile, "embedding_dimension", 0) > 0

    def can_feedback_check(self) -> bool:
        with self._lock:
            return self.active_profile.can_feedback_check() if self.active_profile else False

    def has_translator_pass(self) -> bool:
        with self._lock:
            return self.active_profile.has_translator_pass() if self.active_profile else False

    def has_semantic_compiler(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_semantic_compiler", lambda: False)() if self.active_profile else False

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "compile_semantic_intent"):
                return self.active_profile.compile_semantic_intent(prompt, schema=schema, system_prompt=system_prompt)
            return {"effective_prompt": prompt}

    def score_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
        target_vec: Union[np.ndarray, str, Any],
    ) -> float:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "score_transition"):
                return self.active_profile.score_transition(state_vec, cell_id, target_vec)
            return 0.0

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        with self._lock:
            return self.active_profile.feedback_check(failing_code, traceback_error) if self.active_profile else failing_code

    @property
    def embedding_dimension(self) -> int:
        with self._lock:
            return self.active_profile.embedding_dimension if self.active_profile else 384
