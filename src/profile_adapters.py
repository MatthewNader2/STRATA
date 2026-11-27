"""
src/profile_adapters.py — Capability-driven profile adapters wiring model interfaces into StrataPipeline.

Maps ProfileSpec orthogonal capability flags (`uses_embedder`, `uses_reranker`,
`uses_llm`, `uses_slot_fill`, `uses_choice_scorer`) directly to callbacks.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from .config import settings
    from .schema import (
        coerce_literal_for_port,
        derive_value_class,
        get_attr,
        is_value_compatible_with_primitive_carrier,
    )
    from .strata_pipeline import StrataPipeline
    from .utils import extract_json_object
except (ImportError, ValueError):
    from config import settings
    from schema import (
        coerce_literal_for_port,
        derive_value_class,
        get_attr,
        is_value_compatible_with_primitive_carrier,
    )
    from strata_pipeline import StrataPipeline
    from utils import extract_json_object


_SLOT_FILL_SYSTEM_PROMPT = (
    "You fill non-asset configuration parameters for a typed operation. "
    "Never invent file paths or external resources. Return only a single valid JSON object."
)


def make_embed_fn(model_manager: Any) -> Optional[Callable[..., List[List[float]]]]:
    """Build an asymmetric embedding callback supporting `mode='query'|'document'`."""
    if model_manager is None or not getattr(model_manager, "has_embedder", lambda: False)():
        return None

    def _embed(texts: List[str], *, mode: str = "document") -> List[List[float]]:
        return model_manager.get_embeddings(list(texts), mode=mode)

    return _embed


def make_rerank_fn(reranker: Any) -> Optional[Callable[[str, List[str]], List[float]]]:
    """Build a listwise/cross-encoder document scorer callback when a reranker model is loaded."""
    if reranker is None or getattr(reranker, "model", None) is None or not hasattr(reranker, "score_documents"):
        return None

    def _rerank(query: str, docs: List[str]) -> List[float]:
        return reranker.score_documents(query, list(docs))

    return _rerank


def make_generate_fn(model_manager: Any) -> Optional[Callable[..., str]]:
    """Build a schema-constrained text generation callback for SLM/LLM TypedIR compilation."""
    if model_manager is None or not getattr(model_manager, "has_llm", lambda: False)():
        return None

    accepts_few_shot = False
    gen_method = getattr(model_manager, "generate_text", None)
    if callable(gen_method):
        try:
            sig = inspect.signature(gen_method)
            accepts_few_shot = "few_shot" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_few_shot = False

    def _generate(
        user_prompt: str,
        json_schema: Dict[str, Any],
        system_prompt: str = "",
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        eff_max_tokens = int(getattr(settings, "llm_max_tokens", 1024) or 1024)
        if accepts_few_shot and few_shot is not None:
            return model_manager.generate_text(
                user_prompt,
                max_tokens=eff_max_tokens,
                schema=json_schema,
                system_prompt=system_prompt,
                few_shot=few_shot,
            )
        return model_manager.generate_text(
            user_prompt,
            max_tokens=eff_max_tokens,
            schema=json_schema,
            system_prompt=system_prompt,
        )

    return _generate


def make_choice_scorer_fn(model_manager: Any) -> Optional[Callable[[str, List[str]], List[float]]]:
    """Build a calibrated constrained-choice scorer callback when supported by the profile."""
    if model_manager is None or not getattr(model_manager, "has_choice_scorer", lambda: False)():
        return None

    def _score_choices(prompt: str, candidates: List[str]) -> List[float]:
        cand_list = list(candidates)
        score_map = model_manager.score_choices(prompt, cand_list) or {}
        return [float(score_map.get(c, 0.0)) for c in cand_list]

    return _score_choices


def _carrier_to_json_schema(carrier: Any, enum_vals: Any = None) -> Dict[str, Any]:
    """Map an abstract STRATA port carrier and optional enum choices to JSON Schema constraints."""
    if enum_vals:
        return {"enum": list(enum_vals)}
    base = str(carrier or "").split("[", 1)[0].strip().lower()
    if base in ("int", "integer"):
        return {"type": "integer"}
    if base in ("float", "number", "scalar"):
        return {"type": "number"}
    if base in ("bool", "boolean"):
        return {"type": "boolean"}
    if base in ("str", "string"):
        return {"type": "string"}
    if base in ("list", "tuple", "sequence", "array"):
        return {"type": "array", "items": {"type": ["string", "number", "integer", "boolean"]}}
    if base in ("dict", "mapping"):
        return {"type": "object"}
    return {}


def make_slot_fill_fn(model_manager: Any) -> Optional[Callable[[Any, List[str], Optional[Any]], Dict[str, Any]]]:
    """Build a schema-constrained slot filler strictly restricted to non-asset configuration ports."""
    if model_manager is None or not getattr(model_manager, "has_llm", lambda: False)():
        return None

    def _fill_slots(cell: Any, missing_ports: List[str], ir_node: Optional[Any] = None) -> Dict[str, Any]:
        raw_ins = get_attr(cell, "inputs", {}) or {}
        if not isinstance(raw_ins, dict):
            return {}

        eligible_props: Dict[str, Dict[str, Any]] = {}
        prompt_lines: List[str] = []
        for port_name in (missing_ports or ()):
            p_key = str(port_name)
            p_obj = raw_ins.get(p_key)
            if p_obj is None or derive_value_class(p_obj) == "external_asset":
                continue
            carrier = get_attr(p_obj, "carrier", None) or get_attr(p_obj, "type_name", "any")
            enum_vals = get_attr(p_obj, "enum_values", None)
            eligible_props[p_key] = _carrier_to_json_schema(carrier, enum_vals)
            desc = str(get_attr(p_obj, "description", "") or "").strip()
            line = f"- {p_key} ({carrier}): {desc}" if desc else f"- {p_key} ({carrier})"
            if enum_vals:
                line += f" [allowed: {list(enum_vals)}]"
            prompt_lines.append(line)

        if not eligible_props:
            return {}

        slot_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": eligible_props,
            "required": sorted(eligible_props.keys()),
        }
        cid = str(get_attr(cell, "cell_id", "") or "").strip()
        intent_str = str(getattr(ir_node, "intent", "") or "")
        slot_prompt = (
            f"Operation: {cid}\n"
            f"Intent: {intent_str}\n"
            "Fill concrete JSON literal values for missing parameters (do NOT echo type metadata objects):\n"
            + "\n".join(prompt_lines)
        )
        raw_text = model_manager.generate_text(
            slot_prompt,
            max_tokens=256,
            schema=slot_schema,
            system_prompt=_SLOT_FILL_SYSTEM_PROMPT,
        )
        parsed = extract_json_object(str(raw_text or ""))
        if not isinstance(parsed, dict):
            return {}
        parsed_clean: Dict[str, Any] = {}
        for k in eligible_props:
            if k not in parsed or parsed[k] is None:
                continue
            val = parsed[k]
            p_obj = raw_ins[k]
            coerced = coerce_literal_for_port(val, p_obj)
            c_base = str(get_attr(p_obj, "carrier", None) or "").split("[", 1)[0].strip().lower()
            if isinstance(coerced, dict) and c_base not in ("dict", "mapping", "object", "any", ""):
                continue
            if not is_value_compatible_with_primitive_carrier(coerced, p_obj):
                continue
            parsed_clean[k] = val
        return parsed_clean

    return _fill_slots


def make_sandbox_exec_fn(sandbox: Any, timeout: float = 30.0) -> Optional[Callable[..., Dict[str, Any]]]:
    """Build a declarative sandbox execution adapter forwarding runtime_setups, input_assets, and egress_paths."""
    if sandbox is None:
        return None

    def _exec(
        code: str,
        *,
        runtime_setups: Optional[List[Dict[str, Any]]] = None,
        input_assets: Optional[List[str]] = None,
        egress_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return sandbox.execute(
            code,
            timeout=timeout,
            runtime_setups=runtime_setups,
            input_assets=input_assets,
            egress_paths=egress_paths,
        )

    return _exec


def _resolve_profile_callbacks(
    profile_spec: Any,
    model_manager: Any = None,
    *,
    reranker: Any = None,
    sandbox: Any = None,
    timeout: float = 30.0,
    embed_fn_override: Optional[Callable[..., List[List[float]]]] = None,
) -> Dict[str, Any]:
    """Resolve capability-gated callbacks from a ProfileSpec."""
    has_llm = bool(getattr(profile_spec, "uses_llm", False))
    embed_fn = (
        embed_fn_override
        if (embed_fn_override is not None and getattr(profile_spec, "uses_embedder", False))
        else (make_embed_fn(model_manager) if getattr(profile_spec, "uses_embedder", False) else None)
    )
    return {
        "has_llm": has_llm,
        "embed_fn": embed_fn,
        "rerank_fn": make_rerank_fn(reranker) if getattr(profile_spec, "uses_reranker", False) else None,
        "generate_fn": make_generate_fn(model_manager) if has_llm else None,
        "slot_fill_fn": make_slot_fill_fn(model_manager) if getattr(profile_spec, "uses_slot_fill", False) else None,
        "choice_scorer_fn": (
            make_choice_scorer_fn(model_manager)
            if getattr(profile_spec, "uses_choice_scorer", False)
            else None
        ),
        "sandbox_exec_fn": make_sandbox_exec_fn(sandbox, timeout=timeout),
    }


def build_strata_pipeline_for_profile(
    compiled: Any,
    profile_spec: Any,
    model_manager: Any = None,
    *,
    reranker: Any = None,
    sandbox: Any = None,
    timeout: float = 30.0,
    max_bridge_hops: Optional[int] = None,
    embed_fn_override: Optional[Callable[..., List[List[float]]]] = None,
) -> StrataPipeline:
    """Construct a StrataPipeline wired strictly according to ProfileSpec capability flags."""
    callbacks = _resolve_profile_callbacks(
        profile_spec,
        model_manager=model_manager,
        reranker=reranker,
        sandbox=sandbox,
        timeout=timeout,
        embed_fn_override=embed_fn_override,
    )
    return StrataPipeline.from_compiled(
        compiled,
        max_bridge_hops=max_bridge_hops,
        **callbacks,
    )


def build_strata_pipeline_from_trees(
    trees: Iterable[Any],
    profile_spec: Any,
    model_manager: Any = None,
    *,
    reranker: Any = None,
    sandbox: Any = None,
    timeout: float = 30.0,
    max_bridge_hops: Optional[int] = None,
    embed_fn_override: Optional[Callable[..., List[List[float]]]] = None,
) -> StrataPipeline:
    """Construct a StrataPipeline from explicit tree manifests strictly according to ProfileSpec flags."""
    callbacks = _resolve_profile_callbacks(
        profile_spec,
        model_manager=model_manager,
        reranker=reranker,
        sandbox=sandbox,
        timeout=timeout,
        embed_fn_override=embed_fn_override,
    )
    return StrataPipeline.from_trees(
        trees,
        max_bridge_hops=max_bridge_hops,
        **callbacks,
    )


__all__ = [
    "make_embed_fn",
    "make_rerank_fn",
    "make_generate_fn",
    "make_choice_scorer_fn",
    "make_slot_fill_fn",
    "make_sandbox_exec_fn",
    "build_strata_pipeline_for_profile",
    "build_strata_pipeline_from_trees",
]
