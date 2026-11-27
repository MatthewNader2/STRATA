"""Frontends converting user specifications into TypedIR."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from .formal_regex import FormalParseResult, parse_formal_prompt
from .slm_compiler import (
    FrontendCompileResult,
    compile_slm_typed_ir,
    parse_slm_json_to_typed_ir,
)


def compile_prompt_to_ir(
    prompt: str,
    *,
    has_llm: bool,
    generate_fn: Optional[Callable[[str, Dict[str, Any], str], str]] = None,
    loaded_domains: Iterable[str] = (),
) -> FrontendCompileResult:
    """
    Single frontend entrypoint routing non-LLM profiles strictly through
    the formal regex parser and SLM/LLM profiles through structured TypedIR compilation.
    """
    if not has_llm:
        f_res = parse_formal_prompt(prompt)
        return FrontendCompileResult(ok=f_res.ok, ir=f_res.ir, error=f_res.error, warnings=[])
    if generate_fn is None:
        return FrontendCompileResult(
            ok=False,
            ir=None,
            error="LLM generate_fn is required when has_llm=True.",
        )
    return compile_slm_typed_ir(prompt, generate_fn, loaded_domains=loaded_domains)


__all__ = [
    "FormalParseResult",
    "parse_formal_prompt",
    "FrontendCompileResult",
    "compile_slm_typed_ir",
    "parse_slm_json_to_typed_ir",
    "compile_prompt_to_ir",
]
