"""
src/utils.py — Minimal shared utilities for STRATA without regex or domain assumptions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from .language_backend import get_active_language_backend
    from .log_config import get_logger
except (ImportError, ValueError):
    from language_backend import get_active_language_backend
    from log_config import get_logger

logger = get_logger("utils")


def extract_template_placeholders(template: str) -> List[str]:
    """Extract valid identifier placeholder names from a code template without regex."""
    if not template or "{" not in template:
        return []
    placeholders: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{":
            j = template.find("}", i + 1)
            if j != -1:
                inner = template[i + 1 : j]
                if inner.isidentifier():
                    placeholders.append(inner)
                    i = j + 1
                    continue
        i += 1
    return placeholders


def is_split_gguf_shard(filename: str) -> bool:
    """Return True iff `filename` follows the split GGUF naming pattern `<stem>-00001-of-00005.gguf`."""
    if not (filename.endswith(".gguf") and "-of-" in filename):
        return False
    stem = filename[:-5]
    parts = stem.rsplit("-", 2)
    if len(parts) >= 3 and parts[-2] == "of":
        return parts[-3].isdigit() and parts[-1].isdigit()
    return False


def extract_json_from_llm(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from raw LLM output (balanced-brace state machine)."""
    if not raw:
        return None
    text = raw.strip()

    if text.startswith("```"):
        first_line_end = text.find("\n")
        text = text[first_line_end + 1 :] if first_line_end != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    n = len(text)
    start = 0
    while start < n:
        brace_pos = text.find("{", start)
        if brace_pos == -1:
            break

        depth = 0
        in_string = False
        escape = False
        for idx in range(brace_pos, n):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_pos : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        break
        start = brace_pos + 1

    logger.debug("[UTILS] Failed to extract JSON object from text (%d chars)", len(text))
    return None


extract_json_object = extract_json_from_llm


def validate_code_template(template: str) -> bool:
    """Validate a code template via the active LanguageBackend."""
    if not template or not template.strip():
        return False
    ok, _ = get_active_language_backend().validate_syntax(template, allow_placeholders=True)
    return ok


__all__ = [
    "extract_template_placeholders",
    "is_split_gguf_shard",
    "extract_json_from_llm",
    "extract_json_object",
    "validate_code_template",
]
