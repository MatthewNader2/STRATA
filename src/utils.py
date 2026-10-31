"""
src/utils.py - Shared utilities for NSTL
Extracts common patterns to eliminate code duplication.
"""

from __future__ import annotations
import ast
import json
import string
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    from log_config import get_logger
except ImportError:
    from .log_config import get_logger

logger = get_logger('utils')

# Translation table mapping all ASCII punctuation (except underscore) to spaces for fast tokenization
_PUNCT_EXCEPT_UNDERSCORE = "".join(c for c in string.punctuation if c != "_")
_DELIM_TRANS = str.maketrans({c: " " for c in _PUNCT_EXCEPT_UNDERSCORE})


def extract_template_placeholders(template: str) -> List[str]:
    """Extracts valid Python identifier placeholder names from a code template without regex."""
    if not template or "{" not in template:
        return []
    placeholders: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{":
            j = template.find("}", i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier():
                    placeholders.append(inner)
                i = j + 1
                continue
        i += 1
    return placeholders


def safe_substitute_template(template: str, bindings: Dict[str, Any]) -> str:
    """Substitutes placeholders with bindings, preserving dict literals and syntax.

    Leaves non-identifier braces (like {"a": 1}) and unbound placeholders untouched.
    Zero regular expressions.

    Callee-position rule (host-language semantics): a placeholder immediately
    followed by ``(`` is a CALL CALLEE in the emitted code, so its binding must
    be a bare identifier. When the supplied binding is a quoted string, the
    quotes are stripped so the emitted call names the identifier instead of
    invoking a string constant (which can never execute).
    """
    if not template or "{" not in template:
        return template
    res = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{":
            j = template.find("}", i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier() and inner in bindings and bindings[inner] is not None:
                    val = bindings[inner]
                    is_callee = j + 1 < n and template[j + 1] == "("
                    if is_callee and isinstance(val, str):
                        stripped = val.strip()
                        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
                            stripped = stripped[1:-1]
                        if stripped.isidentifier():
                            val = stripped
                    res.append(str(val))
                    i = j + 1
                    continue
        res.append(template[i])
        i += 1
    return "".join(res)


def tokenize_alphanumeric(text: str, min_len: int = 2) -> List[str]:
    """Tokenizes text into lowercase alphanumeric + underscore tokens without regex.
    
    Uses fast C-level string translation table.
    """
    if not text:
        return []
    cleaned = text.lower().translate(_DELIM_TRANS)
    return [t for t in cleaned.split() if len(t) >= min_len]


def is_split_gguf_shard(filename: str) -> bool:
    """Checks whether a GGUF filename corresponds to a split shard (e.g. model-00001-of-00005.gguf).
    
    Zero regular expressions.
    """
    if not (filename.endswith(".gguf") and "-of-" in filename):
        return False
    stem = filename[:-5]
    parts = stem.rsplit("-", 2)
    if len(parts) >= 3 and parts[-2] == "of":
        return parts[-3].isdigit() and parts[-1].isdigit()
    return False


def extract_json_from_llm(raw: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts a JSON object from LLM output.
    
    Handles: markdown fences, leading prose, multiple JSON blocks.
    Used by: planner.py, synthesis.py, generate_trees.py, llm_harvester.py
    """
    if not raw:
        return None
    text = raw.strip()
    
    # Strip markdown code fences
    if text.startswith("```"):
        first_line_end = text.find("\n")
        if first_line_end != -1:
            text = text[first_line_end + 1 :]
        else:
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    
    # Attempt 1: Direct parse (best case — clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Balanced-brace state machine respecting string quotes and escapes
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
                    candidate = text[brace_pos:idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        break
        start = brace_pos + 1
    
    logger.warning(f"[UTILS] Failed to extract JSON from LLM output ({len(text)} chars)")
    return None


# Alias for backward compatibility across modules
extract_json_object = extract_json_from_llm


def validate_code_template(template: str) -> bool:
    """Validates a code template by substituting placeholders with dummy identifiers.
    
    Used by: cli.py, schema.py, synthesis.py
    Returns True if the template parses as valid Python after placeholder substitution.
    """
    if not template or not template.strip():
        return False
    
    placeholders = set(extract_template_placeholders(template))
    test_code = template
    for ph in placeholders:
        test_code = test_code.replace(f"{{{ph}}}", f"_ph_{ph}")
    try:
        ast.parse(test_code)
        return True
    except SyntaxError:
        return False


def extract_code_from_llm_response(text: str) -> str:
    """Extracts executable Python code from an LLM response.

    Handles:
    - ```python\\n...\\n```
    - ```py\\n...\\n```
    - ```\\n...\\n``` (no language tag)
    - Plain code with no fences at all (return trimmed/unchanged)
    - Fenced code with leading/trailing prose outside the fence (extracts just the fenced block)
    """
    if not text:
        return ""

    start_idx = text.find("```")
    if start_idx != -1:
        newline_idx = text.find("\n", start_idx + 3)
        if newline_idx != -1:
            code_start = newline_idx + 1
        else:
            code_start = start_idx + 3
        end_idx = text.find("```", code_start)
        if end_idx != -1:
            return text[code_start:end_idx].strip()
        else:
            return text[code_start:].strip()

    return text.strip()


def ensure_comprehensive_prompt(prompt: str, candidate_prompt: Optional[str]) -> str:
    """
    Validates a preprocessed, translated, or compiler-generated prompt against the
    authoritative user prompt. If the candidate prompt dropped clauses, truncated
    compound workflows, or suffered from significant semantic loss (e.g. LLM
    prematurely stopping after the first step), this preserves the original prompt
    so downstream routers and planners receive the complete pipeline specification.
    """
    if not prompt or not prompt.strip():
        return candidate_prompt or ""
    if not candidate_prompt or not candidate_prompt.strip():
        return prompt

    p_clean = prompt.strip()
    c_clean = candidate_prompt.strip().strip("`").strip()
    if p_clean == c_clean:
        return c_clean

    try:
        from tokenizer import CellTokenizer
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        stop = set(reg.get_function_words()) | set(reg.get_sentence_connectives())

        p_toks = CellTokenizer.tokenize_prompt(p_clean)
        c_toks = CellTokenizer.tokenize_prompt(c_clean)

        p_content = {t for t in p_toks if t not in stop and reg.is_informative_token(t)}
        c_content = {t for t in c_toks if t not in stop and reg.is_informative_token(t)}

        # If the original prompt is rich, ensure candidate does not discard majority of intent
        if len(p_content) >= 3:
            recall = len(p_content & c_content) / len(p_content)
            if recall < 0.85:
                logger.info(f"[PromptProtection] Candidate prompt dropped key intent (recall {recall:.2f} < 0.85); reverting to original.")
                return p_clean

        # Precision protection against additive hallucinations
        if len(c_content) >= 3 and len(p_content) >= 3:
            precision = len(p_content & c_content) / len(c_content)
            if precision < 0.60:
                logger.info(f"[PromptProtection] Candidate prompt introduced extraneous hallucinations (precision {precision:.2f} < 0.60); reverting to original.")
                return p_clean

        # Clause count protection for compound queries
        p_clauses = CellTokenizer.split_prompt_clauses(p_clean)
        c_clauses = CellTokenizer.split_prompt_clauses(c_clean)
        if len(p_clauses) > 1 and len(c_clauses) < max(2, int(len(p_clauses) * 0.6)):
            logger.info(f"[PromptProtection] Candidate prompt truncated clauses ({len(c_clauses)} vs {len(p_clauses)}); reverting to original.")
            return p_clean

        # Word length sanity check
        p_words = p_clean.split()
        c_words = c_clean.split()
        if len(p_words) >= 8 and len(c_words) < len(p_words) * 0.40:
            logger.info(f"[PromptProtection] Candidate prompt is excessively abbreviated ({len(c_words)} vs {len(p_words)} words); reverting to original.")
            return p_clean

    except Exception as e:
        logger.debug(f"[PromptProtection] Evaluation fallback: {e}")

    return c_clean

