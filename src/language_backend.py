"""
src/language_backend.py
Target-language abstraction for STRATA.

Routes template selection, LHS placeholder detection, literal parsing/formatting,
dependency formatting, control-flow block formatting, and syntax validation through
the active LanguageBackend configured by the loaded core domain's `LanguageSpec`.
Zero regular expressions. Zero domain assumptions.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from .schema import TOP_STATES, UNIVERSAL_BOTTOM_SYMBOLS
except (ImportError, ValueError):
    from schema import TOP_STATES, UNIVERSAL_BOTTOM_SYMBOLS


def _mask_placeholders(template: str) -> Tuple[str, Dict[str, str]]:
    """Replace `{ident}` placeholders with valid `_ph_<i>` identifiers without regex."""
    seen: Dict[str, str] = {}
    res: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{" and i + 1 < n:
            end = template.find("}", i + 1)
            if end != -1:
                inner = template[i + 1 : end]
                if inner.isidentifier():
                    key = f"{{{inner}}}"
                    if key not in seen:
                        seen[key] = f"_ph_{len(seen)}"
                    res.append(seen[key])
                    i = end + 1
                    continue
        res.append(template[i])
        i += 1
    return "".join(res), seen


def validate_placeholder_braces(template: str) -> bool:
    """Structurally verify balanced identifier placeholders without language assumptions."""
    if not template:
        return True
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                i += 2
                continue
            end = template.find("}", i + 1)
            if end == -1:
                return False
            i = end + 1
            continue
        i += 1
    return True


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split `text` by `sep` at bracket depth 0 outside quotes."""
    parts: List[str] = []
    cur: List[str] = []
    depth_p = 0
    depth_b = 0
    depth_c = 0
    in_quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(text)
    sep_len = len(sep)
    while i < n:
        ch = text[i]
        if in_quote is not None:
            cur.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_p += 1
        elif ch == ")" and depth_p > 0:
            depth_p -= 1
        elif ch == "[":
            depth_b += 1
        elif ch == "]" and depth_b > 0:
            depth_b -= 1
        elif ch == "{":
            depth_c += 1
        elif ch == "}" and depth_c > 0:
            depth_c -= 1
        elif depth_p == 0 and depth_b == 0 and depth_c == 0 and text.startswith(sep, i):
            parts.append("".join(cur).strip())
            cur = []
            i += sep_len
            continue
        cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail or parts:
        parts.append(tail)
    return parts


class LexicalLanguageBackend:
    """
    Target-language-agnostic lexical backend driven solely by `LanguageSpec`.
    Supports arbitrary compiled or interpreted target languages without assuming Python AST.
    """

    def __init__(self, spec: Optional[Any] = None) -> None:
        if isinstance(spec, dict):
            from types import SimpleNamespace
            spec = SimpleNamespace(**spec)
        self.spec = spec

    @property
    def is_python_ast(self) -> bool:
        return False

    @property
    def name(self) -> str:
        if self.spec is not None and getattr(self.spec, "name", None):
            return str(self.spec.name).strip().lower()
        return ""

    @property
    def runtime(self) -> Optional[str]:
        return getattr(self.spec, "runtime", None) if self.spec is not None else None

    @property
    def line_comment(self) -> str:
        if self.spec is not None and getattr(self.spec, "line_comment", None):
            return str(self.spec.line_comment)
        return "#"

    @property
    def assignment_operator(self) -> str:
        if self.spec is not None and getattr(self.spec, "assignment_operator", None):
            return str(self.spec.assignment_operator)
        return "="

    @property
    def statement_terminator(self) -> str:
        if self.spec is not None and getattr(self.spec, "statement_terminator", None) is not None:
            return str(self.spec.statement_terminator)
        return ""

    @property
    def dependency_prefixes(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "dependency_prefixes", None):
            return tuple(str(x) for x in self.spec.dependency_prefixes if x)
        return ()

    @property
    def declaration_prefixes(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "declaration_prefixes", None):
            return tuple(str(x) for x in self.spec.declaration_prefixes if x)
        return ()

    @property
    def null_literal(self) -> str:
        if self.spec is not None and getattr(self.spec, "null_literal", None):
            return str(self.spec.null_literal)
        return "None"

    @property
    def bool_literals(self) -> Tuple[str, str]:
        if self.spec is not None and getattr(self.spec, "bool_literals", None):
            vals = [str(x) for x in self.spec.bool_literals if x]
            if len(vals) >= 2:
                return (vals[0], vals[1])
        return ("True", "False")

    @property
    def loop_keyword(self) -> str:
        if self.spec is not None and getattr(self.spec, "loop_keyword", None):
            return str(self.spec.loop_keyword).strip()
        return "for"

    @property
    def loop_in_keyword(self) -> str:
        if self.spec is not None and getattr(self.spec, "loop_in_keyword", None):
            return str(self.spec.loop_in_keyword).strip()
        return "in"

    @property
    def block_suffix(self) -> str:
        if self.spec is not None and getattr(self.spec, "block_suffix", None) is not None:
            return str(self.spec.block_suffix)
        return ":"

    @property
    def noop_statement(self) -> str:
        if self.spec is not None and getattr(self.spec, "noop_statement", None):
            return str(self.spec.noop_statement)
        return "pass"

    def is_dependency_statement(self, line: str) -> bool:
        s = line.lstrip()
        if not s or not self.dependency_prefixes:
            return False
        return any(s.startswith(p) for p in self.dependency_prefixes)

    def format_dependency_statement(self, dep_s: str) -> str:
        s = str(dep_s or "").strip()
        if not s:
            return ""
        term = self.statement_terminator
        if not self.is_dependency_statement(s) and self.dependency_prefixes:
            s = f"{self.dependency_prefixes[0]}{s}"
        if term and not s.endswith(term):
            s = f"{s}{term}"
        return s

    def select_cell_template(self, cell: Any) -> str:
        templates = getattr(cell, "code_templates", None) or {}
        if isinstance(templates, dict) and self.name and templates.get(self.name):
            return str(templates[self.name])
        ct = getattr(cell, "code_template", None)
        if ct:
            return str(ct)
        if isinstance(templates, dict) and templates:
            return str(next(iter(templates.values()), ""))
        return ""

    def parse_literal(self, expr: str) -> Tuple[bool, Any]:
        s = str(expr or "").strip()
        if not s:
            return False, None
        t_lit, f_lit = self.bool_literals
        if s == t_lit or s.lower() == "true":
            return True, True
        if s == f_lit or s.lower() == "false":
            return True, False
        if s == self.null_literal or s.lower() in ("none", "null") or s.lower() in UNIVERSAL_BOTTOM_SYMBOLS:
            return True, None
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            try:
                if s[0] == '"':
                    return True, json.loads(s)
            except Exception:
                pass
            return True, s[1:-1]
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            is_tup = s.startswith("(")
            inner = s[1:-1].strip()
            if not inner:
                return True, () if is_tup else []
            items: List[Any] = []
            for seg in _split_top_level(inner, ","):
                if not seg:
                    continue
                ok, item_val = self.parse_literal(seg)
                if not ok:
                    return False, None
                items.append(item_val)
            return True, tuple(items) if is_tup else items
        if s.startswith("{") and s.endswith("}"):
            try:
                return True, json.loads(s)
            except Exception:
                return False, None
        try:
            if s.lstrip("+-").isdigit():
                return True, int(s)
            return True, float(s)
        except ValueError:
            return False, None

    def parse_schema_default(self, raw_val: Any) -> Any:
        if not isinstance(raw_val, str):
            return raw_val
        s = raw_val.strip()
        if not s:
            return raw_val
        ok, parsed = self.parse_literal(s)
        return parsed if ok else raw_val

    def format_literal(self, val: Any) -> str:
        if val is None:
            return self.null_literal
        if isinstance(val, bool):
            t_lit, f_lit = self.bool_literals
            return t_lit if val else f_lit
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, list):
            return "[" + ", ".join(self.format_literal(v) for v in val) + "]"
        if isinstance(val, tuple):
            items = ", ".join(self.format_literal(v) for v in val)
            if len(val) == 1:
                items += ","
            return f"({items})"
        if isinstance(val, dict):
            pairs = ", ".join(f"{self.format_literal(k)}: {self.format_literal(v)}" for k, v in val.items())
            return "{" + pairs + "}"
        return str(val)

    def format_loop_header(self, condition: str) -> str:
        cond = str(condition or "").strip()
        kw = self.loop_keyword or "for"
        sfx = self.block_suffix
        sfx_s = sfx.strip()
        if cond.lower().startswith(f"{kw.lower()} "):
            base = cond
        else:
            base = f"{kw} {cond}"
        return f"{base} {sfx_s}" if sfx_s == "{" else f"{base}{sfx}"

    def format_branch_header(self, condition: str) -> str:
        cond = str(condition or "").strip() or self.bool_literals[0]
        sfx = self.block_suffix
        sfx_s = sfx.strip()
        base = cond if cond.lower().startswith("if ") else f"if {cond}"
        return f"{base} {sfx_s}" if sfx_s == "{" else f"{base}{sfx}"

    def format_else_header(self) -> str:
        sfx = self.block_suffix
        sfx_s = sfx.strip()
        return "} else {" if sfx_s == "{" else f"else{sfx}"

    def format_block_end(self) -> str:
        return "}" if self.block_suffix.strip() == "{" else ""

    def extract_lhs_placeholders(self, tmpl: str, placeholders: List[str]) -> List[str]:
        if not tmpl or not placeholders:
            return []
        op = self.assignment_operator or "="
        ph_set = set(placeholders)
        lhs_phs: List[str] = []
        for raw_line in tmpl.splitlines():
            for stmt in raw_line.split(";"):
                search_start = 0
                pos = -1
                while True:
                    cand = stmt.find(op, search_start)
                    if cand == -1:
                        break
                    if op == "=" and (
                        (cand > 0 and stmt[cand - 1] in ("=", "!", "<", ">"))
                        or stmt[cand : cand + 2] == "=="
                    ):
                        search_start = cand + len(op)
                        continue
                    pos = cand
                    break
                if pos <= 0:
                    continue
                lhs_part = stmt[:pos]
                for ph in placeholders:
                    if ph in ph_set and f"{{{ph}}}" in lhs_part and ph not in lhs_phs:
                        lhs_phs.append(ph)
        if not lhs_phs and "output_var" in ph_set:
            return ["output_var"]
        return lhs_phs

    def validate_syntax(self, code: str, allow_placeholders: bool = True) -> Tuple[bool, Optional[str]]:
        """Check balanced delimiters and string quotes across arbitrary target languages."""
        if not code or not code.strip():
            return False, "empty code"
        text = _mask_placeholders(code)[0] if allow_placeholders else code
        pairs = {"(": ")", "[": "]", "{": "}"}
        closers = {")": "(", "]": "[", "}": "{"}
        stack: List[str] = []
        in_quote: Optional[str] = None
        escaped = False
        comment = self.line_comment
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if in_quote is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if comment and text.startswith(comment, i):
                nl = text.find("\n", i)
                i = n if nl == -1 else nl + 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                i += 1
                continue
            if ch in pairs:
                stack.append(ch)
            elif ch in closers:
                if not stack or stack[-1] != closers[ch]:
                    return False, f"unmatched closing '{ch}'"
                stack.pop()
            i += 1
        if in_quote is not None:
            return False, "unterminated string literal"
        if stack:
            return False, f"unclosed '{stack[-1]}'"
        return True, None


class PythonASTBackend(LexicalLanguageBackend):
    """Exact AST backend when `language_spec.ast_backend == "python_ast"`."""

    @property
    def is_python_ast(self) -> bool:
        return True

    def validate_syntax(self, code: str, allow_placeholders: bool = True) -> Tuple[bool, Optional[str]]:
        if not code or not code.strip():
            return False, "empty code"
        text = _mask_placeholders(code)[0] if allow_placeholders else code
        try:
            ast.parse(text)
            return True, None
        except SyntaxError as exc:
            return False, str(exc)

    def parse_literal(self, expr: str) -> Tuple[bool, Any]:
        s = (expr or "").strip()
        if not s:
            return False, None
        try:
            return True, ast.literal_eval(s)
        except Exception:
            return False, None

    def format_literal(self, val: Any) -> str:
        return repr(val)

    def extract_lhs_placeholders(self, tmpl: str, placeholders: List[str]) -> List[str]:
        if not tmpl or not placeholders:
            return []
        lhs_phs: List[str] = []
        try:
            dummy = {ph: f"__STRATA_PH_{idx}__" for idx, ph in enumerate(placeholders)}
            inv = {v: k for k, v in dummy.items()}
            parsed = ast.parse(tmpl.format(**dummy))
            seen_lhs = set()
            for node in ast.walk(parsed):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in inv:
                    orig_ph = inv[node.id]
                    if orig_ph not in seen_lhs:
                        seen_lhs.add(orig_ph)
                        lhs_phs.append(orig_ph)
            return lhs_phs
        except Exception:
            return super().extract_lhs_placeholders(tmpl, placeholders)


_ACTIVE_LANGUAGE_SPEC: Optional[Any] = None
_ACTIVE_BACKEND: LexicalLanguageBackend = LexicalLanguageBackend(None)


def get_language_backend(spec: Optional[Any] = None) -> LexicalLanguageBackend:
    """Instantiate the appropriate LanguageBackend from a LanguageSpec."""
    if spec is None:
        return LexicalLanguageBackend(None)
    raw_kind = spec.get("ast_backend", "lexical") if isinstance(spec, dict) else getattr(spec, "ast_backend", "lexical")
    ast_kind = str(raw_kind or "lexical").strip().lower()
    if ast_kind == "python_ast":
        return PythonASTBackend(spec)
    return LexicalLanguageBackend(spec)


def set_active_language_spec(spec: Optional[Any]) -> LexicalLanguageBackend:
    """Set the active LanguageSpec registered by the loaded core domain."""
    global _ACTIVE_LANGUAGE_SPEC, _ACTIVE_BACKEND
    _ACTIVE_LANGUAGE_SPEC = spec
    _ACTIVE_BACKEND = get_language_backend(spec)
    return _ACTIVE_BACKEND


def get_active_language_spec() -> Optional[Any]:
    """Return the currently active LanguageSpec registered by the loaded core domain."""
    return _ACTIVE_LANGUAGE_SPEC


def get_active_language_backend() -> LexicalLanguageBackend:
    """Return the active LanguageBackend."""
    return _ACTIVE_BACKEND


__all__ = [
    "TOP_STATES",
    "LexicalLanguageBackend",
    "PythonASTBackend",
    "get_language_backend",
    "set_active_language_spec",
    "get_active_language_spec",
    "get_active_language_backend",
    "validate_placeholder_braces",
]
