"""
src/language_backend.py
Target-language abstraction for NSTL / STRATA.

All syntax validation, statement/assignment detection, literal parsing,
projection parsing, call parsing, and template selection route through an
active LanguageBackend configured by the loaded core tree's `LanguageSpec`.
Zero regular expressions. Zero domain or target-language assumptions in callers.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set, Tuple


# Canonical universal lattice symbols (domain- and language-agnostic sentinels).
UNIVERSAL_TOP_SYMBOLS: Tuple[str, ...] = ("any", "*", "top", "unknown", "generic", "\u22a4")
DEFAULT_STATE_SYMBOLS: Tuple[str, ...] = ("", "any", "*", "default", "top", "unbound", "\u22a4")
UNIVERSAL_BOTTOM_SYMBOLS: Tuple[str, ...] = ("void", "unit", "never", "bottom", "\u22a5")


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
    """Structurally verifies balanced identifier placeholders without language assumptions."""
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
            # Allow dict/block literals like `{"k": v}` or `{ a; b }` as well as `{ident}`
            i = end + 1
            continue
        i += 1
    return True


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Splits `text` by `sep` at bracket depth 0 outside quotes (no regex)."""
    parts: List[str] = []
    cur: List[str] = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
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
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]" and depth_brack > 0:
            depth_brack -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}" and depth_brace > 0:
            depth_brace -= 1
        elif (
            depth_paren == 0
            and depth_brack == 0
            and depth_brace == 0
            and text.startswith(sep, i)
        ):
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
    Handles Rust, C++, Go, Julia, SQL-like dialects, or unknown core languages
    without importing or invoking Python's `ast` module.
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
        if self.spec is not None:
            return getattr(self.spec, "runtime", None)
        return None

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
    def variadic_markers(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "variadic_markers", None):
            return tuple(str(x).strip().lower() for x in self.spec.variadic_markers if x)
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
    def empty_collection_literals(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "empty_collection_literals", None):
            return tuple(str(x) for x in self.spec.empty_collection_literals if x)
        return ("[]", "()")

    @property
    def method_receiver_names(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "method_receiver_names", None):
            return tuple(str(x) for x in self.spec.method_receiver_names if x)
        return ()

    @property
    def debug_builtins(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "debug_builtins", None):
            return tuple(str(x) for x in self.spec.debug_builtins if x)
        return ()

    @property
    def union_type_constructors(self) -> Tuple[str, ...]:
        if self.spec is not None and getattr(self.spec, "union_type_constructors", None):
            return tuple(str(x) for x in self.spec.union_type_constructors if x)
        return ("Union",)

    def format_bool(self, val: bool) -> str:
        t_lit, f_lit = self.bool_literals
        return t_lit if val else f_lit

    def is_null_literal(self, s: Any, case_insensitive: bool = True) -> bool:
        txt = str(s or "").strip()
        if not txt:
            return False
        nl = self.null_literal
        if txt == nl:
            return True
        if case_insensitive and (txt.lower() == nl.lower() or txt.lower() in UNIVERSAL_BOTTOM_SYMBOLS):
            return True
        return False

    def is_bool_literal(self, s: Any, case_insensitive: bool = False) -> bool:
        txt = str(s or "").strip()
        if not txt:
            return False
        b_lits = self.bool_literals
        if txt in b_lits:
            return True
        if case_insensitive and txt.lower() in tuple(b.lower() for b in b_lits):
            return True
        return False

    def is_null_or_bool_literal(self, s: Any, case_insensitive: bool = False) -> bool:
        return self.is_null_literal(s, case_insensitive=case_insensitive) or self.is_bool_literal(s, case_insensitive=case_insensitive)

    def is_empty_collection_literal(self, s: Any) -> bool:
        txt = str(s or "").strip()
        return txt in self.empty_collection_literals

    def extract_identifier_refs(self, expr: str) -> Optional[Set[str]]:
        """Extracts identifier references from an expression string; returns None on invalid syntax."""
        s = (expr or "").strip()
        if not s:
            return None
        refs: Set[str] = set()
        i = 0
        n = len(s)
        in_q: Optional[str] = None
        while i < n:
            ch = s[i]
            if in_q:
                if ch == in_q and (i == 0 or s[i - 1] != "\\"):
                    in_q = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_q = ch
                i += 1
                continue
            if ch.isalpha() or ch == "_":
                j = i + 1
                while j < n and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                tok = s[i:j]
                if not self.is_null_or_bool_literal(tok):
                    refs.add(tok)
                i = j
                continue
            i += 1
        return refs

    @property
    def noop_statement(self) -> str:
        if self.spec is not None and getattr(self.spec, "noop_statement", None):
            return str(self.spec.noop_statement)
        return ""

    @property
    def return_keyword(self) -> str:
        if self.spec is not None and getattr(self.spec, "return_keyword", None):
            return str(self.spec.return_keyword)
        return "return"

    @property
    def loop_keyword(self) -> str:
        if self.spec is not None and getattr(self.spec, "loop_keyword", None):
            return str(self.spec.loop_keyword)
        return ""

    @property
    def loop_in_keyword(self) -> str:
        if self.spec is not None and getattr(self.spec, "loop_in_keyword", None):
            return str(self.spec.loop_in_keyword)
        return ""

    @property
    def block_suffix(self) -> str:
        if self.spec is not None and getattr(self.spec, "block_suffix", None) is not None:
            return str(self.spec.block_suffix)
        return ""

    @property
    def container_constructors(self) -> Dict[str, str]:
        if self.spec is not None and isinstance(getattr(self.spec, "container_constructors", None), dict):
            return dict(self.spec.container_constructors)
        return {}

    @property
    def collection_append_templates(self) -> Dict[str, str]:
        if self.spec is not None and isinstance(getattr(self.spec, "collection_append_templates", None), dict):
            return dict(self.spec.collection_append_templates)
        return {}

    def format_collection_append(self, container_kind: str, container_var: str, item_expr: str, key_expr: str = "") -> Optional[str]:
        tmpl = self.collection_append_templates.get(container_kind)
        if not tmpl:
            return None
        return tmpl.format(container=container_var, item=item_expr, key=key_expr)

    def _extract_dep_roots(self, dep_s: str, include_aliased_roots: bool = False) -> Set[str]:
        s = str(dep_s or "").strip()
        if not s:
            return set()
        term = self.statement_terminator
        if term and s.endswith(term):
            s = s[: -len(term)].strip()
        changed = True
        prefixes = self.dependency_prefixes + self.declaration_prefixes
        while changed and prefixes:
            changed = False
            for pref in prefixes:
                if s.startswith(pref):
                    s = s[len(pref) :].lstrip()
                    changed = True
                    break
        out: Set[str] = set()
        for part in s.split(","):
            seg = part.strip()
            if not seg:
                continue
            if " as " in seg:
                lhs, _, rhs = seg.partition(" as ")
                alias = rhs.strip()
                if alias.isidentifier():
                    out.add(alias)
                    if not include_aliased_roots:
                        continue
                seg = lhs.strip()
            root = seg.split("::")[0].split(".")[0].split("/")[0].strip("<>\"'()")
            if root.isidentifier():
                out.add(root)
        return out

    def extract_dependency_namespaces(self, dep_s: str) -> Set[str]:
        """Extracts bound namespace / alias identifiers from a declared dependency using LanguageSpec."""
        return self._extract_dep_roots(dep_s, include_aliased_roots=False)

    def extract_dependency_identifiers(self, dep_s: str) -> Set[str]:
        """Extracts both root module names and alias identifiers from a declared dependency."""
        return self._extract_dep_roots(dep_s, include_aliased_roots=True)

    def extract_dependency_target_modules(self, dep_s: str, target_alias: str) -> List[str]:
        """Resolves candidate module paths bound to `target_alias` by a dependency declaration."""
        if target_alias in self.extract_dependency_namespaces(dep_s):
            s = str(dep_s or "").strip()
            term = self.statement_terminator
            if term and s.endswith(term):
                s = s[: -len(term)].strip()
            for pref in self.dependency_prefixes + self.declaration_prefixes:
                if s.startswith(pref):
                    s = s[len(pref) :].lstrip()
            for part in s.split(","):
                seg = part.strip()
                if " as " in seg:
                    mod_p, _, al_p = seg.partition(" as ")
                    if al_p.strip() == target_alias:
                        return [mod_p.strip()]
                elif seg.split(".")[0].split("::")[0] == target_alias:
                    return [seg]
        return []

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

    def extract_loop_item_var(self, template: str) -> Optional[str]:
        kw = self.loop_keyword.strip()
        in_kw = self.loop_in_keyword.strip()
        if not template or not kw or not in_kw:
            return None
        needle = f" {in_kw} "
        kw_pfx = f"{kw} "
        for ln in template.splitlines():
            st = ln.strip()
            if st.startswith(kw_pfx) and needle in st:
                cand = st[len(kw_pfx) : st.index(needle)].strip().strip("()")
                if cand.isidentifier():
                    return cand
        return None

    def is_variadic_marker(self, marker: Any) -> bool:
        if not marker:
            return False
        return str(marker).strip().lower() in self.variadic_markers

    def fn_declaration_position(self, cur_lstrip: str) -> Tuple[bool, bool, bool]:
        """Returns `(is_fn_name, is_fn_params, is_catch_type)` from `LanguageSpec` keywords."""
        fn_kw = str(getattr(self.spec, "function_keyword", "") or "") if self.spec else ""
        catch_kw = str(getattr(self.spec, "catch_keyword", "") or "") if self.spec else ""
        is_fn_name = bool(fn_kw and cur_lstrip == fn_kw)
        is_fn_params = bool(fn_kw and cur_lstrip.startswith(fn_kw) and "(" in cur_lstrip and ")" not in cur_lstrip)
        is_catch_type = bool(catch_kw and cur_lstrip == catch_kw)
        return is_fn_name, is_fn_params, is_catch_type

    def is_dependency_statement(self, line: str) -> bool:
        s = line.lstrip()
        if not s:
            return False
        prefixes = self.dependency_prefixes
        if not prefixes:
            return False
        return any(s.startswith(p) for p in prefixes)

    def format_comment(self, text: str) -> str:
        prefix = self.line_comment.rstrip()
        return f"{prefix} {text}"

    def select_cell_template(self, cell: Any) -> str:
        templates = getattr(cell, "code_templates", None) or {}
        if isinstance(templates, dict) and self.name and self.name in templates and templates[self.name]:
            return str(templates[self.name])
        ct = getattr(cell, "code_template", None)
        if ct:
            return str(ct)
        if isinstance(templates, dict) and templates:
            return str(next(iter(templates.values()), ""))
        return ""

    def validate_syntax(self, code: str, allow_placeholders: bool = True) -> Tuple[bool, Optional[str]]:
        """Checks balanced delimiters and string quotes across arbitrary languages."""
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
                if nl == -1:
                    break
                i = nl + 1
                continue
            if ch in ('"', "'"):
                # Handle Rust lifetime annotations like `'a` or `&'static str`:
                # if single quote is immediately followed by identifier char and NOT closed by `'` at i+2
                if ch == "'" and (i + 1 < n and (text[i + 1].isalpha() or text[i + 1] == "_")):
                    line_end = text.find("\n", i + 1)
                    if line_end == -1:
                        line_end = n
                    if "'" not in text[i + 1 : line_end]:
                        i += 1
                        continue
                in_quote = ch
                i += 1
                continue
            if ch in pairs:
                stack.append(ch)
            elif ch in closers:
                if not stack or stack[-1] != closers[ch]:
                    return False, f"unmatched delimiter '{ch}'"
                stack.pop()
            i += 1
        if in_quote is not None:
            return False, f"unclosed string literal {in_quote}"
        if stack:
            return False, f"unclosed delimiter '{stack[-1]}'"
        return True, None

    def _strip_var_decl_prefix(self, lhs: str) -> str:
        s = lhs.strip()
        changed = True
        while changed and self.declaration_prefixes:
            changed = False
            for pref in self.declaration_prefixes:
                if s.startswith(pref):
                    s = s[len(pref) :].lstrip()
                    changed = True
                    break
        # Strip type annotation suffix on LHS (`x: i32` or `x: DataFrame`) outside brackets
        depth = 0
        in_quote: Optional[str] = None
        for idx, ch in enumerate(s):
            if in_quote:
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            elif ch == ":" and depth == 0:
                # Avoid splitting `::` scope operator
                if idx + 1 < len(s) and s[idx + 1] == ":":
                    continue
                s = s[:idx].strip()
                break
        return s

    def extract_assignment_targets(self, code: str) -> List[str]:
        """Extracts LHS target variable names per top-level statement without AST."""
        targets: List[str] = []
        op = self.assignment_operator
        comment = self.line_comment
        for raw_line in (code or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if comment and line.startswith(comment):
                continue
            if self.is_dependency_statement(line):
                continue
            term = self.statement_terminator
            if term and line.endswith(term):
                line = line[: -len(term)].rstrip()
            # Scan for top-level assignment operator (not ==, !=, <=, >=, =>)
            depth = 0
            in_quote: Optional[str] = None
            split_idx = -1
            i = 0
            n = len(line)
            while i < n:
                ch = line[i]
                if in_quote:
                    if ch == in_quote and (i == 0 or line[i - 1] != "\\"):
                        in_quote = None
                    i += 1
                    continue
                if ch in ('"', "'"):
                    in_quote = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}" and depth > 0:
                    depth -= 1
                elif depth == 0 and line.startswith(op, i):
                    prev_c = line[i - 1] if i > 0 else ""
                    next_c = line[i + len(op)] if i + len(op) < n else ""
                    if prev_c not in ("=", "!", "<", ">") and next_c not in ("=", ">"):
                        split_idx = i
                        break
                i += 1
            if split_idx <= 0:
                continue
            lhs = self._strip_var_decl_prefix(line[:split_idx])
            if not lhs:
                continue
            # Support tuple destructuring `(a, b)` or single `ident` or `a.attr` / `a["col"]`
            if lhs.startswith("(") and lhs.endswith(")"):
                lhs = lhs[1:-1]
            for part in _split_top_level(lhs, ","):
                p = self._strip_var_decl_prefix(part)
                # Take root identifier if indexing/attribute write, or bare identifier
                ident = []
                for ch in p:
                    if ch.isalnum() or ch == "_":
                        ident.append(ch)
                    else:
                        break
                root = "".join(ident)
                if root and root.isidentifier() and root not in targets:
                    targets.append(root)
        return targets

    def parse_literal(self, expr: str) -> Tuple[bool, Any]:
        """Parses common literal values across languages safely without eval."""
        s = (expr or "").strip()
        if not s:
            return False, None
        low = s.lower()
        if low in ("true", "false"):
            return True, low == "true"
        if low in ("none", "null", "nil"):
            return True, None
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            return True, s[1:-1]
        # Numeric check (strip trailing numeric type suffixes like 10usize or 1.0f64 only if needed)
        try:
            if s.startswith(("0x", "0b", "0o", "-0x")):
                return True, int(s, 0)
            return True, int(s)
        except ValueError:
            pass
        try:
            return True, float(s)
        except ValueError:
            pass
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            inner = s[1:-1].strip()
            if not inner:
                return True, [] if s.startswith("[") else ()
            elems: List[Any] = []
            for part in _split_top_level(inner, ","):
                if not part:
                    continue
                ok, val = self.parse_literal(part)
                if not ok:
                    return False, None
                elems.append(val)
            return True, elems if s.startswith("[") else tuple(elems)
        return False, None

    def parse_comparison(self, expr: str) -> Optional[Dict[str, Any]]:
        """Parses `<prop> <op> <val>` into structured predicate fields without Python AST."""
        s = (expr or "").strip()
        if not s:
            return None
        ops = ("==", "!=", ">=", "<=", " not in ", " is not ", " in ", " is ", ">", "<")
        depth = 0
        in_quote: Optional[str] = None
        i = 0
        n = len(s)
        found: Optional[Tuple[int, int, str]] = None
        while i < n:
            ch = s[i]
            if in_quote:
                if ch == in_quote and (i == 0 or s[i - 1] != "\\"):
                    in_quote = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                i += 1
                continue
            if ch in "([{":
                depth += 1
                i += 1
                continue
            if ch in ")]}" and depth > 0:
                depth -= 1
                i += 1
                continue
            if depth == 0:
                matched = False
                for op in ops:
                    if s.startswith(op, i):
                        # Disallow multiple top-level comparisons (`a < x < b`)
                        if found is not None:
                            return None
                        found = (i, i + len(op), op.strip())
                        i += len(op)
                        matched = True
                        break
                if matched:
                    continue
            i += 1
        if found is None:
            return None
        left = s[: found[0]].strip()
        right = s[found[1] :].strip()
        if not left or not right:
            return None
        # Validate left looks like identifier / attribute / subscript chain
        root_part = left.split("[", 1)[0]
        if not all(seg.isidentifier() for seg in root_part.split(".") if seg):
            return None
        ok, val = self.parse_literal(right)
        return {
            "property": left,
            "operator": found[2],
            "value": val if ok else right,
            "expression": s,
        }

    def parse_call(self, expr: str) -> Optional[Tuple[str, List[str], Dict[str, str]]]:
        """Parses `<callee>(<args...>)` at depth 0 into `(callee, pos_args, kw_args)`."""
        s = (expr or "").strip()
        term = self.statement_terminator
        if term and s.endswith(term):
            s = s[: -len(term)].rstrip()
        if not s.endswith(")") or "(" not in s:
            return None
        open_idx = s.find("(")
        callee = s[:open_idx].strip()
        if not callee:
            return None
        inner = s[open_idx + 1 : -1].strip()
        pos_args: List[str] = []
        kw_args: Dict[str, str] = {}
        if not inner:
            return callee, pos_args, kw_args
        for part in _split_top_level(inner, ","):
            if not part:
                continue
            # Check top-level `=` (not `==`)
            eq_idx = -1
            depth = 0
            in_q: Optional[str] = None
            for idx, ch in enumerate(part):
                if in_q:
                    if ch == in_q and (idx == 0 or part[idx - 1] != "\\"):
                        in_q = None
                    continue
                if ch in ('"', "'"):
                    in_q = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}" and depth > 0:
                    depth -= 1
                elif ch == "=" and depth == 0:
                    prev_c = part[idx - 1] if idx > 0 else ""
                    next_c = part[idx + 1] if idx + 1 < len(part) else ""
                    if prev_c not in ("=", "!", "<", ">") and next_c != "=":
                        eq_idx = idx
                        break
            if eq_idx > 0:
                k = part[:eq_idx].strip()
                v = part[eq_idx + 1 :].strip()
                if k.isidentifier():
                    kw_args[k] = v
                    continue
            pos_args.append(part)
        return callee, pos_args, kw_args

    def parse_projection(self, expr: str) -> Optional[Dict[str, Any]]:
        """Parses subscript projection `<base>[<slice>]` cleanly without regex."""
        s = (expr or "").strip()
        sub_open = getattr(self.spec, "subscript_open", "[") if self.spec else "["
        sub_close = getattr(self.spec, "subscript_close", "]") if self.spec else "]"
        if not s.endswith(sub_close) or sub_open not in s:
            return None
        # Find matching opening bracket for trailing `sub_close`
        depth = 0
        in_q: Optional[str] = None
        match_idx = -1
        i = len(s) - len(sub_close)
        while i >= 0:
            ch = s[i]
            if in_q:
                if ch == in_q and (i == 0 or s[i - 1] != "\\"):
                    in_q = None
            else:
                if ch in ('"', "'"):
                    in_q = ch
                elif s.startswith(sub_close, i):
                    depth += 1
                elif s.startswith(sub_open, i):
                    depth -= 1
                    if depth == 0:
                        match_idx = i
                        break
            i -= 1
        if match_idx <= 0:
            return None
        base = s[:match_idx].strip()
        slice_str = s[match_idx + len(sub_open) : -len(sub_close)].strip()
        if not slice_str:
            return None
        ok, val = self.parse_literal(slice_str)
        if ok:
            if isinstance(val, (list, tuple)):
                return {
                    "base": base,
                    "kind": "multi",
                    "keys": [str(x) for x in val],
                }
            if isinstance(val, str):
                return {"base": base, "kind": "single", "key": val}
        return {"base": base, "kind": "single", "key": slice_str}

    def extract_call_names(self, code: str) -> List[str]:
        """Lexically extracts callable names immediately preceding `(` without regex."""
        if not code:
            return []
        masked, _ = _mask_placeholders(code)
        calls: List[str] = []
        i = 0
        n = len(masked)
        in_q: Optional[str] = None
        while i < n:
            ch = masked[i]
            if in_q:
                if ch == in_q and (i == 0 or masked[i - 1] != "\\"):
                    in_q = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_q = ch
                i += 1
                continue
            if ch == "(":
                # Walk backwards past whitespace to find identifier token
                j = i - 1
                while j >= 0 and masked[j].isspace():
                    j -= 1
                end = j + 1
                while j >= 0 and (masked[j].isalnum() or masked[j] == "_"):
                    j -= 1
                tok = masked[j + 1 : end]
                if tok and tok.isidentifier() and not tok.startswith("_ph_"):
                    calls.append(tok.lower())
            i += 1
        return calls


class PythonASTBackend(LexicalLanguageBackend):
    """
    Exact Python AST backend enabled only when the active core tree's
    `language_spec.ast_backend == "python_ast"`.
    """

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

    def parse_comparison(self, expr: str) -> Optional[Dict[str, Any]]:
        s = (expr or "").strip()
        if not s:
            return None
        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError:
            return None
        body = tree.body
        if not (isinstance(body, ast.Compare) and len(body.ops) == 1 and len(body.comparators) == 1):
            return None
        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.In: "in",
            ast.NotIn: "not in",
            ast.Is: "is",
            ast.IsNot: "is not",
        }
        op = op_map.get(type(body.ops[0]))
        if not op:
            return None

        def _is_valid_prop_ast(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id.isidentifier()
            if isinstance(node, ast.Attribute):
                return node.attr.isidentifier() and _is_valid_prop_ast(node.value)
            if isinstance(node, ast.Subscript):
                return _is_valid_prop_ast(node.value)
            return False

        left = body.left
        if not _is_valid_prop_ast(left):
            return None
        try:
            prop = ast.unparse(left)
        except Exception:
            return None
        right = body.comparators[0]
        try:
            val = ast.literal_eval(right)
        except Exception:
            try:
                val = ast.unparse(right)
            except Exception:
                val = None
        return {"property": prop, "operator": op, "value": val, "expression": s}

    def extract_assignment_targets(self, code: str) -> List[str]:
        targets: List[str] = []
        try:
            tree = ast.parse(code or "")
        except SyntaxError:
            return super().extract_assignment_targets(code)

        def _collect(node: ast.AST) -> None:
            if isinstance(node, ast.Name) and node.id not in targets:
                targets.append(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    _collect(elt)
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id not in targets:
                    targets.append(node.value.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id not in targets:
                    targets.append(node.value.id)

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    _collect(tgt)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                _collect(stmt.target)
            elif isinstance(stmt, ast.AugAssign):
                _collect(stmt.target)
        return targets

    def parse_call(self, expr: str) -> Optional[Tuple[str, List[str], Dict[str, str]]]:
        s = (expr or "").strip()
        if not s:
            return None
        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError:
            return super().parse_call(s)
        call = tree.body
        if not isinstance(call, ast.Call):
            return None
        callee = ast.unparse(call.func)
        pos_args = [ast.unparse(a) for a in call.args]
        kw_args = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}
        return callee, pos_args, kw_args

    def parse_projection(self, expr: str) -> Optional[Dict[str, Any]]:
        s = (expr or "").strip()
        if not s:
            return None
        try:
            node = ast.parse(s, mode="eval").body
        except SyntaxError:
            return super().parse_projection(s)
        if not isinstance(node, ast.Subscript):
            return None
        base = ast.unparse(node.value)
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return {"base": base, "kind": "single", "key": sl.value}
        if isinstance(sl, (ast.List, ast.Tuple)):
            keys: List[str] = []
            for elt in sl.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    keys.append(elt.value)
                else:
                    keys.append(ast.unparse(elt))
            return {"base": base, "kind": "multi", "keys": keys}
        return {"base": base, "kind": "single", "key": ast.unparse(sl)}

    def extract_call_names(self, code: str) -> List[str]:
        if not code or not code.strip():
            return []
        sanitized, _ = _mask_placeholders(code)
        try:
            tree = ast.parse(sanitized)
        except SyntaxError:
            return super().extract_call_names(code)
        calls: List[str] = []
        top_calls: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                cname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
                if cname and not cname.startswith("_ph_"):
                    calls.append(cname.lower())
        if len(calls) > 1:
            for stmt in tree.body:
                val = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
                if isinstance(val, ast.Call):
                    fn = val.func
                    cname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
                    if cname and not cname.startswith("_ph_"):
                        top_calls.append(cname.lower())
        return top_calls or calls

    def extract_dependency_namespaces(self, dep_s: str) -> Set[str]:
        s = str(dep_s or "").strip()
        if not s:
            return set()
        out: Set[str] = set()
        try:
            tree = ast.parse(s)
            if tree.body:
                matched = False
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        matched = True
                        for al in node.names:
                            name = al.asname or al.name.split(".")[0]
                            if name and name.isidentifier():
                                out.add(name)
                    elif isinstance(node, ast.ImportFrom):
                        matched = True
                        if node.module:
                            root = node.module.split(".")[0]
                            if root and root.isidentifier():
                                out.add(root)
                        for al in node.names:
                            name = al.asname or al.name
                            if name and name != "*" and name.isidentifier():
                                out.add(name)
                if matched:
                    return out
        except SyntaxError:
            pass
        return super().extract_dependency_namespaces(s)

    def extract_dependency_target_modules(self, dep_s: str, target_alias: str) -> List[str]:
        s = str(dep_s or "").strip()
        if not s:
            return []
        out: List[str] = []
        try:
            tree = ast.parse(s)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for al in node.names:
                        alias = al.asname or al.name.split(".")[0]
                        if alias == target_alias:
                            out.append(al.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for al in node.names:
                        alias = al.asname or al.name
                        if alias == target_alias:
                            out.append(f"{node.module}.{al.name}")
            if out:
                return out
        except SyntaxError:
            pass
        return super().extract_dependency_target_modules(s, target_alias)

    def extract_loop_item_var(self, template: str) -> Optional[str]:
        if not template or not template.strip():
            return None
        sanitized, mask_map = _mask_placeholders(template)
        rev_mask = {v: k[1:-1] for k, v in mask_map.items()}
        try:
            tree = ast.parse(sanitized)
        except SyntaxError:
            return super().extract_loop_item_var(template)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                target = node.target
                t_name = None
                if isinstance(target, ast.Name):
                    t_name = target.id
                elif isinstance(target, (ast.Tuple, ast.List)) and target.elts:
                    if isinstance(target.elts[0], ast.Name):
                        t_name = target.elts[0].id
                if t_name:
                    return rev_mask.get(t_name, t_name)
        return None

    def extract_identifier_refs(self, expr: str) -> Optional[Set[str]]:
        import keyword as _kw
        s = (expr or "").strip()
        if not s:
            return None
        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError:
            return None
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and not _kw.iskeyword(node.id)
        }


_ACTIVE_LANGUAGE_SPEC: Optional[Any] = None
_ACTIVE_BACKEND: LexicalLanguageBackend = LexicalLanguageBackend(None)


def get_language_backend(spec: Optional[Any] = None) -> LexicalLanguageBackend:
    """Instantiates the appropriate LanguageBackend from a LanguageSpec."""
    if spec is None:
        return LexicalLanguageBackend(None)
    raw_kind = spec.get("ast_backend", "lexical") if isinstance(spec, dict) else getattr(spec, "ast_backend", "lexical")
    ast_kind = str(raw_kind or "lexical").strip().lower()
    if ast_kind == "python_ast":
        return PythonASTBackend(spec)
    return LexicalLanguageBackend(spec)


def set_active_language_spec(spec: Optional[Any]) -> LexicalLanguageBackend:
    """Sets the active LanguageSpec registered by the loaded core tree."""
    global _ACTIVE_LANGUAGE_SPEC, _ACTIVE_BACKEND
    _ACTIVE_LANGUAGE_SPEC = spec
    _ACTIVE_BACKEND = get_language_backend(spec)
    return _ACTIVE_BACKEND


def get_active_language_backend(registry: Optional[Any] = None) -> LexicalLanguageBackend:
    """Returns the LanguageBackend for `registry` (if given and configured) or the active core tree."""
    if registry is not None:
        spec_fn = getattr(registry, "get_language_spec", None)
        spec = spec_fn() if callable(spec_fn) else None
        if spec is not None:
            return get_language_backend(spec)
    return _ACTIVE_BACKEND
