"""
src/semantic_repair_engine.py

General dynamic semantic validation and repair engine for STRATA cells.
Contains ZERO hardcoded library checks or cell-name checks.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from typing import Any, Dict, List, Optional, Tuple

try:
    from utils import extract_template_placeholders
except ImportError:
    from .utils import extract_template_placeholders

from signature_introspector import resolve_signature
from template_wiring import (
    clean_malformed_template_braces,
    repair_wiring_invariant,
)


class CallExpressionInfo:
    """
    Holds parsed call expression metadata, preserving surrounding multi-statement code.
    Maintains 4-tuple compatibility for backwards-compatible unpacking.
    """
    def __init__(
        self,
        func_expr: str,
        positional: List[str],
        keywords: Dict[str, str],
        lhs: Optional[str],
        prefix_code: str = "",
        suffix_code: str = "",
    ):
        self.func_expr = func_expr
        self.positional = positional
        self.keywords = keywords
        self.lhs = lhs
        self.prefix_code = prefix_code
        self.suffix_code = suffix_code

    def __iter__(self):
        return iter((self.func_expr, self.positional, self.keywords, self.lhs))

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return (self.func_expr, self.positional, self.keywords, self.lhs)[idx]


def parse_call_expression(
    template: str,
) -> Optional[CallExpressionInfo]:
    """
    Parses the single call expression in a code template.

    Returns:
        CallExpressionInfo (unpacks as (func_expr, positional_arg_names, keyword_arg_names, lhs))

    `lhs` is the original assignment target if the template contains an
    assignment, otherwise None.  This allows callers to preserve multi-target
    assignments and void/sink calls without forcibly injecting `{output_var} =`.
    """
    if not template or "{" not in template:
        return None

    text = template.strip()

    # Substitute placeholders with valid identifiers so AST parsing succeeds.
    placeholders = extract_template_placeholders(text)
    ast_ready = text
    ph_map: Dict[str, str] = {}
    for name in placeholders:
        ast_name = f"_nstl_ph_{name}"
        ast_ready = ast_ready.replace(f"{{{name}}}", ast_name)
        ph_map[ast_name] = name

    def _restore_placeholders(s: str) -> str:
        for ast_name, orig in ph_map.items():
            s = s.replace(ast_name, f"{{{orig}}}")
        return s

    prefix_code = ""
    suffix_code = ""

    # Try expression first (e.g. "func({x})" or "obj.method({x})").
    # If that fails, parse as a statement and extract the statement containing the call.
    try:
        tree = ast.parse(ast_ready, mode="eval")
        rhs_node = tree.body
        lhs_node = None
    except SyntaxError:
        try:
            tree = ast.parse(ast_ready, mode="exec")
        except SyntaxError:
            return None

        rhs_node = None
        lhs_node = None
        target_idx = None
        for idx, node in enumerate(tree.body):
            curr_rhs = None
            curr_lhs = None
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                curr_lhs = node.targets[0] if node.targets else None
                curr_rhs = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
                curr_lhs = node.target
                curr_rhs = node.value
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                curr_rhs = node.value

            if curr_rhs is not None:
                rhs_node = curr_rhs
                lhs_node = curr_lhs
                target_idx = idx
                break

        if rhs_node is None or target_idx is None:
            return None

        if target_idx > 0:
            prefix_stmts = tree.body[:target_idx]
            prefix_code = _restore_placeholders("\n".join(ast.unparse(s) for s in prefix_stmts))

        if target_idx + 1 < len(tree.body):
            suffix_stmts = tree.body[target_idx + 1 :]
            suffix_code = _restore_placeholders("\n".join(ast.unparse(s) for s in suffix_stmts))

    if not isinstance(rhs_node, ast.Call):
        return None

    call = rhs_node

    def _unparse_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _unparse_name(node.value)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Call):
            return _unparse_name(node.func)
        return None

    func_expr = _unparse_name(call.func)
    if not func_expr:
        return None

    # Method calls on data carriers ({df}.head(...), {img}.resize(...)) have a
    # placeholder as their root: there is no static callable to resolve against.
    if func_expr.startswith("_nstl_ph_"):
        return None

    def _ph_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name) and node.id.startswith("_nstl_ph_"):
            return ph_map.get(node.id, node.id[len("_nstl_ph_") :])
        return None

    positional: List[str] = []
    for arg in call.args:
        name = _ph_name(arg)
        if name is not None:
            positional.append(name)
        else:
            try:
                positional.append(ast.unparse(arg))
            except Exception:
                positional.append("")

    keywords: Dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        name = _ph_name(kw.value)
        if name is not None:
            keywords[kw.arg] = name
        else:
            try:
                keywords[kw.arg] = ast.unparse(kw.value) if kw.value is not None else ""
            except Exception:
                keywords[kw.arg] = ""

    lhs: Optional[str] = None
    if lhs_node is not None:
        try:
            lhs = _restore_placeholders(ast.unparse(lhs_node))
        except Exception:
            lhs = None

    if not placeholders and not positional and not keywords:
        return None

    return CallExpressionInfo(func_expr, positional, keywords, lhs, prefix_code, suffix_code)


def resolve_callable_from_expr(func_expr: str, dependencies: Optional[List[str]] = None):
    """
    Resolves a dotted expression (e.g. 'pd.read_csv', 'cv2.threshold') to the
    live callable using ONLY the cell's declared dependencies (alias -> module)
    plus the expression's own root module. Returns None when unresolvable.
    """
    if not func_expr:
        return None

    root, _, rest = func_expr.partition(".")
    candidate_modules: List[str] = []

    for dep in dependencies or []:
        dep_str = str(dep).strip()

        if dep_str.startswith("import "):
            rest_dep = dep_str[len("import ") :].strip()
            for part in rest_dep.split(","):
                part = part.strip()
                if not part:
                    continue
                if " as " in part:
                    mod_name, alias = [x.strip() for x in part.split(" as ", 1)]
                else:
                    mod_name = part
                    alias = part.split(".")[0]
                if alias == root:
                    candidate_modules.append(mod_name)

        elif dep_str.startswith("from "):
            rest_dep = dep_str[len("from ") :].strip()
            if " import " in rest_dep:
                mod_name, imports = rest_dep.split(" import ", 1)
                for part in imports.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if " as " in part:
                        name, alias = [x.strip() for x in part.split(" as ", 1)]
                    else:
                        name = part
                        alias = part
                    if alias == root:
                        candidate_modules.append(f"{mod_name.strip()}.{name}")

    candidate_modules.append(root)

    attr_chain = rest.split(".") if rest else []

    for mod_name in candidate_modules:
        try:
            if "." in mod_name:
                mod_part, _, first_attr = mod_name.partition(".")
                mod = importlib.import_module(mod_part)
                obj = getattr(mod, first_attr, None)
                if obj is None:
                    continue
            else:
                obj = importlib.import_module(mod_name)

            if not rest and root == mod_name.split(".")[0]:
                return obj

            ok = True
            for attr in attr_chain:
                nxt = getattr(obj, attr, None)
                if nxt is None:
                    ok = False
                    break
                obj = nxt

            if ok and callable(obj):
                return obj
        except Exception:
            continue

    # Last resort: the expression's own root module name.
    try:
        obj = importlib.import_module(root)
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj if callable(obj) else None
    except Exception:
        return None


def _format_token(tok: str) -> str:
    if not tok:
        return ""
    if tok.startswith("{") and tok.endswith("}"):
        return tok
    if tok.isidentifier():
        return f"{{{tok}}}"
    return tok


def _format_call(func_expr: str, positional: List[str], keywords: Dict[str, str]) -> str:
    args_src = ", ".join(_format_token(a) for a in positional if a)
    kw_src = ", ".join(f"{k}={_format_token(v)}" for k, v in keywords.items())
    call_src = f"{func_expr}({', '.join(x for x in (args_src, kw_src) if x)})"
    return call_src


def validate_and_reconstruct_call(
    func_obj: Any,
    func_expr: str,
    raw_args: List[str],
    raw_kwargs: Dict[str, str],
) -> str:
    """
    Validates a parsed call against the ground-truth signature of `func_obj`
    and reconstructs a corrected call expression:
      - keyword arguments not accepted by the callable are dropped,
      - required positional parameters missing from the call are restored as
        fresh placeholders ({param_name}),
      - supplied positional placeholders are preserved in declared order.
    """
    sig = resolve_signature(func_obj)
    if sig is None:
        return _format_call(func_expr, raw_args, raw_kwargs)

    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    accepted_kw = {
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    # 1. Drop spurious keyword arguments.
    kept_kwargs: Dict[str, str] = {}
    for k, v in raw_kwargs.items():
        if accepts_kwargs or k in accepted_kw or not accepted_kw:
            kept_kwargs[k] = v

    # 2. Restore missing required positional parameters as placeholders.
    pos_params = [
        p
        for p in params.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    supplied_count = min(len(raw_args), len(pos_params))
    restored_positional: List[str] = []

    for idx, p in enumerate(pos_params):
        if idx < supplied_count:
            continue
        if p.name in kept_kwargs:
            continue
        if p.default is inspect.Parameter.empty:
            restored_positional.append(p.name)
        else:
            # First optional positional gap ends restoration.
            break

    # 3. Restore missing required keyword-only parameters.
    for p in params.values():
        if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty:
            if p.name not in kept_kwargs:
                kept_kwargs[p.name] = p.name

    final_positional = list(raw_args) + [
        r for r in restored_positional if r not in raw_args
    ]

    return _format_call(func_expr, final_positional, kept_kwargs)


def repair_cell_semantics(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Dynamically validates and repairs a single cell against ground-truth runtime signatures.

    Returns True if the cell was modified.
    """
    modified = False

    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    if tmpl != cell.get("code_template", ""):
        cell["code_template"] = tmpl
        modified = True

    call_info = parse_call_expression(tmpl)
    if call_info:
        func_expr, raw_args, raw_kwargs, lhs = call_info
        func_obj = resolve_callable_from_expr(func_expr, cell.get("dependencies"))

        if func_obj is not None and callable(func_obj):
            reconstructed = validate_and_reconstruct_call(
                func_obj, func_expr, raw_args, raw_kwargs
            )

            # Preserve the original assignment target if one existed.
            # Never force `{output_var} =` onto void/sink calls or multi-assignments.
            if lhs:
                call_stmt = f"{lhs} = {reconstructed}"
            else:
                call_stmt = reconstructed

            prefix_code = getattr(call_info, "prefix_code", "")
            suffix_code = getattr(call_info, "suffix_code", "")
            parts = []
            if prefix_code:
                parts.append(prefix_code)
            parts.append(call_stmt)
            if suffix_code:
                parts.append(suffix_code)
            new_tmpl = "\n".join(parts)

            if new_tmpl != tmpl:
                cell["code_template"] = clean_malformed_template_braces(new_tmpl)
                modified = True

    # Enforce wiring invariant (placeholder sets vs declared input ports)
    if repair_wiring_invariant(cell, domain):
        modified = True

    return modified
