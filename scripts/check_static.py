#!/usr/bin/env python3
"""
scripts/check_static.py - Undefined-name gate (R6-3).

Runs pyflakes over src/, tests/ and scripts/ and fails on any `undefined name`
report. py_compile cannot see an undefined name; round 5 shipped a crash that
failed all 45 sweep runs behind exactly this class of defect (R6-1/R6-3).

Usage:
    python scripts/check_static.py

Exit codes: 0 = clean, 1 = undefined names (or pyflakes unavailable).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGETS = ["src", "tests", "scripts"]

UNDEFINED_NAME_RE = re.compile(r": undefined name '([^']+)'")

# Architectural hygiene gates enforced over non-legacy engine source under src/.
# Note: universal_harvester.py is legacy/out-of-scope per project policy.
LEGACY_EXEMPT = {"universal_harvester.py"}
FORBIDDEN_IDENTIFIERS = {
    "frozenset",
    "allow_fallback",
    "role_semantics",
    "get_verification_semantics",
    "register_verification_semantics",
    "SemanticsSet",
}
FORBIDDEN_ATTR_PAT = re.compile(r"\bre\.")

# Subprocess text output is inspected structurally via the compiled pattern
# above; scripts/ tooling may use regex (the no-regex rule applies to src/).


def _check_src_hygiene() -> list[str]:
    violations: list[str] = []
    src_dir = ROOT_DIR / "src"
    for py_path in sorted(src_dir.rglob("*.py")):
        if py_path.name in LEGACY_EXEMPT:
            continue
        rel = py_path.relative_to(ROOT_DIR).as_posix()
        src_text = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src_text, filename=rel)
        except SyntaxError as exc:
            violations.append(f"{rel}:{exc.lineno}: SyntaxError: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "re" or alias.name.startswith("re."):
                        violations.append(f"{rel}:{node.lineno}: forbidden import of regex module 're'")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "re" or (node.module or "").startswith("re."):
                    violations.append(f"{rel}:{node.lineno}: forbidden from-import of regex module 're'")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{rel}:{node.lineno}: forbidden identifier '{node.id}'")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{rel}:{node.lineno}: forbidden attribute '{node.attr}'")

        for lineno, line in enumerate(src_text.splitlines(), start=1):
            code_part = line.split("#", 1)[0]
            if FORBIDDEN_ATTR_PAT.search(code_part):
                violations.append(f"{rel}:{lineno}: forbidden regex call 're.'")
    return violations


def _iter_scope_without_inner_classes(fn_node: ast.AST):
    """Yield AST nodes within a method scope, skipping nested class definitions."""
    stack: list[ast.AST] = list(getattr(fn_node, "body", ()))
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.ClassDef):
            continue
        yield curr
        stack.extend(ast.iter_child_nodes(curr))


def _extract_self_attrs_from_target(tgt: ast.AST) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(tgt):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            attrs.add(node.attr)
    return attrs


def _collect_self_attr_assignments(fn_node: ast.AST) -> set[str]:
    attrs: set[str] = set()
    for node in _iter_scope_without_inner_classes(fn_node):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                attrs.update(_extract_self_attrs_from_target(tgt))
        elif isinstance(node, ast.AnnAssign):
            attrs.update(_extract_self_attrs_from_target(node.target))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attrs.add(node.args[1].value)
    return attrs


def _check_self_method_resolution() -> list[str]:
    """
    AST static gate: for classes in src/ whose inheritance closes within src/ (or object),
    verify every direct `self.<method>(...)` call resolves to a defined method/attribute
    on the class or its ancestors. Catches typos such as calling `self.get_type_kind`
    instead of `self.has_type_kind`.
    """
    src_dir = ROOT_DIR / "src"
    class_records: dict[str, dict] = {}
    parsed_files: list[tuple[str, ast.AST]] = []

    for py_path in sorted(src_dir.rglob("*.py")):
        if py_path.name in LEGACY_EXEMPT:
            continue
        rel = py_path.relative_to(ROOT_DIR).as_posix()
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        parsed_files.append((rel, tree))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            members: set[str] = set()
            has_getattr = False
            bases: list[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)

            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(stmt.name)
                    if stmt.name == "__getattr__":
                        has_getattr = True
                    members.update(_collect_self_attr_assignments(stmt))
                elif isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            members.add(tgt.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    members.add(stmt.target.id)

            rec = class_records.setdefault(
                node.name,
                {"bases": set(), "members": set(), "has_getattr": False},
            )
            rec["bases"].update(bases)
            rec["members"].update(members)
            rec["has_getattr"] = rec["has_getattr"] or has_getattr

    def _is_closed_hierarchy(cls_name: str, seen: set[str]) -> bool:
        if cls_name in seen:
            return True
        rec = class_records.get(cls_name)
        if rec is None:
            return False
        if rec["has_getattr"]:
            return False
        seen.add(cls_name)
        for base in rec["bases"]:
            if base == "object":
                continue
            if base not in class_records or not _is_closed_hierarchy(base, seen):
                return False
        return True

    def _all_members(cls_name: str, seen: set[str]) -> set[str]:
        if cls_name in seen:
            return set()
        seen.add(cls_name)
        rec = class_records.get(cls_name)
        if rec is None:
            return set()
        out = set(rec["members"])
        for base in rec["bases"]:
            if base != "object":
                out.update(_all_members(base, seen))
        return out

    violations: list[str] = []
    for rel, tree in parsed_files:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_closed_hierarchy(node.name, set()):
                continue
            resolved_members = _all_members(node.name, set())
            for stmt in node.body:
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for inner in _iter_scope_without_inner_classes(stmt):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and isinstance(inner.func.value, ast.Name)
                        and inner.func.value.id == "self"
                    ):
                        attr = inner.func.attr
                        if attr not in resolved_members:
                            violations.append(
                                f"{rel}:{inner.lineno}: {node.name}.{stmt.name} calls unresolved `self.{attr}(...)`"
                            )
    return violations


def main() -> int:
    try:
        import pyflakes  # noqa: F401
    except ImportError:
        print(
            "[check_static] pyflakes is not installed; install it with:\n"
            "    python -m pip install pyflakes",
            file=sys.stderr,
        )
        return 1

    hygiene_violations = _check_src_hygiene()
    if hygiene_violations:
        print("[check_static] FAIL — architectural hygiene violations in src/:\n")
        for line in hygiene_violations:
            print(f"  {line}")
        return 1

    method_violations = _check_self_method_resolution()
    if method_violations:
        print("[check_static] FAIL — unresolved self.<method>() calls in src/:\n")
        for line in method_violations:
            print(f"  {line}")
        return 1

    cmd = [sys.executable, "-m", "pyflakes"] + TARGETS
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    undefined = []
    for line in output.splitlines():
        m = UNDEFINED_NAME_RE.search(line)
        if m:
            undefined.append(line.strip())

    if undefined:
        print("[check_static] FAIL — undefined names found:\n")
        for line in undefined:
            print(f"  {line}")
        print(f"\n{len(undefined)} undefined name(s). Fix these before merging.")
        return 1

    other = [l for l in output.splitlines() if l.strip()]
    print("[check_static] OK — no undefined names or architectural hygiene violations.")
    if other:
        print(f"[check_static] note: {len(other)} non-undefined pyflakes report(s) remain (unused imports/variables):")
        for line in other[:10]:
            print(f"  {line}")
        if len(other) > 10:
            print(f"  ... and {len(other) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
