#!/usr/bin/env python3
"""
scripts/check_static.py — STRATA Architectural Hygiene, Neutrality & Static Gate.

Enforces:
  1. Zero forbidden constructs (`re` outside formal_regex.py, `frozenset`, etc.).
  2. Clean core kernel neutrality (zero hardcoded library/domain tokens in src/).
  3. GEVRSandbox import & attribute neutrality (zero hardcoded third-party hooks).
  4. Model-agnostic core/inference modules & backend-neutral PortBinder AST isolation.
  5. Tooling script neutrality (scripts/ and tools/ contain zero hardcoded domain logic).
  6. Closed-hierarchy self.<method>() resolution and pyflakes undefined-name checks.

Usage:
    python3 scripts/check_static.py
"""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGETS = [
    d for d in ("src", "scripts", "tools", "benchmarks", "tests")
    if (ROOT_DIR / d).is_dir()
]

UNDEFINED_NAME_RE = re.compile(r": undefined name '([^']+)'")

# Architectural hygiene gates enforced over engine source under src/.
# formal_regex.py is explicitly quarantined as the ONLY module in src/ allowed to use regex.
LEGACY_EXEMPT: set[str] = set()
REGEX_ALLOWED_REL_PATHS = {"src/frontends/formal_regex.py"}
CLEAN_CORE_MODULES = {
    "src/schema.py",
    "src/type_system.py",
    "src/bridge_graph.py",
    "src/typed_ir.py",
    "src/frontends/__init__.py",
    "src/frontends/formal_regex.py",
    "src/frontends/slm_compiler.py",
    "src/subgoal_retrieval.py",
    "src/typed_planner.py",
    "src/port_binder.py",
    "src/profile_adapters.py",
    "src/strata_pipeline.py",
    "src/language_backend.py",
    "src/utils.py",
    "src/security_policy.py",
}
EXPLICIT_FORBIDDEN_LIBS = {
    "numpy", "pandas", "sklearn", "cv2", "pygame", "pillow", "pil",
    "matplotlib", "seaborn", "scipy", "soundfile", "cryptography",
    "sqlite3", "requests", "torch", "tensorflow",
}
FORBIDDEN_SANDBOX_ATTRS = {
    "_fit_X",
    "__sklearn_is_fitted__",
    "ndarray",
    "patches",
    "artists",
}
FORBIDDEN_MODEL_TOKENS = (
    "qwen",
    "nano",
    "jina",
    "gemma",
    "mistral",
    "deepseek",
    "claude",
    "gpt-4",
    "gpt-oss",
    "phi-3",
    "phi-4",
    "bge-",
    "e5-",
    "minilm",
)
MODEL_CHECK_MODULES = {
    "src/config.py",
    "src/inference.py",
    "src/reranker.py",
    "src/cli.py",
    "src/strata_pipeline.py",
    "src/profile_adapters.py",
    "src/subgoal_retrieval.py",
}
TOOLING_CHECK_MODULES = {
    "scripts/lint_trees.py",
    "scripts/audit_trees.py",
    "scripts/neutralize_trees.py",
    "tools/compile_trees.py",
}
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
# above; scripts/ tooling may use regex (the no-regex rule applies to src/,
# except for the quarantined formal-syntax frontend).


def _check_src_hygiene() -> list[str]:
    violations: list[str] = []
    src_dir = ROOT_DIR / "src"
    for py_path in sorted(src_dir.rglob("*.py")):
        if py_path.name in LEGACY_EXEMPT:
            continue
        rel = py_path.relative_to(ROOT_DIR).as_posix()
        allow_regex = rel in REGEX_ALLOWED_REL_PATHS
        src_text = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src_text, filename=rel)
        except SyntaxError as exc:
            violations.append(f"{rel}:{exc.lineno}: SyntaxError: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not allow_regex and (alias.name == "re" or alias.name.startswith("re.")):
                        violations.append(f"{rel}:{node.lineno}: forbidden import of regex module 're'")
            elif isinstance(node, ast.ImportFrom):
                if not allow_regex and (node.module == "re" or (node.module or "").startswith("re.")):
                    violations.append(f"{rel}:{node.lineno}: forbidden from-import of regex module 're'")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{rel}:{node.lineno}: forbidden identifier '{node.id}'")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
                violations.append(f"{rel}:{node.lineno}: forbidden attribute '{node.attr}'")

        if not allow_regex:
            for lineno, line in enumerate(src_text.splitlines(), start=1):
                code_part = line.split("#", 1)[0]
                if FORBIDDEN_ATTR_PAT.search(code_part):
                    violations.append(f"{rel}:{lineno}: forbidden regex call 're.'")
    return violations


def _check_clean_module_neutrality() -> list[str]:
    """
    Verify clean core modules contain zero imports or string references to concrete
    library or non-core domain names loaded from trees/*.json.
    """
    forbidden_names = set(EXPLICIT_FORBIDDEN_LIBS)
    trees_dir = ROOT_DIR / "trees"
    if trees_dir.is_dir():
        for tree_path in trees_dir.glob("*.json"):
            stem = tree_path.stem.strip().lower()
            if stem and stem not in ("python_core", "control_flow"):
                forbidden_names.add(stem)

    violations: list[str] = []
    for rel in sorted(CLEAN_CORE_MODULES):
        py_path = ROOT_DIR / rel
        if not py_path.is_file():
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".", 1)[0].strip().lower()
                    if top_mod in forbidden_names:
                        violations.append(f"{rel}:{node.lineno}: non-neutral import of domain/library '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                top_mod = (node.module or "").split(".", 1)[0].strip().lower()
                if top_mod in forbidden_names:
                    violations.append(f"{rel}:{node.lineno}: non-neutral from-import of domain/library '{node.module}'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                toks = set(re.findall(r"[A-Za-z0-9_]+", node.value.lower()))
                hits = sorted(toks & forbidden_names)
                if hits:
                    violations.append(f"{rel}:{node.lineno}: non-neutral domain/library reference {hits}")
    return violations


def _check_sandbox_import_neutrality() -> list[str]:
    """Verify src/gevr_sandbox.py contains zero hardcoded third-party library imports or attribute hooks."""
    sb_path = ROOT_DIR / "src" / "gevr_sandbox.py"
    if not sb_path.is_file():
        return []
    third_party_libs = EXPLICIT_FORBIDDEN_LIBS - {"pathlib"}
    violations: list[str] = []
    try:
        tree = ast.parse(sb_path.read_text(encoding="utf-8"), filename="src/gevr_sandbox.py")
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".", 1)[0].strip().lower()
                if top_mod in third_party_libs:
                    violations.append(f"src/gevr_sandbox.py:{node.lineno}: hardcoded third-party import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            top_mod = (node.module or "").split(".", 1)[0].strip().lower()
            if top_mod in third_party_libs:
                violations.append(f"src/gevr_sandbox.py:{node.lineno}: hardcoded third-party from-import '{node.module}'")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_SANDBOX_ATTRS:
                violations.append(f"src/gevr_sandbox.py:{node.lineno}: library-specific attribute '{node.attr}'")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_SANDBOX_ATTRS:
                violations.append(f"src/gevr_sandbox.py:{node.lineno}: library-specific identifier '{node.id}'")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            toks = re.findall(r"[A-Za-z0-9_]+", node.value)
            if node.value in FORBIDDEN_SANDBOX_ATTRS or any(t in FORBIDDEN_SANDBOX_ATTRS for t in toks):
                violations.append(f"src/gevr_sandbox.py:{node.lineno}: library-specific duck-type literal {node.value!r}")
            elif any(t.lower() in third_party_libs for t in toks):
                violations.append(f"src/gevr_sandbox.py:{node.lineno}: hardcoded third-party reference {node.value!r}")
    return violations


def _check_hardcoded_models_and_ast() -> list[str]:
    """Forbid hardcoded model tokens in config/inference/reranker and forbid Python ast parsing in port_binder."""
    violations: list[str] = []
    for rel in sorted(MODEL_CHECK_MODULES):
        p = ROOT_DIR / rel
        if not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                hits = [tok for tok in FORBIDDEN_MODEL_TOKENS if tok in low]
                if hits:
                    violations.append(f"{rel}:{node.lineno}: hardcoded model token {hits} in string literal {node.value!r}")

    pb_path = ROOT_DIR / "src" / "port_binder.py"
    if pb_path.is_file():
        try:
            pb_tree = ast.parse(pb_path.read_text(encoding="utf-8"), filename="src/port_binder.py")
        except SyntaxError:
            pb_tree = None
        if pb_tree is not None:
            for node in ast.walk(pb_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "ast" or alias.name.startswith("ast."):
                            violations.append(f"src/port_binder.py:{node.lineno}: forbidden import '{alias.name}'")
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "ast" or (node.module or "").startswith("ast."):
                        violations.append(f"src/port_binder.py:{node.lineno}: forbidden from-import '{node.module}'")
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "ast"
                    and node.attr == "parse"
                ):
                    violations.append(f"src/port_binder.py:{node.lineno}: forbidden 'ast.parse' call")
    return violations


def _check_tooling_neutrality() -> list[str]:
    """Verify domain tree validators, auditors, normalizers, and compilers stay domain-neutral."""
    forbidden_names = set(EXPLICIT_FORBIDDEN_LIBS) - {"pathlib"}
    trees_dir = ROOT_DIR / "trees"
    if trees_dir.is_dir():
        for tree_path in trees_dir.glob("*.json"):
            stem = tree_path.stem.strip().lower()
            if stem and stem not in ("python_core", "control_flow"):
                forbidden_names.add(stem)

    violations: list[str] = []
    for rel in sorted(TOOLING_CHECK_MODULES):
        py_path = ROOT_DIR / rel
        if not py_path.is_file():
            continue
        allowed_stdlib_imports = {"pathlib", "sqlite3"} if rel == "tools/compile_trees.py" else {"pathlib"}
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".", 1)[0].strip().lower()
                    if top_mod in forbidden_names and top_mod not in allowed_stdlib_imports:
                        violations.append(f"{rel}:{node.lineno}: non-neutral tooling import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                top_mod = (node.module or "").split(".", 1)[0].strip().lower()
                if top_mod in forbidden_names and top_mod not in allowed_stdlib_imports:
                    violations.append(f"{rel}:{node.lineno}: non-neutral tooling from-import '{node.module}'")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                toks = set(re.findall(r"[A-Za-z0-9_]+", node.value.lower()))
                hits = sorted(toks & forbidden_names)
                if hits:
                    violations.append(f"{rel}:{node.lineno}: non-neutral tooling reference {hits}")
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
    if importlib.util.find_spec("pyflakes") is None:
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

    neutrality_violations = _check_clean_module_neutrality()
    if neutrality_violations:
        print("[check_static] FAIL — domain/library neutrality violations in clean core modules:\n")
        for line in neutrality_violations:
            print(f"  {line}")
        return 1

    sandbox_violations = _check_sandbox_import_neutrality()
    if sandbox_violations:
        print("[check_static] FAIL — third-party library hardcodes in src/gevr_sandbox.py:\n")
        for line in sandbox_violations:
            print(f"  {line}")
        return 1

    hardcode_violations = _check_hardcoded_models_and_ast()
    if hardcode_violations:
        print("[check_static] FAIL — hardcoded model tokens or target-AST violations:\n")
        for line in hardcode_violations:
            print(f"  {line}")
        return 1

    tooling_violations = _check_tooling_neutrality()
    if tooling_violations:
        print("[check_static] FAIL — domain/library hardcodes in tooling scripts:\n")
        for line in tooling_violations:
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
