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
