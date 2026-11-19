"""
eval_metrics.py - honest, unit-testable scoring for the NSTL benchmark harness (test.py).

Design rules (research integrity):
  * Failure classes come from the engine's STRUCTURED failure record (`failure` in the run result), never from
    pattern-matching rendered console text. Text matching survives only as a fallback for subprocess runs.
  * "ARCHITECTURAL" is NOT a catch-all. A category is only claimed when the evidence supports it:
        VERIFIER_REJECTED  the planner produced a plan the pre-flight verifier proves wrong/incomplete
        BINDING_UNRESOLVED a required port received no value (prompt under-specified OR binder gap; the
                           unresolved ports are listed so a human can tell which)
        PLANNING           no path / unification failed / no code
        IMPLEMENTATION     engine crash, malformed emitted code, or generated code that crashes
  * A sandbox exit code of 0 only proves the program runs. PASSED additionally requires that every literal the
    prompt states (quoted strings, numbers) is actually consumed by the emitted code; otherwise the run is
    PASSED_UNVERIFIED and does NOT count as a success.
  * Counts are reported with Wilson confidence intervals; n=1 is never presented as a rate.
  * Nothing here is keyed to a preset, prompt, cell id or port name.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUCCESS_STATUSES = ("PASSED",)
# Statuses that mean "program was produced and ran" (used for the secondary, sandbox-only rate).
RAN_STATUSES = ("PASSED", "PASSED_UNVERIFIED")

STATUS_CATEGORY: Dict[str, str] = {
    "PASSED": "SUCCESS",
    "SYNTHESIZED": "SUCCESS",  # --no-exec runs: code emitted, execution intentionally skipped
    "PASSED_UNVERIFIED": "SUSPECT",
    "LINT_HALTED": "VERIFIER_REJECTED",
    "UNRESOLVED_PORT": "BINDING_UNRESOLVED",
    "PLANNING_FAILED": "PLANNING",
    "UNIFICATION_FAILED": "PLANNING",
    "NO_CODE": "PLANNING",
    "CONTRACT_VIOLATION": "PLANNING",
    "CRASHED": "IMPLEMENTATION",
    "EMISSION_DEFECT": "IMPLEMENTATION",
    "SANDBOX_FAILED": "IMPLEMENTATION",
    "FAILED": "IMPLEMENTATION",
    "COMPLETED": "IMPLEMENTATION",
    "SECURITY_BLOCKED": "SECURITY",
    "MODEL_UNAVAILABLE": "ENVIRONMENT",
    "PROFILE_LOAD_FAILED": "ENVIRONMENT",
    "TIMEOUT": "ENVIRONMENT",
    "THEORETICAL_REFUSAL_OK": "EXPECTED_NEGATIVE",
}

FAILURE_KIND_TO_STATUS: Dict[str, str] = {
    "INTERNAL_ERROR": "CRASHED",
    "EMISSION_DEFECT": "EMISSION_DEFECT",
    "PLANNING_FAILED": "PLANNING_FAILED",
    "UNIFICATION_FAILED": "UNIFICATION_FAILED",
    "UNRESOLVED_PORTS": "UNRESOLVED_PORT",
    "LINT_REJECTED": "LINT_HALTED",
    "NO_CODE": "NO_CODE",
    "SANDBOX_FAILED": "SANDBOX_FAILED",
}

# Categories in which a refusal to synthesize is an acceptable answer to an under-specified prompt.
REFUSAL_CATEGORIES = ("BINDING_UNRESOLVED", "VERIFIER_REJECTED", "PLANNING")


def category_of(status: str) -> str:
    return STATUS_CATEGORY.get(status, "UNKNOWN")


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion; (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def format_rate(successes: int, n: int) -> str:
    """'3/20 = 15.0% [95% CI 5.2-36.0]' and an explicit small-n warning."""
    if n <= 0:
        return "n/a (no runs)"
    lo, hi = wilson_interval(successes, n)
    s = f"{successes}/{n} = {successes / n * 100:.1f}% [95% CI {lo * 100:.1f}-{hi * 100:.1f}]"
    if n < 10:
        s += f"  (n={n}: too small to call a rate)"
    return s


# ---------------------------------------------------------------------------------------------
# Literal-consumption verification (generic: derived from the prompt text only)
# ---------------------------------------------------------------------------------------------
def extract_prompt_literals(prompt: str) -> Dict[str, List[Any]]:
    """
    Pulls out what the prompt STATES explicitly: quoted strings and standalone numeric tokens.
    Numbers inside quoted text belong to the string and are not extracted separately.
    """
    strings: List[str] = []
    numbers: List[float] = []
    i, n = 0, len(prompt)
    buf: List[str] = []

    def flush_numbers(text: str) -> None:
        tok = []
        for ch in text + " ":
            if ch.isalnum() or ch in ".":
                tok.append(ch)
            else:
                if tok:
                    t = "".join(tok).strip(".")
                    if t and t.replace(".", "", 1).isdigit():
                        try:
                            numbers.append(float(t))
                        except ValueError:
                            pass
                    tok = []

    while i < n:
        ch = prompt[i]
        if ch in ("'", '"'):
            j = prompt.find(ch, i + 1)
            # an apostrophe inside a word ("don't") is not a quote
            if ch == "'" and i > 0 and prompt[i - 1].isalnum():
                buf.append(ch)
                i += 1
                continue
            if j == -1:
                buf.append(ch)
                i += 1
                continue
            flush_numbers("".join(buf))
            buf = []
            s = prompt[i + 1:j]
            if s:
                strings.append(s)
            i = j + 1
        else:
            buf.append(ch)
            i += 1
    flush_numbers("".join(buf))
    return {"strings": strings, "numbers": numbers}


def literal_consumption(prompt: str, code: str) -> Dict[str, Any]:
    """
    Checks that every stated literal appears as a constant in the emitted program.
    Returns {"required": int, "found": int, "missing": [..], "checked": bool}.
    `checked` is False when the code cannot be parsed (the caller must not treat that as verified).
    """
    lits = extract_prompt_literals(prompt or "")
    required = len(lits["strings"]) + len(lits["numbers"])
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {"required": required, "found": 0, "missing": ["<code does not parse>"], "checked": False}

    str_consts: List[str] = []
    num_consts: List[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                num_consts.append(float(v))
            elif isinstance(v, str):
                str_consts.append(v)
    haystack = "\n".join(str_consts)

    missing: List[str] = []
    for s in lits["strings"]:
        # a literal may be embedded in a larger constant (a path, a format string) but must be present
        if not any(s == c for c in str_consts) and s not in haystack:
            missing.append(f"string {s!r}")
    for x in lits["numbers"]:
        if not any(math.isclose(x, c, rel_tol=1e-9, abs_tol=1e-9) for c in num_consts):
            missing.append(f"number {x:g}")
    return {"required": required, "found": required - len(missing), "missing": missing, "checked": True}


# ---------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------
def _fmt_violations(violations: Sequence[str], limit: int = 6) -> str:
    vs = [str(v).strip() for v in violations if str(v).strip()]
    head = "; ".join(vs[:limit])
    if len(vs) > limit:
        head += f"; (+{len(vs) - limit} more)"
    return head


def classify_from_record(record: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """(status, detail, failure) from the engine's structured record, or None when it has none."""
    failure = record.get("failure") if isinstance(record, dict) else None
    if failure is None:
        sb = (record or {}).get("sandbox_result") or {}
        if (record or {}).get("code") and sb.get("success"):
            return "PASSED", "Sandbox exit 0", {}
        return None
    kind = str(failure.get("kind") or "")
    status = FAILURE_KIND_TO_STATUS.get(kind)
    if status is None:
        return None
    if kind == "LINT_REJECTED":
        detail = _fmt_violations(failure.get("violations") or []) or str(failure.get("detail", ""))
    elif kind == "UNRESOLVED_PORTS":
        ports = failure.get("unresolved_ports") or []
        detail = ("unresolved ports: " + ", ".join(ports)) if ports else str(failure.get("detail", ""))
    else:
        detail = str(failure.get("detail", ""))
    return status, detail, dict(failure)


def finalize_success(status: str, detail: str, prompt: str, code: str, record: Optional[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Upgrades a sandbox pass to a VERIFIED pass only when the emitted program consumes every literal the
    prompt states, and the pre-flight lint actually ran. Returns (status, detail, verification_report).
    """
    report = literal_consumption(prompt, code)
    lint_error = (record or {}).get("lint_error") if isinstance(record, dict) else None
    problems: List[str] = []
    if not report["checked"]:
        problems.append("emitted code does not parse")
    if report["missing"] and report["checked"]:
        problems.append("prompt literals not consumed: " + ", ".join(report["missing"]))
    if lint_error:
        problems.append(f"pre-flight lint did not run ({lint_error})")
    report["lint_error"] = lint_error
    if problems:
        return "PASSED_UNVERIFIED", "Sandbox exit 0 but " + "; ".join(problems), report
    return status, detail, report


def outcome_ok(status: str, expected: str) -> bool:
    """
    Did the system do an acceptable thing for this preset?
      expected == "synthesize": only a verified pass counts.
      expected == "clarify":    a verified pass, OR a refusal that names what is missing (binding / verifier /
                                planning), counts; an engine defect never does.
    """
    if status in SUCCESS_STATUSES:
        return True
    if expected == "clarify":
        return category_of(status) in REFUSAL_CATEGORIES
    return False


def funnel(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Monotone funnel: attempted >= emitted_code >= lint_clean >= ran >= verified."""
    attempted = len(results)
    emitted = sum(1 for r in results if r.get("code"))
    lint_clean = sum(1 for r in results if r.get("code") and r.get("status") not in ("LINT_HALTED", "EMISSION_DEFECT", "NO_CODE"))
    ran = sum(1 for r in results if r.get("status") in RAN_STATUSES)
    verified = sum(1 for r in results if r.get("status") in SUCCESS_STATUSES)
    return {"attempted": attempted, "emitted_code": emitted, "lint_clean": lint_clean, "ran_in_sandbox": ran, "verified": verified}
