"""
benchmarks/evaluators.py - Shared code extraction, headless sandbox runner,
static SDK typestate auditor, and resource telemetry utilities.
"""
from __future__ import annotations

import ast
import csv
import io
import math
import os
import resource
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_metrics import _extract_ast_literals

from .suites import SDK_SPECS


def read_proc_mem_mb() -> Dict[str, Optional[float]]:
    rss_kb = None
    hwm_kb = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = float(line.split()[1])
    except Exception:
        pass
    return {
        "rss_mb": round(rss_kb / 1024.0, 2) if rss_kb is not None else None,
        "hwm_mb": round(hwm_kb / 1024.0, 2) if hwm_kb is not None else None,
    }


def read_gpu_vram_mb() -> Dict[str, Optional[int]]:
    total_used = None
    proc_used = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        if out:
            total_used = int(float(out.splitlines()[0].strip()))
    except Exception:
        pass
    try:
        pid = os.getpid()
        out2 = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        for line in out2.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                proc_used = int(float(parts[1]))
    except Exception:
        pass
    return {"total_used_mb": total_used, "proc_used_mb": proc_used}


def read_cpu_times() -> Dict[str, float]:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_user": round(s.ru_utime, 4),
        "self_sys": round(s.ru_stime, 4),
        "child_user": round(c.ru_utime, 4),
        "child_sys": round(c.ru_stime, 4),
    }


def _strip_ci_closed_spans(text: str, open_tag: str, close_tag: str) -> str:
    """Removes all closed case-insensitive `<open_tag>...<close_tag>` spans."""
    open_l = open_tag.lower()
    close_l = close_tag.lower()
    out: List[str] = []
    pos = 0
    low = text.lower()
    while True:
        start = low.find(open_l, pos)
        if start == -1:
            out.append(text[pos:])
            break
        end = low.find(close_l, start + len(open_l))
        if end == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        pos = end + len(close_l)
    return "".join(out)


def strip_reasoning_traces(raw_text: str) -> str:
    """Removes closed/unclosed <think> traces and harmony/channel reasoning wrappers."""
    if not raw_text:
        return ""
    cleaned = _strip_ci_closed_spans(raw_text, "<think>", "</think>")
    # Strip unclosed/truncated <think>... blocks when generation hits max_tokens mid-thought
    open_idx = cleaned.lower().find("<think>")
    if open_idx != -1:
        cleaned = cleaned[:open_idx]
    # Strip orphaned leading ...</think> when chat template pre-filled <think>
    close_idx = cleaned.lower().find("</think>")
    if close_idx != -1:
        cleaned = cleaned[close_idx + len("</think>") :]

    final_marker = "<|channel|>final<|message|>"
    if final_marker in cleaned:
        cleaned = cleaned.split(final_marker)[-1]
    else:
        open_markers = ("<|channel|>analysis<|message|>", "<|channel|>commentary<|message|>")
        stop_markers = ("<|channel|>", "<|end|>", "<|return|>")
        while True:
            low = cleaned.lower()
            found_start = -1
            found_len = 0
            for om in open_markers:
                idx = low.find(om)
                if idx != -1 and (found_start == -1 or idx < found_start):
                    found_start = idx
                    found_len = len(om)
            if found_start == -1:
                break
            search_from = found_start + found_len
            stop_idx = -1
            for sm in stop_markers:
                idx = low.find(sm, search_from)
                if idx != -1 and (stop_idx == -1 or idx < stop_idx):
                    stop_idx = idx
            if stop_idx == -1:
                cleaned = cleaned[:found_start]
            else:
                cleaned = cleaned[:found_start] + cleaned[stop_idx:]

    for tok in ("<|end|>", "<|return|>", "<|start|>assistant", "<|message|>"):
        cleaned = cleaned.replace(tok, "")
    return cleaned.strip()


def extract_code_clean(raw_text: str) -> str:
    """Extracts the final executable Python code block from raw model output."""
    cleaned = strip_reasoning_traces(raw_text)
    if not cleaned:
        return ""

    blocks: List[str] = []
    open_fallback: Optional[str] = None
    pos = 0
    while True:
        fence_start = cleaned.find("```", pos)
        if fence_start == -1:
            break
        after_ticks = fence_start + 3
        nl_idx = cleaned.find("\n", after_ticks)
        if nl_idx == -1:
            break
        info = cleaned[after_ticks:nl_idx].strip().lower()
        close_idx = cleaned.find("```", nl_idx + 1)
        if info in ("", "python", "py"):
            if close_idx != -1:
                blocks.append(cleaned[nl_idx + 1 : close_idx])
                pos = close_idx + 3
            else:
                if open_fallback is None:
                    open_fallback = cleaned[nl_idx + 1 :].strip()
                break
        else:
            if close_idx != -1:
                pos = close_idx + 3
            else:
                pos = nl_idx + 1

    if blocks:
        # Prefer the last non-empty block (models sometimes draft then output final code)
        for blk in reversed(blocks):
            if blk.strip():
                return blk.strip()

    if open_fallback is not None:
        return open_fallback

    # If the response contains prose (refusal/clarification) and no Python AST, return empty
    try:
        ast.parse(cleaned)
        return cleaned
    except SyntaxError:
        return ""


def check_ast(code: str) -> Dict[str, Any]:
    if not code or not code.strip():
        return {"ast_ok": False, "ast_error": "empty"}
    try:
        ast.parse(code)
        return {"ast_ok": True, "ast_error": None}
    except SyntaxError as exc:
        return {"ast_ok": False, "ast_error": f"{exc.msg} (line {exc.lineno})"}


def verify_literals_consumed(code: str, expected_literals: List[Any]) -> List[Any]:
    """Checks that prompt-specified literals are present as Python AST constants or JoinedStr fragments."""
    if not expected_literals or not code.strip():
        return list(expected_literals) if not code.strip() else []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return list(expected_literals)

    str_consts, num_consts = _extract_ast_literals(tree)
    bool_consts = {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, bool)
    }

    missing: List[Any] = []
    for lit in expected_literals:
        if isinstance(lit, bool):
            if lit not in bool_consts:
                missing.append(lit)
        elif isinstance(lit, float):
            if not any(abs(c - lit) < 1e-6 for c in num_consts):
                missing.append(lit)
        elif isinstance(lit, int):
            if not any(math.isclose(float(lit), c, rel_tol=1e-9, abs_tol=1e-9) for c in num_consts):
                missing.append(lit)
        elif isinstance(lit, str):
            if not any(lit == c or lit in c for c in str_consts):
                missing.append(lit)
        else:
            missing.append(lit)
    return missing


def provision_sandbox_fixtures(workdir: Path) -> None:
    """Ensures input.csv, input.png, and input_1.wav exist in workdir."""
    # Reuse repo fixtures if available, otherwise synthesize minimal valid fixtures
    for fname in ("input.csv", "input.png", "input_1.wav"):
        src = ROOT_DIR / fname
        dst = workdir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    csv_path = workdir / "input.csv"
    if not csv_path.exists():
        lines = ["X,Y,label"]
        for i in range(1, 33):
            lines.append(f"{i * 0.5:.2f},{i * 1.25 + 2.0:.2f},c{i % 3}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    png_path = workdir / "input.png"
    if not png_path.exists():
        import numpy as np
        try:
            import cv2
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.rectangle(img, (12, 12), (52, 52), (220, 220, 220), -1)
            cv2.imwrite(str(png_path), img)
        except Exception:
            pass

    wav_path = workdir / "input_1.wav"
    if not wav_path.exists():
        import math
        sr = 8000
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            frames = bytearray()
            for i in range(sr):
                val = int(12000 * math.sin(2.0 * math.pi * 440.0 * i / sr))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(bytes(frames))


def _task_egress_paths(task: Dict[str, Any]) -> List[str]:
    spec = task.get("artifact_spec")
    if isinstance(spec, dict):
        if str(spec.get("kind") or "").strip().lower() == "sqlite_clean_table" and spec.get("db_path"):
            return [str(spec["db_path"])]
        if spec.get("egress_paths"):
            return [str(p) for p in spec["egress_paths"] if p]
    return [str(p) for p in (task.get("egress_paths") or ()) if p]


def verify_task_artifacts(
    task: Dict[str, Any],
    workdir: Path,
    input_csv_path: Optional[Path] = None,
) -> Optional[str]:
    """Verifies declarative task output artifacts semantically (returns error message or None)."""
    spec = task.get("artifact_spec")
    if not isinstance(spec, dict):
        return None
    if str(spec.get("kind") or "").strip().lower() != "sqlite_clean_table":
        return None

    db_rel = str(spec.get("db_path") or "").strip()
    table_name = str(spec.get("table") or "").strip()
    source_csv = str(spec.get("source_csv") or "").strip()
    if not db_rel or not table_name:
        return "Incomplete sqlite_clean_table artifact_spec (missing db_path or table)"

    wd = Path(workdir)
    db_path = wd / db_rel
    if not db_path.is_file() or db_path.stat().st_size <= 0:
        return f"Output SQLite database '{db_rel}' was not created or is empty"

    csv_candidate: Optional[Path] = Path(input_csv_path) if input_csv_path is not None else None
    if (csv_candidate is None or not csv_candidate.is_file()) and source_csv:
        cand = wd / source_csv
        csv_candidate = cand if cand.is_file() else (ROOT_DIR / source_csv)
    if csv_candidate is None or not csv_candidate.is_file():
        return f"Reference CSV '{source_csv or '<unspecified>'}' unavailable for SQLite artifact verification"

    raw_null_tokens = spec.get("null_tokens")
    null_tokens = (
        {str(tok).strip().lower() for tok in raw_null_tokens}
        if raw_null_tokens is not None
        else {"", "nan", "na", "null", "none"}
    )
    with open(csv_candidate, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        input_cols = [c for c in (reader.fieldnames or []) if c]
        rows_csv = list(reader)

    if not input_cols:
        return f"Reference CSV '{csv_candidate.name}' has no columns"

    expected_clean_rows = sum(
        1
        for r in rows_csv
        if all(r.get(c) is not None and str(r.get(c)).strip().lower() not in null_tokens for c in input_cols)
    )

    quoted_table = '"' + table_name.replace('"', '""') + '"'
    reject_diag = bool(spec.get("reject_square_identity_matrix", True))
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cur.fetchone() is None:
            return f"SQLite table '{table_name}' not found in '{db_rel}'"

        cur.execute(f"PRAGMA table_info({quoted_table})")
        table_cols = [str(r[1]) for r in cur.fetchall()]
        missing_cols = [c for c in input_cols if c not in table_cols]
        if missing_cols:
            return f"Table '{table_name}' missing required columns {missing_cols}; found {table_cols}"

        cur.execute(f"SELECT COUNT(*) FROM {quoted_table}")
        row_count = int(cur.fetchone()[0])
        if row_count == 0:
            return f"Table '{table_name}' is empty (0 rows)"

        quoted_cols = ", ".join('"' + c.replace('"', '""') + '"' for c in input_cols)
        cur.execute(f"SELECT {quoted_cols} FROM {quoted_table}")
        fetched = cur.fetchall()

        if reject_diag and row_count == len(input_cols) and expected_clean_rows > len(input_cols):
            diag_vals: List[float] = []
            for idx_r in range(row_count):
                cell_v = fetched[idx_r][idx_r]
                if isinstance(cell_v, (int, float)) and not isinstance(cell_v, bool):
                    diag_vals.append(float(cell_v))
            if len(diag_vals) == len(input_cols) and all(abs(v - 1.0) < 1e-6 for v in diag_vals):
                return (
                    f"Table '{table_name}' appears to store a {row_count}x{len(input_cols)} "
                    f"correlation matrix rather than the cleaned dataframe ({expected_clean_rows} rows)"
                )

        if row_count != expected_clean_rows:
            return (
                f"Table '{table_name}' has {row_count} rows, expected {expected_clean_rows} "
                f"cleaned non-null rows from '{csv_candidate.name}'"
            )

        for r_vals in fetched:
            for val in r_vals:
                if val is None:
                    return f"Table '{table_name}' contains NULL values in required columns (missing rows not dropped)"
                if isinstance(val, str) and val.strip().lower() in null_tokens:
                    return f"Table '{table_name}' contains blank/NaN string values in required columns"
                if isinstance(val, float) and not math.isfinite(val):
                    return f"Table '{table_name}' contains non-finite values in required columns"
    finally:
        conn.close()

    return None


def run_sandbox_eval(code: str, task: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Executes generated Python code inside STRATA's headless GEVRSandbox and checks literal grounding."""
    ast_info = check_ast(code)
    if not ast_info["ast_ok"]:
        return {
            "status": "NO_CODE" if not code.strip() else "SANDBOX_FAILED",
            "category": "IMPLEMENTATION",
            "detail": ast_info["ast_error"] or "Empty code",
            "ast_ok": False,
            "missing_literals": list(task.get("expected_literals") or ()),
            "stdout": "",
            "stderr": "",
        }

    provision_sandbox_fixtures(ROOT_DIR)
    src_str = str(ROOT_DIR / "src")
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from gevr_sandbox import GEVRSandbox

    expected_egress = _task_egress_paths(task)
    sandbox = GEVRSandbox(default_timeout=timeout, allowed_capabilities=set())
    sb = sandbox.execute(
        code,
        timeout=timeout,
        egress_paths=expected_egress or None,
        capture_egress=bool(expected_egress),
    )
    err_txt = str(sb.get("error") or sb.get("stderr") or "").strip()

    if not sb.get("success", False):
        err_kind = str(sb.get("error_kind") or "").strip().lower()
        if err_kind == "security":
            st, cat = "SECURITY_BLOCKED", "SECURITY"
        elif err_kind in ("egress", "contract"):
            st, cat = "SEMANTIC_FAILED", "IMPLEMENTATION"
        else:
            st, cat = "SANDBOX_FAILED", "IMPLEMENTATION"
        last_line = err_txt.splitlines()[-1] if err_txt else "Runtime failure"
        return {
            "status": st,
            "category": cat,
            "detail": last_line[:160],
            "ast_ok": True,
            "missing_literals": verify_literals_consumed(code, task.get("expected_literals") or []),
            "stdout": str(sb.get("stdout") or "")[-1000:],
            "stderr": err_txt[-1500:],
        }

    if expected_egress:
        art_err: Optional[str] = None
        captured = sb.get("egress_bytes") or {}
        spec = task.get("artifact_spec") if isinstance(task.get("artifact_spec"), dict) else {}
        src_csv = str(spec.get("source_csv") or "").strip() if spec else ""
        ref_csv_path = (ROOT_DIR / src_csv) if src_csv else None
        with tempfile.TemporaryDirectory(prefix="strata_eval_art_") as tmp_dir:
            tmp_root = Path(tmp_dir).resolve()
            for rel_p, raw_bytes in captured.items():
                dst = (tmp_root / str(rel_p)).resolve()
                if dst.is_relative_to(tmp_root) and isinstance(raw_bytes, (bytes, bytearray)):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(bytes(raw_bytes))
            art_err = verify_task_artifacts(task, tmp_root, input_csv_path=ref_csv_path)
        if art_err:
            return {
                "status": "SEMANTIC_FAILED",
                "category": "IMPLEMENTATION",
                "detail": art_err,
                "ast_ok": True,
                "missing_literals": verify_literals_consumed(code, task.get("expected_literals") or []),
                "stdout": str(sb.get("stdout") or "")[-1000:],
                "stderr": err_txt[-1000:],
            }

    missing_lits = verify_literals_consumed(code, task.get("expected_literals") or [])
    if missing_lits:
        return {
            "status": "PASSED_UNVERIFIED",
            "category": "SUSPECT",
            "detail": f"Missing prompt literals: {missing_lits}",
            "ast_ok": True,
            "missing_literals": missing_lits,
            "stdout": str(sb.get("stdout") or "")[-1000:],
            "stderr": err_txt[-1000:],
        }
    return {
        "status": "PASSED",
        "category": "SUCCESS",
        "detail": "Live GEVR Sandbox verified + all literals consumed",
        "ast_ok": True,
        "missing_literals": [],
        "stdout": str(sb.get("stdout") or "")[-1000:],
        "stderr": err_txt[-1000:],
    }


def audit_sdk_code(code: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic AST + pyflakes + typestate-order + kwarg verifier for unseen SDK tasks."""
    sdk_name = task["sdk"]
    sdk_spec = SDK_SPECS[sdk_name]
    mod_name = sdk_spec["module"]
    api_specs = sdk_spec["api_specs"]
    forbidden = set(sdk_spec.get("forbidden_calls") or ())
    prereqs = sdk_spec["prereqs"]

    res: Dict[str, Any] = {
        "ast_ok": False,
        "imports_sdk": False,
        "calls_found": [],
        "hallucinated_calls": [],
        "forbidden_calls_used": [],
        "invalid_kwargs": [],
        "missing_required_calls": [],
        "order_violations": [],
        "missing_literals": [],
        "pyflakes_issues": [],
        "spec_compliant": False,
    }
    if not code or not code.strip():
        res["missing_required_calls"] = list(task.get("required_calls") or [])
        res["missing_literals"] = list(task.get("expected_literals") or [])
        return res

    try:
        tree = ast.parse(code)
        res["ast_ok"] = True
    except SyntaxError as se:
        res["pyflakes_issues"].append(f"SyntaxError: {se}")
        return res

    try:
        from pyflakes import api as pf_api
        from pyflakes import reporter as pf_rep
        out_io, err_io = io.StringIO(), io.StringIO()
        pf_api.check(code, f"<{mod_name}>", reporter=pf_rep.Reporter(out_io, err_io))
        for ln in (out_io.getvalue() + "\n" + err_io.getvalue()).splitlines():
            if ln.strip():
                res["pyflakes_issues"].append(ln.strip())
    except Exception:
        pass

    imported_direct: Dict[str, str] = {}
    mod_aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == mod_name:
                    res["imports_sdk"] = True
                    mod_aliases.add(alias.asname or mod_name)
        elif isinstance(node, ast.ImportFrom) and node.module == mod_name:
            res["imports_sdk"] = True
            for alias in node.names:
                imported_direct[alias.asname or alias.name] = alias.name

    calls_seq: List[str] = []
    call_nodes = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)),
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )
    for call in call_nodes:
        fn_name: Optional[str] = None
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            if call.func.value.id in mod_aliases or call.func.value.id == mod_name:
                fn_name = call.func.attr
        elif isinstance(call.func, ast.Name) and call.func.id in imported_direct:
            fn_name = imported_direct[call.func.id]
        elif isinstance(call.func, ast.Name) and call.func.id in (set(api_specs) | forbidden):
            fn_name = call.func.id
            res["hallucinated_calls"].append(f"unqualified:{fn_name}")

        if fn_name is None:
            continue
        if fn_name in forbidden:
            res["forbidden_calls_used"].append(fn_name)
            continue
        if fn_name not in api_specs:
            res["hallucinated_calls"].append(fn_name)
            continue

        calls_seq.append(fn_name)
        spec = api_specs[fn_name]
        if len(call.args) > len(spec["pos"]):
            res["invalid_kwargs"].append(f"{fn_name}: too many positional args ({len(call.args)})")
        for kw in call.keywords:
            if kw.arg and kw.arg not in spec["kw"]:
                res["invalid_kwargs"].append(f"{fn_name}: unknown kwarg '{kw.arg}'")

    res["calls_found"] = calls_seq
    for req in (task.get("required_calls") or []):
        if req not in calls_seq:
            res["missing_required_calls"].append(req)

    first_idx: Dict[str, int] = {}
    for idx, fn in enumerate(calls_seq):
        first_idx.setdefault(fn, idx)
    for pre, post in prereqs:
        if post in first_idx:
            if pre not in first_idx:
                res["order_violations"].append(f"{post} called without prior {pre}")
            elif first_idx[pre] > first_idx[post]:
                res["order_violations"].append(f"{pre} called AFTER {post}")

    res["missing_literals"] = verify_literals_consumed(code, task.get("expected_literals") or [])
    res["spec_compliant"] = (
        res["ast_ok"]
        and res["imports_sdk"]
        and not res["hallucinated_calls"]
        and not res["forbidden_calls_used"]
        and not res["invalid_kwargs"]
        and not res["missing_required_calls"]
        and not res["order_violations"]
        and not res["missing_literals"]
        and not any("undefined name" in p for p in res["pyflakes_issues"])
    )
    return res


def evaluate_task_response(raw_text: str, code: str, task: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Evaluates raw model output + extracted code for either sandbox or sdk tasks."""
    expected = task.get("expected", "synthesize")
    kind = task["kind"]

    if kind == "sdk":
        audit = audit_sdk_code(code, task)
        ok = bool(audit["spec_compliant"])
        return {
            "kind": "sdk",
            "expected": expected,
            "status": "PASSED" if ok else "SPEC_FAILED",
            "passed": ok,
            "acceptable": ok,
            "audit": audit,
            "detail": "SDK spec compliant" if ok else "; ".join(
                filter(
                    None,
                    [
                        f"missing={audit['missing_required_calls']}" if audit["missing_required_calls"] else "",
                        f"halluc={audit['hallucinated_calls']}" if audit["hallucinated_calls"] else "",
                        f"forbidden={audit['forbidden_calls_used']}" if audit["forbidden_calls_used"] else "",
                        f"order={audit['order_violations']}" if audit["order_violations"] else "",
                        f"kw={audit['invalid_kwargs']}" if audit["invalid_kwargs"] else "",
                        f"lits={audit['missing_literals']}" if audit["missing_literals"] else "",
                    ],
                )
            ) or "Not spec compliant",
        }

    # Sandbox task
    if expected == "clarify" and not code.strip():
        # Clean refusal / clarification request only if visible text remains after stripping reasoning traces
        visible_reply = strip_reasoning_traces(raw_text)
        has_question_or_clarify = bool(visible_reply and visible_reply.strip())
        return {
            "kind": "sandbox",
            "expected": "clarify",
            "status": "REFUSED_OK" if has_question_or_clarify else "NO_CODE",
            "passed": False,
            "acceptable": has_question_or_clarify,
            "detail": (
                "Clean clarification/refusal on underspecified prompt"
                if has_question_or_clarify
                else "Empty reply after stripping reasoning traces"
            ),
            "ast_ok": False,
            "missing_literals": [],
        }

    sb = run_sandbox_eval(code, task, timeout=timeout)
    passed = sb["status"] == "PASSED"
    acceptable = passed or (
        expected == "clarify"
        and sb["status"] in ("REFUSED_OK", "TYPED_HOLE_HALTED", "AMBIGUITY_HALTED", "UNRESOLVED_PORT")
    )
    return {
        "kind": "sandbox",
        "expected": expected,
        "status": sb["status"],
        "passed": passed,
        "acceptable": acceptable,
        "detail": sb["detail"],
        "ast_ok": sb["ast_ok"],
        "missing_literals": sb["missing_literals"],
        "stderr": sb.get("stderr", ""),
    }


def load_sdk_docs(sdk_name: str) -> str:
    return Path(SDK_SPECS[sdk_name]["docs_path"]).read_text(encoding="utf-8")
