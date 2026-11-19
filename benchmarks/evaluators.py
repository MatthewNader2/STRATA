"""
benchmarks/evaluators.py - Shared code extraction, headless sandbox runner,
static SDK typestate auditor, and resource telemetry utilities.
"""
from __future__ import annotations

import ast
import io
import os
import re
import resource
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .suites import SDK_SPECS

ROOT_DIR = Path(__file__).resolve().parent.parent


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


def strip_reasoning_traces(raw_text: str) -> str:
    """Removes <think>...</think> and harmony/channel reasoning wrappers before code extraction."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if "<|channel|>final<|message|>" in cleaned:
        cleaned = cleaned.split("<|channel|>final<|message|>")[-1]
    cleaned = cleaned.replace("<|end|>", "").replace("<|return|>", "")
    return cleaned.strip()


def extract_code_clean(raw_text: str) -> str:
    """Extracts the final executable Python code block from raw model output."""
    cleaned = strip_reasoning_traces(raw_text)
    if not cleaned:
        return ""

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        # Prefer the last non-empty block (models sometimes draft then output final code)
        for blk in reversed(blocks):
            if blk.strip():
                return blk.strip()

    # Fallback: unclosed fence
    m_open = re.search(r"```(?:python|py)?\s*\n(.*)$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m_open:
        return m_open.group(1).strip()

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
    """Checks that prompt-specified literals are present as Python AST constants/values."""
    if not expected_literals or not code.strip():
        return list(expected_literals) if not code.strip() else []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return list(expected_literals)

    consts: List[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            consts.append(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                consts.append(-node.operand.value)

    missing: List[Any] = []
    for lit in expected_literals:
        if isinstance(lit, float):
            if not any(isinstance(c, (int, float)) and not isinstance(c, bool) and abs(float(c) - lit) < 1e-6 for c in consts):
                missing.append(lit)
        elif isinstance(lit, str):
            if not any(isinstance(c, str) and (lit == c or lit in c) for c in consts):
                missing.append(lit)
        else:
            if lit not in consts:
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

    sandbox = GEVRSandbox(default_timeout=timeout, allowed_capabilities=set())
    sb = sandbox.execute(code, timeout=timeout)
    err_txt = str(sb.get("error") or sb.get("stderr") or "").strip()

    if not sb.get("success", False):
        if any(k in err_txt for k in ("SecurityPolicyError", "Security violation", "Capability denied")):
            st, cat = "SECURITY_BLOCKED", "SECURITY"
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
        # Clean refusal / clarification request without hallucinated code
        has_question_or_clarify = bool(raw_text and raw_text.strip())
        return {
            "kind": "sandbox",
            "expected": "clarify",
            "status": "REFUSED_OK" if has_question_or_clarify else "NO_CODE",
            "passed": False,
            "acceptable": has_question_or_clarify,
            "detail": "Clean clarification/refusal on underspecified prompt" if has_question_or_clarify else "Empty reply",
            "ast_ok": False,
            "missing_literals": [],
        }

    sb = run_sandbox_eval(code, task, timeout=timeout)
    passed = sb["status"] == "PASSED"
    acceptable = passed or (expected == "clarify" and sb["status"] in ("LINT_HALTED", "REFUSED_OK"))
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
