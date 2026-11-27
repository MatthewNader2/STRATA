#!/usr/bin/env python3
"""
benchmarks/external_bfcl.py - Standardized External Public Benchmark (BFCL v3).

Evaluates STRATA v2 (domain-neutral JSON schema -> TreeSchema compilation + typed
retrieval/planning/binding) against matched single-shot (`skill_md_gguf`) and
zero-oracle iterative (`react_skill_md_gguf`) GGUF baselines on a stratified N=120
subset of UC Berkeley's Berkeley Function Calling Leaderboard v3
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`: 40 simple, 40 multiple,
40 irrelevance).

Zero hardcoded tool or domain names: every BFCL function schema is compiled on the
fly into a generic STRATA v2 TreeSchema manifest.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (str(ROOT_DIR / "src"), str(ROOT_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from benchmarks.evaluators import (  # noqa: E402
    extract_code_clean,
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
    strip_reasoning_traces,
)
from benchmarks.run_local import LlamaTelemetry, _infer_params_b  # noqa: E402

BFCL_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

_JSON_TYPE_TO_CARRIER: Dict[str, str] = {
    "integer": "int",
    "int": "int",
    "float": "float",
    "number": "float",
    "string": "str",
    "str": "str",
    "boolean": "bool",
    "bool": "bool",
    "array": "list",
    "list": "list",
    "dict": "dict",
    "object": "dict",
    "tuple": "tuple",
}


def _extract_prompt_text(raw_question: Any) -> str:
    if isinstance(raw_question, str):
        return raw_question.strip()
    if isinstance(raw_question, list):
        for item in raw_question:
            if isinstance(item, list):
                txt = _extract_prompt_text(item)
                if txt:
                    return txt
            elif isinstance(item, dict) and str(item.get("role") or "") == "user":
                content = str(item.get("content") or "").strip()
                if content:
                    return content
    return ""


def _even_indices(total: int, count: int) -> List[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    chosen: List[int] = []
    seen: Set[int] = set()
    for i in range(count):
        idx = min(total - 1, int(i * total / count))
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    return chosen


def load_bfcl_v3_tasks(
    *,
    per_split: int = 40,
    splits: Sequence[str] = ("simple", "multiple", "irrelevance"),
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Download/cache official BFCL v3 splits from HuggingFace and sample `per_split` per split."""
    from huggingface_hub import hf_hub_download

    dl_kwargs: Dict[str, Any] = {"repo_id": BFCL_HF_REPO, "repo_type": "dataset"}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dl_kwargs["cache_dir"] = str(cache_dir)

    tasks: List[Dict[str, Any]] = []
    for split in splits:
        q_file = f"BFCL_v3_{split}.json"
        q_path = Path(hf_hub_download(filename=q_file, **dl_kwargs))
        q_rows = [json.loads(ln) for ln in q_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        gt_by_id: Dict[str, Any] = {}
        if split != "irrelevance":
            gt_file = f"possible_answer/BFCL_v3_{split}.json"
            gt_path = Path(hf_hub_download(filename=gt_file, **dl_kwargs))
            for ln in gt_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    gt_by_id[str(rec["id"])] = rec.get("ground_truth") or []

        for idx in _even_indices(len(q_rows), per_split):
            row = q_rows[idx]
            tid = str(row["id"])
            is_irrel = split == "irrelevance"
            tasks.append(
                {
                    "id": tid,
                    "group": f"bfcl_{split}",
                    "split": split,
                    "kind": "bfcl_refusal" if is_irrel else "bfcl_call",
                    "expected": "refuse" if is_irrel else "synthesize",
                    "prompt": _extract_prompt_text(row.get("question")),
                    "functions": list(row.get("function") or ()),
                    "ground_truth": [] if is_irrel else list(gt_by_id.get(tid) or ()),
                }
            )
    return tasks


def bfcl_functions_to_strata_tree(
    functions: Sequence[Dict[str, Any]],
    *,
    domain_name: str = "bfcl_tools",
) -> Dict[str, Any]:
    """
    Domain-neutrally compile arbitrary BFCL JSON tool schemas into a valid STRATA v2 TreeSchema dict.
    Zero hardcoded function names.
    """
    cells: List[Dict[str, Any]] = []
    for fn in functions:
        fn_name = str(fn.get("name") or "").strip()
        if not fn_name:
            continue
        desc = str(fn.get("description") or "").strip()
        params_obj = fn.get("parameters") or {}
        raw_props = params_obj.get("properties") or {}
        req_set = {str(r) for r in (params_obj.get("required") or ())}

        prop_names = [str(pn) for pn in raw_props.keys()]
        out_port = "result"
        while out_port in prop_names:
            out_port = f"{out_port}_out" if out_port != "result" else "out_val"

        inputs_spec: Dict[str, Dict[str, Any]] = {}
        for pname, pspec in raw_props.items():
            p_str = str(pname)
            ps = pspec if isinstance(pspec, dict) else {}
            raw_type = str(ps.get("type") or "any").strip().lower()
            carrier = _JSON_TYPE_TO_CARRIER.get(raw_type, "any")
            raw_enum = ps.get("enum")
            is_enum = isinstance(raw_enum, list) and len(raw_enum) > 0
            has_default = "default" in ps and ps.get("default") is not None
            inputs_spec[p_str] = {
                "carrier": carrier,
                "typestate": "any",
                "value_class": "enum" if is_enum else "literal",
                "resource_mode": "value",
                "required": (p_str in req_set) and not has_default,
                "default_value": ps.get("default") if has_default else None,
                "enum_values": list(raw_enum) if is_enum else None,
                "description": str(ps.get("description") or ""),
            }

        args_tmpl = ", ".join(f"{pn}={{{pn}}}" for pn in prop_names)
        code_tmpl = f"{{{out_port}}} = {fn_name}({args_tmpl})"
        cells.append(
            {
                "cell_id": fn_name,
                "domain_name": domain_name,
                "docstring": desc or f"Invoke {fn_name}.",
                "code_template": code_tmpl,
                "dependencies": [],
                "inputs": inputs_spec,
                "outputs": {
                    out_port: {
                        "carrier": "any",
                        "typestate": f"{domain_name}::call_result",
                        "value_class": "data",
                        "resource_mode": "value",
                        "required": True,
                    }
                },
                "primary_out": out_port,
                "mutation_type": "pure",
            }
        )

    return {
        "domain": domain_name,
        "version": "2.0.0",
        "description": "Dynamically compiled external function-calling domain manifest.",
        "is_core": False,
        "aliases": [domain_name],
        "imports": ["python_core"],
        "types": {},
        "cells": cells,
    }


def _render_bfcl_tools_markdown(functions: Sequence[Dict[str, Any]]) -> str:
    sections: List[str] = [f"## Available Functions ({len(functions)} tools)"]
    for fn in functions:
        fn_name = str(fn.get("name") or "").strip()
        desc = str(fn.get("description") or "").strip()
        params_obj = fn.get("parameters") or {}
        raw_props = params_obj.get("properties") or {}
        req_set = {str(r) for r in (params_obj.get("required") or ())}
        sig_parts: List[str] = []
        param_lines: List[str] = []
        for pname, pspec in raw_props.items():
            ps = pspec if isinstance(pspec, dict) else {}
            ptype = str(ps.get("type") or "any")
            req_tag = "required" if str(pname) in req_set else "optional"
            extra_bits = [req_tag]
            if "default" in ps and ps["default"] is not None:
                extra_bits.append(f"default={ps['default']!r}")
            if isinstance(ps.get("enum"), list) and ps["enum"]:
                extra_bits.append(f"enum={ps['enum']}")
            sig_parts.append(f"{pname}: {ptype}")
            pdesc = str(ps.get("description") or "").strip()
            param_lines.append(f"  - `{pname}` ({ptype}, {', '.join(extra_bits)}): {pdesc}")
        sections.append(f"### `{fn_name}({', '.join(sig_parts)})`")
        if desc:
            sections.append(f"- Summary: {desc}")
        sections.extend(param_lines)
    return "\n".join(sections)


def build_bfcl_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    tools_md = _render_bfcl_tools_markdown(task.get("functions") or ())
    sys_msg = (
        "You are a precise Python function-calling assistant.\n"
        "Rules:\n"
        "1. Emit ONLY a single ```python ... ``` code block containing valid Python calling the single matching function "
        "from the Available Functions below with exact arguments.\n"
        "2. Preserve every literal number, string, unit, boolean, array, and dictionary from the user's request.\n"
        "3. Do NOT invent or call functions outside the Available Functions below.\n"
        "4. If NONE of the Available Functions below matches the user's request (unrelated domain or missing required capability), "
        "refuse cleanly in plain text WITHOUT emitting any code block.\n\n"
        f"{tools_md}"
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": str(task.get("prompt") or "")},
    ]


def _dotted_call_name(node: ast.AST, alias_map: Dict[str, str]) -> Optional[str]:
    if isinstance(node, ast.Name):
        return alias_map.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted_call_name(node.value, alias_map)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _eval_ast_value(node: ast.AST, const_env: Dict[str, Any]) -> Tuple[bool, Any]:
    if isinstance(node, ast.Name):
        if node.id in const_env:
            return True, const_env[node.id]
        return False, None
    if isinstance(node, ast.List):
        vals: List[Any] = []
        for el in node.elts:
            ok, v = _eval_ast_value(el, const_env)
            if not ok:
                return False, None
            vals.append(v)
        return True, vals
    if isinstance(node, ast.Tuple):
        vals_t: List[Any] = []
        for el in node.elts:
            ok, v = _eval_ast_value(el, const_env)
            if not ok:
                return False, None
            vals_t.append(v)
        return True, tuple(vals_t)
    if isinstance(node, ast.Dict):
        out_d: Dict[Any, Any] = {}
        for k_n, v_n in zip(node.keys, node.values):
            if k_n is None:
                return False, None
            ok_k, k_v = _eval_ast_value(k_n, const_env)
            ok_v, v_v = _eval_ast_value(v_n, const_env)
            if not ok_k or not ok_v:
                return False, None
            out_d[k_v] = v_v
        return True, out_d
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is bool and type(expected) is bool and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-6)
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(_values_match(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_values_match(actual[k], expected[k]) for k in expected)
    return actual == expected


def _match_single_ground_truth_entry(
    gt_entry: Dict[str, Dict[str, List[Any]]],
    calls: List[Tuple[str, ast.Call]],
    fn_schemas: Dict[str, Dict[str, Any]],
    const_env: Dict[str, Any],
) -> Tuple[bool, str]:
    if len(gt_entry) != 1:
        return False, "Unsupported multi-entry BFCL ground_truth dict"
    exp_fn, param_spec_map = next(iter(gt_entry.items()))
    exp_suffix = exp_fn.rsplit(".", 1)[-1]

    schema = fn_schemas.get(exp_fn) or {}
    props = (schema.get("parameters") or {}).get("properties") or {}
    prop_order = [str(k) for k in props.keys()]
    allowed_kwargs = set(prop_order)

    schema_tool_names = set(fn_schemas.keys())
    other_tool_calls = [
        cn for cn, _ in calls
        if cn in schema_tool_names and cn != exp_fn
    ]
    if other_tool_calls:
        return False, f"Extraneous schema tool call(s): {other_tool_calls}"

    matching_calls = [
        call_node for cname, call_node in calls
        if cname == exp_fn or ("." in exp_fn and cname == exp_suffix)
    ]
    if not matching_calls:
        found = [cn for cn, _ in calls]
        return False, f"Expected call '{exp_fn}', found {found}"

    last_reason = "Call mismatch"
    for call_node in matching_calls:
        if any(isinstance(a, ast.Starred) for a in call_node.args):
            last_reason = "Unsupported *args unpacking"
            continue
        bound: Dict[str, Any] = {}
        ok_bind = True
        if len(call_node.args) > len(prop_order):
            last_reason = f"Too many positional args for '{exp_fn}'"
            continue
        for pos_idx, arg_node in enumerate(call_node.args):
            ok_v, val = _eval_ast_value(arg_node, const_env)
            if not ok_v:
                ok_bind = False
                last_reason = f"Non-literal positional arg at index {pos_idx}"
                break
            bound[prop_order[pos_idx]] = val
        if not ok_bind:
            continue

        for kw in call_node.keywords:
            if kw.arg is None:
                ok_bind = False
                last_reason = "Unsupported **kwargs unpacking"
                break
            if allowed_kwargs and kw.arg not in allowed_kwargs:
                ok_bind = False
                last_reason = f"Unknown keyword arg '{kw.arg}' for '{exp_fn}'"
                break
            if kw.arg in bound:
                ok_bind = False
                last_reason = f"Duplicate argument '{kw.arg}'"
                break
            ok_v, val = _eval_ast_value(kw.value, const_env)
            if not ok_v:
                ok_bind = False
                last_reason = f"Non-literal value for kwarg '{kw.arg}'"
                break
            bound[kw.arg] = val
        if not ok_bind:
            continue

        all_params_ok = True
        for pname, accepted_vals in (param_spec_map or {}).items():
            accepts_omit = ("" in accepted_vals) or (None in accepted_vals)
            if pname not in bound or bound[pname] is None:
                if not accepts_omit:
                    all_params_ok = False
                    last_reason = f"Missing required parameter '{pname}' on '{exp_fn}'"
                    break
                continue
            actual_val = bound[pname]
            non_empty_cands = [v for v in accepted_vals if v != "" and v is not None]
            if not non_empty_cands:
                # GT only allowed omitted/empty
                all_params_ok = False
                last_reason = f"Parameter '{pname}' had unexpected value {actual_val!r}"
                break
            if not any(_values_match(actual_val, cand) for cand in non_empty_cands):
                all_params_ok = False
                last_reason = (
                    f"Parameter '{pname}' mismatch on '{exp_fn}': "
                    f"got {actual_val!r}, expected one of {non_empty_cands[:3]}"
                )
                break
        if all_params_ok:
            return True, f"Matched BFCL call '{exp_fn}'"

    return False, last_reason


def evaluate_bfcl_response(raw_text: str, code: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic AST / irrelevance evaluator for BFCL v3 tasks."""
    expected = str(task.get("expected") or "synthesize")
    kind = str(task.get("kind") or "bfcl_call")

    if expected == "refuse" or kind == "bfcl_refusal":
        if not str(code or "").strip():
            visible = strip_reasoning_traces(raw_text)
            ok_ref = bool(visible and visible.strip())
            return {
                "kind": kind,
                "expected": expected,
                "status": "REFUSED_OK" if ok_ref else "NO_CODE",
                "passed": False,
                "acceptable": ok_ref,
                "ast_ok": False,
                "pyflakes_issues": [],
                "detail": "Cleanly refused irrelevant/unmappable function request" if ok_ref else "Empty refusal",
            }
        return {
            "kind": kind,
            "expected": expected,
            "status": "IRRELEVANCE_VIOLATION",
            "passed": False,
            "acceptable": False,
            "ast_ok": True,
            "pyflakes_issues": [],
            "detail": "Invoked function on BFCL irrelevance task requiring refusal",
        }

    if not str(code or "").strip():
        return {
            "kind": kind,
            "expected": expected,
            "status": "NO_CODE",
            "passed": False,
            "acceptable": False,
            "ast_ok": False,
            "pyflakes_issues": [],
            "detail": "No executable Python code emitted",
        }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "kind": kind,
            "expected": expected,
            "status": "SYNTAX_ERROR",
            "passed": False,
            "acceptable": False,
            "ast_ok": False,
            "pyflakes_issues": [f"SyntaxError: {exc.msg} (line {exc.lineno})"],
            "detail": f"SyntaxError: {exc.msg}",
        }

    pyflakes_issues: List[str] = []
    try:
        from pyflakes import api as pf_api
        from pyflakes import reporter as pf_rep

        out_io, err_io = io.StringIO(), io.StringIO()
        pf_api.check(code, "<bfcl>", reporter=pf_rep.Reporter(out_io, err_io))
        for ln in (out_io.getvalue() + "\n" + err_io.getvalue()).splitlines():
            if ln.strip():
                pyflakes_issues.append(ln.strip())
    except Exception:
        pass

    alias_map: Dict[str, str] = {}
    const_env: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for al in node.names:
                alias_map[al.asname or al.name.split(".", 1)[0]] = al.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for al in node.names:
                alias_map[al.asname or al.name] = f"{node.module}.{al.name}"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            ok_v, val = _eval_ast_value(node.value, const_env)
            if ok_v:
                const_env[node.targets[0].id] = val

    calls: List[Tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            cname = _dotted_call_name(node.func, alias_map)
            if cname:
                calls.append((cname, node))

    fn_schemas = {
        str(fn.get("name") or ""): fn
        for fn in (task.get("functions") or ())
        if isinstance(fn, dict) and fn.get("name")
    }
    gt_list = list(task.get("ground_truth") or ())
    if not gt_list:
        return {
            "kind": kind,
            "expected": expected,
            "status": "AST_FAILED",
            "passed": False,
            "acceptable": False,
            "ast_ok": True,
            "pyflakes_issues": pyflakes_issues,
            "detail": "Empty BFCL ground_truth",
        }

    reasons: List[str] = []
    for gt_entry in gt_list:
        if isinstance(gt_entry, dict):
            ok_gt, msg = _match_single_ground_truth_entry(gt_entry, calls, fn_schemas, const_env)
            if ok_gt:
                return {
                    "kind": kind,
                    "expected": expected,
                    "status": "PASSED",
                    "passed": True,
                    "acceptable": True,
                    "ast_ok": True,
                    "pyflakes_issues": pyflakes_issues,
                    "detail": msg,
                }
            reasons.append(msg)

    return {
        "kind": kind,
        "expected": expected,
        "status": "AST_FAILED",
        "passed": False,
        "acceptable": False,
        "ast_ok": True,
        "pyflakes_issues": pyflakes_issues,
        "detail": reasons[0] if reasons else "BFCL AST mismatch",
    }


def _build_bfcl_react_feedback(code: str, ev: Dict[str, Any], task: Dict[str, Any]) -> Optional[str]:
    """Zero-oracle syntax/format compiler feedback for BFCL ReAct baselines."""
    if str(task.get("expected") or "synthesize") != "synthesize":
        return None
    if not str(code or "").strip():
        return (
            "No valid ```python ... ``` code block was found in your reply. "
            "Emit ONLY a single valid ```python ... ``` block calling the matching function."
        )
    if not ev.get("ast_ok", False):
        return (
            f"Python syntax error in generated code: {ev.get('detail')}. "
            "Fix the syntax and reply with ONLY a single ```python ... ``` code block."
        )
    return None


def _run_bfcl_strata(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    hook: LlamaTelemetry,
    out_dir: Path,
) -> Path:
    from config import resolve_profile_spec, settings
    from inference import ModelManager
    from profile_adapters import build_strata_pipeline_from_trees, make_embed_fn
    from reranker import LocalReranker

    if args.n_ctx is not None:
        settings.llm_context_length = int(args.n_ctx)
    if args.reranker is not None:
        settings.use_reranker = bool(args.reranker)
    if args.seed is not None:
        settings.llm_seed = int(args.seed)

    spec = resolve_profile_spec(args.profile)
    use_reranker = bool(spec.uses_reranker and settings.use_reranker)
    rr_tag = ("_rr" if use_reranker else "_norr") if spec.uses_reranker else ""
    rr_disp = (" [RR]" if use_reranker else " [No-RR]") if spec.uses_reranker else ""

    mm = ModelManager.get_instance()
    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    t_init0 = time.perf_counter()
    mm.initialize_profile(spec.code, embedder_name=args.embedder or "", llm_name=args.llm or "")
    rr_obj = LocalReranker.get_shared(device="auto") if use_reranker else None
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
    gpu_after_init = read_gpu_vram_mb()

    resolved_llm = mm.llm_name or args.llm or "unknown"
    params_b = _infer_params_b(resolved_llm, fallback=0.8)
    arm_id = f"strata_v2_{spec.code}{rr_tag}"
    model_id = f"strata_v2_{spec.code.lower()}{rr_tag}_{resolved_llm}"
    model_display = f"STRATA v2 Profile {spec.code}{rr_disp} ({resolved_llm})"

    python_core_raw = json.loads((ROOT_DIR / "trees" / "python_core.json").read_text(encoding="utf-8"))
    python_core_base = {**python_core_raw, "cells": []}
    embed_vec_cache: Dict[Tuple[str, str], List[float]] = {}
    embed_override = None
    if spec.uses_embedder:
        raw_embed = make_embed_fn(mm)
        if raw_embed is not None:
            def _cached_embed(texts: List[str], *, mode: str = "document") -> List[List[float]]:
                missing = [t for t in dict.fromkeys(texts) if (mode, t) not in embed_vec_cache]
                if missing:
                    vecs = raw_embed(missing, mode=mode)
                    for txt, vec in zip(missing, vecs):
                        embed_vec_cache[(mode, txt)] = list(vec)
                return [embed_vec_cache[(mode, txt)] for txt in texts]

            embed_override = _cached_embed

    records: List[Dict[str, Any]] = []
    try:
        for idx, task in enumerate(tasks, 1):
            bfcl_tree = bfcl_functions_to_strata_tree(task.get("functions") or ())
            pipe = build_strata_pipeline_from_trees(
                [python_core_base, bfcl_tree],
                spec,
                model_manager=mm,
                reranker=rr_obj,
                sandbox=None,
                embed_fn_override=embed_override,
            )
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()
            expected = str(task.get("expected") or "synthesize")

            res = pipe.run(str(task.get("prompt") or ""), execute=False, top_k=max(4, len(bfcl_tree["cells"])))
            code = res.code or ""

            if expected == "refuse":
                if not res.ok:
                    ev = {
                        "status": "REFUSED_OK",
                        "passed": False,
                        "acceptable": True,
                        "detail": f"STRATA refused cleanly: {res.refusal_kind} ({res.error})",
                    }
                else:
                    ev = evaluate_bfcl_response("", code, task)
            elif not res.ok:
                ev = {
                    "status": res.refusal_kind or "FAILED",
                    "passed": False,
                    "acceptable": False,
                    "detail": str(res.error or res.refusal_kind or "Pipeline failed"),
                }
            else:
                ev = evaluate_bfcl_response("", code, task)

            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            records.append(
                {
                    "arm": arm_id,
                    "model_id": model_id,
                    "model_display": model_display,
                    "family": "STRATA-v2",
                    "params_b": params_b,
                    "active_params_b": params_b,
                    "quant": "Q4_K_M",
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "expected": expected,
                    "seed": args.seed,
                    "prompt": task["prompt"],
                    "status": ev["status"],
                    "passed": bool(ev["passed"]),
                    "acceptable": bool(ev["acceptable"]),
                    "detail": ev.get("detail", ""),
                    "wall_ms": wall_ms,
                    "llm_calls": hook.calls,
                    "prompt_tokens": hook.prompt_tokens,
                    "completion_tokens": hook.completion_tokens,
                    "total_tokens": hook.total_tokens,
                    "llm_wall_ms": round(hook.llm_wall_ms, 2),
                    "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                    "mem_mb": read_proc_mem_mb(),
                    "gpu_mb": read_gpu_vram_mb(),
                    "code": code,
                }
            )
            if idx % 20 == 0 or idx == len(tasks) or not ev["acceptable"]:
                print(
                    f"[BFCL:STRATA-{spec.code}{rr_disp}] [{idx:03d}/{len(tasks):03d}] "
                    f"{task['id']:<18} -> {ev['status']:<20} | acc={ev['acceptable']} | {wall_ms:6.1f} ms"
                )
    finally:
        mm.cleanup(release_reranker=True)

    safe_llm = resolved_llm.replace("/", "_")
    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    out_path = out_dir / f"results_bfcl_strata_v2_{spec.code}{rr_tag}_{safe_llm}{seed_tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "benchmark": "BFCL_v3",
                "arm": arm_id,
                "model_id": model_id,
                "model_display": model_display,
                "family": "STRATA-v2",
                "params_b": params_b,
                "active_params_b": params_b,
                "quant": "Q4_K_M",
                "seed": args.seed,
                "init": {
                    "profile": spec.code,
                    "reranker": use_reranker,
                    "init_ms": init_ms,
                    "mem_before": mem_before,
                    "gpu_before": gpu_before,
                    "gpu_after_init": gpu_after_init,
                },
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_path}")
    return out_path


def _run_bfcl_gguf(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    hook: LlamaTelemetry,
    out_dir: Path,
) -> Path:
    from config import settings
    from hardware import HardwareProfiler
    from inference import ModelManager, select_optimal_llm

    is_react = args.arm == "react_skill_md_gguf"
    max_turns = max(1, int(args.max_turns or 3)) if is_react else 1
    arm_tag = "react_skill_md" if is_react else "skill_md"
    mode_tag = f"ReAct-SKILL.md-{max_turns}T" if is_react else "SKILL.md"

    if args.n_ctx is not None:
        settings.llm_context_length = int(args.n_ctx)
    elif settings.llm_context_length < 4096:
        settings.llm_context_length = 4096

    mm = ModelManager.get_instance()
    resolved_llm = select_optimal_llm(args.llm or "")
    params_b = _infer_params_b(resolved_llm, fallback=0.8)

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    t_init0 = time.perf_counter()
    mm._load_llm(resolved_llm, HardwareProfiler.get_optimal_device(), logits_all=False)
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
    gpu_after_init = read_gpu_vram_mb()

    display_name = f"{resolved_llm} [{mode_tag}]"
    records: List[Dict[str, Any]] = []

    try:
        for idx, task in enumerate(tasks, 1):
            conv_messages = build_bfcl_messages(task)
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()

            turns_used = 0
            raw_reply = ""
            code = ""
            ev: Dict[str, Any] = {
                "status": "LLM_ERROR",
                "passed": False,
                "acceptable": False,
                "detail": "Not executed",
            }
            try:
                for turn in range(1, max_turns + 1):
                    turns_used = turn
                    chat_kw: Dict[str, Any] = {
                        "messages": conv_messages,
                        "max_tokens": int(args.max_tokens),
                        "temperature": float(args.temperature),
                        "top_p": 0.95,
                    }
                    if args.seed is not None:
                        chat_kw["seed"] = int(args.seed)
                    resp = mm.llm.create_chat_completion(**chat_kw)
                    msg = (resp.get("choices") or [{}])[0].get("message") or {}
                    raw_reply = str(msg.get("content") or "").strip()
                    code = extract_code_clean(raw_reply)
                    ev = evaluate_bfcl_response(raw_reply, code, task)
                    if not is_react or turn >= max_turns:
                        break
                    fb = _build_bfcl_react_feedback(code, ev, task)
                    if fb is None:
                        break
                    clean_prev = strip_reasoning_traces(raw_reply)
                    assistant_turn = clean_prev or (f"```python\n{code}\n```" if code else "(empty)")
                    conv_messages.append({"role": "assistant", "content": assistant_turn})
                    conv_messages.append({"role": "user", "content": fb})
            except Exception as exc:
                raw_reply = ""
                code = ""
                ev = {"status": "LLM_ERROR", "passed": False, "acceptable": False, "detail": str(exc)}

            wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            cpu1 = read_cpu_times()
            records.append(
                {
                    "arm": arm_tag,
                    "model_id": f"{resolved_llm}_{arm_tag}",
                    "model_display": display_name,
                    "family": "Local-GGUF",
                    "params_b": params_b,
                    "active_params_b": params_b,
                    "quant": "Q4_K_M",
                    "task_id": task["id"],
                    "group": task["group"],
                    "kind": task["kind"],
                    "expected": task.get("expected", "synthesize"),
                    "seed": args.seed,
                    "react_turns": turns_used,
                    "max_turns": max_turns,
                    "prompt": task["prompt"],
                    "status": ev["status"],
                    "passed": bool(ev["passed"]),
                    "acceptable": bool(ev["acceptable"]),
                    "detail": ev.get("detail", ""),
                    "wall_ms": wall_ms,
                    "llm_calls": hook.calls,
                    "prompt_tokens": hook.prompt_tokens,
                    "completion_tokens": hook.completion_tokens,
                    "total_tokens": hook.total_tokens,
                    "llm_wall_ms": round(hook.llm_wall_ms, 2),
                    "cpu_delta": {k: round(cpu1[k] - cpu0[k], 4) for k in cpu0},
                    "mem_mb": read_proc_mem_mb(),
                    "gpu_mb": read_gpu_vram_mb(),
                    "code": code,
                    "raw_reply_preview": raw_reply[:1200],
                }
            )
            if idx % 20 == 0 or idx == len(tasks) or not ev["acceptable"]:
                print(
                    f"[BFCL:{mode_tag}:{resolved_llm}] [{idx:03d}/{len(tasks):03d}] "
                    f"{task['id']:<18} -> {ev['status']:<20} | acc={ev['acceptable']} | t={turns_used} | {wall_ms:6.1f} ms"
                )
    finally:
        mm.cleanup(release_reranker=True)

    safe_llm = resolved_llm.replace("/", "_")
    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    out_path = out_dir / f"results_bfcl_{args.arm}_{safe_llm}{seed_tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
                "benchmark": "BFCL_v3",
                "arm": arm_tag,
                "model_id": f"{resolved_llm}_{arm_tag}",
                "model_display": display_name,
                "family": "Local-GGUF",
                "params_b": params_b,
                "active_params_b": params_b,
                "quant": "Q4_K_M",
                "seed": args.seed,
                "init": {
                    "init_ms": init_ms,
                    "n_ctx": settings.llm_context_length,
                    "max_turns": max_turns,
                    "temperature": float(args.temperature),
                    "mem_before": mem_before,
                    "gpu_before": gpu_before,
                    "gpu_after_init": gpu_after_init,
                },
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="STRATA v2 Standardized External Benchmark Runner (BFCL v3)")
    ap.add_argument(
        "--arm",
        choices=["strata_v2", "skill_md_gguf", "react_skill_md_gguf"],
        default="strata_v2",
    )
    ap.add_argument("--profile", choices=["C", "D", "E", "O"], default="O")
    ap.add_argument("--llm", default="")
    ap.add_argument("--embedder", default="")
    ap.add_argument("--reranker", dest="reranker", action="store_true", default=None)
    ap.add_argument("--no-reranker", dest="reranker", action="store_false")
    ap.add_argument("--per-split", type=int, default=40, help="Tasks sampled evenly per BFCL split (40 -> N=120 total)")
    ap.add_argument("--splits", nargs="+", default=["simple", "multiple", "irrelevance"])
    ap.add_argument("--n-ctx", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache-dir", default="test_logs/bfcl_cache")
    ap.add_argument("--out-dir", default="test_logs/wave4_bfcl_v3")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_bfcl_v3_tasks(
        per_split=int(args.per_split),
        splits=tuple(args.splits),
        cache_dir=Path(args.cache_dir),
    )
    print(f"[BFCL-v3] Arm={args.arm} | Tasks={len(tasks)} ({args.per_split} x {args.splits})")

    hook = LlamaTelemetry()
    hook.install()

    if args.arm == "strata_v2":
        _run_bfcl_strata(args, tasks, hook, out_dir)
    else:
        _run_bfcl_gguf(args, tasks, hook, out_dir)


if __name__ == "__main__":
    main()
