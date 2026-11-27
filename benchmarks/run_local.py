#!/usr/bin/env python3
"""
benchmarks/run_local.py - Unified v2 local benchmark runner for STRATA and local GGUF baselines.

Supports apples-to-apples local benchmarking across six arms:
  - strata_v2             : Unified StrataPipeline (Profile C/D/E/O) over v2 domain trees
  - pure_gguf             : Local GGUF model, prompt-only (zero SKILL.md docs)
  - skill_md_gguf         : Local GGUF model with domain-scoped SKILL.md loaded in system prompt
  - skill_md_haystack_gguf: Local GGUF model with 15-domain catalog noise + target SDK docs
  - react_pure_gguf       : Zero-oracle multi-turn execution-feedback ReAct loop (pure)
  - react_skill_md_gguf   : Zero-oracle multi-turn execution-feedback ReAct loop (SKILL.md)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    build_react_feedback,
    evaluate_task_response,
    extract_code_clean,
    provision_sandbox_fixtures,
    read_cpu_times,
    read_gpu_vram_mb,
    read_proc_mem_mb,
    strip_reasoning_traces,
)
from benchmarks.skill_docs import (  # noqa: E402
    DOMAIN_SET_TREE_PATHS,
    build_messages_for_arm,
)
from benchmarks.suites import get_suite_tasks  # noqa: E402


class LlamaTelemetry:
    """Thread-safe call and token counter wrapping llama_cpp.Llama."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_wall_ms = 0.0
        self._installed = False

    def reset(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_wall_ms = 0.0

    def install(self) -> None:
        if self._installed:
            return
        from llama_cpp import Llama

        hook = self
        orig_chat = Llama.create_chat_completion
        orig_call = Llama.__call__

        def _rec(resp: Any, dt_ms: float) -> None:
            hook.calls += 1
            hook.llm_wall_ms += dt_ms
            if isinstance(resp, dict):
                usage = resp.get("usage") or {}
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                hook.prompt_tokens += pt
                hook.completion_tokens += ct
                hook.total_tokens += int(usage.get("total_tokens") or (pt + ct))

        @functools.wraps(orig_chat)
        def wrapped_chat(self_l, *args, **kwargs):
            t0 = time.perf_counter()
            r = orig_chat(self_l, *args, **kwargs)
            _rec(r, (time.perf_counter() - t0) * 1000.0)
            return r

        @functools.wraps(orig_call)
        def wrapped_call(self_l, *args, **kwargs):
            t0 = time.perf_counter()
            r = orig_call(self_l, *args, **kwargs)
            _rec(r, (time.perf_counter() - t0) * 1000.0)
            return r

        Llama.create_chat_completion = wrapped_chat
        Llama.__call__ = wrapped_call
        self._installed = True


def _infer_params_b(model_name: str, fallback: float = 0.8) -> float:
    low = str(model_name or "").lower()
    for tag, val in (
        ("0.5b", 0.5),
        ("0.8b", 0.8),
        ("1.5b", 1.5),
        ("3b", 3.0),
        ("7b", 7.6),
        ("9b", 9.0),
        ("14b", 14.0),
        ("20b", 20.0),
        ("27b", 27.0),
        ("30b", 30.5),
        ("31b", 30.7),
        ("32b", 32.0),
        ("35b", 35.0),
    ):
        if tag in low:
            return val
    return fallback


def _get_or_init_strata_pipeline(
    *,
    domain_set: str,
    spec: Any,
    mm: Any,
    reranker_obj: Any,
    sandbox: Any,
    compiled_default: Any,
    pipe_cache: Dict[str, Any],
    embed_vec_cache: Dict[Tuple[str, str], List[float]],
    timeout: float,
) -> Any:
    if domain_set in pipe_cache:
        return pipe_cache[domain_set]

    from profile_adapters import (
        build_strata_pipeline_for_profile,
        build_strata_pipeline_from_trees,
        make_embed_fn,
    )

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

    if domain_set == "default":
        pipe = build_strata_pipeline_for_profile(
            compiled_default,
            spec,
            model_manager=mm,
            reranker=reranker_obj,
            sandbox=sandbox,
            timeout=timeout,
            embed_fn_override=embed_override,
        )
    else:
        trees = [
            json.loads(tp.read_text(encoding="utf-8"))
            for tp in DOMAIN_SET_TREE_PATHS(ROOT_DIR, domain_set)
        ]
        pipe = build_strata_pipeline_from_trees(
            trees,
            spec,
            model_manager=mm,
            reranker=reranker_obj,
            sandbox=sandbox,
            timeout=timeout,
            embed_fn_override=embed_override,
        )
    if spec.uses_embedder:
        pipe.retriever._ensure_doc_embeddings()
    pipe_cache[domain_set] = pipe
    return pipe


def _run_strata_v2_arm(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    hook: LlamaTelemetry,
    out_dir: Path,
) -> Path:
    from config import resolve_profile_spec, settings
    from gevr_sandbox import GEVRSandbox
    from inference import ModelManager
    from lattice_db import ensure_lattice_compiled, load_compiled_lattice
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

    db_path = ROOT_DIR / "trees" / "lattice.db"
    ensure_lattice_compiled(trees_dir=str(ROOT_DIR / "trees"), db_path=str(db_path))
    compiled_default = load_compiled_lattice(db_path)
    sandbox = GEVRSandbox(
        default_timeout=float(args.timeout),
        asset_dirs=[ROOT_DIR / "benchmarks" / "fixtures" / "media"],
    )

    pipe_cache: Dict[str, Any] = {}
    embed_vec_cache: Dict[Tuple[str, str], List[float]] = {}
    records: List[Dict[str, Any]] = []

    try:
        for task in tasks:
            domain_set = str(task.get("domain_set") or ("cryopulse_only" if task.get("sdk") == "cryopulse" else "aetheris_only" if task.get("sdk") == "aetheris" else "default"))
            pipe = _get_or_init_strata_pipeline(
                domain_set=domain_set,
                spec=spec,
                mm=mm,
                reranker_obj=rr_obj,
                sandbox=sandbox,
                compiled_default=compiled_default,
                pipe_cache=pipe_cache,
                embed_vec_cache=embed_vec_cache,
                timeout=float(args.timeout),
            )
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()

            expected = str(task.get("expected") or "synthesize")
            should_exec = bool(task["kind"] == "sandbox" and expected == "synthesize")
            res = pipe.run(task["prompt"], execute=should_exec)
            code = res.code or ""

            if expected in ("clarify", "refuse"):
                if not res.ok:
                    ev = {
                        "status": "REFUSED_OK",
                        "passed": False,
                        "acceptable": True,
                        "detail": f"STRATA refused cleanly: {res.refusal_kind} ({res.error})",
                    }
                else:
                    ev = {
                        "status": "DOMAIN_VIOLATION" if expected == "refuse" else "UNEXPECTED_SYNTHESIS",
                        "passed": False,
                        "acceptable": False,
                        "detail": "STRATA synthesized code when refusal was required",
                    }
            elif not res.ok:
                ev = {
                    "status": res.refusal_kind or "FAILED",
                    "passed": False,
                    "acceptable": False,
                    "detail": str(res.error or res.refusal_kind or "Pipeline failed"),
                }
            else:
                ev = evaluate_task_response("", code, task, timeout=float(args.timeout))

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
                    "domain_set": domain_set,
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
                    "audit": ev.get("audit"),
                    "code": code,
                }
            )
            print(
                f"[STRATA-{spec.code}{rr_disp}] {task['id']:<30} -> {ev['status']:<18} "
                f"| acc={str(ev['acceptable']):<5} | {wall_ms:7.1f} ms | calls={hook.calls}"
            )
    finally:
        mm.cleanup(release_reranker=True)

    safe_llm = resolved_llm.replace("/", "_")
    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    out_path = out_dir / f"results_strata_v2_{spec.code}{rr_tag}_{safe_llm}{seed_tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
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


def _run_gguf_baseline_arm(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    hook: LlamaTelemetry,
    out_dir: Path,
) -> Path:
    from config import settings
    from hardware import HardwareProfiler
    from inference import ModelManager, select_optimal_llm

    max_turns = max(1, int(getattr(args, "max_turns", 3) or 3))
    arm_cfg = {
        "pure_gguf": ("pure", False, "Pure", "pure"),
        "skill_md_gguf": ("skill_md", False, "SKILL.md", "skill_md"),
        "skill_md_haystack_gguf": ("skill_md_haystack", False, "SKILL.md+Haystack", "skill_md_haystack"),
        "react_pure_gguf": ("pure", True, f"ReAct-Pure-{max_turns}T", "react_pure"),
        "react_skill_md_gguf": ("skill_md", True, f"ReAct-SKILL.md-{max_turns}T", "react_skill_md"),
    }[args.arm]
    context_mode, is_react, mode_tag, arm_tag = arm_cfg

    if args.n_ctx is not None:
        settings.llm_context_length = int(args.n_ctx)
    elif (context_mode in ("skill_md", "skill_md_haystack") or is_react) and settings.llm_context_length < 8192:
        settings.llm_context_length = 8192

    mm = ModelManager.get_instance()
    resolved_llm = select_optimal_llm(args.llm or "")
    params_b = _infer_params_b(resolved_llm, fallback=0.8)

    mem_before = read_proc_mem_mb()
    gpu_before = read_gpu_vram_mb()
    t_init0 = time.perf_counter()
    mm._load_llm(resolved_llm, HardwareProfiler.get_optimal_device(), logits_all=False)
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 2)
    gpu_after_init = read_gpu_vram_mb()

    records: List[Dict[str, Any]] = []
    display_name = f"{resolved_llm} [{mode_tag}]"

    try:
        for task in tasks:
            messages, ctx_meta = build_messages_for_arm(
                task,
                context_mode=context_mode,
                max_distractor_chars=int(args.haystack_chars),
                full_haystack=bool(args.full_haystack),
            )
            hook.reset()
            cpu0 = read_cpu_times()
            t0 = time.perf_counter()

            conv_messages = [dict(m) for m in messages]
            turns_limit = max_turns if is_react else 1
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
                for turn in range(1, turns_limit + 1):
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
                    ev = evaluate_task_response(raw_reply, code, task, timeout=float(args.timeout))
                    if not is_react or turn >= turns_limit:
                        break
                    fb = build_react_feedback(code, ev, task)
                    if fb is None:
                        break
                    clean_prev = strip_reasoning_traces(raw_reply)
                    assistant_turn = clean_prev or (f"```python\n{code}\n```" if code else "(empty)")
                    conv_messages.append({"role": "assistant", "content": assistant_turn})
                    conv_messages.append({"role": "user", "content": fb})
            except Exception as exc:
                raw_reply = ""
                code = ""
                ev = {
                    "status": "LLM_ERROR",
                    "passed": False,
                    "acceptable": False,
                    "detail": str(exc),
                }

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
                    "domain_set": ctx_meta["domain_set"],
                    "context_mode": context_mode,
                    "context_meta": ctx_meta,
                    "expected": task.get("expected", "synthesize"),
                    "seed": args.seed,
                    "react_turns": turns_used,
                    "max_turns": turns_limit,
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
                    "audit": ev.get("audit"),
                    "code": code,
                    "raw_reply_preview": raw_reply[:1200],
                }
            )
            print(
                f"[{mode_tag}:{resolved_llm}] {task['id']:<30} -> {ev['status']:<18} "
                f"| acc={str(ev['acceptable']):<5} | t={turns_used} | {wall_ms:7.1f} ms | tok={hook.prompt_tokens}+{hook.completion_tokens}"
            )
    finally:
        mm.cleanup(release_reranker=True)

    safe_llm = resolved_llm.replace("/", "_")
    seed_tag = f"_s{args.seed}" if args.seed is not None else ""
    out_path = out_dir / f"results_{args.arm}_{safe_llm}{seed_tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "strata_bench_v2",
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
                    "max_turns": max_turns if is_react else 1,
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
    ap = argparse.ArgumentParser(description="STRATA v2 Standardized Local Benchmark Runner")
    ap.add_argument(
        "--arm",
        choices=[
            "strata_v2",
            "pure_gguf",
            "skill_md_gguf",
            "skill_md_haystack_gguf",
            "react_pure_gguf",
            "react_skill_md_gguf",
        ],
        default="strata_v2",
        help="Synthesis / baseline arm to benchmark",
    )
    ap.add_argument("--suite", default="v2_expanded", help="Suite name from benchmarks/suites.py")
    ap.add_argument("--profile", choices=["C", "D", "E", "O"], default="O", help="STRATA profile for --arm strata_v2")
    ap.add_argument("--llm", default="", help="Override local GGUF model folder name")
    ap.add_argument("--embedder", default="", help="Override embedder model folder name")
    ap.add_argument("--reranker", dest="reranker", action="store_true", default=None, help="Force enable neural reranker")
    ap.add_argument("--no-reranker", dest="reranker", action="store_false", help="Force disable neural reranker")
    ap.add_argument("--haystack-chars", type=int, default=18000, help="Character budget for multi-domain SKILL.md catalog")
    ap.add_argument("--full-haystack", action="store_true", help="Include every cell across all loaded trees without truncation")
    ap.add_argument("--n-ctx", type=int, default=None, help="Override GGUF context length")
    ap.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens for GGUF baseline arms")
    ap.add_argument("--max-turns", type=int, default=3, help="Max zero-oracle repair turns for react_*_gguf arms")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for GGUF baseline arms")
    ap.add_argument("--timeout", type=float, default=45.0, help="Sandbox timeout seconds")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed forwarded to GGUF completions & recorded")
    ap.add_argument("--out-dir", default="test_logs/local_bench")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provision_sandbox_fixtures(ROOT_DIR / "benchmarks" / "fixtures" / "media")

    hook = LlamaTelemetry()
    hook.install()

    tasks = get_suite_tasks(args.suite)
    print(f"[LOCAL] Arm={args.arm} | Suite='{args.suite}' ({len(tasks)} tasks)")

    if args.arm == "strata_v2":
        _run_strata_v2_arm(args, tasks, hook, out_dir)
    else:
        _run_gguf_baseline_arm(args, tasks, hook, out_dir)


if __name__ == "__main__":
    main()
