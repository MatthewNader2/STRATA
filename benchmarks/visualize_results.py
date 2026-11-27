#!/usr/bin/env python3
"""
benchmarks/visualize_results.py - Aggregates STRATA local, Colab, and hosted API
benchmark JSONs into Markdown/CSV reports and academic-style Matplotlib figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "bg": "#FFFFFF",
        "ink": "#1F2937",
        "muted": "#6B7280",
        "light_ink": "#9CA3AF",
        "grid": "#EAECEF",
        "spine": "#D1D5DB",
        "strata": "#C2592A",
        "strata_dark": "#9A3F18",
        "strata_border": "#C2592A",
        "ctx": "#4C72B0",
        "ctx_dark": "#2F4D7E",
        "pure": "#8C96A5",
        "pure_light": "#A8B1BE",
        "pure_bar": "#CBD2DC",
        "syn": "#4E8A78",
        "safety": "#5B84B1",
        "fallback": "#CFA052",
        "fail": "#C45C5C",
        "na": "#ECEFF2",
        "vram": "#5C6FA3",
        "latency": "#4A7C7E",
        "think": "#3B6998",
        "nothink": "#5C9A82",
        "instruct": "#7A68A6",
        "range_bg": "#E2E6EC",
        "seq_cmap": ["#F3F5F8", "#D6E4F0", "#9BC0DC", "#558BB8", "#235382"],
        "seq_threshold": 55.0,
    },
    "dark": {
        "bg": "#0D1117",
        "ink": "#E6EDF3",
        "muted": "#8B949E",
        "light_ink": "#6E7681",
        "grid": "#21262D",
        "spine": "#30363D",
        "strata": "#E07A4B",
        "strata_dark": "#F09A72",
        "strata_border": "#E07A4B",
        "ctx": "#6B9AD6",
        "ctx_dark": "#A5C4EB",
        "pure": "#8B949E",
        "pure_light": "#6E7681",
        "pure_bar": "#484F58",
        "syn": "#5FA68E",
        "safety": "#6B9AC8",
        "fallback": "#D6AC60",
        "fail": "#CE6B6B",
        "na": "#21262D",
        "vram": "#7B8FD1",
        "latency": "#5CA4A6",
        "think": "#6B9AD6",
        "nothink": "#5FA68E",
        "instruct": "#A58CD1",
        "range_bg": "#262C36",
        "seq_cmap": ["#161B22", "#1E3148", "#2B527A", "#4A82B5", "#86BBE8"],
        "seq_threshold": 60.0,
    },
}

API_ERROR_STATUSES = {"API_ERROR", "HTTP_ERROR", "ERROR"}

GROUP_SYN_ORDER = {
    "v2_nl_core": 0,
    "v2_domain_spec": 1,
    "v2_unseen": 2,
    "v2_haystack": 3,
    "v2_wrong_domain": 4,
}

KNOWN_METADATA: Dict[str, Dict[str, Any]] = {
    "omega_no_fb": {"display": "STRATA O+OMEGA (0.8B, No-FB)", "params_b": 0.8, "active_b": 0.8},
    "omega_with_fb": {"display": "STRATA O+OMEGA (0.8B, With-FB)", "params_b": 0.8, "active_b": 0.8},
    "omega": {"display": "STRATA O+OMEGA (0.8B, No-FB)", "params_b": 0.8, "active_b": 0.8},
    "pure_08b": {"display": "Pure Qwen3.5-0.8B", "params_b": 0.8, "active_b": 0.8},
    "pure_7b": {"display": "Pure Qwen2.5-Coder-7B", "params_b": 7.6, "active_b": 7.6},
}


def wilson_ci_95(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Return asymmetric 95% Wilson score confidence interval (lo_pct, hi_pct) in [0.0, 100.0]."""
    if n <= 0:
        return 0.0, 0.0
    p = min(max(float(k) / float(n), 0.0), 1.0)
    z2 = z * z
    n_f = float(n)
    denom = 1.0 + z2 / n_f
    center = (p + z2 / (2.0 * n_f)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n_f)) / n_f)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return round(lo, 1), round(hi, 1)


def _normalize_legacy_record(r: Dict[str, Any], top_data: Dict[str, Any]) -> Dict[str, Any]:
    is_v2 = top_data.get("schema_version") == "strata_bench_v2"
    arm = str(r.get("arm") or top_data.get("arm") or "unknown")
    tid = str(r.get("task_id") or r.get("preset") or "unknown")
    group = str(r.get("group") or "other")
    model_id = str(r.get("model_id") or top_data.get("model_id") or arm)
    meta = KNOWN_METADATA.get(arm, {})
    display = str(r.get("model_display") or top_data.get("model_display") or meta.get("display") or model_id)
    params_b = float(r.get("params_b") or top_data.get("params_b") or meta.get("params_b") or 0.0)
    active_params_b = float(
        r.get("active_params_b") or top_data.get("active_params_b") or meta.get("active_b") or params_b
    )
    if not is_v2 and arm == "omega":
        display, model_id, params_b, active_params_b = "STRATA O+OMEGA (0.8B, No-FB)", "omega_no_fb", 0.8, 0.8
    elif not is_v2 and arm == "pure_md":
        llm = str((top_data.get("init") or {}).get("resolved_llm") or "")
        if "0.8" in llm:
            display, model_id, params_b, active_params_b = "Pure Qwen3.5-0.8B", "pure_08b", 0.8, 0.8
        elif "7B" in llm or "7b" in llm:
            display, model_id, params_b, active_params_b = "Pure Qwen2.5-Coder-7B", "pure_7b", 7.6, 7.6

    expected = str(r.get("expected") or "")
    if "acceptable" in r:
        passed = bool(r.get("passed"))
        acceptable = bool(r.get("acceptable"))
        status = str(r.get("status") or ("PASSED" if passed else "FAILED"))
    elif "audit" in r and isinstance(r["audit"], dict):
        passed = bool(r["audit"].get("spec_compliant"))
        acceptable = passed
        status = "PASSED" if passed else "SPEC_FAILED"
    else:
        status = str(r.get("status") or "FAILED")
        passed = status == "PASSED"
        if not expected:
            expected = "clarify" if tid in ("vague_game", "vague_ml", "bound_ambiguous") else "synthesize"
        acceptable = passed or (expected == "clarify" and status in ("LINT_HALTED", "REFUSED_OK"))

    return {
        "model_key": model_id if is_v2 else (arm if arm in ("omega_no_fb", "omega_with_fb", "pure_08b", "pure_7b") else display),
        "display": display,
        "params_b": params_b,
        "active_params_b": active_params_b,
        "arm": arm,
        "family": str(r.get("family") or top_data.get("family") or ""),
        "task_id": tid,
        "seed": r.get("seed", top_data.get("seed")),
        "group": group,
        "expected": expected,
        "passed": passed,
        "acceptable": acceptable,
        "status": status,
        "wall_ms": float(r.get("wall_ms") or 0.0),
        "total_tokens": int(r.get("total_tokens") or 0),
        "hwm_mb": float((r.get("mem_mb") or {}).get("hwm_mb") or 0.0),
        "vram_mb": float((r.get("gpu_mb") or {}).get("total_used_mb") or 0.0),
    }


def load_all_results(paths: List[Path], *, clean_only: bool = False) -> List[Dict[str, Any]]:
    """Load and normalize benchmark records, optionally filtering partial or errored runs."""
    parsed: List[Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        recs = list(data.get("records") or [])
        parsed.append((p, data, recs))

    target_n = max((len(recs) for _, _, recs in parsed if recs), default=0) if clean_only else 0
    rows: List[Dict[str, Any]] = []
    for _, data, recs in parsed:
        if clean_only and target_n > 0:
            has_api_err = any(str(r.get("status") or "") in API_ERROR_STATUSES for r in recs)
            if len(recs) < target_n or has_api_err:
                continue
        for r in recs:
            rows.append(_normalize_legacy_record(r, data))
    return rows


def aggregate_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate normalized benchmark rows per evaluated arm with 95% Wilson CIs and split metrics."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["display"], []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for disp, items in grouped.items():
        has_seeds = any(it.get("seed") is not None for it in items)
        deduped: Dict[Any, Dict[str, Any]] = {}
        for it in items:
            key = (it["task_id"], it.get("seed")) if has_seeds else it["task_id"]
            deduped[key] = it
        vals = list(deduped.values())
        seeds_count = len({it.get("seed") for it in vals if it.get("seed") is not None}) or 1
        trials_n = len(vals)

        task_buckets: Dict[str, List[Dict[str, Any]]] = {}
        grp_buckets: Dict[str, List[Dict[str, Any]]] = {}
        seed_buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for v in vals:
            task_buckets.setdefault(v["task_id"], []).append(v)
            grp_buckets.setdefault(v.get("group", "other"), []).append(v)
            seed_buckets.setdefault(v.get("seed"), []).append(v)

        unique_tasks = len(task_buckets)
        by_task: Dict[str, Dict[str, Any]] = {}
        for tid, t_list in task_buckets.items():
            by_task[tid] = {
                "passed": all(r["passed"] for r in t_list),
                "acceptable": all(r["acceptable"] for r in t_list),
                "expected": str(t_list[0].get("expected") or ""),
                "group": str(t_list[0].get("group") or "other"),
                "status": str(t_list[0].get("status") or ""),
            }

        by_group: Dict[str, Dict[str, Any]] = {}
        for gname, gvals in sorted(grp_buckets.items()):
            gn = max(len(gvals), 1)
            g_unique = len({v["task_id"] for v in gvals})
            gacc = sum(1 for v in gvals if v["acceptable"])
            gpass = sum(1 for v in gvals if v["passed"])
            g_eff_k = gacc if seeds_count == 1 else round(gacc / seeds_count)
            ga_lo, ga_hi = wilson_ci_95(g_eff_k, max(g_unique, 1))
            by_group[gname] = {
                "tasks": g_unique,
                "trials": len(gvals),
                "passed": gpass,
                "acceptable": gacc,
                "acc_pct": round(100.0 * gacc / gn, 1),
                "acc_ci_lo": ga_lo,
                "acc_ci_hi": ga_hi,
            }

        strict_k = sum(1 for v in vals if v["passed"])
        acc_k = sum(1 for v in vals if v["acceptable"])
        n_trials = max(trials_n, 1)
        if seeds_count == 1:
            s_lo, s_hi = wilson_ci_95(strict_k, unique_tasks)
            a_lo, a_hi = wilson_ci_95(acc_k, unique_tasks)
            acc_pct = round(100.0 * acc_k / n_trials, 1)
            strict_pct = round(100.0 * strict_k / n_trials, 1)
            seed_std_pct = 0.0
        else:
            seed_acc_pcts = [
                100.0 * sum(1 for sv in svals if sv["acceptable"]) / max(len(svals), 1)
                for svals in seed_buckets.values()
            ]
            mean_acc = sum(seed_acc_pcts) / len(seed_acc_pcts)
            acc_pct = round(mean_acc, 1)
            strict_pct = round(100.0 * strict_k / n_trials, 1)
            seed_std_pct = (
                round(math.sqrt(sum((x - mean_acc) ** 2 for x in seed_acc_pcts) / (len(seed_acc_pcts) - 1)), 1)
                if len(seed_acc_pcts) > 1
                else 0.0
            )
            s_lo, s_hi = wilson_ci_95(int(round(strict_k / seeds_count)), unique_tasks)
            a_lo, a_hi = wilson_ci_95(int(round(acc_k / seeds_count)), unique_tasks)

        syn_vals = [v for v in vals if v.get("expected") == "synthesize"]
        saf_vals = [v for v in vals if v.get("expected") in ("refuse", "clarify")]
        blind_vals = [v for v in syn_vals if v.get("group") in ("v2_unseen", "v2_haystack")]
        syn_tasks = len({v["task_id"] for v in syn_vals})
        syn_passed = sum(1 for v in syn_vals if v["passed"])
        syn_pct = round(100.0 * syn_passed / len(syn_vals), 1) if syn_vals else 0.0
        safety_tasks = len({v["task_id"] for v in saf_vals})
        safety_acc = sum(1 for v in saf_vals if v["acceptable"])
        safety_refused = sum(1 for v in saf_vals if v["acceptable"] and not v["passed"])
        safety_pct = round(100.0 * safety_acc / len(saf_vals), 1) if saf_vals else 0.0

        summary[disp] = {
            "display": disp,
            "params_b": max(float(v["params_b"]) for v in vals),
            "active_params_b": max(float(v.get("active_params_b") or v["params_b"]) for v in vals),
            "arm": next((str(v.get("arm") or "") for v in vals if v.get("arm")), ""),
            "family": next((str(v.get("family") or "") for v in vals if v.get("family")), ""),
            "tasks": unique_tasks,
            "trials": trials_n,
            "seeds": seeds_count,
            "passed": strict_k,
            "acceptable": acc_k,
            "acc_pct": acc_pct,
            "seed_std_pct": seed_std_pct,
            "acc_ci_lo": a_lo,
            "acc_ci_hi": a_hi,
            "strict_pct": strict_pct,
            "strict_ci_lo": s_lo,
            "strict_ci_hi": s_hi,
            "syn_tasks": syn_tasks,
            "syn_trials": len(syn_vals),
            "syn_passed": syn_passed,
            "syn_pct": syn_pct,
            "blind_tasks": len({v["task_id"] for v in blind_vals}),
            "blind_passed": sum(1 for v in blind_vals if v["passed"]),
            "safety_tasks": safety_tasks,
            "safety_trials": len(saf_vals),
            "safety_acc": safety_acc,
            "safety_refused": safety_refused,
            "safety_pct": safety_pct,
            "mean_wall_ms": round(sum(v["wall_ms"] for v in vals) / n_trials, 1),
            "total_tokens": sum(v["total_tokens"] for v in vals),
            "peak_hwm_mb": round(max(v["hwm_mb"] for v in vals), 1),
            "peak_vram_mb": round(max(v["vram_mb"] for v in vals), 1),
            "by_task": by_task,
            "by_group": by_group,
        }
    return summary


# ---------------------------------------------------------------------------
# Formatting, Consolidation & Semantic Ordering Helpers
# ---------------------------------------------------------------------------


def _fmt_b(val: float) -> str:
    if val <= 0:
        return "—"
    s = f"{val:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return f"{s}B"


def _md_cell(val: Any) -> str:
    """Sanitize a string so Markdown table pipes never split GitHub GFM columns."""
    txt = str(val or "").replace(" | ", " · ").replace("\n", " ")
    return txt.replace("|", "\\|").strip()


def _family_key(display: str) -> str:
    if "STRATA" in display:
        return "STRATA v2"
    return str(display).split(" [", 1)[0].strip()


def _is_pure_arm(display: str) -> bool:
    return "[Pure" in str(display)


def _strip_verbose_suffixes(base: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*(?:Q\d|IQ\d|GGUF|GB|SB|OR|API)[^)]*\)", "", str(base or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-Instruct-GGUF$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"-Instruct$", "", cleaned, flags=re.IGNORECASE)
    if re.fullmatch(r"qwen3\.5-0\.8b(?:-instruct)?", str(base or "").strip(), flags=re.IGNORECASE):
        return "Qwen3.5-0.8B Base"
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?B-A\d+(?:\.\d+)?B$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_model_name(
    base: str,
    params_b: float,
    active_b: float,
    *,
    is_strata: bool = False,
    best_display: str = "",
    include_scale: bool = True,
) -> str:
    if is_strata:
        prof_match = re.search(r"Profile\s+([A-Z0-9]+)", str(best_display or ""))
        prof_tag = f"Profile {prof_match.group(1)}" if prof_match else "Best Profile"
        return f"STRATA v2 ({prof_tag}) · {_fmt_b(params_b)}" if include_scale else f"STRATA v2 ({prof_tag})"
    stem = _strip_verbose_suffixes(base)
    if not include_scale:
        return stem
    if 0 < active_b < params_b - 1e-6:
        return f"{stem} · {_fmt_b(params_b)} ({_fmt_b(active_b)} act.)"
    return f"{stem} · {_fmt_b(params_b)}"


def _short_scatter_name(base: str, params_b: float, *, is_strata: bool = False) -> str:
    if is_strata:
        return f"STRATA v2 ({_fmt_b(params_b)})"
    stem = _strip_verbose_suffixes(base)
    stem = stem.replace(" Super VL", "").replace(" Dense", "").replace(" MoE", "").replace(" Base", "")
    if "·" in stem:
        stem = stem.split("·", 1)[0].strip()
    if stem.endswith(f"-{_fmt_b(params_b)}") or f" {_fmt_b(params_b)}" in stem:
        return stem
    return f"{stem} ({_fmt_b(params_b)})"


def _arm_score_key(m: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        float(m["acc_pct"]),
        float(m.get("syn_pct", 0.0)),
        int(m.get("safety_refused", 0)),
        -int(m["total_tokens"]),
        -float(m["peak_vram_mb"]),
        -float(m["mean_wall_ms"]),
    )


def _consolidate_headline(summary: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    families: Dict[str, List[Dict[str, Any]]] = {}
    for m in summary.values():
        families.setdefault(_family_key(m["display"]), []).append(m)

    consolidated: List[Dict[str, Any]] = []
    for base, arms in families.items():
        is_strata = base == "STRATA v2"
        best = max(arms, key=_arm_score_key)
        ctx_arms = [a for a in arms if not _is_pure_arm(a["display"])]
        pure_arms = [a for a in arms if _is_pure_arm(a["display"])]
        best_ctx = max(ctx_arms, key=_arm_score_key) if ctx_arms else None
        best_pure = max(pure_arms, key=_arm_score_key) if pure_arms else None
        params_b = best["params_b"]
        active_b = best.get("active_params_b", params_b)
        consolidated.append({
            "family_key": base,
            "is_strata": is_strata,
            "is_pure_best": _is_pure_arm(best["display"]),
            "label": _clean_model_name(
                base, params_b, active_b, is_strata=is_strata, best_display=best["display"], include_scale=True
            ),
            "short_name": _clean_model_name(
                base, params_b, active_b, is_strata=is_strata, best_display=best["display"], include_scale=False
            ),
            "scatter_label": _short_scatter_name(base, params_b, is_strata=is_strata),
            "best": best,
            "best_ctx": best_ctx,
            "best_pure": best_pure,
            "arms": arms,
        })

    consolidated.sort(
        key=lambda e: (
            0 if e["is_strata"] else 1,
            -float(e["best"]["acc_pct"]),
            -float(e["best"].get("syn_pct", 0.0)),
            -int(e["best"].get("safety_refused", 0)),
            int(e["best"]["total_tokens"]),
            float(e["best"]["params_b"]),
        )
    )
    return consolidated


def _short_plot_label(display: str) -> str:
    text = re.sub(r"\s*\([^)]*GB\)", "", str(display or ""))
    text = re.sub(r"\s*\([^)]*instruct[^)]*\)", " (0.8B)", text, flags=re.IGNORECASE)
    text = text.replace("-Instruct-GGUF", "").replace("-GGUF", "")
    text = text.replace(" | ", ", ")
    return text.strip()


def _short_task_label(tid: str) -> str:
    txt = str(tid or "")
    txt = re.sub(r"^v2_", "", txt)
    txt = re.sub(r"^nl_", "", txt)
    txt = re.sub(r"_nl$", "", txt)
    txt = txt.replace("_1049_noise", " (1K noise)")
    txt = txt.replace("pillow", "PIL").replace("missing_file", "missing").replace("bound_ambiguous", "ambiguous")
    txt = txt.replace("wrong_domain_", "wrong ").replace("image_in_sqlite", "img/sqlite")
    txt = txt.replace("cryo_not_loaded", "cryo unloaded").replace("only_cv2", "only cv2")
    txt = txt.replace("_both_loaded", " (both)").replace("implicit_params", "implicit")
    return txt.replace("_", " ").strip()


def _order_tasks_semantically(task_meta: Dict[str, Tuple[str, str]]) -> List[str]:
    if not any(exp in ("synthesize", "refuse", "clarify") for exp, _ in task_meta.values()):
        return sorted(task_meta.keys())

    def _sort_key(tid: str) -> Tuple[int, int, str]:
        exp, grp = task_meta[tid]
        is_safety = 1 if exp in ("refuse", "clarify") else 0
        sub_rank = (0 if exp == "clarify" else 1) if is_safety else GROUP_SYN_ORDER.get(grp, 9)
        return (is_safety, sub_rank, tid)

    return sorted(task_meta.keys(), key=_sort_key)


def _monotone_pareto_envelope(points: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    if not points:
        return [], []
    max_per_x: Dict[float, float] = {}
    for px, py in points:
        rx = round(px, 2)
        max_per_x[rx] = max(max_per_x.get(rx, -1e9), py)
    env_xs: List[float] = []
    env_ys: List[float] = []
    running = -1e9
    for px in sorted(max_per_x.keys()):
        py = max_per_x[px]
        if py > running:
            running = py
            env_xs.append(px)
            env_ys.append(py)
    last_x = max(max_per_x.keys())
    if env_xs and env_xs[-1] < last_x:
        env_xs.append(last_x)
        env_ys.append(running)
    return env_xs, env_ys


def _arm_marker_style(display: str, theme: Dict[str, Any]) -> Tuple[str, str, str, int, int]:
    if "STRATA" in display:
        return "*", theme["strata"], theme["strata_dark"], 260, 6
    if "[Pure" in display:
        return "^", theme["pure"], theme["ink"], 95, 4
    return "o", theme["ctx"], theme["ctx_dark"], 100, 5


def _decoding_regime(display: str) -> str:
    if ("Think" in display and "NoThink" not in display) or "EffortHigh" in display:
        return "think"
    if "NoThink" in display or "EffortLow" in display:
        return "nothink"
    return "instruct"


# ---------------------------------------------------------------------------
# Understated Paper-Style Matplotlib Primitives
# ---------------------------------------------------------------------------


def _configure_rcparams(plt: Any, theme: Dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.facecolor": theme["bg"],
        "figure.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"],
        "axes.edgecolor": theme["spine"],
        "axes.linewidth": 0.85,
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "xtick.major.size": 3.2,
        "ytick.major.size": 0.0,
    })


def _style_axis(ax: Any, theme: Dict[str, Any], *, grid_axis: str = "x", hide_left: bool = False) -> None:
    ax.set_facecolor(theme["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not hide_left)
    if not hide_left:
        ax.spines["left"].set_color(theme["spine"])
    ax.spines["bottom"].set_color(theme["spine"])
    ax.set_axisbelow(True)
    if grid_axis == "both":
        ax.grid(True, which="major", linestyle="--", color=theme["grid"], alpha=0.9, linewidth=0.75)
    elif grid_axis in ("x", "y"):
        ax.grid(axis=grid_axis, linestyle="--", color=theme["grid"], alpha=0.95, linewidth=0.75)


def _add_header(
    fig: Any,
    theme: Dict[str, Any],
    title: str,
    subtitle: str,
    *,
    title_y: float = 0.985,
    sub_y: float = 0.943,
    x: float = 0.04,
) -> None:
    fig.text(x, title_y, title, fontsize=12.4, fontweight="bold", color=theme["ink"], ha="left", va="top")
    fig.text(x, sub_y, subtitle, fontsize=8.8, color=theme["muted"], ha="left", va="top")


def _draw_strata_divider(ax: Any, rows: List[Dict[str, Any]], theme: Dict[str, Any]) -> None:
    """Draw a subtle separator line below the STRATA reference row (without loud tint bands)."""
    for idx, entry in enumerate(rows):
        if entry["is_strata"] and idx + 1 < len(rows):
            ax.axhline(idx + 0.5, color=theme["spine"], lw=0.95, ls="-", zorder=2)
            break


def _place_direct_scatter_labels(
    fig: Any,
    ax: Any,
    theme: Dict[str, Any],
    items: List[Tuple[float, float, str, bool, float]],
) -> None:
    """Place collision-free plain text labels near scatter markers with subtle leader ticks."""
    if not items:
        return
    probes: List[Tuple[float, float, str, bool, float, float]] = []
    temp_texts = []
    for px, py, txt, is_strata, _sz in items:
        t = ax.text(
            0.5,
            0.5,
            txt,
            transform=ax.transAxes,
            fontsize=8.1 if is_strata else 7.8,
            fontweight="bold" if is_strata else "normal",
        )
        temp_texts.append((t, px, py, txt, is_strata))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)
    for t, px, py, txt, is_strata in temp_texts:
        bb = t.get_window_extent(renderer=renderer)
        probes.append((px, py, txt, is_strata, bb.width + 5.0, bb.height + 4.0))
        t.remove()

    dpi = float(fig.dpi)
    px_per_pt = dpi / 72.0
    pts_px: List[Tuple[float, float, float]] = []
    for px, py, _, _, sz in items:
        mx, my = ax.transData.transform((px, py))
        rad_px = (math.sqrt(max(sz, 36.0) / math.pi) + 2.8) * px_per_pt
        pts_px.append((mx, my, rad_px))

    def _overlaps(r1: Tuple[float, float, float, float], r2: Tuple[float, float, float, float], pad: float = 4.0) -> bool:
        return not (r1[2] + pad <= r2[0] or r2[2] + pad <= r1[0] or r1[3] + pad <= r2[1] or r2[3] + pad <= r1[1])

    placed_boxes: List[Tuple[float, float, float, float]] = []
    angles_deg = [18, -22, 158, -158, 45, -45, 135, -135, 72, -72, 108, -108, 0, 180, 90, -90, 32, -32, 148, -148]
    radii_pts = [13, 21, 31, 43, 57, 72, 88]

    indexed = list(enumerate(probes))
    indexed.sort(key=lambda tup: (0 if tup[1][3] else 1, -tup[1][1], tup[1][0]))

    for idx, (px, py, txt, is_strata, bw, bh) in indexed:
        sx, sy, _ = pts_px[idx]
        best_off = (14.0, 8.0)
        best_ha = "left"
        best_va = "center"
        best_box = (sx + 10, sy - bh / 2, sx + 10 + bw, sy + bh / 2)
        best_penalty = 1e12
        found = False

        for rad_pt in radii_pts:
            for deg in angles_deg:
                ang = math.radians(deg)
                dx_pt = rad_pt * math.cos(ang)
                dy_pt = rad_pt * math.sin(ang)
                dx_px = dx_pt * px_per_pt
                dy_px = dy_pt * px_per_pt

                if abs(math.cos(ang)) < 0.28:
                    ha = "center"
                    x0 = sx + dx_px - bw * 0.5
                elif math.cos(ang) >= 0:
                    ha = "left"
                    x0 = sx + dx_px
                else:
                    ha = "right"
                    x0 = sx + dx_px - bw

                if abs(math.sin(ang)) < 0.25:
                    va = "center"
                    y0 = sy + dy_px - bh * 0.5
                elif math.sin(ang) >= 0:
                    va = "bottom"
                    y0 = sy + dy_px
                else:
                    va = "top"
                    y0 = sy + dy_px - bh

                cand = (x0, y0, x0 + bw, y0 + bh)
                out_bounds = (
                    cand[0] < ax_bb.x0 + 4
                    or cand[2] > ax_bb.x1 - 4
                    or cand[1] < ax_bb.y0 + 4
                    or cand[3] > ax_bb.y1 - 4
                )
                if out_bounds:
                    continue

                pt_hit = any(_overlaps(cand, (mx - mr, my - mr, mx + mr, my + mr), pad=2.0) for mx, my, mr in pts_px)
                box_hit = any(_overlaps(cand, pb, pad=5.0) for pb in placed_boxes)
                if not pt_hit and not box_hit:
                    best_off = (dx_pt, dy_pt)
                    best_ha = ha
                    best_va = va
                    best_box = cand
                    found = True
                    break

                penalty = (1000.0 if pt_hit else 0.0) + (500.0 if box_hit else 0.0) + rad_pt
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_off = (dx_pt, dy_pt)
                    best_ha = ha
                    best_va = va
                    best_box = cand
            if found:
                break

        placed_boxes.append(best_box)
        tc = theme["strata_dark"] if is_strata else theme["ink"]
        dist_pt = math.hypot(best_off[0], best_off[1])
        arrow = (
            {"arrowstyle": "-", "color": theme["pure_light"], "lw": 0.85, "shrinkA": 2, "shrinkB": 5}
            if dist_pt > 12.0
            else None
        )
        ax.annotate(
            txt,
            xy=(px, py),
            xycoords="data",
            xytext=best_off,
            textcoords="offset points",
            ha=best_ha,
            va=best_va,
            fontsize=8.1 if is_strata else 7.8,
            fontweight="bold" if is_strata else "normal",
            color=tc,
            zorder=8 if is_strata else 7,
            arrowprops=arrow,
            bbox={"boxstyle": "square,pad=0.06", "fc": theme["bg"], "ec": "none", "alpha": 0.85},
        )


def _draw_grouped_matrix(
    fig: Any,
    ax: Any,
    theme: Dict[str, Any],
    mat: Any,
    cell_texts: List[List[str]],
    cell_text_colors: List[List[str]],
    row_labels: List[str],
    row_strata_mask: List[bool],
    row_scores: List[str],
    col_ids: List[str],
    task_meta: Dict[str, Tuple[str, str]],
    *,
    cmap: Any,
    norm: Any,
    header_title: str,
    header_subtitle: str,
) -> None:
    import numpy as np

    n_rows, n_cols = mat.shape
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=theme["bg"], linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8.8)
    for ytick, is_s in zip(ax.get_yticklabels(), row_strata_mask):
        if is_s:
            ytick.set_fontweight("bold")
            ytick.set_color(theme["strata_dark"])

    if n_cols <= 45:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([_short_task_label(t) for t in col_ids], rotation=35, ha="right", fontsize=8.3)
    else:
        step = max(1, n_cols // 20)
        ticks = list(range(0, n_cols, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([_short_task_label(col_ids[t]) for t in ticks], rotation=30, ha="right", fontsize=7.6)

    if n_cols <= 32:
        for i in range(n_rows):
            for j in range(n_cols):
                txt = cell_texts[i][j]
                if not txt:
                    continue
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9.0 if len(txt) <= 2 else 7.2,
                    fontweight="bold" if len(txt) <= 2 else "normal",
                    color=cell_text_colors[i][j],
                    zorder=4,
                )

    has_split = any(task_meta[t][0] in ("refuse", "clarify") for t in col_ids) and any(
        task_meta[t][0] == "synthesize" for t in col_ids
    )
    if has_split:
        syn_count = sum(1 for t in col_ids if task_meta[t][0] == "synthesize")
        ax.axvline(syn_count - 0.5, color=theme["ink"], linewidth=2.0, zorder=5)

        sub_spans: List[Tuple[str, int, int]] = []
        sub_name_map = {
            "v2_nl_core": "NL Core",
            "v2_domain_spec": "Domain Spec",
            "v2_unseen": "Blind SDK",
            "v2_haystack": "Haystack",
        }
        cur_label = None
        cur_start = 0
        for j, tid in enumerate(col_ids):
            exp, grp = task_meta[tid]
            lbl = "Clarify" if exp == "clarify" else ("Wrong Domain" if exp == "refuse" else sub_name_map.get(grp, grp))
            if lbl != cur_label:
                if cur_label is not None:
                    sub_spans.append((cur_label, cur_start, j - 1))
                cur_label = lbl
                cur_start = j
        if cur_label is not None:
            sub_spans.append((cur_label, cur_start, n_cols - 1))

        for k_idx, (slbl, s0, s1) in enumerate(sub_spans):
            if k_idx > 0 and s0 != syn_count:
                ax.axvline(s0 - 0.5, color=theme["bg"], linewidth=2.4, zorder=4)
                ax.axvline(s0 - 0.5, color=theme["light_ink"], linewidth=0.9, linestyle=":", zorder=4)
            ax.text(
                (s0 + s1) / 2.0,
                -0.66,
                slbl,
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=theme["muted"],
                clip_on=False,
            )

        ax.text(
            (syn_count - 1) / 2.0,
            -1.15,
            f"Actionable Synthesis ({syn_count})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=theme["syn"],
            clip_on=False,
        )
        ax.text(
            syn_count + (n_cols - syn_count - 1) / 2.0,
            -1.15,
            f"Safety & Refusal ({n_cols - syn_count})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=theme["safety"],
            clip_on=False,
        )

    if any(row_strata_mask):
        first_non_s = next((idx for idx, flag in enumerate(row_strata_mask) if not flag), None)
        if first_non_s is not None and first_non_s > 0:
            ax.axhline(first_non_s - 0.5, color=theme["ink"], linewidth=1.4, zorder=5)

    if row_scores:
        ax.text(n_cols - 0.15, -0.66, "Score", ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=theme["muted"], clip_on=False)
        for i, sc in enumerate(row_scores):
            ax.text(
                n_cols - 0.15,
                i,
                sc,
                ha="left",
                va="center",
                fontsize=8.4,
                fontweight="bold" if row_strata_mask[i] else "normal",
                color=theme["strata_dark"] if row_strata_mask[i] else theme["ink"],
                clip_on=False,
            )

    _add_header(fig, theme, header_title, header_subtitle, title_y=0.98, sub_y=0.935, x=0.04)


# ---------------------------------------------------------------------------
# Headline Executive Figures (--layout headline)
# ---------------------------------------------------------------------------


def _generate_headline_plots(
    summary: Dict[str, Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    theme_name: str = "light",
    suffix: str = "",
) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    theme = THEMES[theme_name]
    _configure_rcparams(plt, theme)
    generated: List[Path] = []
    entries = _consolidate_headline(summary)
    if not entries:
        return generated

    n_rows = len(entries)
    y = np.arange(n_rows)
    row_labels = [e["label"] for e in entries]
    strata_mask = [e["is_strata"] for e in entries]
    best_arms = [e["best"] for e in entries]

    # Fig 1: Overall Score + Task Split + Token Usage
    fig_h1 = max(5.4, 0.46 * n_rows + 1.8)
    fig = plt.figure(figsize=(15.6, fig_h1), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.35, 1.15], wspace=0.14)
    ax_ov = fig.add_subplot(gs[0, 0])
    ax_sp = fig.add_subplot(gs[0, 1], sharey=ax_ov)
    ax_tk = fig.add_subplot(gs[0, 2], sharey=ax_ov)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.81, bottom=0.11)

    for ax_i in (ax_ov, ax_sp, ax_tk):
        _draw_strata_divider(ax_i, entries, theme)

    ov_vals = [b["acc_pct"] for b in best_arms]
    ov_err = [
        [max(0.0, b["acc_pct"] - b["acc_ci_lo"]) for b in best_arms],
        [max(0.0, b["acc_ci_hi"] - b["acc_pct"]) for b in best_arms],
    ]
    bar_colors = [
        theme["strata"] if e["is_strata"] else (theme["pure"] if e["is_pure_best"] else theme["ctx"])
        for e in entries
    ]
    ax_ov.barh(y, ov_vals, height=0.58, color=bar_colors, edgecolor="none", zorder=3)
    ax_ov.errorbar(
        ov_vals,
        y,
        xerr=ov_err,
        fmt="none",
        ecolor=theme["ink"],
        elinewidth=0.95,
        capsize=2.5,
        alpha=0.72,
        zorder=4,
    )
    for idx, e in enumerate(entries):
        b = e["best"]
        pure = e["best_pure"]
        if pure is not None and not e["is_pure_best"]:
            ax_ov.scatter(
                pure["acc_pct"],
                idx,
                marker="^",
                s=46,
                facecolor=theme["bg"],
                edgecolor=theme["ink"],
                linewidth=1.15,
                zorder=5,
            )
        txt_x = max(b["acc_pct"], b["acc_ci_hi"]) + 2.2
        lbl = f"{b['acc_pct']:.1f}% ({b['acceptable']}/{b['trials']})"
        if e["is_pure_best"]:
            lbl += " [Pure]"
        ax_ov.text(
            txt_x,
            idx,
            lbl,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold" if e["is_strata"] else "normal",
            color=theme["strata_dark"] if e["is_strata"] else theme["ink"],
            zorder=6,
        )
    ax_ov.invert_yaxis()
    ax_ov.set_xlim(0, 134)
    ax_ov.set_xticks([0, 25, 50, 75, 100])
    ax_ov.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.4)
    ax_ov.set_yticks(y)
    ax_ov.set_yticklabels(row_labels, fontsize=8.7)
    for ytick, is_s in zip(ax_ov.get_yticklabels(), strata_mask):
        if is_s:
            ytick.set_fontweight("bold")
            ytick.set_color(theme["strata_dark"])
    ax_ov.set_xlabel("Acceptable Rate (%, 95% Wilson CI)", fontsize=8.9, labelpad=6)
    ax_ov.set_title("Overall Score (Best Arm vs Pure ▲)", fontsize=9.8, fontweight="bold", color=theme["ink"], pad=8, loc="left")
    _style_axis(ax_ov, theme, grid_axis="x")

    syn_vals = [b["syn_pct"] for b in best_arms]
    saf_vals = [b["safety_pct"] for b in best_arms]
    h_sp = 0.29
    ax_sp.barh(y - h_sp / 2, syn_vals, height=h_sp, color=theme["syn"], zorder=3)
    ax_sp.barh(y + h_sp / 2, saf_vals, height=h_sp, color=theme["safety"], zorder=3)
    for idx, b in enumerate(best_arms):
        st = max(b["syn_tasks"], 1)
        sf = max(b["safety_tasks"], 1)
        ax_sp.text(b["syn_pct"] + 2.5, idx - h_sp / 2, f"{b['syn_passed']}/{st}", va="center", ha="left", fontsize=7.6, color=theme["ink"])
        sref = b.get("safety_refused", b["safety_acc"])
        saf_tag = f"{b['safety_acc']}/{sf}" if sref >= b["safety_acc"] else f"{b['safety_acc']}/{sf} ({sref} ref)"
        ax_sp.text(b["safety_pct"] + 2.5, idx + h_sp / 2, saf_tag, va="center", ha="left", fontsize=7.6, color=theme["muted"])
    ax_sp.set_xlim(0, 146)
    ax_sp.set_xticks([0, 50, 100])
    ax_sp.set_xticklabels(["0%", "50%", "100%"], fontsize=8.4)
    ax_sp.tick_params(axis="y", labelleft=False)
    ax_sp.set_xlabel("Sub-Axis Accuracy (%)", fontsize=8.9, labelpad=6)
    ax_sp.set_title("Synthesis (12) vs. Safety (7)", fontsize=9.8, fontweight="bold", color=theme["ink"], pad=8, loc="left")
    _style_axis(ax_sp, theme, grid_axis="x", hide_left=True)

    tok_k = [b["total_tokens"] / 1000.0 for b in best_arms]
    max_tok_k = max(tok_k, default=50.0)
    tok_colors = [
        theme["strata"] if e["is_strata"] else (theme["pure_bar"] if e["is_pure_best"] else theme["muted"])
        for e in entries
    ]
    ax_tk.barh(y, tok_k, height=0.54, color=tok_colors, zorder=3)
    for idx, e in enumerate(entries):
        tk = tok_k[idx]
        lbl = f"{tk:.1f}K [Pure]" if e["is_pure_best"] else f"{tk:.1f}K"
        ax_tk.text(
            tk + max_tok_k * 0.028,
            idx,
            lbl,
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold" if e["is_strata"] else "normal",
            color=theme["strata_dark"] if e["is_strata"] else theme["ink"],
        )
    ax_tk.set_xlim(0, max_tok_k * 1.32)
    ax_tk.tick_params(axis="y", labelleft=False)
    ax_tk.set_xlabel("Total Suite Tokens (K)", fontsize=8.9, labelpad=6)
    ax_tk.set_title("Total Token Usage (Best Arm)", fontsize=9.8, fontweight="bold", color=theme["ink"], pad=8, loc="left")
    _style_axis(ax_tk, theme, grid_axis="x", hide_left=True)

    legend_handles = [
        Patch(facecolor=theme["strata"], label="STRATA v2 (0.8B)"),
        Patch(facecolor=theme["ctx"], label="Context Baseline (SKILL.md / Haystack)"),
        Patch(facecolor=theme["pure"], label="Pure-Best Baseline"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=theme["bg"], markeredgecolor=theme["ink"], markeredgewidth=1.15, markersize=7.0, label="Pure Weights Score"),
        Patch(facecolor=theme["syn"], label="Synthesis (12)"),
        Patch(facecolor=theme["safety"], label="Safety & Refusal (7)"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.04, 0.905), ncol=6, frameon=False, fontsize=8.0, columnspacing=1.1, handletextpad=0.45)
    _add_header(
        fig,
        theme,
        f"{prefix}Overall Score, Task Split, and Token Usage",
        "Best complete N = 19 run per model (12 actionable synthesis + 7 safety/refusal tasks); error bars denote 95% Wilson CI.",
    )
    p1 = out_dir / f"fig1_accuracy_comparison{suffix}.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p1)

    # Fig 2: Dual Pareto Frontier (Zero marketing arrows or card borders)
    fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(14.8, 5.8), dpi=200)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.12, wspace=0.12)

    scatter_items_p: List[Tuple[float, float, str, bool, float]] = []
    scatter_items_t: List[Tuple[float, float, str, bool, float]] = []
    baseline_pts_p: List[Tuple[float, float]] = []

    for e in entries:
        b = e["best"]
        px = max(float(b["params_b"]), 0.5)
        tx = float(b["total_tokens"]) / 1000.0
        py = float(b["acc_pct"])
        if e["is_strata"]:
            marker, fc, ec, sz, zord = "*", theme["strata"], theme["strata_dark"], 280, 6
        elif e["is_pure_best"]:
            marker, fc, ec, sz, zord = "^", theme["pure"], theme["ink"], 95, 5
            baseline_pts_p.append((px, py))
        else:
            marker, fc, ec, sz, zord = "o", theme["ctx"], theme["ctx_dark"], 100, 5
            baseline_pts_p.append((px, py))

        ax_p.scatter(px, py, s=sz, c=fc, edgecolors=ec, linewidths=0.95, marker=marker, zorder=zord)
        ax_t.scatter(tx, py, s=sz, c=fc, edgecolors=ec, linewidths=0.95, marker=marker, zorder=zord)
        scatter_items_p.append((px, py, e["scatter_label"], e["is_strata"], float(sz)))
        scatter_items_t.append((tx, py, e["scatter_label"], e["is_strata"], float(sz)))

    env_xs, env_ys = _monotone_pareto_envelope(baseline_pts_p)
    if len(env_xs) >= 2:
        ax_p.plot(env_xs, env_ys, linestyle="--", color=theme["pure_light"], linewidth=1.25, zorder=2)

    strata_entry = next((e for e in entries if e["is_strata"]), None)
    if strata_entry is not None:
        s_acc = float(strata_entry["best"]["acc_pct"])
        ax_p.axhline(s_acc, linestyle=":", color=theme["strata_border"], linewidth=1.1, alpha=0.75, zorder=1)
        ax_t.axhline(s_acc, linestyle=":", color=theme["strata_border"], linewidth=1.1, alpha=0.75, zorder=1)

    max_px = max((x for x, _, _, _, _ in scatter_items_p), default=100.0)
    min_px = min((x for x, _, _, _, _ in scatter_items_p), default=0.8)
    ax_p.set_xscale("log")
    ax_p.set_xlim(min_px * 0.48, max(max_px * 2.1, 180.0))
    ax_p.set_ylim(30, 110)
    param_ticks = [t for t in (1, 3, 10, 30, 100, 300, 500) if min_px * 0.5 <= t <= max(max_px * 2.1, 180.0)]
    if param_ticks:
        ax_p.set_xticks(param_ticks)
        ax_p.set_xticklabels([f"{t}B" for t in param_ticks], fontsize=8.4)
    ax_p.set_yticks([30, 50, 70, 90, 100])
    ax_p.set_yticklabels(["30%", "50%", "70%", "90%", "100%"], fontsize=8.4)
    ax_p.set_xlabel("Model Total Parameters (Billions, log scale)", fontsize=9.0, labelpad=6)
    ax_p.set_ylabel("Acceptable Rate (%)", fontsize=9.0, labelpad=6)
    ax_p.set_title("Accuracy vs. Parameter Scale", fontsize=10.0, fontweight="bold", color=theme["ink"], loc="left", pad=8)
    _style_axis(ax_p, theme, grid_axis="both")

    max_tx = max((x for x, _, _, _, _ in scatter_items_t), default=120.0)
    ax_t.set_xlim(-6.0, max_tx * 1.28)
    ax_t.set_ylim(30, 110)
    ax_t.set_yticks([30, 50, 70, 90, 100])
    ax_t.set_yticklabels(["", "", "", "", ""])
    ax_t.set_xlabel("Total Suite Token Usage (K tokens)", fontsize=9.0, labelpad=6)
    ax_t.set_title("Accuracy vs. Token Budget", fontsize=10.0, fontweight="bold", color=theme["ink"], loc="left", pad=8)
    _style_axis(ax_t, theme, grid_axis="both")

    _place_direct_scatter_labels(fig, ax_p, theme, scatter_items_p)
    _place_direct_scatter_labels(fig, ax_t, theme, scatter_items_t)

    pareto_handles = [
        Line2D([0], [0], marker="*", color="none", label="STRATA v2 (0.8B)", markerfacecolor=theme["strata"], markeredgecolor=theme["strata_dark"], markersize=11),
        Line2D([0], [0], marker="o", color="none", label="Context Baseline Best Arm", markerfacecolor=theme["ctx"], markeredgecolor=theme["ctx_dark"], markersize=7.5),
        Line2D([0], [0], marker="^", color="none", label="Pure Weights Best Arm", markerfacecolor=theme["pure"], markeredgecolor=theme["ink"], markersize=7.5),
        Line2D([0], [0], color=theme["pure_light"], linestyle="--", linewidth=1.25, label="Baseline Pareto Frontier"),
    ]
    fig.legend(handles=pareto_handles, loc="upper left", bbox_to_anchor=(0.06, 0.90), ncol=4, frameon=False, fontsize=8.2)
    _add_header(
        fig,
        theme,
        f"{prefix}Accuracy vs. Parameter Scale and Token Budget",
        "Highest-scoring complete N = 19 configuration per model; dashed curve marks non-dominated baseline frontier.",
        x=0.06,
    )
    p2 = out_dir / f"fig2_pareto_scale{suffix}.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p2)

    # Fig 3: Peak GPU Memory and Mean Task Latency
    fig_h3 = max(5.2, 0.44 * n_rows + 1.8)
    fig, (ax_vr, ax_lt) = plt.subplots(1, 2, figsize=(14.6, fig_h3), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.81, bottom=0.12, wspace=0.10)

    _draw_strata_divider(ax_vr, entries, theme)
    _draw_strata_divider(ax_lt, entries, theme)

    vram_gb = [b["peak_vram_mb"] / 1024.0 for b in best_arms]
    max_gb = max(max(vram_gb, default=15.0), 16.0)
    vr_colors = [theme["strata"] if e["is_strata"] else theme["vram"] for e in entries]
    ax_vr.barh(y, vram_gb, height=0.56, color=vr_colors, zorder=3)
    ax_vr.axvline(8.0, color=theme["pure_light"], linestyle="--", linewidth=0.95, zorder=2)
    ax_vr.axvline(15.0, color=theme["muted"], linestyle=":", linewidth=0.95, zorder=2)
    ax_vr.text(8.15, -0.58, "8 GiB", fontsize=7.5, color=theme["muted"], va="bottom")
    ax_vr.text(14.85, -0.58, "15 GiB", fontsize=7.5, color=theme["muted"], va="bottom", ha="right")

    for idx, gb in enumerate(vram_gb):
        if gb <= 0.0:
            ax_vr.text(0.28, idx, "Cloud API", va="center", ha="left", fontsize=8.0, style="italic", color=theme["muted"], zorder=4)
        else:
            ax_vr.text(
                gb + max_gb * 0.02,
                idx,
                f"{gb:.1f} GiB",
                va="center",
                ha="left",
                fontsize=8.1,
                fontweight="bold" if entries[idx]["is_strata"] else "normal",
                color=theme["strata_dark"] if entries[idx]["is_strata"] else theme["ink"],
                bbox={"boxstyle": "square,pad=0.04", "fc": theme["bg"], "ec": "none", "alpha": 0.85},
                zorder=4,
            )
    ax_vr.invert_yaxis()
    ax_vr.set_xlim(0, max_gb * 1.18)
    ax_vr.set_yticks(y)
    ax_vr.set_yticklabels(row_labels, fontsize=8.7)
    for ytick, is_s in zip(ax_vr.get_yticklabels(), strata_mask):
        if is_s:
            ytick.set_fontweight("bold")
            ytick.set_color(theme["strata_dark"])
    ax_vr.set_xlabel("Peak GPU Memory (GiB)", fontsize=8.9, labelpad=6)
    ax_vr.set_title("Peak GPU Memory", fontsize=10.0, fontweight="bold", color=theme["ink"], loc="left", pad=12)
    _style_axis(ax_vr, theme, grid_axis="x")

    lat_s = [b["mean_wall_ms"] / 1000.0 for b in best_arms]
    max_s = max(lat_s, default=30.0)
    lt_colors = [theme["strata"] if e["is_strata"] else theme["latency"] for e in entries]
    ax_lt.barh(y, lat_s, height=0.56, color=lt_colors, zorder=3)
    for idx, sec in enumerate(lat_s):
        ax_lt.text(
            sec + max_s * 0.02,
            idx,
            f"{sec:.1f}s",
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold" if entries[idx]["is_strata"] else "normal",
            color=theme["strata_dark"] if entries[idx]["is_strata"] else theme["ink"],
        )
    ax_lt.set_xlim(0, max_s * 1.18)
    ax_lt.tick_params(axis="y", labelleft=False)
    ax_lt.set_xlabel("Mean Wall Latency (s / task)", fontsize=8.9, labelpad=6)
    ax_lt.set_title("Mean Execution Time per Task", fontsize=10.0, fontweight="bold", color=theme["ink"], loc="left", pad=12)
    _style_axis(ax_lt, theme, grid_axis="x", hide_left=True)

    _add_header(
        fig,
        theme,
        f"{prefix}Peak GPU Memory and Mean Task Latency",
        "Measured across all 19 tasks on each model's best complete configuration (ordered as in Fig. 1).",
    )
    p3 = out_dir / f"fig3_resources_and_latency{suffix}.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p3)

    # Fig 4: Grouped Task Outcome Matrix (11 × 19)
    task_meta: Dict[str, Tuple[str, str]] = {}
    for b in best_arms:
        for tid, rec in b["by_task"].items():
            if tid not in task_meta or rec.get("expected"):
                task_meta[tid] = (str(rec.get("expected") or ""), str(rec.get("group") or "other"))
    col_ids = _order_tasks_semantically(task_meta)

    if col_ids:
        mat = np.zeros((n_rows, len(col_ids)))
        cell_texts: List[List[str]] = [["" for _ in col_ids] for _ in range(n_rows)]
        cell_colors: List[List[str]] = [["#FFFFFF" for _ in col_ids] for _ in range(n_rows)]
        dark_ink = "#FFFFFF" if theme_name == "light" else "#0D1117"
        for i, b in enumerate(best_arms):
            for j, tid in enumerate(col_ids):
                rec = b["by_task"].get(tid)
                exp = task_meta[tid][0]
                if rec is None:
                    mat[i, j] = -1.0
                    cell_texts[i][j] = "·"
                    cell_colors[i][j] = theme["light_ink"]
                elif not rec["acceptable"]:
                    mat[i, j] = 0.0
                    cell_texts[i][j] = "✗"
                    cell_colors[i][j] = dark_ink
                elif exp in ("refuse", "clarify") and rec["passed"]:
                    mat[i, j] = 1.0
                    cell_texts[i][j] = "~"
                    cell_colors[i][j] = "#1F2937"
                elif rec["passed"]:
                    mat[i, j] = 3.0
                    cell_texts[i][j] = "✓"
                    cell_colors[i][j] = dark_ink
                else:
                    mat[i, j] = 2.0
                    cell_texts[i][j] = "⊘"
                    cell_colors[i][j] = dark_ink

        row_scores = [f"{b['acceptable']}/{b['trials']}" for b in best_arms]
        fig_w = max(13.6, 0.62 * len(col_ids) + 4.2)
        fig_h4 = max(5.4, 0.42 * n_rows + 2.2)
        fig, ax_hm = plt.subplots(figsize=(fig_w, fig_h4), dpi=200)
        fig.subplots_adjust(left=0.22, right=0.92, top=0.77, bottom=0.19)

        cmap = matplotlib.colors.ListedColormap([
            theme["na"],
            theme["fail"],
            theme["fallback"],
            theme["safety"],
            theme["syn"],
        ])
        norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        _draw_grouped_matrix(
            fig,
            ax_hm,
            theme,
            mat,
            cell_texts,
            cell_colors,
            row_labels,
            strata_mask,
            row_scores,
            col_ids,
            task_meta,
            cmap=cmap,
            norm=norm,
            header_title=f"{prefix}Per-Task Outcomes (Best Configuration per Model)",
            header_subtitle="Grouped into 12 actionable synthesis tasks (left) and 7 safety/refusal tasks (right).",
        )
        hm_handles = [
            Patch(facecolor=theme["syn"], label="✓ Verified Synthesis Pass"),
            Patch(facecolor=theme["safety"], label="⊘ Clean Refusal / Clarify"),
            Patch(facecolor=theme["fallback"], label="~ Vague-Prompt Code Fallback"),
            Patch(facecolor=theme["fail"], label="✗ Failed / Hallucinated"),
        ]
        fig.legend(handles=hm_handles, loc="upper left", bbox_to_anchor=(0.04, 0.895), ncol=4, frameon=False, fontsize=8.1)
        p4 = out_dir / f"fig4_task_heatmap{suffix}.png"
        fig.savefig(p4, dpi=200, bbox_inches="tight", pad_inches=0.20)
        plt.close(fig)
        generated.append(p4)

    return generated


# ---------------------------------------------------------------------------
# Frontier Decoding Ablations Suite (--layout ablation)
# ---------------------------------------------------------------------------


def _generate_ablation_plots(
    summary: Dict[str, Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    theme_name: str = "light",
    suffix: str = "",
) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    theme = THEMES[theme_name]
    _configure_rcparams(plt, theme)
    generated: List[Path] = []
    entries = _consolidate_headline(summary)
    if not entries:
        return generated

    strata_ref = next((e for e in entries if e["is_strata"]), None)
    frontier_entries = [e for e in entries if not e["is_strata"]] or entries
    frontier_entries.sort(
        key=lambda e: (-float(e["best"]["acc_pct"]), -float(e["best"].get("syn_pct", 0.0)), -float(e["best"]["params_b"]))
    )

    n_f = len(frontier_entries)
    fy = np.arange(n_f)
    fig_h = max(4.8, 0.54 * n_f + 2.0)
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(15.0, fig_h), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.745, bottom=0.13, wspace=0.08)

    strata_line = float(strata_ref["best"]["acc_pct"]) if strata_ref else None
    regime_style = {
        "think": ("o", theme["think"]),
        "nothink": ("s", theme["nothink"]),
        "instruct": ("D", theme["instruct"]),
    }

    for ax_i, is_pure_panel, p_title in (
        (ax_c, False, "With SKILL.md API Context"),
        (ax_p, True, "Pure Weights (No API Docs)"),
    ):
        if strata_line is not None:
            ax_i.axvline(strata_line, color=theme["strata"], linestyle="--", linewidth=1.3, zorder=2)
            ax_i.text(
                strata_line - 1.2,
                -0.68,
                f"STRATA v2 (0.8B): {strata_line:.0f}%",
                ha="right",
                va="bottom",
                fontsize=7.9,
                fontweight="bold",
                color=theme["strata_dark"],
            )

        for idx, e in enumerate(frontier_entries):
            arms = [a for a in e["arms"] if _is_pure_arm(a["display"]) == is_pure_panel]
            if not arms:
                ax_i.text(
                    50.0,
                    idx,
                    "No complete N=19 run (upstream API 503)",
                    ha="center",
                    va="center",
                    fontsize=7.9,
                    style="italic",
                    color=theme["muted"],
                )
                continue

            vals = [float(a["acc_pct"]) for a in arms]
            vmin, vmax = min(vals), max(vals)
            if vmax > vmin:
                ax_i.plot([vmin, vmax], [idx, idx], color=theme["range_bg"], linewidth=6.5, solid_capstyle="round", zorder=2)
                ax_i.plot([vmin, vmax], [idx, idx], color=theme["pure_light"], linewidth=1.2, zorder=3)

            offsets = np.linspace(-0.16, 0.16, len(arms)) if len(arms) > 1 else [0.0]
            for arm, dy in zip(sorted(arms, key=lambda a: (a["acc_pct"], a["display"])), offsets):
                reg = _decoding_regime(arm["display"])
                mkr, col = regime_style[reg]
                ax_i.scatter(
                    arm["acc_pct"],
                    idx + dy,
                    marker=mkr,
                    s=44,
                    facecolor=col,
                    edgecolor=theme["bg"],
                    linewidth=0.65,
                    alpha=0.92,
                    zorder=5,
                )

            rng_lbl = f"{vmax:.0f}%" if abs(vmax - vmin) < 0.2 else f"{vmin:.0f}–{vmax:.0f}%"
            rng_lbl += f" ({len(arms)})"
            ax_i.text(
                min(vmax + 2.5, 118.0),
                idx,
                rng_lbl,
                va="center",
                ha="left",
                fontsize=7.9,
                color=theme["ink"],
                bbox={"boxstyle": "square,pad=0.06", "fc": theme["bg"], "ec": "none", "alpha": 0.85},
                zorder=6,
            )

        ax_i.set_xlim(18, 121)
        ax_i.set_xticks([25, 50, 75, 100])
        ax_i.set_xticklabels(["25%", "50%", "75%", "100%"], fontsize=8.4)
        ax_i.set_xlabel("Acceptable Rate (%)", fontsize=8.9, labelpad=6)
        ax_i.set_title(p_title, fontsize=10.0, fontweight="bold", color=theme["ink"], loc="left", pad=19)
        _style_axis(ax_i, theme, grid_axis="x", hide_left=is_pure_panel)

    ax_c.invert_yaxis()
    ax_c.set_yticks(fy)
    ax_c.set_yticklabels([f"{e['label']} ({len(e['arms'])}c)" for e in frontier_entries], fontsize=8.6)

    leg_items = [
        Line2D([0], [0], color=theme["strata"], linestyle="--", linewidth=1.35, label="STRATA v2 Reference (0.8B)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=theme["think"], markeredgecolor=theme["bg"], markersize=7.2, label="Think / High Effort"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=theme["nothink"], markeredgecolor=theme["bg"], markersize=6.8, label="NoThink / Low Effort"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=theme["instruct"], markeredgecolor=theme["bg"], markersize=6.5, label="Instruct"),
        Line2D([0], [0], color=theme["range_bg"], linewidth=6.0, label="Min–Max Span"),
    ]
    fig.legend(handles=leg_items, loc="upper left", bbox_to_anchor=(0.04, 0.895), ncol=5, frameon=False, fontsize=8.0)
    _add_header(
        fig,
        theme,
        f"{prefix}Accuracy Across Decoding Configurations",
        "Complete N = 19 runs over T ∈ {0, 1} × Thinking × {Max4096, NoLimit}; labels indicate min–max% (run count).",
    )
    p1 = out_dir / f"fig1_accuracy_comparison{suffix}.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p1)

    # Fig 2: Range vs. Parameter Scale
    fig, ax_sc = plt.subplots(figsize=(11.8, 5.6), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.81, bottom=0.13)

    sc_items: List[Tuple[float, float, str, bool, float]] = []
    for e in entries:
        px = max(float(e["best"]["params_b"]), 0.5)
        vals = [float(a["acc_pct"]) for a in e["arms"]]
        vmin, vmax = min(vals), max(vals)
        if e["is_strata"]:
            sz = 280.0
            ax_sc.scatter(px, vmax, marker="*", s=sz, c=theme["strata"], edgecolors=theme["strata_dark"], lw=0.95, zorder=6)
        else:
            sz = 90.0
            ax_sc.plot([px, px], [vmin, vmax], color=theme["pure_light"], linewidth=2.0, zorder=3)
            ax_sc.scatter(px, vmin, marker="o", s=50, facecolor=theme["bg"], edgecolor=theme["pure"], lw=1.3, zorder=4)
            ax_sc.scatter(px, vmax, marker="o", s=sz, facecolor=theme["ctx"], edgecolor=theme["ctx_dark"], lw=0.9, zorder=5)
        lbl = f"{e['scatter_label']}\n{vmax:.0f}%" if e["is_strata"] else f"{e['scatter_label']}\n{vmax:.0f}% (min {vmin:.0f}%)"
        sc_items.append((px, vmax, lbl, e["is_strata"], sz))

    if strata_line is not None:
        ax_sc.axhline(strata_line, color=theme["strata_border"], linestyle=":", linewidth=1.1, zorder=1)
    max_px = max((x for x, _, _, _, _ in sc_items), default=100.0)
    min_px = min((x for x, _, _, _, _ in sc_items), default=0.8)
    ax_sc.set_xscale("log")
    ax_sc.set_xlim(min_px * 0.48, max(max_px * 2.1, 180.0))
    ax_sc.set_ylim(18, 112)
    pticks = [t for t in (1, 3, 10, 30, 100, 300, 500) if min_px * 0.5 <= t <= max(max_px * 2.1, 180.0)]
    if pticks:
        ax_sc.set_xticks(pticks)
        ax_sc.set_xticklabels([f"{t}B" for t in pticks], fontsize=8.4)
    ax_sc.set_xlabel("Model Total Parameters (Billions, log scale)", fontsize=9.0, labelpad=6)
    ax_sc.set_ylabel("Acceptable Rate (%)", fontsize=9.0, labelpad=6)
    _style_axis(ax_sc, theme, grid_axis="both")
    _place_direct_scatter_labels(fig, ax_sc, theme, sc_items)

    handles_2 = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=theme["strata"], markeredgecolor=theme["strata_dark"], markersize=11, label="STRATA v2 Reference (0.8B)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=theme["ctx"], markeredgecolor=theme["ctx_dark"], markersize=7.5, label="Best Complete Configuration"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=theme["bg"], markeredgecolor=theme["pure"], markersize=6.8, label="Worst Complete Configuration"),
    ]
    fig.legend(handles=handles_2, loc="upper left", bbox_to_anchor=(0.07, 0.90), ncol=3, frameon=False, fontsize=8.2)
    _add_header(
        fig,
        theme,
        f"{prefix}Accuracy Range vs. Parameter Scale",
        "Vertical segments span worst-to-best complete N = 19 runs per model family.",
        x=0.07,
    )
    p2 = out_dir / f"fig2_pareto_scale{suffix}.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p2)

    # Fig 3: Token & Latency Dispersion
    all_rows_3 = ([strata_ref] if strata_ref else []) + frontier_entries
    y3 = np.arange(len(all_rows_3))
    fig_h3 = max(4.8, 0.52 * len(all_rows_3) + 1.8)
    fig, (ax_tk, ax_lt) = plt.subplots(1, 2, figsize=(14.6, fig_h3), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.13, wspace=0.10)

    _draw_strata_divider(ax_tk, all_rows_3, theme)
    _draw_strata_divider(ax_lt, all_rows_3, theme)

    max_tk = max((float(a["total_tokens"]) / 1000.0 for e in all_rows_3 for a in e["arms"]), default=100.0)
    max_lt = max((float(a["mean_wall_ms"]) / 1000.0 for e in all_rows_3 for a in e["arms"]), default=30.0)
    for idx, e in enumerate(all_rows_3):
        tks = sorted(float(a["total_tokens"]) / 1000.0 for a in e["arms"])
        lts = sorted(float(a["mean_wall_ms"]) / 1000.0 for a in e["arms"])
        col = theme["strata"] if e["is_strata"] else theme["ctx"]
        col_l = theme["strata"] if e["is_strata"] else theme["latency"]
        med_t = float(np.median(tks))
        med_l = float(np.median(lts))

        ax_tk.plot([tks[0], tks[-1]], [idx, idx], color=theme["range_bg"], linewidth=6.0, solid_capstyle="round", zorder=2)
        ax_tk.scatter(med_t, idx, s=68 if not e["is_strata"] else 115, marker="D" if e["is_strata"] else "o", color=col, zorder=4)
        t_lbl = f"{med_t:.1f}K" if len(tks) == 1 else f"{tks[0]:.0f}–{tks[-1]:.0f}K (med {med_t:.0f}K)"
        ax_tk.annotate(
            t_lbl, xy=(tks[-1], idx), xycoords="data", xytext=(9, 0), textcoords="offset points",
            va="center", ha="left", fontsize=7.9,
            fontweight="bold" if e["is_strata"] else "normal",
            color=theme["strata_dark"] if e["is_strata"] else theme["ink"],
        )

        ax_lt.plot([lts[0], lts[-1]], [idx, idx], color=theme["range_bg"], linewidth=6.0, solid_capstyle="round", zorder=2)
        ax_lt.scatter(med_l, idx, s=68 if not e["is_strata"] else 115, marker="D" if e["is_strata"] else "o", color=col_l, zorder=4)
        l_lbl = f"{med_l:.1f}s" if len(lts) == 1 else f"{lts[0]:.0f}–{lts[-1]:.0f}s (med {med_l:.0f}s)"
        ax_lt.annotate(
            l_lbl, xy=(lts[-1], idx), xycoords="data", xytext=(9, 0), textcoords="offset points",
            va="center", ha="left", fontsize=7.9,
            fontweight="bold" if e["is_strata"] else "normal",
            color=theme["strata_dark"] if e["is_strata"] else theme["ink"],
        )

    ax_tk.invert_yaxis()
    ax_tk.set_yticks(y3)
    ax_tk.set_yticklabels([e["label"] for e in all_rows_3], fontsize=8.6)
    for ytick, e in zip(ax_tk.get_yticklabels(), all_rows_3):
        if e["is_strata"]:
            ytick.set_fontweight("bold")
            ytick.set_color(theme["strata_dark"])
    ax_tk.set_xlim(-6, max_tk * 1.44)
    ax_lt.set_xlim(-4, max_lt * 1.42)
    ax_tk.set_xlabel("Total Suite Tokens (K)", fontsize=8.9, labelpad=6)
    ax_lt.set_xlabel("Mean Task Latency (s)", fontsize=8.9, labelpad=6)
    ax_tk.set_title("Token Usage (Min–Max & Median)", fontsize=9.8, fontweight="bold", loc="left", pad=8)
    ax_lt.set_title("Task Latency (Min–Max & Median)", fontsize=9.8, fontweight="bold", loc="left", pad=8)
    _style_axis(ax_tk, theme, grid_axis="x")
    _style_axis(ax_lt, theme, grid_axis="x", hide_left=True)

    _add_header(
        fig,
        theme,
        f"{prefix}Token Usage and Latency Dispersion",
        "Horizontal spans cover min-to-max across all complete decoding runs; points show medians.",
    )
    p3 = out_dir / f"fig3_resources_and_latency{suffix}.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p3)

    # Fig 4: Sequential Single-Hue Per-Task Robustness Heatmap
    task_meta: Dict[str, Tuple[str, str]] = {}
    for e in all_rows_3:
        for a in e["arms"]:
            for tid, rec in a["by_task"].items():
                if tid not in task_meta or rec.get("expected"):
                    task_meta[tid] = (str(rec.get("expected") or ""), str(rec.get("group") or "other"))
    col_ids = _order_tasks_semantically(task_meta)

    if col_ids:
        mat = np.zeros((len(all_rows_3), len(col_ids)))
        cell_texts: List[List[str]] = [["" for _ in col_ids] for _ in all_rows_3]
        cell_colors: List[List[str]] = [[theme["ink"] for _ in col_ids] for _ in all_rows_3]
        thr = float(theme["seq_threshold"])
        high_txt = "#FFFFFF" if theme_name == "light" else "#0D1117"
        for i, e in enumerate(all_rows_3):
            for j, tid in enumerate(col_ids):
                recs = [a["by_task"][tid] for a in e["arms"] if tid in a["by_task"]]
                if not recs:
                    mat[i, j] = -1.0
                    cell_texts[i][j] = "—"
                    cell_colors[i][j] = theme["muted"]
                else:
                    pct = 100.0 * sum(1 for r in recs if r["acceptable"]) / len(recs)
                    mat[i, j] = pct
                    cell_texts[i][j] = f"{pct:.0f}"
                    cell_colors[i][j] = high_txt if pct >= thr else theme["ink"]

        row_lbls = [f"{e['label']} ({len(e['arms'])}c)" if not e["is_strata"] else e["label"] for e in all_rows_3]
        row_masks = [e["is_strata"] for e in all_rows_3]
        mean_scores = [f"{np.mean([mat[i, j] for j in range(len(col_ids)) if mat[i, j] >= 0]):.0f}%" for i in range(len(all_rows_3))]

        fig_w = max(13.6, 0.62 * len(col_ids) + 4.2)
        fig_h4 = max(5.2, 0.48 * len(all_rows_3) + 2.2)
        fig, ax_hm = plt.subplots(figsize=(fig_w, fig_h4), dpi=200)
        fig.subplots_adjust(left=0.23, right=0.92, top=0.77, bottom=0.19)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("strata_seq", theme["seq_cmap"], N=256)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=100.0)
        _draw_grouped_matrix(
            fig,
            ax_hm,
            theme,
            mat,
            cell_texts,
            cell_colors,
            row_lbls,
            row_masks,
            mean_scores,
            col_ids,
            task_meta,
            cmap=cmap,
            norm=norm,
            header_title=f"{prefix}Per-Task Acceptable Rate Across Decoding Runs (%)",
            header_subtitle="Cell integers report % of complete configurations acceptable on each task (right column: mean across 19 tasks).",
        )
        p4 = out_dir / f"fig4_task_heatmap{suffix}.png"
        fig.savefig(p4, dpi=200, bbox_inches="tight", pad_inches=0.20)
        plt.close(fig)
        generated.append(p4)

    return generated


# ---------------------------------------------------------------------------
# Standard Layout (Backward Compatible with BFCL v3, Unit Tests & Ad-Hoc Runs)
# ---------------------------------------------------------------------------


def _select_pareto_foreground(
    clusters: Dict[Tuple[float, float], List[Dict[str, Any]]],
    max_badges: int = 22,
) -> List[Tuple[Tuple[float, float], List[Dict[str, Any]]]]:
    if len(clusters) <= 28:
        return sorted(clusters.items(), key=lambda kv: (kv[0][0], -kv[0][1]))

    by_family: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
    for (_, _), members in clusters.items():
        for m in members:
            family = _family_key(m["display"])
            px = round(max(m["params_b"], 0.5), 2)
            by_family.setdefault((px, family), []).append(m)

    keep_coords: set[Tuple[float, float]] = set()
    for (px, _), grp in sorted(by_family.items()):
        skill_arms = [m for m in grp if not _is_pure_arm(m["display"])]
        pure_arms = [m for m in grp if _is_pure_arm(m["display"])]
        if skill_arms:
            best_s = max(skill_arms, key=_arm_score_key)
            keep_coords.add((px, round(best_s["acc_pct"], 1)))
        if pure_arms:
            best_p = max(pure_arms, key=_arm_score_key)
            keep_coords.add((px, round(best_p["acc_pct"], 1)))
        worst_all = min(grp, key=_arm_score_key)
        keep_coords.add((px, round(worst_all["acc_pct"], 1)))

    return [
        (coord, clusters[coord])
        for coord in sorted(keep_coords, key=lambda c: (c[0], -c[1]))
        if coord in clusters
    ][:max_badges]


def _place_numbered_badges(
    ax: Any,
    theme: Dict[str, Any],
    items: List[Tuple[int, Tuple[float, float], List[Dict[str, Any]]]],
    all_points: List[Tuple[float, float]],
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> None:
    lo_x = math.log10(max(x_limits[0], 1e-3))
    hi_x = math.log10(max(x_limits[1], x_limits[0] * 1.1))
    span_x = max(hi_x - lo_x, 1e-6)
    lo_y, hi_y = y_limits
    span_y = max(hi_y - lo_y, 1.0)

    def to_uv(px: float, py: float) -> Tuple[float, float]:
        return (math.log10(max(px, 1e-3)) - lo_x) / span_x, (py - lo_y) / span_y

    def to_xy(u: float, v: float) -> Tuple[float, float]:
        return 10.0 ** (lo_x + u * span_x), lo_y + v * span_y

    point_uvs = [to_uv(px, py) for px, py in all_points]
    placed_badges: List[Tuple[float, float]] = []
    angles_deg = [90, 55, 125, 20, 160, -55, -125, -20, -160, -90, 0, 180, 38, 142, -38, -142]
    radii = [0.050, 0.075, 0.105, 0.135, 0.170, 0.210]

    for badge_num, (px, py), members in items:
        u0, v0 = to_uv(px, py)
        best_uv = (min(max(u0 + 0.05, 0.04), 0.96), min(max(v0 + 0.05, 0.05), 0.95))
        best_score = -1e9
        found_clean = False
        for rad in radii:
            for deg in angles_deg:
                ang = math.radians(deg)
                cu = u0 + rad * math.cos(ang)
                cv = v0 + rad * math.sin(ang)
                if not (0.04 <= cu <= 0.96 and 0.06 <= cv <= 0.95):
                    continue
                pt_clear = all(((cu - pu) / 0.044) ** 2 + ((cv - pv) / 0.052) ** 2 >= 1.0 for pu, pv in point_uvs)
                bd_clear = all(abs(cu - bu) >= 0.058 or abs(cv - bv) >= 0.062 for bu, bv in placed_badges)
                if pt_clear and bd_clear:
                    best_uv = (cu, cv)
                    found_clean = True
                    break
                min_d = min(
                    [math.hypot(cu - bu, cv - bv) for bu, bv in placed_badges]
                    + [math.hypot(cu - pu, cv - pv) for pu, pv in point_uvs]
                    + [1.0]
                )
                score = min_d - 0.15 * rad
                if score > best_score:
                    best_score = score
                    best_uv = (cu, cv)
            if found_clean:
                break

        placed_badges.append(best_uv)
        tx, ty = to_xy(best_uv[0], best_uv[1])
        has_strata = any("STRATA" in m["display"] for m in members)
        ec = theme["strata_border"] if has_strata else theme["muted"]
        tc = theme["strata_dark"] if has_strata else theme["ink"]
        ax.annotate(
            str(badge_num),
            xy=(px, py),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=tc,
            zorder=7,
            arrowprops={"arrowstyle": "-", "color": theme["pure_light"], "lw": 0.85, "shrinkA": 2, "shrinkB": 4},
            bbox={"boxstyle": "circle,pad=0.20", "fc": theme["bg"], "ec": ec, "lw": 0.9, "alpha": 0.95},
        )


def _generate_standard_plots(
    summary: Dict[str, Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    theme_name: str = "light",
    suffix: str = "",
) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    theme = THEMES[theme_name]
    _configure_rcparams(plt, theme)
    generated: List[Path] = []
    models = sorted(summary.values(), key=lambda m: (0 if "STRATA" in m["display"] else 1, -m["acc_pct"], m["params_b"], m["display"]))
    if not models:
        return generated

    labels = [_short_plot_label(m["display"]) for m in models]
    y = np.arange(len(labels))

    strict_vals = [m["strict_pct"] for m in models]
    acc_vals = [m["acc_pct"] for m in models]
    strict_err = [
        [max(0.0, m["strict_pct"] - m["strict_ci_lo"]) for m in models],
        [max(0.0, m["strict_ci_hi"] - m["strict_pct"]) for m in models],
    ]
    acc_err = [
        [max(0.0, m["acc_pct"] - m["acc_ci_lo"]) for m in models],
        [max(0.0, m["acc_ci_hi"] - m["acc_pct"]) for m in models],
    ]
    fig_h = max(4.6, 0.44 * len(models) + 1.8)
    fig, ax = plt.subplots(figsize=(12.6, fig_h), dpi=180)
    h = 0.36
    ax.barh(
        y + h / 2, strict_vals, xerr=strict_err, capsize=2.5,
        error_kw={"elinewidth": 0.9, "ecolor": theme["muted"]},
        height=h, label="Strict Passed (%, 95% Wilson CI)", color=theme["ctx"], zorder=3,
    )
    ax.barh(
        y - h / 2, acc_vals, xerr=acc_err, capsize=2.5,
        error_kw={"elinewidth": 0.9, "ecolor": theme["ink"]},
        height=h, label="Acceptable Outcome (%, 95% Wilson CI)", color=theme["syn"], zorder=3,
    )
    for idx, m in enumerate(models):
        x_txt = max(m["acc_pct"], m["acc_ci_hi"]) + 2.0
        ax.text(x_txt, idx - h / 2, f"{m['acc_pct']:.1f}%", va="center", ha="left", fontsize=8.0, color=theme["ink"])
    ax.invert_yaxis()
    ax.set_xlabel("Rate (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlim(0, 118)
    ax.set_title(f"{prefix}Overall Benchmark Accuracy (95% Wilson CI)", pad=26, fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, frameon=False, fontsize=9.0)
    _style_axis(ax, theme, grid_axis="x")
    p1 = out_dir / f"fig1_accuracy_comparison{suffix}.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p1)

    clusters: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for m in models:
        px = round(max(m["params_b"], 0.5), 2)
        py = round(m["acc_pct"], 1)
        clusters.setdefault((px, py), []).append(m)

    max_px = max((k[0] for k in clusters), default=10.0)
    min_px = min((k[0] for k in clusters), default=0.5)

    if len(clusters) <= 14:
        fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=180)
        sc_items: List[Tuple[float, float, str, bool, float]] = []
        base_pts: List[Tuple[float, float]] = []
        for (px, py), members in sorted(clusters.items()):
            rep = max(members, key=lambda m: (1 if "STRATA" in m["display"] else (0 if _is_pure_arm(m["display"]) else 0.5)))
            marker, color, ec, size, zord = _arm_marker_style(rep["display"], theme)
            ax.scatter(px, py, s=size, color=color, edgecolors=ec, marker=marker, zorder=zord)
            is_s = any("STRATA" in m["display"] for m in members)
            if not is_s:
                base_pts.append((px, py))
            lbl = "\n".join(f"{_short_plot_label(m['display'])} ({m['acc_pct']}%)" for m in members)
            sc_items.append((px, py, lbl, is_s, float(size)))
        env_xs, env_ys = _monotone_pareto_envelope(base_pts)
        if len(env_xs) >= 2:
            ax.plot(env_xs, env_ys, linestyle="--", color=theme["pure_light"], linewidth=1.25, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(min_px * 0.55, max(max_px * 2.4, 22.0))
        ax.set_ylim(-6, 110)
        ax.set_xlabel("Model Total Parameters (Billions, log scale)")
        ax.set_ylabel("Acceptable Rate (%)")
        ax.set_title(f"{prefix}Reliability vs. Model Scale", pad=12, fontweight="bold", loc="left")
        _style_axis(ax, theme, grid_axis="both")
        _place_direct_scatter_labels(fig, ax, theme, sc_items)
    else:
        fg_clusters = _select_pareto_foreground(clusters)
        key_lines: List[str] = []
        badged_items: List[Tuple[int, Tuple[float, float], List[Dict[str, Any]]]] = []
        for idx_num, (coord, members) in enumerate(fg_clusters, 1):
            badged_items.append((idx_num, coord, members))
            for sub_i, m in enumerate(sorted(members, key=lambda m: m["display"])):
                prefix_tag = f"[{idx_num:2d}] " if sub_i == 0 else "     "
                key_lines.append(f"{prefix_tag}{_short_plot_label(m['display'])} ({m['acc_pct']}%)")

        fig_h2 = max(7.0, 0.255 * len(key_lines) + 1.4)
        fig = plt.figure(figsize=(16.0, fig_h2), dpi=180)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.05], wspace=0.04)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.10)
        ax = fig.add_subplot(gs[0, 0])
        ax_key = fig.add_subplot(gs[0, 1])
        ax_key.axis("off")

        fg_set = {coord for _, coord, _ in badged_items}
        for (px, py), members in sorted(clusters.items()):
            if (px, py) not in fg_set:
                marker, color, ec, _, _ = _arm_marker_style(members[0]["display"], theme)
                ax.scatter(px, py, s=52, color=color, edgecolors=ec, marker=marker, alpha=0.28, zorder=2)

        non_strata_pts = [
            (round(max(m["params_b"], 0.5), 2), float(m["acc_pct"]))
            for m in models
            if "STRATA" not in m["display"]
        ]
        env_xs, env_ys = _monotone_pareto_envelope(non_strata_pts)
        if len(env_xs) >= 2:
            ax.plot(env_xs, env_ys, linestyle="--", color=theme["pure_light"], linewidth=1.2, zorder=1)
        strata_peaks = [m["acc_pct"] for m in models if "STRATA" in m["display"]]
        if strata_peaks:
            ax.axhline(max(strata_peaks), linestyle=":", color=theme["strata_border"], linewidth=1.2, alpha=0.75, zorder=1)

        for _, (px, py), members in badged_items:
            rep = next((m for m in members if "STRATA" in m["display"]), None)
            if rep is None:
                rep = next((m for m in members if not _is_pure_arm(m["display"])), members[0])
            marker, color, ec, size, zord = _arm_marker_style(rep["display"], theme)
            ax.scatter(px, py, s=size, color=color, edgecolors=ec, linewidths=0.9, marker=marker, zorder=zord)

        x_lim = (min_px * 0.65, max(max_px * 1.65, 160.0))
        y_lim = (-4.0, 108.0)
        ax.set_xscale("log")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        _place_numbered_badges(ax, theme, badged_items, [coord for _, coord, _ in badged_items], x_lim, y_lim)
        ax.set_xlabel("Model Total Parameters (Billions, log scale)")
        ax.set_ylabel("Acceptable Rate (%)")
        ax.set_title(f"{prefix}Reliability vs. Parameter Scale", pad=10, fontsize=11.5, fontweight="bold", loc="left")
        _style_axis(ax, theme, grid_axis="both")

        legend_handles = [
            Line2D([0], [0], marker="*", color="none", label="STRATA (0.8B)", markerfacecolor=theme["strata"], markeredgecolor=theme["strata_dark"], markersize=11),
            Line2D([0], [0], marker="o", color="none", label="Context (SKILL.md / ReAct)", markerfacecolor=theme["ctx"], markeredgecolor=theme["ctx_dark"], markersize=7.5),
            Line2D([0], [0], marker="^", color="none", label="Pure Weights (No Docs)", markerfacecolor=theme["pure"], markeredgecolor=theme["ink"], markersize=7.5),
            Line2D([0], [0], color=theme["pure_light"], linestyle="--", linewidth=1.2, label="Baseline Pareto Frontier"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.0)

        ax_key.text(0.01, 0.99, "Indexed Clusters", va="top", ha="left", fontsize=10.0, fontweight="bold", transform=ax_key.transAxes)
        y_step = 0.94 / max(len(key_lines), 1)
        font_sz = 7.6 if len(key_lines) <= 30 else 7.0
        for row_i, line_txt in enumerate(key_lines):
            y_pos = 0.945 - row_i * y_step
            is_s = "STRATA" in line_txt
            ax_key.text(
                0.01, y_pos, line_txt, va="top", ha="left", transform=ax_key.transAxes,
                fontsize=font_sz, fontfamily="DejaVu Sans Mono",
                fontweight="bold" if is_s else "normal",
                color=theme["strata_dark"] if is_s else theme["ink"],
            )

    p2 = out_dir / f"fig2_pareto_scale{suffix}.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    generated.append(p2)

    vram_models = [m for m in models if m["peak_vram_mb"] > 0 or m["mean_wall_ms"] > 0]
    if vram_models:
        fig_h3 = max(4.6, 0.40 * len(vram_models) + 1.6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, fig_h3), dpi=180, sharey=True)
        vl = [_short_plot_label(m["display"]) for m in vram_models]
        vx = np.arange(len(vl))
        vr_gb = [m["peak_vram_mb"] / 1024.0 for m in vram_models]
        ax1.barh(vx, vr_gb, color=theme["vram"], zorder=3)
        for idx, gb in enumerate(vr_gb):
            if gb <= 0:
                ax1.text(0.25, idx, "Cloud API", va="center", ha="left", fontsize=7.8, style="italic", color=theme["muted"])
            else:
                ax1.text(gb + max(max(vr_gb), 1.0) * 0.02, idx, f"{gb:.1f} GiB", va="center", ha="left", fontsize=7.8)
        ax1.invert_yaxis()
        ax1.set_yticks(vx)
        ax1.set_yticklabels(vl, fontsize=8.4)
        ax1.set_xlim(0, max(max(vr_gb, default=4.0) * 1.18, 4.0))
        ax1.set_xlabel("Peak GPU VRAM (GiB)")
        ax1.set_title("GPU Memory Footprint", fontweight="bold", loc="left")
        _style_axis(ax1, theme, grid_axis="x")

        lat_s = [m["mean_wall_ms"] / 1000.0 for m in vram_models]
        ax2.barh(vx, lat_s, color=theme["latency"], zorder=3)
        for idx, sec in enumerate(lat_s):
            ax2.text(sec + max(max(lat_s), 1.0) * 0.02, idx, f"{sec:.1f}s", va="center", ha="left", fontsize=7.8)
        ax2.tick_params(axis="y", labelleft=False)
        ax2.set_xlim(0, max(max(lat_s, default=1.0) * 1.18, 1.0))
        ax2.set_xlabel("Mean End-to-End Latency (s / task)")
        ax2.set_title("Mean Execution Time per Task", fontweight="bold", loc="left")
        _style_axis(ax2, theme, grid_axis="x", hide_left=True)
        p3 = out_dir / f"fig3_resources_and_latency{suffix}.png"
        fig.savefig(p3, dpi=180, bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        generated.append(p3)

    task_meta = {
        tid: (str(rec.get("expected") or ""), str(rec.get("group") or "other"))
        for m in models
        for tid, rec in m["by_task"].items()
    }
    all_tasks = _order_tasks_semantically(task_meta)
    if all_tasks:
        mat = np.zeros((len(models), len(all_tasks)))
        cell_texts: List[List[str]] = [["" for _ in all_tasks] for _ in models]
        cell_colors: List[List[str]] = [[theme["ink"] for _ in all_tasks] for _ in models]
        hi_txt = "#FFFFFF" if theme_name == "light" else "#0D1117"
        for i, m in enumerate(models):
            for j, tid in enumerate(all_tasks):
                rec = m["by_task"].get(tid)
                exp = task_meta[tid][0]
                if rec is None:
                    mat[i, j] = -1.0
                    cell_texts[i][j] = "·"
                    cell_colors[i][j] = theme["light_ink"]
                elif not rec["acceptable"]:
                    mat[i, j] = 0.0
                    cell_texts[i][j] = "✗"
                    cell_colors[i][j] = hi_txt
                elif exp in ("refuse", "clarify") and rec["passed"]:
                    mat[i, j] = 1.0
                    cell_texts[i][j] = "~"
                    cell_colors[i][j] = "#1F2937"
                elif rec["passed"]:
                    mat[i, j] = 3.0
                    cell_texts[i][j] = "✓"
                    cell_colors[i][j] = hi_txt
                else:
                    mat[i, j] = 2.0
                    cell_texts[i][j] = "⊘"
                    cell_colors[i][j] = hi_txt

        fig_w = (
            min(22.0, max(10.0, 0.75 * len(all_tasks)))
            if len(all_tasks) <= 35
            else min(20.0, max(12.0, 0.16 * len(all_tasks)))
        )
        fig, ax = plt.subplots(figsize=(fig_w, max(4.8, 0.38 * len(models) + 2.0)), dpi=180)
        cmap = matplotlib.colors.ListedColormap([theme["na"], theme["fail"], theme["fallback"], theme["safety"], theme["syn"]])
        norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        _draw_grouped_matrix(
            fig,
            ax,
            theme,
            mat,
            cell_texts,
            cell_colors,
            labels,
            ["STRATA" in m["display"] for m in models],
            [f"{m['acceptable']}/{m['trials']}" for m in models],
            all_tasks,
            task_meta,
            cmap=cmap,
            norm=norm,
            header_title=f"{prefix}Per-Task Outcomes",
            header_subtitle="✓ Pass | ⊘ Clean Refusal | ~ Vague Code Fallback | ✗ Fail | · N/A",
        )
        p4 = out_dir / f"fig4_task_heatmap{suffix}.png"
        fig.savefig(p4, dpi=180, bbox_inches="tight", pad_inches=0.20)
        plt.close(fig)
        generated.append(p4)

    return generated


def generate_plots(
    summary: Dict[str, Dict[str, Any]],
    out_dir: Path,
    *,
    title_prefix: str = "",
    layout: str = "standard",
    dark_variant: bool = False,
) -> List[Path]:
    """Generate benchmark visualizations using standard, headline, or ablation layouts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(title_prefix or "")
    fn = _generate_headline_plots if layout == "headline" else (_generate_ablation_plots if layout == "ablation" else _generate_standard_plots)
    paths = fn(summary, out_dir, prefix, theme_name="light", suffix="")
    if dark_variant:
        paths.extend(fn(summary, out_dir, prefix, theme_name="dark", suffix="_dark"))
    return paths


# ---------------------------------------------------------------------------
# Tabular Markdown & CSV Reports
# ---------------------------------------------------------------------------


def _format_vram_md(vram_mb: float, display: str) -> str:
    if vram_mb > 0:
        return f"{vram_mb / 1024.0:.1f} GiB"
    if any(tag in display for tag in ("(API)", "(OR)", "(SB)")):
        return "Cloud API"
    return "0.0"


def write_tables(
    summary: Dict[str, Dict[str, Any]],
    out_dir: Path,
    *,
    title_prefix: str = "",
    layout: str = "standard",
) -> Tuple[Path, Path]:
    """Write GFM-safe Markdown & CSV comparison summaries with split Synthesis vs Safety columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(
        summary.values(),
        key=lambda m: (0 if "STRATA" in m["display"] else 1, m["params_b"], -m["acc_pct"], m["display"]),
    )
    title_txt = f"# {title_prefix}STRATA Cross-Model Comparison Summary"
    denoms = sorted({m["tasks"] for m in models})
    md_lines = [title_txt, ""]
    if len(denoms) > 1:
        md_lines.extend([
            f"> **Warning:** Included arms span non-matched unique task denominators ({denoms}); compare identical denominators pairwise.",
            "",
        ])

    if layout in ("headline", "ablation"):
        entries = _consolidate_headline(summary)
        md_lines.extend([
            "## Headline Cross-Model Leaderboard (Best Complete Arm per Model)",
            "",
            "| Model | Params (Active) | Overall (95% CI) | Synthesis | Safety & Refusal | Pure Baseline | Tokens | Peak VRAM / Host | Latency |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for e in entries:
            b = e["best"]
            act_b = b.get("active_params_b", b["params_b"])
            p_str = f"{_fmt_b(b['params_b'])} ({_fmt_b(act_b)})" if 0 < act_b < b["params_b"] - 1e-6 else _fmt_b(b["params_b"])
            syn_str = f"{b['syn_passed']}/{b['syn_tasks']} ({b['syn_pct']}%)" if b.get("syn_tasks", 0) > 0 else "—"
            if b.get("safety_tasks", 0) > 0:
                ref_note = f" ({b['safety_refused']} ref)" if b["safety_refused"] < b["safety_acc"] else ""
                saf_str = f"{b['safety_acc']}/{b['safety_tasks']} ({b['safety_pct']}%){ref_note}"
            else:
                saf_str = "—"
            bp = e["best_pure"]
            pure_str = f"{bp['acceptable']}/{bp['trials']} ({bp['acc_pct']}%)" if bp is not None else "—"
            vr_str = f"{b['peak_vram_mb'] / 1024.0:.1f} GiB" if b["peak_vram_mb"] > 0 else "Cloud API"
            lat_str = f"{b['mean_wall_ms'] / 1000.0:.1f}s"
            md_lines.append(
                "| " + " | ".join([
                    _md_cell(e["short_name"]),
                    _md_cell(p_str),
                    _md_cell(f"{b['acceptable']}/{b['trials']} ({b['acc_pct']}%, [{b['acc_ci_lo']}-{b['acc_ci_hi']}%])"),
                    _md_cell(syn_str),
                    _md_cell(saf_str),
                    _md_cell(pure_str),
                    _md_cell(f"{b['total_tokens']:,}"),
                    _md_cell(vr_str),
                    _md_cell(lat_str),
                ]) + " |"
            )
        md_lines.extend(["", "## All Evaluated Arms", ""])

    md_lines.extend([
        "| Model / Architecture | Params (B) | Tasks | Strict Passed (95% CI) | Acceptable (95% CI) | Mean Wall (s) | Total Tokens | Peak HWM (MB) | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "model", "params_b", "active_params_b", "seeds", "tasks", "trials", "passed", "acceptable",
            "strict_pct", "strict_ci_lo", "strict_ci_hi",
            "acc_pct", "seed_std_pct", "acc_ci_lo", "acc_ci_hi",
            "syn_passed", "syn_tasks", "syn_pct",
            "safety_acc", "safety_refused", "safety_tasks", "safety_pct",
            "mean_wall_ms", "total_tokens", "peak_hwm_mb", "peak_vram_mb",
        ])
        for m in models:
            trials = m.get("trials", m["tasks"])
            task_cell = f"{m['tasks']} ({m.get('seeds', 1)}s)" if m.get("seeds", 1) > 1 else str(m["tasks"])
            std_suffix = f" ±{m['seed_std_pct']}%" if m.get("seeds", 1) > 1 else ""
            md_lines.append(
                "| " + " | ".join([
                    _md_cell(m["display"]),
                    _md_cell(m["params_b"]),
                    _md_cell(task_cell),
                    _md_cell(f"{m['passed']}/{trials} ({m['strict_pct']}%, [{m['strict_ci_lo']}-{m['strict_ci_hi']}%])"),
                    _md_cell(f"{m['acceptable']}/{trials} ({m['acc_pct']}%{std_suffix}, [{m['acc_ci_lo']}-{m['acc_ci_hi']}%])"),
                    _md_cell(f"{m['mean_wall_ms'] / 1000.0:.1f}s"),
                    _md_cell(f"{m['total_tokens']:,}"),
                    _md_cell(f"{m['peak_hwm_mb']:.1f}"),
                    _md_cell(_format_vram_md(m["peak_vram_mb"], m["display"])),
                ]) + " |"
            )
            writer.writerow([
                m["display"], m["params_b"], m.get("active_params_b", m["params_b"]),
                m.get("seeds", 1), m["tasks"], trials, m["passed"], m["acceptable"],
                m["strict_pct"], m["strict_ci_lo"], m["strict_ci_hi"],
                m["acc_pct"], m.get("seed_std_pct", 0.0), m["acc_ci_lo"], m["acc_ci_hi"],
                m.get("syn_passed", 0), m.get("syn_tasks", 0), m.get("syn_pct", 0.0),
                m.get("safety_acc", 0), m.get("safety_refused", 0), m.get("safety_tasks", 0), m.get("safety_pct", 0.0),
                m["mean_wall_ms"], m["total_tokens"], m["peak_hwm_mb"], m["peak_vram_mb"],
            ])

    all_groups = sorted({g for m in models for g in (m.get("by_group") or {}).keys()})
    if all_groups:
        md_lines.extend([
            "",
            "## Per-Category Acceptable Rate Breakdown",
            "",
            "| Model / Architecture | " + " | ".join(_md_cell(g) for g in all_groups) + " |",
            "|---|" + "|".join("---:" for _ in all_groups) + "|",
        ])
        grp_csv_path = out_dir / "comparison_summary_by_group.csv"
        with open(grp_csv_path, "w", newline="", encoding="utf-8") as gfh:
            gwriter = csv.writer(gfh)
            gwriter.writerow(["model"] + all_groups)
            for m in models:
                cells_md: List[str] = []
                cells_csv: List[str] = []
                by_g = m.get("by_group") or {}
                for gname in all_groups:
                    gstat = by_g.get(gname)
                    if gstat is None:
                        cells_md.append("—")
                        cells_csv.append("")
                    else:
                        g_trials = gstat.get("trials", gstat["tasks"])
                        cells_md.append(_md_cell(f"{gstat['acceptable']}/{g_trials} ({gstat['acc_pct']}%)"))
                        cells_csv.append(str(gstat["acc_pct"]))
                md_lines.append(f"| {_md_cell(m['display'])} | " + " | ".join(cells_md) + " |")
                gwriter.writerow([m["display"]] + cells_csv)

    md_path = out_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate and visualize STRATA benchmark JSONs")
    ap.add_argument("inputs", nargs="+", help="JSON files or directories containing results_*.json")
    ap.add_argument("--out-dir", default="test_logs/visualizations")
    ap.add_argument("--title-prefix", default="", help="Optional prefix for summary titles and plot headers")
    ap.add_argument(
        "--clean-only",
        action="store_true",
        help="Skip partial runs or JSONs containing upstream API_ERROR / HTTP_ERROR records.",
    )
    ap.add_argument(
        "--layout",
        choices=("standard", "headline", "ablation"),
        default="standard",
        help="Visualization & table layout mode: standard, headline, or ablation.",
    )
    ap.add_argument(
        "--dark-variant",
        action="store_true",
        help="Also emit matching *_dark.png figures styled for GitHub Dark mode (#0D1117).",
    )
    args = ap.parse_args()

    json_paths: List[Path] = []
    for raw_in in args.inputs:
        p = Path(raw_in)
        if p.is_dir():
            json_paths.extend(sorted(p.rglob("results_*.json")))
        elif p.exists():
            json_paths.append(p)

    if not json_paths:
        raise SystemExit("No results_*.json files found in provided inputs.")

    rows = load_all_results(json_paths, clean_only=args.clean_only)
    summary = aggregate_by_model(rows)
    out_dir = Path(args.out_dir)
    md_path, csv_path = write_tables(summary, out_dir, title_prefix=args.title_prefix, layout=args.layout)
    plots = generate_plots(
        summary,
        out_dir,
        title_prefix=args.title_prefix,
        layout=args.layout,
        dark_variant=args.dark_variant,
    )
    print(md_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    for pl in plots:
        print(f"Plot: {pl}")


if __name__ == "__main__":
    main()
