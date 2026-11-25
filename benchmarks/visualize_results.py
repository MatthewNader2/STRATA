#!/usr/bin/env python3
"""
benchmarks/visualize_results.py - Aggregates STRATA local, Colab, and AI Studio
benchmark JSONs into Markdown/CSV reports and comparative Matplotlib visualizations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


KNOWN_METADATA: Dict[str, Dict[str, Any]] = {
    "omega_no_fb": {"display": "STRATA O+OMEGA (0.8B, No-FB)", "params_b": 0.8, "active_b": 0.8},
    "omega_with_fb": {"display": "STRATA O+OMEGA (0.8B, With-FB)", "params_b": 0.8, "active_b": 0.8},
    "omega": {"display": "STRATA O+OMEGA (0.8B, No-FB)", "params_b": 0.8, "active_b": 0.8},
    "pure_08b": {"display": "Pure Qwen3.5-0.8B", "params_b": 0.8, "active_b": 0.8},
    "pure_7b": {"display": "Pure Qwen2.5-Coder-7B", "params_b": 7.6, "active_b": 7.6},
}


def _normalize_legacy_record(r: Dict[str, Any], top_data: Dict[str, Any]) -> Dict[str, Any]:
    is_v2 = top_data.get("schema_version") == "strata_bench_v2"
    arm = str(r.get("arm") or top_data.get("arm") or "unknown")
    tid = str(r.get("task_id") or r.get("preset") or "unknown")
    group = str(r.get("group") or "other")
    model_id = str(r.get("model_id") or top_data.get("model_id") or arm)
    meta = KNOWN_METADATA.get(arm, {})
    display = str(r.get("model_display") or top_data.get("model_display") or meta.get("display") or model_id)
    params_b = float(r.get("params_b") or top_data.get("params_b") or meta.get("params_b") or 0.0)
    if not is_v2 and arm == "omega":
        display, model_id, params_b = "STRATA O+OMEGA (0.8B, No-FB)", "omega_no_fb", 0.8
    elif not is_v2 and arm == "pure_md":
        llm = str((top_data.get("init") or {}).get("resolved_llm") or "")
        if "0.8" in llm:
            display, model_id, params_b = "Pure Qwen3.5-0.8B", "pure_08b", 0.8
        elif "7B" in llm or "7b" in llm:
            display, model_id, params_b = "Pure Qwen2.5-Coder-7B", "pure_7b", 7.6

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
        expected = "clarify" if tid in ("vague_game", "vague_ml", "bound_ambiguous") else "synthesize"
        acceptable = passed or (expected == "clarify" and status in ("LINT_HALTED", "REFUSED_OK"))

    return {
        "model_key": model_id if is_v2 else (arm if arm in ("omega_no_fb", "omega_with_fb", "pure_08b", "pure_7b") else display),
        "display": display,
        "params_b": params_b,
        "task_id": tid,
        "group": group,
        "passed": passed,
        "acceptable": acceptable,
        "status": status,
        "wall_ms": float(r.get("wall_ms") or 0.0),
        "total_tokens": int(r.get("total_tokens") or 0),
        "hwm_mb": float((r.get("mem_mb") or {}).get("hwm_mb") or 0.0),
        "vram_mb": float((r.get("gpu_mb") or {}).get("total_used_mb") or 0.0),
    }


def load_all_results(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("records") or []:
            rows.append(_normalize_legacy_record(r, data))
    return rows


def aggregate_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["display"], []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for disp, items in grouped.items():
        # De-duplicate by task_id keeping latest entry passed in
        by_task = {it["task_id"]: it for it in items}
        vals = list(by_task.values())
        n = max(len(vals), 1)
        by_group: Dict[str, Dict[str, Any]] = {}
        grp_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for v in vals:
            grp_buckets.setdefault(v.get("group", "other"), []).append(v)
        for gname, gvals in sorted(grp_buckets.items()):
            gn = max(len(gvals), 1)
            gacc = sum(1 for v in gvals if v["acceptable"])
            gpass = sum(1 for v in gvals if v["passed"])
            by_group[gname] = {
                "tasks": len(gvals),
                "passed": gpass,
                "acceptable": gacc,
                "acc_pct": round(100.0 * gacc / gn, 1),
            }
        summary[disp] = {
            "display": disp,
            "params_b": max(v["params_b"] for v in vals),
            "tasks": len(vals),
            "passed": sum(1 for v in vals if v["passed"]),
            "acceptable": sum(1 for v in vals if v["acceptable"]),
            "acc_pct": round(100.0 * sum(1 for v in vals if v["acceptable"]) / n, 1),
            "strict_pct": round(100.0 * sum(1 for v in vals if v["passed"]) / n, 1),
            "mean_wall_ms": round(sum(v["wall_ms"] for v in vals) / n, 1),
            "total_tokens": sum(v["total_tokens"] for v in vals),
            "peak_hwm_mb": round(max(v["hwm_mb"] for v in vals), 1),
            "peak_vram_mb": round(max(v["vram_mb"] for v in vals), 1),
            "by_task": by_task,
            "by_group": by_group,
        }
    return summary


def generate_plots(summary: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    models = sorted(summary.values(), key=lambda m: (m["params_b"], m["display"]))
    if not models:
        return generated

    labels = [m["display"] for m in models]
    x = np.arange(len(labels))

    # 1. Accuracy bar chart (Strict Verified vs Acceptable)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.36
    ax.bar(x - w / 2, [m["strict_pct"] for m in models], width=w, label="Strict Passed (%)", color="#2b6cb0")
    ax.bar(x + w / 2, [m["acc_pct"] for m in models], width=w, label="Acceptable Outcome (%)", color="#2f855a")
    ax.set_ylabel("Rate (%)")
    ax.set_title("STRATA (0.8B) vs Pure LLMs — Overall Benchmark Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0, 112)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p1 = out_dir / "fig1_accuracy_comparison.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    generated.append(p1)

    # 2. Pareto frontier: Accuracy vs Parameters (log scale)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in models:
        is_strata = "STRATA" in m["display"]
        color = "#d69e2e" if is_strata else "#3182ce"
        marker = "*" if is_strata else "o"
        size = 260 if is_strata else 110
        ax.scatter(max(m["params_b"], 0.5), m["acc_pct"], s=size, color=color, marker=marker, zorder=4)
        ax.annotate(
            f"{m['display']} ({m['acc_pct']}%)",
            (max(m["params_b"], 0.5), m["acc_pct"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Model Total Parameters (Billions, log scale)")
    ax.set_ylabel("Acceptable Rate (%)")
    ax.set_title("Pareto Frontier: Code Synthesis Reliability vs Model Scale")
    ax.set_ylim(-5, 112)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    p2 = out_dir / "fig2_pareto_scale.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    generated.append(p2)

    # 3. Resource footprint & latency (VRAM + Mean Wall ms)
    vram_models = [m for m in models if m["peak_vram_mb"] > 0 or m["mean_wall_ms"] > 0]
    if vram_models:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vl = [m["display"] for m in vram_models]
        vx = np.arange(len(vl))
        ax1.barh(vx, [m["peak_vram_mb"] for m in vram_models], color="#805ad5")
        ax1.set_yticks(vx)
        ax1.set_yticklabels(vl, fontsize=8.5)
        ax1.set_xlabel("Peak GPU VRAM (MB)")
        ax1.set_title("GPU Memory Footprint")
        ax1.grid(axis="x", linestyle="--", alpha=0.35)

        ax2.barh(vx, [m["mean_wall_ms"] / 1000.0 for m in vram_models], color="#dd6b20")
        ax2.set_yticks(vx)
        ax2.set_yticklabels([""] * len(vl))
        ax2.set_xlabel("Mean End-to-End Latency (s / task)")
        ax2.set_title("Mean Execution Time per Task")
        ax2.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        p3 = out_dir / "fig3_resources_and_latency.png"
        fig.savefig(p3, dpi=160)
        plt.close(fig)
        generated.append(p3)

    # 4. Task-level pass/fail matrix heatmap
    all_tasks = sorted({tid for m in models for tid in m["by_task"].keys()})
    if all_tasks:
        mat = np.zeros((len(models), len(all_tasks)))
        for i, m in enumerate(models):
            for j, tid in enumerate(all_tasks):
                rec = m["by_task"].get(tid)
                if rec is None:
                    mat[i, j] = -1.0  # missing
                elif rec["passed"]:
                    mat[i, j] = 1.0
                elif rec["acceptable"]:
                    mat[i, j] = 0.65  # clean refusal
                else:
                    mat[i, j] = 0.0
        fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(all_tasks)), max(4.5, 0.55 * len(models) + 1.8)))
        cmap = matplotlib.colors.ListedColormap(["#e2e8f0", "#e53e3e", "#ecc94b", "#38a169"])
        bounds = [-1.5, -0.25, 0.3, 0.85, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([m["display"] for m in models], fontsize=9)
        ax.set_xticks(np.arange(len(all_tasks)))
        ax.set_xticklabels(all_tasks, rotation=38, ha="right", fontsize=8.5)
        ax.set_title("Per-Task Outcome Matrix (Green=Pass, Yellow=Clean Refusal, Red=Fail, Grey=Not Run)")
        fig.tight_layout()
        p4 = out_dir / "fig4_task_heatmap.png"
        fig.savefig(p4, dpi=160)
        plt.close(fig)
        generated.append(p4)

    return generated


def write_tables(summary: Dict[str, Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(summary.values(), key=lambda m: (m["params_b"], m["display"]))
    md_lines = [
        "# STRATA Cross-Model Comparison Summary",
        "",
        "| Model / Architecture | Params (B) | Tasks | Strict Passed | Acceptable | Mean Wall (ms) | Total Tokens | Peak HWM (MB) | Peak VRAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "params_b", "tasks", "passed", "acceptable", "strict_pct", "acc_pct", "mean_wall_ms", "total_tokens", "peak_hwm_mb", "peak_vram_mb"])
        for m in models:
            md_lines.append(
                f"| {m['display']} | {m['params_b']} | {m['tasks']} | "
                f"{m['passed']}/{m['tasks']} ({m['strict_pct']}%) | "
                f"{m['acceptable']}/{m['tasks']} ({m['acc_pct']}%) | "
                f"{m['mean_wall_ms']} | {m['total_tokens']} | {m['peak_hwm_mb']} | {m['peak_vram_mb']} |"
            )
            writer.writerow([
                m["display"], m["params_b"], m["tasks"], m["passed"], m["acceptable"],
                m["strict_pct"], m["acc_pct"], m["mean_wall_ms"], m["total_tokens"],
                m["peak_hwm_mb"], m["peak_vram_mb"],
            ])
    all_groups = sorted({g for m in models for g in (m.get("by_group") or {}).keys()})
    if all_groups:
        md_lines.extend([
            "",
            "## Per-Category Acceptable Rate Breakdown",
            "",
            "| Model / Architecture | " + " | ".join(all_groups) + " |",
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
                        cells_md.append(f"{gstat['acceptable']}/{gstat['tasks']} ({gstat['acc_pct']}%)")
                        cells_csv.append(str(gstat["acc_pct"]))
                md_lines.append(f"| {m['display']} | " + " | ".join(cells_md) + " |")
                gwriter.writerow([m["display"]] + cells_csv)

    md_path = out_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate and visualize STRATA benchmark JSONs")
    ap.add_argument("inputs", nargs="+", help="JSON files or directories containing results_*.json")
    ap.add_argument("--out-dir", default="test_logs/visualizations")
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

    rows = load_all_results(json_paths)
    summary = aggregate_by_model(rows)
    out_dir = Path(args.out_dir)
    md_path, csv_path = write_tables(summary, out_dir)
    plots = generate_plots(summary, out_dir)
    print(md_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    for pl in plots:
        print(f"Plot: {pl}")


if __name__ == "__main__":
    main()
