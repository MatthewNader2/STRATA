"""
src/cli_helpers.py - STRATA (Neuro-Symbolic Topological Lattice)
Track 2.3 & Track 3.3: Extracted CLI Diagnostics, Failure Surface Summary,
and Environment/Database Sync Helpers.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

try:
    from .diagnostics import DiagnosticTelemetry, GLOBAL_DIAGNOSTICS
except (ImportError, ValueError):
    from diagnostics import DiagnosticTelemetry, GLOBAL_DIAGNOSTICS


def render_failure_surface_summary(
    console: Console,
    ctx: Optional[Any] = None,
    router: Optional[Any] = None,
    gate: Optional[Any] = None,
) -> None:
    """
    Track 3.3: Renders the `--debug` Failure Surface Summary table reporting
    total swallowed warnings, caught exceptions, failed unifications, and
    fallback invocations recorded during pipeline execution.
    """
    telemetry = DiagnosticTelemetry()
    telemetry.merge(GLOBAL_DIAGNOSTICS)
    if ctx is not None and isinstance(getattr(ctx, "diagnostics", None), DiagnosticTelemetry):
        telemetry.merge(ctx.diagnostics)

    fb_reason = getattr(router, "last_fallback_reason", None) if router else None
    req_route = getattr(router, "last_requested_route", None) if router else None
    eff_route = getattr(router, "last_effective_route", None) if router else None
    if fb_reason:
        telemetry.record_fallback(
            "LatticeRouter.plan_path",
            str(req_route or "M?"),
            str(eff_route or "M0"),
            str(fb_reason),
        )

    refusal = getattr(getattr(router, "planner", None), "last_refusal", None) if router else None
    if isinstance(refusal, dict) and refusal.get("reason"):
        telemetry.record_refusal(
            "LatticePlanner.plan",
            f"{refusal.get('reason')} (coverage={refusal.get('coverage_fraction', 0.0):.1%}, "
            f"floor={refusal.get('coverage_floor', 0.85):.1%})",
        )

    unresolved = getattr(ctx, "unresolved_ports", []) if ctx else []
    for cid, pname in unresolved:
        telemetry.record_refusal("UnificationGate.unify_pipeline", "Unresolved required port", cell_id=cid, port_name=pname)

    summary = telemetry.summary()
    counts = summary["counts"]

    summary_table = Table(
        title="🛡️ Failure Surface & Diagnostic Telemetry Summary (--debug)",
        box=box.ROUNDED,
        expand=True,
        border_style="magenta",
    )
    summary_table.add_column("Diagnostic Category", style="bold cyan", ratio=2)
    summary_table.add_column("Count", justify="right", style="bold yellow", width=10)
    summary_table.add_column("Status / Impact", style="white", ratio=4)

    exc_cnt = counts.get("exception", 0)
    warn_cnt = counts.get("warning", 0)
    fb_cnt = counts.get("fallback", 0)
    ref_cnt = counts.get("refusal", 0)

    summary_table.add_row(
        "Caught Exceptions (Non-Silent)",
        str(exc_cnt),
        "[green]0 (Clean)[/green]" if exc_cnt == 0 else f"[yellow]{exc_cnt} logged to diagnostic trace[/yellow]",
    )
    summary_table.add_row(
        "Warnings & Unexpanded Macros",
        str(warn_cnt),
        "[green]0 (Clean)[/green]" if warn_cnt == 0 else f"[yellow]{warn_cnt} warning(s) recorded[/yellow]",
    )
    summary_table.add_row(
        "Route / Planner Fallbacks",
        str(fb_cnt),
        "[green]0 (Primary Route Held)[/green]" if fb_cnt == 0 else f"[bold yellow]{fb_cnt} fallback(s) triggered[/bold yellow]",
    )
    summary_table.add_row(
        "Unification / Coverage Refusals",
        str(ref_cnt),
        "[green]0 (All Constraints Satisfied)[/green]" if ref_cnt == 0 else f"[bold red]{ref_cnt} refusal(s) recorded[/bold red]",
    )

    console.print(summary_table)

    events = summary.get("events", [])
    if events:
        detail_table = Table(
            title="📋 Recorded Diagnostic Events (Latest 10)",
            box=box.SIMPLE_HEAD,
            expand=True,
            border_style="dim magenta",
        )
        detail_table.add_column("Kind", style="bold", width=10)
        detail_table.add_column("Subsystem / Site", style="cyan", width=30)
        detail_table.add_column("Type", style="yellow", width=20)
        detail_table.add_column("Detail", style="white", ratio=3)

        for ev in events[-10:]:
            loc = ev["site"]
            if ev.get("cell_id"):
                loc += f" [{ev['cell_id']}" + (f".{ev['port_name']}]" if ev.get("port_name") else "]")
            detail_table.add_row(
                ev["kind"].upper(),
                loc,
                ev["error_type"],
                ev["message"][:120],
            )
        console.print(detail_table)
    console.print("")
    GLOBAL_DIAGNOSTICS.clear()


__all__ = [
    "render_failure_surface_summary",
]
