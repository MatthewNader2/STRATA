"""
src/debug_panels.py - STRATA (Neuro-Symbolic Topological Lattice)

`--debug` / `/debug on` diagnostics that explain WHY a pipeline looks the way it
does, instead of only showing what it is. Every panel is built from telemetry the
engine records while it works (planner trace, tagging decisions, Layer-3 binding
trace) plus the corpus-derived coverage audit. Rendering never affects synthesis.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rich import box
from rich.table import Table


def _t(title: str, border: str = "cyan") -> Table:
    return Table(title=title, box=box.SIMPLE_HEAVY, expand=True, border_style=border, show_lines=False)


def render_planning_diagnostics(console: Any, router: Any, cells: List[Any], prompt: str) -> None:
    """Planner telemetry: clause anchor candidates, dropped/inserted anchors, tag decisions."""
    trace: List[Dict[str, Any]] = list(getattr(router, "last_route_trace", []) or [])
    if trace:
        cand = _t("🔬 Planner Anchor Evidence (per clause: recall = share of clause explained, name-prec = clause names the cell id, F = identity F)")
        for col in ("Clause", "Cell", "Recall", "Name-P", "Prec", "F", "Matched tokens", "Verdict"):
            cand.add_column(col, overflow="fold")
        shown: Dict[int, int] = {}
        for e in trace:
            if e["event"] != "anchor_candidate":
                continue
            # accepted rows always; rejected rows capped to the 3 strongest per clause
            # (candidates are already emitted best-rank first)
            if not e["accepted"]:
                shown[e["clause_idx"]] = shown.get(e["clause_idx"], 0) + 1
                if shown[e["clause_idx"]] > 3:
                    continue
            cand.add_row(str(e["clause_idx"] + 1), e["cell"], f'{e["recall"]:.3f}', f'{e["name_precision"]:.3f}',
                         f'{e["precision"]:.3f}', f'{e["f"]:.3f}', ",".join(e["tokens"]),
                         ("[green]ACCEPT[/green] " if e["accepted"] else "[dim]rejected[/dim] ") + e["reason"][:48])
        console.print(cand)
        for e in trace:
            if e["event"] == "anchor_dropped":
                console.print(f"[bold red]✗ ANCHOR DROPPED[/bold red] {e['cell']} (clause {e['clause_idx'] + 1 if e['clause_idx'] is not None else '?'}): {e['reason']}\n"
                              f"    needs: {e['needs']}\n    ancestor outputs: {e['ancestor_outputs']}")
            elif e["event"] == "method_fallback":
                console.print(f"[bold yellow]⚠ METHOD FELL BACK[/bold yellow] requested {e['requested']} → effective {e['to']}: {e['reason']}\n"
                              f"    (the path below is {e['to']}'s, not {e['requested']}'s)")
            elif e["event"] == "mcts_space":
                console.print(f"[bold cyan]🎲 MONTE CARLO SEARCH SPACE[/bold cyan] {e['clauses']} clauses; alternatives per clause "
                              f"(tied evidence): {[[g[0] if len(g) == 1 else g for g in gl] for gl in e['groups']]}; "
                              f"bridge candidates (tree postconditions that satisfy preconditions): {e['bridges']}")
            elif e["event"] == "mcts_summary":
                v = e.get("verdict") or {}
                console.print(f"[bold cyan]🎲 MONTE CARLO RESULT[/bold cyan] iterations={e['iterations']} distinct evaluations={e['evaluations']} "
                              f"cache hits={e['cache_hits']} seed={e['seed']} {e['elapsed_ms']} ms (stopped by {e['stopped_by']}); "
                              f"best reward={e['best_reward']}  verdict: served={v.get('served')} quality={v.get('quality')} "
                              f"violations={v.get('violations')} {v.get('first') or ''}")
                tb = _t("🎲 Root statistics (UCT): how the budget was spent")
                for col in ("First decision", "Visits", "Mean reward"):
                    tb.add_column(col, overflow="fold")
                for r in e.get("root", []):
                    tb.add_row(str(r["action"]), str(r["visits"]), str(r["mean_reward"]))
                console.print(tb)
                console.print(f"    most-visited line: {[(x['action'], x['visits']) for x in e.get('best_line', [])]}")
            elif e["event"] in ("mcts_bridge_removed", "mcts_baseline_kept"):
                console.print(f"[cyan]🎲 {e['event']}[/cyan] {({k: v for k, v in e.items() if k != 'event'})}")
            elif e["event"] in ("pruned_unrequested", "prune_blocked"):
                col = "green" if e["event"] == "pruned_unrequested" else "yellow"
                console.print(f"[bold {col}]✂ {e['event'].upper()}[/bold {col}] {e['cell']}: {e['reason']} (better explained by {e.get('winners')})")
            elif e["event"] in ("column_commit_inserted", "column_commit_blocked"):
                col = "green" if e["event"] == "column_commit_inserted" else "yellow"
                console.print(f"[bold {col}]↩ {e['event'].upper()}[/bold {col}] {e.get('commit')} after {e['after']} "
                              f"(column {e.get('column')}): {e['reason']}")
            elif e["event"] == "producer_inserted":
                console.print(f"[bold green]+ PRODUCER INSERTED[/bold green] {e['cell']} for {e['for_anchor']}: {e['reason']}")
            elif e["event"] in ("anchor_fallback_retrieval_only", "clause_without_anchor"):
                console.print(f"[yellow]! {e['event']}[/yellow] {e}")
    dec = list(getattr(router, "last_tagging_decisions", []) or [])
    if dec:
        tg = _t("🏷️ Clause Tagging Decisions (a cell outranked on a clause stays untagged)")
        for col in ("Cell", "Tagged clause", "Rank (recall, name-P, F)", "Outranked on clauses"):
            tg.add_column(col, overflow="fold")
        for d in dec:
            tg.add_row(d["cell"], "-" if d["tagged_clause"] is None else str(d["tagged_clause"] + 1),
                       str(d["best_rank"]), ",".join(str(j + 1) for j in d["outranked_on"]) or "-")
        console.print(tg)


def render_synthesis_diagnostics(console: Any, ctx: Any, cells: List[Any], prompt: str) -> None:
    """Layer-3 binding trace + clause coverage audit."""
    bt = list(getattr(ctx, "binding_trace", []) or [])
    if bt:
        tb = _t("🧵 Layer-3 Binding Decisions (which code site bound each port; rejections are the interesting rows)", "magenta")
        for col in ("Event", "Cell.port", "Value / var", "Clause", "Target key", "Detail"):
            tb.add_column(col, overflow="fold")
        for e in bt:
            if e["event"] == "bind" and e["port"] == "output_var":
                continue
            detail = e.get("site") or ""
            if "var_origins" in e:
                detail = f"origins={e['var_origins']}"
            if "candidates" in e:
                detail = f"candidates={e['candidates']}"
            if "expr" in e:
                detail = f"expr={e['expr']}"
            style = "red" if "rejected" in e["event"] or "failed" in e["event"] else ""
            tb.add_row(f"[{style}]{e['event']}[/{style}]" if style else e["event"], f"{e.get('cell')}.{e.get('port')}",
                       str(e.get("value", e.get("var", e.get("rejected", "")))), str(e.get("clause", "")),
                       str(e.get("target_key", "")), str(detail))
        console.print(tb)
    try:
        try:
            from .token_evidence import audit
        except (ImportError, ValueError):
            from token_evidence import audit
        res = audit(cells, prompt)
    except Exception as exc:  # pragma: no cover
        console.print(f"[yellow]coverage audit unavailable: {exc}[/yellow]")
        return
    if not res.get("available"):
        return
    ct = _t("📋 Prompt-Clause Coverage (served = on-path winner explains as much as the best lattice cell could)", "green")
    for col in ("#", "Clause", "Winner on path", "Path recall", "Lattice best", "Status"):
        ct.add_column(col, overflow="fold")
    for c in res["clauses"]:
        st = c["status"]
        ct.add_row(str(c["idx"] + 1), c["text"], str(c["winner"]), str(c["path_recall"]), str(c["lattice_best_recall"]),
                   f"[bold red]{st}[/bold red]" if st == "UNSERVED" else st)
    console.print(ct)
    cl = _t("🧩 Cell Roles (UNREQUESTED = outranked everywhere and explains no leftover clause token)", "green")
    for col in ("Cell", "Wins clauses", "Serves leftover of", "Status"):
        cl.add_column(col, overflow="fold")
    for c in res["cells"]:
        cl.add_row(c["cell"], ",".join(str(j + 1) for j in c["wins"]) or "-",
                   ",".join(str(j + 1) for j in c["residual_serves"]) or "-",
                   f"[bold red]{c['status']}[/bold red]" if c["status"] == "UNREQUESTED" else c["status"])
    console.print(cl)
