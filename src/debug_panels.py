"""
src/debug_panels.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

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
        anchor_rows = 0
        for e in trace:
            if e.get("event") != "anchor_candidate" or "accepted" not in e or "clause_idx" not in e:
                continue
            # accepted rows always; rejected rows capped to the 3 strongest per clause
            # (candidates are already emitted best-rank first)
            accepted = bool(e.get("accepted"))
            cl_idx = int(e.get("clause_idx", 0))
            if not accepted:
                shown[cl_idx] = shown.get(cl_idx, 0) + 1
                if shown[cl_idx] > 3:
                    continue
            cand.add_row(str(cl_idx + 1), str(e.get("cell", "")), f'{float(e.get("recall", 0.0)):.3f}', f'{float(e.get("name_precision", 0.0)):.3f}',
                         f'{float(e.get("precision", 0.0)):.3f}', f'{float(e.get("f", 0.0)):.3f}', ",".join(e.get("tokens") or ()),
                         ("[green]ACCEPT[/green] " if accepted else "[dim]rejected[/dim] ") + str(e.get("reason", ""))[:48])
            anchor_rows += 1
        if anchor_rows == 0:
            for e in trace:
                if e.get("event") != "omega_target_candidates":
                    continue
                tgt = e.get("target")
                cl_str = str(tgt[0] + 1) if isinstance(tgt, (tuple, list)) and tgt else str(tgt)
                c_list = list(e.get("candidates") or ())
                cand.add_row(cl_str, ", ".join(str(x) for x in c_list[:4]), "-", "-", "-", "-", "-", "[cyan]OMEGA target[/cyan]")
        console.print(cand)
        for e in trace:
            ev = e.get("event")
            if ev == "anchor_dropped":
                _cidx = e.get("clause_idx")
                console.print(f"[bold red]✗ ANCHOR DROPPED[/bold red] {e.get('cell')} (clause {_cidx + 1 if _cidx is not None else '?'}): {e.get('reason', '')}\n"
                              f"    needs: {e.get('needs')}\n    ancestor outputs: {e.get('ancestor_outputs')}")
            elif ev == "method_fallback":
                console.print(f"[bold yellow]⚠ METHOD FELL BACK[/bold yellow] requested {e.get('requested')} → effective {e.get('to')}: {e.get('reason', '')}\n"
                              f"    (the path below is {e.get('to')}'s, not {e.get('requested')}'s)")
            elif ev == "mcts_space":
                console.print(f"[bold cyan]🎲 MONTE CARLO SEARCH SPACE[/bold cyan] {e.get('clauses')} clauses; alternatives per clause "
                              f"(tied evidence): {[[g[0] if len(g) == 1 else g for g in gl] for gl in (e.get('groups') or [])]}; "
                              f"bridge candidates (tree postconditions that satisfy preconditions): {e.get('bridges')}")
            elif ev == "mcts_summary":
                v = e.get("verdict") or {}
                console.print(f"[bold cyan]🎲 MONTE CARLO RESULT[/bold cyan] iterations={e.get('iterations')} distinct evaluations={e.get('evaluations')} "
                              f"cache hits={e.get('cache_hits')} seed={e.get('seed')} {e.get('elapsed_ms')} ms (stopped by {e.get('stopped_by')}); "
                              f"best reward={e.get('best_reward')}  verdict: served={v.get('served')} quality={v.get('quality')} "
                              f"violations={v.get('violations')} {v.get('first') or ''}")
                tb = _t("🎲 Root statistics (UCT): how the budget was spent")
                for col in ("First decision", "Visits", "Mean reward"):
                    tb.add_column(col, overflow="fold")
                for r in e.get("root", []):
                    tb.add_row(str(r.get("action")), str(r.get("visits")), str(r.get("mean_reward")))
                console.print(tb)
                console.print(f"    most-visited line: {[(x.get('action'), x.get('visits')) for x in e.get('best_line', [])]}")
            elif ev in ("mcts_bridge_removed", "mcts_baseline_kept"):
                console.print(f"[cyan]🎲 {ev}[/cyan] {({k: v for k, v in e.items() if k != 'event'})}")
            elif ev in ("pruned_unrequested", "prune_blocked"):
                col = "green" if ev == "pruned_unrequested" else "yellow"
                console.print(f"[bold {col}]✂ {ev.upper()}[/bold {col}] {e.get('cell')}: {e.get('reason', '')} (better explained by {e.get('winners')})")
            elif ev in ("column_commit_inserted", "column_commit_blocked"):
                col = "green" if ev == "column_commit_inserted" else "yellow"
                console.print(f"[bold {col}]↩ {ev.upper()}[/bold {col}] {e.get('commit', e.get('cell'))} after {e.get('after')} "
                              f"(column {e.get('column')}): {e.get('reason', '')}")
            elif ev == "producer_inserted":
                console.print(f"[bold green]+ PRODUCER INSERTED[/bold green] {e.get('cell')} for {e.get('for_anchor')}: {e.get('reason', '')}")
            elif ev in ("anchor_fallback_retrieval_only", "clause_without_anchor"):
                console.print(f"[yellow]! {ev}[/yellow] {e}")
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
