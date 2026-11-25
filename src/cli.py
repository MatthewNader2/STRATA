"""
src/cli.py — Clean CLI & Interactive Studio for the unified STRATA synthesis engine.

Subcommands: compile, validate, audit, run, shell.
"""
from __future__ import annotations

import argparse
import cmd
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

try:
    from .bridge_graph import LiveBridgeGraph, SimpleTypePoset
    from .config import MODELS_DIR, PROFILE_SPECS, resolve_profile_spec, settings
    from .gevr_sandbox import GEVRSandbox
    from .hardware import HardwareProfiler
    from .inference import ModelManager, select_optimal_embedder, select_optimal_llm
    from .language_backend import get_active_language_backend, get_language_backend
    from .lattice_db import (
        COMPILER_SCHEMA_VERSION,
        CompiledLattice,
        compute_trees_fingerprint,
        ensure_lattice_compiled,
        init_sqlite_db,
        load_compiled_lattice,
        record_trees_fingerprint,
    )
    from .log_config import get_logger
    from .profile_adapters import build_strata_pipeline_for_profile
    from .reranker import LocalReranker, get_available_rerankers
    from .schema import TreeSchema, derive_value_class, get_attr
except (ImportError, ValueError):
    from bridge_graph import LiveBridgeGraph, SimpleTypePoset
    from config import MODELS_DIR, PROFILE_SPECS, resolve_profile_spec, settings
    from gevr_sandbox import GEVRSandbox
    from hardware import HardwareProfiler
    from inference import ModelManager, select_optimal_embedder, select_optimal_llm
    from language_backend import get_active_language_backend, get_language_backend
    from lattice_db import (
        COMPILER_SCHEMA_VERSION,
        CompiledLattice,
        compute_trees_fingerprint,
        ensure_lattice_compiled,
        init_sqlite_db,
        load_compiled_lattice,
        record_trees_fingerprint,
    )
    from log_config import get_logger
    from profile_adapters import build_strata_pipeline_for_profile
    from reranker import LocalReranker, get_available_rerankers
    from schema import TreeSchema, derive_value_class, get_attr

logger = get_logger("cli")
console = Console()

DEFAULT_SANDBOX_TIMEOUT: float = float(os.environ.get("NSTL_SANDBOX_TIMEOUT", settings.sandbox_timeout))

QUICK_PROFILE_SHORTCUTS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        ((spec.quick_shortcut, spec.code) for spec in PROFILE_SPECS.values() if spec.quick_shortcut),
        key=lambda kv: kv[0],
    )
)


def _parse_bool_flag(arg: str, current: bool) -> bool:
    s = str(arg or "").strip().lower()
    if s in ("1", "true", "on", "yes", "enable", "enabled"):
        return True
    if s in ("0", "false", "off", "no", "disable", "disabled"):
        return False
    return not current


def _format_step_lines(steps: List[Any], indent: str = "") -> List[str]:
    lines: List[str] = []
    for idx, step in enumerate(steps, start=1):
        if step.kind in ("loop", "branch"):
            cond = f" ({step.condition})" if step.condition else ""
            lines.append(f"{indent}{idx}. [{step.kind.upper()}]{cond}")
            if step.body_steps:
                lines.extend(_format_step_lines(list(step.body_steps), indent + "  "))
            if step.else_steps:
                lines.append(f"{indent}  [ELSE]")
                lines.extend(_format_step_lines(list(step.else_steps), indent + "  "))
        else:
            tag = " [LIVE BRIDGE]" if step.is_live_bridge else ""
            lines.append(
                f"{indent}{idx}. {step.cell_id}{tag} -> {step.output_var} "
                f"({step.output_sig.carrier}[{step.output_sig.typestate}])"
            )
    return lines


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile canonical domain tree JSONs into SQLite schema v4."""
    trees_dir = getattr(args, "trees_dir", None) or getattr(args, "dir", "trees")
    out_db = getattr(args, "output", "trees/lattice.db")
    domain_filter = getattr(args, "domains", None) or None

    try:
        from tools.compile_trees import compile_database
    except ImportError:
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)
        from tools.compile_trees import compile_database

    init_sqlite_db(Path(out_db), clean=bool(getattr(args, "clean", False)))
    written = compile_database(output_db=str(out_db), domain_filter=domain_filter, trees_dir=trees_dir)
    if not domain_filter:
        record_trees_fingerprint(Path(out_db), compute_trees_fingerprint(Path(trees_dir)))
    print(f"[*] Compilation Complete: {written} total verified nodes compiled into '{out_db}'.")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate TreeSchema and LanguageBackend syntax across a domain JSON or compiled SQLite v4 DB."""
    target_str = getattr(args, "db", None) or getattr(args, "target", "trees/lattice.db")
    target_path = Path(target_str)
    if not target_path.exists():
        print(f"[!] Target file '{target_path}' does not exist!")
        sys.exit(1)

    if target_str.endswith(".json"):
        print(f"[*] Validating Domain Tree JSON '{target_path}'...")
        try:
            raw_data = json.loads(target_path.read_text(encoding="utf-8"))
            tree = TreeSchema.model_validate(raw_data)
        except Exception as exc:
            print(f"[X] JSON / Schema Validation Error: {exc}")
            sys.exit(1)

        backend = get_language_backend(tree.language_spec) if tree.language_spec is not None else get_active_language_backend()
        cells = list(tree.cells)
        label = f"Domain: {tree.domain}"
    else:
        print(f"[*] Validating SQLite v{COMPILER_SCHEMA_VERSION} Database '{target_path}'...")
        try:
            compiled = load_compiled_lattice(target_path)
        except Exception as exc:
            print(f"[X] Database Validation Error: {exc}")
            sys.exit(1)
        backend = get_language_backend(compiled.language_spec) if compiled.language_spec is not None else get_active_language_backend()
        cells = list(compiled.cells)
        label = f"Schema v{compiled.schema_version}"

    mut_counts: Counter[str] = Counter()
    valid_count = 0
    failed_count = 0
    errors: List[str] = []

    for cell in cells:
        mut = str(get_attr(cell, "mutation_type", "pure") or "pure").strip().lower()
        mut_counts[mut] += 1
        tmpl = backend.select_cell_template(cell)
        if not tmpl or not tmpl.strip():
            failed_count += 1
            errors.append(f"{cell.cell_id}: Empty code template")
            continue
        ok, err = backend.validate_syntax(tmpl, allow_placeholders=True)
        if ok:
            valid_count += 1
        else:
            failed_count += 1
            errors.append(f"{cell.cell_id}: Syntax Error: {err}")

    total = len(cells)
    print("\n==================================================")
    print(f" VALIDATION RESULTS FOR: {target_path.name}")
    print("==================================================")
    print(f" Source              : {label}")
    print(f" Total Cells Checked : {total}")
    print(f" Mutation Breakdown  : {dict(sorted(mut_counts.items()))}")
    print(f" Syntactically Valid : {valid_count}")
    print(f" Failed Nodes        : {failed_count}")
    print(f" Success Rate        : {(valid_count / total * 100):.2f}%" if total else "0.00%")

    if errors:
        print("\n[!] Top Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    print("\n[+] Validation passed 100% cleanly.")


def cmd_audit(args: argparse.Namespace) -> None:
    """Audit compiled SQLite v4 lattice topology, posets, and live bridge morphisms."""
    db_path = Path(getattr(args, "db", None) or "trees/lattice.db")
    trees_dir = str(getattr(args, "trees_dir", "trees") or "trees")
    ensure_lattice_compiled(trees_dir=trees_dir, db_path=str(db_path))

    try:
        compiled = load_compiled_lattice(db_path)
    except Exception as exc:
        print(f"[X] Audit Load Error: {exc}")
        sys.exit(1)

    poset = SimpleTypePoset(
        type_parents=compiled.type_parents,
        state_parents=compiled.state_parents,
        top_types=compiled.top_types,
    )
    bridge_graph = LiveBridgeGraph.build(compiled.cells, poset)

    total_cells = len(compiled.cells)
    loaded_domains = sorted(
        {
            str(get_attr(c, "domain_name", "") or "").strip().lower()
            for c in compiled.cells
            if str(get_attr(c, "domain_name", "") or "").strip()
        }
    )
    loaded_set = set(loaded_domains)
    type_edge_count = sum(len(v) for v in compiled.type_parents.values())
    state_edge_count = sum(len(v) for v in compiled.state_parents.values())
    tier1_bridges = sum(1 for e in bridge_graph.edges if not e.lossy and not e.requires_intent)
    tier2_bridges = len(bridge_graph.edges) - tier1_bridges

    entry_cells = 0
    terminal_cells = 0
    for cell in compiled.cells:
        ins = list((get_attr(cell, "inputs", {}) or {}).values())
        outs = list((get_attr(cell, "outputs", {}) or {}).values())
        has_req_data = any(
            bool(get_attr(p, "required", True)) and derive_value_class(p) == "data"
            for p in ins
        )
        if not has_req_data:
            entry_cells += 1

        mut = str(get_attr(cell, "mutation_type", "pure") or "pure").strip().lower()
        has_release = any(
            str(get_attr(p, "resource_mode", "value") or "value").strip().lower() == "release"
            for p in ins
        )
        has_non_void_out = any(
            str(get_attr(p, "carrier", "void") or "void").strip().lower() not in ("", "void", "none")
            for p in outs
        )
        if mut == "sink" or has_release or not has_non_void_out:
            terminal_cells += 1

    missing_imports = {
        dom: missing
        for dom, imps in sorted(compiled.domain_imports.items())
        for missing in ([imp for imp in imps if imp.lower() not in loaded_set],)
        if missing
    }

    summary = {
        "schema_version": str(compiled.schema_version),
        "total_cells": total_cells,
        "domains_loaded": loaded_domains,
        "domain_count": len(loaded_domains),
        "type_poset_edges": type_edge_count,
        "typestate_edges": state_edge_count,
        "live_bridges_total": len(bridge_graph.edges),
        "live_bridges_tier1": tier1_bridges,
        "live_bridges_tier2": tier2_bridges,
        "entry_cells": entry_cells,
        "terminal_release_cells": terminal_cells,
        "missing_domain_imports": missing_imports,
    }

    table = Table(
        title=f"STRATA SQLite v{compiled.schema_version} Lattice Audit ({total_cells} Cells)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="bold white")
    table.add_column("Status", style="bold green")

    table.add_row("Total Cells", str(total_cells), "[green]PASS[/green]" if total_cells > 0 else "[red]EMPTY[/red]")
    table.add_row("Domains Loaded", f"{len(loaded_domains)} ({', '.join(loaded_domains)})", "[green]OK[/green]")
    table.add_row("Type Poset Edges", str(type_edge_count), "[green]OK[/green]")
    table.add_row("Typestate Edges", str(state_edge_count), "[green]OK[/green]")
    table.add_row(
        "Live Bridge Morphisms",
        f"{len(bridge_graph.edges)} (Tier-1={tier1_bridges}, Tier-2={tier2_bridges})",
        "[green]OK[/green]",
    )
    table.add_row("Entry Cells", str(entry_cells), "Zero required data inputs")
    table.add_row("Terminal / Release Cells", str(terminal_cells), "Sink / release / void outputs")
    table.add_row(
        "Missing Domain Imports",
        str(missing_imports) if missing_imports else "0",
        "[green]PASS (0)[/green]" if not missing_imports else "[red]MISSING[/red]",
    )
    console.print(table)

    out_path_str = str(getattr(args, "output", "") or "").strip()
    if out_path_str:
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[+] Audit report saved to '{out_path}'.")

    if total_cells == 0 or missing_imports:
        print("[X] Lattice audit failed due to empty lattice or missing domain imports.")
        sys.exit(1)


def _scan_models(subdir: str, preferred: Tuple[str, ...] = (), *, require_dir_only: bool = False) -> List[str]:
    base_dir = Path(MODELS_DIR) / subdir
    if not base_dir.is_dir():
        return []
    raw = sorted(
        p.name
        for p in base_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (not require_dir_only or not p.name.endswith("-GGUF"))
    )
    prefs = [str(p).strip() for p in preferred if str(p).strip()]
    if not prefs:
        return raw
    ordered = [d for p in prefs for d in raw if d == p]
    ordered.extend(d for d in raw if d not in ordered)
    return ordered


def execute_unified_pipeline(
    prompt: str,
    *,
    compiled_lattice: Any = None,
    orchestrator: Any = None,
    profile_code: str = "0",
    model_manager: Any = None,
    reranker: Any = None,
    sandbox: Any = None,
    execute_sandbox: bool = False,
    timeout: float = DEFAULT_SANDBOX_TIMEOUT,
    debug: bool = False,
    render: bool = True,
    console_obj: Optional[Console] = None,
    pipeline: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single canonical synthesis entry point wrapping StrataPipeline."""
    out_c = console_obj or console
    spec = resolve_profile_spec(profile_code)
    t0 = time.perf_counter()

    source_lattice = compiled_lattice if compiled_lattice is not None else orchestrator
    active_pipeline = pipeline
    if active_pipeline is None:
        active_pipeline = build_strata_pipeline_for_profile(
            source_lattice,
            spec,
            model_manager=model_manager,
            reranker=reranker,
            sandbox=sandbox,
            timeout=timeout,
        )

    res = active_pipeline.run(prompt, execute=bool(execute_sandbox))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    steps = list(res.plan.steps) if res.plan is not None else []
    path_ids = [s.cell_id for s in steps]
    sb_res = dict(res.sandbox_result) if isinstance(res.sandbox_result, dict) else {}
    if not execute_sandbox:
        sb_status = "SKIPPED (--no-exec)"
    elif sb_res.get("success", False) or res.ok:
        sb_status = "PASSED"
    else:
        sb_status = f"FAILED ({res.refusal_kind or 'SANDBOX_ERROR'})"

    if render:
        if debug:
            if res.ir is not None:
                ir_summary = (
                    f"source_mode={res.ir.source_mode} | nodes={len(res.ir.flatten_nodes())}"
                    + (f" | deliverable={res.ir.deliverable}" if res.ir.deliverable else "")
                )
                if res.warnings:
                    ir_summary += "\nwarnings: " + "; ".join(res.warnings)
                out_c.print(
                    Panel(
                        ir_summary,
                        title=f"[cyan]Layer 0 TypedIR ({spec.display_name})[/cyan]",
                        border_style="cyan",
                        padding=(0, 1),
                    )
                )
            if steps:
                out_c.print(
                    Panel(
                        "\n".join(_format_step_lines(steps)),
                        title=f"[yellow]Layer 2 Plan ({res.live_bridge_count} live bridge(s))[/yellow]",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )

        if res.ok and res.code:
            arrows = " -> ".join(path_ids)
            out_c.print(
                f"\n[bold yellow]Unified Path ({latency_ms:.2f}ms, bridges={res.live_bridge_count}):[/bold yellow] "
                f"[bold cyan]{arrows}[/bold cyan]"
            )
            backend = active_pipeline.language_backend or get_active_language_backend()
            syn_lang = (backend.name if backend is not None else "") or "python"
            out_c.print(
                Panel(
                    Syntax(res.code, syn_lang, theme="monokai", line_numbers=True),
                    title=f"[bold green]Synthesized Code ({spec.display_name})[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                )
            )
            if debug or execute_sandbox:
                rt_note = f" | runtime_setups={len(res.runtime_setups)}" if res.runtime_setups else ""
                out_c.print(f"[dim]Sandbox: {sb_status}{rt_note}[/dim]\n")
        else:
            out_c.print(
                f"\n[bold red][!] Refused at stage '{res.stage_failed}' "
                f"({res.refusal_kind}):[/bold red] {res.error}\n"
            )
            if debug and res.code:
                out_c.print(Panel(res.code, title="[red]Emitted Code Before Failure[/red]", border_style="red", padding=(0, 1)))

    return {
        "ok": bool(res.ok),
        "engine": "unified",
        "prompt": prompt,
        "profile": spec.code,
        "path": path_ids,
        "live_bridges": int(res.live_bridge_count),
        "latency_ms": latency_ms,
        "code": res.code,
        "stage_failed": res.stage_failed,
        "refusal_kind": res.refusal_kind,
        "error": res.error,
        "warnings": list(res.warnings),
        "runtime_setups": list(res.runtime_setups),
        "sandbox_status": sb_status,
        "sandbox_result": sb_res,
    }


class NSTLInteractiveShell(cmd.Cmd):
    """Interactive studio shell for STRATA unified program synthesis."""

    prompt = "\033[1;36mSTRATA [Profile 0: Pure Symbolic]\033[0m > "

    def emptyline(self) -> None:
        pass

    def __init__(
        self,
        db_path: str = "trees/lattice.db",
        initial_profile: str = "0",
        embedder: str = "",
        llm: str = "",
        device: str = "auto",
        debug: bool = False,
        interactive: bool = True,
        no_exec: bool = False,
        exec_sandbox: bool = False,
        timeout: float = DEFAULT_SANDBOX_TIMEOUT,
        reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        allowed_capabilities: Optional[Union[str, List[str], Set[str]]] = None,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.device = device
        self.embedder_request = embedder
        self.llm_request = llm
        try:
            self.embedder_name = select_optimal_embedder(embedder)
        except Exception:
            self.embedder_name = embedder
        try:
            self.llm_name = select_optimal_llm(llm)
        except Exception:
            self.llm_name = llm
        self.active_profile = "0"
        self.debug = bool(debug)
        self.interactive = bool(interactive)
        self.timeout = float(timeout)
        self.history: List[Dict[str, Any]] = []
        self._pipeline_cache: Optional[Tuple[Tuple[Any, ...], Any]] = None

        if exec_sandbox:
            self.no_exec = False
        elif no_exec:
            self.no_exec = True
        else:
            self.no_exec = not settings.sandbox_enabled

        self._reranker_override: Optional[bool] = reranker
        if reranker is not None:
            settings.use_reranker = bool(reranker)
            os.environ["NSTL_USE_RERANKER"] = "1" if reranker else "0"
        self.use_reranker = bool(settings.use_reranker)

        if reranker_model is not None:
            settings.reranker_model = reranker_model
            os.environ["NSTL_RERANKER_MODEL"] = str(reranker_model)
        self.reranker_model = settings.reranker_model

        if isinstance(allowed_capabilities, str):
            self.allowed_capabilities: Set[str] = {c.strip() for c in allowed_capabilities.split(",") if c.strip()}
        else:
            self.allowed_capabilities = set(allowed_capabilities or ())
        if self.allowed_capabilities:
            os.environ["NSTL_ALLOWED_CAPABILITIES"] = ",".join(sorted(self.allowed_capabilities))

        env_trees = os.environ.get("NSTL_TREES_DIR")
        db_parent = Path(db_path).parent
        trees_dir = env_trees or (
            db_parent if db_path and str(db_parent) not in ("", ".", "trees") else (settings.trees_dir or db_parent)
        )
        self.trees_dir = str(trees_dir)

        if self.interactive:
            console.print("\n[bold cyan][*] Initializing STRATA Engine...[/bold cyan]")
        t0 = time.perf_counter()
        ensure_lattice_compiled(trees_dir=self.trees_dir, db_path=db_path)
        self.compiled_lattice: CompiledLattice = load_compiled_lattice(db_path)
        self.sandbox = GEVRSandbox(allowed_capabilities=self.allowed_capabilities)
        load_ms = (time.perf_counter() - t0) * 1000.0
        if self.interactive:
            console.print(f"[bold green][+] Loaded {len(self.compiled_lattice.cells):,} cells ({load_ms:.1f}ms)[/bold green]\n")

        self._switch_profile(initial_profile, embedder=embedder, llm=llm, verbose=False)
        if self.interactive:
            self._render_dashboard()

    def _switch_profile(self, target: str, *, embedder: str = "", llm: str = "", verbose: bool = True) -> None:
        spec = resolve_profile_spec(target)
        mm = ModelManager.get_instance()
        emb_arg = embedder or self.embedder_request
        llm_arg = llm or self.llm_request

        if spec.pure_symbolic:
            mm.cleanup()
        else:
            mm.initialize_profile(spec.code, embedder_name=emb_arg, llm_name=llm_arg)
            if mm.embedder_name:
                self.embedder_name = mm.embedder_name
            if mm.llm_name:
                self.llm_name = mm.llm_name

        self.active_profile = spec.code
        os.environ["NSTL_PROFILE"] = spec.code
        settings.profile = spec.code
        self._pipeline_cache = None
        self.prompt = f"\033[1;36mSTRATA [{spec.display_name}]\033[0m > "
        if verbose:
            console.print(f"[bold green][+] Active Profile -> {spec.display_name}[/bold green]")

    def _render_dashboard(self) -> None:
        domains = sorted({c.domain_name for c in self.compiled_lattice.cells if c.domain_name})
        spec = resolve_profile_spec(self.active_profile)
        dev = HardwareProfiler.get_optimal_device()
        shortcuts = " | ".join(f"{k}=Profile {v}" for k, v in QUICK_PROFILE_SHORTCUTS)

        grid = Table(box=box.ROUNDED, show_header=False, border_style="cyan", expand=True)
        grid.add_column("Property", style="bold yellow", width=22)
        grid.add_column("Value", style="white")
        grid.add_row("Active Profile", spec.display_name)
        grid.add_row("Hardware", dev.upper())
        grid.add_row("Compiled Lattice", f"{len(self.compiled_lattice.cells):,} cells across {len(domains)} domains")
        grid.add_row("Sandbox Execution", "DISABLED (--no-exec)" if self.no_exec else "ENABLED")
        grid.add_row("Reranker", "ON" if self.use_reranker else "OFF")
        grid.add_row("Shortcuts", f"{shortcuts} | debug | exec | status | domains | exit")
        console.print(Panel(grid, title="[bold cyan]STRATA Unified Synthesis Studio[/bold cyan]", border_style="cyan"))

    def do_profile(self, arg: str) -> None:
        val = (arg or "").strip()
        if not val:
            console.print(f"[*] Active Profile: {resolve_profile_spec(self.active_profile).display_name}")
            return
        try:
            self._switch_profile(val)
        except Exception as exc:
            console.print(f"[bold red][!] Profile switch failed:[/bold red] {exc}")

    def do_debug(self, arg: str) -> None:
        self.debug = _parse_bool_flag(arg, self.debug)
        console.print(f"[bold yellow][*] Debug mode -> {'ON' if self.debug else 'OFF'}[/bold yellow]")

    def do_exec(self, arg: str) -> None:
        enabled = _parse_bool_flag(arg, not self.no_exec)
        self.no_exec = not enabled
        console.print(f"[bold yellow][*] Sandbox execution -> {'ENABLED' if enabled else 'DISABLED'}[/bold yellow]")

    def do_reranker(self, arg: str) -> None:
        self.use_reranker = _parse_bool_flag(arg, self.use_reranker)
        self._reranker_override = self.use_reranker
        settings.use_reranker = self.use_reranker
        self._pipeline_cache = None
        console.print(f"[bold yellow][*] Neural reranker -> {'ON' if self.use_reranker else 'OFF'}[/bold yellow]")

    def do_status(self, _arg: str) -> None:
        self._render_dashboard()

    def do_models(self, _arg: str) -> None:
        embs = _scan_models("embeddings", settings.preferred_embedders, require_dir_only=True)
        llms = _scan_models("llms", settings.preferred_llms)
        rrs = get_available_rerankers()
        console.print(f"Embedders : {embs}")
        console.print(f"LLMs      : {llms}")
        console.print(f"Rerankers : {rrs}")

    def do_domains(self, _arg: str) -> None:
        counts = Counter(c.domain_name for c in self.compiled_lattice.cells if c.domain_name)
        for dom, cnt in sorted(counts.items()):
            console.print(f"  {dom:16s} {cnt:4d} cells")

    def do_history(self, _arg: str) -> None:
        for i, item in enumerate(self.history, start=1):
            console.print(f"{i}. [{item['profile']}] ok={item['ok']} :: {item['prompt']}")

    def do_clear(self, _arg: str) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        self._render_dashboard()

    def do_new(self, _arg: str) -> None:
        self.history.clear()
        self._pipeline_cache = None
        console.print("[green][+] Session cleared.[/green]")

    def do_exit(self, _arg: str) -> bool:
        ModelManager.get_instance().cleanup()
        return True

    def do_quit(self, arg: str) -> bool:
        return self.do_exit(arg)

    def default(self, line: str) -> Optional[bool]:
        prompt = (line or "").strip()
        if not prompt:
            return None

        if prompt.startswith("/"):
            parts = prompt[1:].split(maxsplit=1)
            cmd_name = parts[0].lower()
            cmd_arg = parts[1] if len(parts) > 1 else ""
            if cmd_name == "profile":
                self.do_profile(cmd_arg)
                return None
            if cmd_name in ("debug", "dbg"):
                self.do_debug(cmd_arg)
                return None
            if cmd_name == "exec":
                self.do_exec(cmd_arg)
                return None
            if cmd_name in ("reranker", "rerank"):
                self.do_reranker(cmd_arg)
                return None
            if cmd_name == "status":
                self.do_status(cmd_arg)
                return None
            if cmd_name == "domains":
                self.do_domains(cmd_arg)
                return None
            if cmd_name == "models":
                self.do_models(cmd_arg)
                return None
            if cmd_name in ("history", "hist"):
                self.do_history(cmd_arg)
                return None
            if cmd_name in ("clear", "cls"):
                self.do_clear(cmd_arg)
                return None
            if cmd_name in ("new", "reset"):
                self.do_new(cmd_arg)
                return None
            if cmd_name in ("exit", "quit", "q"):
                return self.do_exit(cmd_arg)

        shortcut_map = dict(QUICK_PROFILE_SHORTCUTS)
        if prompt in shortcut_map:
            self._switch_profile(shortcut_map[prompt])
            return None

        query_debug = self.debug
        if "--debug" in prompt:
            prompt = prompt.replace("--debug", "").strip()
            query_debug = True

        spec = resolve_profile_spec(self.active_profile)
        mm = ModelManager.get_instance()
        want_reranker = (
            bool(spec.uses_reranker and settings.use_reranker)
            if self._reranker_override is None
            else (spec.uses_reranker and self._reranker_override)
        )
        rr_obj = None
        if want_reranker:
            try:
                rr_obj = LocalReranker.get_shared(model_name_or_path=self.reranker_model, device=self.device)
            except Exception:
                rr_obj = None

        cache_key = (
            spec.code,
            (mm.current_profile_name or "", mm.embedder_name or "", mm.llm_name or ""),
            want_reranker,
            self.reranker_model or "",
            self.timeout,
            len(self.compiled_lattice.cells),
        )
        if self._pipeline_cache is not None and self._pipeline_cache[0] == cache_key:
            active_pipe = self._pipeline_cache[1]
        else:
            active_pipe = build_strata_pipeline_for_profile(
                self.compiled_lattice,
                spec,
                model_manager=mm,
                reranker=rr_obj,
                sandbox=self.sandbox,
                timeout=self.timeout,
            )
            self._pipeline_cache = (cache_key, active_pipe)

        record = execute_unified_pipeline(
            prompt,
            compiled_lattice=self.compiled_lattice,
            profile_code=spec.code,
            model_manager=mm,
            reranker=rr_obj,
            sandbox=self.sandbox,
            execute_sandbox=(not self.no_exec),
            timeout=self.timeout,
            debug=query_debug,
            pipeline=active_pipe,
        )
        self.history.append(record)
        return None


def cmd_shell(args: argparse.Namespace) -> None:
    shell = NSTLInteractiveShell(
        db_path=getattr(args, "db", "trees/lattice.db"),
        initial_profile=getattr(args, "profile", "0"),
        embedder=getattr(args, "embedder", ""),
        llm=getattr(args, "llm", ""),
        device=getattr(args, "device", "auto"),
        debug=bool(getattr(args, "debug", False)),
        interactive=True,
        no_exec=bool(getattr(args, "no_exec", False)),
        exec_sandbox=bool(getattr(args, "exec_sandbox", False)),
        timeout=float(getattr(args, "timeout", DEFAULT_SANDBOX_TIMEOUT)),
        reranker=getattr(args, "reranker", None),
        reranker_model=getattr(args, "reranker_model", None),
        allowed_capabilities=getattr(args, "allow_capabilities", ""),
    )
    shell.cmdloop()


def cmd_run(args: argparse.Namespace) -> None:
    prompt = str(getattr(args, "prompt", "") or "").strip()
    if not prompt:
        console.print("[bold red][!] Prompt must not be empty.[/bold red]")
        sys.exit(1)

    profile = getattr(args, "profile", "0")
    shell = NSTLInteractiveShell(
        db_path=getattr(args, "db", "trees/lattice.db"),
        initial_profile=profile,
        embedder=getattr(args, "embedder", ""),
        llm=getattr(args, "llm", ""),
        device=getattr(args, "device", "auto"),
        debug=bool(getattr(args, "debug", False)),
        interactive=False,
        no_exec=bool(getattr(args, "no_exec", False)),
        exec_sandbox=bool(getattr(args, "exec_sandbox", False)),
        timeout=float(getattr(args, "timeout", DEFAULT_SANDBOX_TIMEOUT)),
        reranker=getattr(args, "reranker", None),
        reranker_model=getattr(args, "reranker_model", None),
        allowed_capabilities=getattr(args, "allow_capabilities", ""),
    )
    try:
        shell.default(f"{prompt} --debug" if getattr(args, "debug", False) else prompt)
    finally:
        ModelManager.get_instance().cleanup()

    if shell.history:
        last = shell.history[-1]
        if not last.get("code"):
            sys.exit(2)
        if getattr(args, "exec_sandbox", False) and not last.get("sandbox_result", {}).get("success", False):
            sys.exit(3)


def _add_runtime_flags(parser: argparse.ArgumentParser, profile_help: str) -> None:
    parser.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    parser.add_argument("--profile", type=str, default="0", help=profile_help)
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug output across pipeline layers")
    parser.add_argument("--embedder", type=str, default="", help="Embedding model name (empty = auto-discover)")
    parser.add_argument("--llm", type=str, default="", help="LLM model name (empty = auto-discover)")
    parser.add_argument("--reranker", dest="reranker", action="store_true", default=None, help="Enable neural reranker")
    parser.add_argument("--no-reranker", dest="reranker", action="store_false", help="Disable neural reranker")
    parser.add_argument("--reranker-model", type=str, default=None, help="Neural reranker model name")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--no-exec", action="store_true", help="Skip GEVR sandbox execution")
    parser.add_argument("--exec", dest="exec_sandbox", action="store_true", default=False, help="Enable GEVR sandbox execution")
    parser.add_argument("--allow-capabilities", type=str, default="", help="Comma-separated sandbox capabilities (e.g. net.http,fs.read)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_SANDBOX_TIMEOUT, help="Sandbox timeout in seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="STRATA Unified Toolchain CLI & Studio",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug mode")
    subparsers = parser.add_subparsers(dest="command", required=False)

    p_compile = subparsers.add_parser("compile", help="Compile canonical domain JSONs into SQLite v4 database")
    p_compile.add_argument("--trees-dir", type=str, default="trees", help="Directory containing domain JSON files")
    p_compile.add_argument("--output", type=str, default="trees/lattice.db", help="Target SQLite DB path")
    p_compile.add_argument("--domains", nargs="*", default=None, help="Optional domain filter")
    p_compile.add_argument("--clean", action="store_true", help="Purge target database before compiling")
    p_compile.set_defaults(func=cmd_compile)

    p_validate = subparsers.add_parser("validate", help="Validate schema and syntax across SQLite v4 DB or domain JSON")
    p_validate.add_argument("target", nargs="?", default="trees/lattice.db", help="Path to SQLite .db or domain .json file")
    p_validate.add_argument("--db", type=str, default=None, help="Optional explicit SQLite DB path")
    p_validate.set_defaults(func=cmd_validate)

    p_audit = subparsers.add_parser("audit", help="Audit compiled SQLite v4 lattice topology, posets, and live bridges")
    p_audit.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_audit.add_argument("--trees-dir", type=str, default="trees", help="Directory containing domain JSON files")
    p_audit.add_argument("--output", type=str, default="", help="Optional JSON output report path")
    p_audit.set_defaults(func=cmd_audit)

    p_run = subparsers.add_parser("run", help="Synthesize code for one prompt from CLI")
    p_run.add_argument("prompt", type=str, help="Formal or natural-language specification")
    _add_runtime_flags(p_run, "Inference profile (0, A, B, C, D, E, O)")
    p_run.set_defaults(func=cmd_run)

    p_shell = subparsers.add_parser("shell", help="Launch interactive STRATA synthesis studio")
    _add_runtime_flags(p_shell, "Initial inference profile (0, A, B, C, D, E, O)")
    p_shell.set_defaults(func=cmd_shell)

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        NSTLInteractiveShell().cmdloop()
        return

    if len(sys.argv) == 2 and sys.argv[1] in ("--debug", "-d"):
        NSTLInteractiveShell(debug=True).cmdloop()
        return

    known_cmds = {"compile", "validate", "audit", "run", "shell", "-h", "--help"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_cmds:
        if sys.argv[1] in ("--debug", "-d") and len(sys.argv) > 2 and sys.argv[2] not in known_cmds:
            sys.argv = [sys.argv[0], "run", sys.argv[2], "--debug"] + sys.argv[3:]
        elif sys.argv[1] not in ("-h", "--help"):
            sys.argv.insert(1, "run")

    args = build_parser().parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        cmd_shell(argparse.Namespace(db="trees/lattice.db", profile="0", debug=getattr(args, "debug", False)))


STRATAInteractiveShell = NSTLInteractiveShell


if __name__ == "__main__":
    main()
