#!/usr/bin/env python3
"""
test.py - Comprehensive Multi-Profile & Multi-RouteMethod Empirical Test Runner for NSTL

Evaluates `nstl_cli.py --debug` across Profiles (0, A, C, D, E, S) and RouteMethods (M0-M11).
Designed to distinguish Architectural / Mathematical issues (lattice reachability, typestate contracts,
coverage floor refusals) from Implementation issues (runtime exceptions, missing imports).

Features:
    - Multi-Prompt Presets: Canonical, Vague Game (Pygame), Vague ML, Specific CV2 with explicit
      parameters, Intensive Control Flow (def/while/if-else/try-except), Cryptography, Graph, NLP, and Audio.
    - Multi-Layer In-Process Caching Engine (`CachedNSTLRunner`):
      * Global Lattice Graph Cache (lattice loaded once across all runs).
      * Neural Model Cache (SentenceTransformers + GGUF LLMs stay resident across method runs).
      * Layer 1 Semantic Tunnel Cache (reuses query decomposition and FAISS lookups across methods).
      * Avoids repeated model/lattice loading (speed-up is not asserted here; timings are measured per run).
    - Synthetic Asset Auto-Provisioning: Automatically ensures `input.csv` and `input.png` exist,
      and sets headless environment variables (SDL dummy drivers) for clean sandbox execution.
    - Overnight Execution: Continuous loop mode (`--loop` / `--overnight`) and multi-iteration runs (`--iterations N`).
    - Comparative Evaluation & Master Reporting: Dual-pass logs (`total_no_feedback.log`,
      `total_with_feedback.log`, `total.log`) with rescue matrix and overnight summary tables.
"""

import argparse
import io
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Self-healing environment resolver: ensure pydantic and NSTL dependencies are present
try:
    import pydantic
except ImportError:
    candidate_pythons = [
        "/home/matthew/.pyenv/versions/3.13.5/bin/python3",
        os.path.expanduser("~/.pyenv/versions/3.13.5/bin/python3"),
    ]
    resolved_python = None
    for cand in candidate_pythons:
        if os.path.exists(cand):
            try:
                res = subprocess.run([cand, "-c", "import pydantic"], capture_output=True)
                if res.returncode == 0:
                    resolved_python = cand
                    break
            except Exception:
                pass
    if resolved_python and resolved_python != sys.executable:
        os.execv(resolved_python, [resolved_python] + sys.argv)
    else:
        sys.stderr.write(
            f"[FATAL ERROR] NSTL core dependencies (pydantic) are not installed in the active Python environment:\n"
            f"  {sys.executable}\n"
            f"Please run test.py using the correct virtual environment / pyenv.\n"
        )
        sys.exit(1)

# Rich UI setup for animated progress and live code preview
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.markup import escape as escape_markup
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        TaskProgressColumn,
        ProgressColumn,
        Task,
    )
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None  # type: ignore

# Project root setup
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import eval_metrics as EM  # noqa: E402  (scoring lives in a unit-tested module, not in the runner)

CLI_SCRIPT = ROOT_DIR / "nstl_cli.py"
if not CLI_SCRIPT.exists():
    CLI_SCRIPT = ROOT_DIR / "src" / "cli.py"

# ==============================================================================
# Prompt Archetype Presets Catalog
# ==============================================================================

DEFAULT_PROMPT = (
    'load a csv file named "input.csv", normalize X column, drop null values, '
    'get the mean of the Y column, and perform FFT and write it to a new column called Z, '
    'then make a regression model trained on X and Y to predict Z'
)

# Each preset may declare "expected": "synthesize" (default) or "clarify". For "clarify" (deliberately
# under-specified prompts) an acceptable outcome is a verified program OR a refusal that names what is missing;
# an engine defect never is. Strict pass rate always counts verified programs only.
PROMPT_PRESETS: Dict[str, Dict[str, str]] = {
    "canonical": {
        "name": "Canonical Multi-Stage Pipeline (Pandas + FFT + Sklearn Regression)",
        "prompt": DEFAULT_PROMPT,
        "description": "Cross-domain pandas/numpy/sklearn pipeline with signal processing and regression.",
    },
    "vague_game": {
        "name": "Vague / Abstract 2D Game Loop (Pygame)",
        "prompt": "make a 2D game in pygame with a display window, game loop, clock, and quit",
        "description": "Abstract game loop prompt testing lifecycle, screen refresh, and event poll combinators.",
        "expected": "clarify",
    },
    "unvague_game": {
        "name": "Detailed Closed-Domain Snake Game (Pygame)",
        "prompt": (
            "initialize pygame and font system, create an 800 by 600 display window with title 'Snake Game', "
            "initialize clock, create an Arial size 24 font, create snake head rect at x 400 y 300 width 20 height 20, "
            "create food rect at x 200 y 200 width 20 height 20, create boundary rect at x 0 y 0 width 800 height 600, "
            "create background color black 0 0 0 255, green snake color 0 255 0 255, red food color 255 0 0 255, and white text color 255 255 255 255, "
            "run main game loop while running is True: get events and key presses, move snake rect by dx 20 and dy 0, "
            "clamp snake rect to boundary rect, check rect collision between snake rect and food rect, "
            "fill screen surface with black color, draw green snake rect and red food rect on screen surface, "
            "render score text surface 'Score: 0' with white color and blit it at position (10, 10) onto screen, "
            "tick clock at 15 fps, flip display, and finally quit font and quit pygame"
        ),
        "description": "Detailed closed-domain Snake game prompt contrasting vague_game and exercising long-chain routing across pygame primitives.",
    },
    "vague_ml": {
        "name": "Vague / Abstract Machine Learning Model",
        "prompt": "load dataset and train a machine learning model to predict target",
        "description": "Vague predictive modeling prompt testing abstract semantic tunneling across tabular ML.",
        "expected": "clarify",
    },
    "cv2_params": {
        "name": "Deep Single-Domain Computer Vision with Explicit Parameters (CV2)",
        "prompt": (
            'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, '
            'detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"'
        ),
        "description": "Precise computer vision pipeline with explicit slot literals, filtering, and edge detection.",
    },
    "control_flow": {
        "name": "Intensive Python Control Flow (def, while, if/else, try/except)",
        "prompt": (
            "define a function that processes data with a while loop, "
            "branches on condition with if else, and handles exceptions with try except"
        ),
        "description": "Higher-order control combinators: lambda abstraction, coproduct branching, and exception guards.",
    },
    "crypto": {
        "name": "Cryptographic Pipeline (SHA256, AES Cipher, Base64)",
        "prompt": "read file 'input.csv', hash data using sha256, encrypt payload with aes cipher, and encode with base64",
        "description": "Cryptographic hashing, symmetric encryption, and base64 encoding pipeline.",
    },
    "audio": {
        "name": "Audio Signal Processing Pipeline (Scipy / Librosa)",
        "prompt": "load audio file, apply high-pass filter, compute spectrogram, and export filtered audio",
        "description": "Audio signal filtering, spectrogram transform, and acoustic export.",
    },
}

# Supported profiles and their corresponding route methods (M0 - M11, OMEGA)
ALL_PROFILES = ["0", "A", "B", "C", "D", "E", "S", "O", "U", "AE"]
ALL_METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"]

SUPPORTED_METHODS: Dict[str, List[str]] = {
    "0": ["M0", "M1", "M2", "M3", "M6", "M10", "OMEGA"],
    "A": ["M0", "M1", "M2", "M3", "M6", "M10", "M11", "OMEGA"],
    "B": ["M0", "M1", "M2", "M3", "M6", "M10", "M11", "OMEGA"],
    "C": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"],
    "D": ["M0", "M1", "M2", "M3", "M6", "M10", "M11", "OMEGA"],
    "E": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"],
    "S": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"],
    "O": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"],
    "U": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "OMEGA"],
    "AE": ["M0", "M1", "M2", "M3", "M6", "M10", "M11", "OMEGA"],
}

# Regex to strip ANSI escape sequences for pure readable text logs
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Removes terminal ANSI styling sequences from text."""
    return ANSI_ESCAPE_RE.sub("", text)


def extract_code_from_output(output: str) -> str:
    """
    Extracts synthesized python code from raw or clean CLI output when running in subprocess mode.
    Handles both rich panel box formats and markdown code blocks.
    """
    if not output:
        return ""
    m_fence = re.search(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if m_fence:
        return m_fence.group(1).strip()
    if "Synthesized Python Code" in output:
        lines = output.splitlines()
        capturing = False
        code_lines = []
        for line in lines:
            if "Synthesized Python Code" in line:
                capturing = True
                continue
            if capturing:
                if any(c in line for c in ("╰", "+--", "+==")) or "Total End-to-End" in line or "Layer 4:" in line or "Method:" in line:
                    break
                m_line = re.match(r"^\s*[│|]\s*(?:\d+\s+)?(.*)$", line)
                if m_line:
                    content = m_line.group(1)
                    if content.endswith("│") or content.endswith("|"):
                        content = content[:-1].rstrip()
                    code_lines.append(content)
        if code_lines:
            return "\n".join(code_lines).rstrip()
    return ""


if RICH_AVAILABLE:
    class RollingMetricsColumn(ProgressColumn):
        """Custom Rich ProgressColumn that displays real-time scoreboard & speed."""
        def __init__(self, stats_ref: Dict[str, Any]):
            super().__init__()
            self.stats = stats_ref

        def render(self, task: "Task") -> "Text":
            rem = self.stats.get("remaining", 0)
            p = self.stats.get("passed", 0)
            s = self.stats.get("synth", 0)
            a = self.stats.get("arch", 0)
            i = self.stats.get("impl", 0)
            rate = self.stats.get("rate", 0.0)
            txt = Text()
            txt.append(f"Rem:{rem} ", style="bold yellow")
            txt.append("│ ", style="dim")
            sus = self.stats.get("suspect", 0)
            txt.append(f"✓ verified:{p} ", style="bold green")
            if s > 0:
                txt.append(f"emitted:{s} ", style="cyan")
            if sus > 0:
                txt.append(f"⚠ unverified:{sus} ", style="bold yellow")
            if a > 0:
                txt.append(f"refused:{a} ", style="magenta")
            if i > 0:
                txt.append(f"⚙ defects:{i} ", style="bold red")
            txt.append("│ ", style="dim")
            txt.append(f"{rate:.0f}% verified", style="bold green" if rate >= 80 else "yellow")
            return txt
else:
    RollingMetricsColumn = None  # type: ignore


def render_hero_banner(
    mode: str,
    target_profiles: List[str],
    total_matrix_runs: int,
    prompts_to_test: List[Tuple[str, str]],
    cache_layers: bool,
    explain_plan: bool,
    reranker_info: str,
    output_dir: Path,
    timeout: float,
    dry_run: bool,
    show_code: bool,
    full_code: bool,
    code_max_lines: int,
) -> None:
    """Renders high-visibility startup banner with all configuration telemetry."""
    if not (RICH_AVAILABLE and console):
        return

    banner_tbl = Table(box=None, expand=True, padding=(0, 1), show_header=False)
    banner_tbl.add_column("Key", style="bold cyan", width=22)
    banner_tbl.add_column("Val", style="bold white")

    banner_tbl.add_row("Evaluation Mode", f"[bold yellow]{mode.upper()}[/bold yellow]")
    banner_tbl.add_row("Target Profiles", f"[cyan]{', '.join(target_profiles)}[/cyan]")
    banner_tbl.add_row("Runs per Pass", f"[green]{total_matrix_runs} tests[/green]")
    banner_tbl.add_row("Prompt Presets", f"[white]{len(prompts_to_test)} preset(s)[/white]")
    banner_tbl.add_row("In-Process Cache", "[bold green]ENABLED (Lattice + models resident; route cache cold per run unless --warm-route-cache)[/bold green]" if cache_layers else "[red]DISABLED (Subprocess per test)[/red]")
    banner_tbl.add_row("Diagnostic Telemetry", "[green]--explain-plan ENABLED[/green]" if explain_plan else "[dim]DISABLED[/dim]")
    banner_tbl.add_row("Neural Reranker", f"[magenta]{reranker_info}[/magenta]")
    code_mode_desc = "ENABLED (Full Uncut Code)" if full_code else f"ENABLED (Max lines: {code_max_lines})"
    banner_tbl.add_row("Live Code Preview", f"[bold green]{code_mode_desc}[/bold green]" if show_code else "[dim]DISABLED[/dim]")
    banner_tbl.add_row("Timeout per test", f"[yellow]{timeout}s[/yellow]")
    banner_tbl.add_row("Logs Directory", f"[dim]{output_dir}[/dim]")

    prompt_tbl = Table(title="Prompts Scheduled for Evaluation", title_style="bold cyan", border_style="dim", expand=True)
    prompt_tbl.add_column("Preset", style="bold yellow", width=20)
    prompt_tbl.add_column("Prompt Specification", style="white")

    for pk, pt in prompts_to_test:
        display_text = f"{pt[:80]}..." if len(pt) > 80 else pt
        prompt_tbl.add_row(escape_markup(f"[{pk}]"), escape_markup(display_text))

    panel = Panel(
        Group(
            Text("NSTL EMPIRICAL MULTI-PROFILE & MULTI-METHOD TEST HARNESS", style="bold white on #1f4068", justify="center"),
            Text(""),
            banner_tbl,
            Text(""),
            prompt_tbl,
        ),
        title="[bold yellow]🚀 NSTL TEST BENCHMARK[/bold yellow]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print(panel)


def render_test_card(
    res: Dict[str, Any],
    test_idx: int,
    total_tests: int,
    pass_label: str,
    preset_key: str,
    show_code: bool = True,
    full_code: bool = False,
    code_max_lines: int = 35,
) -> Any:
    """Renders syntax-highlighted synthesized Python code and live execution telemetry in a Rich card."""
    if not (RICH_AVAILABLE and console):
        return None

    prof = res.get("profile", "")
    meth = res.get("method", "")
    status = res.get("status", "UNKNOWN")
    category = res.get("category", "UNKNOWN")
    detail = res.get("detail", "")
    elapsed = res.get("elapsed", 0.0)
    latencies = res.get("latencies", {})
    code = res.get("code", "")
    sandbox_res = res.get("sandbox_res", {})
    log_path = res.get("flat_log_path") or res.get("nested_log_path", "")

    # Status badge styling (one badge per evidence-based category; no catch-all "architectural")
    badge_styles = {
        "SUCCESS": ("[bold white on #1e7e34]  ✓ {s}  [/bold white on #1e7e34]", "green"),
        "SUSPECT": ("[bold black on #ffc107]  ⚠ {s}  [/bold black on #ffc107]", "yellow"),
        "VERIFIER_REJECTED": ("[bold white on #6f42c1]  🔎 VERIFIER REJECTED  [/bold white on #6f42c1]", "magenta"),
        "BINDING_UNRESOLVED": ("[bold white on #fd7e14]  🔌 UNRESOLVED PORTS  [/bold white on #fd7e14]", "dark_orange"),
        "PLANNING": ("[bold white on #20c997]  🧭 {s}  [/bold white on #20c997]", "cyan"),
        "IMPLEMENTATION": ("[bold white on #d9534f]  ⚙️ IMPLEMENTATION  [/bold white on #d9534f]", "red"),
        "SECURITY": ("[bold black on #ffc107]  🛡️ SECURITY  [/bold black on #ffc107]", "yellow"),
        "ENVIRONMENT": ("[bold white on #e83e8c]  ⏳ {s}  [/bold white on #e83e8c]", "dark_red"),
        "EXPECTED_NEGATIVE": ("[bold white on #6c757d]  ∅ EXPECTED REFUSAL  [/bold white on #6c757d]", "bright_black"),
    }
    badge_tpl, border_color = badge_styles.get(category, ("[bold white on #6c757d]  ✗ {s}  [/bold white on #6c757d]", "bright_black"))
    badge = badge_tpl.format(s=status)

    # Latency breakdown badges
    tot_ms = latencies.get("total_ms", elapsed * 1000.0)
    timing_parts = [f"[bold white]{tot_ms:.1f}ms[/bold white]"]
    if "route_ms" in latencies:
        timing_parts.append(f"[cyan]Route:[/cyan] {latencies['route_ms']:.1f}ms")
    if "plan_ms" in latencies:
        timing_parts.append(f"[yellow]Plan:[/yellow] {latencies['plan_ms']:.1f}ms")
    if "synth_ms" in latencies:
        timing_parts.append(f"[green]Synth:[/green] {latencies['synth_ms']:.1f}ms")
    if "sandbox_ms" in latencies:
        timing_parts.append(f"[blue]SB:[/blue] {latencies['sandbox_ms']:.1f}ms")
    if "repair_ms" in latencies and latencies["repair_ms"] > 0:
        timing_parts.append(f"[bold magenta]Repair:[/bold magenta] {latencies['repair_ms']:.1f}ms")

    timing_line = " │ ".join(timing_parts)

    # Header table
    header_table = Table(box=None, expand=True, padding=(0, 1), show_header=False)
    header_table.add_column("Left", justify="left")
    header_table.add_column("Right", justify="right")

    # Long diagnostics never go in the header row (Rich ellipsizes them); they get their own wrapped panel below.
    left_markup = f"[bold cyan]Profile {prof}[/bold cyan] │ [bold yellow]Method {meth}[/bold yellow] │ {badge}"
    header_table.add_row(left_markup, timing_line)

    body_elements: List[Any] = [header_table]

    # Synthesized Code Display
    if code and show_code:
        code_lines = code.splitlines()
        truncated = False
        display_code = code
        if not full_code and len(code_lines) > code_max_lines:
            truncated = True
            display_code = "\n".join(code_lines[:code_max_lines])

        syntax_widget = Syntax(
            display_code,
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        code_title = f"[bold green]✨ Synthesized Python Code ({len(code_lines)} lines)[/bold green]"
        code_panel = Panel(syntax_widget, title=code_title, border_style="green", padding=(0, 1))
        body_elements.append(code_panel)

        if truncated:
            hidden_cnt = len(code_lines) - code_max_lines
            body_elements.append(
                Text.from_markup(f"[dim yellow]... (+{hidden_cnt} lines truncated. Pass --full-code or inspect individual log to view all)[/dim yellow]")
            )

        # Sandbox execution info
        sb_success = sandbox_res.get("success", False)
        sb_rc = sandbox_res.get("returncode", 0)
        sb_stdout = str(sandbox_res.get("stdout") or "").strip()
        sb_stderr = str(sandbox_res.get("stderr") or "").strip()

        if status == "PASSED" or (sb_success and sb_rc == 0):
            sb_msg = f"[bold green]✓ Sandbox Live Execution Succeeded[/bold green] [dim](Exit Code 0)[/dim]"
            if sb_stdout:
                first_lines = "\n".join(sb_stdout.splitlines()[:2])
                sb_msg += f"\n[dim green]Output: {escape_markup(first_lines)}[/dim green]"
            body_elements.append(Text.from_markup(sb_msg))
        elif sandbox_res:
            sb_msg = f"[bold red]✗ Sandbox Execution Failed[/bold red] [dim](Exit Code {sb_rc})[/dim]"
            if sb_stderr:
                err_snippet = "\n".join(sb_stderr.splitlines()[:3])
                sb_msg += f"\n[dim red]Error: {escape_markup(err_snippet)}[/dim red]"
            body_elements.append(Text.from_markup(sb_msg))

    if not code or status not in ("PASSED", "SYNTHESIZED"):
        failure = res.get("failure") or {}
        lines = [f"[bold yellow]{'No Synthesized Code Emitted' if not code else 'Run did not verify'}[/bold yellow]",
                 f"[dim]• Category:       {escape_markup(category)}",
                 f"• Pipeline Stage: {escape_markup(str(failure.get('stage') or status))}  ({escape_markup(status)})[/dim]"]
        viols = failure.get("violations") or []
        ports = failure.get("unresolved_ports") or []
        if viols:
            lines.append(f"[dim]• Verifier violations ({len(viols)}):[/dim]")
            lines += [f"[dim]    - {escape_markup(str(v))}[/dim]" for v in viols]
        elif ports:
            lines.append(f"[dim]• Unresolved ports ({len(ports)}):[/dim]")
            lines += [f"[dim]    - {escape_markup(str(v))}[/dim]" for v in ports]
        else:
            lines.append(f"[dim]• Diagnostic:     {escape_markup(str(detail))}[/dim]")
        ver = res.get("verification") or {}
        if ver.get("missing"):
            lines.append(f"[dim]• Prompt literals not consumed by the program: {escape_markup(', '.join(ver['missing']))}[/dim]")
        body_elements.append(Panel(Text.from_markup("\n".join(lines)), border_style="yellow", padding=(0, 1)))

    if log_path:
        body_elements.append(Text.from_markup(f"[dim]Log file: {escape_markup(str(log_path))}[/dim]"))

    card_title = f"[bold white on blue] Test {test_idx}/{total_tests} [/bold white on blue] [bold]{escape_markup(preset_key.upper())}[/bold] [dim]({escape_markup(pass_label)})[/dim]"
    return Panel(Group(*body_elements), title=card_title, border_style=border_color, padding=(0, 1))


def configure_headless_env() -> None:
    """Configures headless environment variables for GUI, game loops, and audio."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def ensure_input_csv(root_dir: Path) -> Path:
    """
    Ensures a valid 'input.csv' exists in the working directory so that
    GEVR sandbox live execution can locate the source dataset if executed.
    """
    csv_file = root_dir / "input.csv"
    if not csv_file.exists():
        lines = ["X,Y\n"]
        for i in range(100):
            if i in (12, 45, 78):
                lines.append(",\n")
            else:
                x_val = f"{i * 0.1:.3f}"
                y_val = f"{5.0 + (i % 7) * 0.5:.3f}"
                lines.append(f"{x_val},{y_val}\n")
        csv_file.write_text("".join(lines), encoding="utf-8")
        print(f"[*] Created synthetic dataset: '{csv_file}' (columns: X, Y; 100 rows with nulls)")
    return csv_file


def ensure_input_image(root_dir: Path) -> Path:
    """
    Ensures a valid 'input.png' exists for computer vision / CV2 / PIL pipelines.
    """
    img_file = root_dir / "input.png"
    if not img_file.exists():
        try:
            import cv2
            import numpy as np
            synthetic_img = np.zeros((128, 128, 3), dtype=np.uint8)
            synthetic_img[16:112, 16:112] = [200, 150, 100]
            synthetic_img[32:96, 32:96] = [50, 180, 240]
            cv2.imwrite(str(img_file), synthetic_img)
            print(f"[*] Created synthetic test image: '{img_file}' (128x128 RGB)")
        except Exception:
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
                b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            img_file.write_bytes(png_bytes)
            print(f"[*] Created fallback 1x1 PNG: '{img_file}'")
    return img_file


def ensure_input_audio(root_dir: Path) -> Path:
    """
    Ensures a valid 16-bit PCM 'input_1.wav' exists for audio DSP pipelines.
    """
    wav_file = root_dir / "input_1.wav"
    if not wav_file.exists():
        import math
        import struct
        import wave

        sample_rate = 16000
        duration_sec = 0.5
        n_samples = int(sample_rate * duration_sec)
        freq = 440.0
        with wave.open(str(wav_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                sample = int(16000 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
                frames.extend(struct.pack("<h", sample))
            wf.writeframes(bytes(frames))
        print(f"[*] Created synthetic test audio: '{wav_file}' (16kHz mono PCM, {duration_sec}s)")
    return wav_file


def extract_pipeline_latencies(output: str) -> Dict[str, float]:
    """Extracts granular pipeline layer latencies (ms) from debug telemetry."""
    lat: Dict[str, float] = {}
    m_tot = re.search(r"Total End-to-End Latency.*?([0-9.]+)\s*ms", output)
    if m_tot:
        lat["total_ms"] = float(m_tot.group(1))
    m_r = re.search(r"Layer 1: Semantic Routing.*?([0-9.]+)\s*ms", output)
    if m_r:
        lat["route_ms"] = float(m_r.group(1))
    m_p = re.search(r"Layer 2: Topological Planning.*?([0-9.]+)\s*ms", output)
    if m_p:
        lat["plan_ms"] = float(m_p.group(1))
    m_u = re.search(r"Layer 3: Monadic Unification.*?([0-9.]+)\s*ms", output)
    if m_u:
        lat["synth_ms"] = float(m_u.group(1))
    m_s = re.search(r"Layer 4: GEVR Sandbox Execution.*?([0-9.]+)\s*ms", output)
    if m_s:
        lat["sandbox_ms"] = float(m_s.group(1))
    m_rep = re.search(r"Layer 5: LLM Self-Repair Cycle.*?([0-9.]+)\s*ms", output)
    if m_rep:
        lat["repair_ms"] = float(m_rep.group(1))
    return lat


def categorize_status(status: str) -> str:
    """
    Maps a run status to an evidence-based category (see eval_metrics.STATUS_CATEGORY).
    Deliberately has no catch-all "architectural" bucket: VERIFIER_REJECTED / BINDING_UNRESOLVED / PLANNING /
    IMPLEMENTATION each name what actually happened.
    """
    return EM.category_of(status)


# ==============================================================================
# In-Process Multi-Layer Caching Engine
# ==============================================================================

class CachedNSTLRunner:
    """
    High-performance in-process execution engine for NSTL.
    Caches:
      1. Global Lattice Graph: LatticeOrchestrator and loaded cells (1,023 nodes) parsed once.
      2. Neural Models: ModelManager resident weights (SentenceTransformers, Qwen GGUF LLMs) kept in RAM/VRAM.
      3. Layer 1 Semantic Tunnel: Query spans, RAG embeddings, and FAISS lookups cached across methods of same profile.
    Avoids repeated model/lattice loading. The Layer-1 route cache is cleared per run by default (--warm-route-cache to keep it).
    """
    _instance: Optional["CachedNSTLRunner"] = None

    def __init__(self, root_dir: Path, db_path: Optional[Path] = None):
        self.root_dir = root_dir
        self.db_path = str(db_path or (root_dir / "trees" / "lattice.db"))
        self.trees_dir = str(root_dir / "trees")
        self.shells: Dict[str, Any] = {}
        self.shared_orchestrator: Optional[Any] = None
        self._init_orchestrator()

    @classmethod
    def get_instance(cls, root_dir: Path, db_path: Optional[Path] = None) -> "CachedNSTLRunner":
        if cls._instance is None:
            cls._instance = cls(root_dir, db_path)
        return cls._instance

    def _init_orchestrator(self) -> None:
        """Initializes global LatticeOrchestrator once."""
        if str(self.root_dir / "src") not in sys.path:
            sys.path.insert(0, str(self.root_dir / "src"))
        try:
            from cli import ensure_lattice_compiled, LatticeOrchestrator
            ensure_lattice_compiled(trees_dir=self.trees_dir, db_path=self.db_path)
            self.shared_orchestrator = LatticeOrchestrator(trees_directory=self.trees_dir, db_path=self.db_path, auto_load=False)
            self.shared_orchestrator.load_from_database(self.db_path)
            self.shared_orchestrator.build_topology()
        except Exception as e:
            print(f"[!] Warning: Failed to pre-initialize LatticeOrchestrator: {e}")

    def get_or_create_shell(
        self,
        profile: str,
        embedder: Optional[str] = None,
        llm: Optional[str] = None,
        device: Optional[str] = None,
        reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        macros: Optional[bool] = None,
        topology: Optional[str] = None,
        dev: bool = False,
        no_lint: bool = False,
        no_exec: bool = False,
        timeout: float = 180.0,
        allowed_capabilities: str = "",
        llm_feedback: bool = False,
        edge_mode: Optional[str] = None,
    ) -> Any:
        import cli
        from config import settings
        from experiments import ACTIVE_EXPERIMENTS

        prof = str(profile).strip().upper()
        os.environ["NSTL_PROFILE"] = prof
        settings.profile = prof

        if llm_feedback:
            os.environ["NSTL_LLM_FEEDBACK"] = "1"
        else:
            os.environ.pop("NSTL_LLM_FEEDBACK", None)

        if allowed_capabilities:
            os.environ["NSTL_ALLOWED_CAPABILITIES"] = str(allowed_capabilities)

        if reranker is not None:
            settings.use_reranker = bool(reranker)
            os.environ["NSTL_USE_RERANKER"] = "1" if reranker else "0"

        if reranker_model:
            settings.reranker_model = str(reranker_model)
            os.environ["NSTL_RERANKER_MODEL"] = str(reranker_model)

        if macros is not None:
            settings.macros_enabled = bool(macros)
        if dev:
            settings.dev_mode = True
        if topology:
            settings.topology_mode = str(topology).lower()
        em = str(edge_mode).lower() if edge_mode else None
        if em:
            settings.edge_mode = em
            ACTIVE_EXPERIMENTS.edge_mode = em
            if self.shared_orchestrator is not None and getattr(self.shared_orchestrator, "edge_mode", None) != em:
                self.shared_orchestrator.build_topology(edge_mode=em)

        shell = self.shells.get(prof)
        if shell is None:
            shell = cli.NSTLInteractiveShell(
                db_path=self.db_path,
                initial_profile=prof,
                embedder=embedder or "",
                llm=llm or "",
                device=device or "auto",
                debug=True,
                interactive=False,
                no_exec=no_exec,
                no_lint=no_lint,
                macros=macros,
                topology=topology,
                edge_mode=edge_mode,
                dev=dev,
                timeout=timeout,
                reranker=reranker,
                reranker_model=reranker_model,
                exec_sandbox=not no_exec,
                llm_feedback=llm_feedback,
                allowed_capabilities=allowed_capabilities,
            )
            if self.shared_orchestrator is not None and shell.orchestrator is not self.shared_orchestrator:
                shell.orchestrator = self.shared_orchestrator
                shell.gate.orchestrator = self.shared_orchestrator
                if shell.router:
                    shell.router.orchestrator = self.shared_orchestrator
            if shell.active_profile == prof:
                self.shells[prof] = shell
            else:
                self.shells.pop(prof, None)
        else:
            shell.no_exec = no_exec
            shell.no_lint = no_lint
            shell.timeout = timeout
            shell.llm_feedback = llm_feedback
            if reranker is not None:
                shell.use_reranker = bool(reranker)
                if shell.router:
                    shell.router.use_reranker = bool(reranker)
            if reranker_model and shell.router:
                shell.router.reranker_model = str(reranker_model)
            if topology and shell.router and hasattr(shell.router, "planner"):
                shell.router.planner.topology_mode = str(topology).lower()
            if em and shell.orchestrator and getattr(shell.orchestrator, "edge_mode", None) != em:
                shell.orchestrator.build_topology(edge_mode=em)

        # Ensure ModelManager and LatentVectorExtractor reflect active profile
        try:
            from inference import ModelManager
            from cognitive_pathfinder import LatentVectorExtractor
            LatentVectorExtractor.reset_instance()
            mm = ModelManager.get_instance()
            if mm.current_profile_name != prof:
                shell._switch_profile(
                    prof,
                    embedder=embedder or getattr(shell, "embedder_name", "") or "",
                    llm=llm or getattr(shell, "llm_name", "") or "",
                    device=device or "auto",
                    verbose=False,
                )
        except Exception:
            pass

        return shell

    def run_test(
        self,
        profile: str,
        method: str,
        prompt: str,
        llm_feedback: bool = False,
        no_lint: bool = False,
        no_exec: bool = False,
        timeout: float = 180.0,
        reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        macros: Optional[bool] = None,
        topology: Optional[str] = None,
        dev: bool = False,
        explain_plan: bool = True,
        allow_capabilities: str = "",
        embedder: Optional[str] = None,
        llm: Optional[str] = None,
        device: Optional[str] = None,
        edge_mode: Optional[str] = None,
        cold_route: bool = True,
    ) -> Tuple[int, str, float, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Executes a test in-process, capturing console and stdout into raw_output.
        Returns: (returncode, raw_output, elapsed_seconds, gen_code, sandbox_result, record)
        `record` is the engine's structured run record for THIS run (None if the run logged nothing).
        `cold_route=True` clears the Layer-1 route cache first, so every run's routing latency and result are
        measured rather than replayed from an earlier pass. Models and the lattice stay resident.
        """
        import cli
        from config import settings

        settings.explain_plan = bool(explain_plan)

        shell = self.get_or_create_shell(
            profile=profile,
            embedder=embedder,
            llm=llm,
            device=device,
            reranker=reranker,
            reranker_model=reranker_model,
            macros=macros,
            topology=topology,
            dev=dev,
            no_lint=no_lint,
            no_exec=no_exec,
            timeout=timeout,
            allowed_capabilities=allow_capabilities,
            llm_feedback=llm_feedback,
            edge_mode=edge_mode,
        )

        requested_prof = "O" if str(profile).strip().upper() in ("OMEGA", "Ω") else str(profile).strip().upper()
        if str(getattr(shell, "active_profile", "")).strip().upper() != requested_prof:
            self.shells.pop(requested_prof, None)
            err_detail = getattr(shell, "last_profile_load_error", None) or (
                f"Active profile remained {getattr(shell, 'active_profile', 'unknown')}"
            )
            msg = f"[PROFILE_LOAD_FAILED] Profile {requested_prof} failed to load: {err_detail}\n"
            return 4, msg, 0.0, "", {}, None

        shell.route_method = method.upper()
        if shell.router:
            shell.router.default_route_method = method.lower()

        if cold_route and getattr(shell, "router", None) is not None and hasattr(shell.router, "clear_route_cache"):
            shell.router.clear_route_cache()
        history_len_before = len(shell.history)

        buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_file = cli.console.file
        t0 = time.time()
        timed_out = False
        old_alarm_handler = None
        has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
        if has_alarm and timeout and timeout > 0:
            def _alarm_handler(signum, frame):
                raise TimeoutError(f"Execution timed out after {timeout} seconds")
            old_alarm_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(timeout))

        try:
            sys.stdout = buf
            sys.stderr = buf
            cli.console.file = buf
            shell.default(f"{prompt} --debug")
        except TimeoutError as exc:
            timed_out = True
            buf.write(f"\n[ERROR] In-process execution timed out after {timeout} seconds.\n")
        except Exception as exc:
            buf.write(f"\n[ERROR] In-process execution exception: {exc}\n")
        finally:
            if has_alarm and timeout and timeout > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if old_alarm_handler is not None:
                    signal.signal(signal.SIGALRM, old_alarm_handler)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            cli.console.file = old_file

        elapsed = round(time.time() - t0, 3)
        raw_output = buf.getvalue()

        if timed_out:
            return -1, raw_output, elapsed, "", {}, None

        returncode = 0
        gen_code = ""
        sandbox_res: Dict[str, Any] = {}
        record: Optional[Dict[str, Any]] = None
        # Only trust a history entry written by THIS run. Without this guard a run that dies before logging
        # would inherit the previous test's code and sandbox result and be scored on someone else's output.
        if len(shell.history) > history_len_before:
            last = shell.history[-1]
            record = dict(last)
            gen_code = str(last.get("code") or "")
            sandbox_res = dict(last.get("sandbox_result") or {})
            if last.get("internal_error"):
                returncode = 1
            elif not last.get("code"):
                returncode = 2
            elif not no_exec and not last.get("sandbox_result", {}).get("success", False):
                returncode = 3
        else:
            returncode = 1
            buf.write("\n[ERROR] Run produced no history record (engine aborted before logging a result).\n")
            raw_output = buf.getvalue()

        return returncode, raw_output, elapsed, gen_code, sandbox_res, record


# ==============================================================================
# Single Test Execution Driver
# ==============================================================================

def run_single_test(
    profile: str,
    method: str,
    prompt: str,
    output_dir: Path,
    total_log_path: Path,
    timeout: float = 180.0,
    dry_run: bool = False,
    reranker: Optional[bool] = None,
    reranker_model: Optional[str] = None,
    macros: Optional[bool] = None,
    topology: Optional[str] = None,
    dev: bool = False,
    no_lint: bool = False,
    no_exec: bool = False,
    explain_plan: bool = True,
    early_exit_layer: Optional[int] = None,
    no_whitening: bool = False,
    sparse_sae: bool = False,
    no_markov_prior: bool = False,
    edge_mode: Optional[str] = None,
    llm_feedback: bool = False,
    allow_capabilities: str = "",
    embedder: Optional[str] = None,
    llm: Optional[str] = None,
    device: Optional[str] = None,
    pass_label: str = "",
    use_cache: bool = True,
    runner: Optional[CachedNSTLRunner] = None,
    cold_route: bool = True,
    expected: str = "synthesize",
    verbose_plain: bool = False,
    show_code: bool = True,
    full_code: bool = False,
    code_max_lines: int = 35,
) -> Dict[str, Any]:
    """
    Executes a single test run of `nstl_cli.py run <prompt> --profile <profile> --route-method <method> --debug`
    using either the in-process cached runner (fast) or an isolated subprocess (fallback).
    """
    cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "run",
        prompt,
        "--profile",
        profile,
        "--route-method",
        method,
        "--debug",
    ]

    if llm_feedback:
        cmd.append("--llm-feedback")
    if allow_capabilities:
        cmd.extend(["--allow-capabilities", str(allow_capabilities)])
    if embedder:
        cmd.extend(["--embedder", str(embedder)])
    if llm:
        cmd.extend(["--llm", str(llm)])
    if device:
        cmd.extend(["--device", str(device)])
    if reranker is True:
        cmd.append("--reranker")
    elif reranker is False:
        cmd.append("--no-reranker")
    if reranker_model:
        cmd.extend(["--reranker-model", str(reranker_model)])
    if macros is True:
        cmd.append("--macros")
    elif macros is False:
        cmd.append("--no-macros")
    if topology:
        cmd.extend(["--topology", str(topology)])
    if dev:
        cmd.append("--dev")
    if no_lint:
        cmd.append("--no-lint")
    if no_exec:
        cmd.append("--no-exec")
    if explain_plan:
        cmd.append("--explain-plan")
    if early_exit_layer is not None:
        cmd.extend(["--early-exit-layer", str(early_exit_layer)])
    if no_whitening:
        cmd.append("--no-whitening")
    if sparse_sae:
        cmd.append("--sparse-sae")
    if no_markov_prior:
        cmd.append("--no-markov-prior")
    if edge_mode:
        cmd.extend(["--edge-mode", str(edge_mode).lower()])

    fb_str = "ENABLED (--llm-feedback)" if llm_feedback else "DISABLED (Deterministic)"
    mode_str = "CACHED (In-Process)" if (use_cache and runner is not None) else "SUBPROCESS"
    pass_prefix = f"{pass_label}_" if pass_label else ""
    tag = f"{pass_prefix}profile_{profile}_method_{method}"
    if verbose_plain:
        print(f"\n{'=' * 78}")
        print(f"[*] RUNNING: Profile {profile} | RouteMethod {method} | Mode: {mode_str} | Feedback: {fb_str}")
        print(f"[*] Command: {' '.join(cmd)}")
        print(f"{'=' * 78}")

    target_dir = output_dir / pass_label if pass_label else output_dir
    profile_dir = target_dir / f"profile_{profile}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    flat_log_path = target_dir / f"{tag}.log"
    nested_log_path = profile_dir / f"method_{method}.log"

    banner = (
        f"================================================================================\n"
        f"PROFILE: {profile} | METHOD: {method} | MODE: {mode_str} | LLM FEEDBACK: {fb_str}\n"
        f"PROMPT: {prompt}\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"================================================================================\n\n"
    )

    if dry_run:
        if verbose_plain:
            print(f"[DRY RUN] Would execute test and record output to {total_log_path}")
        return {
            "profile": profile,
            "method": method,
            "llm_feedback": llm_feedback,
            "status": "DRY_RUN",
            "category": "DRY_RUN",
            "detail": "Skipped execution (Dry Run)",
            "elapsed": 0.0,
            "latencies": {},
            "code": "",
            "sandbox_res": {},
            "flat_log_path": flat_log_path,
            "nested_log_path": nested_log_path,
        }

    if method.upper() in ("M4", "M5", "M7", "M8", "M9") and method.upper() not in SUPPORTED_METHODS.get(str(profile), ALL_METHODS):
        clean_output = f"[THEORETICAL_REFUSAL_OK] Route method {method.upper()} requires generative LLM synthesis which is disabled in Profile {profile}."
        flat_log_path.write_text(clean_output, encoding="utf-8")
        nested_log_path.write_text(clean_output, encoding="utf-8")
        with open(total_log_path, "a", encoding="utf-8") as tf:
            tf.write(banner)
            tf.write(clean_output)
            tf.write("\n\n\n")
        return {
            "profile": profile,
            "method": method,
            "status": "THEORETICAL_REFUSAL_OK",
            "detail": f"Method {method} unsupported in Profile {profile} (Expected Negative Control)",
            "elapsed": 0.001,
            "latencies": {},
            "code": "",
            "sandbox_res": {},
            "flat_log_path": flat_log_path,
            "nested_log_path": nested_log_path,
        }

    raw_output = ""
    returncode = 0
    elapsed = 0.0
    gen_code = ""
    sandbox_res: Dict[str, Any] = {}
    record: Optional[Dict[str, Any]] = None

    # 1. Attempt cached in-process execution if enabled
    if use_cache and runner is not None:
        try:
            returncode, raw_output, elapsed, gen_code, sandbox_res, record = runner.run_test(
                profile=profile,
                method=method,
                prompt=prompt,
                llm_feedback=llm_feedback,
                no_lint=no_lint,
                no_exec=no_exec,
                timeout=timeout,
                reranker=reranker,
                reranker_model=reranker_model,
                macros=macros,
                topology=topology,
                dev=dev,
                explain_plan=explain_plan,
                allow_capabilities=allow_capabilities,
                embedder=embedder,
                llm=llm,
                device=device,
                edge_mode=edge_mode,
                cold_route=cold_route,
            )
        except Exception as exc:
            if verbose_plain:
                print(f"[!] In-process runner encountered error: {exc}. Falling back to subprocess...")
            raw_output = ""
            record = None

    # 2. Subprocess fallback if in-process is disabled or encountered failure
    if not raw_output:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        configure_headless_env()

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
            elapsed = round(time.time() - t0, 3)
            raw_output = proc.stdout or ""
            returncode = proc.returncode
            gen_code = extract_code_from_output(raw_output)
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.time() - t0, 3)
            out_str = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_output = out_str + f"\n\n[ERROR] Command timed out after {timeout} seconds."
            returncode = -1
        except Exception as exc:
            elapsed = round(time.time() - t0, 3)
            raw_output = f"[ERROR] Subprocess launch failed: {exc}"
            returncode = -2

    clean_output = strip_ansi(raw_output)

    # Fallback extraction if runner didn't have code directly
    if not gen_code:
        gen_code = extract_code_from_output(clean_output)

    # Write individual pure text logs
    flat_log_path.write_text(clean_output, encoding="utf-8")
    nested_log_path.write_text(clean_output, encoding="utf-8")

    # Append to combined total.log
    with open(total_log_path, "a", encoding="utf-8") as tf:
        tf.write(banner)
        tf.write(clean_output.rstrip())
        tf.write("\n\n\n")

    # Status detection. Primary source: the engine's structured failure record (exact, untruncated, no text
    # scraping). The text-pattern chain below is only the fallback for subprocess runs that have no record.
    structured = EM.classify_from_record(record) if record else None
    failure_info: Dict[str, Any] = {}
    verification: Dict[str, Any] = {}
    if returncode == -1:
        status = "TIMEOUT"
        detail = f"Timed out after {timeout}s"
    elif method not in SUPPORTED_METHODS.get(str(profile), ALL_METHODS):
        status = "THEORETICAL_REFUSAL_OK"
        detail = f"Method {method} unsupported in Profile {profile} (Expected Negative Control)"
    elif structured is not None:
        status, detail, failure_info = structured
        if status == "SANDBOX_FAILED" and (
            "SecurityPolicyError" in clean_output or "Security violation" in clean_output
            or "Execution blocked by security floor" in clean_output or "Capability denied" in clean_output
        ):
            status, detail = "SECURITY_BLOCKED", "Blocked by Security Floor / Capability Policy"
        elif status == "PASSED" and not no_exec:
            status, detail, verification = EM.finalize_success(status, detail, prompt, gen_code, record)
        elif status == "PASSED" and no_exec:
            status, detail = "SYNTHESIZED", "Code emitted (execution skipped by --no-exec)"
    elif "INTERNAL ERROR" in clean_output or "Internal Engine Error" in clean_output:
        status = "CRASHED"
        tb_lines = [line.strip() for line in clean_output.splitlines() if line.strip()]
        last_err = next((line for line in tb_lines if any(k in line for k in ("INTERNAL ERROR", "NameError:", "Error:", "Exception:"))), None)
        detail = last_err if last_err else "Internal Engine Exception"
    elif "Status: ✓ PASSED" in clean_output or "Post-Repair Result: PASSED" in clean_output:
        status = "PASSED"
        detail = "Live Sandbox Execution Verified"
        status, detail, verification = EM.finalize_success(status, detail, prompt, gen_code or extract_code_from_output(clean_output), None)
    elif (
        "SecurityPolicyError" in clean_output
        or "Security violation" in clean_output
        or "Execution blocked by security floor" in clean_output
        or "Capability denied" in clean_output
    ):
        status = "SECURITY_BLOCKED"
        detail = "Blocked by Security Floor / Capability Policy"
    elif "PLANNING FAILURE" in clean_output:
        status = "PLANNING_FAILED"
        detail = "No valid compositional path through lattice"
    elif "UNIFICATION FAILED" in clean_output:
        status = "UNRESOLVED_PORT"
        m_fail = re.search(r"UNIFICATION FAILED:\s*(.*)", clean_output)
        detail = m_fail.group(1).strip() if m_fail else "Type-Monadic Unification Failure"
    elif "SYNTHESIS REFUSED" in clean_output:
        status = "UNRESOLVED_PORT"
        m_ref = re.search(r"SYNTHESIS REFUSED:\s*(.*)", clean_output)
        detail = m_ref.group(1).strip() if m_ref else "Synthesis Refused"
    elif "<UNRESOLVED_PORT>" in clean_output or "UnresolvedPlaceholderError" in clean_output:
        status = "UNRESOLVED_PORT"
        detail = "Layer 3 Port Unification Failure"
    elif "CONTRACT VIOLATION" in clean_output or "Algebraic contract violation" in clean_output:
        status = "CONTRACT_VIOLATION"
        detail = "Verification Contract Violated"
    elif "pre-flight lint violations" in clean_output.lower():
        status = "LINT_HALTED"
        detail = "Pre-flight Linter Violations"
    elif "[PROFILE_LOAD_FAILED]" in clean_output or "Failed to load Profile " in clean_output:
        status = "PROFILE_LOAD_FAILED"
        m_load = re.search(r"(?:\[PROFILE_LOAD_FAILED\]|Failed to load Profile [^:]+:)\s*(.*)", clean_output)
        detail = m_load.group(1).strip() if m_load else f"Profile {profile} failed to load"
    elif "ModelUnavailableError" in clean_output:
        status = "MODEL_UNAVAILABLE"
        detail = "Required Neural Model Not Available"
    elif "Status: ✗ FAILED" in clean_output or "Post-Repair Result: FAILED" in clean_output:
        status = "SANDBOX_FAILED"
        detail = "Runtime Exception During Execution"
    elif "✨ Synthesized Python Code" in clean_output:
        status = "SYNTHESIZED"
        detail = "Synthesized Code Generated"
    elif returncode != 0:
        status = "CRASHED"
        tb_lines = [line.strip() for line in clean_output.splitlines() if line.strip()]
        last_err = next((line for line in reversed(tb_lines) if any(kw in line for kw in ("Error:", "Exception:", "Error", "Exception"))), None)
        detail = last_err if last_err else f"CLI process exited with code {returncode}"
    else:
        status = "FAILED"
        detail = "Execution Finished Without Synthesized Code"

    latencies = extract_pipeline_latencies(clean_output)
    if record:
        for src_k, dst_k in (("latency_ms", "total_ms"), ("route_ms", "route_ms"), ("plan_ms", "plan_ms"),
                             ("synth_ms", "synth_ms"), ("sandbox_ms", "sandbox_ms")):
            if record.get(src_k) is not None:
                latencies[dst_k] = float(record[src_k])
    category = categorize_status(status)

    if verbose_plain:
        status_icon = "✓" if status in ("PASSED", "SYNTHESIZED") else ("~" if status == "COMPLETED" else "✗")
        print(f"[{status_icon}] Profile {profile} | Method {method} -> {status} [{category}] ({detail}) [{elapsed}s]")
        if latencies.get("total_ms"):
            r_ms = latencies.get('route_ms', 0)
            p_ms = latencies.get('plan_ms', 0)
            s_ms = latencies.get('synth_ms', 0)
            sb_ms = latencies.get('sandbox_ms', 0)
            rep_ms = latencies.get('repair_ms', 0)
            print(f"    Latency: {latencies['total_ms']:.1f}ms (Route:{r_ms:.1f}ms, Plan:{p_ms:.1f}ms, Synth:{s_ms:.1f}ms, SB:{sb_ms:.1f}ms, Rep:{rep_ms:.1f}ms)")
        if gen_code and show_code:
            code_lines = gen_code.splitlines()
            print(f"    --- Synthesized Python Code ({len(code_lines)} lines) ---")
            lines_to_show = code_lines if full_code else code_lines[:code_max_lines]
            for cl in lines_to_show:
                print(f"    {cl}")
            if not full_code and len(code_lines) > code_max_lines:
                print(f"    ... (+{len(code_lines) - code_max_lines} lines truncated. Pass --full-code to view all)")
            print(f"    ---------------------------------------------------------")
        print(f"    Individual log: {flat_log_path}")
        print(f"    Appended to:    {total_log_path}")

    return {
        "profile": profile,
        "method": method,
        "llm_feedback": llm_feedback,
        "status": status,
        "category": category,
        "detail": detail,
        "elapsed": elapsed,
        "latencies": latencies,
        "prompt": prompt,
        "code": gen_code,
        "sandbox_res": sandbox_res,
        "flat_log_path": flat_log_path,
        "nested_log_path": nested_log_path,
        "clean_output": clean_output,
        "record": record,
        "failure": failure_info,
        "verification": verification,
        "expected": expected,
        "outcome_ok": EM.outcome_ok(status, expected),
        "cold_route": cold_route,
        "repair_attempted": bool((latencies or {}).get("repair_ms", 0)),
    }


# ==============================================================================
# Pure LLM Chat Baseline Runner (No Lattice / RAG / Unification / Repair)
# ==============================================================================

DEFAULT_PURE_SYSTEM_PROMPT = (
    "You are a helpful Python coding assistant. "
    "Respond with a single runnable Python script inside a ```python ... ``` code block "
    "that accomplishes the user's request."
)


class PureLLMRunner:
    """
    Runs ONLY the raw GGUF LLM in plain chat mode (zero NSTL framework:
    no lattice, no embedder/RAG, no unification, no IR compiler, no repair loop),
    reusing a loaded llama_cpp instance when available to save reload time.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._models: Dict[Tuple[str, str], Tuple[Any, Any, str]] = {}

    def _ensure_src_on_path(self) -> None:
        src_str = str(self.root_dir / "src")
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def _get_llm(self, llm_name: Optional[str] = None, device: Optional[str] = None) -> Tuple[Any, Any, str]:
        self._ensure_src_on_path()
        import threading
        from inference import select_optimal_llm, ModelManager, is_split_gguf_shard, MODELS_DIR
        from config import settings
        from llama_cpp import Llama

        resolved_llm = select_optimal_llm(llm_name or "")
        dev = (device or getattr(settings, "device", None) or "auto").strip().lower()
        if dev == "auto":
            try:
                from router import HardwareProfiler
                dev = HardwareProfiler.get_optimal_device()
            except Exception:
                dev = "cpu"

        key = (resolved_llm, dev)
        if key in self._models:
            return self._models[key]

        # Reuse the active llama_cpp.Llama instance if already loaded by ModelManager
        try:
            mm = ModelManager.get_instance()
            ap = getattr(mm, "active_profile", None)
            if ap is not None and getattr(ap, "llm", None) is not None and getattr(ap, "llm_name", "") == resolved_llm:
                lock = getattr(ap, "_llm_lock", None) or threading.Lock()
                self._models[key] = (ap.llm, lock, resolved_llm)
                return self._models[key]
        except Exception:
            pass

        llm_dir = os.path.join(MODELS_DIR, "llms", resolved_llm)
        if not os.path.isdir(llm_dir):
            raise FileNotFoundError(f"LLM directory not found: {llm_dir}")

        ggufs = sorted(f for f in os.listdir(llm_dir) if f.endswith(".gguf"))
        if not ggufs:
            raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
        unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
        chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
        model_file = os.path.join(llm_dir, chosen_gguf)

        gpu_layers = -1 if dev in ("cuda", "mps", "metal", "rocm") else 0
        n_threads = max(1, (os.cpu_count() or 4) - 1)
        llama_obj = Llama(
            model_path=model_file,
            n_ctx=settings.llm_context_length,
            n_gpu_layers=gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        lock = threading.Lock()
        self._models[key] = (llama_obj, lock, resolved_llm)
        return self._models[key]

    def generate_chat_code(
        self,
        prompt: str,
        llm_name: Optional[str] = None,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Tuple[str, str, float, str]:
        self._ensure_src_on_path()
        from config import settings
        from utils import extract_code_from_llm_response

        llama_obj, lock, resolved_llm = self._get_llm(llm_name=llm_name, device=device)
        sys_msg = system_prompt if system_prompt is not None else DEFAULT_PURE_SYSTEM_PROMPT
        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        with lock:
            resp = llama_obj.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                repeat_penalty=1.15,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
        gen_ms = round((time.time() - t0) * 1000.0, 2)
        raw_reply = (resp["choices"][0]["message"]["content"] or "").strip()
        code = extract_code_from_llm_response(raw_reply)
        return raw_reply, code, gen_ms, resolved_llm


def run_pure_llm_test(
    prompt: str,
    output_dir: Path,
    total_log_path: Path,
    pure_runner: PureLLMRunner,
    timeout: float = 180.0,
    dry_run: bool = False,
    no_exec: bool = False,
    allow_capabilities: str = "",
    llm: Optional[str] = None,
    device: Optional[str] = None,
    system_prompt: Optional[str] = None,
    pass_label: str = "pure",
    verbose_plain: bool = False,
    show_code: bool = True,
    full_code: bool = False,
    code_max_lines: int = 35,
) -> Dict[str, Any]:
    """Evaluates raw normal-chat LLM code generation against the same GEVR sandbox."""
    target_dir = output_dir / pass_label if pass_label else output_dir
    profile_dir = target_dir / "profile_PURE"
    profile_dir.mkdir(parents=True, exist_ok=True)
    flat_log_path = target_dir / f"{pass_label}_PURE_CHAT.log"
    nested_log_path = profile_dir / "method_CHAT.log"

    banner = (
        f"================================================================================\n"
        f"PROFILE: PURE (Normal Chat Baseline) | METHOD: CHAT | LLM: {llm or 'auto'}\n"
        f"PROMPT: {prompt}\n"
        f"================================================================================\n\n"
    )

    if dry_run:
        return {
            "profile": "PURE",
            "method": "CHAT",
            "llm_feedback": False,
            "status": "DRY_RUN",
            "category": "DRY_RUN",
            "detail": "Skipped execution (Dry Run)",
            "elapsed": 0.0,
            "latencies": {},
            "prompt": prompt,
            "code": "",
            "sandbox_res": {},
            "flat_log_path": flat_log_path,
            "nested_log_path": nested_log_path,
            "clean_output": "",
        }

    t_start = time.time()
    raw_reply = ""
    gen_code = ""
    gen_ms = 0.0
    sb_ms = 0.0
    resolved_llm = llm or "auto"
    sandbox_res: Dict[str, Any] = {}
    status = "FAILED"
    detail = ""

    try:
        raw_reply, gen_code, gen_ms, resolved_llm = pure_runner.generate_chat_code(
            prompt=prompt,
            llm_name=llm,
            device=device,
            system_prompt=system_prompt,
        )
        if not gen_code.strip():
            status = "FAILED"
            detail = "Pure LLM produced empty code"
        elif no_exec:
            import ast as _ast
            try:
                _ast.parse(gen_code)
                status = "SYNTHESIZED"
                detail = "Syntax-valid Python generated (--no-exec)"
            except SyntaxError as se:
                status = "SANDBOX_FAILED"
                detail = f"SyntaxError: {se}"
        else:
            src_str = str(ROOT_DIR / "src")
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            from gevr_sandbox import GEVRSandbox

            caps = {c.strip() for c in (allow_capabilities or "").split(",") if c.strip()}
            sandbox = GEVRSandbox(default_timeout=timeout, allowed_capabilities=caps)
            t_sb = time.time()
            sandbox_res = sandbox.execute(gen_code, timeout=timeout)
            sb_ms = round((time.time() - t_sb) * 1000.0, 2)

            err_txt = str(sandbox_res.get("error") or sandbox_res.get("stderr") or "").strip()
            if sandbox_res.get("success", False):
                status = "PASSED"
                detail = f"Live Sandbox Execution Verified ({resolved_llm})"
            elif any(k in err_txt for k in ("SecurityPolicyError", "Security violation", "Capability denied")):
                status = "SECURITY_BLOCKED"
                detail = err_txt.splitlines()[-1][:140]
            else:
                status = "SANDBOX_FAILED"
                last_line = err_txt.splitlines()[-1] if err_txt else "Runtime failure"
                detail = last_line[:160]
    except Exception as exc:
        status = "MODEL_UNAVAILABLE" if "FileNotFoundError" in type(exc).__name__ else "CRASHED"
        detail = str(exc)[:160]

    elapsed = round(time.time() - t_start, 3)
    total_ms = round(elapsed * 1000.0, 2)
    latencies = {"synth_ms": gen_ms, "sandbox_ms": sb_ms, "total_ms": total_ms}
    category = categorize_status(status)

    log_body = (
        f"[PURE LLM MODEL]: {resolved_llm}\n"
        f"[STATUS]: {status} [{category}]\n"
        f"[DETAIL]: {detail}\n"
        f"[LATENCIES]: llm_gen={gen_ms:.2f}ms, sandbox={sb_ms:.2f}ms, total={total_ms:.2f}ms\n\n"
        f"--- RAW LLM REPLY ---\n{raw_reply}\n\n"
        f"--- EXTRACTED CODE ---\n{gen_code}\n\n"
        f"--- SANDBOX STDOUT ---\n{sandbox_res.get('stdout', '')}\n\n"
        f"--- SANDBOX ERROR/STDERR ---\n{sandbox_res.get('error', '') or sandbox_res.get('stderr', '')}\n"
    )
    flat_log_path.write_text(log_body, encoding="utf-8")
    nested_log_path.write_text(log_body, encoding="utf-8")
    with open(total_log_path, "a", encoding="utf-8") as tf:
        tf.write(banner)
        tf.write(log_body.rstrip())
        tf.write("\n\n\n")

    if verbose_plain:
        icon = "✓" if status in ("PASSED", "SYNTHESIZED") else "✗"
        print(f"[{icon}] PURE ({resolved_llm}) | CHAT -> {status} [{category}] ({detail}) [{elapsed}s]")
        if gen_code and show_code:
            code_lines = gen_code.splitlines()
            print(f"    --- Pure Chat Python Code ({len(code_lines)} lines) ---")
            lines_to_show = code_lines if full_code else code_lines[:code_max_lines]
            for cl in lines_to_show:
                print(f"    {cl}")
            if not full_code and len(code_lines) > code_max_lines:
                print(f"    ... (+{len(code_lines) - code_max_lines} lines truncated)")
            print("    ---------------------------------------------------------")

    return {
        "profile": "PURE",
        "method": "CHAT",
        "llm_name": resolved_llm,
        "llm_feedback": False,
        "status": status,
        "category": category,
        "detail": detail,
        "elapsed": elapsed,
        "latencies": latencies,
        "prompt": prompt,
        "code": gen_code,
        "sandbox_res": sandbox_res,
        "flat_log_path": flat_log_path,
        "nested_log_path": nested_log_path,
        "clean_output": log_body,
    }


def print_pure_comparison_table(
    preset_key: str,
    framework_results: List[Dict[str, Any]],
    pure_result: Dict[str, Any],
    use_rich: bool = True,
) -> None:
    """Displays side-by-side comparison between NSTL framework run(s) and Pure LLM chat."""
    if not pure_result:
        return

    pure_st = pure_result.get("status", "N/A")
    pure_ok = pure_st in ("PASSED", "SYNTHESIZED")
    pure_ms = pure_result.get("latencies", {}).get("total_ms", 0) or (pure_result.get("elapsed", 0) * 1000.0)
    pure_det = str(pure_result.get("detail", ""))[:60]

    if use_rich and RICH_AVAILABLE and console:
        tbl = Table(
            title=f"⚔️ NSTL FRAMEWORK vs PURE LLM CHAT [{preset_key.upper()}]",
            title_style="bold yellow",
            border_style="yellow",
            header_style="bold white on #3a2f0b",
            expand=True,
        )
        tbl.add_column("Pipeline / Model", style="bold cyan", width=22)
        tbl.add_column("Status", justify="center", width=18)
        tbl.add_column("Latency", justify="right", width=12)
        tbl.add_column("Outcome vs Pure Baseline / Detail", justify="left")

        p_style = f"[green]{pure_st}[/green]" if pure_ok else f"[red]{pure_st}[/red]"
        tbl.add_row(f"PURE ({pure_result.get('llm_name', 'LLM')})", p_style, f"{pure_ms:.0f}ms", f"Baseline: {pure_det}")

        for r in framework_results:
            fw_st = r.get("status", "N/A")
            fw_ok = fw_st in ("PASSED", "SYNTHESIZED")
            fw_ms = r.get("latencies", {}).get("total_ms", 0) or (r.get("elapsed", 0) * 1000.0)
            fw_style = f"[green]{fw_st}[/green]" if fw_ok else f"[red]{fw_st}[/red]"
            if fw_ok and not pure_ok:
                verdict = f"[bold green]✓ FRAMEWORK WINS (+Pure failed: {pure_det})[/bold green]"
            elif fw_ok and pure_ok:
                verdict = "[cyan]= BOTH PASSED[/cyan]"
            elif not fw_ok and pure_ok:
                verdict = f"[bold red]✗ PURE WINS (FW: {str(r.get('detail',''))[:45]})[/bold red]"
            else:
                verdict = f"[dim]~ BOTH FAILED ({str(r.get('detail',''))[:45]})[/dim]"
            tbl.add_row(f"Profile {r['profile']} + {r['method']}", fw_style, f"{fw_ms:.0f}ms", verdict)

        console.print("")
        console.print(tbl)
        return

    print("\n" + "-" * 80)
    print(f" ⚔️ NSTL FRAMEWORK vs PURE LLM CHAT [{preset_key.upper()}]")
    print("-" * 80)
    print(f"  PURE ({pure_result.get('llm_name', 'LLM')}) : {pure_st} ({pure_ms:.0f}ms) | {pure_det}")
    for r in framework_results:
        fw_st = r.get("status", "N/A")
        fw_ms = r.get("latencies", {}).get("total_ms", 0) or (r.get("elapsed", 0) * 1000.0)
        print(f"  Profile {r['profile']} + {r['method']} : {fw_st} ({fw_ms:.0f}ms) | {r.get('detail', '')}")
    print("-" * 80)


# ==============================================================================
# Summary & Analytics Printers
# ==============================================================================

def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v]
    return (sum(vals) / len(vals)) if vals else None


def print_pass_summary(
    title: str,
    results: List[Dict[str, Any]],
    duration: float,
    log_path: Path,
    prompt: str = "",
    use_rich: bool = True,
) -> None:
    """
    Honest per-pass report: a monotone funnel, a failure breakdown that names what actually happened, and rates
    with 95% Wilson intervals. Only VERIFIED programs count as passes (sandbox exit 0 AND every literal the
    prompt states is consumed by the code); sandbox-only passes are shown separately as unverified.
    """
    total = len(results)
    if not total:
        return
    fun = EM.funnel(results)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "UNKNOWN"), []).append(r)

    cat_help = {
        "SUCCESS": "Verified: ran in sandbox and consumed every stated literal",
        "SUSPECT": "Ran, but a stated literal was ignored or the lint did not run - NOT counted as a pass",
        "VERIFIER_REJECTED": "Planner output rejected by the pre-flight verifier (wrong/incomplete plan caught)",
        "BINDING_UNRESOLVED": "A required port got no value (under-specified prompt OR binder gap - see ports)",
        "PLANNING": "No path / unification failed / no code emitted",
        "IMPLEMENTATION": "Engine crash, malformed emitted code, or generated code that crashed",
        "SECURITY": "Blocked by capability sandbox",
        "ENVIRONMENT": "Model unavailable, profile load failure or timeout",
        "EXPECTED_NEGATIVE": "Method intentionally unsupported in this profile (negative control)",
    }
    order = ["SUCCESS", "SUSPECT", "VERIFIER_REJECTED", "BINDING_UNRESOLVED", "PLANNING", "IMPLEMENTATION",
             "SECURITY", "ENVIRONMENT", "EXPECTED_NEGATIVE", "UNKNOWN"]
    lat = lambda k: _mean([r.get("latencies", {}).get(k) for r in results])  # noqa: E731
    lat_rows = [("Layer 1: Semantic Routing", lat("route_ms")), ("Layer 2: Topological Planning", lat("plan_ms")),
                ("Layer 3: Monadic Unification", lat("synth_ms")), ("Layer 4: GEVR Sandbox Execution", lat("sandbox_ms")),
                ("Layer 5: LLM Self-Repair Cycle", lat("repair_ms")), ("End-to-End Total Mean", lat("total_ms"))]
    warm = any(not r.get("cold_route", True) for r in results)
    exp_ok = sum(1 for r in results if r.get("outcome_ok"))
    expected_modes = sorted({r.get("expected", "synthesize") for r in results})

    if use_rich and RICH_AVAILABLE and console:
        ft = Table(title=f"📊 {title.upper()} - FUNNEL", title_style="bold cyan", border_style="blue", expand=True)
        ft.add_column("Stage", style="bold")
        ft.add_column("Count", justify="center", width=10)
        ft.add_column("% of runs", justify="center", width=12)
        for label, key in (("Attempted", "attempted"), ("Emitted code", "emitted_code"), ("Lint-clean", "lint_clean"),
                           ("Ran in sandbox (exit 0)", "ran_in_sandbox"), ("Verified (literals consumed)", "verified")):
            ft.add_row(label, str(fun[key]), f"{fun[key] / total * 100:.1f}%")
        console.print("")
        console.print(ft)

        ct = Table(title="Failure & outcome classification", title_style="bold cyan", border_style="blue", expand=True)
        ct.add_column("Category", style="bold")
        ct.add_column("Count", justify="center", width=8)
        ct.add_column("%", justify="center", width=8)
        ct.add_column("Meaning", justify="left")
        for cat in order:
            rs = by_cat.get(cat)
            if rs:
                ct.add_row(cat, str(len(rs)), f"{len(rs) / total * 100:.1f}%", cat_help.get(cat, ""))
        console.print(ct)

        console.print(f"[bold]Verified pass rate:[/bold] {EM.format_rate(fun['verified'], total)}")
        if "clarify" in expected_modes:
            console.print(f"[bold]Acceptable outcome (per preset 'expected'):[/bold] {EM.format_rate(exp_ok, total)}")
        console.print(f"[bold]Sandbox-only pass rate (not verified):[/bold] {EM.format_rate(fun['ran_in_sandbox'], total)}")

        lt = Table(title="⏱️ Pipeline Layer Latencies (Mean)" + (" - WARM route cache: not comparable across passes" if warm else " - cold route cache"),
                   title_style="bold yellow", border_style="yellow")
        lt.add_column("Layer", style="bold")
        lt.add_column("Mean Duration", justify="right", style="bold green")
        for name, v in lat_rows:
            if v is not None:
                lt.add_row(name, f"{v:.2f} ms")
        console.print(lt)
        console.print(f"[dim]Total Duration: {duration:.2f}s │ Log: {log_path}[/dim]\n")
        return

    print("\n" + "=" * 80)
    print(f" 📊 {title.upper()}")
    if prompt:
        print(f" Prompt: \"{prompt[:72]}...\"" if len(prompt) > 72 else f" Prompt: \"{prompt}\"")
    print("=" * 80)
    print(f"Funnel: attempted {fun['attempted']} > emitted code {fun['emitted_code']} > lint-clean {fun['lint_clean']} "
          f"> ran in sandbox {fun['ran_in_sandbox']} > verified {fun['verified']}")
    for cat in order:
        for r in by_cat.get(cat, []):
            if cat == "SUCCESS":
                continue
            print(f"  [{cat}] Profile {r['profile']} {r['method']}: {r['status']} ({r['detail']})")
    print(f"Verified pass rate:     {EM.format_rate(fun['verified'], total)}")
    if "clarify" in expected_modes:
        print(f"Acceptable outcome:     {EM.format_rate(exp_ok, total)}")
    for name, v in lat_rows:
        if v is not None:
            print(f"  • {name:<34} {v:.2f} ms")
    print(f"Execution Duration:     {duration:.2f}s\nLog Output File:        {log_path}")
    print("=" * 80 + "\n")


def _feedback_outcome(r_no: Dict[str, Any], r_with: Dict[str, Any]) -> str:
    ok_no = r_no.get("status") in EM.SUCCESS_STATUSES
    ok_with = r_with.get("status") in EM.SUCCESS_STATUSES
    if ok_no and ok_with:
        return "consistent_pass"
    if ok_with and not ok_no:
        return "rescued" if r_with.get("repair_attempted") else "differs_without_repair"
    if ok_no and not ok_with:
        return "regressed"
    if not r_with.get("repair_attempted"):
        return "not_entered"
    return "repair_failed"


def print_comparative_table(
    results_no: List[Dict[str, Any]],
    results_with: List[Dict[str, Any]],
    use_rich: bool = True,
) -> None:
    """
    Feedback-effectiveness matrix. "Rescued" requires that the repair stage actually ran. A run that failed
    before any code was emitted never enters the repair loop, so it is reported as such instead of as a
    "persistent limit". Latency overhead is only reported when both passes used a cold route cache.
    """
    map_no = {(r["profile"], r["method"]): r for r in results_no}
    map_with = {(r["profile"], r["method"]): r for r in results_with}
    keys = sorted(set(map_no) | set(map_with))
    labels = {
        "consistent_pass": "✓ Verified in both passes",
        "rescued": "✨ RESCUED by feedback (repair stage ran)",
        "differs_without_repair": "~ Pass 2 differs, no repair ran (non-determinism)",
        "regressed": "⚠ REGRESSED with feedback",
        "not_entered": "∅ Feedback loop never entered (failed before repair stage)",
        "repair_failed": "✗ Repair stage ran and did not fix it",
    }
    tally: Dict[str, int] = {k: 0 for k in labels}
    rows = []
    for k in keys:
        r_no, r_with = map_no.get(k, {}), map_with.get(k, {})
        oc = _feedback_outcome(r_no, r_with)
        tally[oc] += 1
        rep_ms = r_with.get("latencies", {}).get("repair_ms", 0) or 0
        rows.append((k, r_no, r_with, oc, rep_ms))

    both_cold = all(r.get("cold_route", True) for r in list(results_no) + list(results_with))
    repair_times = [x[4] for x in rows if x[4] > 0]
    t_no = [r["latencies"].get("total_ms") for r in results_no if r.get("latencies", {}).get("total_ms")]
    t_with = [r["latencies"].get("total_ms") for r in results_with if r.get("latencies", {}).get("total_ms")]

    if use_rich and RICH_AVAILABLE and console:
        t = Table(title="🔬 LLM FEEDBACK EFFECTIVENESS MATRIX", title_style="bold cyan", border_style="bright_blue",
                  header_style="bold white on #1f4068", expand=True)
        t.add_column("Prof", justify="center", style="bold cyan", width=6)
        t.add_column("Method", justify="center", style="bold yellow", width=8)
        t.add_column("Pass 1 (No FB)", justify="center", width=20)
        t.add_column("Pass 2 (With FB)", justify="center", width=20)
        t.add_column("Repair (ms)", justify="right", width=12)
        t.add_column("Outcome", justify="left")
        for k, r_no, r_with, oc, rep_ms in rows:
            t.add_row(k[0], k[1], str(r_no.get("status", "N/A")), str(r_with.get("status", "N/A")),
                      f"{rep_ms:.1f}" if rep_ms > 0 else "-", labels[oc])
        console.print("")
        console.print(t)
        st = Table(title="Feedback Telemetry", title_style="bold yellow", border_style="yellow")
        st.add_column("Metric", style="bold")
        st.add_column("Value", style="bold green", justify="right")
        for key, label in labels.items():
            if tally[key]:
                st.add_row(label, str(tally[key]))
        if repair_times:
            st.add_row("Repair-stage time (mean, runs where it ran)", f"{sum(repair_times) / len(repair_times):.1f} ms")
        if both_cold and t_no and t_with:
            st.add_row("End-to-end mean latency, pass 1 vs pass 2", f"{sum(t_no) / len(t_no):.1f} vs {sum(t_with) / len(t_with):.1f} ms")
        elif not both_cold:
            st.add_row("Latency comparison", "omitted (warm route cache: not comparable)")
        console.print(st)
        console.print("")
        return

    print("\n" + "=" * 94)
    print(" 🔬 LLM FEEDBACK EFFECTIVENESS MATRIX")
    print("=" * 94)
    for k, r_no, r_with, oc, rep_ms in rows:
        print(f"Profile {k[0]:<2} | {k[1]:<6} | {str(r_no.get('status', 'N/A')):<18} | {str(r_with.get('status', 'N/A')):<18} | {labels[oc]}")
    print("-" * 94)
    for key, label in labels.items():
        if tally[key]:
            print(f"  • {label}: {tally[key]}")
    if repair_times:
        print(f"  • Repair-stage time (mean, runs where it ran): {sum(repair_times) / len(repair_times):.1f} ms")
    if both_cold and t_no and t_with:
        print(f"  • End-to-end mean latency pass 1 vs pass 2: {sum(t_no) / len(t_no):.1f} vs {sum(t_with) / len(t_with):.1f} ms")
    print("=" * 94 + "\n")


def print_overnight_master_summary(
    prompt_summaries: List[Dict[str, Any]],
    total_suite_duration: float,
    cold_route: bool = True,
    use_rich: bool = True,
) -> None:
    """Multi-prompt summary. Rates carry 95% Wilson intervals; no timing figure is estimated or hardcoded."""
    rows = []
    all_results: List[Dict[str, Any]] = []
    for s_ in prompt_summaries:
        rs = list(s_.get("fw_results") or [])
        all_results += rs
        fun = EM.funnel(rs)
        n = len(rs)
        refused = sum(1 for r in rs if r.get("category") in EM.REFUSAL_CATEGORIES)
        defects = sum(1 for r in rs if r.get("category") == "IMPLEMENTATION")
        suspect = sum(1 for r in rs if r.get("category") == "SUSPECT")
        ok = sum(1 for r in rs if r.get("outcome_ok"))
        rows.append((s_.get("preset_key", "custom"), s_.get("expected", "synthesize"), n, fun["verified"], suspect, refused, defects, ok))

    tot_n = sum(r[2] for r in rows)
    tot_ver = sum(r[3] for r in rows)
    tot_ok = sum(r[7] for r in rows)
    fun_all = EM.funnel(all_results)
    n_expected_clarify = sum(1 for r in rows if r[1] == "clarify")

    if use_rich and RICH_AVAILABLE and console:
        t = Table(title="🌌 NSTL BENCHMARK MASTER SUMMARY", title_style="bold magenta", border_style="magenta",
                  header_style="bold white on #4a154b", expand=True)
        t.add_column("Preset", style="bold yellow", width=16)
        t.add_column("Expected", width=11)
        t.add_column("Runs", justify="center", width=6)
        t.add_column("Verified", justify="center", width=10, style="green")
        t.add_column("Unverified", justify="center", width=11, style="yellow")
        t.add_column("Refused", justify="center", width=9, style="magenta")
        t.add_column("Defects", justify="center", width=9, style="red")
        t.add_column("Verified rate (95% CI)", justify="right")
        for key, exp, n, ver, sus, refd, dfc, ok in rows:
            lo, hi = EM.wilson_interval(ver, n)
            t.add_row(key, exp, str(n), str(ver), str(sus), str(refd), str(dfc),
                      f"{ver / n * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]" + (" (n<10)" if n < 10 else "") if n else "n/a")
        console.print("")
        console.print(t)
        g = Table(title="Grand Totals", title_style="bold green", border_style="green")
        g.add_column("Metric", style="bold")
        g.add_column("Value", style="bold white", justify="right")
        g.add_row("Runs evaluated", str(tot_n))
        g.add_row("Verified passes", EM.format_rate(tot_ver, tot_n))
        g.add_row("Ran in sandbox (unverified incl.)", EM.format_rate(fun_all["ran_in_sandbox"], tot_n))
        if n_expected_clarify:
            g.add_row("Acceptable outcome (per-preset expectation)", EM.format_rate(tot_ok, tot_n))
        g.add_row("Total suite duration", f"{total_suite_duration:.2f}s")
        g.add_row("Route cache", "cold per run (measured)" if cold_route else "WARM (timings not comparable)")
        console.print(g)
        console.print("")
        return

    print("\n" + "#" * 94)
    print(" 🌌 NSTL BENCHMARK MASTER SUMMARY")
    print("#" * 94)
    print(f"{'Preset':<15} | {'Expect':<10} | {'Runs':<5} | {'Verified':<8} | {'Unverif':<7} | {'Refused':<7} | {'Defects':<7}")
    print("-" * 94)
    for key, exp, n, ver, sus, refd, dfc, ok in rows:
        print(f"{key[:14]:<15} | {exp:<10} | {n:<5} | {ver:<8} | {sus:<7} | {refd:<7} | {dfc:<7}")
    print("=" * 94)
    print(f"  • Runs evaluated:      {tot_n}")
    print(f"  • Verified passes:     {EM.format_rate(tot_ver, tot_n)}")
    print(f"  • Ran in sandbox:      {EM.format_rate(fun_all['ran_in_sandbox'], tot_n)}")
    if n_expected_clarify:
        print(f"  • Acceptable outcome:  {EM.format_rate(tot_ok, tot_n)}")
    print(f"  • Total suite duration: {total_suite_duration:.2f}s  | route cache: {'cold per run' if cold_route else 'WARM (timings not comparable)'}")
    print("#" * 94 + "\n")


# ==============================================================================
# Argument Parsing & CLI Setup
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for test.py runner."""
    parser = argparse.ArgumentParser(
        prog="python3 test.py",
        description="Comprehensive Multi-Profile & Multi-RouteMethod Empirical Test Runner for NSTL. Analyzes latency, performance, and failure root causes (architectural vs implementation), generating both deterministic and LLM-feedback logs for comparative evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--warm-route-cache",
        action="store_true",
        help=(
            "Keep the Layer-1 route cache between runs/passes. Default is COLD: the cache is cleared before every "
            "run so routing latency and results are measured, not replayed from an earlier pass. With a warm cache "
            "pass-2 timings are not comparable to pass-1."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["dual", "without-feedback", "with-feedback", "pure", "compare-pure"],
        default="dual",
        help="Evaluation mode: 'dual' runs Pass 1 (without feedback) and Pass 2 (with feedback); 'pure' runs ONLY the raw GGUF LLM in normal chat mode without NSTL; 'compare-pure' runs the framework alongside 'pure' baseline for comparison. (default: dual)",
    )
    parser.add_argument(
        "--with-pure",
        action="store_true",
        default=False,
        help="Also run the pure LLM chat baseline alongside --mode (without-feedback / with-feedback / dual) and print a side-by-side comparison table.",
    )
    parser.add_argument(
        "--pure-system-prompt",
        type=str,
        default=None,
        help="Optional custom system prompt for pure LLM chat mode.",
    )
    parser.add_argument(
        "--profiles",
        "-p",
        nargs="+",
        default=["0", "A", "C", "D", "E", "S"],
        help=f"List of profiles to test: choose one or more from {ALL_PROFILES} or 'all' to run all profiles. Accepts space- or comma-separated lists. (default: 0 A C D E S)",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=None,
        help="Optional override to test specific route methods across profiles (e.g. M0 M1 M6 M11 OMEGA or M0,M1,OMEGA, or 'all'). Accepts space- or comma-separated lists.",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        default=False,
        help="Test all route methods (M0-M11) across all profiles, including Profile 0 and A",
    )
    parser.add_argument(
        "--preset",
        "-pr",
        nargs="+",
        default=["canonical"],
        help=f"Prompt preset(s) to evaluate. Choose one or more: {list(PROMPT_PRESETS.keys())} or 'all' to evaluate all presets. (default: canonical)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a text file (one prompt per line) or JSON file with custom prompts to evaluate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override with a custom single pipeline prompt specification",
    )
    parser.add_argument(
        "--cache-layers",
        dest="cache_layers",
        action="store_true",
        default=True,
        help="Enable in-memory layer caching (Lattice graph, Model weights, Layer 1 semantic tunnels) to save 5x-10x execution time (default: True)",
    )
    parser.add_argument(
        "--no-cache-layers",
        dest="cache_layers",
        action="store_false",
        help="Disable layer caching and execute each test in an isolated CLI subprocess",
    )
    parser.add_argument(
        "--subprocess",
        dest="cache_layers",
        action="store_false",
        help="Alias for --no-cache-layers: run isolated CLI subprocesses for each test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run across the prompt matrix (default: 1)",
    )
    parser.add_argument(
        "--loop",
        "--overnight",
        dest="overnight_loop",
        action="store_true",
        default=False,
        help="Run continuously in an overnight benchmarking loop until interrupted (Ctrl+C)",
    )
    parser.add_argument(
        "--reranker",
        dest="reranker",
        action="store_true",
        default=None,
        help="Enable neural reranker (jina-reranker-v3.5) across all test runs",
    )
    parser.add_argument(
        "--no-reranker",
        dest="reranker",
        action="store_false",
        help="Explicitly disable neural reranker across all test runs",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Specify reranker model name (e.g. jina-reranker-v3.5)",
    )
    parser.add_argument(
        "--macros",
        dest="macros",
        action="store_true",
        default=None,
        help="Enable macro-goal routing",
    )
    parser.add_argument(
        "--no-macros",
        dest="macros",
        action="store_false",
        help="Disable macro-goal routing",
    )
    parser.add_argument(
        "--topology",
        choices=["frontier", "linear"],
        default=None,
        help="Topological planning approach (frontier DAG vs linear monadic)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Enable Dev Mode during test execution",
    )
    parser.add_argument(
        "--no-lint",
        action="store_true",
        default=False,
        help="Skip static pre-flight linter",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        default=False,
        help="Skip GEVR live sandbox execution for rapid routing verification",
    )
    parser.add_argument(
        "--explain-plan",
        dest="explain_plan",
        action="store_true",
        default=True,
        help="Print and log clause-level coverage and precision diagnostics table (default: True)",
    )
    parser.add_argument(
        "--no-explain-plan",
        dest="explain_plan",
        action="store_false",
        help="Disable clause-level coverage diagnostics table",
    )
    parser.add_argument(
        "--early-exit-layer",
        type=int,
        default=None,
        help="Experimental: Extract residual stream from transformer layer K instead of final layer.",
    )
    parser.add_argument(
        "--no-whitening",
        action="store_true",
        default=False,
        help="Experimental: Disable ZCA whitening on latent cell embeddings.",
    )
    parser.add_argument(
        "--sparse-sae",
        action="store_true",
        default=False,
        help="Experimental: Enable Top-K Sparse Autoencoder (OMP) multi-clause concept decomposition.",
    )
    parser.add_argument(
        "--no-markov-prior",
        action="store_true",
        default=False,
        help="Experimental: Disable Absorbing Markov Chain N=(I-Q)^-1 hitting-probability prior.",
    )
    parser.add_argument(
        "--edge-mode",
        choices=["static", "live"],
        default=None,
        help="5-axis edge calculator mode: static (eager full graph) or live (lazy frontier).",
    )
    parser.add_argument(
        "--llm-feedback",
        action="store_true",
        default=False,
        help="Shortcut to force single pass with LLM feedback enabled (sets mode to with-feedback)",
    )
    parser.add_argument(
        "--allow-capabilities",
        type=str,
        default="",
        help="Comma-separated list of capabilities to allow in sandbox (e.g. net.http,fs.read)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        help="Override embedding model name (e.g. jina-embeddings-v5-text-nano)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="Override LLM model name / path (e.g. qwen2.5-coder-0.5b-instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Compute device for neural models",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="test_logs",
        help="Directory to dump log files (default: test_logs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds per test run (default: 180.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned test matrix without running tests",
    )
    parser.add_argument(
        "--show-code",
        dest="show_code",
        action="store_true",
        default=True,
        help="Display synthesized Python code live per test in syntax-highlighted terminal panel (default: True)",
    )
    parser.add_argument(
        "--no-show-code",
        dest="show_code",
        action="store_false",
        help="Hide live synthesized Python code previews in terminal",
    )
    parser.add_argument(
        "--full-code",
        dest="full_code",
        action="store_true",
        default=False,
        help="Display complete synthesized Python code without line truncation (default: False)",
    )
    parser.add_argument(
        "--code-max-lines",
        type=int,
        default=35,
        help="Maximum number of code lines to display in live terminal card before truncation (default: 35)",
    )
    parser.add_argument(
        "--rich",
        dest="use_rich",
        action="store_true",
        default=True,
        help="Enable Rich animated terminal UI, live progress bar, and formatted tables (default: True)",
    )
    parser.add_argument(
        "--no-rich",
        dest="use_rich",
        action="store_false",
        help="Disable Rich terminal UI and fall back to plain text console output",
    )
    return parser


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure headless environment for game engine / audio sandbox execution
    configure_headless_env()

    # Provision required synthetic test assets
    ensure_input_csv(ROOT_DIR)
    ensure_input_image(ROOT_DIR)
    ensure_input_audio(ROOT_DIR)

    # Resolve execution mode
    if args.llm_feedback:
        mode = "with-feedback"
    else:
        mode = args.mode
    run_pure = (mode in ("pure", "compare-pure")) or bool(getattr(args, "with_pure", False))
    run_framework = (mode != "pure")
    run_fw_no = run_framework and (mode in ("dual", "without-feedback", "compare-pure"))
    run_fw_with = run_framework and (mode in ("dual", "with-feedback"))

    # Resolve prompts to evaluate
    prompts_to_test: List[Tuple[str, str]] = []  # (preset_key, prompt_text)
    if args.prompt:
        prompts_to_test.append(("custom", args.prompt.strip()))
    elif args.prompts_file:
        pf = Path(args.prompts_file)
        if pf.exists():
            import json
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for idx, p in enumerate(data):
                        prompts_to_test.append((f"file_{idx+1}", str(p).strip()))
                elif isinstance(data, dict):
                    for k, v in data.items():
                        prompts_to_test.append((str(k), str(v).strip()))
            except Exception:
                for idx, line in enumerate(pf.read_text(encoding="utf-8").splitlines()):
                    l_s = line.strip()
                    if l_s and not l_s.startswith("#"):
                        prompts_to_test.append((f"file_{idx+1}", l_s))
        else:
            print(f"[!] Warning: prompts file '{pf}' not found. Falling back to presets.")

    if not prompts_to_test:
        raw_presets = []
        for p in args.preset:
            for item in p.split(","):
                item_s = item.lower().strip()
                if item_s:
                    raw_presets.append(item_s)
        if "all" in raw_presets:
            for k, info in PROMPT_PRESETS.items():
                prompts_to_test.append((k, info["prompt"]))
        else:
            for k in raw_presets:
                if k in PROMPT_PRESETS:
                    prompts_to_test.append((k, PROMPT_PRESETS[k]["prompt"]))
                else:
                    print(f"[!] Unknown preset '{k}'. Valid options: {list(PROMPT_PRESETS.keys())} or 'all'.")

    if not prompts_to_test:
        prompts_to_test.append(("canonical", DEFAULT_PROMPT))

    if getattr(args, "edge_mode", None):
        em_val = str(args.edge_mode).lower()
        os.environ["NSTL_EDGE_MODE"] = em_val
        try:
            sys.path.insert(0, str(ROOT_DIR / "src"))
            from config import settings
            from experiments import ACTIVE_EXPERIMENTS
            settings.edge_mode = em_val
            ACTIVE_EXPERIMENTS.edge_mode = em_val
        except Exception:
            pass

    # Initialize in-process caching runner if enabled
    runner: Optional[CachedNSTLRunner] = None
    if run_framework and args.cache_layers and not args.dry_run:
        print("[*] Initializing CachedNSTLRunner (multi-layer in-memory caching enabled)...")
        runner = CachedNSTLRunner.get_instance(ROOT_DIR)

    pure_runner: Optional[PureLLMRunner] = None
    if run_pure and not args.dry_run:
        pure_runner = PureLLMRunner(ROOT_DIR)

    # Build matrix of profile & methods
    raw_profiles = []
    for p in args.profiles:
        for item in p.split(","):
            item_s = item.strip().upper()
            if item_s in ("OMEGA", "Ω"):
                item_s = "O"
            if item_s:
                raw_profiles.append(item_s)
    if "ALL" in raw_profiles:
        target_profiles = ALL_PROFILES
    else:
        target_profiles = raw_profiles

    raw_methods = []
    if args.methods:
        for m in args.methods:
            for item in m.split(","):
                item_s = item.strip().upper()
                if item_s:
                    raw_methods.append(item_s)
        if "ALL" in raw_methods:
            raw_methods = ALL_METHODS

    test_matrix: List[Tuple[str, str]] = []
    if run_framework:
        for prof in target_profiles:
            if raw_methods:
                methods = raw_methods
            elif args.all_methods:
                methods = ALL_METHODS
            else:
                methods = SUPPORTED_METHODS.get(prof, ALL_METHODS)
            for meth in methods:
                test_matrix.append((prof, meth))
    if run_pure:
        target_profiles_display = list(target_profiles) + ["PURE"] if run_framework else ["PURE"]
    else:
        target_profiles_display = list(target_profiles)

    reranker_info = "ENABLED" if args.reranker is True else ("DISABLED" if args.reranker is False else "DEFAULT (Auto)")
    if args.reranker_model:
        reranker_info += f" ({args.reranker_model})"

    caching_info = "ENABLED (In-Process Models & Layer 1 Caching)" if (args.cache_layers and not args.dry_run) else "DISABLED (Subprocess per test)"

    total_matrix_runs_display = len(test_matrix) + (1 if run_pure else 0)
    if args.use_rich and RICH_AVAILABLE and console:
        render_hero_banner(
            mode=mode,
            target_profiles=target_profiles_display,
            total_matrix_runs=total_matrix_runs_display,
            prompts_to_test=prompts_to_test,
            cache_layers=args.cache_layers and not args.dry_run,
            explain_plan=args.explain_plan,
            reranker_info=reranker_info,
            output_dir=output_dir,
            timeout=args.timeout,
            dry_run=args.dry_run,
            show_code=args.show_code,
            full_code=args.full_code,
            code_max_lines=args.code_max_lines,
        )
    else:
        print("\n" + "=" * 80)
        print(" 🚀 NSTL MULTI-PROFILE & MULTI-METHOD EMPIRICAL TEST MATRIX")
        print("=" * 80)
        print(f"Evaluation Mode:      {mode.upper()}")
        print(f"Target Profiles:      {target_profiles_display}")
        print(f"Total Runs per Pass:  {total_matrix_runs_display}")
        print(f"Total Prompts:        {len(prompts_to_test)} presets/specs")
        print(f"Layer Caching:        {caching_info}")
        print(f"Diagnostic Telemetry: {'--explain-plan ENABLED' if args.explain_plan else 'DISABLED'}")
        print(f"Neural Reranker:      {reranker_info}")
        print(f"Output Directory:     {output_dir}")
        print(f"Timeout per run:      {args.timeout}s")
        print(f"Dry Run Mode:         {args.dry_run}")
        print("Active Prompts to Evaluate:")
        for pk, pt in prompts_to_test:
            p_name = PROMPT_PRESETS.get(pk, {}).get("name", pk)
            print(f"  • [{pk}] {p_name}: \"{pt[:64]}...\"" if len(pt) > 64 else f"  • [{pk}] {p_name}: \"{pt}\"")
        print("=" * 80 + "\n")

    iteration = 0
    overnight_prompt_summaries: List[Dict[str, Any]] = []
    t_suite_start = time.time()
    total_cached_runs = 0

    log_combined = output_dir / "total.log"
    master_log_combined = ROOT_DIR / "total.log"
    if not args.dry_run:
        log_combined.write_text("", encoding="utf-8")

    try:
        while True:
            iteration += 1
            if args.overnight_loop:
                print(f"\n{'#' * 80}")
                print(f" 🌙 OVERNIGHT BENCHMARK ITERATION #{iteration}")
                print(f"{'#' * 80}\n")
            elif args.iterations > 1:
                print(f"\n{'#' * 80}")
                print(f" 🔁 ITERATION #{iteration} of {args.iterations}")
                print(f"{'#' * 80}\n")

            for p_idx, (preset_key, current_prompt) in enumerate(prompts_to_test, 1):
                p_desc = PROMPT_PRESETS.get(preset_key, {}).get("name", preset_key)
                prompt_tag = f"{preset_key}" if len(prompts_to_test) > 1 else ""
                tag_suffix = f"_{preset_key}" if prompt_tag else ""

                log_no_feedback = output_dir / f"total_no_feedback{tag_suffix}.log"
                log_with_feedback = output_dir / f"total_with_feedback{tag_suffix}.log"

                if not (args.use_rich and RICH_AVAILABLE):
                    print("\n" + "=" * 80)
                    print(f" ▶️ [{p_idx}/{len(prompts_to_test)}] EVALUATING PRESET: '{preset_key.upper()}' ({p_desc})")
                    print(f" Prompt: \"{current_prompt}\"")
                    print("=" * 80)

                def execute_pass(
                    llm_feedback: bool,
                    log_path: Path,
                    pass_label: str,
                    pass_title: str,
                ) -> Tuple[List[Dict[str, Any]], float]:
                    if not args.dry_run:
                        log_path.write_text("", encoding="utf-8")
                    t0 = time.time()
                    res_list: List[Dict[str, Any]] = []
                    total_tests = len(test_matrix)

                    is_rich = RICH_AVAILABLE and args.use_rich and not args.dry_run

                    if is_rich and console:
                        stats_ref: Dict[str, Any] = {
                            "passed": 0,
                            "synth": 0,
                            "arch": 0,
                            "impl": 0,
                            "sec": 0,
                            "remaining": total_tests,
                            "rate": 0.0,
                        }
                        with Progress(
                            SpinnerColumn(spinner_name="dots"),
                            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                            BarColumn(bar_width=None, complete_style="green", finished_style="bold green"),
                            TaskProgressColumn(),
                            MofNCompleteColumn(),
                            TimeElapsedColumn(),
                            TimeRemainingColumn(),
                            RollingMetricsColumn(stats_ref) if RollingMetricsColumn else TextColumn(""),
                            console=console,
                            transient=False,
                        ) as progress:
                            task_id = progress.add_task(f"{pass_title}", total=total_tests)
                            for test_idx, (profile, method) in enumerate(test_matrix, 1):
                                progress.update(task_id, description=f"{pass_title} │ Profile {profile} {method}")
                                res = run_single_test(
                                    profile=profile,
                                    method=method,
                                    prompt=current_prompt,
                                    output_dir=output_dir,
                                    total_log_path=log_path,
                                    timeout=args.timeout,
                                    dry_run=args.dry_run,
                                    reranker=args.reranker,
                                    reranker_model=args.reranker_model,
                                    macros=args.macros,
                                    topology=args.topology,
                                    dev=args.dev,
                                    no_lint=args.no_lint,
                                    no_exec=args.no_exec,
                                    explain_plan=args.explain_plan,
                                    early_exit_layer=args.early_exit_layer,
                                    no_whitening=args.no_whitening,
                                    sparse_sae=args.sparse_sae,
                                    no_markov_prior=args.no_markov_prior,
                                    edge_mode=args.edge_mode,
                                    llm_feedback=llm_feedback,
                                    allow_capabilities=args.allow_capabilities,
                                    embedder=args.embedder,
                                    llm=args.llm,
                                    device=args.device,
                                    pass_label=f"{pass_label}{tag_suffix}",
                                    use_cache=args.cache_layers,
                                    runner=runner,
                                    cold_route=not args.warm_route_cache,
                                    expected=PROMPT_PRESETS.get(preset_key, {}).get("expected", "synthesize"),
                                    verbose_plain=False,
                                )
                                res_list.append(res)

                                st = res.get("status")
                                cat = res.get("category")
                                if st == "PASSED":
                                    stats_ref["passed"] += 1
                                elif st == "SYNTHESIZED":
                                    stats_ref["synth"] += 1
                                if cat == "SUSPECT":
                                    stats_ref["suspect"] = stats_ref.get("suspect", 0) + 1
                                if cat in EM.REFUSAL_CATEGORIES:
                                    stats_ref["arch"] += 1
                                elif cat == "IMPLEMENTATION":
                                    stats_ref["impl"] += 1
                                elif cat == "SECURITY":
                                    stats_ref["sec"] += 1

                                stats_ref["remaining"] = total_tests - test_idx
                                stats_ref["rate"] = (stats_ref["passed"] / test_idx) * 100.0

                                card = render_test_card(
                                    res=res,
                                    test_idx=test_idx,
                                    total_tests=total_tests,
                                    pass_label=pass_title,
                                    preset_key=preset_key,
                                    show_code=args.show_code,
                                    full_code=args.full_code,
                                    code_max_lines=args.code_max_lines,
                                )
                                if card:
                                    progress.print(card)
                                progress.advance(task_id)
                    else:
                        for test_idx, (profile, method) in enumerate(test_matrix, 1):
                            res = run_single_test(
                                profile=profile,
                                method=method,
                                prompt=current_prompt,
                                output_dir=output_dir,
                                total_log_path=log_path,
                                timeout=args.timeout,
                                dry_run=args.dry_run,
                                reranker=args.reranker,
                                reranker_model=args.reranker_model,
                                macros=args.macros,
                                topology=args.topology,
                                dev=args.dev,
                                no_lint=args.no_lint,
                                no_exec=args.no_exec,
                                explain_plan=args.explain_plan,
                                early_exit_layer=args.early_exit_layer,
                                no_whitening=args.no_whitening,
                                sparse_sae=args.sparse_sae,
                                no_markov_prior=args.no_markov_prior,
                                edge_mode=args.edge_mode,
                                llm_feedback=llm_feedback,
                                allow_capabilities=args.allow_capabilities,
                                embedder=args.embedder,
                                llm=args.llm,
                                device=args.device,
                                pass_label=f"{pass_label}{tag_suffix}",
                                use_cache=args.cache_layers,
                                runner=runner,
                                cold_route=not args.warm_route_cache,
                                expected=PROMPT_PRESETS.get(preset_key, {}).get("expected", "synthesize"),
                                verbose_plain=True,
                                show_code=args.show_code,
                                full_code=args.full_code,
                                code_max_lines=args.code_max_lines,
                            )
                            res_list.append(res)

                    duration = time.time() - t0
                    return res_list, duration

                results_no: List[Dict[str, Any]] = []
                results_with: List[Dict[str, Any]] = []
                results_pure: List[Dict[str, Any]] = []
                dur_no = 0.0
                dur_with = 0.0
                dur_pure = 0.0
                log_pure = output_dir / f"total_pure{tag_suffix}.log"

                if run_fw_no:
                    if not (args.use_rich and RICH_AVAILABLE):
                        print("\n" + "#" * 80)
                        print(f" ### PASS 1: DETERMINISTIC BASELINE [{preset_key.upper()}] (WITHOUT LLM FEEDBACK)")
                        print("#" * 80)
                    results_no, dur_no = execute_pass(
                        llm_feedback=False,
                        log_path=log_no_feedback,
                        pass_label="without_feedback",
                        pass_title=f"Pass 1: Deterministic [{preset_key.upper()}]",
                    )
                    if not args.dry_run and log_no_feedback.exists():
                        shutil.copyfile(log_no_feedback, ROOT_DIR / f"total_no_feedback{tag_suffix}.log")
                        if preset_key == "canonical" or not (ROOT_DIR / "total_no_feedback.log").exists():
                            shutil.copyfile(log_no_feedback, ROOT_DIR / "total_no_feedback.log")
                    if args.cache_layers:
                        total_cached_runs += len(results_no)

                if run_fw_with:
                    if not (args.use_rich and RICH_AVAILABLE):
                        print("\n" + "#" * 80)
                        print(f" ### PASS 2: AUTONOMOUS SELF-REPAIR [{preset_key.upper()}] (WITH LLM FEEDBACK)")
                        print("#" * 80)
                    results_with, dur_with = execute_pass(
                        llm_feedback=True,
                        log_path=log_with_feedback,
                        pass_label="with_feedback",
                        pass_title=f"Pass 2: Autonomous Self-Repair [{preset_key.upper()}]",
                    )
                    if not args.dry_run and log_with_feedback.exists():
                        shutil.copyfile(log_with_feedback, ROOT_DIR / f"total_with_feedback{tag_suffix}.log")
                        if preset_key == "canonical" or not (ROOT_DIR / "total_with_feedback.log").exists():
                            shutil.copyfile(log_with_feedback, ROOT_DIR / "total_with_feedback.log")
                    if args.cache_layers:
                        total_cached_runs += len(results_with)

                if run_pure:
                    if not args.dry_run:
                        log_pure.write_text("", encoding="utf-8")
                    if not (args.use_rich and RICH_AVAILABLE):
                        print("\n" + "#" * 80)
                        print(f" ### PURE LLM CHAT BASELINE [{preset_key.upper()}] (NO FRAMEWORK)")
                        print("#" * 80)
                    t_p0 = time.time()
                    if pure_runner is None and not args.dry_run:
                        pure_runner = PureLLMRunner(ROOT_DIR)
                    pure_res = run_pure_llm_test(
                        prompt=current_prompt,
                        output_dir=output_dir,
                        total_log_path=log_pure,
                        pure_runner=pure_runner if pure_runner is not None else PureLLMRunner(ROOT_DIR),
                        timeout=args.timeout,
                        dry_run=args.dry_run,
                        no_exec=args.no_exec,
                        allow_capabilities=args.allow_capabilities,
                        llm=args.llm,
                        device=args.device,
                        system_prompt=getattr(args, "pure_system_prompt", None),
                        pass_label=f"pure{tag_suffix}",
                        verbose_plain=not (args.use_rich and RICH_AVAILABLE),
                        show_code=args.show_code,
                        full_code=args.full_code,
                        code_max_lines=args.code_max_lines,
                    )
                    dur_pure = time.time() - t_p0
                    results_pure = [pure_res]
                    if not args.dry_run and log_pure.exists():
                        shutil.copyfile(log_pure, ROOT_DIR / f"total_pure{tag_suffix}.log")
                        if preset_key == "canonical" or not (ROOT_DIR / "total_pure.log").exists():
                            shutil.copyfile(log_pure, ROOT_DIR / "total_pure.log")

                # Append to master combined total.log
                if not args.dry_run:
                    with open(log_combined, "a", encoding="utf-8") as tf:
                        tf.write(f"\n{'#' * 80}\n### PROMPT PRESET: {preset_key.upper()} ({p_desc})\n### PROMPT: {current_prompt}\n{'#' * 80}\n\n")
                        if log_no_feedback.exists():
                            tf.write("### PASS 1: DETERMINISTIC RUNS (WITHOUT LLM FEEDBACK)\n\n")
                            tf.write(log_no_feedback.read_text(encoding="utf-8") + "\n\n")
                        if log_with_feedback.exists():
                            tf.write("### PASS 2: AUTONOMOUS SELF-REPAIR RUNS (WITH LLM FEEDBACK)\n\n")
                            tf.write(log_with_feedback.read_text(encoding="utf-8") + "\n\n")
                        if run_pure and log_pure.exists():
                            tf.write("### PURE LLM CHAT BASELINE (NO FRAMEWORK)\n\n")
                            tf.write(log_pure.read_text(encoding="utf-8") + "\n\n")
                    shutil.copyfile(log_combined, master_log_combined)

                # Display per-preset summaries
                active_res = results_no or results_with or results_pure
                if results_no:
                    print_pass_summary(
                        title=f"Pass 1: Deterministic Baseline [{preset_key.upper()}]",
                        results=results_no,
                        duration=dur_no,
                        log_path=log_no_feedback,
                        prompt=current_prompt,
                        use_rich=args.use_rich,
                    )
                if results_with:
                    print_pass_summary(
                        title=f"Pass 2: Autonomous Self-Repair [{preset_key.upper()}]",
                        results=results_with,
                        duration=dur_with,
                        log_path=log_with_feedback,
                        prompt=current_prompt,
                        use_rich=args.use_rich,
                    )
                if results_pure and (not run_framework or mode == "pure"):
                    print_pass_summary(
                        title=f"Pure LLM Chat Baseline [{preset_key.upper()}]",
                        results=results_pure,
                        duration=dur_pure,
                        log_path=log_pure,
                        prompt=current_prompt,
                        use_rich=args.use_rich,
                    )
                if mode == "dual" and results_no and results_with:
                    print_comparative_table(results_no, results_with, use_rich=args.use_rich)
                if results_pure and (results_no or results_with):
                    fw_for_cmp = results_with or results_no
                    print_pure_comparison_table(
                        preset_key=preset_key,
                        framework_results=fw_for_cmp,
                        pure_result=results_pure[0],
                        use_rich=args.use_rich,
                    )

                # Record preset metrics for master overnight summary
                passed_cnt = sum(1 for r in active_res if r.get("status") == "PASSED")
                synth_cnt = sum(1 for r in active_res if r.get("status") == "SYNTHESIZED")
                arch_cnt = sum(1 for r in active_res if r.get("category") in EM.REFUSAL_CATEGORIES)
                impl_cnt = sum(1 for r in active_res if r.get("category") == "IMPLEMENTATION")
                pure_status = results_pure[0].get("status") if results_pure else None
                overnight_prompt_summaries.append({
                    "preset_key": preset_key,
                    "expected": PROMPT_PRESETS.get(preset_key, {}).get("expected", "synthesize"),
                    "prompt": current_prompt,
                    "total_runs": len(active_res),
                    "passed": passed_cnt,
                    "synthesized": synth_cnt,
                    "arch": arch_cnt,
                    "impl": impl_cnt,
                    "pure_status": pure_status,
                    "pure_detail": results_pure[0].get("detail") if results_pure else "",
                    "fw_results": list(results_no or results_with),
                })

            if not args.overnight_loop and iteration >= args.iterations:
                break

    except KeyboardInterrupt:
        print("\n[!] Overnight test run interrupted by user (Ctrl+C). Generating summary of completed runs...")

    total_suite_duration = time.time() - t_suite_start
    # NOTE: no "estimated time saved" figure is reported. The former value was a hardcoded constant per run
    # (8.5 s), not a measurement. Layer-1 routing is cleared per run by default, so its latency is measured.

    if len(prompts_to_test) > 1 or iteration > 1:
        print_overnight_master_summary(
            prompt_summaries=overnight_prompt_summaries,
            total_suite_duration=total_suite_duration,
            cold_route=not args.warm_route_cache,
            use_rich=args.use_rich,
        )

    if run_pure and overnight_prompt_summaries and not args.dry_run:
        print("\n" + "=" * 80)
        print(" ⚔️ CROSS-PRESET COMPARISON: NSTL FRAMEWORK vs PURE LLM CHAT")
        print("=" * 80)
        fw_pass_tot = 0
        fw_tot = 0
        pure_pass_tot = 0
        pure_tot = 0
        for s in overnight_prompt_summaries:
            pk = s.get("preset_key", "custom")
            p_st = s.get("pure_status") or "N/A"
            p_det = s.get("pure_detail") or ""
            if p_st != "N/A":
                pure_tot += 1
                if p_st in ("PASSED", "SYNTHESIZED"):
                    pure_pass_tot += 1
            fw_res_list = s.get("fw_results") or []
            if fw_res_list:
                for fr in fw_res_list:
                    fw_tot += 1
                    f_st = fr.get("status", "N/A")
                    if f_st in ("PASSED", "SYNTHESIZED"):
                        fw_pass_tot += 1
                    print(f"  [{pk:<12}] Profile {fr.get('profile','?'):<2} + {fr.get('method','?'):<5} -> {f_st:<16} | PURE -> {p_st:<15} ({p_det[:55]})")
            else:
                print(f"  [{pk:<12}] PURE -> {p_st:<15} ({p_det[:65]})")
        if fw_tot:
            print("-" * 80)
            print(f"  Framework Pass Rate: {fw_pass_tot}/{fw_tot} ({fw_pass_tot/fw_tot*100:.1f}%)")
        if pure_tot:
            print(f"  Pure Chat Pass Rate: {pure_pass_tot}/{pure_tot} ({pure_pass_tot/pure_tot*100:.1f}%)")
        print("=" * 80)

    print("=" * 80)
    print(" 📁 Master Generated Log Artifacts:")
    print(f"  • Combined Master Log:  {master_log_combined}")
    if (ROOT_DIR / "total_no_feedback.log").exists():
        print(f"  • Without Feedback Log: {ROOT_DIR / 'total_no_feedback.log'}")
    if (ROOT_DIR / "total_with_feedback.log").exists():
        print(f"  • With Feedback Log:    {ROOT_DIR / 'total_with_feedback.log'}")
    if (ROOT_DIR / "total_pure.log").exists():
        print(f"  • Pure Chat Log:        {ROOT_DIR / 'total_pure.log'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
