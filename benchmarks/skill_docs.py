"""
benchmarks/skill_docs.py - Neutral domain-tree to SKILL.md serializer & prompt builder.

Builds controlled comparison prompts across three context arms:
  - pure              : Prompt only (no SKILL.md documentation; optional env availability note)
  - skill_md          : Loaded domain_set rendered cleanly as SKILL.md / SDK reference docs
  - skill_md_haystack : Full 15-domain catalog noise + target SDK docs at midpoint
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
TREES_DIR = ROOT_DIR / "trees"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

DEFAULT_PURE_SYS = (
    "You are a concise, precise Python developer. Write ONLY clean executable Python code inside a single "
    "```python ... ``` code block solving the user's request. Use standard/installed libraries only. "
    "Preserve every literal value, filename, column name, and parameter specified in the prompt. "
    "If the prompt is genuinely too underspecified to write concrete working code without guessing critical missing rules, "
    "missing required file paths, or requesting an unavailable library, ask 2 concise clarifying questions or state the "
    "missing requirement instead of writing speculative or substitute code."
)

VALID_CONTEXT_MODES: Tuple[str, ...] = ("pure", "skill_md", "skill_md_haystack")


def DOMAIN_SET_TREE_PATHS(root_dir: Path, domain_set: str) -> List[Path]:
    """Return ordered list of canonical tree JSON paths matching a named domain_set."""
    trees_dir = Path(root_dir) / "trees"
    fixtures_dir = Path(root_dir) / "benchmarks" / "fixtures"
    std_trees = sorted(
        p for p in trees_dir.glob("*.json")
        if p.name not in {"manifest.json", "index.json"}
    )

    if domain_set == "default":
        return std_trees
    if domain_set == "cryopulse_only":
        return [trees_dir / "python_core.json", fixtures_dir / "cryopulse" / "cryopulse.json"]
    if domain_set == "aetheris_only":
        return [trees_dir / "python_core.json", fixtures_dir / "aetheris" / "aetheris.json"]
    if domain_set == "haystack_cryopulse":
        return std_trees + [fixtures_dir / "cryopulse" / "cryopulse.json"]
    if domain_set == "haystack_aetheris":
        return std_trees + [fixtures_dir / "aetheris" / "aetheris.json"]
    if domain_set == "sqlite_only":
        return [trees_dir / "python_core.json", trees_dir / "sqlite3.json"]
    if domain_set == "cv2_only":
        return [trees_dir / "python_core.json", trees_dir / "numpy.json", trees_dir / "cv2.json"]
    if domain_set == "cv2_and_pillow":
        return [
            trees_dir / "python_core.json",
            trees_dir / "numpy.json",
            trees_dir / "cv2.json",
            trees_dir / "pillow.json",
        ]
    raise ValueError(f"Unknown domain_set: {domain_set!r}")


def _evenly_sample(items: List[Any], max_items: Optional[int]) -> List[Any]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return items
    n = len(items)
    chosen_idx: List[int] = []
    seen = set()
    for i in range(max_items):
        idx = int(i * n / max_items)
        if idx >= n:
            idx = n - 1
        if idx not in seen:
            seen.add(idx)
            chosen_idx.append(idx)
    return [items[i] for i in chosen_idx]


def _first_sentence(text: str, limit: int = 90) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return ""
    dot = clean.find(". ")
    if 0 < dot <= limit:
        return clean[: dot + 1]
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 3)].rstrip() + "..."


def _format_port_brief(pname: str, pspec: Dict[str, Any], *, compact: bool) -> str:
    carrier = str(pspec.get("carrier") or "any")
    tstate = str(pspec.get("typestate") or "")
    rmode = str(pspec.get("resource_mode") or "value")
    req = bool(pspec.get("required", True))
    has_def = "default_value" in pspec and pspec.get("default_value") is not None
    type_str = f"{carrier}[{tstate}]" if tstate and tstate != "any" else carrier
    if compact:
        suffix = f"={pspec['default_value']!r}" if (not req and has_def) else ("" if req else "=None")
        if rmode in ("acquire", "release", "borrow"):
            type_str = f"{type_str}({rmode})"
        return f"{pname}: {type_str}{suffix}"
    flags = [f"role={pspec.get('value_class', 'data')}", f"mode={rmode}", "req" if req else f"opt={pspec.get('default_value')!r}"]
    return f"{pname}: {type_str} ({', '.join(flags)})"


def _format_cell_line(cell: Dict[str, Any], *, compact: bool) -> str:
    cid = str(cell.get("cell_id") or "")
    ins = cell.get("inputs") or {}
    outs = cell.get("outputs") or {}
    in_parts = [_format_port_brief(pn, ps, compact=compact) for pn, ps in ins.items() if isinstance(ps, dict)]
    out_parts = [_format_port_brief(pn, ps, compact=compact) for pn, ps in outs.items() if isinstance(ps, dict)]
    out_str = ", ".join(out_parts) if out_parts else "None"
    tmpl = " ".join(str(cell.get("code_template") or "").split())
    doc = _first_sentence(str(cell.get("docstring") or ""), limit=75 if compact else 140)
    if compact:
        return f"- `{cid}({', '.join(in_parts)}) -> {out_str}` | `{tmpl}` — {doc}"
    deps = ", ".join(str(d) for d in (cell.get("dependencies") or ()))
    mut = str(cell.get("mutation_type") or "pure")
    return (
        f"### `{cid}` (`{mut}`)\n"
        f"- Inputs: {', '.join(in_parts) if in_parts else 'none'}\n"
        f"- Outputs: {out_str}\n"
        f"- Imports: {deps or 'none'} | Template: `{tmpl}`\n"
        f"- Summary: {doc}"
    )


def _extract_type_names(raw_types: Any) -> List[str]:
    if isinstance(raw_types, dict):
        return sorted(str(k) for k in raw_types.keys() if str(k).strip())
    if isinstance(raw_types, list):
        return sorted(str(item.get("name")) for item in raw_types if isinstance(item, dict) and item.get("name"))
    return []


def render_tree_skill_md(
    tree_path: Path,
    *,
    compact: bool = False,
    max_cells: Optional[int] = None,
) -> Tuple[str, int]:
    """Serialize one domain JSON manifest into structured SKILL.md reference text."""
    data = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    dom = str(data.get("domain") or Path(tree_path).stem).strip()
    aliases = [str(a) for a in (data.get("aliases") or ()) if str(a).strip()]
    imports = [str(i) for i in (data.get("imports") or ()) if str(i).strip()]
    desc = str(data.get("description") or "").strip()
    cells = [c for c in (data.get("cells") or ()) if isinstance(c, dict)]
    sampled = _evenly_sample(cells, max_cells)

    header_parts = [f"## Domain: `{dom}` ({len(cells)} total cells; showing {len(sampled)})"]
    meta_bits: List[str] = []
    if aliases:
        meta_bits.append(f"aliases={aliases}")
    if imports:
        meta_bits.append(f"imports={imports}")
    type_names = _extract_type_names(data.get("types"))
    if type_names:
        shown_types = ", ".join(type_names[:10]) + ("..." if len(type_names) > 10 else "")
        meta_bits.append(f"types=[{shown_types}]")
    if meta_bits:
        header_parts.append("Metadata: " + " | ".join(meta_bits))
    if desc:
        header_parts.append(_first_sentence(desc, limit=180))

    lines = list(header_parts)
    for cell in sampled:
        lines.append(_format_cell_line(cell, compact=compact))
    return "\n".join(lines), len(sampled)


def _fit_tree_sections_to_budget(
    tree_paths: List[Path],
    *,
    max_chars: Optional[int],
    compact: bool,
) -> Tuple[List[str], List[str], int, int]:
    """Render each tree JSON, scaling per-domain cell caps evenly if constrained by character budget."""
    raw_metas: List[Tuple[Path, str, int]] = []
    total_all_cells = 0
    for tp in tree_paths:
        data = json.loads(tp.read_text(encoding="utf-8"))
        dom = str(data.get("domain") or tp.stem).strip()
        n_cells = len(data.get("cells") or ())
        total_all_cells += n_cells
        raw_metas.append((tp, dom, n_cells))

    if max_chars is None:
        sections: List[str] = []
        shown_ops = 0
        for tp, _, _ in raw_metas:
            sec, n_op = render_tree_skill_md(tp, compact=compact, max_cells=None)
            sections.append(sec)
            shown_ops += n_op
        return sections, [m[1] for m in raw_metas], shown_ops, total_all_cells

    per_dom_budget = max(450, int(max_chars / max(1, len(raw_metas))))
    approx_chars_per_cell = 115 if compact else 240
    sections = []
    shown_ops = 0
    for tp, _, n_cells in raw_metas:
        est_cap = max(2, (per_dom_budget - 160) // approx_chars_per_cell)
        cap = None if n_cells <= est_cap else est_cap
        sec, n_op = render_tree_skill_md(tp, compact=compact, max_cells=cap)
        if len(sec) > per_dom_budget and cap is not None and cap > 2:
            sec, n_op = render_tree_skill_md(tp, compact=compact, max_cells=max(2, int(cap * per_dom_budget / len(sec))))
        sections.append(sec)
        shown_ops += n_op
    return sections, [m[1] for m in raw_metas], shown_ops, total_all_cells


def _load_authored_sdk_doc(sdk_name: str) -> str:
    return (FIXTURES_DIR / sdk_name / f"{sdk_name}_docs.md").read_text(encoding="utf-8").strip()


def _infer_domain_set(task: Dict[str, Any]) -> str:
    explicit = str(task.get("domain_set") or "").strip()
    if explicit:
        return explicit
    sdk = str(task.get("sdk") or "").strip().lower()
    if sdk == "cryopulse":
        return "cryopulse_only"
    if sdk == "aetheris":
        return "aetheris_only"
    return "default"


def build_skill_md_for_task(
    task: Dict[str, Any],
    *,
    force_haystack: bool = False,
    max_distractor_chars: int = 18000,
    full_haystack: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Build SKILL.md reference context strictly scoped to the task's domain_set (or haystack)."""
    domain_set = _infer_domain_set(task)
    sdk_name = str(task.get("sdk") or "").strip().lower()
    is_haystack = bool(force_haystack or domain_set.startswith("haystack_"))
    budget: Optional[int] = None if full_haystack else int(max_distractor_chars)

    if is_haystack:
        std_paths = DOMAIN_SET_TREE_PATHS(ROOT_DIR, "default")
        sections, doms, shown_ops, total_cells = _fit_tree_sections_to_budget(
            std_paths,
            max_chars=budget,
            compact=True,
        )
        all_doms = list(doms)
        needle_ops = 0
        if sdk_name in ("cryopulse", "aetheris"):
            sdk_doc = _load_authored_sdk_doc(sdk_name)
            sdk_tree = json.loads((FIXTURES_DIR / sdk_name / f"{sdk_name}.json").read_text(encoding="utf-8"))
            needle_ops = len(sdk_tree.get("cells") or ())
            total_cells += needle_ops
            shown_ops += needle_ops
            mid = len(sections) // 2
            sections.insert(mid, sdk_doc)
            all_doms.insert(mid, sdk_name)
        header = (
            f"# Loaded STRATA SKILL.md Haystack ({len(all_doms)} domains, "
            f"{total_cells} total cells in lattice, {shown_ops} operations documented)\n"
            f"Loaded domains: {', '.join(all_doms)}"
        )
        md_text = header + "\n\n" + "\n\n".join(sections)
        return md_text, {
            "domain_set": f"haystack_{sdk_name}" if sdk_name else "haystack_default",
            "domains": all_doms,
            "ops_documented": shown_ops,
            "total_lattice_cells": total_cells,
            "chars": len(md_text),
            "haystack": True,
        }

    if domain_set in ("cryopulse_only", "aetheris_only") and sdk_name in ("cryopulse", "aetheris"):
        sdk_doc = _load_authored_sdk_doc(sdk_name)
        sdk_tree = json.loads((FIXTURES_DIR / sdk_name / f"{sdk_name}.json").read_text(encoding="utf-8"))
        n_cells = len(sdk_tree.get("cells") or ())
        header = (
            f"# Loaded STRATA SKILL.md (domains: python_core, {sdk_name}; "
            f"{n_cells} target SDK cells)\n"
        )
        md_text = header + "\n" + sdk_doc
        return md_text, {
            "domain_set": domain_set,
            "domains": ["python_core", sdk_name],
            "ops_documented": n_cells,
            "total_lattice_cells": n_cells,
            "chars": len(md_text),
            "haystack": False,
        }

    tree_paths = DOMAIN_SET_TREE_PATHS(ROOT_DIR, domain_set)
    use_compact = len(tree_paths) >= 3
    sections, doms, shown_ops, total_cells = _fit_tree_sections_to_budget(
        tree_paths,
        max_chars=budget,
        compact=use_compact,
    )
    header = (
        f"# Loaded STRATA SKILL.md ({len(doms)} domains, {total_cells} total cells, "
        f"{shown_ops} operations shown)\n"
        f"Loaded domains ONLY: {', '.join(doms)}\n"
        f"Do NOT import or call any library outside these loaded domains plus Python stdlib."
    )
    md_text = header + "\n\n" + "\n\n".join(sections)
    return md_text, {
        "domain_set": domain_set,
        "domains": doms,
        "ops_documented": shown_ops,
        "total_lattice_cells": total_cells,
        "chars": len(md_text),
        "haystack": False,
    }


def build_messages_for_arm(
    task: Dict[str, Any],
    context_mode: str = "skill_md",
    *,
    max_distractor_chars: int = 18000,
    full_haystack: bool = False,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Build chat messages and metadata for 'pure', 'skill_md', or 'skill_md_haystack'."""
    if context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"Invalid context_mode {context_mode!r}; expected one of {VALID_CONTEXT_MODES}")

    domain_set = _infer_domain_set(task)
    avail_note = str(task.get("available_libraries_note") or "").strip()

    if context_mode == "pure":
        sys_msg = DEFAULT_PURE_SYS
        if avail_note:
            sys_msg += f"\n\n{avail_note}"
        meta = {
            "context_mode": "pure",
            "domain_set": domain_set,
            "domains": [],
            "ops_documented": 0,
            "total_lattice_cells": 0,
            "chars": len(sys_msg),
            "haystack": False,
        }
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": task["prompt"]},
        ], meta

    md_text, ctx_meta = build_skill_md_for_task(
        task,
        force_haystack=(context_mode == "skill_md_haystack"),
        max_distractor_chars=max_distractor_chars,
        full_haystack=full_haystack,
    )
    ctx_meta["context_mode"] = context_mode

    extra_rule = f"\nAdditional constraint: {avail_note}" if avail_note else ""
    sys_msg = (
        "You are a precise Python synthesis assistant.\n"
        "Strict Rules:\n"
        "1. Use ONLY the APIs and libraries documented in the loaded SKILL.md specification below (plus Python stdlib when needed).\n"
        "2. Preserve every literal filename, value, identifier, parameter, and order constraint from the prompt.\n"
        "3. Always properly close or release any acquired resource handles using the corresponding cleanup operation documented below.\n"
        "4. If the prompt is underspecified, omits required external file paths, or requests a library/domain NOT present in the loaded SKILL.md below, "
        "do NOT substitute another library or write speculative code — instead ask 2 concise clarifying questions or state the missing domain/asset WITHOUT emitting a code block."
        f"{extra_rule}\n\n"
        + md_text
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": task["prompt"]},
    ], ctx_meta
