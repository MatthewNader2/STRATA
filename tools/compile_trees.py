"""
tools/compile_trees.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Compiles trees/*.json into SQLite.

Table and column names match the loader contract in src/lattice.py
(see the SELECT FROM nodes at ~line 1650):

    cell_id, domain_name, node_type, node_role, stage, keywords,
    input_type, input_state, output_type, output_state, code,
    dependencies, configuration_schema, slots, verified, docstring,
    source_priority

The loader's extended-state fields (inputs, outputs, slots, bound_slots,
postconditions, ...) are packed as a JSON dict into ``configuration_schema``.
Additional columns are kept as pass-through for downstream tooling; the
loader ignores unknown columns.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DB_PATH = os.path.join(PROJECT_ROOT, "trees", "lattice.db")


# --------------------------------------------------------------------------- #
# Column layout — matches the loader's SELECT FROM nodes field order.
# --------------------------------------------------------------------------- #

# Required by the loader (or synthesized by it).
_LOADER_COLUMNS: Tuple[str, ...] = (
    "cell_id",
    "domain_name",
    "node_type",
    "node_role",
    "stage",
    "keywords",
    "input_type",
    "input_state",
    "output_type",
    "output_state",
    "code",
    "dependencies",
    "configuration_schema",
    "slots",
    "verified",
    "docstring",
    "source_priority",
)

# Extra pass-through columns for downstream tooling; loader ignores them.
_EXTRA_COLUMNS: Tuple[str, ...] = (
    "domain",
    "doc",
    "inputs",
    "outputs",
    "code_templates",
    "topology_type",
    "mutation_type",
    "is_context_manager",
    "bound_slots",
    "preconditions",
    "postconditions",
    "effects",
    "type_vars",
    "raises",
    "semantic_tags",
    "primary_in",
    "primary_out",
    "is_public",
    "source_provenance",
    "raw_cell",
)

_COLUMNS: Tuple[str, ...] = _LOADER_COLUMNS + _EXTRA_COLUMNS

_INT_COLUMNS: frozenset = frozenset({
    "stage", "verified", "is_context_manager", "is_public", "source_priority",
})

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    cell_id               TEXT PRIMARY KEY,
    domain_name           TEXT,
    node_type             TEXT,
    node_role             TEXT,
    stage                 INTEGER,
    keywords              TEXT,
    input_type            TEXT,
    input_state           TEXT,
    output_type           TEXT,
    output_state          TEXT,
    code                  TEXT,
    dependencies          TEXT,
    configuration_schema  TEXT,
    slots                 TEXT,
    verified              INTEGER,
    docstring             TEXT,
    source_priority       INTEGER,
    domain                TEXT,
    doc                   TEXT,
    inputs                TEXT,
    outputs               TEXT,
    code_templates        TEXT,
    topology_type         TEXT,
    mutation_type         TEXT,
    is_context_manager    INTEGER,
    bound_slots           TEXT,
    preconditions         TEXT,
    postconditions        TEXT,
    effects               TEXT,
    type_vars             TEXT,
    raises                TEXT,
    semantic_tags         TEXT,
    primary_in            TEXT,
    primary_out           TEXT,
    is_public             INTEGER,
    source_provenance     TEXT,
    raw_cell              TEXT
);
"""

INSERT_SQL = f"""
    INSERT INTO nodes ({", ".join(_COLUMNS)})
    VALUES ({", ".join("?" * len(_COLUMNS))})
    ON CONFLICT(cell_id) DO UPDATE SET
        {", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "cell_id")}
"""


# --------------------------------------------------------------------------- #
# Structural helpers — no domain vocabulary, no extension lists.
# --------------------------------------------------------------------------- #

_DOTTED_NAME_RE = re.compile(
    r'(?<![\'"\w.])([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b(?![\'"])'
)
_QUOTED_PATH_RE = re.compile(r'(["\'])([^"\']*[/\\][^"\']*)\1')
_PLACEHOLDER_SUBST_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def _module_aliases(dependencies: Any) -> Set[str]:
    aliases: Set[str] = set()

    if isinstance(dependencies, str):
        dependencies = [dependencies]
    if not dependencies:
        return aliases

    def _add_dotted(path: str) -> None:
        aliases.add(path)
        aliases.add(path.split(".")[0])
        tail = path.rsplit(".", 1)[-1]
        if tail != path:
            aliases.add(tail)

    for dep in dependencies:
        if isinstance(dep, dict):
            for key in ("import", "module", "name", "statement"):
                if isinstance(dep.get(key), str):
                    dep = dep[key]
                    break
            else:
                for v in dep.values():
                    if isinstance(v, str):
                        dep = v
                        break
                else:
                    continue

        dep_s = str(dep).strip()
        if not dep_s:
            continue

        m = re.match(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?", dep_s)
        if m:
            _add_dotted(m.group(1))
            if m.group(2):
                aliases.add(m.group(2))
            continue

        m = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)$", dep_s)
        if m:
            _add_dotted(m.group(1))
            for name in m.group(2).split(","):
                name = name.strip()
                if not name or name == "*":
                    continue
                as_m = re.match(r"^(\w+)\s+as\s+(\w+)$", name)
                aliases.add(as_m.group(2) if as_m else name)
            continue

        m = re.match(r"^([\w.]+)(?:\s+as\s+(\w+))?$", dep_s)
        if m:
            _add_dotted(m.group(1))
            if m.group(2):
                aliases.add(m.group(2))
            continue

    return aliases


def _unbound_dotted_reference(token: str, declared_aliases: Set[str]) -> Optional[str]:
    if "." not in token:
        return None
    prefix = token.split(".", 1)[0]
    return None if prefix in declared_aliases else prefix


def _validate_template(
    code: str,
    cell_id: str,
    node_role: str = "function",
    declared_aliases: Optional[Set[str]] = None,
    supported_formats: Optional[Iterable[str]] = None,
) -> Tuple[bool, Optional[str]]:
    if not code or not code.strip():
        return False, "Empty code template"

    for m in _QUOTED_PATH_RE.finditer(code):
        return False, f"Hardcoded path literal '{m.group(2)}'"

    dummy_code = _PLACEHOLDER_SUBST_RE.sub("dummy_var", code)
    try:
        parsed_tree = ast.parse(dummy_code)
    except SyntaxError as e:
        return False, f"AST SyntaxError: {e}"

    import builtins
    valid_prefixes = set(dir(builtins))
    if declared_aliases is not None:
        valid_prefixes.update(declared_aliases)

    # Collect lexically bound names within the template itself
    for node in ast.walk(parsed_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        valid_prefixes.add(n.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                valid_prefixes.add(node.target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            valid_prefixes.add(n.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    valid_prefixes.add(n.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if hasattr(node, "name"):
                valid_prefixes.add(node.name)
            for arg in node.args.args:
                valid_prefixes.add(arg.arg)
        elif hasattr(ast, "NamedExpr") and isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                valid_prefixes.add(node.target.id)

    # Also add template placeholder variables
    for m in _PLACEHOLDER_SUBST_RE.finditer(code):
        var_name = m.group(0)[1:-1]
        valid_prefixes.add(var_name)

    if declared_aliases is not None:
        for m in _DOTTED_NAME_RE.finditer(code):
            token = m.group(1)
            if f"{{{token}}}" in code:
                continue
            unbound = _unbound_dotted_reference(token, valid_prefixes)
            if unbound is not None:
                return False, (
                    f"Unbound reference '{token}' "
                    f"(prefix '{unbound}' not in declared dependencies)"
                )

    check_exts = {f.lower().lstrip(".") for f in (supported_formats or []) if f != "*"}
    if check_exts:
        # AST Walk: inspect Attribute nodes for unquoted bare filenames matching declared formats
        for parent in ast.walk(parsed_tree):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, ast.Attribute):
                    if isinstance(parent, ast.Call) and parent.func is child:
                        continue
                    if child.attr.lower() in check_exts:
                        if isinstance(child.value, ast.Name):
                            if declared_aliases is None or child.value.id not in declared_aliases:
                                return False, f"Bare unquoted filename detected as attribute argument: {child.value.id}.{child.attr}"

                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    val = child.value.strip()
                    if any(val.lower().endswith(f".{ext}") for ext in check_exts):
                        if not ("{" in val and "}" in val):
                            if "/" not in val and "\\" not in val and len(val.split(".")) == 2:
                                return False, f"Hardcoded string filename detected: '{val}' (must use a placeholder)"

    return True, None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _json_or_null(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return None


def _extract_python_template(node: dict) -> Tuple[str, dict]:
    code_templates = node.get("code_templates", {}) or {}
    if not isinstance(code_templates, dict):
        code_templates = {}
    py_template = ""
    if code_templates.get("python"):
        py_template = code_templates["python"]
    elif node.get("code_template"):
        py_template = node["code_template"]
    elif node.get("template"):
        py_template = node["template"]
    if py_template and "python" not in code_templates:
        code_templates = {"python": py_template, **code_templates}
    return py_template, code_templates


def _primary_port_sig(node: dict, direction: str) -> Tuple[str, str]:
    """(type_name, state) for the primary port in ``direction``.

    Prefers the port named by ``node['primary_in']`` / ``node['primary_out']``;
    falls back to the first declared port. Returns ``("any", "any")`` when no
    port exists.
    """
    ports = node.get(direction, {}) or {}
    if not isinstance(ports, dict) or not ports:
        return ("any", "any")

    primary_key = "primary_in" if direction == "inputs" else "primary_out"
    prim_name = node.get(primary_key, "") or ""
    chosen = None
    if prim_name and prim_name in ports:
        chosen = ports[prim_name]
    else:
        chosen = next(iter(ports.values()))

    if not isinstance(chosen, dict):
        return ("any", "any")
    return (
        str(chosen.get("type_name", "any") or "any"),
        str(chosen.get("state", "any") or "any"),
    )


def compute_structural_edge(producer: dict, consumer: dict, p_out: dict, p_in: dict) -> Optional[dict]:
    if producer.get("cell_id") == consumer.get("cell_id"):
        return None

    # Universal mathematical edge-validity predicate e = (u -> v)
    from unification import is_edge_valid
    if not is_edge_valid(p_out, p_in, producer, consumer):
        return None

    score = 0.70
    if producer.get("domain") == consumer.get("domain"):
        score += 0.15

    t_out = str(p_out.get("type_name") or "").lower()
    t_in = str(p_in.get("type_name") or "").lower()
    if t_out and t_in and t_out == t_in and t_out != "any":
        score += 0.10

    state_out = str(p_out.get("state") or "any").lower()
    state_in = str(p_in.get("state") or "any").lower()
    if state_out == state_in and state_out != "any":
        score += 0.05

    score = max(0.1, min(1.0, round(score, 3)))

    return {
        "target_cell_id": consumer["cell_id"],
        "affinity_score": score,
        "bridging_precondition": "true",
        "score_provenance": "structural_unification",
    }


def synthesize_dynamic_edges(all_cells: List[Tuple[str, dict]]) -> Dict[str, List[dict]]:
    """Synthesizes dynamic transition edges between compatible cells using carrier-bucketed indexing."""
    from lattice import TypeRegistry, build_cell_from_raw, _apply_method_owner_qualification
    from experiments import build_indexed_topology

    edges_by_cell: Dict[str, List[dict]] = {
        node["cell_id"]: [] for _, node in all_cells if isinstance(node, dict) and "cell_id" in node
    }
    loaded_cells: Dict[str, Any] = {}
    raw_by_id: Dict[str, Tuple[str, dict]] = {}
    for domain, c in all_cells:
        if not isinstance(c, dict) or "cell_id" not in c:
            continue
        cid = c["cell_id"]
        c_domain = c.get("domain_name") or domain
        raw_by_id[cid] = (c_domain, c)
        try:
            cell_obj = build_cell_from_raw(c, domain=c_domain)
            _apply_method_owner_qualification(cell_obj)
            loaded_cells[cid] = cell_obj
        except Exception:
            pass

    reg = TypeRegistry.get_instance()
    adjacency, _ = build_indexed_topology(list(loaded_cells.values()), loaded_cells, reg)

    for u_id, target_ids in adjacency.items():
        if u_id not in raw_by_id:
            continue
        u_domain, u_raw = raw_by_id[u_id]
        u_out_type, u_out_state = _primary_port_sig(u_raw, "outputs")
        u_out_t_low = u_out_type.lower()
        u_out_s_low = u_out_state.lower()
        c_edges = []
        for v_id in target_ids:
            if v_id == u_id or v_id not in raw_by_id:
                continue
            v_domain, v_raw = raw_by_id[v_id]
            v_in_type, v_in_state = _primary_port_sig(v_raw, "inputs")
            score = 0.70
            if u_domain == v_domain:
                score += 0.15
            if u_out_t_low and u_out_t_low == v_in_type.lower() and u_out_t_low != "any":
                score += 0.10
            if u_out_s_low and u_out_s_low == v_in_state.lower() and u_out_s_low != "any":
                score += 0.05
            score = max(0.1, min(1.0, round(score, 3)))
            c_edges.append({
                "target_cell_id": v_id,
                "affinity_score": score,
                "bridging_precondition": "true",
                "score_provenance": "structural_unification",
            })
        c_edges.sort(key=lambda x: x["affinity_score"], reverse=True)
        edges_by_cell[u_id] = c_edges

    return edges_by_cell


def extract_node_data(node: dict, domain: str, dynamic_edges: Optional[List[dict]] = None) -> tuple:
    py_template, code_templates = _extract_python_template(node)

    node_role = node.get("node_role") or node.get("role") or "function"
    node_type = node.get("node_type") or "function"

    dependencies = node.get("dependencies", []) or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]

    slots = node.get("slots", {}) or {}

    in_type, in_state = _primary_port_sig(node, "inputs")
    out_type, out_state = _primary_port_sig(node, "outputs")

    # configuration_schema carries every extended-state field the loader reads
    # after the initial SELECT — inputs, outputs, slots, postconditions, etc.
    # edges are dynamically synthesized based on structural type compatibility.
    configuration_schema = {
        "inputs": node.get("inputs", {}) or {},
        "outputs": node.get("outputs", {}) or {},
        "slots": slots,
        "bound_slots": node.get("bound_slots", {}) or {},
        "preconditions": node.get("preconditions", []) or [],
        "postconditions": node.get("postconditions", []) or [],
        "effects": node.get("effects", []) or [],
        "type_vars": node.get("type_vars", []) or [],
        "raises": node.get("raises", []) or [],
        "semantic_tags": node.get("semantic_tags", []) or [],
        "topology_type": node.get("topology_type", "") or "",
        "mutation_type": node.get("mutation_type", "") or "",
        "is_context_manager": bool(node.get("is_context_manager", False)),
        "is_public": bool(node.get("is_public", True)),
        "source_provenance": node.get("source_provenance", "") or "",
        "primary_in": node.get("primary_in", "") or "",
        "primary_out": node.get("primary_out", "") or "",
        "code_templates": code_templates,
        "edges": dynamic_edges if dynamic_edges is not None else [],
        "endable": bool(node["endable"]) if "endable" in node else None,
        "metadata_tags": node.get("metadata_tags", {}) or {},
        "sub_cells": node.get("sub_cells", []) or [],
        "algorithmic_steps": node.get("algorithmic_steps", []) or [],
        "internal_topology": node.get("internal_topology", {}) or {},
        "projection": node.get("projection"),
        "optic_role": node.get("optic_role"),
        "container_port": node.get("container_port"),
        "key_port": node.get("key_port"),
        "focus_port": node.get("focus_port"),
    }

    return (
        # --- loader contract ---
        node["cell_id"],
        # Per-cell declared domain wins over the file-level domain: macro-goal
        # trees (domain "macros") declare their cells' functional domain so
        # domain-coherence scoring sees the expansion's true home domain.
        node.get("domain_name") or domain,
        node_type,
        node_role,
        node["stage"],
        _json_or_null(node.get("keywords", [])),
        in_type,
        in_state,
        out_type,
        out_state,
        py_template,
        _json_or_null(dependencies),
        _json_or_null(configuration_schema),
        _json_or_null(slots),
        1 if node.get("verified", True) else 0,
        node.get("docstring", "") or "",
        int(node.get("source_priority", 100) or 100),
        # --- extras ---
        domain,
        node.get("doc", "") or "",
        _json_or_null(node.get("inputs", {})),
        _json_or_null(node.get("outputs", {})),
        _json_or_null(code_templates) if code_templates else _json_or_null({"python": py_template}),
        node.get("topology_type", "") or "",
        node.get("mutation_type", "") or "",
        1 if node.get("is_context_manager", False) else 0,
        _json_or_null(node.get("bound_slots", {})),
        _json_or_null(node.get("preconditions", [])),
        _json_or_null(node.get("postconditions", [])),
        _json_or_null(node.get("effects", [])),
        _json_or_null(node.get("type_vars", [])),
        _json_or_null(node.get("raises", [])),
        _json_or_null(node.get("semantic_tags", [])),
        node.get("primary_in", "") or "",
        node.get("primary_out", "") or "",
        1 if node.get("is_public", True) else 0,
        node.get("source_provenance", "") or "",
        _json_or_null(node),
    )


# --------------------------------------------------------------------------- #
# Tree iteration / schema
# --------------------------------------------------------------------------- #

def _iter_tree_nodes(path: Path) -> Iterable[Tuple[str, dict]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        default_domain = path.stem
        for node in data:
            yield (node.get("domain", default_domain), node)
        return
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported tree JSON shape in {path}: {type(data).__name__}")
    domain = data.get("domain", path.stem)
    for key in ("cells", "nodes", "morphisms"):
        if key in data and isinstance(data[key], list):
            for node in data[key]:
                yield (node.get("domain", domain), node)
            return
    yield (domain, data)


def _resolve_tree_files(
    trees_dir: Path, domain_filter: Optional[List[str]]
) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(trees_dir / "*.json")))
    files = [p for p in files if p.name not in {"manifest.json", "index.json"}]
    if domain_filter:
        wanted = {d.lower() for d in domain_filter}
        files = [p for p in files if p.stem.lower() in wanted]
    return files


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the ``nodes`` table if missing; add any new columns."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS types;")
    cur.execute("DROP TABLE IF EXISTS typestates;")
    cur.execute("DROP TABLE IF EXISTS aliases;")
    cur.execute("DROP TABLE IF EXISTS artifact_readers;")
    cur.execute("DROP TABLE IF EXISTS structural_metadata;")
    cur.execute(_CREATE_TABLE_SQL)
    cur.execute(_CREATE_TYPES_TABLE_SQL)
    cur.execute(_CREATE_TYPESTATES_TABLE_SQL)
    cur.execute(_CREATE_ALIASES_TABLE_SQL)
    cur.execute(_CREATE_ARTIFACT_READERS_TABLE_SQL)
    cur.execute(_CREATE_STRUCTURAL_METADATA_TABLE_SQL)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(nodes)")}
    for col in _COLUMNS:
        if col not in existing:
            col_type = "INTEGER" if col in _INT_COLUMNS else "TEXT"
            cur.execute(f"ALTER TABLE nodes ADD COLUMN {col} {col_type}")
    conn.commit()


_CREATE_TYPES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS types (
    type_name   TEXT NOT NULL,
    parent_type TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (type_name, parent_type)
);
"""

_CREATE_TYPESTATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS typestates (
    state_name   TEXT PRIMARY KEY,
    parent_state TEXT,
    carrier_type TEXT,
    properties   TEXT
);
"""

_CREATE_ALIASES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT
);
"""

_CREATE_ARTIFACT_READERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS artifact_readers (
    category      TEXT NOT NULL,
    module_name   TEXT NOT NULL,
    function_name TEXT NOT NULL,
    PRIMARY KEY (category, module_name, function_name)
);
"""

_CREATE_STRUCTURAL_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS structural_metadata (
    category TEXT,
    item     TEXT,
    extra    TEXT,
    PRIMARY KEY (category, item)
);
"""


def _persist_artifact_readers(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    cur = conn.cursor()
    written = 0
    for path in tree_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        readers_map = data.get("artifact_readers")
        if isinstance(readers_map, dict):
            for cat, readers in readers_map.items():
                if not isinstance(readers, list):
                    continue
                for r in readers:
                    parts = str(r).replace(":", ".").rsplit(".", 1)
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        cur.execute(
                            "INSERT OR REPLACE INTO artifact_readers (category, module_name, function_name) "
                            "VALUES (?, ?, ?)",
                            (str(cat).strip().lower(), parts[0].strip(), parts[1].strip()),
                        )
                        written += 1
    conn.commit()
    return written


def _persist_aliases(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    cur = conn.cursor()
    written = 0
    for path in tree_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        aliases = data.get("aliases")
        if isinstance(aliases, dict):
            for a_k, a_v in aliases.items():
                cur.execute(
                    "INSERT OR REPLACE INTO aliases (alias, canonical) VALUES (?, ?)",
                    (str(a_k), str(a_v))
                )
                written += 1
    conn.commit()
    return written


def _merge_list_mapping(acc: dict, incoming: dict, lower_keys: bool = True) -> None:
    if not isinstance(incoming, dict):
        return
    for raw_k, raw_v in incoming.items():
        k = str(raw_k).strip().lower() if lower_keys else str(raw_k).strip()
        if not k:
            continue
        if isinstance(raw_v, (list, tuple, set)):
            vals = list(raw_v)
            prev = acc.get(k)
            if isinstance(prev, list):
                seen = {json.dumps(x, sort_keys=True) for x in prev}
                for item in vals:
                    marker = json.dumps(item, sort_keys=True)
                    if marker not in seen:
                        prev.append(item)
                        seen.add(marker)
            else:
                deduped = []
                seen = set()
                for item in vals:
                    marker = json.dumps(item, sort_keys=True)
                    if marker not in seen:
                        deduped.append(item)
                        seen.add(marker)
                acc[k] = deduped
        elif isinstance(raw_v, dict):
            prev = acc.get(k)
            if isinstance(prev, dict):
                _merge_list_mapping(prev, raw_v, lower_keys=lower_keys)
            else:
                acc[k] = dict(raw_v)
        else:
            acc[k] = raw_v


def _persist_structural_metadata(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    cur = conn.cursor()
    written = 0
    core_domains = {"", "python_core", "core", "base", "builtin", "builtins"}

    def _insert(cat: str, it: str, ex: Optional[str] = None):
        nonlocal written
        cur.execute(
            "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
            (cat, str(it), ex)
        )
        written += 1

    merged_role_semantics: dict = {}
    merged_capabilities: dict = {}
    merged_carrier_roles: dict = {}
    merged_state_role_ancestors: dict = {}
    merged_state_role_markers: dict = {}

    for path in tree_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        dom = str(data.get("domain") or path.stem).strip().lower()

        mappings = [
            ("dest_port_tokens", "dest_port_token"),
            ("source_port_tokens", "source_port_token"),
            ("data_bearing_roles", "data_bearing_role"),
            ("abstract_carriers", "abstract_carrier"),
            ("type_vars", "type_var"),
            ("top_types", "top_type"),
            ("top_states", "top_state"),
            ("product_constructors", "product_constructor"),
            ("materialization_states", "materialization_state"),
            ("top_level_carriers", "top_level_carrier"),
            ("path_port_tokens", "path_port_token"),
            ("identifier_root_states", "identifier_root_state"),
            ("supported_formats", "supported_format"),
        ]
        for key, cat in mappings:
            items = data.get(key)
            if isinstance(items, list):
                for item in items:
                    _insert(cat, item)

        ts_info = data.get("typestates")
        if isinstance(ts_info, dict) and isinstance(ts_info.get("terminal_states"), list):
            for ts_item in ts_info["terminal_states"]:
                _insert("terminal_state", ts_item)

        adv_q = data.get("advisory_qualifiers")
        if isinstance(adv_q, list):
            for q in adv_q:
                q_json = json.dumps(q, sort_keys=True)
                _insert("advisory_qualifier", q_json, q_json)

        acm = data.get("abstract_carrier_mapping")
        if isinstance(acm, dict):
            for k, val in acm.items():
                _insert("abstract_carrier_mapping", k, str(val))

        pdefs = data.get("pointed_defaults")
        if isinstance(pdefs, dict):
            for k, val in pdefs.items():
                k_low = str(k).strip().lower()
                if not k_low:
                    continue
                if dom:
                    _insert("type_default", f"{dom}:{k_low}", json.dumps(val))
                if dom in core_domains:
                    _insert("type_default", k_low, json.dumps(val))

        if isinstance(data.get("enum_constant_rules"), list):
            for rule in data["enum_constant_rules"]:
                if isinstance(rule, dict):
                    rule_json = json.dumps(rule, sort_keys=True)
                    _insert("enum_constant_rule", rule_json, rule_json)

        probes = data.get("mock_probes")
        if not probes:
            for k_p, v_p in data.items():
                if k_p.startswith("mock_") and isinstance(v_p, list):
                    probes = v_p
                    break
        if isinstance(probes, list):
            for probe in probes:
                if isinstance(probe, dict):
                    probe_json = json.dumps(probe, sort_keys=True)
                    _insert("mock_probe", probe_json, probe_json)

        _merge_list_mapping(merged_role_semantics, data.get("role_semantics"), lower_keys=True)
        _merge_list_mapping(merged_capabilities, data.get("capabilities"), lower_keys=False)
        _merge_list_mapping(merged_carrier_roles, data.get("carrier_roles"), lower_keys=True)
        _merge_list_mapping(merged_state_role_ancestors, data.get("state_role_ancestors"), lower_keys=True)
        _merge_list_mapping(merged_state_role_markers, data.get("state_role_markers"), lower_keys=True)

        cells = data.get("cells") or data.get("nodes") or data.get("morphisms") or []
        if isinstance(cells, list):
            for c in cells:
                if not isinstance(c, dict):
                    continue
                for tv in c.get("type_vars") or ():
                    _insert("type_var", tv)
                inputs = c.get("inputs")
                if isinstance(inputs, dict):
                    for p in inputs.values():
                        if isinstance(p, dict):
                            role = p.get("port_role") or p.get("role")
                            if role:
                                _insert("role_carrier", str(role).strip().lower())

    for k, val in merged_role_semantics.items():
        _insert("role_semantics", k, json.dumps(val))
    for k, val in merged_capabilities.items():
        _insert("capability", k, json.dumps(val))
    for k, val in merged_carrier_roles.items():
        _insert("carrier_role", k, json.dumps(val))
    for k, val in merged_state_role_ancestors.items():
        _insert("state_role_ancestor", k, json.dumps(val))
    for k, val in merged_state_role_markers.items():
        _insert("state_role_marker", k, json.dumps(val))

    conn.commit()
    return written


def _persist_type_hierarchies(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    """Persists per-tree ``types`` hierarchies into the ``types`` table.

    Properly extracts multi-parent DAG edges when ``types[child]`` is a dict
    with ``parent`` or ``parents`` (never stringifying the dict itself).
    """
    cur = conn.cursor()
    written = 0
    core_domains = {"", "python_core", "core", "base", "builtin", "builtins"}
    for path in tree_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        dom = str(data.get("domain") or path.stem).strip().lower()
        entries: List[Tuple[str, str]] = []
        raw_types = data.get("types")
        if isinstance(raw_types, list):
            for item in raw_types:
                if isinstance(item, str):
                    entries.append((item, ""))
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("type_name")
                    if not name:
                        continue
                    k_low = str(name).strip().lower()
                    if "default_value" in item:
                        if dom:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", f"{dom}:{k_low}", json.dumps(item["default_value"])),
                            )
                        if dom in core_domains:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", k_low, json.dumps(item["default_value"])),
                            )
                    elif "pointed_default" in item:
                        if dom:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", f"{dom}:{k_low}", json.dumps(item["pointed_default"])),
                            )
                        if dom in core_domains:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", k_low, json.dumps(item["pointed_default"])),
                            )
                    parents = item.get("parents")
                    if isinstance(parents, list) and parents:
                        for p in parents:
                            if p:
                                entries.append((str(name), str(p)))
                    else:
                        p = item.get("parent") or item.get("parent_type") or ""
                        entries.append((str(name), str(p)))
        elif isinstance(raw_types, dict):
            for name, meta in raw_types.items():
                if isinstance(meta, dict):
                    k_low = str(name).strip().lower()
                    if "default_value" in meta:
                        if dom:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", f"{dom}:{k_low}", json.dumps(meta["default_value"])),
                            )
                        if dom in core_domains:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", k_low, json.dumps(meta["default_value"])),
                            )
                    elif "pointed_default" in meta:
                        if dom:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", f"{dom}:{k_low}", json.dumps(meta["pointed_default"])),
                            )
                        if dom in core_domains:
                            cur.execute(
                                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                                ("type_default", k_low, json.dumps(meta["pointed_default"])),
                            )
                    parents = meta.get("parents")
                    if isinstance(parents, list) and parents:
                        for p in parents:
                            if p:
                                entries.append((str(name), str(p)))
                    else:
                        p = meta.get("parent") or meta.get("parent_type") or ""
                        entries.append((str(name), str(p)))
                elif isinstance(meta, list):
                    for p in meta:
                        if p:
                            entries.append((str(name), str(p)))
                else:
                    entries.append((str(name), str(meta) if meta else ""))

        for name, parent in entries:
            cur.execute(
                "INSERT OR REPLACE INTO types (type_name, parent_type) VALUES (?, ?)",
                (name, parent),
            )
            written += 1
    conn.commit()
    return written


def _persist_typestates(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    """Persists per-tree ``typestates`` declarations into the ``typestates``
    table (state hierarchy + declared carrier_type + verifiable properties).

    The loader (LatticeOrchestrator.load_from_database) reads this table to
    restore the typestate vocabulary in SQLite mode.
    """
    cur = conn.cursor()
    written = 0
    for path in tree_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        ts_info = data.get("typestates")
        states: List[Any] = []
        if isinstance(ts_info, dict):
            states = ts_info.get("states", []) or []
        elif isinstance(ts_info, list):
            states = ts_info
        for s in states:
            if isinstance(s, str):
                row = (s, None, None, None)
            elif isinstance(s, dict) and s.get("name"):
                props = s.get("properties")
                row = (
                    str(s["name"]),
                    s.get("parent_state"),
                    s.get("carrier_type"),
                    json.dumps(props, ensure_ascii=False) if props else None,
                )
            else:
                continue
            cur.execute(
                "INSERT INTO typestates (state_name, parent_state, carrier_type, properties) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(state_name) DO UPDATE SET "
                "parent_state=excluded.parent_state, carrier_type=excluded.carrier_type, "
                "properties=excluded.properties",
                row,
            )
            written += 1
    conn.commit()
    return written


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #

def compile_database(
    output_db: str = DB_PATH,
    domain_filter: Optional[List[str]] = None,
    trees_dir: Optional[Union[str, Path]] = None,
) -> int:
    t_dir = Path(trees_dir) if trees_dir else (PROJECT_ROOT / "trees")
    tree_files = _resolve_tree_files(t_dir, domain_filter)

    if not tree_files:
        print(f"[compile_trees] No tree files found in {t_dir}"
              + (f" for domains={domain_filter}" if domain_filter else ""))
        return 0

    Path(output_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_db)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        written = 0
        skipped = 0
        signature_only = 0
        signature_only_by_domain: dict = {}
        warnings: List[str] = []

        types_written = _persist_type_hierarchies(conn, tree_files)
        typestates_written = _persist_typestates(conn, tree_files)
        aliases_written = _persist_aliases(conn, tree_files)
        readers_written = _persist_artifact_readers(conn, tree_files)
        structural_written = _persist_structural_metadata(conn, tree_files)
        if types_written:
            print(f"[compile_trees] Persisted {types_written} type-hierarchy entrie(s) into the types table.")
        if typestates_written:
            print(f"[compile_trees] Persisted {typestates_written} typestate declaration(s) into the typestates table.")
        if aliases_written:
            print(f"[compile_trees] Persisted {aliases_written} alias declaration(s) into the aliases table.")
        if readers_written:
            print(f"[compile_trees] Persisted {readers_written} artifact reader(s) into the artifact_readers table.")
        if structural_written:
            print(f"[compile_trees] Persisted {structural_written} structural metadata entrie(s) into the structural_metadata table.")

        raw_trees: List[Tuple[Path, dict, List[Tuple[str, dict]]]] = []
        all_cells: List[Tuple[str, dict]] = []

        for path in tree_files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    tree_data = json.load(fh)
                nodes = list(_iter_tree_nodes(path))
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[compile_trees] SKIP {path.name}: {exc}", file=sys.stderr)
                continue
            raw_trees.append((path, tree_data if isinstance(tree_data, dict) else {}, nodes))
            for domain, node in nodes:
                if "cell_id" in node and "stage" in node:
                    all_cells.append((domain, node))

        dynamic_edges_map = synthesize_dynamic_edges(all_cells)
        total_synthesized_edges = sum(len(e) for e in dynamic_edges_map.values())
        print(f"[compile_trees] Synthesized {total_synthesized_edges} dynamic structural edges across {len(all_cells)} cells.")

        for path, tree_data, nodes in raw_trees:
            supported_formats = tree_data.get("supported_formats", [])
            for domain, node in nodes:
                if "cell_id" not in node or "stage" not in node:
                    print(f"[compile_trees] SKIP node in {path.name}: missing cell_id/stage",
                          file=sys.stderr)
                    skipped += 1
                    continue

                cell_edges = dynamic_edges_map.get(node["cell_id"], [])
                row = extract_node_data(node, domain, dynamic_edges=cell_edges)
                if len(row) != len(_COLUMNS):
                    raise RuntimeError(
                        f"extract_node_data returned {len(row)} fields, "
                        f"schema expects {len(_COLUMNS)}."
                    )

                template = row[10]                      # `code` column
                if not template or not template.strip():
                    signature_only += 1
                    signature_only_by_domain[domain] = (
                        signature_only_by_domain.get(domain, 0) + 1
                    )
                else:
                    aliases = _module_aliases(node.get("dependencies", []) or [])
                    ok, err = _validate_template(
                        template, row[0], node.get("node_role", ""), aliases, supported_formats=supported_formats,
                    )
                    if not ok:
                        warnings.append(f"{path.name}::{row[0]}: {err}")

                cur.execute(INSERT_SQL, row)
                written += 1

        conn.commit()

        print(f"[compile_trees] Compiled {written} node(s) from "
              f"{len(tree_files)} file(s) into {output_db}"
              + (f" ({skipped} skipped)" if skipped else ""))

        if signature_only:
            parts = ", ".join(f"{d}={n}" for d, n in sorted(signature_only_by_domain.items()))
            print(f"[compile_trees] {signature_only} signature-only cell(s) "
                  f"(body synthesized on demand): {parts}")

        if warnings:
            print(f"[compile_trees] {len(warnings)} template warning(s):",
                  file=sys.stderr)
            for w in warnings[:40]:
                print(f"  WARN {w}", file=sys.stderr)
            if len(warnings) > 40:
                print(f"  ... and {len(warnings) - 40} more", file=sys.stderr)

        return written
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="STRATA Tree Compiler")
    parser.add_argument("--output", type=str, default=DB_PATH)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--dir", "--trees-dir", dest="trees_dir", type=str, default=None, help="Directory containing domain tree JSON files")
    args = parser.parse_args()
    compile_database(output_db=args.output, domain_filter=args.domains, trees_dir=args.trees_dir)


if __name__ == "__main__":
    main()
