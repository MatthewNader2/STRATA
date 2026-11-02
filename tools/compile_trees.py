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
    source_provenance     TEXT
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
    """Synthesizes dynamic transition edges between compatible cells based on structural type unification."""
    edges_by_cell: Dict[str, List[dict]] = {node["cell_id"]: [] for _, node in all_cells if isinstance(node, dict) and "cell_id" in node}
    consumers = []
    for domain, c in all_cells:
        if not isinstance(c, dict) or "cell_id" not in c:
            continue
        c_id = c["cell_id"]
        c_domain = c.get("domain_name") or domain
        ins = c.get("inputs", {})
        if not ins or not isinstance(ins, dict):
            continue
        prim_in_name = c.get("primary_in")
        p_in = ins.get(prim_in_name) if prim_in_name and prim_in_name in ins else next(iter(ins.values()))
        if isinstance(p_in, dict):
            c_copy = dict(c)
            c_copy["domain"] = c_domain
            consumers.append((c_id, c_copy, p_in))

    for domain, c in all_cells:
        if not isinstance(c, dict) or "cell_id" not in c:
            continue
        c_id = c["cell_id"]
        c_domain = c.get("domain_name") or domain
        outs = c.get("outputs", {})
        if not outs or not isinstance(outs, dict):
            continue
        prim_out_name = c.get("primary_out")
        p_out = outs.get(prim_out_name) if prim_out_name and prim_out_name in outs else next(iter(outs.values()))
        if not isinstance(p_out, dict):
            continue

        c_copy = dict(c)
        c_copy["domain"] = c_domain
        c_edges = []
        for target_id, target_cell, p_in in consumers:
            if target_id == c_id:
                continue
            edge = compute_structural_edge(c_copy, target_cell, p_out, p_in)
            if edge:
                c_edges.append(edge)
                
        c_edges.sort(key=lambda x: x["affinity_score"], reverse=True)
        edges_by_cell[c_id] = c_edges

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
    """Create the ``nodes`` table if missing; add any new columns.

    Note: this does NOT migrate an older ``morphisms`` table. If you compiled
    with a previous version of this script, delete the DB and recompile:
        rm -f trees/lattice.db*
    """
    cur = conn.cursor()
    cur.execute(_CREATE_TABLE_SQL)
    cur.execute(_CREATE_TYPES_TABLE_SQL)
    cur.execute(_CREATE_TYPESTATES_TABLE_SQL)
    cur.execute(_CREATE_ALIASES_TABLE_SQL)
    cur.execute(_CREATE_STRUCTURAL_METADATA_TABLE_SQL)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(nodes)")}
    for col in _COLUMNS:
        if col not in existing:
            col_type = "INTEGER" if col in _INT_COLUMNS else "TEXT"
            cur.execute(f"ALTER TABLE nodes ADD COLUMN {col} {col_type}")
    conn.commit()


_CREATE_TYPES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS types (
    type_name   TEXT PRIMARY KEY,
    parent_type TEXT
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

_CREATE_STRUCTURAL_METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS structural_metadata (
    category TEXT,
    item     TEXT,
    extra    TEXT,
    PRIMARY KEY (category, item)
);
"""


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


def _persist_structural_metadata(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
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

        def _insert(cat: str, it: str, ex: Optional[str] = None):
            nonlocal written
            cur.execute(
                "INSERT OR REPLACE INTO structural_metadata (category, item, extra) VALUES (?, ?, ?)",
                (cat, str(it), ex)
            )
            written += 1

        mappings = [
            ("estimator_verbs", "estimator_verb"),
            ("preposition_triggers", "preposition_trigger"),
            ("prompt_filler_tokens", "prompt_filler_token"),
            ("sentence_connectives", "sentence_connective"),
            ("coordinating_conjunctions", "coordinating_conjunction"),
            ("sequencing_connectives", "sequencing_connective"),
            ("filter_context_tokens", "filter_context_token"),
            ("column_projection_tokens", "column_projection_token"),
            ("dest_port_tokens", "dest_port_token"),
            ("data_bearing_roles", "data_bearing_role"),
            ("egress_intent_tokens", "egress_intent_token"),
            ("stopwords", "stopword"),
            ("abstract_carriers", "abstract_carrier"),
        ]
        for key, cat in mappings:
            items = data.get(key)
            if isinstance(items, list):
                for item in items:
                    _insert(cat, item)

        vs = data.get("verification_semantics")
        if isinstance(vs, dict):
            for k, val in vs.items():
                _insert("verification_semantics", k, json.dumps(val))
                if k in ("estimator_verbs", "target_verbs") and isinstance(val, list):
                    for v in val:
                        _insert("estimator_verb", v)
                elif k == "preposition_triggers" and isinstance(val, list):
                    for p in val:
                        _insert("preposition_trigger", p)
                elif k in ("destination_roles", "dest_port_tokens") and isinstance(val, list):
                    for d in val:
                        _insert("dest_port_token", d)
                elif k in ("terminal_states",) and isinstance(val, list):
                    for s in val:
                        _insert("terminal_state", s)
                elif k in ("materialization_states",) and isinstance(val, list):
                    for s in val:
                        _insert("materialization_state", s)
                elif k in ("bridge_roles",) and isinstance(val, list):
                    for r in val:
                        _insert("role_carrier", r)

        ap = data.get("asset_placeholders")
        if isinstance(ap, dict):
            for k, val in ap.items():
                _insert("asset_placeholder", k, str(val))
        oap = data.get("output_asset_placeholders")
        if isinstance(oap, dict):
            for k, val in oap.items():
                _insert("output_asset_placeholder", k, str(val))

    conn.commit()
    return written


def _persist_type_hierarchies(conn: sqlite3.Connection, tree_files: List[Path]) -> int:
    """Persists per-tree ``types`` hierarchies into the ``types`` table.

    The loader (LatticeOrchestrator.load_from_database) reads this table to
    restore the type poset in SQLite mode; without it, subtype relations
    declared in tree JSONs were silently lost on compilation.
    Accepts both list-form entries ("Name" or {"name": ..., "parent": ...})
    and dict-mappings {name: parent}.
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

        entries: List[Tuple[str, Optional[str]]] = []
        raw_types = data.get("types")
        if isinstance(raw_types, list):
            for item in raw_types:
                if isinstance(item, str):
                    entries.append((item, None))
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("type_name")
                    if name:
                        entries.append((str(name), item.get("parent") or item.get("parent_type")))
        elif isinstance(raw_types, dict):
            for name, parent in raw_types.items():
                entries.append((str(name), str(parent) if parent else None))

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
        structural_written = _persist_structural_metadata(conn, tree_files)
        if types_written:
            print(f"[compile_trees] Persisted {types_written} type-hierarchy entrie(s) into the types table.")
        if typestates_written:
            print(f"[compile_trees] Persisted {typestates_written} typestate declaration(s) into the typestates table.")
        if aliases_written:
            print(f"[compile_trees] Persisted {aliases_written} alias declaration(s) into the aliases table.")
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
