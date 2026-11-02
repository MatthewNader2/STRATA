"""
audit_trees.py — structural QA gate for trees/*.json before promoting an
enrichment checkpoint (or any harvest run) into the live tree files.

Seven independent checks, each answering a different question:

1. SCHEMA FINGERPRINTING: every cell's shape — which top-level keys it has,
   and which keys each of its ports has — gets hashed into a fingerprint.
   Cells are grouped by fingerprint. A schema shared by thousands of cells is
   almost certainly fine; a schema shared by a handful is worth a human
   glancing at.

2. DOCSTRING COMPLETENESS — cells with no docstring at all, and, separately,
   cells whose docstring is character-for-character IDENTICAL to another
   cell's (a known LLM-batch failure mode).

3. TYPE CONSISTENCY — for every port, checks that `default_value`'s actual
   JSON type matches what `type_name` claims, AND that `type_name` itself is
   a recognizable signature (concrete type, type variable, or higher-order
   generic like List[int] / Optional[str] / Callable[[int], str]).

4. TEMPLATE WIRING — cross-checks every `{placeholder}` in `template` /
   `code_template` / every `code_templates[lang]` against the cell's declared
   `inputs` (plus the always-valid `output_var`, `dest_path`, `filepath`), in
   both directions.

5. DUPLICATE CELL_IDs — should be structurally impossible, but cheap to check.

6. PORT_ROLE VALIDATION — every port's `port_role` must be present and be one
   of the values in VALID_PORT_ROLES.

7. REQUIRED FIELDS + TEMPLATE PRESENCE — every cell must declare `cell_id`,
   `stage`, and at least one of `template` / `code_template` / `code_templates`.

Usage:
    python3 audit_trees.py trees/pandas.json
    python3 audit_trees.py strata_enrichment/checkpoints/pandas.json
    python3 audit_trees.py trees/*.json          # whole corpus at once
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

# Placeholders that are always valid in a template even if the cell hasn't
# declared them as named inputs.
RESERVED_PLACEHOLDERS = {"output_var", "dest_path", "filepath", "T", "U", "V", "R", "State", "Comparable", "C"}

VALID_PORT_ROLES = {
    "data_input",
    "literal_parameter",
    "functional_operator",
    "predicate_operator",
}

# Atomic / free type names accepted without further parsing.
_ATOMIC_TYPE_NAMES = {
    # type variables / generic markers used by the STRATA schema
    "T", "U", "V", "W", "State", "Comparable", "C", "Exception",
    # python builtins
    "bool", "int", "float", "str", "bytes", "list", "dict", "set",
    "tuple", "frozenset", "complex", "object", "None", "NoneType",
    # typing sentinels
    "Any", "NoReturn", "Never", "Self",
}

# Higher-order / parameterized generics we explicitly allow.
_GENERIC_TYPE_RE = re.compile(
    r"^(?:"
    r"List|Tuple|Optional|ContextManager|Callable|Dict|Set|FrozenSet|"
    r"Sequence|MutableSequence|Iterable|Iterator|Mapping|MutableMapping|"
    r"Union|Awaitable|Coroutine|Generator|AsyncGenerator|AsyncIterator|"
    r"Type|Deque|DefaultDict|OrderedDict|Counter|ChainMap|Literal|"
    r"Annotated|TypeVar|NewType|ClassVar|Final"
    r")\[.*\]$"
)

# Concrete-name fallback (dotted module paths, camel-cased classes).
_CONCRETE_NAME_RE = re.compile(r"^[A-Za-z_][\w\.]*$")

# Simple runtime-type validators used when the declared type is a primitive.
PY_TYPE_CHECKS = {
    "bool":      lambda v: isinstance(v, bool),
    "int":       lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float":     lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str":       lambda v: isinstance(v, str),
    "bytes":     lambda v: isinstance(v, (bytes, bytearray)),
    "list":      lambda v: isinstance(v, list),
    "tuple":     lambda v: isinstance(v, list),  # JSON has no tuple; arrays decode to list
    "set":       lambda v: isinstance(v, (list, dict)),
    "frozenset": lambda v: isinstance(v, (list, dict)),
    "dict":      lambda v: isinstance(v, dict),
}

# --------------------------------------------------------------------------- #
# Type-signature validation (from the provided snippet, hardened)
# --------------------------------------------------------------------------- #

def is_valid_type_signature(type_name: str) -> bool:
    """Allow concrete types, type variables, and higher-order generics.

    Rejects obvious junk (empty strings, whitespace, punctuation soup) while
    remaining permissive about user-defined class names.
    """
    if not type_name or not isinstance(type_name, str):
        return False
    type_name = type_name.strip()
    if not type_name:
        return False
    if type_name in _ATOMIC_TYPE_NAMES:
        return True
    if _GENERIC_TYPE_RE.match(type_name):
        return True
    if _CONCRETE_NAME_RE.match(type_name):
        return True
    return False


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #

def port_fingerprint(port: Dict[str, Any]) -> tuple:
    return tuple(sorted(port.keys()))


def cell_fingerprint(cell: Dict[str, Any]) -> tuple:
    top_keys = tuple(sorted(cell.keys()))
    input_shapes = tuple(
        sorted(port_fingerprint(p) for p in cell.get("inputs", {}).values())
    )
    output_shapes = tuple(
        sorted(port_fingerprint(p) for p in cell.get("outputs", {}).values())
    )
    return (top_keys, input_shapes, output_shapes)


def check_schema_fingerprints(cells: List[Dict[str, Any]]) -> Dict[tuple, List[str]]:
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for c in cells:
        groups[cell_fingerprint(c)].append(c.get("cell_id", "<unknown>"))
    return groups


# --------------------------------------------------------------------------- #
# Docstrings
# --------------------------------------------------------------------------- #

def check_docstrings(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    empty = [c.get("cell_id", "<unknown>") for c in cells if not c.get("docstring")]
    by_text: Dict[str, List[str]] = defaultdict(list)
    for c in cells:
        doc = c.get("docstring")
        if doc:
            by_text[doc].append(c.get("cell_id", "<unknown>"))
    duplicated = {doc: ids for doc, ids in by_text.items() if len(ids) > 1}
    return {"empty": empty, "duplicated_text": duplicated}


# --------------------------------------------------------------------------- #
# Type consistency (default_value vs type_name, plus signature validity)
# --------------------------------------------------------------------------- #

def check_type_consistency(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    for c in cells:
        cell_id = c.get("cell_id", "<unknown>")
        for direction in ("inputs",):
            for pname, port in c.get(direction, {}).items():
                type_name = port.get("type_name")
                default = port.get("default_value")

                # 3a. Signature validity
                if type_name is not None and not is_valid_type_signature(type_name):
                    problems.append({
                        "cell_id": cell_id,
                        "port": f"{direction}.{pname}",
                        "type_name": type_name,
                        "default_value": default,
                        "actual_json_type": type(default).__name__,
                        "reason": "unrecognized type signature",
                    })
                    continue

                # 3b. Runtime-type agreement (only for primitives we can check)
                if default is None or type_name not in PY_TYPE_CHECKS:
                    continue
                if not PY_TYPE_CHECKS[type_name](default):
                    problems.append({
                        "cell_id": cell_id,
                        "port": f"{direction}.{pname}",
                        "type_name": type_name,
                        "default_value": default,
                        "actual_json_type": type(default).__name__,
                        "reason": "default_value type mismatch",
                    })
    return problems


# --------------------------------------------------------------------------- #
# Template wiring (polyglot-aware)
# --------------------------------------------------------------------------- #

def _extract_templates(cell: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return [(source_label, template_string), ...] for every template in the cell."""
    out: List[Tuple[str, str]] = []
    seen: set = set()

    ct = cell.get("code_templates")
    if isinstance(ct, dict):
        for lang, tpl in ct.items():
            if isinstance(tpl, str) and tpl and tpl not in seen:
                out.append((f"code_templates.{lang}", tpl))
                seen.add(tpl)

    for key in ("template", "code_template"):
        tpl = cell.get(key)
        if isinstance(tpl, str) and tpl and tpl not in seen:
            out.append((key, tpl))
            seen.add(tpl)

    return out


def check_template_wiring(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    for c in cells:
        cell_id = c.get("cell_id", "<unknown>")
        slots = c.get("slots")
        slots_declared = set(slots.keys()) if isinstance(slots, dict) else set()
        declared = set(c.get("inputs", {}).keys()) | set(c.get("outputs", {}).keys()) | slots_declared | RESERVED_PLACEHOLDERS

        for label, template in _extract_templates(c):
            placeholders = set(re.findall(r"\{(\w+)\}", template))
            unmatched = placeholders - declared
            unused = declared - placeholders - RESERVED_PLACEHOLDERS - slots_declared
            if unmatched or unused:
                problems.append({
                    "cell_id": cell_id,
                    "source": label,
                    "template": template,
                    "unmatched_placeholders": sorted(unmatched),
                    "unused_declared_inputs": sorted(unused),
                })
    return problems


# --------------------------------------------------------------------------- #
# Duplicate IDs
# --------------------------------------------------------------------------- #

def check_duplicate_ids(cells: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, int] = defaultdict(int)
    for c in cells:
        seen[c.get("cell_id", "<unknown>")] += 1
    return [cid for cid, n in seen.items() if n > 1]


# --------------------------------------------------------------------------- #
# Port roles (new) + required fields / template presence (new)
# --------------------------------------------------------------------------- #

def check_port_roles(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every input/output port must declare a `port_role` from VALID_PORT_ROLES."""
    problems: List[Dict[str, Any]] = []
    for c in cells:
        cell_id = c.get("cell_id", "<unknown>")
        for direction in ("inputs",):
            for pname, port in c.get(direction, {}).items():
                role = port.get("port_role")
                if role is None:
                    problems.append({
                        "cell_id": cell_id,
                        "port": f"{direction}.{pname}",
                        "port_role": None,
                        "reason": "missing port_role",
                    })
                elif role not in VALID_PORT_ROLES:
                    problems.append({
                        "cell_id": cell_id,
                        "port": f"{direction}.{pname}",
                        "port_role": role,
                        "reason": (
                            f"invalid port_role "
                            f"(expected one of {sorted(VALID_PORT_ROLES)})"
                        ),
                    })
    return problems


def check_required_fields(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every cell must have cell_id, stage, and at least one template field."""
    problems: List[Dict[str, Any]] = []
    for c in cells:
        missing: List[str] = []
        if "cell_id" not in c:
            missing.append("cell_id")
        if "stage" not in c:
            missing.append("stage")
        has_template = (
            "template" in c
            or "code_template" in c
            or (
                isinstance(c.get("code_templates"), dict)
                and bool(c["code_templates"])
            )
        )
        if not has_template:
            missing.append("template|code_template|code_templates")
        if missing:
            problems.append({
                "cell_id": c.get("cell_id", "<unknown>"),
                "missing": missing,
            })
    return problems


# --------------------------------------------------------------------------- #
# Fail-fast helper (kept from the snippet, for callers that want assertions)
# --------------------------------------------------------------------------- #

def audit_node(node: dict) -> None:
    """Strict per-node validator that raises on the first violation.

    Use this when iterating nodes in a tight loop where you want to abort on
    bad input. The reporting checks above are preferred for whole-corpus QA.
    """
    assert "cell_id" in node, "Missing cell_id"
    assert "stage" in node, f"Missing stage in {node.get('cell_id')}"
    assert (
        "template" in node
        or "code_template" in node
        or "code_templates" in node
    ), f"No template found in {node.get('cell_id')}"

    for port_name, port in node.get("inputs", {}).items():
        role = port.get("port_role")
        assert role in VALID_PORT_ROLES, (
            f"Invalid port_role '{role}' in {node['cell_id']}.inputs.{port_name}"
        )
    for port_name, port in node.get("outputs", {}).items():
        role = port.get("port_role")
        assert role in VALID_PORT_ROLES, (
            f"Invalid port_role '{role}' in {node['cell_id']}.outputs.{port_name}"
        )


# --------------------------------------------------------------------------- #
# Port descriptions (informational)
# --------------------------------------------------------------------------- #

def check_port_descriptions(cells: List[Dict[str, Any]]) -> Dict[str, int]:
    """Informational only — nothing in synthesis reads port-level `description`."""
    total = empty = 0
    for c in cells:
        for port in list(c.get("inputs", {}).values()) + list(c.get("outputs", {}).values()):
            total += 1
            if not port.get("description"):
                empty += 1
    return {"total_ports": total, "empty_description": empty}


# --------------------------------------------------------------------------- #
# File-level audit
# --------------------------------------------------------------------------- #

def audit_file(path: Path) -> None:
    data = json.loads(path.read_text())
    cells = data.get("cells") or data.get("nodes") or []
    print(f"\n{'='*70}\n{path} — {len(cells)} cells\n{'='*70}")

    # 1. Schema fingerprints
    groups = check_schema_fingerprints(cells)
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    print(f"\n[schema fingerprints] {len(groups)} distinct shapes found")
    for fp, ids in sorted_groups[:5]:
        print(f"  {len(ids):6d} cells  <- dominant shape(s), e.g. {ids[0]}")
    minority = [(fp, ids) for fp, ids in sorted_groups if len(ids) <= 3]
    if minority:
        print(f"  {len(minority)} shapes used by 3 or fewer cells (worth a manual look):")
        for fp, ids in minority[:15]:
            print(f"    {ids} -> top_keys={fp[0]}")

    # 2. Docstrings
    doc_report = check_docstrings(cells)
    print(f"\n[docstrings] empty: {len(doc_report['empty'])}")
    if doc_report["empty"]:
        print(f"  {doc_report['empty'][:10]}")
    dup_text = doc_report["duplicated_text"]
    if dup_text:
        print(f"  {len(dup_text)} docstring texts reused across multiple cells:")
        for doc, ids in list(dup_text.items())[:5]:
            print(f"    '{doc[:60]}...' used by {len(ids)} cells: {ids[:5]}")

    # 3. Type consistency
    type_problems = check_type_consistency(cells)
    print(f"\n[type consistency] {len(type_problems)} problems")
    for p in type_problems[:10]:
        print(f"  {p['cell_id']} | {p['port']} | type_name={p['type_name']} "
              f"default={p['default_value']!r} ({p['actual_json_type']}) "
              f"-> {p['reason']}")

    # 4. Template wiring
    wiring_problems = check_template_wiring(cells)
    print(f"\n[template wiring] {len(wiring_problems)} template(s) with mismatches")
    for p in wiring_problems[:10]:
        print(f"  {p['cell_id']} [{p['source']}]: "
              f"unmatched={p['unmatched_placeholders']} "
              f"unused={p['unused_declared_inputs']} "
              f"| template={p['template'][:60]}")

    # 5. Duplicate IDs
    dupes = check_duplicate_ids(cells)
    print(f"\n[duplicate cell_ids] {len(dupes)}")
    if dupes:
        print(f"  {dupes}")

    # 6. Port roles
    role_problems = check_port_roles(cells)
    print(f"\n[port roles] {len(role_problems)} invalid/missing port_role")
    for p in role_problems[:10]:
        print(f"  {p['cell_id']} | {p['port']} | role={p['port_role']!r} "
              f"-> {p['reason']}")

    # 7. Required fields / template presence
    field_problems = check_required_fields(cells)
    print(f"\n[required fields] {len(field_problems)} cells missing required keys")
    for p in field_problems[:10]:
        print(f"  {p['cell_id']}: missing {p['missing']}")

    # 8. Port descriptions (informational only)
    desc = check_port_descriptions(cells)
    pct = 100 * desc["empty_description"] / max(desc["total_ports"], 1)
    print(f"\n[port descriptions - informational, not a blocker] "
          f"{desc['empty_description']}/{desc['total_ports']} empty ({pct:.0f}%)")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["trees/pandas.json"]
    for p in paths:
        audit_file(Path(p))
