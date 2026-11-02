"""
src/template_wiring.py

Core shared functions for maintaining and repairing the STRATA template wiring invariant:
1. Every {placeholder} in `code_template` must have a matching key in `inputs`
   (except `output_var`).
2. Every declared input in `inputs` must be referenced in `code_template`.

Zero regular expressions.
"""

from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Set


def clean_malformed_template_braces(template: str) -> str:
    """Cleans malformed double braces around identifier placeholders or trailing artifacts in templates without regex."""
    if not template or "{" not in template:
        return template
    cleaned = template
    for artifact in (", {/*}", "{/*},", "{/*}", ", {\\*}", "{\\*},", "{\\*}"):
        cleaned = cleaned.replace(artifact, "")

    res = []
    i = 0
    n = len(cleaned)
    while i < n:
        if cleaned[i:i+2] == "{{" and i + 2 < n:
            end = cleaned.find("}}", i + 2)
            if end != -1:
                inner = cleaned[i + 2:end].strip()
                if inner.isidentifier():
                    res.append(f"{{{inner}}}")
                    i = end + 2
                    continue
        res.append(cleaned[i])
        i += 1
    return "".join(res)


def extract_placeholders(template: str) -> List[str]:
    """Extracts all {var} placeholder names from a template string without regex."""
    placeholders: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == '{':
            j = template.find('}', i + 1)
            if j != -1:
                inner = template[i + 1:j]
                if inner.isidentifier():
                    placeholders.append(inner)
                i = j + 1
                continue
        i += 1
    return placeholders


# --------------------------------------------------------------------------- #
# Type resolution — DECLARED schemas only (Solution 2)
# --------------------------------------------------------------------------- #
# The engine never guesses a port type from a parameter name. Resolution
# order for a template placeholder:
#   1. The cell's DECLARED input port type (CellSchema.inputs / PortSchema.
#      type_name) — the typed-lattice source of truth.
#   2. Port-name hints DECLARED in tree JSON (`port_name_type_hints` /
#      `port_name_morphology_hints`), harvested by TypeRegistry. Domain
#      authors extend typing by adding data to their tree — never by
#      patching engine code.
#   3. The structural default "any".
# Zero hardcoded name dictionaries live in the engine.

_PREFIXES: tuple = ("input_", "in_", "arg_", "param_")
_SUFFIXES: tuple = ("_input", "_in", "_arg", "_param", "_value", "_val")


def _canonicalize_param_name(param_name: str) -> str:
    name = str(param_name).strip().lower()
    for pre in _PREFIXES:
        if name.startswith(pre) and len(name) > len(pre):
            name = name[len(pre):]
            break
    for suf in _SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            name = name[: -len(suf)]
            break
    return name


def _registry():
    try:
        from .lattice import TypeRegistry
    except (ImportError, ValueError):
        try:
            from lattice import TypeRegistry
        except Exception:
            return None
    try:
        return TypeRegistry.get_instance()
    except Exception:
        return None


def resolve_placeholder_type(cell: Any, placeholder: str) -> str:
    """
    Returns the DECLARED type of a template placeholder port.
    Reads the port schema off the cell itself; falls back to tree-declared
    name hints, then to the structural default "any". No lexical guessing.
    """
    inputs = getattr(cell, "inputs", None)
    if isinstance(cell, dict):
        inputs = cell.get("inputs", inputs)
    if isinstance(inputs, dict):
        port = inputs.get(placeholder)
        if port is not None:
            t_name = (
                port.get("type_name") if isinstance(port, dict)
                else getattr(port, "type_name", None)
            )
            if t_name:
                return str(t_name)
    return infer_port_type(placeholder)


def infer_port_type(param_name: str, domain: str = "generic") -> str:
    """
    Resolves a port name to its declared type_name using ONLY declared data:
    exact-name hints and affix morphology hints harvested from trees via the
    TypeRegistry. Unknown names resolve to "any" — the engine contains no
    name-to-type vocabulary of its own.
    """
    if not param_name:
        return "any"
    reg = _registry()
    if reg is not None:
        try:
            declared = reg.lookup_port_name_type(str(param_name))
            if declared:
                return declared
        except Exception:
            pass
        # Also try the canonicalized name (strip input_/arg_/_value etc.) so
        # tree hints can be written against the semantic port name.
        canonical = _canonicalize_param_name(str(param_name))
        if canonical and canonical != str(param_name).strip().lower():
            try:
                declared = reg.lookup_port_name_type(canonical)
                if declared:
                    return declared
            except Exception:
                pass
    return "any"


def transform_call_ast_with_flag(code_snippet: str, mod_alias: str, flag_attr: str) -> str:
    """Uses AST Node Transformation to inject a flag attribute into a function call snippet dynamically."""
    if not code_snippet:
        return f"{{output_var}} = {mod_alias}.{flag_attr}()"

    cleaned_snippet = clean_malformed_template_braces(code_snippet)

    placeholders = list(dict.fromkeys(extract_placeholders(cleaned_snippet)))
    safe_code = cleaned_snippet
    ph_map = {}
    for i, ph in enumerate(placeholders):
        safe_id = f"__ph_{i}_{ph}__"
        ph_map[safe_id] = ph
        safe_code = safe_code.replace(f"{{{ph}}}", safe_id)

    try:
        tree = ast.parse(safe_code)

        class FlagASTReplacer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                flag_node = ast.Attribute(
                    value=ast.Name(id=mod_alias, ctx=ast.Load()),
                    attr=flag_attr,
                    ctx=ast.Load(),
                )
                # Prevent duplicate injection if flag is already present
                for arg in node.args:
                    if isinstance(arg, ast.Attribute) and arg.attr == flag_attr:
                        return node
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Attribute) and kw.value.attr == flag_attr:
                        return node

                # Safely append flag without overwriting existing arguments or keywords
                node.args.append(flag_node)
                return node

        transformed = FlagASTReplacer().visit(tree)
        ast.fix_missing_locations(transformed)
        unparsed = ast.unparse(transformed)

        for safe_id, ph in ph_map.items():
            unparsed = unparsed.replace(safe_id, f"{{{ph}}}")

        return clean_malformed_template_braces(unparsed)
    except Exception:
        return cleaned_snippet


def wire_default_flag(cell: Any, code: str, mod_alias: str) -> str:
    """Injects default flag attributes via AST without text surgery hacks."""
    slots = getattr(cell, "slots", {}) if not isinstance(cell, dict) else cell.get("slots", {})
    tree_meta = getattr(cell, "tree_meta", {}) if not isinstance(cell, dict) else cell.get("tree_meta", {})
    flag_attr = (slots or {}).get("default_flag_attr") or (tree_meta or {}).get("default_flag_attr")
    if not flag_attr:
        return code
    return transform_call_ast_with_flag(code, mod_alias, str(flag_attr))


def _port_is_preservable(spec: Any) -> bool:
    """
    A port is preservable (i.e. must NOT be pruned even if unreferenced by the
    template) when it is declared optional, config-role, or variadic. These
    ports are supplied out-of-band (via **kwargs or configuration dicts) and
    deleting them would destroy the port contract.
    """
    if not isinstance(spec, dict):
        return False
    if not spec.get("required", True):
        return True
    if spec.get("kind") == "config" or spec.get("role") == "config":
        return True
    if spec.get("variadic") in ("kwargs", "kw", "**kwargs"):
        return True
    if spec.get("supplied_out_of_band") is True:
        return True
    return False


def repair_wiring_invariant(cell: Dict[str, Any], domain: str = "generic") -> bool:
    """Enforces the template wiring invariant on a cell without regex."""
    modified = False

    tmpl = clean_malformed_template_braces(cell.get("code_template", ""))
    if tmpl != cell.get("code_template", ""):
        cell["code_template"] = tmpl
        modified = True

    # Normalize inputs to dict format if needed
    raw_inputs = cell.get("inputs")
    if isinstance(raw_inputs, list):
        inputs_dict = {}
        for item in raw_inputs:
            if isinstance(item, dict):
                pname = item.get("name", "input_var")
                inputs_dict[pname] = {
                    "type_name": item.get("type", item.get("type_name", "any")),
                    "state": item.get("state", "any"),
                    "required": item.get("required", True),
                    "default_value": item.get("default_value", None),
                    "description": item.get("description", ""),
                    "kind": item.get("kind"),
                    "role": item.get("role"),
                    "variadic": item.get("variadic"),
                    "supplied_out_of_band": item.get("supplied_out_of_band", False),
                }
        cell["inputs"] = inputs_dict
        modified = True
    elif not isinstance(raw_inputs, dict):
        cell["inputs"] = {}
        modified = True

    inputs = cell["inputs"]
    placeholders = set(extract_placeholders(cell.get("code_template", ""))) - {"output_var"}

    # Ensure 1-to-1 input key alignment with template placeholder if single port differs
    if len(inputs) == 1 and len(placeholders) == 1:
        inp_key = next(iter(inputs.keys()))
        ph_key = next(iter(placeholders))
        if inp_key != ph_key:
            inputs[ph_key] = inputs.pop(inp_key)
            modified = True

    # Ensure every placeholder has a matching declared input port
    for ph in placeholders:
        if ph not in inputs:
            port_type = infer_port_type(ph, domain)
            inputs[ph] = {
                "type_name": port_type,
                "state": "any",
                "required": True,
                "default_value": None,
                "description": f"Input {ph}",
            }
            modified = True

    # Prune declared inputs that are not referenced in the template, EXCEPT
    # optional / config / variadic ports which may be supplied out-of-band.
    for inp in list(inputs.keys()):
        if inp in placeholders:
            continue
        if _port_is_preservable(inputs[inp]):
            continue
        del inputs[inp]
        modified = True

    return modified
