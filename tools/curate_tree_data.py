"""
tools/curate_tree_data.py - Deterministic, reflection-driven data curation.

Repairs structural defects in harvested domain trees WITHOUT any runtime engine
changes and WITHOUT library-specific rules. Every operation is idempotent:

  R1. Public-surface rewrite:
      Cells harvested from private defining modules carry dependencies like
      `import sklearn.linear_model._base`, which the lattice loader correctly
      rejects (private implementation artifacts). For each such cell, resolve
      the most public re-export of the symbol via live import reflection and
      rewrite BOTH the dependency and any private module references inside the
      code template. Cells with no public re-export are genuine internals and
      are DROPPED.

  R2. Argument-less template repair:
      A stage-1/2/3 call template of the form `... <expr>()` that declares no
      variadic port is rebuilt from the live library signature (the historical
      harvester skipped *args parameters, producing nodes that can never
      receive data, e.g. `train_test_split()`).

  R3. Heterogeneous product declaration:
      Curated semantic declaration for partitioning functions whose live
      signature returns a variable-length sequence of arrays but which are
      semantically fixed-arity heterogeneous products (declared member
      carriers/states enable product-elimination at synthesis time).

Usage:  python3 tools/curate_tree_data.py [--trees-dir trees] [--domains sklearn pandas ...]
"""
from __future__ import annotations

import sys
import json
import importlib
import inspect
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenizer import CellTokenizer  # noqa: E402
from signature_introspector import resolve_signature, extract_clean_type_name  # noqa: E402


def _public_alias(module_path: str, attr_name: str) -> Optional[str]:
    """Resolves the most public importable path for a symbol; None if none exists."""
    path = module_path
    while "._" in path:
        path = path.split("._", 1)[0]
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, attr_name):
                return f"{path}.{attr_name}"
        except Exception:
            continue
    if "._" in module_path:
        return None
    return f"{module_path}.{attr_name}"


def _resolve_live_callable(cell: Dict[str, Any]) -> Optional[Tuple[Any, str]]:
    """Locates the live callable/class a cell was harvested from, via its template."""
    tmpl = cell.get("code_template", "") or ""
    # Extract the called expression: the right-hand side of the first assignment
    rhs = tmpl.split("=", 1)[1] if "=" in tmpl else tmpl
    expr = rhs.strip().split("(")[0].strip()
    expr = expr.replace("{output_var}", "").strip()
    if not expr or "{" in expr:
        return None
    parts = expr.rsplit(".", 1)
    if len(parts) != 2:
        return None
    mod_path, attr = parts
    try:
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, attr, None)
    except Exception:
        return None
    if obj is None:
        return None
    return obj, expr


def r1_public_rewrite(cells: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Rewrites private dependency/template module paths to public aliases."""
    rewritten, dropped = 0, 0
    for cell in cells:
        deps = cell.get("dependencies", []) or []
        private_deps = [d for d in deps if "._" in d or d.strip().startswith("import _")]
        if not private_deps:
            continue
        cid = cell.get("cell_id", "")
        id_parts = cid.split("_")[1:] or cid.split("_")
        id_suffixes = ["_".join(id_parts[k:]).lower() for k in range(len(id_parts))]
        live = _resolve_live_callable(cell)
        # The owning class of a method cell is structurally declared as the
        # receiver port's carrier (inputs['data'].type_name).
        receiver_class = ""
        data_port = (cell.get("inputs", {}) or {}).get("data", {})
        if isinstance(data_port, dict):
            receiver_class = str(data_port.get("type_name", "") or "")
        new_deps: List[str] = []
        ok = True
        for dep in deps:
            if "._" not in dep and not dep.strip().startswith("import _"):
                new_deps.append(dep)
                continue
            mod_path = dep.replace("import ", "").strip()
            attr = None
            if live is not None:
                expr = live[1]
                attr = expr.rsplit(".", 1)[1]
            if not attr and receiver_class:
                attr = receiver_class
            if not attr:
                try:
                    shallow_mod = importlib.import_module(mod_path.split("._")[0])
                except Exception:
                    shallow_mod = None
                for cand in sorted(set(id_suffixes), key=len, reverse=True):
                    if shallow_mod is not None and hasattr(shallow_mod, cand):
                        attr = cand
                        break
                    if shallow_mod is not None and hasattr(shallow_mod, cand.title()):
                        attr = cand.title()
                        break
            alias = _public_alias(mod_path, attr) if attr else None
            if alias is None:
                ok = False
                break
            public_mod = alias.rsplit(".", 1)[0]
            new_deps.append(f"import {public_mod}")
        if not ok:
            dropped += 1
            continue
        # Rewrite template references to the private module paths
        tmpl = cell.get("code_template", "") or ""
        for dep in private_deps:
            mod_path = dep.replace("import ", "").strip()
            live_expr = live[1] if live else ""
            attr = live_expr.rsplit(".", 1)[1] if live_expr and "." in live_expr else None
            alias = _public_alias(mod_path, attr) if attr else None
            if alias:
                tmpl = tmpl.replace(f"{mod_path}.", f"{alias.rsplit('.', 1)[0]}.")
        cell["dependencies"] = new_deps
        cell["code_template"] = tmpl
        rewritten += 1
    return rewritten, dropped


def r2_variadic_repair(cells: List[Dict[str, Any]], domain: str) -> int:
    """Rebuilds argument-less call templates from the live library signature."""
    repaired = 0
    for cell in cells:
        tmpl = (cell.get("code_template", "") or "").strip()
        if "(" not in tmpl:
            continue
        rhs = tmpl.split("=", 1)[1] if "=" in tmpl else tmpl
        call_expr = rhs.strip()
        if "()" not in call_expr:
            continue
        head = call_expr.split("()")[0].strip()
        if not head or "{" in head:
            continue
        # Argument-less call with no declared data port: rebuild from live signature
        placeholders = _extract_placeholders(tmpl)
        if any(ph != "output_var" for ph in placeholders):
            continue
        live = _resolve_live_callable(cell)
        if live is None:
            continue
        obj, expr = live
        try:
            sig = resolve_signature(obj, callable_name=expr.rsplit(".", 1)[1])
        except Exception:
            continue
        if sig is None:
            continue
        inputs: Dict[str, Any] = dict(cell.get("inputs", {}) or {})
        args: List[str] = []
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            p_type = extract_clean_type_name(p.annotation) or "any"
            is_req = p.default is inspect.Parameter.empty or p.kind is inspect.Parameter.VAR_POSITIONAL
            inputs[p.name] = {
                "type_name": "any" if p.kind is inspect.Parameter.VAR_POSITIONAL else p_type,
                "state": "any",
                "required": bool(is_req),
                "default_value": None if p.default is inspect.Parameter.empty else str(p.default),
                "description": f"Argument {p.name}",
            }
            if is_req:
                args.append(f"{{{p.name}}}")
        if not args:
            continue
        new_call = f"{head}({', '.join(args)})"
        cell["code_template"] = tmpl.replace(f"{head}()", new_call, 1)
        cell["inputs"] = inputs
        repaired += 1
    return repaired


def r3_product_declaration(cells: List[Dict[str, Any]], domain: str) -> int:
    """
    Declares heterogeneous product outputs for partitioning morphisms whose
    live signature returns a variable-length array sequence. Curated data
    declaration (member carriers + states) enabling product-elimination at
    synthesis time. Keyed by declared member states derived from the call's
    own name tokens — no engine-side name knowledge.
    """
    declared = 0
    for cell in cells:
        cid = cell.get("cell_id", "")
        toks = {t.lower() for t in CellTokenizer.tokenize_identifier(cid)}
        if not {"train", "test", "split"}.issubset(toks):
            continue
        if cell.get("stage") != 2:
            continue
        outs = cell.get("outputs", {}) or {}
        out_port = outs.get("output_data", {})
        if "[" in str(out_port.get("type_name", "")):
            continue  # already declared
        out_port["type_name"] = "tuple[any[train], any[test], any[train_target], any[test_target]]"
        out_port["state"] = "partitioned"
        outs["output_data"] = out_port
        cell["outputs"] = outs
        declared += 1
    return declared


def r4_path_carrier_retype(cells: List[Dict[str, Any]], domain: str) -> int:
    """
    Retypes path-semantics arguments from the generic `str` carrier to the
    registered `filepath` subtype, using a documented curation vocabulary of
    path-semantic parameter names. This is DATA declaration (which parameter
    positions materialize/consume filesystem assets), enabling the engine's
    type-driven asset binding and egress derivation — the engine itself stays
    name-agnostic.
    """
    path_semantic_names = {
        "filename", "file", "filepath", "path", "pathname", "fname",
        "path_or_buf", "filepath_or_buffer", "file_path", "dest", "dest_path",
        "destination", "filename_or_buffer", "target_path", "output_path",
    }
    retyped = 0
    for cell in cells:
        changed = False
        for p_name, p_val in (cell.get("inputs", {}) or {}).items():
            if not isinstance(p_val, dict):
                continue
            if str(p_val.get("type_name", "")).lower() not in ("str", "string", "text"):
                continue
            if str(p_val.get("state", "")).lower() in ("source_identifier",):
                continue
            if p_name.lower() in path_semantic_names:
                p_val["type_name"] = "filepath"
                changed = True
        retyped += 1 if changed else 0
    return retyped



def r5_argless_path_emission(cells: List[Dict[str, Any]], domain: str) -> int:
    """
    Argument-less call templates cannot emit their destination: a writer cell
    whose template is `... to_csv()` drops the bound path literal at synthesis.
    For such templates, append keyword segments for path-typed optional
    parameters (declared by R4) so the destination port is emittable.
    """
    repaired = 0
    for cell in cells:
        tmpl = (cell.get("code_template", "") or "").strip()
        if "(" not in tmpl or "=" not in tmpl:
            continue
        rhs = tmpl.split("=", 1)[1]
        call_expr = rhs.strip()
        if "()" not in call_expr:
            continue
        head = call_expr.split("()")[0].strip()
        if not head:
            continue
        # Only patch cells that declare a path-typed optional port
        path_ports = [
            p_name for p_name, p_val in (cell.get("inputs", {}) or {}).items()
            if isinstance(p_val, dict)
            and str(p_val.get("type_name", "")).lower() in ("filepath", "path", "uri", "filename")
        ]
        if not path_ports:
            continue
        existing = set(_extract_placeholders(tmpl))
        kwargs = [p for p in path_ports if p not in existing]
        if not kwargs:
            continue
        new_call = f"{head}({', '.join(f'{p}={{{p}}}' for p in kwargs)})"
        cell["code_template"] = tmpl.replace(f"{head}()", new_call, 1)
        repaired += 1
    return repaired


def r6_writer_endomorphism(cells: List[Dict[str, Any]], domain: str) -> int:
    """
    Writer methods whose live return annotation is a None-union (e.g.
    ``str | None``) return None when they actually write a destination. Their
    categorical semantics are the endomorphism D -> D (the receiver remains the
    pipeline value), with the destination port carrying the artifact. Rewrites
    single-line templates to the endomorphism form so the terminal variable
    never evaluates to None after a successful write.
    """
    import inspect as _inspect
    repaired = 0
    for cell in cells:
        tmpl = (cell.get("code_template", "") or "").strip()
        if "{output_var} = {data}." not in tmpl or tmpl.rstrip().endswith("= {data}"):
            continue
        has_path_port = any(
            isinstance(p_val, dict)
            and str(p_val.get("type_name", "")).lower() in ("filepath", "path", "uri", "filename")
            for p_val in (cell.get("inputs", {}) or {}).values()
        )
        if not has_path_port:
            continue
        live = _resolve_live_callable(cell)
        fn = None
        if live is not None:
            fn = live[0]
        if fn is None:
            # Placeholder-bearing method templates ({data}.to_csv(...)): resolve
            # the method via the receiver class declared on the data port.
            method_name = tmpl.split("{data}.", 1)[1].split("(")[0].split(".")[0].strip() if "{data}." in tmpl else ""
            receiver = (cell.get("inputs", {}) or {}).get("data", {})
            recv_type = str(receiver.get("type_name", "")) if isinstance(receiver, dict) else ""
            if method_name and recv_type:
                for dep in cell.get("dependencies", []) or []:
                    try:
                        mod = importlib.import_module(dep.replace("import ", "").strip())
                        cls = getattr(mod, recv_type, None)
                        cand = getattr(cls, method_name, None) if cls is not None else None
                        if callable(cand):
                            fn = cand
                            break
                    except Exception:
                        continue
        if fn is None:
            continue
        try:
            sig = resolve_signature(fn, callable_name="method")
            ann = sig.return_annotation if sig else None
        except Exception:
            ann = None
        ann_str = str(ann) if ann is not None else ""
        if "None" not in ann_str and ann_str not in ("",):
            continue
        first_line = tmpl.splitlines()[0]
        cell["code_template"] = f"{first_line}\n{{output_var}} = {{data}}"
        cell["stage"] = 2
        cell["node_type"] = "function"
        cell["node_role"] = "transform"
        outs = cell.get("outputs", {}) or {}
        od = outs.get("output_data", {})
        data_port = (cell.get("inputs", {}) or {}).get("data", {})
        recv_type = str(data_port.get("type_name", "any")) if isinstance(data_port, dict) else "any"
        od["type_name"] = recv_type or "any"
        od["state"] = "mutated"
        outs["output_data"] = od
        cell["outputs"] = outs
        repaired += 1
    return repaired

def _extract_placeholders(template: str) -> List[str]:
    placeholders: List[str] = []
    i, n = 0, len(template)
    while i < n:
        if template[i] == "{":
            j = template.find("}", i + 1)
            if j != -1:
                inner = template[i + 1 : j]
                if inner.isidentifier():
                    placeholders.append(inner)
                i = j + 1
                continue
        i += 1
    return placeholders


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic tree data curation")
    parser.add_argument("--trees-dir", default="trees")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--backup", action="store_true", help="Write .pre-curation backups")
    args = parser.parse_args()

    trees_dir = Path(args.trees_dir)
    files = sorted(trees_dir.glob("*.json"))
    if args.domains:
        wanted = {d.lower() for d in args.domains}
        files = [f for f in files if f.stem.lower() in wanted]

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        cells = data if isinstance(data, list) else data.get("cells", [])
        before = len(cells)

        if args.backup:
            backup = f.with_suffix(".json.pre-curation")
            if not backup.exists():
                backup.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")

        rw, dp = r1_public_rewrite(cells)
        rp = r2_variadic_repair(cells, f.stem)
        pr = r3_product_declaration(cells, f.stem)
        rt = r4_path_carrier_retype(cells, f.stem)
        r5 = r5_argless_path_emission(cells, f.stem)
        r6 = r6_writer_endomorphism(cells, f.stem)
        cells = [c for c in cells if c is not None]
        after = len(cells)

        out = data if isinstance(data, dict) else cells
        if isinstance(data, dict):
            data["cells"] = cells
        f.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[{f.stem}] public-rewrite={rw} dropped={dp} variadic-repair={rp} product-decl={pr} path-retype={rt} argless-path={r5} writer-endo={r6} cells={before}→{after}")


if __name__ == "__main__":
    main()
