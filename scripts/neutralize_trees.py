#!/usr/bin/env python3
"""
scripts/neutralize_trees.py — Domain-neutral canonical manifest normalizer & validator.

Checks and normalizes domain plugin manifests (`trees/*.json` and optional fixtures)
to strict canonical v2 `TreeSchema` (`extra="forbid"`) using purely schema-driven rules:
  1. Infers `imports` deterministically from cell dependencies, qualified typestates,
     and loaded `is_core` foundational trees.
  2. Strips non-canonical legacy fields and dotted noise invariants generically.
  3. Canonicalizes 5-axis ports (`layout`, `value_class`, `resource_mode`, `invariants`)
     and validates strict `TreeSchema` conformance before writing (or under `--check`).
"""
from __future__ import annotations

import argparse
import ast
import copy
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schema import (  # noqa: E402
    BridgeProfileSchema,
    ConditionPredicate,
    LanguageSpec,
    LayoutContract,
    PolymorphicPropagationSchema,
    RuntimeSetupSchema,
    TOP_CARRIERS,
    TreeSchema,
    TypestateDefinition,
    derive_value_class,
    is_literal_config_carrier,
)
from type_system import BoundedPoset  # noqa: E402

_CANONICAL_LAYOUT_KEYS: Tuple[str, ...] = tuple(LayoutContract.model_fields.keys())
_CANONICAL_PREDICATE_KEYS: Tuple[str, ...] = tuple(ConditionPredicate.model_fields.keys())
_CANONICAL_BRIDGE_KEYS: Tuple[str, ...] = tuple(BridgeProfileSchema.model_fields.keys())
_CANONICAL_POLY_KEYS: Tuple[str, ...] = tuple(PolymorphicPropagationSchema.model_fields.keys())
_CANONICAL_TS_DEF_KEYS: Tuple[str, ...] = tuple(TypestateDefinition.model_fields.keys())
_CANONICAL_RUNTIME_KEYS: Tuple[str, ...] = tuple(RuntimeSetupSchema.model_fields.keys())
_CANONICAL_LANG_KEYS: Tuple[str, ...] = tuple(LanguageSpec.model_fields.keys())
_VALID_VALUE_CLASSES: Set[str] = {"data", "literal", "external_asset", "identifier", "enum"}
_VALID_RESOURCE_MODES: Set[str] = {"value", "acquire", "borrow", "release"}
_NOISE_INVARIANTS: Set[str] = {
    "primary",
    "optional",
    "scalar",
    "estimator",
    "collection",
    "const",
}


def _is_dotted_type_noise_invariant(inv_l: str) -> bool:
    """Detect dotted package/class type tags (e.g. pkg.type) mistakenly stored in invariants."""
    if "." not in inv_l or any(ch in inv_l for ch in "=<>:()"):
        return False
    parts = inv_l.split(".")
    return len(parts) >= 2 and all(seg.isidentifier() for seg in parts)


def _clean_invariants(raw_invs: Any) -> List[str]:
    """Strip legacy role/tag noise tokens from port invariants."""
    out: List[str] = []
    for raw in (raw_invs or ()):
        s = str(raw or "").strip()
        if not s:
            continue
        sl = s.lower()
        if sl in _NOISE_INVARIANTS or _is_dotted_type_noise_invariant(sl):
            continue
        out.append(s)
    return out


def _parse_dep_roots(deps: List[Any]) -> Set[str]:
    roots: Set[str] = set()
    for raw in deps or ():
        s = str(raw or "").strip()
        if not s:
            continue
        if not s.startswith(("import ", "from ")):
            s = f"import {s}"
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".", 1)[0].strip().lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0].strip().lower())
    return roots


def _derive_resource_mode(
    port: Dict[str, Any],
    direction: str,
    cell: Dict[str, Any],
    incoming_handle_types: Set[str],
) -> str:
    if "structure_kind" not in port and "abstract_type" not in port:
        existing_rm = str(port.get("resource_mode") or "value").strip().lower()
        return existing_rm if existing_rm in _VALID_RESOURCE_MODES else "value"

    sk = str(port.get("structure_kind") or "").strip().lower()
    at = str(port.get("abstract_type") or "").strip().lower()
    is_handle = (sk == "handle") or (at == "handle")
    if not is_handle:
        return "value"

    t_low = str(port.get("type_name") or port.get("carrier") or "").strip().lower()
    cid_low = str(cell.get("cell_id") or "").strip().lower()
    stage = int(cell.get("stage") or 2)
    mut = str(cell.get("mutation_type") or "").strip().lower()

    if direction == "inputs":
        is_release = (
            stage == 3
            or mut in ("cleanup", "release", "close")
            or any(cid_low.endswith(sfx) for sfx in ("_quit", "_close", "_release", "_destroy", "_free", ".close", ".quit"))
        )
        return "release" if is_release else "borrow"

    if stage == 1 or (t_low and t_low not in incoming_handle_types):
        return "acquire"
    return "borrow"


def _canonicalize_layout(pv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_layout = pv.get("layout") if isinstance(pv.get("layout"), dict) else {}
    sc = pv.get("shape_contract") if isinstance(pv.get("shape_contract"), dict) else {}

    rank_v = raw_layout.get("rank")
    if rank_v is None:
        rank_v = sc.get("ndim")
    if rank_v is None:
        rank_v = pv.get("rank")

    shape_v = raw_layout.get("shape")
    if shape_v is None:
        shape_v = sc.get("shape")

    dtype_v = raw_layout.get("dtype")
    if dtype_v is None:
        dtype_v = pv.get("element_type")

    rp_v = bool(raw_layout.get("rank_polymorphic", sc.get("rank_polymorphic", False)))

    cc_v = raw_layout.get("channel_count")
    if cc_v is None:
        cc_v = sc.get("channel_count")

    ca_v = raw_layout.get("channel_axis")
    if ca_v is None:
        ca_v = sc.get("channel_axis")

    len_v = raw_layout.get("length")
    if len_v is None:
        len_v = sc.get("len")

    if not any(v is not None for v in (rank_v, shape_v, dtype_v, cc_v, ca_v, len_v)) and not rp_v:
        return None

    out = {
        "rank": rank_v,
        "shape": shape_v,
        "dtype": dtype_v,
        "rank_polymorphic": rp_v,
        "channel_count": cc_v,
        "channel_axis": ca_v,
        "length": len_v,
    }
    return {k: out[k] for k in _CANONICAL_LAYOUT_KEYS}


def _canonicalize_port(
    pv: Dict[str, Any],
    direction: str,
    cell: Dict[str, Any],
    incoming_handle_types: Set[str],
) -> Dict[str, Any]:
    raw_vc = str(pv.get("value_class") or "").strip().lower()
    resource_mode = _derive_resource_mode(pv, direction, cell, incoming_handle_types)
    layout = _canonicalize_layout(pv)
    carrier = str(pv.get("carrier") or pv.get("type_name") or pv.get("type") or "any").strip()
    role_val = str(pv.get("port_role") if pv.get("port_role") is not None else (pv.get("role") or "")).strip().lower()
    dtype_val = str(layout.get("dtype") or "").strip().lower() if isinstance(layout, dict) else ""
    if (
        raw_vc == "data"
        and role_val in ("config", "configuration", "literal_parameter", "parameter", "dim_size")
        and resource_mode == "value"
        and (
            is_literal_config_carrier(carrier)
            or (layout is not None and layout.get("rank") == 0 and dtype_val != "composite_record")
        )
    ):
        value_class = "literal"
    else:
        value_class = raw_vc if raw_vc in _VALID_VALUE_CLASSES else derive_value_class(pv)

    typestate = str(pv.get("typestate") or pv.get("state") or "any").strip()
    accepted_states = [
        str(s).strip()
        for s in (pv.get("accepted_states") or ())
        if str(s or "").strip()
    ]
    invariants = _clean_invariants(pv.get("invariants"))

    clean_port: Dict[str, Any] = {
        "carrier": carrier,
        "typestate": typestate,
        "accepted_states": accepted_states,
        "layout": layout,
        "invariants": invariants,
        "value_class": value_class,
        "resource_mode": resource_mode,
        "required": bool(pv.get("required", True)),
    }

    if "default_value" in pv and pv["default_value"] is not None:
        clean_port["default_value"] = pv["default_value"]
    elif "default" in pv and pv["default"] is not None:
        clean_port["default_value"] = pv["default"]

    if pv.get("enum_values") is not None:
        clean_port["enum_values"] = list(pv["enum_values"])

    if pv.get("description") is not None:
        clean_port["description"] = pv["description"]

    role = pv.get("port_role") if pv.get("port_role") is not None else pv.get("role")
    if role is not None and str(role).strip():
        clean_port["port_role"] = str(role).strip()

    if pv.get("domain") is not None and str(pv["domain"]).strip():
        clean_port["domain"] = str(pv["domain"]).strip().lower()

    return clean_port


def _canonicalize_effects(raw_effs: Any, raw_postconds: Any) -> List[Any]:
    out: List[Any] = []
    seen: Set[str] = set()
    for item in list(raw_effs or ()) + list(raw_postconds or ()):
        if isinstance(item, dict):
            cleaned = {k: item[k] for k in _CANONICAL_PREDICATE_KEYS if k in item}
            if not cleaned:
                continue
            key = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                out.append(cleaned)
        elif isinstance(item, str):
            s = item.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _canonicalize_cell(cell: Dict[str, Any], dom: str) -> Dict[str, Any]:
    in_ports = cell.get("inputs") or {}
    incoming_handle_types: Set[str] = set()
    if isinstance(in_ports, dict):
        for pv in in_ports.values():
            if not isinstance(pv, dict):
                continue
            sk = str(pv.get("structure_kind") or "").strip().lower()
            at = str(pv.get("abstract_type") or "").strip().lower()
            if sk == "handle" or at == "handle" or str(pv.get("resource_mode") or "").strip().lower() in ("borrow", "release", "acquire"):
                t_low = str(pv.get("carrier") or pv.get("type_name") or "").strip().lower()
                if t_low:
                    incoming_handle_types.add(t_low)

    clean_inputs: Dict[str, Dict[str, Any]] = {}
    if isinstance(in_ports, dict):
        for pname, pval in in_ports.items():
            if isinstance(pval, dict):
                clean_inputs[str(pname)] = _canonicalize_port(pval, "inputs", cell, incoming_handle_types)

    out_ports = cell.get("outputs") or {}
    clean_outputs: Dict[str, Dict[str, Any]] = {}
    if isinstance(out_ports, dict):
        for pname, pval in out_ports.items():
            if isinstance(pval, dict):
                clean_outputs[str(pname)] = _canonicalize_port(pval, "outputs", cell, incoming_handle_types)

    tmpl_parts = [str(cell.get("code_template") or "")]
    raw_tmpls = cell.get("code_templates")
    if isinstance(raw_tmpls, dict):
        tmpl_parts.extend(str(v or "") for v in raw_tmpls.values())
    tmpl_txt = "\n".join(tmpl_parts)

    deps: List[str] = []
    seen_deps: Set[str] = set()
    for raw_dep in list(cell.get("dependencies") or ()) + list(cell.get("imports") or ()):
        s = str(raw_dep or "").strip()
        if s and "." in s and not s.startswith(("import ", "from ")):
            pkg, leaf = s.rsplit(".", 1)
            if pkg and leaf.isidentifier() and f"{leaf}." in tmpl_txt and f"{s}." not in tmpl_txt:
                s = f"from {pkg} import {leaf}"
        if s and s not in seen_deps:
            seen_deps.add(s)
            deps.append(s)

    mut = str(cell.get("mutation_type") or "pure").strip().lower() or "pure"
    if mut == "immutable":
        mut = "pure"

    cid_str = str(cell.get("cell_id") or "").strip()
    eff_list = _canonicalize_effects(cell.get("effects"), cell.get("postconditions"))
    for p_clean in clean_inputs.values():
        if p_clean.get("value_class") != "external_asset":
            continue
        cur_role = str(p_clean.get("port_role") or "").strip().lower()
        if cur_role in ("destination", "source_data", "data_input"):
            continue
        ts_raw = str(p_clean.get("typestate") or "").strip().lower()
        bare_ts = ts_raw.split("::", 1)[-1] if "::" in ts_raw else ts_raw
        if bare_ts == "destination_path":
            dom_v = p_clean.pop("domain", None)
            p_clean["port_role"] = "destination"
            if dom_v is not None:
                p_clean["domain"] = dom_v

    clean_cell: Dict[str, Any] = {
        "cell_id": cid_str,
        "domain_name": str(cell.get("domain_name") or dom).strip().lower(),
        "docstring": str(cell.get("docstring") or cell.get("doc") or "").strip(),
        "code_template": str(cell.get("code_template") or ""),
        "code_templates": dict(cell.get("code_templates") or {}),
        "dependencies": deps,
        "inputs": clean_inputs,
        "outputs": clean_outputs,
        "primary_in": cell.get("primary_in"),
        "primary_out": cell.get("primary_out"),
        "effects": eff_list,
        "mutation_type": mut,
        "is_context_manager": bool(cell.get("is_context_manager", False)),
    }

    bp = cell.get("bridge_profile")
    if isinstance(bp, dict) and bp:
        clean_cell["bridge_profile"] = {k: bp[k] for k in _CANONICAL_BRIDGE_KEYS if k in bp}

    pp = cell.get("polymorphic_propagation")
    if isinstance(pp, dict) and pp:
        clean_cell["polymorphic_propagation"] = {k: pp[k] for k in _CANONICAL_POLY_KEYS if k in pp}

    return clean_cell


def _canonicalize_typestates(raw_ts: Any, dom: str) -> Optional[Dict[str, Any]]:
    if raw_ts is None:
        return None
    if isinstance(raw_ts, list):
        items = raw_ts
        ts_dict: Dict[str, Any] = {"domain": dom}
    elif isinstance(raw_ts, dict):
        items = raw_ts.get("states") or []
        ts_dict = dict(raw_ts)
    else:
        return None

    clean_states: List[Any] = []
    for entry in items:
        if isinstance(entry, str) and entry.strip():
            clean_states.append(entry.strip())
        elif isinstance(entry, dict) and entry.get("name"):
            clean_states.append({k: entry[k] for k in _CANONICAL_TS_DEF_KEYS if k in entry})

    return {
        "domain": str(ts_dict.get("domain") or dom).strip().lower(),
        "states": clean_states,
        "transitions": list(ts_dict.get("transitions") or []),
        "initial_state": ts_dict.get("initial_state"),
        "terminal_states": list(ts_dict.get("terminal_states") or []),
    }


def _canonicalize_tree(data: Dict[str, Any], dom: str, imports_list: List[str]) -> Dict[str, Any]:
    raw_cells = data.get("cells") or []
    clean_cells = [_canonicalize_cell(c, dom) for c in raw_cells if isinstance(c, dict)]

    clean_tree: Dict[str, Any] = {
        "domain": dom,
        "version": str(data.get("version") or "2.0.0"),
        "description": str(data.get("description") or ""),
        "is_core": bool(data.get("is_core", False)),
        "imports": imports_list,
        "types": data.get("types") if data.get("types") is not None else {},
        "top_types": [str(x) for x in (data.get("top_types") or ()) if str(x or "").strip()],
        "typestates": _canonicalize_typestates(data.get("typestates"), dom),
        "cells": clean_cells,
    }

    rs = data.get("runtime_setup")
    if isinstance(rs, dict):
        clean_tree["runtime_setup"] = {k: rs[k] for k in _CANONICAL_RUNTIME_KEYS if k in rs}

    ls = data.get("language_spec")
    if isinstance(ls, dict):
        clean_tree["language_spec"] = {k: ls[k] for k in _CANONICAL_LANG_KEYS if k in ls}

    raw_aliases = [str(x).strip().lower() for x in (data.get("aliases") or ()) if str(x or "").strip()]
    if raw_aliases:
        clean_tree["aliases"] = list(dict.fromkeys(raw_aliases))

    TreeSchema(**clean_tree)
    return clean_tree


def _collect_tree_files(include_fixtures: bool) -> List[Path]:
    files = sorted(glob.glob(str(ROOT / "trees" / "*.json")))
    if include_fixtures:
        files.extend(sorted(glob.glob(str(ROOT / "benchmarks" / "fixtures" / "*" / "*.json"))))
    seen: Set[Path] = set()
    all_files: List[Path] = []
    for f in files:
        p = Path(f).resolve()
        if p not in seen:
            seen.add(p)
            all_files.append(p)
    return all_files


def neutralize_all(*, check: bool = False, include_fixtures: bool = True) -> int:
    all_files = _collect_tree_files(include_fixtures=include_fixtures)
    parsed: List[Tuple[Path, str, Dict[str, Any], Dict[str, Any]]] = []
    known_domains: Set[str] = set()
    core_domains: List[str] = []
    dom_dep_counts: Dict[str, Dict[str, int]] = {}

    for p in all_files:
        orig_data = json.loads(p.read_text(encoding="utf-8"))
        data = copy.deepcopy(orig_data)
        dom = str(data.get("domain") or p.stem).strip().lower()
        known_domains.add(dom)
        if bool(data.get("is_core", False)):
            core_domains.append(dom)
        counts: Dict[str, int] = {}
        for c in data.get("cells") or []:
            deps = list(c.get("dependencies") or ()) + list(c.get("imports") or ())
            for r in _parse_dep_roots(deps):
                counts[r] = counts.get(r, 0) + 1
        dom_dep_counts[dom] = counts
        parsed.append((p, dom, orig_data, data))

    pkg_to_dom: Dict[str, str] = {d: d for d in known_domains}
    for dom, counts in dom_dep_counts.items():
        foreign_only = {r: cnt for r, cnt in counts.items() if r not in pkg_to_dom}
        if foreign_only:
            top_pkg = max(foreign_only, key=lambda r: (foreign_only[r], r))
            if counts.get(dom, 0) == 0 and foreign_only[top_pkg] >= max(1, sum(foreign_only.values()) // 2):
                pkg_to_dom[top_pkg] = dom

    prepared: List[Tuple[Path, str, Dict[str, Any], Dict[str, Any]]] = []
    for p, dom, orig_data, data in parsed:
        is_core = bool(data.get("is_core", False))
        foreign_imports: Set[str] = set()
        if not is_core:
            for core_dom in core_domains:
                if core_dom != dom:
                    foreign_imports.add(core_dom)
            for existing_imp in data.get("imports") or ():
                imp_l = str(existing_imp or "").strip().lower()
                if imp_l and imp_l != dom and imp_l in known_domains:
                    foreign_imports.add(imp_l)

        for cell in data.get("cells") or []:
            all_deps = list(cell.get("dependencies") or ()) + list(cell.get("imports") or ())
            for root in _parse_dep_roots(all_deps):
                owner = pkg_to_dom.get(root)
                if owner and owner != dom:
                    foreign_imports.add(owner)
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if not isinstance(pv, dict):
                        continue
                    for s_key in ("typestate", "state"):
                        sv = str(pv.get(s_key) or "").strip().lower()
                        if "::" in sv:
                            pref = sv.split("::", 1)[0].strip().lower()
                            if pref and pref != dom and pref in known_domains:
                                foreign_imports.add(pref)
                    for acc_s in pv.get("accepted_states") or ():
                        sv = str(acc_s or "").strip().lower()
                        if "::" in sv:
                            pref = sv.split("::", 1)[0].strip().lower()
                            if pref and pref != dom and pref in known_domains:
                                foreign_imports.add(pref)

        imports_list = [] if is_core else sorted(foreign_imports)
        clean_tree = _canonicalize_tree(data, dom, imports_list)
        prepared.append((p, dom, orig_data, clean_tree))

    trees_dir = (ROOT / "trees").resolve()
    global_dom_map: Dict[str, Dict[str, Any]] = {}
    dir_dom_map: Dict[Tuple[Path, str], Dict[str, Any]] = {}
    for p, dom, _, ct in prepared:
        dir_dom_map[(p.resolve().parent, dom)] = ct
        if p.resolve().parent == trees_dir:
            global_dom_map[dom] = ct
    for p, dom, _, ct in prepared:
        if dom not in global_dom_map:
            global_dom_map[dom] = ct

    updated_count = 0
    drift_count = 0
    total_cells = 0

    for p, dom, orig_data, clean_tree in prepared:
        closure_trees: List[Dict[str, Any]] = [clean_tree]
        seen_doms: Set[str] = {dom}
        queue: List[str] = [str(imp).strip().lower() for imp in (clean_tree.get("imports") or ()) if str(imp).strip()]
        p_dir = p.resolve().parent
        while queue:
            imp_dom = queue.pop(0)
            if imp_dom in seen_doms:
                continue
            seen_doms.add(imp_dom)
            imp_tree = dir_dom_map.get((p_dir, imp_dom)) or global_dom_map.get(imp_dom)
            if not imp_tree:
                continue
            closure_trees.append(imp_tree)
            for sub_imp in imp_tree.get("imports") or ():
                sub_l = str(sub_imp).strip().lower()
                if sub_l and sub_l not in seen_doms:
                    queue.append(sub_l)

        type_parents: Dict[str, Set[str]] = {}
        top_types: Set[str] = set()
        released_carriers: Set[str] = set()
        state_layout_specs: Dict[str, Tuple[Optional[str], int]] = {}

        for ct in closure_trees:
            ct_dom = str(ct.get("domain") or "").strip().lower()
            for t in (ct.get("top_types") or ()):
                tl = str(t or "").strip().lower()
                if tl:
                    top_types.add(tl)
            raw_types = ct.get("types")
            if isinstance(raw_types, dict):
                t_items = [(str(k), v) for k, v in raw_types.items()]
            elif isinstance(raw_types, list):
                t_items = [
                    (str(e.get("name") or e.get("type_name")), e)
                    for e in raw_types
                    if isinstance(e, dict) and (e.get("name") or e.get("type_name"))
                ]
            else:
                t_items = []
            for tname, tdef in t_items:
                tn = tname.strip().lower()
                if not tn:
                    continue
                type_parents.setdefault(tn, set())
                if isinstance(tdef, dict):
                    if tdef.get("parent"):
                        type_parents[tn].add(str(tdef["parent"]).strip().lower())
                    for p_it in (tdef.get("parents") or ()):
                        ps = str(p_it or "").strip().lower()
                        if ps:
                            type_parents[tn].add(ps)
            for cell in ct.get("cells") or []:
                for pv in (cell.get("inputs") or {}).values():
                    if isinstance(pv, dict) and str(pv.get("resource_mode") or "").lower() == "release":
                        c_l = str(pv.get("carrier") or "").strip().lower()
                        if c_l and c_l not in TOP_CARRIERS:
                            released_carriers.add(c_l)

            ts_b = ct.get("typestates")
            if isinstance(ts_b, dict):
                ts_dom = str(ts_b.get("domain") or ct_dom).strip().lower()
                for s_entry in ts_b.get("states") or ():
                    if isinstance(s_entry, dict) and s_entry.get("name"):
                        nd = (s_entry.get("properties") or {}).get("ndim")
                        if isinstance(nd, int) and not isinstance(nd, bool):
                            sname = str(s_entry["name"]).strip().lower()
                            sc_raw = s_entry.get("carrier_type")
                            sc_l = str(sc_raw).strip().lower() if sc_raw else None
                            state_layout_specs[f"{ts_dom}::{sname}"] = (sc_l, nd)
                            if ts_dom == dom:
                                state_layout_specs[sname] = (sc_l, nd)

        poset = BoundedPoset(type_parents=type_parents, top_types=top_types)

        def _is_releasable(carrier_str: Any) -> bool:
            c_l = str(carrier_str or "").strip().lower()
            if not c_l or c_l in TOP_CARRIERS:
                return False
            if c_l in released_carriers:
                return True
            return any(
                anc != "any" and anc not in TOP_CARRIERS and anc in released_carriers
                for anc in poset.type_ancestors(c_l)
            )

        for cell in clean_tree.get("cells") or []:
            ins = cell.get("inputs") or {}
            outs = cell.get("outputs") or {}
            has_handle_in = any(
                isinstance(pv, dict) and str(pv.get("resource_mode") or "").lower() in ("acquire", "borrow")
                for pv in ins.values()
            )
            has_releasable_out = any(
                isinstance(pv, dict)
                and str(pv.get("resource_mode") or "").lower() == "acquire"
                and _is_releasable(pv.get("carrier"))
                for pv in outs.values()
            )
            for pv in list(ins.values()) + list(outs.values()):
                if not isinstance(pv, dict):
                    continue
                carrier_l = str(pv.get("carrier") or "").strip().lower()
                lay = pv.get("layout")
                if isinstance(lay, dict) and not lay.get("rank_polymorphic"):
                    st_raw = str(pv.get("typestate") or "").strip().lower()
                    st_qual = st_raw if "::" in st_raw else f"{dom}::{st_raw}"
                    spec = state_layout_specs.get(st_raw) or state_layout_specs.get(st_qual)
                    if spec is not None:
                        exp_carrier, exp_ndim = spec
                        if exp_carrier is None or exp_carrier == carrier_l:
                            lay["rank"] = exp_ndim
            for pv in outs.values():
                if isinstance(pv, dict) and str(pv.get("resource_mode") or "").lower() == "acquire":
                    if not _is_releasable(pv.get("carrier")):
                        pv["resource_mode"] = "borrow" if (has_handle_in or has_releasable_out) else "value"

        TreeSchema(**clean_tree)
        total_cells += len(clean_tree["cells"])
        rel = p.relative_to(ROOT)
        if check:
            if orig_data != clean_tree:
                drift_count += 1
                print(f"[neutralize:check] DRIFT: {rel} differs from canonical schema form.")
        else:
            p.write_text(json.dumps(clean_tree, indent=2) + "\n", encoding="utf-8")
            updated_count += 1
            print(f"[neutralize] {rel}: imports={clean_tree['imports']} cells={len(clean_tree['cells'])}")

    if check:
        if drift_count:
            print(f"\n[neutralize:check] FAIL: {drift_count} manifest(s) need canonical normalization.")
            return 1
        print(f"[neutralize:check] OK: All {len(prepared)} domain tree manifests are canonical & idempotent ({total_cells} cells).")
        return 0

    print(f"\n[neutralize] Completed: {updated_count} domain tree files normalized ({total_cells} total validated cells).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize or verify canonical STRATA v2 domain manifests.")
    parser.add_argument("--check", action="store_true", help="Verify manifests are canonical & idempotent without modifying files")
    parser.add_argument("--include-fixtures", action="store_true", help="Include benchmark fixture trees under benchmarks/fixtures/*/*.json")
    args = parser.parse_args()
    return neutralize_all(check=bool(args.check), include_fixtures=bool(args.include_fixtures))


if __name__ == "__main__":
    raise SystemExit(main())
