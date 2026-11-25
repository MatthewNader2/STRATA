#!/usr/bin/env python3
"""
scripts/neutralize_trees.py — Normalizes domain trees to strict canonical v2 STRATA schema.

Performs an idempotent cleanup across trees/*.json and benchmarks/fixtures/*/*.json:
  1. Infers declared `imports` automatically from each tree's dependencies & qualified typestates.
  2. Applies neutral cross-domain dependency alignment for shared types/typestates.
  3. Stamps explicit `value_class`, `resource_mode`, and canonical `layout` on every port before
     stripping legacy fields (`stage`, `keywords`, `semantic_tags`, `edges`, `postconditions`, etc.).
  4. Merges `postconditions` -> `effects` and cell `imports` -> `dependencies`, normalizes
     `mutation_type == "immutable"` -> `"pure"`, and strips all non-canonical tree/cell/port keys.
  5. Validates every normalized manifest strictly against `TreeSchema` (`extra="forbid"`) prior to write.
"""
from __future__ import annotations

import ast
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
    "os.path",
    "cv2.mask",
}


def _clean_invariants(raw_invs: Any) -> List[str]:
    """Strip legacy role/tag noise tokens from port invariants."""
    out: List[str] = []
    for raw in (raw_invs or ()):
        s = str(raw or "").strip()
        if not s:
            continue
        sl = s.lower()
        if sl in _NOISE_INVARIANTS or sl.startswith("cv2."):
            continue
        out.append(s)
    return out


def _parse_dep_roots(deps: List[Any]) -> Set[str]:
    roots: Set[str] = set()
    for raw in deps or []:
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


def _ensure_state_decls(data: Dict[str, Any], dom: str, required_states: List[Dict[str, Any]]) -> None:
    ts = data.get("typestates")
    if isinstance(ts, dict):
        states = ts.setdefault("states", [])
    elif isinstance(ts, list):
        data["typestates"] = {"domain": dom, "states": list(ts)}
        states = data["typestates"]["states"]
    else:
        data["typestates"] = {"domain": dom, "states": []}
        states = data["typestates"]["states"]

    existing: Set[str] = set()
    for entry in states:
        if isinstance(entry, dict) and entry.get("name"):
            existing.add(str(entry["name"]).strip().lower())
        elif isinstance(entry, str) and entry.strip():
            existing.add(entry.strip().lower())

    for spec in required_states:
        sname = str(spec["name"]).strip().lower()
        if sname not in existing:
            states.append(dict(spec))
            existing.add(sname)


def _prune_state_decls(data: Dict[str, Any], remove_names: Set[str]) -> None:
    ts = data.get("typestates")
    if isinstance(ts, dict) and isinstance(ts.get("states"), list):
        ts["states"] = [
            entry for entry in ts["states"]
            if (
                str(entry.get("name") if isinstance(entry, dict) else entry).strip().lower()
                not in remove_names
            )
        ]
    elif isinstance(ts, list):
        data["typestates"] = [
            entry for entry in ts
            if (
                str(entry.get("name") if isinstance(entry, dict) else entry).strip().lower()
                not in remove_names
            )
        ]


def _retarget_port_states(port: Dict[str, Any], mapping: Dict[str, str], *, require_carrier: Optional[str] = None) -> None:
    carrier_l = str(port.get("carrier") or port.get("type_name") or "").strip().lower()
    raw_ts = str(port.get("typestate") or port.get("state") or "").strip().lower()
    bare_ts = raw_ts.split("::", 1)[-1] if raw_ts else ""
    if bare_ts in mapping and (require_carrier is None or carrier_l == require_carrier.lower()):
        port["typestate"] = mapping[bare_ts]

    acc = port.get("accepted_states")
    if isinstance(acc, list) and acc:
        new_acc: List[str] = []
        for s in acc:
            s_str = str(s or "").strip()
            if not s_str:
                continue
            bare = s_str.lower().split("::", 1)[-1]
            if bare in mapping and (require_carrier is None or carrier_l == require_carrier.lower()):
                new_acc.append(mapping[bare])
            else:
                new_acc.append(s_str)
        port["accepted_states"] = new_acc


def _retarget_parent_states(data: Dict[str, Any], mapping: Dict[str, str]) -> None:
    ts = data.get("typestates")
    states = ts.get("states", []) if isinstance(ts, dict) else (ts if isinstance(ts, list) else [])
    for entry in states:
        if not isinstance(entry, dict):
            continue
        ps = entry.get("parent_state")
        if ps:
            bare = str(ps).strip().lower().split("::", 1)[-1]
            if bare in mapping:
                entry["parent_state"] = mapping[bare]


def _apply_shared_dependency_cleanup(dom: str, data: Dict[str, Any]) -> None:
    if dom == "python_core":
        if isinstance(data.get("types"), dict):
            data["types"].setdefault("db_connection", {"parent": "any", "kind": "handle"})
        _ensure_state_decls(
            data,
            dom,
            [
                {
                    "name": "raw_bytes",
                    "parent_state": None,
                    "carrier_type": "bytes",
                    "description": "Opaque raw byte sequence.",
                    "properties": {},
                },
                {
                    "name": "db_connection",
                    "parent_state": None,
                    "carrier_type": "db_connection",
                    "description": "Open PEP 249 (DB-API 2.0) compatible database connection handle.",
                    "properties": {},
                },
            ],
        )
    elif dom == "sqlite3":
        if isinstance(data.get("types"), dict):
            data["types"]["connection"] = {"parent": "db_connection", "kind": "handle"}
        ts = data.get("typestates")
        states = ts.get("states", []) if isinstance(ts, dict) else (ts if isinstance(ts, list) else [])
        for entry in states:
            if isinstance(entry, dict) and str(entry.get("name") or "").strip().lower() in (
                "open",
                "in_transaction",
                "committed",
                "rolled_back",
            ):
                entry["parent_state"] = "python_core::db_connection"
    elif dom == "numpy":
        _ensure_state_decls(
            data,
            dom,
            [
                {
                    "name": "color_rgb",
                    "parent_state": "ndarray_generic",
                    "carrier_type": "ndarray",
                    "description": "3-channel ndarray arranged in RGB channel order.",
                    "properties": {"channels": 3, "color_order": "RGB"},
                },
                {
                    "name": "color_bgr",
                    "parent_state": "ndarray_generic",
                    "carrier_type": "ndarray",
                    "description": "3-channel ndarray arranged in BGR channel order.",
                    "properties": {"channels": 3, "color_order": "BGR"},
                },
                {
                    "name": "gray",
                    "parent_state": "ndarray_generic",
                    "carrier_type": "ndarray",
                    "description": "2D single-channel grayscale ndarray.",
                    "properties": {"channels": 1},
                },
            ],
        )
    elif dom == "pillow":
        _prune_state_decls(data, {"color_bgr", "color_rgb", "gray", "ndarray_generic"})
        pil_map = {
            "color_bgr": "numpy::color_bgr",
            "color_rgb": "numpy::color_rgb",
            "gray": "numpy::gray",
            "ndarray_generic": "numpy::ndarray_generic",
        }
        for cell in data.get("cells") or []:
            cid = str(cell.get("cell_id") or "")
            if cid == "PILLOW_RGB_TO_NDARRAY_BGR_ORDER":
                cell["docstring"] = "Exports a Pillow RGB Image to a 3-channel uint8 NumPy ndarray with channels in BGR order."
            elif cid == "PILLOW_FROM_NDARRAY_BGR":
                cell["docstring"] = "Ingests a 3-channel uint8 BGR-ordered NumPy ndarray into a Pillow RGB Image by reversing the channel axis."
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if not isinstance(pv, dict):
                        continue
                    _retarget_port_states(pv, pil_map, require_carrier="ndarray")
                    pv["invariants"] = [
                        inv for inv in (pv.get("invariants") or [])
                        if str(inv).strip().lower() != "cv2.image"
                    ]
    elif dom == "cv2":
        cv_map = {
            "color_bgr": "numpy::color_bgr",
            "color_rgb": "numpy::color_rgb",
            "gray": "numpy::gray",
        }
        _prune_state_decls(data, {"color_bgr", "color_rgb", "gray"})
        _retarget_parent_states(data, {"gray": "numpy::gray"})
        for cell in data.get("cells") or []:
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if not isinstance(pv, dict):
                        continue
                    _retarget_port_states(pv, cv_map)
                    pv["invariants"] = [
                        inv for inv in (pv.get("invariants") or [])
                        if str(inv).strip().lower() != "cv2.image"
                    ]
    elif dom == "matplotlib":
        _prune_state_decls(data, {"color_rgb"})
        for cell in data.get("cells") or []:
            cid = str(cell.get("cell_id") or "")
            if cid == "matplotlib_ax_imshow":
                x_port = (cell.get("inputs") or {}).get("X")
                if isinstance(x_port, dict):
                    x_port["accepted_states"] = ["numpy::color_rgb", "numpy::gray", "numpy::2d_matrix"]
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if isinstance(pv, dict):
                        _retarget_port_states(pv, {"color_rgb": "numpy::color_rgb"}, require_carrier="ndarray")
    elif dom == "pandas":
        _prune_state_decls(data, {"ndarray_generic", "db_connection"})
        if isinstance(data.get("types"), dict):
            data["types"]["Engine"] = {"parent": "db_connection", "kind": "handle"}
        for cell in data.get("cells") or []:
            cid = str(cell.get("cell_id") or "")
            ins = cell.get("inputs") or {}
            if cid == "PD_READ_SQL":
                con_port = ins.get("con") if isinstance(ins.get("con"), dict) else ins.get("port_1")
                if isinstance(con_port, dict):
                    con_port["carrier"] = "db_connection"
                    con_port["typestate"] = "python_core::db_connection"
                    con_port["accepted_states"] = []
                    con_port["description"] = "Database connection object (DB-API 2.0 or SQLAlchemy connectable)."
            elif cid == "PD_TO_SQL":
                con_port = ins.get("con") if isinstance(ins.get("con"), dict) else ins.get("port_2")
                if isinstance(con_port, dict):
                    con_port["carrier"] = "db_connection"
                    con_port["typestate"] = "python_core::db_connection"
                    con_port["accepted_states"] = []
                    con_port["description"] = "Database connection object (DB-API 2.0 or SQLAlchemy connectable)."
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if isinstance(pv, dict):
                        _retarget_port_states(pv, {"ndarray_generic": "numpy::ndarray_generic"})
    elif dom == "sklearn":
        sk_map = {
            "ndarray_generic": "numpy::ndarray_generic",
            "dataframe_2d_generic": "pandas::dataframe_2d_generic",
            "pca_components": "reduced_features",
        }
        _prune_state_decls(data, {"ndarray_generic", "dataframe_2d_generic"})
        _retarget_parent_states(data, sk_map)
        for cell in data.get("cells") or []:
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if isinstance(pv, dict):
                        _retarget_port_states(pv, sk_map)
    elif dom == "seaborn":
        sns_map = {
            "axes_initialized": "matplotlib::axes_initialized",
            "axes_with_content": "matplotlib::axes_with_content",
            "figure_saved": "matplotlib::figure_saved",
            "dataframe_2d_generic": "pandas::dataframe_2d_generic",
        }
        _prune_state_decls(
            data,
            {"axes_initialized", "axes_with_content", "figure_saved", "dataframe_2d_generic"},
        )
        _retarget_parent_states(
            data,
            {
                "axes_initialized": "matplotlib::axes_initialized",
                "axes_with_content": "matplotlib::axes_with_content",
            },
        )
        for cell in data.get("cells") or []:
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if isinstance(pv, dict):
                        _retarget_port_states(pv, sns_map)
    elif dom == "sqlite3":
        if isinstance(data.get("types"), list):
            for entry in data["types"]:
                if isinstance(entry, dict) and entry.get("name") == "connection":
                    entry["parent"] = "db_connection"
                elif isinstance(entry, dict) and entry.get("name") == "path":
                    entry["parent"] = "object"
        elif isinstance(data.get("types"), dict):
            if isinstance(data["types"].get("connection"), dict):
                data["types"]["connection"]["parent"] = "db_connection"
            if isinstance(data["types"].get("path"), dict):
                data["types"]["path"]["parent"] = "object"
    elif dom in ("requests", "cryptography"):
        _prune_state_decls(data, {"raw_bytes"})
        for cell in data.get("cells") or []:
            for direction in ("inputs", "outputs"):
                ports = cell.get(direction) or {}
                if not isinstance(ports, dict):
                    continue
                for pv in ports.values():
                    if isinstance(pv, dict):
                        _retarget_port_states(pv, {"raw_bytes": "python_core::raw_bytes"})


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
    for pname, p_clean in clean_inputs.items():
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
        clean_bp = {k: bp[k] for k in _CANONICAL_BRIDGE_KEYS if k in bp}
        clean_cell["bridge_profile"] = clean_bp

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

    out: Dict[str, Any] = {
        "domain": str(ts_dict.get("domain") or dom).strip().lower(),
        "states": clean_states,
        "transitions": list(ts_dict.get("transitions") or []),
        "initial_state": ts_dict.get("initial_state"),
        "terminal_states": list(ts_dict.get("terminal_states") or []),
    }
    return out


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

    # Strict schema enforcement prior to serialization
    TreeSchema(**clean_tree)
    return clean_tree


def neutralize_all() -> None:
    files = sorted(glob.glob(str(ROOT / "trees" / "*.json")))
    fixture_files = sorted(glob.glob(str(ROOT / "benchmarks" / "fixtures" / "*" / "*.json")))
    seen: Set[Path] = set()
    all_files: List[Path] = []
    for f in files + fixture_files:
        p = Path(f).resolve()
        if p not in seen:
            seen.add(p)
            all_files.append(p)

    parsed: List[Tuple[Path, str, Dict[str, Any]]] = []
    known_domains: Set[str] = set()
    dom_dep_counts: Dict[str, Dict[str, int]] = {}

    for p in all_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        dom = str(data.get("domain") or p.stem).strip().lower()
        known_domains.add(dom)
        counts: Dict[str, int] = {}
        for c in data.get("cells") or []:
            deps = list(c.get("dependencies") or ()) + list(c.get("imports") or ())
            for r in _parse_dep_roots(deps):
                counts[r] = counts.get(r, 0) + 1
        dom_dep_counts[dom] = counts
        parsed.append((p, dom, data))

    pkg_to_dom: Dict[str, str] = {d: d for d in known_domains}
    for dom, counts in dom_dep_counts.items():
        foreign_only = {r: cnt for r, cnt in counts.items() if r not in pkg_to_dom}
        if foreign_only:
            top_pkg = max(foreign_only, key=lambda r: (foreign_only[r], r))
            if counts.get(dom, 0) == 0 and foreign_only[top_pkg] >= max(1, sum(foreign_only.values()) // 2):
                pkg_to_dom[top_pkg] = dom

    updated_count = 0
    total_cells = 0
    prepared: List[Tuple[Path, str, Dict[str, Any]]] = []
    for p, dom, data in parsed:
        _apply_shared_dependency_cleanup(dom, data)
        is_core = bool(data.get("is_core", False))
        foreign_imports: Set[str] = set()
        if not is_core and "python_core" in known_domains:
            foreign_imports.add("python_core")
        if not is_core:
            for existing_imp in data.get("imports") or ():
                imp_l = str(existing_imp or "").strip().lower()
                if imp_l and imp_l != dom and imp_l in known_domains:
                    foreign_imports.add(imp_l)

        cells = data.get("cells") or []
        for cell in cells:
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

        if dom == "pygame":
            data["runtime_setup"] = {
                "env_vars": {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
                "pre_exec_hooks": ["pygame.sysfont.initsysfonts"],
                "notes": "Headless SDL display/audio drivers and read-only sysfont index preload.",
            }

        imports_list = [] if is_core else sorted(foreign_imports)
        clean_tree = _canonicalize_tree(data, dom, imports_list)
        prepared.append((p, dom, clean_tree))

    trees_dir = (ROOT / "trees").resolve()
    global_dom_map: Dict[str, Dict[str, Any]] = {}
    dir_dom_map: Dict[Tuple[Path, str], Dict[str, Any]] = {}
    for p, dom, ct in prepared:
        dir_dom_map[(p.resolve().parent, dom)] = ct
        if p.resolve().parent == trees_dir:
            global_dom_map[dom] = ct
    for p, dom, ct in prepared:
        if dom not in global_dom_map:
            global_dom_map[dom] = ct

    for p, dom, clean_tree in prepared:
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
        for ct in closure_trees:
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

        state_ndim_map: Dict[str, int] = {}
        for ct in closure_trees:
            ct_dom = str(ct.get("domain") or "").strip().lower()
            ts_b = ct.get("typestates")
            if isinstance(ts_b, dict):
                ts_dom = str(ts_b.get("domain") or ct_dom).strip().lower()
                for s_entry in ts_b.get("states") or ():
                    if isinstance(s_entry, dict) and s_entry.get("name"):
                        nd = (s_entry.get("properties") or {}).get("ndim")
                        if isinstance(nd, int) and not isinstance(nd, bool):
                            sname = str(s_entry["name"]).strip().lower()
                            state_ndim_map[f"{ts_dom}::{sname}"] = nd
                            if ts_dom == dom:
                                state_ndim_map[sname] = nd

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
                if str(pv.get("carrier") or "").strip().lower() == "ndarray":
                    lay = pv.get("layout")
                    if isinstance(lay, dict) and not lay.get("rank_polymorphic"):
                        st_raw = str(pv.get("typestate") or "").strip().lower()
                        st_qual = st_raw if "::" in st_raw else f"{dom}::{st_raw}"
                        exp_ndim = state_ndim_map.get(st_raw)
                        if exp_ndim is None:
                            exp_ndim = state_ndim_map.get(st_qual)
                        if exp_ndim is not None:
                            lay["rank"] = exp_ndim
            for pv in outs.values():
                if isinstance(pv, dict) and str(pv.get("resource_mode") or "").lower() == "acquire":
                    if not _is_releasable(pv.get("carrier")):
                        pv["resource_mode"] = "borrow" if (has_handle_in or has_releasable_out) else "value"

        TreeSchema(**clean_tree)
        p.write_text(json.dumps(clean_tree, indent=2) + "\n", encoding="utf-8")
        updated_count += 1
        total_cells += len(clean_tree["cells"])
        print(f"[neutralize] {p.relative_to(ROOT)}: imports={clean_tree['imports']} cells={len(clean_tree['cells'])}")

    print(f"\n[neutralize] Completed: {updated_count} domain tree files normalized ({total_cells} total validated cells).")


if __name__ == "__main__":
    neutralize_all()
