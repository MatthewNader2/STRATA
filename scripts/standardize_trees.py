#!/usr/bin/env python3
"""
scripts/standardize_trees.py - Domain-Agnostic 5-Axis + Relational/Effect Tree Migrator.

Uniformly upgrades all domain files in `trees/*.json` to the canonical 5-axis
contract without hardcoding any domain names, cell IDs, or symbol lists.
All semantic vocabularies (`role_semantics`, `typestates`) are read dynamically
from the trees themselves, and existing explicit tree annotations are preserved.
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from schema import TreeSchema  # noqa: E402


def _collect_role_vocabularies(loaded: List[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Set[str]]:
    """Aggregates `role_semantics` vocabularies dynamically from all loaded trees."""
    vocab: Dict[str, Set[str]] = {}
    for _, data in loaded:
        rs = data.get("role_semantics")
        if not isinstance(rs, dict):
            continue
        for key, vals in rs.items():
            if isinstance(vals, list):
                vocab.setdefault(key, set()).update(str(v).strip().lower() for v in vals if isinstance(v, str) and v.strip())
        ts = data.get("typestates")
        if isinstance(ts, dict):
            for st_obj in ts.get("states") or []:
                if isinstance(st_obj, dict) and st_obj.get("is_top"):
                    sn = str(st_obj.get("name") or "").strip().lower()
                    if sn:
                        vocab.setdefault("top_states", set()).add(sn)
    return vocab


def _compute_semantics(dom: str, state: str, port: Dict[str, Any], top_states: Set[str]) -> str:
    existing = str(port.get("semantics") or "").strip().lower()
    if existing:
        return existing
    st = str(state or "").strip().lower()
    if not st or st in top_states:
        return "any"
    if "::" in st or "." in st:
        return st
    return f"{dom}.{st}"


def _compute_typestate(dom: str, state: str, port: Dict[str, Any], top_states: Set[str]) -> str:
    existing = str(port.get("typestate") or "").strip().lower()
    if existing:
        return existing
    st = str(state or "").strip().lower()
    if not st or st in top_states:
        return "any"
    if "::" in st:
        return st
    return f"{dom}::{st}"


def _compute_layout(
    port: Dict[str, Any],
    dom_ranks: Dict[str, Set[int]],
    is_poly_op: bool = False,
    polymorphic_sks: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    existing = port.get("layout")
    if isinstance(existing, dict) and existing:
        return existing
    sc = port.get("shape_contract") if isinstance(port.get("shape_contract"), dict) else {}
    sk = str(port.get("structure_kind") or "").lower()
    tn = str(port.get("type_name") or "").lower()
    st = str(port.get("state") or "").lower()
    rank = sc.get("ndim", port.get("rank"))
    if rank is None and st in dom_ranks and len(dom_ranks[st]) == 1:
        rank = next(iter(dom_ranks[st]))
    if rank is None and sk == "scalar":
        rank = 0
    poly = bool(sc.get("rank_polymorphic", False)) or is_poly_op
    if rank is None and polymorphic_sks and sk in polymorphic_sks:
        poly = True
    if sk == "handle" and rank is None and not sc and port.get("element_type") in (None, "opaque"):
        return None
    return {
        "rank": "R" if is_poly_op else rank,
        "shape": sc.get("shape"),
        "dtype": "D" if is_poly_op else port.get("element_type"),
        "memory_order": "any",
        "rank_polymorphic": poly,
        "channel_count": sc.get("channel_count"),
        "channel_axis": sc.get("channel_axis"),
        "length": sc.get("len"),
    }


def _compute_invariants(port: Dict[str, Any], structural_tags: Set[str]) -> List[str]:
    invs: List[str] = list(port.get("invariants") or [])
    for q in port.get("qualifiers") or []:
        if isinstance(q, (list, tuple)):
            qs = ".".join(str(x).lower() for x in q)
        elif isinstance(q, str) and q.strip():
            qs = q.strip().lower()
        else:
            continue
        if qs and qs not in structural_tags:
            invs.append(qs)
    vc = port.get("value_constraints")
    if isinstance(vc, dict):
        for k, v in sorted(vc.items()):
            invs.append(f"{str(k).lower()}={v}")
    return list(dict.fromkeys(invs))


def _compute_bridge_profile(cell: Dict[str, Any]) -> Dict[str, Any]:
    existing = cell.get("bridge_profile")
    if isinstance(existing, dict) and existing:
        return existing
    ins = list((cell.get("inputs") or {}).values())
    outs = list((cell.get("outputs") or {}).values())
    p_in = ins[0] if ins else {}
    p_out = outs[0] if outs else {}
    in_c = str(p_in.get("carrier") or p_in.get("type_name") or "").lower()
    out_c = str(p_out.get("carrier") or p_out.get("type_name") or "").lower()
    in_sem = str(p_in.get("semantics") or "").lower()
    out_sem = str(p_out.get("semantics") or "").lower()
    in_lay = p_in.get("layout") if isinstance(p_in.get("layout"), dict) else {}
    out_lay = p_out.get("layout") if isinstance(p_out.get("layout"), dict) else {}

    deltas: List[str] = []
    if in_c != out_c:
        deltas.append("carrier")
    if in_sem != out_sem and out_sem != "any":
        deltas.append("semantics")
    if in_lay.get("rank") != out_lay.get("rank") or in_lay.get("dtype") != out_lay.get("dtype"):
        deltas.append("layout")
    if not deltas:
        deltas.append("carrier")

    optic_r = str(cell.get("optic_role") or "").lower()
    in_rk = in_lay.get("rank") if isinstance(in_lay.get("rank"), int) else None
    out_rk = out_lay.get("rank") if isinstance(out_lay.get("rank"), int) else None
    rank_reduced = (in_rk is not None and out_rk is not None and out_rk < in_rk and optic_r != "getter")
    is_lossy = bool(cell.get("lossy")) or rank_reduced
    cost_tier = 10.0 if is_lossy else 1.0
    return {
        "is_bridge": True,
        "delta_axes": deltas,
        "cost_tier": cost_tier,
        "lossy": is_lossy,
        "requires_intent": bool(cell.get("requires_intent", is_lossy)),
    }


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "trees" / "*.json")))
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    state_usage_by_dom: Dict[str, Set[str]] = {}
    state_defs_count: Dict[str, int] = collections.Counter()
    state_ranks_by_dom: Dict[str, Dict[str, Set[int]]] = {}
    state_poly_by_dom: Dict[str, Set[str]] = {}

    # Pass 1: Load and record declared usage & metadata across all domains
    for fp in files:
        p = Path(fp)
        data = json.loads(p.read_text(encoding="utf-8"))
        dom = str(data.get("domain") or p.stem).strip().lower()
        used: Set[str] = set()
        dom_ranks: Dict[str, Set[int]] = {}
        dom_poly: Set[str] = set()
        for cell in data.get("cells") or []:
            for io in ("inputs", "outputs"):
                ports = cell.get(io) or {}
                if isinstance(ports, dict):
                    for pv in ports.values():
                        if not isinstance(pv, dict):
                            continue
                        st_v = str(pv.get("state") or "").strip().lower()
                        if st_v:
                            used.add(st_v)
                            if io == "outputs":
                                sc_v = pv.get("shape_contract") if isinstance(pv.get("shape_contract"), dict) else {}
                                rk_v = sc_v.get("ndim", pv.get("rank"))
                                if isinstance(rk_v, int) and not sc_v.get("rank_polymorphic", False):
                                    dom_ranks.setdefault(st_v, set()).add(rk_v)
                        if pv.get("parent_state"):
                            used.add(str(pv["parent_state"]).strip().lower())
                        for acc in pv.get("accepted_states") or []:
                            used.add(str(acc).strip().lower())

        ts_block = data.get("typestates") or {}
        states_list = ts_block.get("states", []) if isinstance(ts_block, dict) else []
        parent_lookup = {
            str(s.get("name")).strip().lower(): str(s.get("parent_state") or "").strip().lower()
            for s in states_list
            if isinstance(s, dict) and s.get("name")
        }
        for s in states_list:
            if isinstance(s, dict) and s.get("name"):
                sn = str(s["name"]).strip().lower()
                props = s.get("properties")
                if isinstance(props, dict):
                    if props.get("is_rank_polymorphic"):
                        dom_poly.add(sn)
                    rk = props.get("ndim")
                    if isinstance(rk, int) and not props.get("is_rank_polymorphic", False):
                        dom_ranks.setdefault(sn, set()).add(rk)
                state_defs_count[sn] += 1

        state_ranks_by_dom[dom] = dom_ranks
        state_poly_by_dom[dom] = dom_poly
        expanded = set(used)
        for st in list(used):
            curr = parent_lookup.get(st, "")
            seen: Set[str] = set()
            while curr and curr not in seen:
                seen.add(curr)
                expanded.add(curr)
                curr = parent_lookup.get(curr, "")
        state_usage_by_dom[dom] = expanded
        loaded.append((p, data))

    vocab = _collect_role_vocabularies(loaded)
    top_states = vocab.get("top_states", set())
    hof_roles = vocab.get("hof_roles", set())
    receiver_roles = vocab.get("receiver_roles", set())
    model_output_roles = vocab.get("model_output_roles", set())
    structural_tags = vocab.get("structural_kinds", set())
    polymorphic_sks = vocab.get("polymorphic_structure_kinds", set())

    # Pass 2: Standardize each tree uniformly without domain/cell hardcodes
    for tree_path, data in loaded:
        dom = str(data.get("domain") or tree_path.stem).strip().lower()
        used_dom = state_usage_by_dom[dom]

        # 2a. Prune unused copy-pasted colliding typestates
        ts_block = data.get("typestates")
        if isinstance(ts_block, dict) and isinstance(ts_block.get("states"), list):
            kept_states = []
            for s in ts_block["states"]:
                if isinstance(s, dict) and s.get("name"):
                    sn = str(s["name"]).strip().lower()
                    if sn not in used_dom and state_defs_count.get(sn, 0) > 1:
                        continue
                kept_states.append(s)
            ts_block["states"] = kept_states

        # 2b. Standardize each cell & port preserving declared metadata
        for cell in data.get("cells") or []:
            cid = str(cell.get("cell_id") or "")
            cell.pop("source_priority", None)

            poly_prop = cell.get("polymorphic_propagation")
            is_poly_op = isinstance(poly_prop, dict) and bool(poly_prop.get("preserves_layout", False))

            if "effects" not in cell:
                cell["effects"] = list(cell.get("postconditions") or [])
            if "mutation_type" not in cell:
                role = str(cell.get("node_role") or cell.get("role") or "").lower()
                if role == "sink" or bool(cell.get("effects")):
                    cell["mutation_type"] = "io"
                elif role == "combinator":
                    cell["mutation_type"] = "combinator"
                else:
                    cell["mutation_type"] = "pure"

            is_oo_method = (str(cell.get("node_type") or "").lower() == "method" and "." in cid)
            owner_qual = cid.rsplit(".", 1)[0].lower() if is_oo_method else None
            owner_short = owner_qual.rsplit(".", 1)[-1] if owner_qual else None

            rels: List[Dict[str, Any]] = list(cell.get("relations") or [])
            seen_rels = {(r.get("kind"), r.get("source_port"), r.get("target_port")) for r in rels if isinstance(r, dict)}

            dom_ranks_map = state_ranks_by_dom.get(dom, {})
            dom_poly_set = state_poly_by_dom.get(dom, set())

            for io in ("inputs", "outputs"):
                ports = cell.get(io) or {}
                if not isinstance(ports, dict):
                    continue
                for pname, p in ports.items():
                    if not isinstance(p, dict):
                        continue

                    tn = str(p.get("type_name") or "any")
                    st = str(p.get("state") or "any")
                    p_role = str(p.get("port_role") or p.get("role") or "").lower()

                    if not p.get("carrier"):
                        p["carrier"] = tn

                    if not p.get("nominal_type") and owner_qual and (
                        (io == "outputs" and (p_role in model_output_roles or tn.lower() == owner_short))
                        or (io == "inputs" and (pname == "self" or p_role in receiver_roles))
                    ):
                        p["nominal_type"] = owner_qual

                    lay = _compute_layout(
                        p,
                        dom_ranks_map,
                        is_poly_op=is_poly_op,
                        polymorphic_sks=polymorphic_sks,
                    )
                    if isinstance(lay, dict) and p.get("accepted_states"):
                        acc_low = [str(st).strip().lower()] + [
                            str(a).strip().lower() for a in (p.get("accepted_states") or [])
                        ]
                        obs_ranks: Set[int] = set()
                        if isinstance(lay.get("rank"), int):
                            obs_ranks.add(lay["rank"])
                        has_poly_state = any(s in dom_poly_set for s in acc_low)
                        for s in acc_low:
                            obs_ranks |= dom_ranks_map.get(s, set())
                        if len(obs_ranks) > 1 or has_poly_state:
                            lay["rank"] = None
                            lay["rank_polymorphic"] = True
                    p["layout"] = lay
                    p["semantics"] = _compute_semantics(dom, st, p, top_states)
                    p["typestate"] = _compute_typestate(dom, st, p, top_states)
                    p["invariants"] = _compute_invariants(p, structural_tags)
                    if p["semantics"] == "any" and p.get("accepted_states"):
                        p.pop("accepted_states", None)

                    if p_role in hof_roles and "effect_suspended" not in p:
                        p["effect_suspended"] = True

                    # Harvest relational constraints into structured cell.relations
                    if io == "inputs":
                        if p.get("complement_of_port"):
                            key = ("complement_of", pname, p["complement_of_port"])
                            if key not in seen_rels:
                                rels.append({"kind": "complement_of", "source_port": pname, "target_port": p["complement_of_port"]})
                                seen_rels.add(key)
                        if p.get("disjoint_from_port"):
                            key = ("disjoint_from", pname, p["disjoint_from_port"])
                            if key not in seen_rels:
                                rels.append({"kind": "disjoint_from", "source_port": pname, "target_port": p["disjoint_from_port"]})
                                seen_rels.add(key)
                        if p.get("binds"):
                            p_role_b = p.get("binds")
                            for other_name, other_p in ports.items():
                                if isinstance(other_p, dict) and other_name != pname and other_p.get("binds") == "data_carrier" and p_role_b == "column_key":
                                    key = ("key_of", pname, other_name)
                                    if key not in seen_rels:
                                        rels.append({"kind": "key_of", "source_port": pname, "target_port": other_name, "role": p_role_b})
                                        seen_rels.add(key)

            if rels:
                cell["relations"] = rels

            role = str(cell.get("node_role") or cell.get("role") or "").lower()
            if role == "bridge":
                cell["bridge_profile"] = _compute_bridge_profile(cell)

        TreeSchema(**data)
        tree_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Standardized {tree_path.name} ({len(data.get('cells') or [])} cells)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
