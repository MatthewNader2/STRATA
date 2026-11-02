"""
tools/migrate_tree_roles.py
One-time data migration: moves domain knowledge OUT of engine hardcodes and
INTO the tree data files where it belongs (declarative, per-domain, editable
without engine changes).

What it does (idempotent):
  1. port_role declarations:
     - sklearn-style ports named X*  -> port_role "feature_input"
     - sklearn-style ports named y*  -> port_role "target_input"
     - estimator/model-typed outputs -> port_role "model_sink"
     - primary DataFrame/table-typed inputs on transformer cells
                                       -> port_role "data_input"
     (The engine's PortSignature.derived_role name/keyword table is retired;
      roles are consumed from these declarations first.)
  2. effect declarations:
     - cells whose postconditions already declare property has_nans == False
       get effect "cleans_missing" (declared channel for the verification
       contract; the engine's expression-substring sniffing is retired).
     - cells whose postconditions already declare is_deduped == True get
       effect "deduplicates".
     - cv2-style annotation cells (tensor in + tensor out + drawing keywords)
       get effect "draws_annotation" (replaces engine-side port-name sniffing).
  3. endable declarations:
     - stage 3 / sink / terminal / evaluator / display nodes get endable: true
       (planner terminal validation consumes Cell.is_endable only).
  4. typestate vocabulary migration:
     - every state referenced by any cell port is declared in its domain
       tree's typestates.states block (parent links mirror the former engine
       bootstrap), with carrier_type + universal properties where derivable.
  5. carrier type migration:
     - dataframe/series/matlike carrier registrations moved from the engine
       bootstrap into pandas.json / cv2.json types blocks.
  6. adds the DataFrame column-projection cell used by generic feature
     selection (replaces the engine's DATAFRAME_TO_NUMPY special case).

Usage:
    python tools/migrate_tree_roles.py [--trees-dir trees] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

DRAW_VERBS = ("draw", "put_text", "circle", "rectangle", "polylines", "line",
              "arrowed", "ellipse", "contour")

# Universal (domain-agnostic) typestate taxonomy retained by the engine
# bootstrap. Everything else found in engine code was domain vocabulary and
# is migrated here into the owning tree's typestates block.
UNIVERSAL_EGRESS_CHAIN = {
    "saved": "written_to_disk",
    "figure_saved": "written_to_disk",
    "saved_npy": "written_to_disk",
    "saved_npz": "written_to_disk",
    "written_to_disk": "exported",
    "exported": None,
    "default": None,
}

# State -> (parent, carrier_type) migrated from the retired engine bootstrap.
# Keys may appear in several domains; each domain tree declares the subset
# its cells use.
MIGRATED_STATES: Dict[str, Dict[str, Any]] = {
    # --- machine-learning partition / feature states (sklearn) ---
    "split_train_features": {"parent": "unscaled_features", "carrier_type": "ndarray"},
    "split_test_features": {"parent": "unscaled_features", "carrier_type": "ndarray"},
    "unscaled_features": {"parent": "feature_matrix", "carrier_type": "ndarray"},
    "scaled_features": {"parent": "feature_matrix", "carrier_type": "ndarray"},
    "imputed_features": {"parent": "unscaled_features", "carrier_type": "ndarray"},
    "encoded_features": {"parent": "unscaled_features", "carrier_type": "ndarray"},
    "reduced_features": {"parent": "feature_matrix", "carrier_type": "ndarray"},
    "feature_matrix": {"parent": "ndarray_generic", "carrier_type": "ndarray"},
    "split_train_targets": {"parent": "target_vector", "carrier_type": "ndarray"},
    "split_test_targets": {"parent": "target_vector", "carrier_type": "ndarray"},
    "target_vector": {"parent": "ndarray_generic", "carrier_type": "ndarray"},
    "fit_regressor": {"parent": "fit_estimator", "carrier_type": "Estimator"},
    "fit_classifier": {"parent": "fit_estimator", "carrier_type": "Estimator"},
    "fit_clusterer": {"parent": "fit_estimator", "carrier_type": "Estimator"},
    "fit_transformer": {"parent": "fit_estimator", "carrier_type": "Estimator"},
    "fit_estimator": {"parent": "trained", "carrier_type": "Estimator"},
    "unfit_estimator": {"parent": "default", "carrier_type": "Estimator"},
    "trained": {"parent": None, "carrier_type": "Estimator"},
    "raw_dataset": {"parent": "dataframe_2d_generic", "carrier_type": "DataFrame"},
    "dataframe_2d_generic": {"parent": None, "carrier_type": "DataFrame"},
    # --- tabular states (pandas) ---
    "series_cleaned": {"parent": "series_numeric", "carrier_type": "Series"},
    "series_numeric": {"parent": "series_raw", "carrier_type": "Series"},
    "series_raw": {"parent": None, "carrier_type": "Series"},
    "cleaned": {"parent": "raw_dataset", "carrier_type": "DataFrame"},
    "normalized": {"parent": "cleaned", "carrier_type": "DataFrame"},
    "transformed": {"parent": "cleaned", "carrier_type": "DataFrame"},
    "deduped": {"parent": "cleaned", "carrier_type": "DataFrame"},
    "filtered": {"parent": "cleaned", "carrier_type": "DataFrame"},
    "indexed": {"parent": "cleaned", "carrier_type": "DataFrame"},
    "numeric_only": {"parent": "cleaned", "carrier_type": "DataFrame"},
    # --- computer-vision states (cv2) ---
    "binary": {"parent": "gray", "carrier_type": "ndarray",
               "properties": {"channels": 1}},
    "grayscale": {"parent": "gray", "carrier_type": "ndarray",
                  "properties": {"channels": 1}},
    "gray": {"parent": None, "carrier_type": "ndarray",
             "properties": {"channels": 1}},
    "computed_threshold": {"parent": "binary", "carrier_type": "ndarray",
                           "properties": {"channels": 1}},
    "blurred": {"parent": "gray", "carrier_type": "ndarray"},
    "edge_map": {"parent": "gray", "carrier_type": "ndarray",
                 "properties": {"channels": 1}},
    "dilated": {"parent": "binary", "carrier_type": "ndarray",
                "properties": {"channels": 1}},
    "eroded": {"parent": "binary", "carrier_type": "ndarray",
               "properties": {"channels": 1}},
    "morphology_processed": {"parent": "binary", "carrier_type": "ndarray",
                             "properties": {"channels": 1}},
    "contours": {"parent": "collection", "carrier_type": "List[Contour]"},
    "source_identifier": {"parent": None, "carrier_type": "str"},
    "valid_path": {"parent": "source_identifier", "carrier_type": "str"},
    "file_path": {"parent": "source_identifier", "carrier_type": "str"},
    "file_path_str": {"parent": "source_identifier", "carrier_type": "str"},
    "source_path": {"parent": "source_identifier", "carrier_type": "str"},
    "destination_path": {"parent": "source_identifier", "carrier_type": "str"},
    # --- generic helper states used across domain trees ---
    "sort_column": {"parent": None, "carrier_type": "str"},
    "limit_count": {"parent": None, "carrier_type": "int"},
    "order_flag": {"parent": None, "carrier_type": "bool"},
    "column_projection": {"parent": None, "carrier_type": "List[str]"},
    "hyperparameter": {"parent": None, "carrier_type": "numeric"},
}

# Carrier registrations migrated out of the retired engine bootstrap.
MIGRATED_CARRIERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "pandas": {
        "DataFrame": {"parents": ["table", "array-like"]},
        "Series": {"parents": ["table", "array-like"]},
    },
    "cv2": {
        "MatLike": {"parent": "tensor"},
    },
}


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: str, data: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _ensure_typestates(data: Dict[str, Any]) -> Dict[str, Any]:
    ts = data.get("typestates")
    if not isinstance(ts, dict):
        ts = {"domain": data.get("domain", "generic"), "states": []}
        data["typestates"] = ts
    if not isinstance(ts.get("states"), list):
        ts["states"] = []
    return ts


def _declared_state_names(data: Dict[str, Any]) -> set:
    names = set()
    for s in _ensure_typestates(data)["states"]:
        if isinstance(s, dict) and s.get("name"):
            names.add(str(s["name"]))
        elif isinstance(s, str):
            names.add(s)
    for c in data.get("cells", []):
        for direction in ("inputs", "outputs"):
            for p in (c.get(direction) or {}).values():
                st = p.get("state") if isinstance(p, dict) else None
                if st and st not in ("any", "*"):
                    names.add(str(st))
    return names


def migrate_typestates(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    """Declare every referenced state in the tree's typestates block."""
    ts = _ensure_typestates(data)
    existing = {s.get("name") if isinstance(s, dict) else s for s in ts["states"]}
    index = {s["name"]: s for s in ts["states"] if isinstance(s, dict) and s.get("name")}

    for name in sorted(_declared_state_names(data)):
        if name in ("any", "*", "raw"):
            continue
        meta = MIGRATED_STATES.get(name)
        if meta is None:
            continue  # already fully declared by the tree author
        if name in existing:
            entry = index.get(name)
            # Backfill parent/properties if the tree declared it bare
            if entry is not None:
                changed = False
                if not entry.get("parent_state") and meta.get("parent"):
                    entry["parent_state"] = meta["parent"]
                    changed = True
                if not entry.get("carrier_type") and meta.get("carrier_type"):
                    entry["carrier_type"] = meta["carrier_type"]
                    changed = True
                if meta.get("properties") and not entry.get("properties"):
                    entry["properties"] = meta["properties"]
                    changed = True
                if changed:
                    stats["typestates_backfilled"] += 1
            continue
        entry = {
            "name": name,
            "parent_state": meta.get("parent"),
            "carrier_type": meta.get("carrier_type"),
            "description": f"Declared by data migration (formerly engine bootstrap state '{name}').",
        }
        if meta.get("properties"):
            entry["properties"] = meta["properties"]
        ts["states"].append(entry)
        stats["typestates_added"] += 1


def migrate_carriers(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    """Register migrated carrier types into the tree's types block."""
    domain = data.get("domain", "")
    additions = MIGRATED_CARRIERS.get(domain, {})
    if not additions:
        return
    types = data.setdefault("types", {})
    for name, meta in additions.items():
        if name in types:
            continue
        types[name] = dict(meta)
        stats["carriers_added"] += 1


def _iter_ports(cell: Dict[str, Any]):
    for direction in ("inputs", "outputs"):
        ports = cell.get(direction) or {}
        if isinstance(ports, dict):
            for pname, pmeta in ports.items():
                if isinstance(pmeta, dict):
                    yield direction, pname, pmeta


def migrate_port_roles(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    for cell in data.get("cells", []):
        cid = str(cell.get("cell_id", ""))
        low = cid.lower()
        for direction, pname, pmeta in _iter_ports(cell):
            if pmeta.get("port_role"):
                continue
            plow = pname.lower()
            tname = str(pmeta.get("type_name", "")).lower()
            state = str(pmeta.get("state", "")).lower()
            role = None
            if direction == "inputs":
                if plow == "x" or plow.startswith("x_") or plow.startswith("x_train"):
                    role = "feature_input"
                elif plow == "y" or plow.startswith("y_") or plow.startswith("y_train"):
                    role = "target_input"
                elif "dataframe" in tname or "table" in tname or state == "raw_dataset":
                    role = "data_input"
                elif "estimator" in tname and pmeta.get("required"):
                    role = "model_input"
            else:  # outputs
                if "estimator" in tname or "model" in tname or state.startswith("fit_"):
                    role = "model_sink"
            if role:
                pmeta["port_role"] = role
                stats["port_roles"] += 1


def migrate_effects(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    for cell in data.get("cells", []):
        effects = cell.get("effects")
        if not isinstance(effects, list):
            effects = []
            cell["effects"] = effects
        eff_set = {str(e) for e in effects}
        posts = cell.get("postconditions") or []

        def _post(prop: str, value: Any) -> bool:
            for p in posts:
                if not isinstance(p, dict):
                    continue
                if p.get("property") == prop and p.get("value") == value:
                    return True
            return False

        if _post("has_nans", False) and "cleans_missing" not in eff_set:
            effects.append("cleans_missing")
            stats["effects"] += 1
        if _post("is_deduped", True) and "deduplicates" not in eff_set:
            effects.append("deduplicates")
            stats["effects"] += 1

        # Annotation cells: tensor carrier in AND out, with a declared drawing
        # verb in the cell identity (offline curation heuristic; the outcome is
        # now a declared effect consumed generically by the engine).
        cid = str(cell.get("cell_id", "")).lower()
        kws = {str(k).lower() for k in (cell.get("keywords") or [])}
        kws |= {t for t in cid.replace("_", " ").split()}
        is_draw = any(v in kws for v in DRAW_VERBS)
        # Annotation cells: tensor carrier in AND out, with a declared drawing
        # verb in the cell identity (offline curation heuristic; the outcome is
        # now a declared effect consumed generically by the engine).
        tensor_t = {"ndarray", "matlike", "tensor", "image"}
        ins_tensor = False
        outs_tensor = False
        for direction, pname, pmeta in _iter_ports(cell):
            tn = str(pmeta.get("type_name", "")).lower()
            if tn in tensor_t or "tensor" in str(pmeta.get("abstract_type", "")).lower():
                if direction == "inputs" and pmeta.get("required", True) and pname.lower() in ("image", "img", "src", "input", "canvas"):
                    ins_tensor = True
                elif direction == "outputs":
                    outs_tensor = True
        if is_draw and ins_tensor and outs_tensor and "draws_annotation" not in {str(e) for e in effects}:
            effects.append("draws_annotation")
            stats["effects"] += 1


def migrate_endable(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    for cell in data.get("cells", []):
        if cell.get("endable") is not None:
            continue
        stage = cell.get("stage")
        role = str(cell.get("node_role", "")).lower()
        if stage == 3 or role in ("sink", "terminal", "evaluator", "consumer",
                                  "display", "visualizer", "cleanup", "destructor"):
            cell["endable"] = True
            stats["endable"] += 1


PROJECTION_CELL = {
    "cell_id": "PD_DATAFRAME_PROJECT_TO_NUMPY",
    "stage": 2,
    "keywords": ["to_numpy", "values", "array_bridge", "columns", "project", "projection", "feature", "features"],
    "cell_type": "micro",
    "node_type": "function",
    "node_role": "transformer",
    "mutation_type": "pure",
    "is_context_manager": False,
    "verified": True,
    "source_priority": 100,
    "is_public": True,
    "endable": False,
    "docstring": (
        "Project the given columns of a DataFrame and convert them to a NumPy "
        "ndarray. Composes column projection with the array bridge so feature "
        "selection for supervised tasks is expressed by declared lattice "
        "morphisms rather than engine-side special cases."
    ),
    "dependencies": ["pandas as pd"],
    "semantic_tags": ["tabular", "projection", "array_bridge"],
    "inputs": {
        "port_0": {
            "type_name": "DataFrame", "state": "cleaned", "qualifiers": [],
            "default_value": None, "description": "Source DataFrame.",
            "domain": "pandas", "required": True, "abstract_type": "table",
            "enum_values": None, "param_kind": "positional_or_keyword",
            "value_constraints": None, "shape_contract": None,
        },
        "columns": {
            "type_name": "List[str]", "state": "column_projection", "qualifiers": [],
            "default_value": None,
            "description": "Column names to project before array conversion.",
            "domain": "pandas", "required": True, "abstract_type": None,
            "enum_values": None, "param_kind": "positional_or_keyword",
            "value_constraints": None, "shape_contract": None,
        },
    },
    "outputs": {
        "port_0": {
            "type_name": "ndarray", "state": "ndarray_generic", "qualifiers": [],
            "default_value": None, "description": "Projected feature matrix.",
            "domain": "pandas", "required": True, "abstract_type": "tensor",
            "enum_values": None, "param_kind": "positional_or_keyword",
            "value_constraints": None, "shape_contract": None,
        }
    },
    "slots": ["output_var", "df", "columns"],
    "bound_slots": {},
    "code_template": "{output_var} = {df}[{columns}].to_numpy()",
    "edges": [], "preconditions": [], "postconditions": [], "effects": [],
    "type_vars": [], "raises": [],
    "domain_name": "pandas",
}


def add_projection_cell(data: Dict[str, Any], stats: Dict[str, int]) -> None:
    if data.get("domain") != "pandas":
        return
    ids = {c.get("cell_id") for c in data.get("cells", [])}
    if "PD_DATAFRAME_PROJECT_TO_NUMPY" in ids:
        return
    data.setdefault("cells", []).append(copy.deepcopy(PROJECTION_CELL))
    stats["cells_added"] += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees-dir", default="trees")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats: Dict[str, int] = {
        "typestates_added": 0, "typestates_backfilled": 0, "carriers_added": 0,
        "port_roles": 0, "effects": 0, "endable": 0, "cells_added": 0,
    }

    files = sorted(
        os.path.join(args.trees_dir, f)
        for f in os.listdir(args.trees_dir)
        if f.endswith(".json")
    )
    for path in files:
        data = _load(path)
        domain = data.get("domain", os.path.basename(path))
        before = json.dumps(data, sort_keys=True)
        migrate_typestates(data, stats)
        migrate_carriers(data, stats)
        migrate_port_roles(data, stats)
        migrate_effects(data, stats)
        migrate_endable(data, stats)
        add_projection_cell(data, stats)
        after = json.dumps(data, sort_keys=True)
        if before != after:
            _save(path, data, args.dry_run)
            print(f"[migrated] {path} (domain={domain})")
        else:
            print(f"[unchanged] {path}")

    print("\nMigration summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
