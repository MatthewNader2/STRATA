"""
tools/consolidate_domains.py
STRATA Domain Consolidation & Audit Tool.

NOTE: As of Phase 4+, the canonical live format is exactly ONE TreeSchema JSON
file per domain under `trees/{domain}.json`. DB compilation is handled by
`tools/compile_trees.py` (via `src.cli.cmd_compile`), and live harvesting is
handled by `src/harvester.py::IntelligentHarvester`.

This script consolidates and verifies single-file domain trees under `trees/{domain}.json`
without destructively overwriting valid trees when legacy staging directories are absent.
"""

from __future__ import annotations
import json
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import CellSchema, TreeSchema, PortSchema
from src.tokenizer import CellTokenizer

CONTAINER_TYPES = {
    "pandas": "DataFrame",
    "cv2": "Mat",
    "sklearn": "ndarray",
    "matplotlib": "Figure",
    "numpy": "ndarray",
    "scipy": "ndarray",
    "python_core": "any"
}

DOMAIN_CONFIGS = [
    {
        "domain": "pandas",
        "primary_tree": "trees/pandas.json",
        "seed_file": "seeds/pandas_seeds.json",
        "harvest_files": ["harvests/verified_pandas.json", "harvests/enriched_pandas.json", "harvests/structural_pandas.json"],
        "default_deps": ["import pandas as pd"]
    },
    {
        "domain": "cv2",
        "primary_tree": "trees/cv2.json",
        "seed_file": "seeds/cv2_seeds.json",
        "harvest_files": ["harvests/verified_cv2.json", "harvests/enriched_cv2.json", "harvests/structural_cv2.json"],
        "default_deps": ["import cv2"]
    },
    {
        "domain": "sklearn",
        "primary_tree": "trees/sklearn.json",
        "seed_file": "seeds/sklearn_seeds.json",
        "harvest_files": ["harvests/verified_sklearn.json", "harvests/enriched_sklearn.json", "harvests/structural_sklearn.json"],
        "default_deps": ["import sklearn"]
    },
    {
        "domain": "matplotlib",
        "primary_tree": "trees/matplotlib.json",
        "seed_file": "seeds/matplotlib_seeds.json",
        "harvest_files": [],
        "default_deps": ["import matplotlib.pyplot as plt"]
    },
    {
        "domain": "numpy",
        "primary_tree": "trees/numpy.json",
        "seed_file": None,
        "harvest_files": ["harvests/verified_numpy.json", "harvests/enriched_numpy.json", "harvests/structural_numpy.json"],
        "default_deps": ["import numpy as np"]
    },
    {
        "domain": "scipy",
        "primary_tree": "trees/scipy.json",
        "seed_file": None,
        "harvest_files": ["harvests/verified_scipy.json", "harvests/enriched_scipy.json", "harvests/structural_scipy.json"],
        "default_deps": ["import scipy"]
    },
    {
        "domain": "python_core",
        "primary_tree": "trees/python_core.json",
        "seed_file": "seeds/python_core_seeds.json",
        "harvest_files": [],
        "default_deps": []
    }
]


def sanitize_type_name(t: str) -> str:
    if not t or t.lower() in ("any", "anyobject", "object", ""):
        return "any"
    return t.strip()

def normalize_ports(raw_ports: Any, default_type: str = "any", default_state: str = "raw") -> Dict[str, PortSchema]:
    ports = {}
    if isinstance(raw_ports, dict):
        for k, v in raw_ports.items():
            if isinstance(v, dict):
                t = sanitize_type_name(v.get("type_name", v.get("type", default_type)))
                s = v.get("state", default_state)
                req = v.get("required", True)
                def_v = v.get("default_value", v.get("default"))
                ports[k] = PortSchema(
                    type_name=t,
                    state=str(s),
                    required=bool(req),
                    default_value=str(def_v) if def_v is not None else None,
                    description=v.get("description", v.get("doc", ""))
                )
            elif isinstance(v, str):
                ports[k] = PortSchema(type_name=sanitize_type_name(v), state=default_state)
    elif isinstance(raw_ports, list):
        for item in raw_ports:
            if isinstance(item, dict) and "name" in item:
                name = item["name"]
                t = sanitize_type_name(item.get("type_name", item.get("type", default_type)))
                s = item.get("state", default_state)
                req = item.get("required", True)
                def_v = item.get("default_value", item.get("default"))
                ports[name] = PortSchema(
                    type_name=t,
                    state=str(s),
                    required=bool(req),
                    default_value=str(def_v) if def_v is not None else None,
                    description=item.get("description", item.get("doc", item.get("param_doc", "")))
                )
    return ports

def consolidate_domain(config: Dict[str, Any], output_dir: Path) -> Optional[TreeSchema]:
    domain = config["domain"]
    default_container = CONTAINER_TYPES.get(domain, "any")
    cells_map: Dict[str, CellSchema] = {}

    out_file = output_dir / f"{domain}.json"

    # 1. Load existing primary tree file if present
    primary_tree = config.get("primary_tree")
    if primary_tree and (PROJECT_ROOT / primary_tree).exists():
        try:
            with open(PROJECT_ROOT / primary_tree, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "cells" in data:
                for c in data["cells"]:
                    cell_obj = CellSchema(**c)
                    cells_map[cell_obj.cell_id.upper()] = cell_obj
            elif isinstance(data, list):
                for c in data:
                    if isinstance(c, dict) and "cell_id" in c:
                        cell_obj = CellSchema(**c)
                        cells_map[cell_obj.cell_id.upper()] = cell_obj
        except Exception as e:
            print(f"[!] Warning loading primary tree {primary_tree}: {e}")

    # 2. Ingest curated seeds if file exists (highest priority: 1)
    seed_file = config.get("seed_file")
    if seed_file and (PROJECT_ROOT / seed_file).exists():
        with open(PROJECT_ROOT / seed_file, "r", encoding="utf-8") as f:
            seeds = json.load(f)
            for s in seeds:
                cid = s["cell_id"].strip().upper()
                c = CellSchema(
                    cell_id=cid,
                    stage=s["stage"],
                    inputs=normalize_ports(s.get("inputs", {}), default_type=default_container, default_state="any"),
                    outputs=normalize_ports(s.get("outputs", {}), default_type=default_container, default_state="raw"),
                    code_template=s["code_template"],
                    dependencies=s.get("dependencies", config["default_deps"]),
                    semantic_tags=s.get("semantic_tags", []),
                    keywords=s.get("keywords", []),
                    docstring=s.get("docstring", ""),
                    domain_name=domain,
                    node_type=s.get("node_type", "function"),
                    node_role=s.get("node_role", "function"),
                    source_priority=1
                )
                cells_map[cid] = c

    # 3. Ingest harvest files if present
    harvest_files = config.get("harvest_files", [])
    for rel_path in harvest_files:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading {rel_path}: {e}")
            continue

        nodes = data.get("nodes", data.get("cells", data)) if isinstance(data, dict) else data
        if not isinstance(nodes, list):
            continue

        for raw_cell in nodes:
            if not isinstance(raw_cell, dict):
                continue
            cid = raw_cell.get("cell_id", "").upper().strip()
            if not cid or cid.startswith("_"):
                continue
            if cid in cells_map and cells_map[cid].source_priority == 1:
                continue

            kws = sorted(CellTokenizer.tokenize_identifier(cid))
            code = raw_cell.get("code_template") or raw_cell.get("code", "")
            if not code:
                continue

            try:
                c = CellSchema(
                    cell_id=cid,
                    stage=raw_cell.get("stage", 2),
                    inputs=normalize_ports(raw_cell.get("inputs", {}), default_type=default_container, default_state="any"),
                    outputs=normalize_ports(raw_cell.get("outputs", {}), default_type=default_container, default_state="processed"),
                    code_template=code,
                    dependencies=raw_cell.get("dependencies", config["default_deps"]),
                    semantic_tags=kws,
                    keywords=kws,
                    docstring=raw_cell.get("docstring", raw_cell.get("description", "")),
                    domain_name=domain,
                    node_type=raw_cell.get("node_type", "function"),
                    node_role=raw_cell.get("node_role", "function"),
                    source_priority=100
                )
                cells_map[cid] = c
            except Exception:
                pass

    # Safety Guard: Never write empty tree if no cells loaded
    if not cells_map:
        print(f"[!] SKIP: Domain '{domain}' has 0 cells mapped. Preserving existing {out_file}.")
        return None

    tree_schema = TreeSchema(
        domain=domain,
        version="1.0.0",
        cells=list(cells_map.values())
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(tree_schema.model_dump_json(indent=2))

    print(f"[+] Consolidated domain '{domain}' -> {out_file} ({len(tree_schema.cells)} cells, {sum(1 for c in tree_schema.cells if c.source_priority <= 10)} verified/seeds)")
    return tree_schema


def main():
    trees_dir = PROJECT_ROOT / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(" STRATA DOMAIN CONSOLIDATION & VERIFICATION")
    print("======================================================================")

    for cfg in DOMAIN_CONFIGS:
        consolidate_domain(cfg, trees_dir)

    print("\n[*] Domain check complete.")


if __name__ == "__main__":
    main()
