import json
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

print("=" * 70)
print(" STRATA CANONICAL NODE SPECIFICATION & CELL ID EXPORTER")
print("=" * 70)

# 1. Collect all existing Cell IDs
tree_files = sorted(glob.glob(str(ROOT / "trees/*.json")))
domain_cell_map = {}
all_cell_ids = []

for tf in tree_files:
    p = Path(tf)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain = data.get("domain", p.stem)
    cells = data.get("cells", [])

    c_ids = sorted(
        c.get("cell_id")
        for c in cells
        if c.get("cell_id")
    )

    domain_cell_map[domain] = {
        "file": p.name,
        "count": len(c_ids),
        "types": data.get("types", {}),
        "cell_ids": c_ids,
    }

    all_cell_ids.extend(c_ids)

print(f"[*] Found {len(all_cell_ids)} cells across {len(domain_cell_map)} domains:")

for dom, info in domain_cell_map.items():
    print(
        f"    - {dom:<15} ({info['file']}): "
        f"{info['count']} cells | {len(info['types'])} custom types"
    )

# 2. Canonical Schema Documentation
SCHEMA_DOCS = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "STRATA_DomainTreeSchema",
    "description": (
        "Category-theoretic domain plugin tree definition for "
        "Symbolic Topological Routing Architecture for Typed Automata (STRATA)"
    ),
    "tree_structure": {
        "domain": (
            "string (name of domain/library, "
            "e.g. cv2, pandas, torch, audio)"
        ),
        "version": "string (semantic version, e.g. 1.0.0)",
        "types": (
            "object (domain plugin taxonomy declaring custom "
            "terms and their parent in the poset)"
        ),
        "typestates": "object (declared typestate vocabulary)",
        "cells": "array of CellSchema objects",
    },
    "cell_fields": {
        "cell_id": (
            "string (UNIQUE screaming snake-case or dot-notation "
            "identifier, e.g. CV2_CANNY, np.linalg.svd)"
        ),
        "stage": (
            "integer (Morphism Stage): "
            "1 = Ingress/Source (Env -> C), "
            "2 = Transform/Endomorphism (C x P -> C), "
            "3 = Egress/Sink (C -> Env), "
            "0 = Constructor"
        ),
        "node_type": (
            "string ('function' | 'method' | "
            "'constructor' | 'macro')"
        ),
        "node_role": (
            "string ('source' | 'transform' | "
            "'estimator' | 'sink' | 'bridge')"
        ),
        "topology_type": (
            "string ('sequential' | 'monoidal_product' | "
            "'coproduct_branch' | 'traced_loop')"
        ),
        "inputs": "object (map of port_name -> PortSchema)",
        "outputs": "object (map of port_name -> PortSchema)",
        "code_template": (
            "string (AST Python template using {port_name} "
            "placeholders, e.g. {output_var} = "
            "cv2.Canny({image}, {threshold1}, {threshold2}))"
        ),
        "dependencies": (
            "array of strings (exact import statements, "
            "e.g. ['import cv2', 'import numpy as np'])"
        ),
        "semantic_tags": (
            "array of strings (retrieval tokens for dense "
            "FAISS and BM25 router indexing)"
        ),
        "keywords": "array of strings (lexical triggers)",
        "docstring": "string (concise technical docstring)",
        "mutation_type": "string ('pure' | 'in_place')",
        "preconditions": (
            "array of condition objects "
            "(Phase-1 verification contracts)"
        ),
        "postconditions": (
            "array of condition objects "
            "(Phase-1 verification contracts)"
        ),
        "effects": (
            "array of strings (semantic effect tags, "
            "e.g. ['draws_annotation', 'cleans_missing'])"
        ),
    },
    "port_fields": {
        "type_name": (
            "string (Ground type, higher-kinded functor, or "
            "refined carrier: e.g. File[Image, PNG], "
            "List[Contour], Tuple[int, ...], Dict[str, Any], "
            "ndarray, DataFrame, int, float, str, bool)"
        ),
        "state": (
            "string (Algebraic typestate: e.g. file_path, "
            "color_bgr, binary, raw_dataset, "
            "column_projection, fitted_estimator)"
        ),
        "required": (
            "boolean (true for mandatory positional parameters; "
            "false for optional configuration knobs)"
        ),
        "default_value": (
            "any (literal Python default value, or null if required)"
        ),
        "abstract_type": (
            "string ('tensor' | 'table' | 'collection' | "
            "'scalar' | 'path' | 'logical' | null)"
        ),
        "param_kind": (
            "string ('standard' | 'positional_only' | "
            "'keyword_only' | 'var_positional' | 'var_keyword')"
        ),
        "enum_values": (
            "array of values (if port restricts values to "
            "an enum set, else null)"
        ),
        "value_constraints": (
            "object (optional bounds, e.g. "
            "{'min': 0, 'max': 255})"
        ),
        "shape_contract": (
            "object (optional tensor contract, "
            "e.g. {'ndim': 2})"
        ),
    },
}

# 3. Canonical Few-Shot Exemplars
EXEMPLARS = [
    {
        "_comment": "STAGE 1: Environmental Ingress (File -> Memory Carrier)",
        "cell_id": "CV2_IMREAD",
        "stage": 1,
        "node_type": "function",
        "node_role": "source",
        "topology_type": "sequential",
        "inputs": {
            "image_path": {
                "type_name": "File[Image, Any]",
                "state": "file_path",
                "required": True,
                "default_value": None,
                "abstract_type": "path",
                "param_kind": "standard",
            },
            "flags": {
                "type_name": "int",
                "state": "flag",
                "required": False,
                "default_value": 1,
                "abstract_type": "scalar",
                "enum_values": [0, 1, -1, 2, 4, 8],
                "param_kind": "standard",
            },
        },
        "outputs": {
            "output_var": {
                "type_name": "ndarray",
                "state": "color_bgr",
                "required": True,
                "abstract_type": "tensor",
                "value_constraints": {"dtype": "uint8"},
                "shape_contract": {"ndim": 3},
            }
        },
        "code_template": "{output_var} = cv2.imread({image_path}, {flags})",
        "dependencies": ["import cv2"],
        "semantic_tags": [
            "read",
            "load",
            "image",
            "decode",
            "bgr",
            "imread",
        ],
    },
    {
        "_comment": (
            "STAGE 2: Endomorphism / Higher-Kinded Container Transform"
        ),
        "cell_id": "CV2_FIND_CONTOURS",
        "stage": 2,
        "node_type": "function",
        "node_role": "transform",
        "topology_type": "sequential",
        "inputs": {
            "image": {
                "type_name": "ndarray",
                "state": "binary",
                "accepted_states": ["edge_map"],
                "required": True,
                "default_value": None,
                "abstract_type": "tensor",
                "shape_contract": {"ndim": 2},
            },
            "mode": {
                "type_name": "int",
                "state": "retrieval_mode",
                "required": False,
                "default_value": 0,
                "abstract_type": "scalar",
                "enum_values": [0, 1, 2, 3],
            },
            "method": {
                "type_name": "int",
                "state": "approx_method",
                "required": False,
                "default_value": 2,
                "abstract_type": "scalar",
                "enum_values": [1, 2, 3, 4],
            },
        },
        "outputs": {
            "contours": {
                "type_name": "List[Contour]",
                "state": "contours",
                "required": True,
                "abstract_type": "collection",
            },
            "hierarchy": {
                "type_name": "ndarray",
                "state": "hierarchy",
                "required": True,
                "abstract_type": "tensor",
                "value_constraints": {"dtype": "int32"},
            },
        },
        "code_template": (
            "{contours}, {hierarchy} = "
            "cv2.findContours({image}, {mode}, {method})"
        ),
        "dependencies": ["import cv2"],
        "semantic_tags": [
            "find",
            "contours",
            "boundary",
            "shapes",
            "detect",
        ],
    },
    {
        "_comment": "STAGE 3: Egress / Sink (Memory Carrier -> File on Disk)",
        "cell_id": "PD_TO_CSV",
        "stage": 3,
        "node_type": "method",
        "node_role": "sink",
        "topology_type": "sequential",
        "inputs": {
            "data": {
                "type_name": "DataFrame",
                "state": "any",
                "required": True,
                "default_value": None,
                "abstract_type": "table",
            },
            "filepath": {
                "type_name": "File[Tabular, CSV]",
                "state": "destination_path",
                "required": True,
                "default_value": None,
                "abstract_type": "path",
            },
            "index": {
                "type_name": "bool",
                "state": "index_flag",
                "required": False,
                "default_value": False,
                "abstract_type": "logical",
            },
        },
        "outputs": {
            "output_var": {
                "type_name": "File[Tabular, CSV]",
                "state": "written_to_disk",
                "required": True,
                "abstract_type": "path",
            }
        },
        "code_template": "{data}.to_csv({filepath}, index={index})",
        "dependencies": ["import pandas as pd"],
        "semantic_tags": [
            "save",
            "write",
            "csv",
            "export",
            "dataframe",
        ],
    },
]

# Export JSON Specification
out_json_path = ROOT / "tools/strata_node_spec.json"

out_json_data = {
    "schema": SCHEMA_DOCS,
    "existing_domains": domain_cell_map,
    "exemplars": EXEMPLARS,
}

out_json_path.write_text(
    json.dumps(out_json_data, indent=2),
    encoding="utf-8",
)

print(f"[✓] Exported complete JSON specification: {out_json_path}")

# Export AI Markdown Prompt
out_md_path = ROOT / "tools/ai_node_generation_prompt.md"

fence = "```"

existing_ids_json = json.dumps(
    {d: info["cell_ids"] for d, info in domain_cell_map.items()},
    indent=2,
)

exemplars_json = json.dumps(EXEMPLARS, indent=2)

md_content = f"""# STRATA Node Generation Prompt Template

You are an expert compiler and category theorist generating domain plugin cells for the **Symbolic Topological Routing Architecture for Typed Automata (STRATA)**.

## Core Directives

1. **Never collide with existing Cell IDs**: Below is the index of all {len(all_cell_ids)} existing Cell IDs across all domains. Any new cell MUST have a unique identifier.

2. **Strict Morphism Stages**:
   - `stage: 1` = Ingress/Source (takes a `File[Modality, Format]` or external source, outputs an in-memory carrier like `ndarray`, `DataFrame`).
   - `stage: 2` = Transform/Endomorphism (takes in-memory carrier, outputs in-memory carrier).
   - `stage: 3` = Egress/Sink (takes in-memory carrier and destination `File[Modality, Format]`, writes to disk).
   - `stage: 0` = Constructor (zero incoming required data ports).

3. **Sound Higher-Kinded Types**:
   - Never use unstructured `str` for files. Use `File[Modality, Format]` (e.g. `File[Image, PNG]`, `File[Tabular, CSV]`, `File[Audio, WAV]`, `File[Model, Joblib]`).
   - Never use untyped `list` for specific items. Use `List[T]` (e.g. `List[Contour]`, `List[str]`, `List[ndarray]`, `List[KeyPoint]`).
   - Use `Tuple[int, ...]` for tensor shapes; `Tuple[int, int]` for 2D geometry/size.

4. **Valid Code Templates**:
   - The `code_template` MUST be valid Python where every input and output port is enclosed in single braces `{{port_name}}`.

## Existing Cell IDs ({len(all_cell_ids)} total)

{fence}json
{existing_ids_json}
{fence}

## Few-Shot Canonical Exemplars

{fence}json
{exemplars_json}
{fence}

## Task

When asked to generate nodes for a domain (e.g. `torch`, `scipy`, `polars`, or expanding `cv2`/`sklearn`), output a strictly valid JSON array of `CellSchema` objects following this exact schema.
"""

out_md_path.write_text(
    md_content,
    encoding="utf-8",
)

print(f"[✓] Exported prompt template for AI: {out_md_path}")
print("=" * 70)
