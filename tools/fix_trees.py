"""
tools/fix_trees.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Canonical type and typestate normalizer for harvested cell definitions.
"""

from __future__ import annotations
import json
import sys
from typing import Dict, Any


def sanitize_type(type_name: str) -> str:
    """Normalizes type names to standard canonical identifiers."""
    if not type_name or type_name.lower() in ("any", "anyobject", "object", "*"):
        return "any"

    t_clean = type_name.strip()
    t_lower = t_clean.lower()

    if "dataframe" in t_lower:
        return "DataFrame"
    if "series" in t_lower:
        return "Series"
    if "ndarray" in t_lower or "array" in t_lower or "matrix" in t_lower:
        return "ndarray"
    if "mat" in t_lower or "image" in t_lower:
        return "Mat"
    if "str" in t_lower or "filepath" in t_lower:
        return "str"
    if "int" in t_lower:
        return "int"
    if "float" in t_lower or "double" in t_lower:
        return "float"
    if "bool" in t_lower:
        return "bool"
    if "dict" in t_lower or "graph" in t_lower:
        return "dict"
    if "list" in t_lower or "tuple" in t_lower:
        return "list"

    return t_clean


def fix_node(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures that cell ports, dependencies, and typestates adhere to lattice invariants."""
    inputs = cell.get("inputs", {})
    outputs = cell.get("outputs", {})

    if isinstance(inputs, dict):
        for port_name, port in inputs.items():
            if isinstance(port, dict):
                port["type_name"] = sanitize_type(port.get("type_name", port.get("type", "any")))
                port["state"] = port.get("state", "any").lower()

    if isinstance(outputs, dict):
        for port_name, port in outputs.items():
            if isinstance(port, dict):
                port["type_name"] = sanitize_type(port.get("type_name", port.get("type", "any")))
                port["state"] = port.get("state", "computed").lower()

    domain = cell.get("domain_name", cell.get("domain", "generic")).lower()
    deps = cell.get("dependencies", [])
    if not deps:
        if domain in ("pandas", "numpy", "cv2", "scipy", "sklearn"):
            alias = "import pandas as pd" if domain == "pandas" else (
                "import numpy as np" if domain == "numpy" else f"import {domain}"
            )
            cell["dependencies"] = [alias]

    return cell


def main():
    if len(sys.argv) < 3:
        print("Usage: python fix_trees.py <input.json> <output.json>")
        sys.exit(1)

    in_file, out_file = sys.argv[1], sys.argv[2]
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = data.get("cells", data.get("nodes", data)) if isinstance(data, dict) else data
    if isinstance(cells, list):
        fixed = [fix_node(c) for c in cells if isinstance(c, dict)]
        out_data = {"cells": fixed} if isinstance(data, dict) and "cells" in data else fixed
    else:
        out_data = data

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[+] Successfully normalized {in_file} -> {out_file}")


if __name__ == "__main__":
    main()
