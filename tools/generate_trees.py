#!/usr/bin/env python3
"""
STRATA Automated Tree Generator
==============================
Generates STRATA micro-cell and macro-cell tree JSON files automatically using
three complementary strategies:

  1. SCRAPE  — Introspects installed Python modules via `inspect` to extract
               fully type-annotated functions. Zero LLM tokens needed.

  2. CURATE  — Writes hand-crafted seed cells for libraries where introspection
               misses the most important patterns (e.g. pandas, flask).
               These seeds are defined as Python dicts in this file.

  3. EXPAND  — Uses the local LLM (BenchmarkProfile_B via ModelManager) to
               generate additional cells or macro nodes that introspection
               cannot produce (algorithms, pipelines, patterns).

Usage:
    # Scrape a module
    python tools/generate_trees.py scrape pandas numpy sklearn.preprocessing

    # Generate macro nodes for algorithmic concepts
    python tools/generate_trees.py macro "Binary Search" "Merge Sort" "Dijkstra"

    # Run the full seed-and-expand pipeline
    python tools/generate_trees.py seed

    # Show available strategies
    python tools/generate_trees.py --help
"""

import argparse
import importlib
import inspect
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger("STRATA-Generator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREES_DIR  = os.path.join(ROOT_DIR, "trees")
MICRO_DIR  = os.path.join(TREES_DIR, "micro")
MACRO_DIR  = os.path.join(TREES_DIR, "macro")

os.makedirs(MICRO_DIR, exist_ok=True)
os.makedirs(MACRO_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STRATEGY 1 — SCRAPE (introspection, no LLM)
# ---------------------------------------------------------------------------

class PythonScraper:
    """
    Introspects an installed Python module and generates micro-cell JSON for
    every public, fully type-annotated function found.

    Design notes:
    - We skip functions without return annotations to keep type-states meaningful.
    - For multi-param functions we collapse all inputs into a Tuple[…] signature.
    - The `domain_implementations` block only contains Python_Core for now; Rust /
      C++ domain cells can be added later via the `merge` CLI of tree_manager.py.
    """

    def _fmt(self, annotation) -> str:
        if annotation is inspect.Parameter.empty or annotation is None:
            return "Any"
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        s = str(annotation)
        # Clean up typing module noise
        for prefix in ("typing.", "collections.abc.", "<class '", "'>"):
            s = s.replace(prefix, "")
        return s.strip("'\"")

    def scrape(self, module_name: str) -> dict:
        sys.path.insert(0, ROOT_DIR)
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            log.error(f"Cannot import '{module_name}': {e}")
            return {}

        cells = []
        seen = set()

        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isbuiltin(obj)):
                continue
            if name in seen:
                continue
            seen.add(name)

            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                continue

            # Skip if return type is unknown
            if sig.return_annotation is inspect.Parameter.empty:
                continue

            params = [p for p in sig.parameters.values()
                      if p.name not in ("self", "cls")]

            # Build input signature
            typed_params = [p for p in params
                            if p.annotation is not inspect.Parameter.empty]

            if not params:
                input_type = "None"
                code = (f"import {module_name}\n"
                        f"{{output_var}} = {module_name}.{name}()")
            elif len(typed_params) == 1:
                input_type = self._fmt(typed_params[0].annotation)
                code = (f"import {module_name}\n"
                        f"{{output_var}} = {module_name}.{name}({{input_var}})")
            else:
                parts = [self._fmt(p.annotation) for p in typed_params] or ["Any"]
                input_type = f"Tuple[{', '.join(parts)}]"
                code = (f"import {module_name}\n"
                        f"{{output_var}} = {module_name}.{name}(*{{input_var}})")

            output_type = self._fmt(sig.return_annotation)

            cells.append({
                "cell_id": f"micro_{module_name.replace('.', '_')}_{name}",
                "type": "micro",
                "stage": 1,
                "keywords": [module_name.split(".")[-1], name],
                "inputs":  {"type_name": input_type,  "state": "raw"},
                "outputs": {"type_name": output_type, "state": "computed"},
                "domain_implementations": {
                    "Python_Core": {
                        "code": code,
                        "dependencies": [module_name.split(".")[0]]
                    }
                }
            })

        log.info(f"Scraped {len(cells)} cells from '{module_name}'")
        return {
            "domain_name": f"{module_name}_domain",
            "version": "2.0.0",
            "cells": cells
        }


def cmd_scrape(args):
    scraper = PythonScraper()
    for mod in args.modules:
        result = scraper.scrape(mod)
        if not result.get("cells"):
            log.warning(f"No typed cells found in '{mod}' — skipping.")
            continue
        safe_name = mod.replace(".", "_")
        out_path  = os.path.join(MICRO_DIR, f"auto_{safe_name}.json")
        _atomic_write(out_path, result)
        log.info(f"✓ Wrote {len(result['cells'])} cells → {out_path}")


# ---------------------------------------------------------------------------
# STRATEGY 2 — CURATE (hand-written seed expansions)
# These seeds define the cells that introspection misses because pandas, flask,
# etc. don't expose their primary API as top-level typed functions.
# ---------------------------------------------------------------------------

SEED_MICRO_CELLS = {
    # Key   = library slug  →  added to trees/micro/<slug>_seed.json
    # Value = list of cell dicts using the STRATA schema

    "numpy_seed": {
        "domain_name": "NumPy_Arrays",
        "version": "2.0.0",
        "cells": [
            {
                "cell_id": "NUMPY_ARRAY_FROM_LIST",
                "type": "micro", "stage": 1,
                "keywords": ["numpy", "array", "create", "list", "ndarray"],
                "inputs":  {"type_name": "list", "state": "tokens"},
                "outputs": {"type_name": "ndarray", "state": "raw"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = np.array({input_var})",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_ZEROS",
                "type": "micro", "stage": 1,
                "keywords": ["numpy", "zeros", "empty", "initialize", "array"],
                "inputs":  {"type_name": "int", "state": "count"},
                "outputs": {"type_name": "ndarray", "state": "initialized"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = np.zeros({input_var})",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_MEAN",
                "type": "micro", "stage": 2,
                "keywords": ["mean", "average", "statistics", "numpy"],
                "inputs":  {"type_name": "ndarray", "state": "raw"},
                "outputs": {"type_name": "float", "state": "computed"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = float(np.mean({input_var}))",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_RESHAPE",
                "type": "micro", "stage": 2,
                "keywords": ["reshape", "shape", "matrix", "tensor", "numpy"],
                "inputs":  {"type_name": "ndarray", "state": "raw"},
                "outputs": {"type_name": "ndarray", "state": "reshaped"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = {input_var}.reshape(-1, 1)",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_NORMALIZE",
                "type": "micro", "stage": 2,
                "keywords": ["normalize", "l2", "unit vector", "scale", "numpy"],
                "inputs":  {"type_name": "ndarray", "state": "raw"},
                "outputs": {"type_name": "ndarray", "state": "normalized"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n_norm = np.linalg.norm({input_var})\n{output_var} = {input_var} / (_norm if _norm > 0 else 1.0)",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_DOT",
                "type": "micro", "stage": 2,
                "keywords": ["dot product", "matmul", "matrix multiply", "numpy"],
                "inputs":  {"type_name": "ndarray", "state": "normalized"},
                "outputs": {"type_name": "ndarray", "state": "computed"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = np.dot({input_var}, {input_var}.T)",
                    "dependencies": ["numpy"]
                }}
            },
            {
                "cell_id": "NUMPY_LINSPACE",
                "type": "micro", "stage": 1,
                "keywords": ["linspace", "range", "evenly spaced", "sequence", "numpy"],
                "inputs":  {"type_name": "int", "state": "count"},
                "outputs": {"type_name": "ndarray", "state": "sequence"},
                "domain_implementations": {"Python_Core": {
                    "code": "import numpy as np\n{output_var} = np.linspace(0, 1, {input_var})",
                    "dependencies": ["numpy"]
                }}
            }
        ]
    }
}


def cmd_seed(_args):
    """Write all hand-curated seed cells to their respective tree files."""
    for slug, tree in SEED_MICRO_CELLS.items():
        out_path = os.path.join(MICRO_DIR, f"{slug}.json")
        _atomic_write(out_path, tree)
        log.info(f"✓ Seeded {len(tree['cells'])} cells → {out_path}")
    log.info("Seed complete.")


# ---------------------------------------------------------------------------
# STRATEGY 3 — EXPAND via LLM (macro + gap-filling micro cells)
# ---------------------------------------------------------------------------

MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cell_id", "type", "stage", "keywords",
                             "inputs", "outputs", "algorithmic_steps",
                             "sub_cells", "internal_topology"],
                "properties": {
                    "cell_id": {"type": "string"},
                    "type":    {"type": "string", "enum": ["macro"]},
                    "stage":   {"type": "integer"},
                    "keywords":{"type": "array", "items": {"type": "string"}},
                    "inputs":  {"type": "object",
                                "required": ["type_name", "state"],
                                "properties": {
                                    "type_name": {"type": "string"},
                                    "state":     {"type": "string"}
                                }},
                    "outputs": {"type": "object",
                                "required": ["type_name", "state"],
                                "properties": {
                                    "type_name": {"type": "string"},
                                    "state":     {"type": "string"}
                                }},
                    "algorithmic_steps": {"type": "array", "items": {"type": "string"}},
                    "sub_cells": {"type": "array", "items": {"type": "string"}},
                    "internal_topology": {"type": "object"}
                }
            }
        }
    },
    "required": ["cells"]
}

MICRO_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["cell_id", "type", "stage", "keywords",
                             "inputs", "outputs", "domain_implementations"],
                "properties": {
                    "cell_id": {"type": "string"},
                    "type":    {"type": "string", "enum": ["micro"]},
                    "stage":   {"type": "integer"},
                    "keywords":{"type": "array", "items": {"type": "string"}},
                    "inputs":  {"type": "object",
                                "required": ["type_name", "state"]},
                    "outputs": {"type": "object",
                                "required": ["type_name", "state"]},
                    "domain_implementations": {"type": "object"}
                }
            }
        }
    },
    "required": ["cells"]
}


def _get_rag_context(concept: str, top_k: int = 8) -> str:
    """Pull relevant micro-node IDs from LocalRAG to provide LLM with real options."""
    try:
        sys.path.insert(0, ROOT_DIR)
        from internal_rag import LocalRAG
        rag = LocalRAG(trees_dir=TREES_DIR)
        return rag.get_relevant_context(concept, top_k=top_k)
    except Exception as e:
        log.warning(f"RAG unavailable ({e}), proceeding without context.")
        return "(No existing micro-nodes available — invent clean semantic IDs)"


def cmd_macro(args):
    """Generate macro-cell JSON for one or more algorithmic concepts via LLM."""
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
    sys.path.insert(0, ROOT_DIR)

    # Lazy-import to avoid requiring LLM for scrape/seed commands
    from inference import ModelManager
    log.info("Loading ModelManager with BenchmarkProfile_C …")
    mm = ModelManager.get_instance()
    if mm.profile is None:
        mm.initialize_profile("C")

    for concept in args.concepts:
        log.info(f"Generating macro-cell for: '{concept}'")
        rag_ctx = _get_rag_context(concept)

        prompt = f"""You are a senior Software Architect building an STRATA Macro-Cell.
Your task: encode the algorithm '{concept}' as a Macro-Cell JSON object.

RULES:
1. Output ONLY valid JSON — no markdown, no explanation.
2. Use the schema below exactly.
3. sub_cells must reference real IDs from the pool. If a step has no match,
   invent a clean semantic placeholder like 'micro_custom_<step>'.
4. internal_topology maps each sub_cell to its successors (directed graph).

Schema:
{{
  "cells": [{{
    "cell_id": "macro_<concept_snake_case>",
    "type": "macro",
    "stage": 2,
    "keywords": ["<keyword>", ...],
    "inputs":  {{"type_name": "<input_type>", "state": "<input_state>"}},
    "outputs": {{"type_name": "<output_type>", "state": "<output_state>"}},
    "algorithmic_steps": ["1. ...", "2. ...", ...],
    "sub_cells": ["<cell_id>", ...],
    "internal_topology": {{"<src>": ["<dst>"], ...}}
  }}]
}}

Available Micro-Node IDs (prefer these in sub_cells):
{rag_ctx}

Algorithm to encode: {concept}"""

        try:
            raw = mm.generate_text(prompt, max_tokens=2048)
            clean_raw = raw.strip()
            if clean_raw.startswith("```"):
                clean_raw = clean_raw.split("\n", 1)[-1]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw.rsplit("```", 1)[0]
            data = json.loads(clean_raw.strip())
        except Exception as e:
            log.error(f"LLM produced invalid JSON for '{concept}': {e}")
            continue

        slug     = concept.lower().replace(" ", "_").replace("*", "star").replace("/", "_")
        out_path = os.path.join(MACRO_DIR, f"{slug}.json")
        _atomic_write(out_path, data)
        log.info(f"✓ Macro '{concept}' → {out_path}")


def cmd_expand(args):
    """
    Expand a library's micro-cells by asking the LLM to produce cells for
    common use-cases that introspection cannot detect (e.g. method chaining,
    context managers, decorator patterns).
    """
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
    sys.path.insert(0, ROOT_DIR)
    from inference import ModelManager
    log.info("Loading ModelManager with BenchmarkProfile_C …")
    mm = ModelManager.get_instance()
    if mm.profile is None:
        mm.initialize_profile("C")

    for lib in args.libraries:
        log.info(f"Expanding micro-cells for library: '{lib}'")

        prompt = f"""You are an expert Python developer building an STRATA micro-cell tree.
Generate 8-12 practical, commonly-used micro-cells for the '{lib}' Python library.

Each cell captures ONE single, idiomatic operation (like read_csv, drop_na, fit_transform).

Output ONLY valid JSON — no markdown, no explanation.

Schema for each cell:
{{
  "cell_id": "micro_{lib}_<operation>",
  "type": "micro",
  "stage": 1,
  "keywords": ["{lib}", "<operation>", "<alias>"],
  "inputs":  {{"type_name": "<Python type>", "state": "<semantic state>"}},
  "outputs": {{"type_name": "<Python type>", "state": "<semantic state>"}},
  "domain_implementations": {{
    "Python_Core": {{
      "code": "import {lib}\\n{{output_var}} = {lib}.<method>({{input_var}})",
      "dependencies": ["{lib}"]
    }}
  }}
}}

Wrap all cells in: {{"domain_name": "{lib}_domain", "cells": [...]}}

Library: {lib}"""

        try:
            raw  = mm.generate_text(prompt, max_tokens=3000)
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error(f"LLM produced invalid JSON for '{lib}': {e}")
            continue

        out_path = os.path.join(MICRO_DIR, f"llm_{lib}.json")
        _atomic_write(out_path, data)
        log.info(f"✓ Expanded '{lib}' → {out_path} ({len(data.get('cells', []))} cells)")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _atomic_write(path: str, data: dict):
    """Write JSON atomically via tmp + os.replace to prevent corrupt files."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"Failed to write {path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STRATA Automated Tree Generator — three strategies in one tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape",
        help="Introspect installed Python modules and generate micro-cell JSON")
    p_scrape.add_argument("modules", nargs="+",
        help="Module names to scrape (e.g. os json typing math)")
    p_scrape.set_defaults(func=cmd_scrape)

    # seed
    p_seed = sub.add_parser("seed",
        help="Write hand-curated seed cells for numpy, etc.")
    p_seed.set_defaults(func=cmd_seed)

    # macro
    p_macro = sub.add_parser("macro",
        help="Use LLM to generate macro-cell JSON for algorithmic concepts")
    p_macro.add_argument("concepts", nargs="+",
        help='Concepts to encode (e.g. "A* Pathfinding" "Merge Sort")')
    p_macro.set_defaults(func=cmd_macro)

    # expand
    p_expand = sub.add_parser("expand",
        help="Use LLM to generate additional micro-cells for a library")
    p_expand.add_argument("libraries", nargs="+",
        help="Library names (e.g. requests httpx sqlalchemy)")
    p_expand.set_defaults(func=cmd_expand)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
