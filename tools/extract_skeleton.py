"""
tools/extract_skeleton.py - Symbolic Topological Routing Architecture for Typed Automata (STRATA)
Extracts type-annotated skeletons for LLM parameter configuration.
"""

import inspect
import importlib
import sys
import json
from typing import Dict, List, Any


def extract_skeleton(root_module_name: str, output_file: str):
    try:
        root_mod = importlib.import_module(root_module_name)
    except ImportError:
        raise ImportError(f"Could not import module '{root_module_name}'")

    visited = set()
    skeletons = {}

    def process_routine(obj, name, parent_name, is_method=False):
        dedup_key = f"{root_module_name}.{name}"
        if dedup_key in skeletons:
            skeletons[dedup_key]["contexts"].append(parent_name)
            return

        params = []
        try:
            sig = inspect.signature(obj)
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls'):
                    continue
                has_default = param.default is not inspect.Parameter.empty
                params.append({
                    "name": param_name,
                    "required": not has_default,
                    "default": None if not has_default else repr(param.default)
                })
        except (ValueError, TypeError):
            params = [{"name": "input_var", "required": True, "default": None}]

        doc = inspect.getdoc(obj) or ""
        doc_short = "\n".join(doc.split('\n')[:3])

        skeletons[dedup_key] = {
            "name": name,
            "contexts": [parent_name],
            "is_method": is_method,
            "params": params,
            "doc": doc_short
        }

    def walk_module(mod, parent_path):
        if mod in visited:
            return
        visited.add(mod)

        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue

            if inspect.ismodule(obj) and getattr(obj, '__name__', '').startswith(root_module_name):
                walk_module(obj, obj.__name__)
            elif inspect.isclass(obj) and getattr(obj, '__module__', '').startswith(root_module_name):
                if obj not in visited:
                    visited.add(obj)
                    for method_name, method_obj in inspect.getmembers(obj):
                        if not method_name.startswith("_") and (inspect.isfunction(method_obj) or inspect.ismethod(method_obj)):
                            process_routine(method_obj, method_name, f"{parent_path}.{name}", is_method=True)
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                process_routine(obj, name, parent_path, is_method=False)

    walk_module(root_mod, root_module_name)
    skeleton_list = list(skeletons.values())

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(skeleton_list, f, indent=2)

    print(f"[+] Extracted {len(skeleton_list)} structured skeletons from {root_module_name} -> {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_skeleton.py <module_name> <output_file.json>")
        sys.exit(1)
    extract_skeleton(sys.argv[1], sys.argv[2])
