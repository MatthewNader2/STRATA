import importlib
import inspect
import json
import argparse
from abc import ABC, abstractmethod
import typing

class BaseScraper(ABC):
    @abstractmethod
    def scrape(self, target: str) -> dict:
        """Parses a target module/crate and returns a dict matching the Macro-Lattice schema."""
        pass

class PythonModuleScraper(BaseScraper):
    def scrape(self, module_name: str) -> dict:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            print(f"Error: Module {module_name} not found.")
            return {}

        cells = []
        for name, obj in inspect.getmembers(module):
            # Target only functions and built-ins, ignore private members
            if not (inspect.isfunction(obj) or inspect.isbuiltin(obj)) or name.startswith("_"):
                continue

            try:
                sig = inspect.signature(obj)
            except ValueError:
                # Builtins without exposed signatures are skipped for Accuracy
                continue

            # Ensure strict type annotations exist
            if sig.return_annotation == inspect._empty:
                continue
            
            params = list(sig.parameters.values())
            if any(p.annotation == inspect._empty for p in params):
                continue

            input_types = [self._format_type(p.annotation) for p in params]
            output_type = self._format_type(sig.return_annotation)

            # Map arguments to tuple if multiple, or direct if single/none
            if len(params) == 0:
                code = f"import {module_name}\n{{output_var}} = {module_name}.{name}()"
                input_sig = "None"
            elif len(params) == 1:
                code = f"import {module_name}\n{{output_var}} = {module_name}.{name}({{input_var}})"
                input_sig = input_types[0]
            else:
                code = f"import {module_name}\n{{output_var}} = {module_name}.{name}(*{{input_var}})"
                input_sig = f"Tuple[{', '.join(input_types)}]"

            cell_id = f"micro_{module_name}_{name}"
            
            cells.append({
                "cell_id": cell_id,
                "type": "micro",
                "stage": 1,
                "keywords": [module_name, name],
                "inputs": {
                    "type_name": input_sig,
                    "state": "raw"
                },
                "outputs": {
                    "type_name": output_type,
                    "state": "computed"
                },
                "domain_implementations": {
                    "Python_Core": {
                        "code": code,
                        "dependencies": [module_name]
                    }
                }
            })

        return {"domain_name": f"{module_name}_domain", "cells": cells}

    def _format_type(self, annotation) -> str:
        if annotation is None:
            return "None"
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation).replace("typing.", "")

def main():
    parser = argparse.ArgumentParser(description="Scrape Python modules to Lattice JSON")
    parser.add_argument("modules", nargs='+', help="Target modules (e.g. typing os json)")
    args = parser.parse_args()

    scraper = PythonModuleScraper()
    import os
    
    for mod in args.modules:
        result = scraper.scrape(mod)
        out_path = f"trees/micro/auto_{mod}.json"
        
        if not result.get("cells"):
            print(f"Warning: No strictly typed functions found in '{mod}'.")
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Success: Scraped {len(result['cells'])} verified Micro-Nodes to {out_path}.")

if __name__ == "__main__":
    main()
