import inspect
import importlib
import sys
import json
import types

def extract_expansions(root_module_name, output_file):
    try:
        root_mod = importlib.import_module(root_module_name)
    except ImportError:
        if root_module_name == "builtins":
            import builtins as root_mod
        else:
            raise ImportError(f"Could not import module '{root_module_name}'")
        
    visited = set()
    skeletons = {} 

    def process_node(obj, name, parent_name, node_type, is_method=False):
        dedup_key = f"{root_module_name}.{name}"
        
        if dedup_key in skeletons:
            if parent_name not in skeletons[dedup_key]["contexts"]:
                skeletons[dedup_key]["contexts"].append(parent_name)
            return

        params = []
        if node_type == "dunder":
            try:
                sig = inspect.signature(obj)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    params.append(param_name)
            except (ValueError, TypeError):
                params = ["*args", "**kwargs"]
        
        # Properties and constants have no params
        
        try:
            doc = inspect.getdoc(obj) or ""
            doc_short = "\n".join(doc.split('\n')[:4])
        except Exception:
            doc_short = ""

        skeletons[dedup_key] = {
            "name": name,
            "contexts": [parent_name],
            "is_method": is_method,
            "params": params,
            "doc": doc_short,
            "node_type": node_type # "constant", "property", or "dunder"
        }

    def walk_module(mod, parent_path):
        if id(mod) in visited:
            return
        visited.add(id(mod))
        
        try:
            members = inspect.getmembers(mod)
        except Exception:
            return
            
        for name, obj in members:
            # Skip standard private methods
            if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                continue
                
            # 1. Constants (Module level)
            if inspect.ismodule(mod):
                if name.isupper() and isinstance(obj, (int, float, str, bytes)):
                    process_node(obj, name, parent_path, "constant", is_method=False)
                    continue

            # 2. Submodules
            if inspect.ismodule(obj):
                if getattr(obj, '__name__', '').startswith(root_module_name):
                    walk_module(obj, obj.__name__)
                    
            # 3. Classes
            elif inspect.isclass(obj):
                if getattr(obj, '__module__', '').startswith(root_module_name) or root_module_name == "builtins":
                    if id(obj) not in visited:
                        visited.add(id(obj))
                        try:
                            class_members = inspect.getmembers(obj)
                        except Exception:
                            continue
                            
                        for member_name, member_obj in class_members:
                            # Skip standard private
                            if member_name.startswith("_") and not (member_name.startswith("__") and member_name.endswith("__")):
                                continue
                                
                            # Dunders
                            if member_name.startswith("__") and member_name.endswith("__"):
                                if member_name in ['__class__', '__delattr__', '__dir__', '__doc__', '__getattribute__', '__new__', '__setattr__', '__subclasshook__', '__module__', '__dict__', '__weakref__']:
                                    continue # Skip Python internals that aren't useful in graphs
                                if inspect.isfunction(member_obj) or inspect.ismethod(member_obj) or inspect.isroutine(member_obj):
                                    process_node(member_obj, member_name, f"{parent_path}.{name}", "dunder", is_method=True)
                            
                            # Properties
                            elif isinstance(member_obj, property) or inspect.isdatadescriptor(member_obj):
                                # Ensure it's not a standard method
                                if not inspect.isroutine(member_obj):
                                    process_node(member_obj, member_name, f"{parent_path}.{name}", "property", is_method=True)

    walk_module(root_mod, root_module_name)
    
    skeleton_list = list(skeletons.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(skeleton_list, f, indent=2)
        
    print(f"[+] Extracted {len(skeleton_list)} expansion skeletons from {root_module_name} into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_skeleton_expansions.py <module_name> <output_file.json>")
        sys.exit(1)
        
    extract_expansions(sys.argv[1], sys.argv[2])
