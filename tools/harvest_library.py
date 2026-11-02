import inspect
import importlib
import sys
import os

def harvest_library(root_module_name, output_file):
    try:
        root_mod = importlib.import_module(root_module_name)
    except ImportError:
        print(f"Error: Could not import module '{root_module_name}'")
        sys.exit(1)

    generated_code = []
    generated_code.append(f"# Auto-harvested from {root_module_name}")
    generated_code.append("import typing")
    generated_code.append(f"import {root_module_name}")
    generated_code.append("")
    
    visited = set()
    count = 0

    def process_routine(obj, name, parent_name, is_method=False):
        nonlocal count
        in_type = "any"
        out_type = "any"
        
        try:
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            if not params:
                # If we explicitly know it takes 0 args, we still need {input_var} for STRATA chaining, 
                # so we fallback instead of skipping.
                first_param = "input_var"
            else:
                first_param = params[0]
                
            param_obj = sig.parameters.get(first_param)
            if param_obj and param_obj.annotation != inspect.Parameter.empty:
                if hasattr(param_obj.annotation, '__name__'):
                    in_type = param_obj.annotation.__name__
                else:
                    in_type = str(param_obj.annotation)
                    
            if sig.return_annotation != inspect.Signature.empty:
                if hasattr(sig.return_annotation, '__name__'):
                    out_type = sig.return_annotation.__name__
                else:
                    out_type = str(sig.return_annotation)
        except (ValueError, TypeError):
            # Fallback for native C/C++ extensions (like OpenCV cv2.cvtColor) that lack signature metadata
            first_param = "input_var"
                
        in_type = in_type.replace("typing.", "").replace("'", "")
        out_type = out_type.replace("typing.", "").replace("'", "")
                
        doc = inspect.getdoc(obj) or ""
        doc = doc.replace('"""', "'''")
        doc_lines = [f"    {line}" for line in doc.split('\n')[:5] if line.strip()]
        doc_str = "\n".join(doc_lines)
        if doc_str:
            doc_str += "\n"
            
        safe_name = f"{parent_name}_{name}".replace(".", "_")
        
        # Determine invocation syntax
        if is_method:
            # {input_var} is the object instance (self)
            invocation = f"output_var = input_var.{name}()"
        else:
            # {input_var} is the first argument to the function
            invocation = f"output_var = {parent_name}.{name}(input_var)"
            
        stub = f"""def {safe_name}({first_param}: '{in_type}') -> '{out_type}_computed':
    r\"\"\"
{doc_str}    Keywords: {parent_name}, {name}
    \"\"\"
    {invocation}
"""
        generated_code.append(stub)
        count += 1

    def walk_module(mod, parent_path):
        if mod in visited:
            return
        visited.add(mod)
        
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
                
            # If it's a submodule belonging to the same root package, recurse!
            if inspect.ismodule(obj):
                if getattr(obj, '__name__', '').startswith(root_module_name):
                    walk_module(obj, obj.__name__)
                    
            # If it's a class, extract its methods!
            elif inspect.isclass(obj):
                # Don't process classes imported from other libraries
                if getattr(obj, '__module__', '').startswith(root_module_name):
                    if obj not in visited:
                        visited.add(obj)
                        for method_name, method_obj in inspect.getmembers(obj):
                            if method_name.startswith("_"):
                                continue
                            if inspect.isfunction(method_obj) or inspect.ismethod(method_obj) or inspect.isroutine(method_obj):
                                process_routine(method_obj, method_name, f"{parent_path}.{name}", is_method=True)
                                
            # If it's a top-level function
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj):
                process_routine(obj, name, parent_path, is_method=False)

    # Start recursion
    walk_module(root_mod, root_module_name)
        
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(generated_code))
        
    print(f"[+] Recursively harvested {count} functions/methods from {root_module_name} into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python harvest_library.py <module_name> <output_file.py>")
        sys.exit(1)
        
    harvest_library(sys.argv[1], sys.argv[2])
