import json
import sys
import os
import argparse
import time
from tqdm import tqdm

# Add parent directory to path to import inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import ModelManager

def run_llm_harvester(skeleton_file, output_file, limit=None):
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
        
    if limit:
        skeletons = skeletons[:limit]
        
    print(f"[*] Initializing ModelManager Profile C for LLM harvesting (Strict JSON Schema)...")
    manager = ModelManager.get_instance()
    manager.initialize_profile("C")
    llm = manager.profile
    
    generated_code = []
    generated_code.append(f"# Auto-synthesized by LLM Harvester from {skeleton_file}")
    generated_code.append("import typing")
    generated_code.append("import pandas as pd")
    generated_code.append("import numpy as np")
    generated_code.append("import cv2")
    generated_code.append("")

    prompt_template = """You are an AST parameter generator.
Function: {name}
Params: {params}

Generate 1 configuration variant JSON object.
Rules for args_string:
- ONLY use literal values (numbers, strings, True/False) or valid library constants.
- NEVER use imaginary variable names. If a parameter requires a dynamic variable, you MUST leave args_string empty "".

JSON Keys:
- suffix: A short, unique identifier describing the variant. If args_string is empty, use "default".
- args_string: The exact string of python arguments. 
- keywords: 3-5 relevant search keywords.
"""

    json_schema = {
        "type": "object",
        "properties": {
            "suffix": {"type": "string"},
            "args_string": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["suffix", "args_string", "keywords"]
    }

    print(f"[*] Processing {len(skeletons)} skeletons with strict JSON grammar...")
    start_time = time.time()
    
    for skel in tqdm(skeletons):
        name = skel["name"]
        parent = skel["contexts"][0]
        
        prompt = prompt_template.format(
            name=name,
            params=", ".join(skel["params"]),
            doc=skel["doc"]
        )
        
        response = llm.generate_text(prompt, max_tokens=512, schema=json_schema)
        print(f"Raw response for {name}: {response}")
        
        # Try to parse JSON
        try:
            # Clean up markdown if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            variants = json.loads(response)
            if not isinstance(variants, list):
                variants = [variants]
        except Exception as e:
            # Fallback if parsing fails
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {response}")
            variants = [{"suffix": "default", "args_string": "", "keywords": [name]}]

        for var in variants:
            suffix = var.get("suffix", "default")
            args_str = var.get("args_string", "")
            keywords = ", ".join(var.get("keywords", [name]))
            
            safe_name = f"{parent}_{name}_{suffix}".replace(".", "_")
            
            if skel["is_method"]:
                if args_str:
                    invocation = f"output_var = input_var.{name}({args_str})"
                else:
                    invocation = f"output_var = input_var.{name}()"
            else:
                if args_str:
                    invocation = f"output_var = {parent}.{name}(input_var, {args_str})"
                else:
                    invocation = f"output_var = {parent}.{name}(input_var)"
                    
            stub = f'''def {safe_name}(input_var: 'any') -> 'any_computed':
    """
    Keywords: {parent}.{name}, {keywords}
    """
    {invocation}
'''
            generated_code.append(stub)

    end_time = time.time()
    total_time = end_time - start_time
    avg = total_time / len(skeletons) if skeletons else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(generated_code))
        
    print(f"[+] Successfully harvested {len(skeletons)} skeletons into {output_file}")
    print(f"[!] Performance stats:")
    print(f"    - Total Inference Time: {total_time:.2f} seconds")
    print(f"    - Average Time per node: {avg:.2f} seconds/node")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton", help="Input skeleton JSON file")
    parser.add_argument("output", help="Output python file for AST compilation")
    parser.add_argument("--limit", type=int, help="Limit number of functions to process for testing")
    args = parser.parse_args()
    
    run_llm_harvester(args.skeleton, args.output, args.limit)
