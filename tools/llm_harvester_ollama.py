import json
import sys
import os
import argparse
import time
import requests
from tqdm import tqdm

def query_ollama(prompt, model=None):
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    model = model or os.environ.get("OLLAMA_MODEL", "gemma4:26b")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "stop": ["TARGET FUNCTION TO SYNTHESIZE:"]
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying Ollama API: {e}")
        return ""

def run_llm_harvester(skeleton_file, output_file, limit=None):
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
        
    if limit:
        skeletons = skeletons[:limit]
        
    print("[*] Connecting to local Ollama API (gemma4:26b)...")
    
    generated_code = []
    generated_code.append(f"# Auto-synthesized by Ollama (gemma4:26b) LLM Harvester from {skeleton_file}")
    generated_code.append("import typing")
    generated_code.append("import pandas as pd")
    generated_code.append("import numpy as np")
    generated_code.append("import cv2")
    generated_code.append("")

    prompt_template = """You are an expert Python and STRATA Node Engineer.
I will give you a function signature and docstring. You must generate 1 to 3 self-contained, highly useful Python AST wrapper functions that implement this function for the STRATA engine.

RULES:
1. The wrapper function must take EXACTLY ONE primary input named `input_var`.
2. The wrapper function must assign its final result to a variable named `output_var`.
3. If the original function requires multiple parameters (like `cv2.cvtColor(src, code)` or `dropna(axis)`), you MUST create specific variants of the wrapper function where you HARDCODE the secondary parameters to useful defaults. Do NOT leave required parameters missing!
4. Include a detailed docstring with a `Keywords:` section containing rich, semantic keywords separated by commas (e.g., `Keywords: image, color, grayscale, conversion`).
5. Only output the raw python code. No markdown formatting, no explanations.

Example Output format:
def cv2_cvtColor_BGR2GRAY(input_var: 'any') -> 'any_computed':
    '''
    Converts an image from BGR to grayscale.
    Keywords: cv2.cvtColor, grayscale, black and white, color conversion, image
    '''
    output_var = cv2.cvtColor(input_var, cv2.COLOR_BGR2GRAY)

---
TARGET FUNCTION TO SYNTHESIZE:
Name: {name}
Contexts (where it was found): {contexts}
Signature Params: {params}
Docstring Summary:
{doc}

Generate the wrapper functions now. Output ONLY Python code."""

    print(f"[*] Processing {len(skeletons)} skeletons via Ollama API...")
    start_inference = time.time()
    
    for skel in tqdm(skeletons):
        prompt = prompt_template.format(
            name=skel["name"],
            contexts=", ".join(skel["contexts"]),
            params=", ".join(skel["params"]),
            doc=skel["doc"]
        )
        
        output_text = query_ollama(prompt, model="gemma4:26b")
        
        # Clean up response if it wrapped in markdown
        if "```python" in output_text:
            output_text = output_text.split("```python")[1].split("```")[0].strip()
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0].strip()
        
        # Guard: skip blank/empty responses to prevent corrupting the output file
        if not output_text.strip():
            print(f"[!] WARNING: Empty response for {skel['name']} — skipping to avoid blank code injection.")
            continue
            
        generated_code.append(output_text)
        generated_code.append("\n")

    end_inference = time.time()
    total_time = end_inference - start_inference
    avg_time = total_time / len(skeletons) if skeletons else 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(generated_code))
        
    print(f"[+] Successfully harvested {len(skeletons)} skeletons into {output_file}")
    print(f"[!] Performance stats:")
    print(f"    - Total Inference Time: {total_time:.2f} seconds")
    print(f"    - Average Time per node: {avg_time:.2f} seconds/node")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton", help="Input skeleton JSON file")
    parser.add_argument("output", help="Output python file for AST compilation")
    parser.add_argument("--limit", type=int, help="Limit number of functions to process for testing")
    args = parser.parse_args()
    
    run_llm_harvester(args.skeleton, args.output, args.limit)
