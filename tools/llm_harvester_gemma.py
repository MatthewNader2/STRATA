import json
import sys
import os
import argparse
import time
from tqdm import tqdm
from llama_cpp import Llama

def run_llm_harvester(skeleton_file, output_file, model_path, limit=None):
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
        
    if limit:
        skeletons = skeletons[:limit]
        
    print("[*] Loading 26B MoE Model (Gemma4) into VRAM...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=True
    )
    t1 = time.time()
    print(f"[+] Model loaded in {t1 - t0:.2f} seconds.")
    
    generated_code = []
    generated_code.append(f"# Auto-synthesized by 26B MoE LLM Harvester from {skeleton_file}")
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

    print(f"[*] Processing {len(skeletons)} skeletons...")
    start_inference = time.time()
    
    for skel in tqdm(skeletons):
        prompt = prompt_template.format(
            name=skel["name"],
            contexts=", ".join(skel["contexts"]),
            params=", ".join(skel["params"]),
            doc=skel["doc"]
        )
        
        response = llm(
            prompt,
            max_tokens=1024,
            temperature=0.2,
            stop=["TARGET FUNCTION TO SYNTHESIZE:"],
            echo=False
        )
        
        output_text = response['choices'][0]['text'].strip()
        
        # Clean up response if it wrapped in markdown
        if "```python" in output_text:
            output_text = output_text.split("```python")[1].split("```")[0].strip()
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0].strip()
            
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
    parser.add_argument("--model-path", default=os.environ.get("STRATA_GEMMA_MODEL"), help="Path to the GGUF model (or set STRATA_GEMMA_MODEL)")
    parser.add_argument("--limit", type=int, help="Limit number of functions to process for testing")
    args = parser.parse_args()
    
    if not args.model_path:
        parser.error("--model-path or STRATA_GEMMA_MODEL is required")
    run_llm_harvester(args.skeleton, args.output, args.model_path, args.limit)
