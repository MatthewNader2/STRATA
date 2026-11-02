import os
import sys
import json
import logging

# Resolve the project root from this file's location (cwd-independent) so the
# src/ modules import correctly regardless of where the script is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import ModelManager
from internal_rag import LocalRAG

# Constants
TREES_DIR = os.path.join(_PROJECT_ROOT, "trees")
MACRO_DIR = os.path.join(TREES_DIR, "macro")

# MacroCell JSON Schema for llama-cpp-python
MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "node_type": {"type": "string", "enum": ["macro"]},
                    "node_role": {"type": "string", "enum": ["macro"]},
                    "stage": {"type": "integer"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "docstring": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "inputs": {
                        "type": "object",
                        "description": "Map of port name -> PortSignature dict (type_name, state, required, ...). Must mirror the port schema used by trees/*.json cells.",
                        "type_name": {"type": "string"},
                        "state": {"type": "string"}
                    },
                    "outputs": {
                        "type": "object",
                        "description": "Map of port name -> PortSignature dict (type_name, state, required, ...). Must mirror the port schema used by trees/*.json cells.",
                        "type_name": {"type": "string"},
                        "state": {"type": "string"}
                    },
                    "algorithmic_steps": {"type": "array", "items": {"type": "string"}},
                    "sub_cells": {"type": "array", "items": {"type": "string"}},
                    "internal_topology": {
                        "type": "object"
                    }
                },
                "required": ["cell_id", "node_type", "stage", "keywords", "inputs", "outputs", "algorithmic_steps", "sub_cells", "internal_topology"]
            }
        }
    },
    "required": ["cells"]
}
# NOTE: the discriminator is `node_type: "macro"` (NOT `type`) to match the
# engine loader contract in src/lattice.py, which classifies a node as a
# MacroCell via node_type/node_role in {"macro", "higher_order"} or macro_*.


def generate_macro_nodes():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MacroGenerator")
    
    os.makedirs(MACRO_DIR, exist_ok=True)
    
    # 1. Initialize Profile B explicitly (local reasoning)
    logger.info("Initializing ModelManager with BenchmarkProfile_B...")
    ModelManager.get_instance().initialize_profile("B")
    
    # 2. Initialize LocalRAG to fetch existing Micro-Nodes context
    logger.info("Initializing LocalRAG engine...")
    rag = LocalRAG(trees_dir=TREES_DIR)
    
    # Predefined target concepts
    concepts = [
        "A* Pathfinding",
        "ETL Pipeline",
        "REST API CRUD",
        "Binary Search"
    ]
    
    for concept in concepts:
        logger.info(f"Generating Macro-Node for concept: {concept}")
        
        # Query LocalRAG for available Micro-Nodes
        rag_context = rag.get_relevant_context(f"Implementation details for {concept}", top_k=5)
        
        system_prompt = f"""You are an expert Software Architect defining abstract logic graphs.
Your task is to implement the algorithmic logic for: {concept}.

You must output a strictly formatted JSON object matching the MacroCell Schema.

Available Scraped Micro-Nodes (Selection Pool):
{rag_context}

RULES FOR SUB_CELLS:
1. ALWAYS try to use the IDs of the scraped Micro-Nodes listed above if they fit the algorithmic step.
2. If an algorithmic step requires functionality that is MISSING from the scraped pool, you are AUTHORIZED to invent a clean, semantic placeholder ID (e.g., 'micro_custom_sort_step'). The system will catch this placeholder later.
"""
        user_prompt = f"Write the Macro-Node JSON for {concept}."
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.info(f"Invoking LLM for {concept}...")
        try:
            result_text = ModelManager.get_instance().generate_text(
                prompt=full_prompt, 
                max_tokens=2048
            )
            
            # Validate JSON
            macro_json = json.loads(result_text)
            
            filename = concept.lower().replace(" ", "_").replace("*", "star") + ".json"
            out_path = os.path.join(MACRO_DIR, filename)
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(macro_json, f, indent=2)
                
            logger.info(f"Successfully generated and saved: {out_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate Macro-Node for {concept}: {e}")

if __name__ == "__main__":
    generate_macro_nodes()
