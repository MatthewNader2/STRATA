import os
import glob
import json

def merge_expansions(harvests_dir):
    """
    Repurposed Tier 3 expansion merger:
    Merges keywords and descriptions into base structural/enriched node files,
    strictly discarding any unapproved fields ('type', 'params', 'inputs', 'outputs', 'code').
    """
    expansion_files = glob.glob(os.path.join(harvests_dir, "*_EXPANSION_*.json"))
    
    for exp_file in expansion_files:
        basename = os.path.basename(exp_file)
        lib_name = basename.split("_EXPANSION_")[0].replace("skeleton_", "").replace("qwen_", "").replace("enriched_", "")
        
        target_file = os.path.join(harvests_dir, f"enriched_{lib_name}.json")
        if not os.path.exists(target_file):
            target_file = os.path.join(harvests_dir, f"structural_{lib_name}.json")
            
        if not os.path.exists(target_file):
            print(f"[-] Target base file not found for {lib_name}, skipping expansion {basename}.")
            continue
            
        with open(exp_file, 'r', encoding='utf-8') as f:
            try:
                exp_data = json.load(f)
            except Exception as e:
                print(f"[!] Error parsing expansion {exp_file}: {e}")
                continue
                
        with open(target_file, 'r', encoding='utf-8') as f:
            base_nodes = json.load(f)
            
        node_map = {n["cell_id"]: n for n in base_nodes if "cell_id" in n}
        
        if isinstance(exp_data, dict) and "items" in exp_data:
            exp_items = exp_data["items"]
        elif isinstance(exp_data, list):
            exp_items = exp_data
        else:
            exp_items = []
            
        merged_count = 0
        for item in exp_items:
            cell_id = item.get("cell_id")
            if not cell_id or cell_id not in node_map:
                continue
                
            base_node = node_map[cell_id]
            
            # Discard any unapproved authority fields if present
            for forbidden in ("type", "params", "inputs", "outputs", "code", "domain_implementations"):
                if forbidden in item:
                    print(f"[!] Warning: Discarded model field '{forbidden}' from expansion for {cell_id}")
                    
            if "keywords" in item and isinstance(item["keywords"], list):
                base_node["keywords"] = item["keywords"]
            if "description" in item and item["description"]:
                base_node["description"] = item["description"]
            merged_count += 1
            
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(base_nodes, f, indent=2)
            
        print(f"[+] Merged {merged_count} metadata entries from {basename} into {os.path.basename(target_file)}")

if __name__ == "__main__":
    import sys
    h_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harvests")
    merge_expansions(h_dir)

