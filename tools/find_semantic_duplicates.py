import sys
import os
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference import ModelManager

def find_semantic_duplicates(threshold: float = 0.92):
    db_path = PROJECT_ROOT / "trees" / "lattice.db"
    if not db_path.exists():
        print(f"[ERROR] Database file not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT cell_id, domain_name, node_type, keywords, input_type, output_type FROM nodes WHERE node_type != 'special_variant'")
    rows = cursor.fetchall()
    conn.close()

    print(f"[+] Sampling {len(rows)} Master & Standard nodes for semantic duplicate discovery...")

    cell_records = []
    descriptions = []
    for r in rows:
        cell_id, domain, node_type, keywords, in_type, out_type = r
        desc = f"Function ID: {cell_id}. Keywords: {keywords or ''}. Input: {in_type}, Output: {out_type}."
        descriptions.append(desc)
        cell_records.append({
            "cell_id": cell_id,
            "domain_name": domain,
            "node_type": node_type,
            "description": desc
        })

    ModelManager.get_instance().initialize_profile("E")
    print("[+] Computing embeddings for semantic duplicate discovery...")
    
    batch_size = 256
    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        emb_batch = ModelManager.get_instance().get_embeddings(batch)
        embeddings.extend(emb_batch)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    print("[+] Computing cross-domain cosine similarities...")
    domain_indices = {}
    for idx, rec in enumerate(cell_records):
        d = rec["domain_name"]
        domain_indices.setdefault(d, []).append(idx)

    duplicate_clusters = []
    seen_pairs = set()
    domain_names = list(domain_indices.keys())

    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            d1, d2 = domain_names[i], domain_names[j]
            idx1, idx2 = domain_indices[d1], domain_indices[d2]
            m1 = emb_matrix[idx1]
            m2 = emb_matrix[idx2]
            sim_block = np.dot(m1, m2.T)
            match_i, match_j = np.where(sim_block >= threshold)
            for r, c in zip(match_i, match_j):
                score = float(sim_block[r, c])
                cell_a = cell_records[idx1[r]]
                cell_b = cell_records[idx2[c]]
                pair_key = tuple(sorted([str(cell_a["cell_id"]), str(cell_b["cell_id"])]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    duplicate_clusters.append({
                        "cell_a": str(cell_a["cell_id"]),
                        "domain_a": str(cell_a["domain_name"]),
                        "cell_b": str(cell_b["cell_id"]),
                        "domain_b": str(cell_b["domain_name"]),
                        "similarity_score": float(round(score, 4))
                    })
                    if len(duplicate_clusters) >= 1000:
                        break
            if len(duplicate_clusters) >= 1000:
                break
        if len(duplicate_clusters) >= 1000:
            break

    duplicate_clusters.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_duplicates = duplicate_clusters[:1000]

    print("\n" + "="*80)
    print(" SEMANTIC-DUPLICATE CLUSTERING REPORT")
    print("="*80)
    print(f" Total Cross-Library Semantic Duplicate Pairs Found (>= {threshold}): {len(duplicate_clusters)}")
    print("-" * 80)
    for dup in top_duplicates[:15]:
        print(f"   * [{dup['similarity_score']:.4f}] {dup['cell_a']} ({dup['domain_a']}) <---> {dup['cell_b']} ({dup['domain_b']})")
    print("="*80)

    # Write report to logs/semantic_duplicates_report.json
    report_file = PROJECT_ROOT / "logs" / "semantic_duplicates_report.json"
    report_data = {
        "threshold": threshold,
        "total_nodes_sampled": len(rows),
        "total_cross_domain_duplicates": len(duplicate_clusters),
        "duplicate_pairs": top_duplicates
    }
    print(f"[DEBUG] Writing {len(top_duplicates)} duplicate pairs to {report_file}...")
    with open(str(report_file), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"[+] Saved semantic duplicate report ({len(top_duplicates)} top pairs) to: {report_file}")

if __name__ == "__main__":
    find_semantic_duplicates(0.92)
