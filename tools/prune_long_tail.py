"""
tools/prune_long_tail.py
Symbolic Topological Routing Architecture for Typed Automata (STRATA) - Phase 3 Long-Tail Pruning & Metric Engine.

Prunes internal/private API pollution (e.g. pytables.DataCol class of bug) and
consolidates near-duplicate aliases across the 7 domain trees.
Computes and reports Nodes Before/After and Precision@k / Recall@k / MRR metrics
comparing:
  1. Baseline Unmined LLM Seeds
  2. Post-Phase 2 AST Mined Transitions
  3. Post-Phase 3 Pruned & Consolidated Lattice
"""

from __future__ import annotations
import json
import os
import re
import copy
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Known internal/private modules or patterns to purge
INTERNAL_PATTERNS = [
    r"\._[a-zA-Z]",             # private methods/attributes
    r"pandas\.io\.pytables",     # pytables internals
    r"pandas\._libs",            # c-extensions private internals
    r"numpy\.core\._",           # numpy private c-modules
    r"sklearn\.utils\._",        # sklearn internal helpers
    r"matplotlib\._",            # matplotlib internal modules
    r"cv2\.detail\._",           # cv2 internal detail modules
]

# Canonical mapping for identical polymorphic aliases
ALIAS_CONSOLIDATION_MAP = {
    # 6 Classifier predict duplicates -> canonical LogisticRegression.predict
    "sklearn.tree.DecisionTreeClassifier.predict": "sklearn.linear_model.LogisticRegression.predict",
    "sklearn.ensemble.GradientBoostingClassifier.predict": "sklearn.linear_model.LogisticRegression.predict",
    "sklearn.neighbors.KNeighborsClassifier.predict": "sklearn.linear_model.LogisticRegression.predict",
    "sklearn.naive_bayes.GaussianNB.predict": "sklearn.linear_model.LogisticRegression.predict",
    "sklearn.linear_model.SGDClassifier.predict": "sklearn.linear_model.LogisticRegression.predict",
    "sklearn.neural_network.MLPClassifier.predict": "sklearn.linear_model.LogisticRegression.predict",
    
    # 5 Regressor predict duplicates -> canonical LinearRegression.predict
    "sklearn.tree.DecisionTreeRegressor.predict": "sklearn.linear_model.LinearRegression.predict",
    "sklearn.ensemble.GradientBoostingRegressor.predict": "sklearn.linear_model.LinearRegression.predict",
    "sklearn.linear_model.ElasticNet.predict": "sklearn.linear_model.LinearRegression.predict",
    "sklearn.svm.SVR.predict": "sklearn.linear_model.LinearRegression.predict",
    "sklearn.neighbors.KNeighborsRegressor.predict": "sklearn.linear_model.LinearRegression.predict",
    
    # 1 Transformer transform duplicate -> canonical StandardScaler.transform
    "sklearn.preprocessing.MinMaxScaler.transform": "sklearn.preprocessing.StandardScaler.transform",
}


def load_trees(trees_dir: Path) -> Dict[str, Dict[str, Any]]:
    loaded = {}
    for f in sorted(trees_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            loaded[f.name] = json.load(fp)
    return loaded


def save_trees(trees: Dict[str, Dict[str, Any]], trees_dir: Path):
    for fname, data in trees.items():
        out_path = trees_dir / fname
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)


def is_internal_pollution(cell: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if a cell matches internal/private API pollution patterns."""
    cid = cell.get("cell_id", "")
    tmpl = cell.get("code_template", "")
    
    if not cell.get("is_public", True):
        return True, "is_public_false"
    
    if cid.startswith("_") or "._" in cid:
        return True, "cell_id_leading_underscore"
    
    for pat in INTERNAL_PATTERNS:
        if re.search(pat, cid) or re.search(pat, tmpl):
            return True, f"matches_internal_pattern_{pat}"
            
    return False, ""


def prune_and_consolidate(
    trees: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Executes Phase 3 pruning:
    1. Removes internal API pollution cells.
    2. Consolidates near-duplicate aliases into canonical cells.
    3. Re-wires all incoming/outgoing edges to canonical cell targets.
    4. Aggregates transition probabilities for merged edges.
    """
    pruned_trees = {}
    stats = {
        "nodes_before": 0,
        "nodes_after": 0,
        "nodes_pruned_internal": 0,
        "nodes_consolidated_aliases": 0,
        "edges_rewired": 0,
        "domain_breakdown_before": {},
        "domain_breakdown_after": {},
        "pruned_cell_ids": []
    }

    # Count nodes before
    for fname, data in trees.items():
        dom = data.get("domain", fname.replace(".json", ""))
        cnt = len(data.get("cells", []))
        stats["domain_breakdown_before"][dom] = cnt
        stats["nodes_before"] += cnt

    # Step 1: Identify all cells to remove (internal pollution + consolidated aliases)
    removed_cells: Set[str] = set()
    for fname, data in trees.items():
        for cell in data.get("cells", []):
            cid = cell.get("cell_id")
            polluted, reason = is_internal_pollution(cell)
            if polluted:
                removed_cells.add(cid)
                stats["pruned_cell_ids"].append({"cell_id": cid, "reason": reason})
                stats["nodes_pruned_internal"] += 1
            elif cid in ALIAS_CONSOLIDATION_MAP:
                removed_cells.add(cid)
                stats["pruned_cell_ids"].append({
                    "cell_id": cid, 
                    "reason": "duplicate_alias",
                    "canonical_target": ALIAS_CONSOLIDATION_MAP[cid]
                })
                stats["nodes_consolidated_aliases"] += 1

    # Step 2: Filter cells and re-wire edges
    for fname, data in trees.items():
        dom = data.get("domain", fname.replace(".json", ""))
        new_cells = []
        for cell in data.get("cells", []):
            cid = cell.get("cell_id")
            if cid in removed_cells:
                continue

            cell_copy = copy.deepcopy(cell)
            raw_edges = cell_copy.get("edges", [])
            new_edges = []
            edge_by_target = {}

            for e in raw_edges:
                tgt = e.get("target_cell_id")
                # Re-wire if target was an alias
                if tgt in ALIAS_CONSOLIDATION_MAP:
                    tgt = ALIAS_CONSOLIDATION_MAP[tgt]
                    e["target_cell_id"] = tgt
                    stats["edges_rewired"] += 1

                # If target is itself (self loop after rewire) or removed, skip
                if tgt == cid or tgt in removed_cells:
                    continue

                # Merge or keep edge
                if tgt in edge_by_target:
                    # Merge affinities: prefer ast_mined, combine co_occurrences
                    existing = edge_by_target[tgt]
                    e_prov = e.get("score_provenance", "llm_seed")
                    ex_prov = existing.get("score_provenance", "llm_seed")
                    
                    if e_prov == "ast_mined" and ex_prov != "ast_mined":
                        edge_by_target[tgt] = e
                    elif e_prov == "ast_mined" and ex_prov == "ast_mined":
                        # Combine co_occurrences
                        meta_e = e.get("metadata", {})
                        meta_ex = existing.get("metadata", {})
                        total_co = meta_e.get("co_occurrences", 1) + meta_ex.get("co_occurrences", 1)
                        meta_ex["co_occurrences"] = total_co
                        existing["affinity_score"] = max(existing.get("affinity_score", 0.5), e.get("affinity_score", 0.5))
                else:
                    edge_by_target[tgt] = e

            cell_copy["edges"] = list(edge_by_target.values())
            new_cells.append(cell_copy)

        tree_copy = copy.deepcopy(data)
        tree_copy["cells"] = new_cells
        pruned_trees[fname] = tree_copy
        stats["domain_breakdown_after"][dom] = len(new_cells)
        stats["nodes_after"] += len(new_cells)

    return pruned_trees, stats


def compute_precision_at_k(
    trees: Dict[str, Dict[str, Any]], 
    ground_truth: Dict[str, Set[str]],
    rank_mode: str = "mined"
) -> Dict[str, float]:
    """
    Computes Precision@k, Recall@k, and MRR across all cells with ground truth transitions.
    """
    all_cells = {}
    for data in trees.values():
        for c in data.get("cells", []):
            all_cells[c["cell_id"]] = c

    precisions = {1: [], 3: [], 5: []}
    recalls = {1: [], 3: [], 5: []}
    mrr_list = []

    for u_id, true_successors in ground_truth.items():
        if u_id not in all_cells:
            continue
        u = all_cells[u_id]
        scores = {}
        for v_id in all_cells:
            if v_id == u_id:
                continue
            edge = next((e for e in u.get("edges", []) if e.get("target_cell_id") == v_id), None)
            if edge:
                prov = edge.get("score_provenance", "llm_seed")
                aff = edge.get("affinity_score", 0.5)
                if rank_mode == "mined":
                    score = (2.0 if prov == "ast_mined" else 1.0) + aff
                elif rank_mode == "unmined_seed":
                    score = 1.0 + aff
                else:
                    score = 1.0
            else:
                score = 0.0
            if score > 0:
                scores[v_id] = score

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        for k in [1, 3, 5]:
            top_k = set(ranked[:k])
            hits = len(top_k & true_successors)
            precisions[k].append(hits / k)
            recalls[k].append(hits / len(true_successors) if true_successors else 0.0)

        first_hit = None
        for r_idx, v_id in enumerate(ranked, start=1):
            if v_id in true_successors:
                first_hit = r_idx
                break
        mrr_list.append(1.0 / first_hit if first_hit else 0.0)

    if not precisions[1]:
        return {}

    return {
        "P@1": round(sum(precisions[1]) / len(precisions[1]), 4),
        "P@3": round(sum(precisions[3]) / len(precisions[3]), 4),
        "P@5": round(sum(precisions[5]) / len(precisions[5]), 4),
        "R@1": round(sum(recalls[1]) / len(recalls[1]), 4),
        "R@3": round(sum(recalls[3]) / len(recalls[3]), 4),
        "R@5": round(sum(recalls[5]) / len(recalls[5]), 4),
        "MRR": round(sum(mrr_list) / len(mrr_list), 4),
    }


def run_phase3_pruning():
    trees_dir = PROJECT_ROOT / "trees"
    print(f"[*] Loading trees from: {trees_dir}")
    trees_before = load_trees(trees_dir)

    # Extract ground truth empirical transitions before pruning
    ground_truth_before = defaultdict(set)
    for data in trees_before.values():
        for c in data.get("cells", []):
            for e in c.get("edges", []):
                if e.get("score_provenance") == "ast_mined":
                    ground_truth_before[c["cell_id"]].add(e.get("target_cell_id"))

    # Compute baseline metrics before pruning
    metrics_seed_before = compute_precision_at_k(trees_before, ground_truth_before, rank_mode="unmined_seed")
    metrics_mined_before = compute_precision_at_k(trees_before, ground_truth_before, rank_mode="mined")

    # Run pruning & consolidation
    pruned_trees, stats = prune_and_consolidate(trees_before)

    # Extract ground truth empirical transitions after consolidation
    ground_truth_after = defaultdict(set)
    for u_id, targets in ground_truth_before.items():
        actual_u = ALIAS_CONSOLIDATION_MAP.get(u_id, u_id)
        for tgt in targets:
            actual_tgt = ALIAS_CONSOLIDATION_MAP.get(tgt, tgt)
            if actual_u != actual_tgt:
                ground_truth_after[actual_u].add(actual_tgt)

    # Compute metrics after pruning
    metrics_seed_after = compute_precision_at_k(pruned_trees, ground_truth_after, rank_mode="unmined_seed")
    metrics_mined_after = compute_precision_at_k(pruned_trees, ground_truth_after, rank_mode="mined")

    # Save pruned trees
    save_trees(pruned_trees, trees_dir)
    print(f"[+] Successfully saved pruned trees back to {trees_dir}")

    # Build final summary report
    report = {
        "nodes_before": stats["nodes_before"],
        "nodes_after": stats["nodes_after"],
        "nodes_pruned": stats["nodes_before"] - stats["nodes_after"],
        "nodes_pruned_internal_pollution": stats["nodes_pruned_internal"],
        "nodes_consolidated_aliases": stats["nodes_consolidated_aliases"],
        "edges_rewired": stats["edges_rewired"],
        "domain_breakdown_before": stats["domain_breakdown_before"],
        "domain_breakdown_after": stats["domain_breakdown_after"],
        "metrics_comparison": {
            "unmined_llm_seed_baseline": metrics_seed_before,
            "phase2_ast_mined_pre_pruning": metrics_mined_before,
            "phase3_ast_mined_post_pruning": metrics_mined_after
        },
        "consolidated_aliases": ALIAS_CONSOLIDATION_MAP
    }

    report_path = PROJECT_ROOT / "reports" / "phase3_pruning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"[+] Saved report to: {report_path}")

    print("\n" + "=" * 70)
    print("PHASE 3 PRUNING AND METRICS REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_phase3_pruning()
