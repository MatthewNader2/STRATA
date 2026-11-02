"""
src/tree_merger.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Universal Knowledge Tree Merger & Schema Reconciler.

Merges newly harvested ground-truth types, states, ports, and bridge roles
with existing curated trees (preserving rich docstrings, semantic tags, and priorities).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

try:
    import config
    PROJECT_ROOT = config.PROJECT_ROOT
except ImportError:
    try:
        from . import config
        PROJECT_ROOT = config.PROJECT_ROOT
    except ImportError:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from schema import CellSchema, TreeSchema
    from universal_harvester import UniversalHarvester
    from log_config import get_logger
except ImportError:
    from .schema import CellSchema, TreeSchema
    from .universal_harvester import UniversalHarvester
    from .log_config import get_logger

logger = get_logger("tree_merger")


def merge_domain_tree(
    domain_name: str,
    package_name: Optional[str] = None,
    tree_path: Optional[Union[str, Path]] = None,
) -> TreeSchema:
    """
    Harvests fresh ground-truth cells from domain package using UniversalHarvester,
    and merges with existing tree JSON (preserving rich docstrings, semantic tags, and priority).
    """
    pkg = package_name or domain_name
    harvester = UniversalHarvester(
        domain_name=domain_name,
        package_name=pkg
    )

    new_cells = harvester.harvest_all()
    trees_dir = Path(PROJECT_ROOT) / "trees"
    t_path = Path(tree_path) if tree_path else trees_dir / f"{domain_name}.json"
    existing_paths = []
    if t_path.exists():
        existing_paths.append(t_path)
    ckpt = Path(PROJECT_ROOT) / "nstl_enrichment" / "checkpoints" / f"{domain_name}.json"
    if ckpt.exists():
        existing_paths.append(ckpt)

    if existing_paths:
        merged_cells = UniversalHarvester.enrich_from_existing_trees(
            domain_name=domain_name,
            new_cells=new_cells,
            existing_tree_paths=existing_paths
        )
    else:
        merged_cells = new_cells

    tree = TreeSchema(
        domain=domain_name,
        cells=merged_cells
    )

    t_path.parent.mkdir(parents=True, exist_ok=True)
    with open(t_path, "w", encoding="utf-8") as f:
        f.write(tree.model_dump_json(indent=2))

    logger.info(f"[{domain_name}] Successfully merged and saved {len(merged_cells)} cells to {t_path}")
    return tree


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge and update domain knowledge trees")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. pandas, numpy, cv2)")
    parser.add_argument("--package", default=None, help="Python package name if different from domain")
    args = parser.parse_args()

    merge_domain_tree(args.domain, args.package)

