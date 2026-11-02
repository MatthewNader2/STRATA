"""
tools/reharvest_domain_trees.py

Executes IntelligentHarvester systematically across domain submodules for:
  - numpy: numpy, numpy.linalg, numpy.fft, numpy.random
  - sklearn: sklearn.preprocessing, sklearn.linear_model, sklearn.model_selection, sklearn.ensemble, sklearn.cluster, sklearn.decomposition, sklearn.metrics, sklearn.neighbors
  - cv2: cv2
  - scipy: scipy.stats, scipy.optimize, scipy.signal, scipy.spatial, scipy.interpolate, scipy.integrate, scipy.linalg, scipy.ndimage, scipy.special
  - matplotlib: matplotlib.pyplot
  - pandas: pandas, pandas.DataFrame

Preserves all curated seeds (source_priority <= 10) and merges newly-harvested callables.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.harvester import IntelligentHarvester

DOMAIN_CONFIGS = {
    "numpy": {
        "package_name": "numpy",
        "submodules": ["numpy", "numpy.linalg", "numpy.fft", "numpy.random"],
        "target_file": PROJECT_ROOT / "trees" / "numpy.json"
    },
    "sklearn": {
        "package_name": "sklearn",
        "submodules": [
            "sklearn.preprocessing",
            "sklearn.linear_model",
            "sklearn.model_selection",
            "sklearn.svm",
            "sklearn.tree",
            "sklearn.naive_bayes",
            "sklearn.ensemble",
            "sklearn.cluster",
            "sklearn.decomposition",
            "sklearn.metrics",
            "sklearn.neighbors",
            "sklearn.feature_extraction",
            "sklearn.pipeline"
        ],
        "target_file": PROJECT_ROOT / "trees" / "sklearn.json"
    },
    "cv2": {
        "package_name": "cv2",
        "submodules": ["cv2"],
        "target_file": PROJECT_ROOT / "trees" / "cv2.json"
    },
    "scipy": {
        "package_name": "scipy",
        "submodules": [
            "scipy.stats",
            "scipy.optimize",
            "scipy.signal",
            "scipy.spatial",
            "scipy.interpolate",
            "scipy.integrate",
            "scipy.linalg",
            "scipy.ndimage",
            "scipy.special",
            "scipy.cluster"
        ],
        "target_file": PROJECT_ROOT / "trees" / "scipy.json"
    },
    "matplotlib": {
        "package_name": "matplotlib",
        "submodules": ["matplotlib.pyplot"],
        "target_file": PROJECT_ROOT / "trees" / "matplotlib.json"
    },
    "pandas": {
        "package_name": "pandas",
        "submodules": ["pandas"],
        "target_file": PROJECT_ROOT / "trees" / "pandas.json"
    }
}


def harvest_domain(domain: str) -> None:
    cfg = DOMAIN_CONFIGS.get(domain)
    if not cfg:
        print(f"[!] Unknown domain '{domain}'")
        return

    print(f"\n{'='*70}\n[*] Harvesting domain: {domain.upper()} (submodules: {cfg['submodules']})\n{'='*70}")
    harvester = IntelligentHarvester(domain=domain, package_name=cfg["package_name"])
    new_cells = harvester.harvest_modules(cfg["submodules"])

    if domain == "pandas":
        import pandas as pd
        seen_ids = {c.cell_id for c in new_cells}
        for name in dir(pd.DataFrame):
            if name.startswith("_"):
                continue
            obj = getattr(pd.DataFrame, name, None)
            if obj is not None and callable(obj):
                cell = harvester.harvest_function(name, obj, parent_mod_name="pandas.DataFrame")
                if cell and cell.cell_id not in seen_ids:
                    new_cells.append(cell)
                    seen_ids.add(cell.cell_id)

    print(f"[+] Introspected {len(new_cells)} public callables for {domain}")
    harvester.merge_and_save(new_cells, cfg["target_file"])
    print(f"[+] Saved merged tree to {cfg['target_file']}")


if __name__ == "__main__":
    domains_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["numpy", "sklearn", "cv2", "scipy", "pandas", "matplotlib"]
    for d in domains_to_run:
        harvest_domain(d)
