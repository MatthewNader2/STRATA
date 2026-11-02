"""
tools/test_cli_all_domains.py

Evaluates the enriched STRATA corpus through the CLI (STRATAInteractiveShell):
1. Each domain/tree alone:
   - Pandas
   - NumPy
   - Scikit-Learn
   - OpenCV (cv2)
   - Matplotlib
2. Multi-domain / cross-tree pipelines:
   - Pandas -> Scikit-Learn
   - OpenCV -> Matplotlib
   - Pandas -> Matplotlib
   - NumPy -> OpenCV
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cli import STRATAInteractiveShell

TEST_CASES = [
    # =========================================================================
    # Part 1: Single Domain / Tree Alone
    # =========================================================================
    {
        'category': 'Single Tree: Pandas',
        'title': 'Pandas: Clean, Sort & Export',
        'prompt': 'read dataset from input.csv, drop missing values, sort by salary descending, and save output to cleaned.csv'
    },
    {
        'category': 'Single Tree: NumPy',
        'title': 'NumPy: Load, Compute Mean & Save',
        'prompt': 'load array from matrix.npy, calculate mean along axis 0, and save array to result.npy'
    },
    {
        'category': 'Single Tree: Scikit-Learn',
        'title': 'Scikit-Learn: Scale Features & Cluster',
        'prompt': 'scale features with StandardScaler and fit KMeans clustering model'
    },
    {
        'category': 'Single Tree: OpenCV (cv2)',
        'title': 'OpenCV: Load, Grayscale & Save',
        'prompt': 'load image from photo.jpg, convert color to grayscale with cvtColor, and save image to output.png'
    },
    {
        'category': 'Single Tree: Matplotlib',
        'title': 'Matplotlib: Line Plot & Save',
        'prompt': 'create plot figure from data and save figure to plot.png'
    },

    # =========================================================================
    # Part 2: Multi-Domain / Cross-Tree Pipelines
    # =========================================================================
    {
        'category': 'Multi-Domain: Pandas -> Scikit-Learn',
        'title': 'Tabular ETL into ML Clustering',
        'prompt': 'read dataset from data.csv, standardize features with StandardScaler, and train KMeans clustering'
    },
    {
        'category': 'Multi-Domain: OpenCV -> Matplotlib',
        'title': 'Computer Vision to Visualization',
        'prompt': 'read image from photo.jpg, compute histogram, and plot histogram'
    },
    {
        'category': 'Multi-Domain: Pandas -> Matplotlib',
        'title': 'Tabular ETL to Plotting',
        'prompt': 'read sales data from sales.csv, calculate total sum, and plot chart'
    },
    {
        'category': 'Multi-Domain: NumPy -> OpenCV',
        'title': 'Numerical Array to Image Processing',
        'prompt': 'load array from input.npy, apply gaussian blur, and save image to blurred.png'
    }
]


def setup_dummy_fixtures():
    """Generates realistic test fixtures for all domains to ensure end-to-end sandbox execution."""
    tabular_csv = (
        "id,name,department,region,sales,salary,age\n"
        "1,Alice,Engineering,North,150.5,85000,32\n"
        "2,Bob,Marketing,South,120.0,62000,28\n"
        "3,Charlie,Engineering,East,200.0,95000,45\n"
        "4,David,Sales,North,95.2,54000,24\n"
        "5,Eve,Marketing,West,180.0,68000,39\n"
        "6,Frank,Sales,South,310.0,71000,50\n"
        "7,Grace,Engineering,West,120.0,92000,29\n"
    )
    (PROJECT_ROOT / "input.csv").write_text(tabular_csv)
    (PROJECT_ROOT / "sales.csv").write_text(tabular_csv)

    rows = ["feature1,feature2,feature3,sales,salary,age"]
    for i in range(1, 35):
        rows.append(f"{100.0 + i*5.0},{1.0 + (i%5)*0.5},{5.0 + i*0.8},{100.0 + i*10.0},{50000.0 + i*1000.0},{20.0 + (i%30)}")
    numeric_csv = "\n".join(rows) + "\n"
    (PROJECT_ROOT / "data.csv").write_text(numeric_csv)
    (PROJECT_ROOT / "input_data.csv").write_text(numeric_csv)

    import numpy as np
    arr_2d = np.random.rand(50, 4).astype(np.float32)
    img_arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    np.save(str(PROJECT_ROOT / "matrix.npy"), arr_2d)
    np.save(str(PROJECT_ROOT / "input.npy"), img_arr)

    try:
        import cv2
        cv2.imwrite(str(PROJECT_ROOT / "photo.jpg"), img_arr)
    except Exception:
        pass


def run_all_domain_tests():
    print('=' * 80)
    print('STRATA MULTI-DOMAIN & SINGLE-TREE EVALUATION VIA CLI (Profile C)')
    print('=' * 80)

    setup_dummy_fixtures()

    shell = STRATAInteractiveShell(
        initial_profile='C',
        embedder='jina-embeddings-v5-text-nano',
        llm='qwen2.5-coder-1.5b-instruct'
    )

    summary_records = []

    for i, t in enumerate(TEST_CASES, 1):
        print()
        print("=" * 80)
        print(f"[*] [{i}/{len(TEST_CASES)}] {t['category']} — {t['title']}")
        print(f"    Prompt: \"{t['prompt']}\"")
        print("-" * 80)

        history_len_before = len(shell.history)
        t0 = time.perf_counter()

        shell.default(t['prompt'])

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if len(shell.history) > history_len_before:
            last_entry = shell.history[-1]
            sb_status = last_entry.get("sandbox_status", "UNKNOWN")
            is_passed = (sb_status == "PASSED")
            status = 'PASSED' if is_passed else 'FAILED'
            summary_records.append({
                'category': t['category'],
                'title': t['title'],
                'prompt': t['prompt'],
                'routed_path': last_entry.get('path', []),
                'code': last_entry.get('code', ''),
                'status': status,
                'latency_ms': last_entry.get('latency_ms', total_elapsed_ms),
                'route_ms': last_entry.get('route_ms', 0.0),
                'synth_ms': last_entry.get('synth_ms', 0.0),
                'exec_ms': last_entry.get('sandbox_ms', 0.0),
                'error': last_entry.get('sandbox_error', '') or (sb_status if not is_passed else '')
            })
        else:
            summary_records.append({
                'category': t['category'],
                'title': t['title'],
                'prompt': t['prompt'],
                'routed_path': [],
                'code': '',
                'status': 'NO_PATH',
                'latency_ms': total_elapsed_ms,
                'route_ms': 0.0,
                'synth_ms': 0.0,
                'exec_ms': 0.0,
                'error': 'No path found'
            })

    print()
    print("=" * 80)
    print("FINAL MULTI-DOMAIN & SINGLE-TREE CLI SUMMARY")
    print("=" * 80)

    for rec in summary_records:
        badge = f'[{rec["status"]: <12}]'
        path_str = ' -> '.join(rec['routed_path']) if rec['routed_path'] else 'NONE'
        print(f"{badge} {rec['category']} ({rec['title']})")
        print(f"              Routed: {path_str}")
        print(f"              Total Latency: {rec['latency_ms']:.1f} ms (Route: {rec['route_ms']:.1f}ms, Synth: {rec['synth_ms']:.1f}ms, Exec: {rec['exec_ms']:.1f}ms)")
        if rec['status'] != 'PASSED' and rec['error']:
            print(f"              Error: {rec['error']}")
        print()

    total_passed = sum(1 for r in summary_records if r['status'] == 'PASSED')
    print(f'Overall Results: {total_passed}/{len(summary_records)} Passed ({(total_passed/len(summary_records)*100):.1f}%)')
    print('=' * 80)


if __name__ == '__main__':
    run_all_domain_tests()
