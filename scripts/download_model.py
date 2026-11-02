#!/usr/bin/env python3
"""
scripts/download_model.py - Automated Model Downloader for STRATA

Downloads and organizes pre-trained embedding, language (GGUF), and reranking models
into the standardized `models/` directory hierarchy expected by the engine.

Usage:
    python scripts/download_model.py                    # Downloads default lightweight models (~750 MB)
    python scripts/download_model.py --tier default     # Same as above (0.5B LLM + Nano Embedder)
    python scripts/download_model.py --tier medium      # 1.5B LLM + Nano Embedder (~1.4 GB)
    python scripts/download_model.py --tier heavy       # 7B LLM + Nano Embedder (~5.1 GB)
    python scripts/download_model.py --tier reranker    # Optional Jina v3.5 Reranker (~500 MB)
    python scripts/download_model.py --tier all         # Downloads all tested tiers
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    print(
        "[ERROR] huggingface_hub is required. Install it with:\n"
        "        pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_embedder_nano():
    target_dir = MODELS_DIR / "embeddings" / "jina-embeddings-v5-text-nano"
    print(f"\n[1/2] Checking Embedder: jina-embeddings-v5-text-nano (~350 MB)...")
    if (target_dir / "model.safetensors").exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading from Hugging Face: jinaai/jina-embeddings-v5-text-nano ...")
    snapshot_download(
        repo_id="jinaai/jina-embeddings-v5-text-nano",
        local_dir=str(target_dir),
    )
    print("      Embedder download complete!")


def download_llm_05b():
    target_dir = MODELS_DIR / "llms" / "qwen2.5-coder-0.5b-instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    print(f"\n[2/2] Checking LLM: Qwen2.5-Coder-0.5B-Instruct-GGUF (~398 MB)...")
    if (target_dir / filename).exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading {filename} from Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF ...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        filename=filename,
        local_dir=str(target_dir),
    )
    print("      LLM download complete!")


def download_llm_15b():
    target_dir = MODELS_DIR / "llms" / "qwen2.5-coder-1.5b-instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    print(f"\n[*] Checking LLM: Qwen2.5-Coder-1.5B-Instruct-GGUF (~986 MB)...")
    if (target_dir / filename).exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading {filename} from Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF ...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename=filename,
        local_dir=str(target_dir),
    )
    print("      LLM download complete!")


def download_llm_7b():
    target_dir = MODELS_DIR / "llms" / "Qwen2.5-Coder-7B-Instruct-GGUF"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    print(f"\n[*] Checking LLM: Qwen2.5-Coder-7B-Instruct-GGUF (~4.68 GB)...")
    if (target_dir / filename).exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading {filename} from Qwen/Qwen2.5-Coder-7B-Instruct-GGUF ...")
    hf_hub_download(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename=filename,
        local_dir=str(target_dir),
    )
    print("      LLM download complete!")


def download_reranker():
    target_dir = MODELS_DIR / "rerankers" / "jina-reranker-v3.5"
    print(f"\n[*] Checking Reranker: jina-reranker-v3.5 (~500 MB)...")
    if (target_dir / "model.safetensors").exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading from Hugging Face: jinaai/jina-reranker-v3.5 ...")
    snapshot_download(
        repo_id="jinaai/jina-reranker-v3.5",
        local_dir=str(target_dir),
    )
    print("      Reranker download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download and configure local AI models for STRATA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["default", "medium", "heavy", "reranker", "all"],
        default="default",
        help=(
            "Model tier to download:\n"
            "  default  : 0.5B LLM + Nano Embedder (~750 MB total, fast on CPU/GPU)\n"
            "  medium   : 1.5B LLM + Nano Embedder (~1.4 GB total)\n"
            "  heavy    : 7B LLM + Nano Embedder (~5.1 GB total)\n"
            "  reranker : Jina v3.5 Neural Cross-Encoder (~500 MB)\n"
            "  all      : Download all tested model variants"
        ),
    )
    args = parser.parse_args()

    print("==================================================")
    print(f"  STRATA Model Provisioner (Tier: {args.tier})")
    print("==================================================")
    print(f"Destination root: {MODELS_DIR}")

    if args.tier in ("default", "medium", "heavy", "all"):
        download_embedder_nano()

    if args.tier == "default":
        download_llm_05b()
    elif args.tier == "medium":
        download_llm_15b()
    elif args.tier == "heavy":
        download_llm_7b()
    elif args.tier == "reranker":
        download_reranker()
    elif args.tier == "all":
        download_llm_05b()
        download_llm_15b()
        download_llm_7b()
        download_reranker()

    print("\n==================================================")
    print("  Model provisioning verified successfully!")
    print("==================================================")
    print("Next steps:")
    print("  1. Compile domain lattice : python nstl_cli.py compile")
    print("  2. Validate domain nodes  : python nstl_cli.py validate")
    print("  3. Run interactive shell  : python nstl_cli.py shell")
    print("  4. Start REST API server  : python src/main.py")


if __name__ == "__main__":
    main()
