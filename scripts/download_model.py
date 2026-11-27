#!/usr/bin/env python3
"""
scripts/download_model.py - Automated Model Downloader for STRATA v2.

Downloads and organizes pre-trained embedding, language (GGUF), and reranking models
into the `models/` directory hierarchy expected by the engine.

Usage:
    python3 scripts/download_model.py                    # Default tier (Qwen3.5-0.8B + Nano Embedder)
    python3 scripts/download_model.py --tier 0.8b        # Explicit Qwen3.5-0.8B-Instruct (~533 MB) + Embedder
    python3 scripts/download_model.py --tier 0.5b        # Legacy ultra-light Qwen2.5-Coder-0.5B (~398 MB)
    python3 scripts/download_model.py --tier medium      # Qwen2.5-Coder-1.5B (~986 MB) + Embedder
    python3 scripts/download_model.py --tier heavy       # Qwen2.5-Coder-7B (~4.68 GB) + Embedder
    python3 scripts/download_model.py --tier reranker    # Optional Jina v3.5 Reranker (~500 MB)
    python3 scripts/download_model.py --tier all         # All supported local tiers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print(
        "[ERROR] huggingface_hub is required. Install it with:\n"
        "        pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_embedder_nano() -> None:
    target_dir = MODELS_DIR / "embeddings" / "jina-embeddings-v5-text-nano"
    print("\n[1/2] Checking Embedder: jina-embeddings-v5-text-nano (~350 MB)...")
    if (target_dir / "model.safetensors").exists():
        print(f"      Already present in {target_dir}")
        return
    print("      Downloading from Hugging Face: jinaai/jina-embeddings-v5-text-nano ...")
    snapshot_download(
        repo_id="jinaai/jina-embeddings-v5-text-nano",
        local_dir=str(target_dir),
    )
    print("      Embedder download complete!")


def download_llm_08b() -> None:
    target_dir = MODELS_DIR / "llms" / "qwen3.5-0.8b-instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "Qwen3.5-0.8B-Q4_K_M.gguf"
    print("\n[2/2] Checking LLM: Qwen3.5-0.8B-Instruct (Q4_K_M, ~533 MB)...")
    if (target_dir / filename).exists():
        print(f"      Already present in {target_dir}")
        return
    print(f"      Downloading {filename} from unsloth/Qwen3.5-0.8B-GGUF ...")
    hf_hub_download(
        repo_id="unsloth/Qwen3.5-0.8B-GGUF",
        filename=filename,
        local_dir=str(target_dir),
    )
    print("      LLM download complete!")


def download_llm_05b() -> None:
    target_dir = MODELS_DIR / "llms" / "qwen2.5-coder-0.5b-instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    print("\n[*] Checking LLM: Qwen2.5-Coder-0.5B-Instruct-GGUF (~398 MB)...")
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


def download_llm_15b() -> None:
    target_dir = MODELS_DIR / "llms" / "qwen2.5-coder-1.5b-instruct"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    print("\n[*] Checking LLM: Qwen2.5-Coder-1.5B-Instruct-GGUF (~986 MB)...")
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


def download_llm_7b() -> None:
    target_dir = MODELS_DIR / "llms" / "Qwen2.5-Coder-7B-Instruct-GGUF"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    print("\n[*] Checking LLM: Qwen2.5-Coder-7B-Instruct-GGUF (~4.68 GB)...")
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


def download_reranker() -> None:
    target_dir = MODELS_DIR / "rerankers" / "jina-reranker-v3.5"
    print("\n[*] Checking Reranker: jina-reranker-v3.5 (~500 MB)...")
    if (target_dir / "model.safetensors").exists():
        print(f"      Already present in {target_dir}")
        return
    print("      Downloading from Hugging Face: jinaai/jina-reranker-v3.5 ...")
    snapshot_download(
        repo_id="jinaai/jina-reranker-v3.5",
        local_dir=str(target_dir),
    )
    print("      Reranker download complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and configure local AI models for STRATA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["default", "0.8b", "0.5b", "medium", "heavy", "reranker", "all"],
        default="default",
        help=(
            "Model tier to download:\n"
            "  default  : Qwen3.5-0.8B + Nano Embedder (~880 MB total, STRATA v2 default)\n"
            "  0.8b     : Same as default (Qwen3.5-0.8B-Instruct + Nano Embedder)\n"
            "  0.5b     : Qwen2.5-Coder-0.5B + Nano Embedder (~750 MB total)\n"
            "  medium   : Qwen2.5-Coder-1.5B + Nano Embedder (~1.4 GB total)\n"
            "  heavy    : Qwen2.5-Coder-7B + Nano Embedder (~5.1 GB total)\n"
            "  reranker : Jina v3.5 Neural Cross-Encoder (~500 MB)\n"
            "  all      : Download all supported model tiers"
        ),
    )
    args = parser.parse_args()

    print("==================================================")
    print(f"  STRATA Model Provisioner (Tier: {args.tier})")
    print("==================================================")
    print(f"Destination root: {MODELS_DIR}")

    if args.tier in ("default", "0.8b", "0.5b", "medium", "heavy", "all"):
        download_embedder_nano()

    if args.tier in ("default", "0.8b"):
        download_llm_08b()
    elif args.tier == "0.5b":
        download_llm_05b()
    elif args.tier == "medium":
        download_llm_15b()
    elif args.tier == "heavy":
        download_llm_7b()
    elif args.tier == "reranker":
        download_reranker()
    elif args.tier == "all":
        download_llm_08b()
        download_llm_05b()
        download_llm_15b()
        download_llm_7b()
        download_reranker()

    print("\n==================================================")
    print("  Model provisioning verified successfully!")
    print("==================================================")
    print("Next steps:")
    print("  1. Compile domain lattice : python3 strata_cli.py compile --clean")
    print("  2. Validate domain nodes  : python3 strata_cli.py validate")
    print("  3. Audit live bridges     : python3 strata_cli.py audit")
    print("  4. Run interactive shell  : python3 strata_cli.py shell")


if __name__ == "__main__":
    main()
