# STRATA Model Provisioning & Specifications

This document details the recommended, tested, and supported AI models for the **STRATA** (Neuro-Symbolic Topological Lattice) synthesis engine, where each model belongs in the directory layout, and how to download and configure them.

---

## 1. Architecture: The Neuro-Symbolic Duality

STRATA uses a dual-engine architecture:
- **Neural Layer (Semantic Extraction & Intent Translation)**: Uses local dense embedding models and GGUF quantized language models to parse natural language, embed user intent, and guide topological beam searches.
- **Symbolic Layer (Constructive Category Theory)**: Deterministically handles all lattice routing, monadic typestate unification, AST code synthesis, and sandboxed execution without relying on stochastic token generation for code correctness.

Because the symbolic engine guarantees 100% type soundness, you do not need a massive 70B parameter model. Lightweight 0.5B–1.5B models work exceptionally well for semantic intent translation while remaining fast on consumer laptops and development machines.

---

## 2. Recommended & Tested Models

| Model Role | Identifier / Hugging Face Source | Format | Local Directory Path | Footprint | Recommended Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Default Embedder** | [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | Safetensors / ST | `models/embeddings/jina-embeddings-v5-text-nano/` | ~350 MB | Default retrieval & semantic similarity scoring across all platforms |
| **Default LLM** *(Recommended)* | [`Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) | GGUF (`q4_k_m`) | `models/llms/qwen2.5-coder-0.5b-instruct/` | ~398 MB | Fast local execution, CI/CD, development laptops, CPU-friendly |
| **Medium LLM** | [`Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) | GGUF (`q4_k_m`) | `models/llms/qwen2.5-coder-1.5b-instruct/` | ~986 MB | Higher semantic reasoning capacity for ambiguous multi-clause prompts |
| **Heavy LLM** | [`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | GGUF (`q4_k_m`) | `models/llms/Qwen2.5-Coder-7B-Instruct-GGUF/` | ~4.68 GB | Full production benchmark evaluation (requires GPU or >=8 GB RAM) |
| **Optional Reranker** | [`jinaai/jina-reranker-v3.5`](https://huggingface.co/jinaai/jina-reranker-v3.5) | Safetensors / ST | `models/rerankers/jina-reranker-v3.5/` | ~500 MB | Deep cross-encoder candidate reranking (`NSTL_USE_RERANKER=1`) |

---

## 3. Directory Layout Expected by the Engine

The engine resolves paths automatically relative to `models/` (configured in `src/config.py`):

```
models/
├── embeddings/
│   └── jina-embeddings-v5-text-nano/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
├── llms/
│   ├── qwen2.5-coder-0.5b-instruct/
│   │   └── qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
│   ├── qwen2.5-coder-1.5b-instruct/            # Optional
│   │   └── qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
│   └── Qwen2.5-Coder-7B-Instruct-GGUF/          # Optional
│       └── qwen2.5-coder-7b-instruct-q4_k_m.gguf
└── rerankers/                                   # Optional
    └── jina-reranker-v3.5/
        ├── config.json
        ├── model.safetensors
        └── ...
```

---

## 4. Downloading the Models

### Automated Download (Recommended)

Run the bundled download utility from the repository root:

```bash
# 1. Download default tested models (~750 MB total; 0.5B LLM + Nano Embedder)
python scripts/download_model.py

# 2. (Optional) Download 1.5B medium tier model
python scripts/download_model.py --tier medium

# 3. (Optional) Download 7B heavy tier model
python scripts/download_model.py --tier heavy

# 4. (Optional) Download neural cross-encoder reranker
python scripts/download_model.py --tier reranker

# 5. Download all tested variants
python scripts/download_model.py --tier all
```

### Manual Download via Hugging Face CLI

If you prefer using `huggingface-cli` directly:

```bash
# Install Hugging Face Hub CLI
pip install "huggingface_hub[cli]>=0.23"

# Download Default Embedder
huggingface-cli download jinaai/jina-embeddings-v5-text-nano \
  --local-dir models/embeddings/jina-embeddings-v5-text-nano \
  --local-dir-use-symlinks False

# Download Default LLM (0.5B Coder Q4_K_M)
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF \
  qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
  --local-dir models/llms/qwen2.5-coder-0.5b-instruct \
  --local-dir-use-symlinks False
```

---

## 5. Hardware & Environment Recommendations

| Workload | Recommended Model Tier | Minimum RAM | Recommended Hardware | Execution Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Development & Testing** | `default` (0.5B + Nano) | 4 GB | Any modern CPU or integrated GPU | Real-time (< 1.5s / query) |
| **Extended Pipeline Synthesis** | `medium` (1.5B + Nano) | 8 GB | CPU with 4 cores or 2GB VRAM GPU | Fast (~ 2–3s / query) |
| **Benchmark Suite & Full Sweeps** | `heavy` (7B + Nano) | 16 GB | NVIDIA GPU with >= 6GB VRAM (CUDA) | Near real-time on GPU (~ 4s on CPU) |

To customize which model is loaded at runtime, copy `.env.example` to `.env` and set:
```bash
NSTL_EMBEDDER_MODEL=jina-embeddings-v5-text-nano
NSTL_LLM_MODEL=qwen2.5-coder-0.5b-instruct
```
