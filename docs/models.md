# STRATA v2 Local Model & Profile Reference

STRATA v2 discovers local embedding, GGUF language, and cross-encoder reranking models dynamically from `models/` (configurable via `NSTL_MODELS_DIR` and `.env`) without hardcoded model paths in inference kernels.

## Inference Profiles (`0, A, B, C, D, E, O`)

| Profile | Label | Pure Symbolic | Dense Embedder | Cross-Encoder Reranker | Local GGUF SLM Compiler | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`0`** | Pure Symbolic | Yes | No | No | No | Deterministic formal frontend + lexical retrieval + full 5-axis planner |
| **`A`** | Dense Embedder Only | No | Yes | No | No | Lexical + dense embedding retrieval (no LLM IR compilation) |
| **`B`** | Dense Embedder + Neural Reranker | No | Yes | Optional | No | Dense retrieval + CrossEncoder candidate reranking (`--use-reranker`) |
| **`C`** | SLM Compiler + Dense Embedder | No | Yes | No | Yes | Constrained JSON-Schema NL -> `TypedIR` compilation + dense retrieval |
| **`D`** | SLM Compiler + Dense + Slot-Fill | No | Yes | No | Yes | Adds neural/lexical slot-fill extraction |
| **`E`** | SLM Compiler + Dense + Reranker + Slot-Fill | No | Yes | Optional | Yes | Enables cross-encoder reranking when `--use-reranker` is active |
| **`O`** | Unified Flagship (`Profile Omega`) | No | Yes | Optional | Yes | Full v2 synthesis pipeline (reranker off by default per `settings.use_reranker=False`) |

## Supported Local Models (`scripts/download_model.py`)

### Recommended Default Bundle (`--tier default` / `--tier 0.8b`)

| Component | Source Repository | Format | Target Directory (`models/...`) | Approx Size |
| :--- | :--- | :--- | :--- | :--- |
| **Embedder** | [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | Safetensors | `models/embeddings/jina-embeddings-v5-text-nano/` | ~350 MB |
| **Default SLM** | [`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | GGUF (`Q4_K_M`) | `models/llms/qwen3.5-0.8b-instruct/` | ~533 MB |

### Additional Tiers

| Tier Flag | Model | Target Directory (`models/...`) | Notes |
| :--- | :--- | :--- | :--- |
| `--tier 0.5b` | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF` (`q4_k_m`) | `models/llms/qwen2.5-coder-0.5b-instruct/` | Ultra-light legacy CPU tier |
| `--tier medium` | `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` (`q4_k_m`) | `models/llms/qwen2.5-coder-1.5b-instruct/` | 1.5B coder GGUF tier |
| `--tier heavy` | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` (`q4_k_m`) | `models/llms/Qwen2.5-Coder-7B-Instruct-GGUF/` | 7B baseline / heavy GPU tier |
| `--tier reranker` | `jinaai/jina-reranker-v3.5` | `models/rerankers/jina-reranker-v3.5/` | Optional CrossEncoder reranker (`--use-reranker`) |

Run `python3 scripts/download_model.py --tier default` to provision the flagship local models cleanly.

## Selecting Models at Runtime

Use `.env` (see [.env.example](../.env.example)) or pass flags directly to `strata_cli.py` / `test.py`:

```bash
python3 strata_cli.py run --profile O --llm qwen3.5-0.8b-instruct --exec 'load csv file "input.csv", drop missing rows, and save to "clean_out.csv"'
python3 test.py --profiles O --llm qwen3.5-0.8b-instruct --early-stop
```

For frontier GGUF comparisons on Google Colab T4 GPUs (Pure vs `SKILL.md` vs `SKILL.md+Haystack`), see [../benchmarks/colab_eval.py](../benchmarks/colab_eval.py) and [../benchmarks/STRATA_Colab_Benchmark.ipynb](../benchmarks/STRATA_Colab_Benchmark.ipynb).
