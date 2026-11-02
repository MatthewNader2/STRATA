# STRATA: Symbolic Topological Routing Architecture for Typed Automata

[![CI](https://github.com/MatthewNader2/STRATA/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthewNader2/STRATA/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Category Theory](https://img.shields.io/badge/foundations-category--theory-purple.svg)]()
[![Type Soundness](https://img.shields.io/badge/verification-Robinson%20Unifier-brightgreen.svg)]()

**STRATA** (**S**ymbolic **T**opological **R**outing **A**rchitecture for **T**yped **A**utomata) is a category-theoretic neuro-symbolic program synthesis engine. It translates natural language user specifications into provably type-sound, executable, multi-domain Python code pipelines by combining continuous neural semantic intent extraction with constructive category-theoretic compiler passes over a precomputed monoidal lattice of typed morphism cells.

---

## 1. System Architecture

STRATA frames program synthesis as pathfinding and diagrammatic composition over a category of typed computational morphisms:

```
[ Natural Language Specification ]
               │
               ▼
   [ Layer 1: Semantic Routing ] ──────── Dense Sentence Embeddings & Stage-Stratified Tunneling
               │
               ▼
   [ Layer 2: Lattice Planner ] ───────── Monadic Morphism Composition & Curation Priority
               │
               ▼
   [ Layer 3: Robinson Unifier ] ──────── Dual-Port Typestate Unification (τ_out ⊓ τ_in ≠ ⊥)
               │
               ▼
   [ Layer 4: AST Synthesizer ] ───────── Deterministic Python Emission, Scope & Lens Rewiring
               │
               ▼
   [ Layer 5: GEVR Sandbox ] ──────────── Multi-Process Isolated Execution & Output Verification
```

### Core Engine Layers (`src/`)

- **`src/lattice.py`**: Category-theoretic lattice orchestrator, partially ordered set (poset) type hierarchy, and high-performance SQLite storage.
- **`src/router.py`**: Stage-stratified semantic routing engine computing the active topological search tunnel $T = T_1 \cup T_2 \cup T_3$ via dense embedding similarity.
- **`src/planner.py`**: Category-theoretic planner constructing valid morphism paths with universal multi-clause coverage and bridge insertion.
- **`src/unification.py`**: Robinson first-order unification gate enforcing algebraic typestates, bracketed container structures, qualifier polarities, and in-scope variable bindings.
- **`src/synthesis.py`**: Deterministic Python Abstract Syntax Tree (AST) code emission, import resolution, and variable hygiene.
- **`src/gevr_sandbox.py`**: Process-isolated sandbox runtime enforcing CPU, memory, and timeout bounds during execution.
- **`src/main.py`**: Production FastAPI REST server with real-time endpoints for health monitoring, synthesis, and interactive visualization.
- **`strata_cli.py` / `src/cli.py`**: Unified command-line interface for harvesting, compilation, validation, auditing, and interactive synthesis (aliased as `nstl_cli.py`).

---

## 2. Repository Layout

```
STRATA/
├── .github/
│   ├── workflows/ci.yml         # Automated GitHub Actions CI workflow
│   ├── ISSUE_TEMPLATE/          # Structured Bug & Feature request forms
│   ├── CODEOWNERS               # Required code ownership review rules
│   └── pull_request_template.md # Standardized PR review checklist
├── docs/
│   ├── development.md           # Team setup, conventions, and Git workflow
│   ├── models.md                # Comprehensive model specifications & download guide
│   └── team_proposal.md         # Project academic proposal and theoretical foundations
├── scripts/
│   ├── download_model.py        # Automated Hugging Face model downloader
│   ├── lint_trees.py            # CI schema & integrity validator for domain trees
│   ├── check_static.py          # Undefined names and static syntax gate
│   └── setup_llama_server.py    # Local llama.cpp inference server setup
├── src/                         # Core STRATA engine source code
├── tests/                       # Unit, regression, and topological test suites
├── trees/                       # 18 canonical domain trees (*.json) & SQLite lattice
├── .env.example                 # Environment configuration template
├── .gitignore                   # Cleaned and categorized Git exclusion rules
├── CONTRIBUTING.md              # Contributor guide and branch conventions
├── install.sh                   # Automated one-step setup script
├── nstl_cli.py                  # Official root CLI entry point
├── pytest.ini                   # Pytest path and runner configuration
├── requirements.txt             # Core runtime dependencies
└── requirements-dev.txt         # Lightweight development, linting, and CI dependencies
```

---

## 3. Recommended & Tested Models

STRATA's neuro-symbolic design allows lightweight models to excel: the neural layer extracts semantic intent, while the category-theoretic engine guarantees 100% type soundness and program termination.

### Model Catalog

| Role | Model Identifier | Hugging Face Source | Local Destination Path | Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **Default Embedder** | `jina-embeddings-v5-text-nano` | [`jinaai/jina-embeddings-v5-text-nano`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) | `models/embeddings/jina-embeddings-v5-text-nano/` | ~350 MB |
| **Default LLM** *(Recommended)* | `qwen2.5-coder-0.5b-instruct` | [`Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) | `models/llms/qwen2.5-coder-0.5b-instruct/` | ~398 MB |
| **Medium LLM** | `qwen2.5-coder-1.5b-instruct` | [`Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) | `models/llms/qwen2.5-coder-1.5b-instruct/` | ~986 MB |
| **Heavy LLM** | `Qwen2.5-Coder-7B-Instruct-GGUF` | [`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | `models/llms/Qwen2.5-Coder-7B-Instruct-GGUF/` | ~4.68 GB |
| **Optional Reranker** | `jina-reranker-v3.5` | [`jinaai/jina-reranker-v3.5`](https://huggingface.co/jinaai/jina-reranker-v3.5) | `models/rerankers/jina-reranker-v3.5/` | ~500 MB |

> [!TIP]
> The **Default** tier (`0.5B LLM` + `Nano Embedder`) requires less than 1 GB of memory, runs in real-time (< 1.5s/query) on modern CPUs and modest laptops, and is fully sufficient for end-to-end synthesis.

For complete specifications, manual download links, and hardware recommendations, see [docs/models.md](docs/models.md).

---

## 4. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/MatthewNader2/STRATA.git
cd STRATA

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 2: Provision AI Models

Download the tested default models (`0.5B Coder GGUF` + `Jina Embedder Nano` ~750 MB total) with one command:

```bash
python scripts/download_model.py --tier default
```

*(Alternatively, run `python scripts/download_model.py --tier heavy` if you want the 7B model for GPU evaluations).*

### Step 3: Compile the Domain Knowledge Lattice

Compile the 18 canonical domain trees into the high-speed SQLite graph database:

```bash
python strata_cli.py compile
python strata_cli.py validate
```

---

## 5. CLI Usage & Capabilities

STRATA provides a unified command-line toolchain via `strata_cli.py` (with `nstl_cli.py` supported as a backward-compatible alias):

### 1. Interactive Synthesis Studio (TUI)
Launch the interactive terminal studio to synthesize, inspect DAG plans, and execute pipelines in real time:
```bash
python strata_cli.py shell
```

### 2. Direct Pipeline Synthesis
Synthesize and execute verified code directly from natural language:
```bash
python strata_cli.py run "load a csv file named input.csv, normalize X column, drop null values, and compute the mean of Y"
```

### 3. Domain Knowledge Management
- **Compile Trees**: `python strata_cli.py compile` (compiles `trees/*.json` into `trees/lattice.db`)
- **Validate Nodes**: `python strata_cli.py validate` (performs dry-run AST and schema validation over all nodes)
- **Topological Audit**: `python strata_cli.py audit` (analyzes reachability, dead ends, and cross-domain bridges)

### 4. REST API Server
Start the high-performance FastAPI synthesis server:
```bash
python src/main.py
```
API documentation will be available at `http://127.0.0.1:58102/docs`.

---

## 6. Domain Knowledge Trees (`trees/`)

STRATA includes 18 canonical domain trees compiling to **1,004 verified morphism cells** and over **34,000 dynamic topological edges**:

- **Core Analytics & Data**: `numpy.json`, `pandas.json`, `scikit-learn.json` (`sklearn.json`), `scipy` (core).
- **Visualization**: `matplotlib.json`, `seaborn.json`.
- **Media & Perception**: `cv2.json` (OpenCV), `pillow.json`, `audio.json`.
- **Specialized Domains**: `bio.json`, `cryptography.json`, `graph.json`, `nlp.json`, `pygame.json`, `requests.json`, `sqlite3.json`.
- **Control & Primitives**: `control_flow.json`, `python_core.json`, `pathlib.json`.

---

## 7. Development, Testing & CI/CD

### Pre-PR Quality Gates
Run our automated verification scripts before opening a PR:

```bash
# 1. Tree schema & integrity linter (0 violations required)
python scripts/lint_trees.py

# 2. Static syntax & undefined names check
python scripts/check_static.py

# 3. Lattice compilation & AST validation
python strata_cli.py compile
python strata_cli.py validate

# 4. Topological and symbolic unit tests
pytest tests/test_soundness_topology.py
```

### Team Git Workflow
- **`main`**: Protected, release-ready code only. Changes require a PR from `dev`, codeowner approval, and passing CI checks.
- **`dev`**: Active integration branch where team members merge feature PRs.
- **`feat/*`, `fix/*`, `docs/*`**: Isolated task branches. Always target PRs to `dev`.

For full development details, please review [docs/development.md](docs/development.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
