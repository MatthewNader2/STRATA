<div align="center">

# STRATA v2
### Symbolic Topological Routing Architecture for Typed Automata

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Schema Pydantic v2](https://img.shields.io/badge/schema-Pydantic%20v2-E92063)](src/schema.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Type-directed program synthesis via a 5-axis graded/indexed type system and resource-aware Symmetric Monoidal Category planner.*

[Formal Spec](docs/STRATA_FORMAL_SPEC.md) · [Development Guide](docs/development.md) · [Models & Profiles](docs/models.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

**STRATA v2** formulates library-grounded code synthesis as **proof search in a Resource-Aware Symmetric Monoidal Category (SMC)** paired with constrained small language models (`Qwen3.5-0.8B`). Instead of letting a generator memorize third-party APIs or improvise unsound glue code, every target library, hardware SDK, or DSL lives in a hot-swappable JSON domain manifest (`trees/*.json`) whose operations expose formal **5-axis port contracts**:

$$\mathcal{T} = \langle \tau,\ \rho,\ (\sigma, \mu),\ \varphi \rangle$$

1. **Carrier Poset ($\tau$)**: Multi-parent DAG carrier subsumption (`BoundedPoset`), higher-kinded file/container types (`File[Modality, Format]`, `List[T]`, `Tuple[int, ...]`), polymorphic type-variable bounds, and branch joins.
2. **Layout & Shape Index ($\rho$)**: Gradual rank/dtype/channel checks plus non-linear symbolic dimension unification (`unify_shapes`).
3. **Domain-Qualified Typestate ($\sigma$)**: Algebraic state machines (e.g. `cv2:color_bgr`, `pandas:clean_table`, `written_to_disk`) with `accepted_states` disjunctions.
4. **Numeric & Structural Refinements ($\varphi$)**: Decidable interval bounds (`gt(0)`, `lte(255)`), `nonzero`, `non_empty`, and parity/literal constraints (`RefinementSet`).
5. **Value Grounding & Linear Multiplicity ($\mu$)**: Explicit literal/asset/identifier binding alongside `acquire` / `borrow` / `release` resource lifetime verification so handles cannot leak or be used after release.

The core kernel (`src/`) contains **zero hardcoded domain/library names**, enforces strict AST neutrality via [scripts/check_static.py](scripts/check_static.py), compiles all manifests into SQLite v4 (`trees/lattice.db`), resolves cross-domain conversions exclusively through explicit `bridge_profile` edges, and validates emitted programs inside the POSIX-isolated `GEVRSandbox`.

---

## Architecture

```
Prompt (Formal DSL or Natural Language)
  │
  ├─► Frontend Compiler (src/frontends/)
  │     ├── formal_regex.py  (Profile 0 deterministic parser -> TypedIR)
  │     └── slm_compiler.py  (JSON-Schema GGUF decoding + literal grounding -> TypedIR)
  │
  ├─► Candidate Retrieval (src/subgoal_retrieval.py, src/reranker.py)
  │     └── Lexical / Dense Embeddings + Domain Scope Isolation (Profiles A–O)
  │
  ├─► Type-Directed Monoidal Planner (src/typed_planner.py, src/bridge_graph.py, src/type_system.py)
  │     └── Multi-hop Tier-1/Tier-2 Live Bridge Search + Resource Lifetime Tracking + Refusal Proofs
  │
  ├─► Symbolic Port Binder & Language Backend (src/port_binder.py, src/language_backend.py)
  │     └── Parameter Alignment, Input Asset / Egress Binding, Control-Flow Lowering
  │
  └─► Headless GEVR Sandbox (src/gevr_sandbox.py, src/security_policy.py)
        └── AST Security Gate, RLIMIT_AS/CPU Isolation, Network Block, Ephemeral Staging
```

---

## Domain Ecosystem

The repository compiles **15 canonical domain manifests** (`trees/*.json`) totaling **1,038 typed cells** and **22 verified live cross-domain bridges** (Tier-1 = 16, Tier-2 = 6; `0` dropped), plus **2 completely unseen synthetic SDK fixtures** used to verify zero-pretraining synthesis without engine changes:

| Category | Manifests | Purpose |
| :--- | :--- | :--- |
| **Core Language Foundation** | `trees/python_core.json` (`is_core: true`, 20 cells) | Core language primitives & syntax specification (`LanguageSpec`) |
| **Control-Flow Combinators** | `trees/control_flow.json` (33 cells) | Typed loops, branches, and monoidal combinators |
| **Swappable Standard Domains (13)** | `audio`, `cryptography`, `cv2`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `pillow`, `pygame`, `requests`, `seaborn`, `sklearn`, `sqlite3` (`trees/*.json`) | Typed operations with explicit `aliases` & declarative `runtime_setup` |
| **Blind SDK Test Fixtures (2)** | `benchmarks/fixtures/cryopulse/cryopulse.json` (11 cells), `benchmarks/fixtures/aetheris/aetheris.json` (15 cells) | Synthetic APIs absent from model pretraining weights |

---

## Quick Start

### 1. Environment & Models

```bash
# Python 3.10+
python3 -m pip install -e ".[domains]" -r requirements.txt

# Optional: download flagship 0.8B local GGUF + Jina nano embedder (~880 MB total)
python3 scripts/download_model.py --tier default
```

Copy [.env.example](.env.example) to `.env` if overriding defaults. Inspect environment health anytime with:

```bash
python3 tools/check_env.py --check
```

### 2. Compile, Validate & Audit the Domain Lattice

```bash
python3 strata_cli.py compile --clean
python3 strata_cli.py validate
python3 strata_cli.py audit
```

### 3. Run Interactive Studio or Single-Shot Synthesis

```bash
# Launch interactive STRATA studio (Profile 0 pure symbolic by default)
python3 strata_cli.py shell

# Single-shot synthesis + GEVR sandbox execution with Profile O (Qwen3.5-0.8B)
python3 strata_cli.py run --profile O --exec \
  'read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"'
```

*(Note: tracked sample media assets `input.png`, `input_1.wav`, and `input.csv` reside cleanly in `benchmarks/fixtures/media/` and are wired automatically into the CLI studio and test runners.)*

---

## Empirical Verification & Benchmark Suite

### 1. Theory-First Verification Matrix (`test.py`)

[test.py](test.py) evaluates **178 profile-expectation cells** (`73` theoretically `PASSABLE`, `105` `EXPECTED_REFUSAL` Proofs of Theory) across all **7 orthogonal profiles (`0, A, B, C, D, E, O`)** and **6 test groups**:

- `formal`: Formal-syntax pipelines, multi-hop live bridges, semantic loop/branch verification, dangling-borrow & formal refusals.
- `nl_core`: Detailed unvague multi-library NL workflows vs vague/missing-asset refusal discipline.
- `unseen`: Blind synthetic SDKs (`cryopulse`, `aetheris`) zero-pretraining & domain-leak refusal proofs.
- `haystack`: Needle-in-a-haystack isolation (target SDK cells embedded inside ~1,038 noise cells).
- `wrong_domain`: Refusal guarantees when unrelated/insufficient domains are loaded.
- `domain_spec`: Explicit library specification adherence and cross-library substitution refusal (`pillow` vs `cv2`).

Run Profile 0 without any neural models (~1s) or run the full 7-profile matrix:

```bash
python3 test.py --profiles 0 --early-stop
python3 test.py --early-stop
```

### 2. Cross-Model Comparative Evaluation Suite (`benchmarks/`)

STRATA v2 introduces an expanded comparative benchmark suite in [benchmarks/suites.py](benchmarks/suites.py):

- `v2_expanded` (**19** tasks across 5 groups: `v2_nl_core` 10, `v2_unseen` 2, `v2_haystack` 2, `v2_wrong_domain` 2, `v2_domain_spec` 3)
- `v2_full` (**27** tasks adding legacy unseen, real-world, and boundary stress cases)
- Legacy compatibility suites: `core8` (8), `core9` (11), `stress` (8), `all` (19)

These compare `STRATA v2` (local 0.8B) against pure models and doc-augmented prompts (`Pure`, `SKILL.md`, `SKILL.md+Haystack`) using:

- Local HuggingFace/GGUF evaluation: [benchmarks/run_local.py](benchmarks/run_local.py)
- Google Colab T4 frontier GGUF runner & notebook: [benchmarks/colab_eval.py](benchmarks/colab_eval.py), [benchmarks/STRATA_Colab_Benchmark.ipynb](benchmarks/STRATA_Colab_Benchmark.ipynb)
- Google AI Studio (Gemma 4 / Gemini API / manual protocol) ingestion: [benchmarks/AI_STUDIO_GEMMA4_GUIDE.md](benchmarks/AI_STUDIO_GEMMA4_GUIDE.md), [benchmarks/ingest_aistudio.py](benchmarks/ingest_aistudio.py)

> **Benchmark Status Notice:** Archived **STRATA v1** baseline charts and tables remain preserved under [docs/benchmarks/comparison_summary.md](docs/benchmarks/comparison_summary.md) for historical provenance. Complete **v2** (`v2_expanded` / `v2_full`) cross-model benchmark tables (`Pure` vs `SKILL.md` vs `SKILL.md+Haystack` vs `STRATA v2`) are currently queued pending cross-model Colab / AI Studio execution and will populate here once completed.

### 3. Unit Tests, Static & Tree Audits

```bash
python3 -m pytest tests/
python3 scripts/check_static.py
python3 scripts/lint_trees.py --include-fixtures
python3 scripts/audit_trees.py --strict-five-axis --strict-neutral --include-fixtures
```

---

## Formal Documentation

- [docs/STRATA_FORMAL_SPEC.md](docs/STRATA_FORMAL_SPEC.md) — Type rules, coercion coherence, linear resource discipline & soundness theorems.
- [docs/development.md](docs/development.md) — Pipeline architecture, repository layout & local audit gates.
- [docs/models.md](docs/models.md) — Profile definitions (`0, A–E, O`) and local GGUF / embedding tiers.
- [docs/PEER_REVIEW_TASK_LIST.md](docs/PEER_REVIEW_TASK_LIST.md) — Peer-review & ICML readiness execution checklist (Waves 1–4).
- [docs/FUTURE_ROADMAP.md](docs/FUTURE_ROADMAP.md) — Post-review roadmap (Layer 6 reactive control, tree harvester & end-to-end UI).
- [CONTRIBUTING.md](CONTRIBUTING.md) — Kernel neutrality guidelines and plugin authoring rules.

## Citation

If you build upon STRATA in research, please cite this repository and inspect [docs/STRATA_FORMAL_SPEC.md](docs/STRATA_FORMAL_SPEC.md).
