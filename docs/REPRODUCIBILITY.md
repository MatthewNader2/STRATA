# STRATA v2 Artifact Evaluation & Reproducibility Guide

This document provides exact instructions for reproducing STRATA v2's static neutrality audits, formal type-theory invariants, unit/integration suite, 7-profile empirical matrix, and comparative cross-model benchmarks.

---

## 1. Hardware & Environment Profiles

| Tier | Capabilities Tested | Hardware Requirements | Neural Models Required |
| :--- | :--- | :--- | :--- |
| **Core / Profile 0** | Static gates, tree linter/auditor, SQLite v4 compilation, 41 pytest tests, Profile 0 synthesis & sandbox execution | Any Linux/macOS Python 3.10+ CPU (`>= 4 GB` RAM) | **None** |
| **Local SLM (Profiles A–O)** | 7-profile `test.py --early-stop` (`178` assertions) & `benchmarks/run_local.py --arm strata_v2` | Linux x86_64 with NVIDIA GPU (`>= 4 GB` VRAM) or modern CPU (`>= 8 GB` RAM) | Nano embedder + `0.8B` GGUF SLM (`scripts/download_model.py --tier default`) |
| **Colab T4 / AI Studio** | Frontier 20B–35B comparative baselines (`Pure`, `SKILL.md`, `SKILL.md+Haystack`) | Google Colab T4 (`16 GB` VRAM) or Google AI Studio API | Auto-downloaded in `benchmarks/colab_eval.py` |

Install base and domain verification dependencies cleanly:

```bash
python3 -m pip install -e ".[dev,domains]"
```

For local neural profiles (`A–O`), also install neural dependencies and fetch local weights:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_model.py --tier default
```

---

## 2. Deterministic Codebase, Schema & Soundness Verification (CPU-Only)

Every command below runs deterministically in seconds with **zero network access** and **zero neural weights**:

```bash
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen

# 1. Architecture neutrality, zero-re, zero-frozenset, model-token, and tooling-neutrality gates
python3 scripts/check_static.py

# 2. Strict Pydantic v2 schema, template placeholder & invariant linting (15 trees + 2 blind fixtures)
python3 scripts/lint_trees.py --include-fixtures

# 3. 5-axis port contract & foreign-leak neutrality audit
python3 scripts/audit_trees.py --strict-five-axis --strict-neutral --include-fixtures

# 4. Canonical manifest idempotency check (non-mutating dry run)
python3 scripts/neutralize_trees.py --check --include-fixtures

# 5. Recompile SQLite v4 database, validate AST/templates, and audit live bridge topology
python3 strata_cli.py compile --clean
python3 strata_cli.py validate
python3 strata_cli.py audit

# 6. 8-module deterministic pytest suite (type system, bridge graph, planner, binder, frontends, sandbox, benchmarks)
python3 -m pytest tests/ -v

# 7. Pure-symbolic theory matrix (Profile 0 passable & proof-of-theory refusals)
python3 test.py --profiles 0 --early-stop
```

---

## 3. Full 7-Profile Empirical Matrix (`0, A, B, C, D, E, O`)

With local embedder and GGUF SLM weights installed under `models/`:

```bash
python3 test.py --early-stop
```

This evaluates **178** theoretical expectations (`73` Passable + `105` Expected Refusals) across all 6 test groups (`formal`, `nl_core`, `unseen`, `haystack`, `wrong_domain`, `domain_spec`), halting on any passable failure with full diagnostic traces.

---

## 4. Controlled Cross-Model Comparative Evaluation (`v2_expanded` / `v2_full` / `BFCL v3`)

### Local Arms (Including Reranker Ablations, ReAct-3T & Multi-Seed `--seed`)

```bash
# STRATA v2 Profiles & Reranker Ablations
python3 benchmarks/run_local.py --arm strata_v2 --profile C --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm strata_v2 --profile D --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm strata_v2 --profile E --no-reranker --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm strata_v2 --profile O --no-reranker --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm strata_v2 --profile O --reranker --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded

# Matched Pure, SKILL.md, Haystack, and Zero-Oracle 3-Turn ReAct Baselines
python3 benchmarks/run_local.py --arm pure_gguf --suite v2_expanded --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm skill_md_gguf --suite v2_expanded --n-ctx 8192 --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm skill_md_haystack_gguf --suite v2_expanded --n-ctx 8192 --seed 42 --out-dir test_logs/wave4_v2_expanded
python3 benchmarks/run_local.py --arm react_skill_md_gguf --suite v2_expanded --n-ctx 8192 --max-turns 3 --seed 42 --out-dir test_logs/wave4_v2_expanded
```

### External Public Benchmark: BFCL v3 (`N = 120`: 40 Simple, 40 Multiple, 40 Irrelevance)

```bash
python3 benchmarks/external_bfcl.py --arm strata_v2 --profile C --llm qwen3.5-0.8b-instruct --per-split 40 --seed 42 --out-dir test_logs/wave4_bfcl_v3
python3 benchmarks/external_bfcl.py --arm strata_v2 --profile O --no-reranker --llm qwen3.5-0.8b-instruct --per-split 40 --seed 42 --out-dir test_logs/wave4_bfcl_v3
python3 benchmarks/external_bfcl.py --arm skill_md_gguf --llm qwen3.5-0.8b-instruct --per-split 40 --seed 42 --out-dir test_logs/wave4_bfcl_v3
python3 benchmarks/external_bfcl.py --arm react_skill_md_gguf --llm qwen3.5-0.8b-instruct --per-split 40 --max-turns 3 --seed 42 --out-dir test_logs/wave4_bfcl_v3
```

### Colab T4 & AI Studio Arms

- **Colab T4:** Open [../benchmarks/STRATA_Colab_Benchmark.ipynb](../benchmarks/STRATA_Colab_Benchmark.ipynb) or invoke `benchmarks/colab_eval.py` (with optional `--react-turns 3`).
- **AI Studio:** Follow [../benchmarks/AI_STUDIO_GEMMA4_GUIDE.md](../benchmarks/AI_STUDIO_GEMMA4_GUIDE.md) and ingest outputs with `benchmarks/ingest_aistudio.py`.

### Aggregation, Multi-Seed Tables & 95% Wilson Confidence Intervals

```bash
python3 benchmarks/visualize_results.py test_logs/wave4_v2_expanded --out-dir docs/benchmarks --title-prefix "v2_expanded (N=19) — "
python3 benchmarks/visualize_results.py test_logs/wave4_bfcl_v3 --out-dir docs/benchmarks/bfcl_v3 --title-prefix "BFCL v3 (N=120) — "
```

`benchmarks/visualize_results.py` aggregates multi-seed records `(task_id, seed)` per model, computes 95% Wilson score confidence intervals (`strict_ci_lo/hi`, `acc_ci_lo/hi`) over independent unique tasks alongside per-seed `seed_std_pct` in Markdown/CSV tables, and plots error bars on comparative figures.
