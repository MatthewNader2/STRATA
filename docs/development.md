# STRATA v2 Development & Architecture Guide

## Architecture Overview

STRATA v2 executes end-to-end neuro-symbolic synthesis along a clean 6-stage pipeline with strict domain-kernel separation:

1. **Prompt Compilation Frontend (`src/frontends/`)**:
   - `formal_regex.py`: Deterministic parser for explicit formal STRATA syntax (`domain::CELL_ID(...)`, `loop ...`, `if ...`) producing `TypedIR`.
   - `slm_compiler.py`: Natural-language compiler via constrained JSON-Schema GGUF decoding (`llama-cpp-python`), deterministic prompt literal grounding, and structured clause decomposition (`TypedIR`).
2. **Compiled Lattice & Bounded Type Poset (`src/lattice_db.py`, `src/type_system.py`)**:
   - `tools/compile_trees.py` / `strata_cli.py compile` compiles `trees/*.json` into SQLite v4 (`trees/lattice.db`).
   - `BoundedPoset` manages multi-parent DAG carrier subtyping ($\tau$) and domain-qualified typestates ($\sigma$), while `RefinementSet` ($\varphi$) and `LayoutContract` ($\rho$) check decidable numeric intervals and symbolic/concrete array shapes (`unify_shapes`, `join_shapes`).
3. **Subgoal Retrieval & Reranking (`src/subgoal_retrieval.py`, `src/reranker.py`)**:
   - Combines structural carrier/lexical scoring with dense vector embeddings (`SentenceTransformer` on CUDA/CPU) and optional cross-encoder reranking (`LocalReranker`).
   - Enforces explicit library domain scoping and structural noise rejection across all loaded trees.
4. **Type-Directed Monoidal Planner & Bridge Graph (`src/typed_planner.py`, `src/bridge_graph.py`)**:
   - Performs backward-chaining goal search and forward linear/substructural state threading over $(\mathcal{C}, \mathcal{S}, \mu, \varphi)$.
   - Resolves multi-hop cross-domain coercions through explicit `bridge_profile` morphisms in `LiveBridgeGraph`, prioritizing Tier-1 zero-loss carriers before Tier-2 explicit conversions, while enforcing `acquire` / `borrow` / `release` resource lifetimes.
5. **Symbolic Port Binding & Language Lowering (`src/port_binder.py`, `src/language_backend.py`)**:
   - Binds quoted external assets (`input_assets`), egress targets (`egress_paths`), numeric/string literals, enums, and loop variables against `PortSchema`, emitting syntax-checked code cleanly via `LexicalLanguageBackend` / `PythonASTLanguageBackend`.
6. **GEVR Isolated Sandbox (`src/gevr_sandbox.py`, `src/security_policy.py`)**:
   - Runs generated programs in a fresh subprocess with AST security validation (`SecurityPolicy`), POSIX memory/CPU `RLIMIT_*`, disabled network sockets, declarative tree `runtime_setup` environment hooks, and an ephemeral scratch working directory seeded solely with validated `input_assets` (resolved from CWD or configured `asset_dirs`).

## Project Directory Map

```
├── strata_cli.py          # Unified CLI & interactive synthesis studio entrypoint
├── test.py                # 178-cell theory-first matrix runner across Profiles 0, A..O
├── pyproject.toml         # Package metadata, optional extras & pytest/ruff config
├── requirements.txt       # Core + neural runtime dependencies
├── .env.example           # NSTL_* environment variable reference matching src/config.py
├── src/
│   ├── config.py          # NSTLSettings & 7 orthogonal ProfileSpec definitions
│   ├── schema.py          # Canonical Pydantic v2 TreeSchema / CellSchema / PortSchema
│   ├── type_system.py     # 4-component type lattice (carrier, layout, typestate, refinement)
│   ├── typed_ir.py        # TypedIR intermediate representation
│   ├── typed_planner.py   # Resource-aware type-directed planner
│   ├── bridge_graph.py    # Cross-domain live bridge graph (Tier-1 / Tier-2)
│   ├── subgoal_retrieval.py
│   ├── port_binder.py
│   ├── language_backend.py
│   ├── lattice_db.py
│   ├── gevr_sandbox.py
│   ├── security_policy.py
│   ├── profile_adapters.py
│   ├── strata_pipeline.py
│   ├── inference.py
│   ├── reranker.py
│   ├── hardware.py
│   ├── log_config.py
│   ├── utils.py
│   ├── cli.py
│   └── frontends/
├── trees/                 # 15 canonical domain manifests (blind SDK fixtures live under benchmarks/fixtures/)
├── tools/                 # Lattice compiler, env sync check & v2 schema spec/prompt
├── scripts/               # Static neutrality check, tree linter, 5-axis auditor, downloader
├── tests/                 # Deterministic unit & Profile-0 integration pytest suite
└── benchmarks/            # v2_expanded (19 tasks) / v2_full (27 tasks) multi-arm evaluation suite, Colab & AI Studio harnesses
```

## Verification & Quality Gates

Always run these commands before pushing to `dev` or updating a PR:

```bash
# 1. Architecture neutrality, AST hygiene, zero-re / zero-frozenset checks
python3 scripts/check_static.py

# 2. Syntax and template placeholder lint across all trees (including blind SDK fixtures)
python3 scripts/lint_trees.py --include-fixtures

# 3. Strict 5-axis port contract & benchmark-leak neutrality audit
python3 scripts/audit_trees.py --strict-five-axis --strict-neutral --include-fixtures

# 4. Recompile SQLite v4 database, validate schemas, and audit live bridges
python3 strata_cli.py compile --clean
python3 strata_cli.py validate
python3 strata_cli.py audit

# 5. Deterministic unit & integration pytest suite (zero neural models required)
python3 -m pytest tests/

# 6. Pure-symbolic theory verification (runs anywhere with zero neural models)
python3 test.py --profiles 0 --early-stop

# 6. Full 7-profile 178-cell verification matrix (requires local models)
python3 test.py --early-stop
```

Look at [STRATA_FORMAL_SPEC.md](STRATA_FORMAL_SPEC.md) for formal type rules, coercion theorems, and soundness proofs.
