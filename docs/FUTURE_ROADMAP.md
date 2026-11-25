# Task List 2: Post-Review Roadmap — State-Dependent Control, Automation & End-to-End Product System

*(Lower priority than **Task List 1 ([PEER_REVIEW_TASK_LIST.md](PEER_REVIEW_TASK_LIST.md))** — execute after Peer-Review Waves 1–3 are clean so product features build on a sound, uncontaminated core.)*

---

## Track 1 — Layer 6: Reactive State-Dependent Execution (Robotics FSMs & Game/Simulator Wrappers)

- [ ] **Design `StepController` as an additive Layer 6 on top of static `StrataPipeline`** *(zero breaking changes to static whole-DAG `TypedIRPlanner`)*:
  - Define an environment protocol (`DynamicStateAdapter`) owning external ground-truth mutable state $s_t$, emitting active typestates/refinement facts + legal action masks $\mathcal{A}(s_t)$.
  - Implement closed-loop cycle: `observe(s_t) -> propose candidate(s) a_t -> STRATA verify legality/types/refinements/resources -> execute 1 step (or horizon H) -> update s_{t+1}`.
- [ ] **Robotics FSM / Hardware Control Pilot:**
  - Author sample robotics/hardware FSM domain trees (`ArmHomed`, `InterlockArmed`, `GripperOpen/Closed`, joint-limit `RefinementSet` invariants, `CalibrationFresh` handles).
  - Verify illegal hardware/safety sequences fail statically at verification time before reaching actuators.
- [ ] **Chess / Combinatorial Environment Tool Wrapper:**
  - Connect a deterministic simulator (`python-chess` or equivalent) implementing `DynamicStateAdapter` where STRATA acts as the legality/action verifier & ranker without enumerating $10^{44}$ board states inside `BoundedPoset`.
  - Benchmark SLM move legality rate (illegal-move rate, phantom/dead-piece rate) with STRATA Layer 6 vs. raw LLM stateful play across multi-turn games.

---

## Track 2 — Automated Tree Harvester & True Cross-Language Support

- [ ] **Automated Library-to-Tree Harvester (`tools/harvest_tree.py`):**
  - Extract deterministic signatures, type hints, return types, defaults, and docstrings from `.pyi` stubs, `inspect.signature`, and ASTs first (before invoking an LLM to synthesize descriptions, `typestates`, and `resource_mode`).
  - Enforce automatic validation gate: every harvested manifest must pass `scripts/lint_trees.py` and `scripts/audit_trees.py --strict-five-axis --strict-neutral` before acceptance.
  - Detect common external alias conventions automatically (`numpy -> np`, `pandas -> pd`, `matplotlib -> plt`, etc.) and emit `aliases`.
- [ ] **Non-Python Multi-Language Support (`Rust`, `TypeScript`):**
  - Support merging multiple `is_core=True` core language trees cleanly instead of taking the alphabetically first `python_core.json`.
  - Replace Python-only carrier/keyword/import/`str.format` assumptions in `port_binder.py`, `type_system.py` (angle-bracket generics `Vec<T>`, `A | B` unions), `schema.py`, and `language_backend.py` (`if (...)`, `'a` lifetimes, `/* */` comments).
  - Configure optional external compiler/container sandboxes (`cargo check`, `tsc --noEmit` / `node`) inside `GEVRSandbox` for non-Python runtimes.

---

## Track 3 — End-to-End Interactive Product System (`--mode product` vs. `--mode strict-theory`)

- [ ] **Strict Mode Separation (`--mode strict-theory` vs. `--mode product`):**
  - Keep strict refusal behavior, zero fallback, and zero asset fabrication **100% locked** under `--mode strict-theory` (default for CLI/benchmarks/CI).
  - Gate any clarifying Q&A, smart defaults, and fallbacks explicitly behind `--mode product --allow-fallback`.
- [ ] **Continuous Multi-Turn Stateful Sessions & History:**
  - Persist session state across user turns (`SessionContext`: cumulative code history, live `env_map`, open linear `acquire` handles, released tombstones, active `domain_set`, and turn rollback).
  - Support incremental follow-up prompts (`"now plot column Z"`, `"actually save to PNG instead"`, `"close the database connection"`).
- [ ] **Interactive GUI & Lattice / Active-Path Visualizer:**
  - Build a web UI (e.g., React/Vite frontend + FastAPI backend wired cleanly to `StrataPipeline`) showing:
    - Domain tree explorer & live hot-swap loader.
    - Decomposed `TypedIR` subgoal DAG and per-node retrieval candidates (`top_k` + rerank scores).
    - Full composed active path, auto-inserted `LiveBridgeGraph` morphisms, linear resource lifetimes ($\Delta \to \mathcal{R}$), rejected paths/refusals, and sandbox outputs.
- [ ] **Hybrid Internet/Docs RAG + Domain Tree RAG:**
  - Combine dense/hybrid retrieval over compiled STRATA trees with optional live official-documentation / web search retrieval when discovering new libraries or synthesizing on-demand candidate cells (strictly disabled in offline benchmarks).
