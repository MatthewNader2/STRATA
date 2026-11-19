# NSTL Prototype — Detailed Remediation Task Plan

## Phase 0: Documentation
- [x] Save `CODEBASE_AUDIT.md` in repo root and `codebase_audit.md` artifact
- [x] Save `REMEDIATION_PLAN.md` in repo root and `task.md` artifact

## Phase 1: Unification, Synthesis, Template Wiring & IR Compiler
- [x] **Step 1.1** Add `.base` property on `TypeTerm` subclasses (`src/unification.py:2544-2700`) and fix `Substitution.compose` (`src/unification.py:2768-2774`)
- [x] **Step 1.2** Fix `CallOptimizer` turning unbound required ports into string literals (`src/unification.py:9195-9230`) and generalize `.startswith("var_")` bound variable detection across `src/unification.py`
- [x] **Step 1.3** Fix tautological `>= 0` condition (`src/unification.py:8214-8230`) and `_synthesize_pipeline_bindings_with_llm` output var counter desync (`src/unification.py:9399-9415`)
- [x] **Step 1.4** Fix `_resolve_direction_is_max` (`src/synthesis.py:848-852`), fail-closed baselines + dep parsing cache in `ASTCapabilityValidator` (`src/synthesis.py:85-226`)
- [x] **Step 1.5** Fix `FlagASTReplacer.visit_Call` nested call corruption (`src/template_wiring.py:174-193`), `IR.validate` 3-char substring matching and `IRCompiler.compile` error propagation (`src/ir_compiler.py:145-158, 312-320`)

## Phase 2: Errors, Runtime Poset, Lattice, Algebraic Effects, Schema, Tree Merger, Node Resolver & Unfolder
- [x] **Step 2.1** Cache dynamic deprecated exception classes in `src/errors.py:133-157`
- [x] **Step 2.2** Fix `bool`/`logical` subtyping under `int`/`numeric` and `has_lhs` keyword arg misclassification in `src/runtime_poset.py`
- [x] **Step 2.3** Add missing 5-axis effect attributes to `PortSignature` (`src/lattice.py:3173-3260`), fix SQLite `domain_name` query (`L5234-5240`), fix `self.get_verification_semantics` `AttributeError` (`L5537-5542`), and gate implicit `advisory_qualifiers` auto-waiver (`L5918-5942`)
- [x] **Step 2.4** Fix `build_universal_verification_contract` multi-partition coherence check in `src/algebraic_effects.py:268-294`
- [x] **Step 2.5** Add missing fields + `extra="allow"` in `PortSchema`/`TreeSchema` (`src/schema.py`) and preserve top-level tree metadata in `tree_merger.py` & `universal_harvester.py`
- [x] **Step 2.6** Remove permissive `_NSTLMockCarrier.__getattr__` & unverified fallback registration (`src/node_resolver.py:203-209, 435-442`); fix `_stem_match` and cycle leaf dropping (`src/coalgebraic_unfolder.py:158-167, 778-785`)

## Phase 3: Planner, DAG Plan, Preflight, Semantic Repair & Sandbox
- [x] **Step 3.1** Fix `LatticeVocabulary` set-difference erasure (`src/planner.py:221-247`), remove unconditional forced selection of failed `scored_candidates[0][0]` (`L3730-3751`), validate post-dry-run greedy completion cells (`L3816-3844`), and stop ignoring `verify_traced_loop_invariant` (`L5101-5105`)
- [x] **Step 3.2** Stop converting disconnected transform nodes into fake ingress nodes in `src/dag_plan.py:460-473`
- [x] **Step 3.3** Fix `PreflightLinter` Check 1 (`var_1` matching `"1"`), Check 3, and Check 4 (`var_1` matching `var_10`) substring collisions in `src/preflight.py`
- [x] **Step 3.4** Fix `True`/`False`/`None` corruption and `_nstl_ph_` leak in `src/semantic_repair_engine.py:172-189, 291-298`
- [x] **Step 3.5** Harden `SecurityPolicy` (`__import__`, `shutil`, `importlib`, `pickle`, `_posixsubprocess` in `src/security_policy.py`) and `GEVRSandbox` (`os.chdir(ephemeral_dir)`, `os.path` import, unfitted sklearn estimator check, `<unpicklable:...>` check, NumPy/Pandas equality in `src/gevr_sandbox.py`)

## Phase 4: Router, Route Methods, Token Evidence & Tokenizer
- [x] **Step 4.1** Preserve multi-hop bridge chains in `find_bridge` / `_find_type_producer` (`src/route_methods/base.py:736-742, 1080-1112`)
- [x] **Step 4.2** Flag non-bridge zero-overlap cells as `UNREQUESTED` in `src/token_evidence.py:390-425`
- [x] **Step 4.3** Include `omega_astar` in `route_method_ids()` (`src/route_methods/__init__.py:88-91`) and stop silently swallowing `UnificationFailure` / preflight failures by default across `omega_astar` and `m3`–`m11`

## Phase 5: Harvesters, Introspector & Domain Trees (`trees/*.json`)
- [x] **Step 5.1** Preserve computed `source_priority` in `_populate_five_axis_contract` (`src/universal_harvester.py:383`), fix sink `dest_port` positional injection (`L2128`), remove duplicate `_classify_mutator` branch (`L765-768`), preserve actual default literals in `SignatureIntrospector` (`L1165-1321`), and fix `_ast_clean_type` for `Tuple[...]` (`L125-128`)
- [x] **Step 5.2** Fix structural bugs in `trees/*.json`: `NUMPY_EYE`/`NUMPY_IDENTITY` 2.15 GB defaults & 3 unbound-output sinks in `trees/numpy.json`, `seaborn.json` slot mismatches, `control_flow.json` missing `node_type`/`state` & try-except fallback order, and `graph.json` fixture mismatch

## Phase 6: CLI, API, RAG, Autoencoder & Benchmark Evaluation
- [x] **Step 6.1** Add `llm_feedback` param to `PipelineDebugger.__init__` and fix `final_code = ""` overwrite in `src/cli.py:591-614, 1115-1133`
- [x] **Step 6.2** Add `emitted_code`/`repaired_code`/`repaired` to `RunResponse` (`src/main.py:158-168`), fix failure attribution in `src/eval_runner.py:211-245`, and stop counting failed runs as passes in `src/eval_metrics.py:47-48`
- [x] **Step 6.3** Fix FAISS multi-embedder cache deletion in `src/internal_rag.py:569-580` and add typestate token lookup before hash fallback in `src/latent_autoencoder.py:300-320`

## Phase 7: Verification
- [x] Run pytest suite and targeted regression checks verifying all fixes
- [x] Update task status and summarize results
