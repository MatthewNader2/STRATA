# NSTL Prototype — Full Architectural & Codebase Audit (`src/` & `trees/*.json`)

**Scope:** All **62 Python files in `src/` (59,518 LOC)** and all **18 domain tree files in `trees/*.json` (1,062 cells, ~168k lines of JSON)**.

---

## 1. Hardcodes, Predefined Lists, and "Hardcode Laundering"

### 1.1 The "Hardcode Laundering" Shell Game (`src/` ↔ `trees/*.json`)
Recent patches (`nstl-hardcode-removal.patch`, `nstl_sweep6.patch`..`nstl_sweep8.patch`) added comments claiming *"Zero hardcoded name dictionaries live in the engine"* (`src/unification.py:1205-1208`, `src/planner.py:159-166`, `src/template_wiring.py:63-72`, `src/runtime_poset.py:11`).

In reality, string-matching heuristics were **not** replaced by genuine type theory. Instead, literal `frozenset` word lists were **laundered** out of Python modules and into JSON arrays inside `trees/*.json` (under `"role_semantics"`, `"capabilities"`, `"dest_port_tokens"`, `"source_port_tokens"`, `"materialization_states"`, `"pointed_defaults"`, `"mock_fixtures"`, and `"enum_constant_rules"`), which the Python engine loads into global sets via `TypeRegistry.get_verification_semantics(key)` (`src/lattice.py:1663-1697`) and `_declared_role_semantics(key)` (**called 344 times** in `src/unification.py:1211-1233`).

In particular, `trees/python_core.json` (which only has **7 code cells**) devotes **lines 961–1536 (77 keys)** to dumping environment variables, engine tuning parameters, and domain keywords belonging to **Pandas, NumPy, Sklearn, Matplotlib, Pygame, and OpenCV/Qt**.

### 1.2 Hardcoded Predefined Lists/Sets/Dicts Still Directly in `src/`
- **Unification & Synthesis**:
  - `src/unification.py:4012-4066`: `_DEFAULT_STOPWORDS` (240+ English words plus domain/variable tokens like `"df"`, `"arr"`, `"img"`, `"row"`, `"col"`, `"val"`, `"str"`, `"int"`), `_QUESTION_MODALS`, `_CONNECTORS`, `_PROSE_VERBS`, `_ARTICLES`, `_DEMONSTRATIVES`, `_PREPOSITIONS`, `_CONJUNCTIONS`.
  - `src/unification.py:305-454`: `_CANONICAL_REWRITES`, `_TYPO_REWRITES`, `_STRUCTURAL_ALIASES`, `_IDENTIFIER_GROUP_SYNONYMS` (mapping `"regularization_strength"` → `{"c", "alpha", "l1_ratio"}`, `"kernel_type"` → `{"kernel"}`, `"random_state"` → `{"random_state", "seed"}`).
  - `src/unification.py:6492-6541`: Hardcoded aggregation word sets (`"sum"`, `"mean"`, `"average"`, `"median"`, `"min"`, `"max"`, `"std"`, `"var"`, `"count"`, `"first"`, `"last"`, `"nunique"`) and arithmetic word sets.
  - `src/synthesis.py:61-82`: `SAFE_AST_NODES`.
  - `src/synthesis.py:454-473`: `_CANONICAL_SUBGOALS` — 18 hardcoded NL descriptions matched by cosine similarity to Python AST templates.
  - `src/synthesis.py:622-645`: Hardcoded regex templates for distance metrics (`"euclidean"`, `"manhattan"`, `"cosine"`, `"chebyshev"`, `"hamming"`, `"minkowski"`).
  - `src/ir_compiler.py:33-108`: `_BASE_IR_OPCODES`, `_OPCODE_SYNONYMS`, `_CANONICAL_PORT_SYNONYMS`, `_STAGE_TO_DEFAULT_OPCODE`, `_ROLE_TO_OPCODE_KEYWORDS`.
- **Lattice, Poset & Coalgebraic Unfolder**:
  - `src/lattice.py:266-300`: `_BUILTIN_PRIMITIVE_POSETS`, `_TOP_TYPE_LITERALS`, `_TOP_STATE_LITERALS`, `_BOTTOM_LITERALS`, `_VOID_OUTPUT_LITERALS`.
  - `src/lattice.py:1401-1408`: Hardcoded 5-entry `char_map` (`"\n"`, `"\t"`, `"\r"`, `" "`, `","`).
  - `src/lattice.py:2528-2547`: Hardcoded `priority_map` (`"numeric": 0, ..., "temporal": 10`).
  - `src/coalgebraic_unfolder.py:39-152`: `_MARKERS_SEQ`, `_MARKERS_PARALLEL`, `_MARKERS_GUARD`, `_MARKERS_BIND`, `_CLAUSE_START_VERBS` (70+ imperative verbs), `_SUBGOAL_STOPWORDS`.
  - `src/node_resolver.py:24-40`: `_FORBIDDEN_SYNTHESIS_BASES` and `_TYPE_COMPAT`.
- **Planner, Sandbox & Preflight**:
  - `src/planner.py:140-155, 1178-1200`: `_CLAUSE_CONNECTORS`, `_DECOMPOSITION_NOISE`, `STOPWORDS`.
  - `src/cognitive_pathfinder.py:22-120`: `_STOPWORDS`, `_IMPERATIVE_STARTERS`, `_CONJUNCTION_SPLITS`, `_ERROR_HINTS`, `_GENERIC_CARRIERS`.
  - `src/gevr_sandbox.py:340-356, 463-468`: `_BLOCKED_MODULES`, `blocked_ops`.
- **Router, Route Methods & Tokenization**:
  - `src/router.py:140-148`: `RETRIEVAL_STOPWORDS` (75 words).
  - `src/token_evidence.py:24-40`: `PROMPT_STOPWORDS` (duplicated a third time).
  - `src/tokenizer.py:46-246`: `DOMAIN_COMPOUNDS` (~60 terms), `CONCEPT_EXPANSIONS` (~70 prompt-tuned synonym expansions), `CONCEPT_BIGRAMS` (~45 bigrams).
  - `src/route_methods/base.py:168-185`: `_ACTION_SYNONYMS`.
  - `src/route_methods/omega_astar.py:81-125`: `_CLAUSE_DOMAIN_KEYWORDS` (12 domains mapped to trigger words) and `_FORMAT_DISAMBIGUATION`.
- **Harvesters, RAG, Inference**:
  - `src/library_adapters.py:89-143`: Hardcoded fallback sets in `_get_excluded_module_tokens`, `_get_enum_citation_triggers` (OpenCV prefixes like `"fourcc"`, `"cap_prop"`), and `_get_enum_suffixes`.
  - `src/signature_introspector.py:228-229, 328-334`: OpenCV `"retval"` return stripping and `os`/`pathlib` module/class checks.
  - `src/external_rag.py:31-58`: `TRUSTED_DOC_DOMAINS`, `PACKAGE_MAPPING`.
  - `src/inference.py:46-78`: `PROFILES` registry.

### 1.3 Fragile Dynamic Derivation Replacing Hardcodes (`LatticeVocabulary`)
In `src/planner.py:159-633`, `LatticeVocabulary` computes `ingress_roles = at_ing - elsewhere` (`L221-232`) and `terminal_roles = at_terminal - elsewhere` (`L235-247`). If **even one** non-stage-1/5 cell in any tree uses `node_role="source"` or `"sink"`, set difference globally deletes `"source"` or `"sink"` across the entire engine.

---

## 2. Brittle String Matching & Regex Hacks

1. **Variable Naming Checks via `.startswith("var_")`**: `_semantic_var_name` (`src/unification.py:6862-6920`) emits carrier-based names like `dataframe_1`, `ndarray_2`, yet `src/unification.py:3563, 6970, 9038, 9112, 9187, 10397, 10416`, `src/preflight.py:305, 314, 317`, and `src/algebraic_effects.py:337` check `.startswith("var_")`.
2. **Template Placeholder Substitution via `_nstl_ph_<slot>` Regex & `ast.parse`**: Used in `src/unification.py:9069-9098` and `src/semantic_repair_engine.py:140-189`; breaks when `{slot}` is inside string literals or non-identifier expressions.
3. **3-Character Prefix Stemmer**: `src/coalgebraic_unfolder.py:158-167` (`_stem_match`) equates any words ≥ 3 chars whose lengths differ by ≤ 3 and share a prefix (`"cat"` == `"catch"`, `"min"` == `"mind"`, `"col"` == `"color"`, `"mat"` == `"matrix"`). Meanwhile `src/tokenizer.py:28-37` (`normalize_token`) claims Porter stemming in its docstring while doing `return token.strip().lower()`.
4. **3-Character Substring Validation in `IR.validate`**: `src/ir_compiler.py:149-152` checks `any((op in v) or (v in op) for v in vocab if len(v) >= 3)` — any hallucinated opcode containing `"get"`, `"set"`, `"fit"`, `"map"` passes.
5. **Naive `"="` Splitting for LHS Assignment**: `src/runtime_poset.py:264-270` (`"=" in code_template` before refactoring, or `index("=")` checking LHS chars) and `src/lattice.py:3834-3848` (`line.split("=")[0]`).
6. **Comma Splitting in Prompt Segmentation**: `src/tokenizer.py:325-327` splits on every `,` at bracket depth 0, shattering `"Select columns a, b, and c"` into 3 separate sequential clauses.

---

## 3. Dangerous & Silent Fallbacks That Hide Bugs (Enabled by Default)

| Location | Fallback Mechanism | Consequence / Hidden Bug |
|---|---|---|
| `src/planner.py:3730-3731` | `if chosen_candidate is None: chosen_candidate = scored_candidates[0][0]` | Forcibly selects a candidate that **just failed unification/preflight and repair**, making MCTS fallback & refusal (`L3732-3751`) **100% unreachable dead code**. |
| `src/synthesis.py:96-127` | `except Exception: pass` returning empty `set()` in `ASTCapabilityValidator` | If `TypeRegistry` fails or isn't loaded, `FORBIDDEN_NAMES` and `FORBIDDEN_SIDE_EFFECT_METHODS` become empty sets — **security validation turns into a no-op**. |
| `src/unification.py:2347-2455, 6929` | `strict_io: bool = False` by default | Fabricates `'input_1.csv'` / `'output_1.png'` when file paths are missing from the prompt. |
| `src/unification.py:2243-2345` | `_pointed_default` | Fabricates `0`, `""`, or `list(enum_vals)[0]` for missing required scalar/enum ports. |
| `src/unification.py:9052-9057, 9309-9314` | `except SyntaxError:` raw `.replace()` | Emits unparsed/syntactically invalid Python strings when AST template instantiation fails. |
| `src/route_methods/omega_astar.py:1683-1694` | Partial prefix `best_partial` returned on failure | Returns incomplete partial plans without running `UnificationGate` or `PreflightLinter`. |
| `src/route_methods/m3..m11` | `try: UnificationGate.unify_pipeline(...) except Exception: pass` | 7 route methods swallow `UnificationFailure` and return invalid pipelines anyway. |
| `src/route_methods/m8_llm_stepwise.py:393-403` | `chosen_next = active_transitions[0]` | Silently turns LLM routing into greedy first-transition walk on LLM error or early `FINISH`. |
| `src/route_methods/m10_monte_carlo.py:317-338` | Accepts `0 < nviol < 10` | Accepts pipelines with up to 9 `PreflightLinter` violations. |
| `src/dag_plan.py:463-471` | `else: frontier.add_ingress_node(cell)` | Converts disconnected transform cells into fake root ingress nodes. |
| `src/lattice.py:5918-5941, 3045-3048` | `advisory_qualifiers` auto-waiver | Registers almost every input port qualifier across all cells and waives them during unification. |
| `src/latent_autoencoder.py:302-312` & `src/inference.py:1024-1049` | Silent fallback to FNV-1a feature hashing | `Profile AE` `typestate_to_idx` only holds short state names (`"DataFrame[raw]"`), so 100% of `LocalRAG` description/prompt queries miss and silently use untrained FNV-1a hashing. |
| `src/node_resolver.py:203-209, 439` | `_NSTLMockCarrier.__getattr__` + `except Exception` in sandbox | Mock carrier allows any nonexistent method call, and sandbox exceptions still register the unverified LLM cell. |

---

## 4. Bugs, Logic Errors, Security Vulnerabilities & Benchmark Inflation

### 4.1 Unification, Synthesis, Template Wiring & IR Compiler
1. **`AttributeError` on `TypeTerm.base` in 4 places (`src/unification.py:3546, 9619`; `src/synthesis.py:1139, 1200`)**: `TypeTerm`, `AtomicType`, and `GenericTypeTerm` lack `.base`; caught by `except Exception:`, forcing `is_callable_type = False`, `t_base = ""`, and generic containers like `Dict[str, int]` / `Set[int]` to initialize as `[]` (`list`).
2. **`CallOptimizer` turns unbound required ports into string literals (`src/unification.py:9199-9222`)**: When `bindings.get(orig_name) is None`, sets `val_str = orig_name` and builds `ast.Constant(value=orig_name)`.
3. **`_resolve_direction_is_max` treats `"min"`, `"argmin"`, and `"ascending"` as `True` (`src/synthesis.py:848-852`)**.
4. **`FlagASTReplacer.visit_Call` injects flags into nested inner `ast.Call` nodes (`src/template_wiring.py:174-193`)**.
5. **`Substitution.compose` is broken (`src/unification.py:2768-2774`)** (drops keys from `other` and fails to apply `self` to `other`'s values).
6. **Variable counter desync in `_synthesize_pipeline_bindings_with_llm` (`src/unification.py:9399-9415` vs `L8642-8653`)**: Increments per cell instead of per output port / optic cell.
7. **Tautological condition (`src/unification.py:8214-8230`)**: `candidate_vars[0][0][1] >= 0` is always True, making fallback branch unreachable.
8. **`IRCompiler.compile` logs `IR.validate` errors at `DEBUG` and returns invalid IR (`src/ir_compiler.py:315-318`)**.

### 4.2 Lattice, Runtime Poset, Algebraic Effects, Schema & Tree Merger
1. **No lattice meet/join in `lattice.py`**; `_TOP_STATE_LITERALS` and `_BOTTOM_LITERALS` (`L279-285`) overlap (`""`, `"none"`).
2. **`RuntimeTypePoset` registers `bool`/`"logical"` as subtype of `int` and `"numeric"` (`src/runtime_poset.py:84-98, 129-132`)** because `issubclass(bool, int)` and `issubclass(bool, numbers.Number)` are True in Python.
3. **`algebraic_effects.py:155-191` reads 5 nonexistent `PortSignature` attributes** (`mutates`, `key_of`, `key_effect`, `partition_index`, `partition` missing from `PortSignature.__slots__` in `src/lattice.py:3173`), and `build_universal_verification_contract` (`L268-294`) only tracks `p_idx == 0`, missing `(X_train, y_test)` train/test leakage.
4. **`LatticeOrchestrator.load_from_database` crash (`src/lattice.py:5537-5541`)** calling `self.get_verification_semantics(...)`, plus SQLite column mismatch (`domain_name` in `src/lattice_db.py:79` vs `domain` in `src/lattice.py:5236`).
5. **`TreeSchema` & `PortSchema` (`src/schema.py:316-352, 702-726`) use `extra="ignore"` while omitting 12 top-level fields and 6 port fields**, and `merge_domain_tree` (`src/tree_merger.py:70-78`) / `UniversalHarvester.harvest_and_save` (`src/universal_harvester.py:2274-2284`) **wipe out all top-level metadata** when saving.
6. **`coalgebraic_unfolder.py:781, 931, 962`** drops child subgoals on cycle detection or unknown combinators instead of returning atomic leaves.

### 4.3 Planner, Preflight, Semantic Repair & Sandbox
1. **`planner.py:5102-5105`**: `if valid: return [best_child]` followed by `return [best_child]` ignores `verify_traced_loop_invariant`.
2. **`planner.py:3816-3844`**: Post-dry-run greedy completion appends uncloned, unlinted cells after verification.
3. **`preflight.py`**:
   - Check 1 (`L366-388`) checks `clean_lit in b` against `bound_values_str` containing `"var_1"`, `"var_2"`, so numeric literals `"1"`/`"2"` always match `"var_1"`/`"var_2"`.
   - Check 3 (`L737`) `f"({pv}" in down_str` and Check 4 (`L948`) `cons_in_v in v` cause substring collisions between `"var_1"` and `"var_10"`.
   - Check 10 (`_verify_carrier_extent_epochs`, `L260-265`) is a stub returning `[]`.
4. **`semantic_repair_engine.py`**: `_format_token` (`L291-298`) checks `if tok.isidentifier()` without `not keyword.iskeyword(tok)`, turning `True`/`False`/`None` into `{True}`/`{False}`/`{None}`! And `_try_repair_call_ast` (`L173, 186`) fails to restore `_nstl_ph_` inside compound AST arguments.
5. **`gevr_sandbox.py` & `security_policy.py`**:
   - `SecurityPolicy` (`src/security_policy.py:21-55`) omits `os`, `sys`, `shutil`, `importlib`, `_posixsubprocess`, `pickle`, and misses `__import__` in `NON_OVERRIDABLE_BLOCKED_CALLS`, allowing `__import__("subprocess")` and `import os; os.system(...)`.
   - `_sandbox_worker_exec` (`src/gevr_sandbox.py:608`) never calls `os.chdir(ephemeral_dir)`, leaking `df.to_csv()`, `cv2.imwrite()`, `np.save()`, and `sqlite3.connect()` writes to the host CWD.
   - `from os.path import join` fails with `AttributeError` (`L487-488`).
   - `_attribute_convention_check` (`L124-127`) passes unfitted sklearn estimators (which have `__dict__` params like `C=1.0`) and `None` as fitted!
   - `_ipc_safe` (`L569-574`) turns unpicklable outputs into `"<unpicklable:...>"`, which passes non-empty deliverable verification (`L951-974`).
   - Array/DataFrame equality (`L996-999`) throws `ValueError` on `bool(expected_val == actual_val)`.

### 4.4 Router, Route Methods, CLI, RAG & Benchmarks
1. **`RouteMethod.find_bridge` (`src/route_methods/base.py:736-741`)** drops `chain[1:]`; **`_find_type_producer` (`L1088-1111`)** drops `p_bridges`.
2. **`omega_astar.py`** is a weighted greedy beam search with non-admissible heuristic, not A\*.
3. **`token_evidence.py:392, 411-417`** classifies zero-overlap unrequested cells as `"connector"` instead of `"UNREQUESTED"`.
4. **`route_method_ids()` (`src/route_methods/__init__.py:88-91`)** excludes `omega_astar`.
5. **`src/errors.py:141-156`**: Uncached dynamic `type()` breaks `except errors.NoPathFoundError:`.
6. **`src/internal_rag.py:569-584`**: Deletes cached `.faiss` indexes of all other embedder models using prefix `split("__")[0]`.
7. **`src/experiments.py:98-129`**: `EarlyExitLatentProbe` runs all transformer layers despite its name.
8. **`src/cli.py:591-614, 1129-1132`**: `PipelineDebugger.__init__` omits `self.llm_feedback`, and `L1132` overwrites `final_code = extracted` with `final_code = ""`.
9. **`src/eval_runner.py:154-252` & `src/eval_metrics.py:47-48`**: `RunResponse` (`src/main.py:158-167`) omits `emitted_code`/`repaired_code`/`repaired` so repair attribution is always `"path"`, and `eval_metrics.py` counts failed-assertion runs (`passed=False, tier='type-safe'`) as passes in the attribution table!

---

## 5. Performance Bottlenecks & Code Duplication
- Linear scans over all 1,062 cells calling `unify()` inside routing loops (`src/router.py:1009, 1058, 1273-1332`, `src/route_methods/m2_endpoint_anchor.py:190-224`).
- Repeated `copy.deepcopy(scope)` inside inner search loops (`src/planner.py:3161`, `src/mcts.py:96-97`).
- Triple-nested loops in `unify_cell_with_scope` (`src/unification.py:11092-11106`), repeated clause tokenization in `resolve_literal_for_port` (`L4705-4775`), uncached `ast.parse` in `ASTCapabilityValidator` (`src/synthesis.py:129-170`).
- Monolithic God-files (`unification.py` 11.6k LOC, `lattice.py` 6.3k LOC, `planner.py` 5.3k LOC) and triplicated pipeline runners (`cli.py:616`, `cli.py:2875`, `main.py:368`).

---

## 6. Domain Trees (`trees/*.json`) & Harvesters Audit
- **1,062 cells across 18 files**, heavily skewed: `graph.json` (1 cell), `nlp.json` (2 cells), `bio.json` (3 cells), `python_core.json` (7 cells — 3 full textbook algorithm scripts + 575 lines of cross-domain `role_semantics`).
- **Positional port keys (`port_0`, `port_1`)** on `pandas.json` (90/95 cells) and `seaborn.json` (58/58 cells, with slot count mismatches on `sns_load_dataset` and `sns_color_palette`).
- **35 cells have 0 outputs (`outputs: {}`)** (15 in `pillow.json`, 8 in `seaborn.json`, 5 in `cv2.json`, 3 in `matplotlib.json`, 2 in `sklearn.json`, 2 in `requests.json`), breaking downstream wiring; 3 sinks in `numpy.json` (`NUMPY_SAVE_NPY`, `NUMPY_SAVEZ_COMPRESSED`, `NUMPY_TOFILE`) declare `output_var` not assigned in `code_template`.
- **2.15 GB RAM default**: `NUMPY_EYE` and `NUMPY_IDENTITY` in `trees/numpy.json:3326, 3537` set `n.default_value = 16384`.
- **11 of 18 trees have zero preconditions/postconditions**.
- **Harvester/Introspector bugs**: `_populate_five_axis_contract` (`src/universal_harvester.py:383`) overwrites every cell's `source_priority = 100`; `SignatureIntrospector` (`src/signature_introspector.py:1165, 1181, 1300, 1321, 1455`) replaces Tier 2/3/4 defaults with `"default"` which `_clean_default_value` deletes; `_ast_clean_type` (`L125-128`) drops tuple elements after index 0; `_emit_method_cell` (`src/universal_harvester.py:2128`) forces `dest_port` to positional index 0.

---

## 7. Recommendations Summary
1. Make strict verification the default (no silent swallowing of `UnificationFailure`, no forced selection of failed candidates, no auto-waived `advisory_qualifiers`, no fabricated file paths).
2. Fix all concrete type/AST/sandbox/router/preflight/evaluation bugs identified above.
3. Preserve top-level tree metadata in `schema.py`, `tree_merger.py`, and `universal_harvester.py`, fix broken cell schemas/defaults (`numpy.json`, `seaborn.json`, `control_flow.json`, `graph.json`), and segregate cross-domain engine config out of `python_core.json`.
4. Decompose monoliths (`unification.py`, `lattice.py`, `planner.py`) and unify the 3 execution loops into a single `PipelineExecutor`.
