# Task List 1: STRATA Peer-Review & ICML Codebase Readiness

## Wave 1 — P0 Scientific Integrity & Formal Soundness *(Completed & Verified)*

- [x] **Remove test-set leakage in SLM prompt & few-shot** ([slm_compiler.py](../src/frontends/slm_compiler.py))
  - Strip benchmark-specific closing verbs (`vent/park/quit`) and held-out SDK units (`nm/mK/mW/GHz/ns`) from `SLM_SYSTEM_PROMPT_TEMPLATE`.
  - Replace the mirrored 2nd few-shot example (`mypkg, read ... tag "A1" at 4.2, convert format, and save to`) with a neutral non-mirroring example (`demo_pkg`, plus pure-Python numeric unit stripping & quoted intent recovery).
- [x] **Tighten filesystem asset vs. colon/version string classifier** ([slm_compiler.py](../src/frontends/slm_compiler.py))
  - Extract shared `_is_path_or_uri_shape` so `"user:123"`, `"localhost:8080"`, `"v1.2"`, and `"127.0.0.1"` are not falsely classified as `external_asset` file paths, and drop ungrounded non-path `"assets"` entries instead of synthesizing false missing-file blocks.
- [x] **Remove `"State"` type-variable hardcode** ([type_system.py](../src/type_system.py), [control_flow.json](../trees/control_flow.json))
  - Restrict type variables in `parse_carrier` strictly to single uppercase letters (`len(s) == 1 and s.isalpha() and s.isupper()`), stopping collisions with `pygame::State`.
  - Migrate the 7 polymorphic `State` cells in `trees/control_flow.json` cleanly to `"S"`.
- [x] **Implement bound-tracked type-variable unification** ([type_system.py](../src/type_system.py))
  - Thread `lo_bounds` / `hi_bounds` across `_is_carrier_subtype` so `Tuple[int, str] <= Tuple[T, T]` and `Callable[[int], str] <= Callable[[T], T]` are rejected (`False`), while consistent subtype instantiation (`Tuple[bool, int] <= Tuple[T, T]`) succeeds (`True`).
- [x] **Implement symbolic-to-symbolic shape unification** ([type_system.py](../src/type_system.py))
  - Track `sym:X` bindings alongside concrete `int` bindings in `unify_shapes` so `["M", "K"]` vs `["N", "N"]` returns `False` while `["M", "M"]` vs `["N", "N"]` returns `True`.
- [x] **Harden `RefinementSet.check_literal`** ([type_system.py](../src/type_system.py))
  - Reject `None`, `bool`, and non-numeric values against active numeric/parity constraints; validate tuple/list elements recursively; enforce `odd`/`even` parity predicates.
- [x] **Implement borrow-owner lifetime tracking & condition checks** ([typed_planner.py](../src/typed_planner.py))
  - Track `borrow_owners: Dict[str, Tuple[str, ...]]` in `_plan_block` & `_wire_data_inputs` so dereferencing a borrowed view (`FluxChannel`, `Axes`, `Cursor`) after its owning handle has been released (`vent_cryo_vessel`, `plt.close`, `sqlite3.close`) is statically rejected (`dangling_borrow`).
  - Check identifiers referenced in `loop`/`branch` `condition` expressions against released handles and dangling borrows.
- [x] **Enforce runtime DAG acyclicity on pipeline construction** ([strata_pipeline.py](../src/strata_pipeline.py))
  - Call `poset.validate_acyclic()` in `StrataPipeline.__init__` and raise `ValueError` on cycles.
- [x] **Fix coercion ordering, carrier guard, format extensions & config sync** ([bridge_graph.py](../src/bridge_graph.py), [typed_planner.py](../src/typed_planner.py), [profile_adapters.py](../src/profile_adapters.py), [gevr_sandbox.py](../src/gevr_sandbox.py))
  - Add `LiveBridgePath.is_tier2` and sort by `(is_tier2, total_cost, hop_count, cell_ids)`; guard `_allows_tier2_step` carrier delta so lossy carrier conversions don't fire when `src.carrier <= dst.carrier`; remove dead `self._by_src_carrier`.
  - Add domain-neutral `_FORMAT_EXT_EQUIV` groups (`.jpg/.jpeg`, `.pq/.parquet`, `.pkl/.pickle`, `.xlsx/.xls/.excel`, etc.) in `_file_format_mismatch_penalty`.
  - Align `profile_adapters.py` (`settings.llm_max_tokens`) and `gevr_sandbox.py` defaults (`8192` MB, `30` s) with `src/config.py`.
- [x] **Align Formal Spec & fix local links** ([STRATA_FORMAL_SPEC.md](STRATA_FORMAL_SPEC.md))
  - Replace hardcoded local URIs with relative `../src/...` repository paths.
  - Align §2.1 (`[Sub-Var]` bound-tracked $\theta$, no `State`), §2.2 (symbolic-to-symbolic shapes + honest gradual `None` layout note), §2.3 & Theorem 2 (borrow-owner lifetime safety), and Theorem 4 (`(is_tier2, total_cost, hop_count)` ordering).

---

## Wave 2 — Domain Tree Integrity & Neutrality *(Completed & Verified)*

- [x] **Fix `scripts/neutralize_trees.py`** (`_canonicalize_tree`) to preserve `TreeSchema.aliases` instead of silently deleting them, and isolate or generalize legacy migration fixups (supporting both `con` and legacy `port_1`/`port_2` plus `sqlite3` `connection -> db_connection`).
- [x] **Eliminate hidden `sqlite3 -> pandas` coupling** in `trees/sqlite3.json` (`connection -> db_connection`, `path -> object` consistent with `python_core.json`).
- [x] **Repair invalid `resource_mode="release"` annotations:**
  - `trees/pygame.json`: changed `screen`/`surface` from `release` to `borrow` on `PYGAME_DISPLAY_FLIP` / `PYGAME_DISPLAY_UPDATE` / `PYGAME_IMAGE_SAVE`, and wired `PYGAME_QUIT` cleanly to release `screen` without breaking loops.
  - `trees/pandas.json` (`PD_TO_SQL`), `trees/cv2.json` (`CV2_VIDEOWRITER_WRITE`), `trees/matplotlib.json` (`matplotlib_fig_savefig`), `trees/seaborn.json` (`sns_axes_savefig`, `sns_bridge_export_figure`), `trees/sqlite3.json` (`SQLITE_INTERRUPT`, `SQLITE_BACKUP_TO_FILE`), `trees/requests.json` (`REQ_RESP_SAVE_*`), `trees/sklearn.json` (`cross_val_score`/`cross_validate`/`joblib.dump`).
- [x] **Fix broken audio, numpy, matplotlib & sklearn manifests:**
  - `trees/audio.json`: fixed 18 processing/export `sample_rate` / `orig_sr` input ports from `value_class="literal"` to `"data"`, and removed unused `sample_rate` on `AUDIO_SAVE_SPECTROGRAM_IMAGE`.
  - `trees/numpy.json`: replaced 11 unquoted-eval-failing `"np.float64"`/`"np.float32"`/`"np.int64"` string defaults with valid defaults (`"'float64'"` / `"'float32'"` / `"'int64'"`).
  - `trees/matplotlib.json` & `trees/seaborn.json`: added declarative `runtime_setup` (`env_vars: {"MPLBACKEND": "Agg"}`) and fixed `matplotlib_ax_annotate` `"dict(...)"` string default to JSON dict.
  - `trees/sklearn.json` & `trees/pathlib.json`: removed 11 sandbox-blocked `pickle.*` / `SHUTIL_*` cells; simplified `StandardScaler.fit_transform` to standard `{X}` template matching peer scalers.
- [x] **Audit `bridge_profile`, `aliases`, `value_class` & port names across all 15 trees:**
  - Added consistent `aliases` across all 13 non-core domain trees (`audio`, `cryptography`, `cv2`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `pillow`, `pygame`, `requests`, `seaborn`, `sklearn`, `sqlite3`).
  - Fixed 390 optional/default scalar config ports misclassified as `"data"`, fixed `LexicalLanguageBackend.extract_lhs_placeholders` top-level `=` scanning, renamed all 608 opaque `port_0`/`port_1` keys in `pandas.json` & `seaborn.json`, removed 10 non-coercion `bridge_profile` markers so `22/22` declared bridges load with 0 drops, and aligned `CV2_IMWRITE` output to `bool`.

---

## Wave 3 — Pytest Suite, CI Hardening, Packaging & Repository Cleanup *(Completed & Verified)*

- [x] **Create deterministic unit & integration `tests/` pytest suite (`36` tests across 8 modules + `conftest.py`):**
  - `tests/test_type_system.py`, `tests/test_bridge_graph.py`, `tests/test_typed_planner.py`, `tests/test_port_binder.py`, `tests/test_frontends.py`, `tests/test_security_and_sandbox.py`, `tests/test_profile0_e2e.py`, `tests/test_benchmarks_v2.py`.
- [x] **Harden CI (`.github/workflows/ci.yml`) & `scripts/check_static.py`:**
  - Scan `src/`, `scripts/`, `tools/`, `benchmarks/`, and `tests/` cleanly in `scripts/check_static.py`.
  - Run `--include-fixtures --strict-neutral` tree checks, lattice compile/validate/audit, `pytest tests/ -v`, zero-`frozenset` enforcement, and `python test.py --profiles 0 --early-stop` in CI.
- [x] **Fix packaging, dependencies & `.env.example`:**
  - Added `pyproject.toml` (`strata-synthesis` v2.0.0) with `[project.dependencies]` vs `[project.optional-dependencies]` (`neural`, `domains`, `dev`) and `strata = "src.cli:main"`.
  - Pruned 8 unused v1 packages from `requirements.txt`, added `huggingface_hub`, and synchronized `.env.example` with actual `src/config.py` fields/defaults.
- [x] **Fix or remove stale/broken scripts, docs & root clutter:**
  - Updated `install.sh`, `run_strata.*`, `scripts/download_model.py` (added `0.8b` tier for `unsloth/Qwen3.5-0.8B-GGUF`, cleaned unused `os`/`HfApi` & empty f-strings), and `tools/check_env.py` (`--check` flag + `strata_cli.py` hints); removed `nstl_cli.py`, `run_nstl.*`, `scripts/setup_llama_server.py`, `eval_metrics.py`, `trees/dev_unreviewed/.dev_cells.lock`, `docs/team_proposal.md`, and `docs/archive/*`.
  - Renamed & regenerated `tools/strata_node_spec.json` and updated `tools/ai_node_generation_prompt.md` from canonical v2 schema (`15` trees, `1,038` cells).
  - Moved `input.png` and `input_1.wav` + tracked `input.csv` into `benchmarks/fixtures/media/` and wired `asset_dirs` cleanly into `GEVRSandbox`, CLI, evaluators, and `test.py` (keeping repo root free of media clutter).
  - Eliminated all remaining domain/library hardcodes across `scripts/` (`lint_trees.py`, `neutralize_trees.py --check`), enforced `_check_tooling_neutrality()` and expanded `FORBIDDEN_MODEL_TOKENS` in `scripts/check_static.py`, and fixed 3 dangling `acquire` ports in `trees/seaborn.json` & `trees/sklearn.json`.
  - Aligned PolyForm Noncommercial 1.0.0 licensing across `LICENSE`, `pyproject.toml`, `README.md`, and `CITATION.cff`; added `docs/REPRODUCIBILITY.md`, multi-seed `--seed` support in `benchmarks/run_local.py`, and 95% Wilson CIs in `benchmarks/visualize_results.py`.
  - Updated `.github/CODEOWNERS`, PR/issue templates, `CONTRIBUTING.md`, `docs/development.md`, `docs/models.md`, verified zero dead functions in `src/`, and rewrote `README.md` for STRATA v2.

---

## Wave 4 — Empirical Evaluation & Benchmark Update *(Completed & Verified)*

- [x] **Run controlled local `v2_expanded` (`N = 19`) cross-model & ablation sweeps (`14` arms, `--seed 42`):**
  - Ran local `benchmarks/run_local.py --arm strata_v2` across Profiles `C` (`18/19`), `D` (`19/19`), `E [No-RR/RR]` (`19/19`), and `O [No-RR/RR]` (`19/19`), plus `LlamaTelemetry` wrapper `@functools.wraps` fix, typed slot-fill JSON Schema fix ([profile_adapters.py](../src/profile_adapters.py)), and primitive carrier `TYPE_MISMATCH` guard ([port_binder.py](../src/port_binder.py), [schema.py](../src/schema.py)) verified cleanly against all `178/178` cases in `test.py --early-stop` and `41/41` pytest tests.
  - Ran matched `qwen3.5-0.8b-instruct` (`Pure` `5/19`, `SKILL.md` `8/19`, `SKILL.md+Haystack` `9/19`, `ReAct-SKILL.md-3T` `8/19`) and `Qwen2.5-Coder-7B-Instruct-GGUF` (`Pure` `11/19`, `SKILL.md` `13/19`, `SKILL.md+Haystack` `11/19`, `ReAct-SKILL.md-3T` `13/19`) baselines.
- [x] **Run remote Google Colab T4 & API frontier sweeps (`94` clean `N = 19` cells across 8 models):**
  - Executed all `72/72` Colab T4 cells via [benchmarks/colab_eval.py](../benchmarks/colab_eval.py) across `GPT-OSS-20B` (`16` cells), `Qwen3.5-27B Dense` (`16` cells), `Qwen3.8-27B` (`16` cells), `Qwen3-Coder-30B-A3B` (`8` cells), and `Qwen3.5-35B-A3B` (`16` cells) over `T ∈ {0, 1}` × thinking/effort toggle × `{Max4096, NoLimit}` × `{Pure, SKILL.md}` (`--skip-chat-parsing` Harmony fix in `colab_eval.py` and `<|constrain|>` parser fix in `evaluators.py`).
  - Added multi-gateway free-tier support in [benchmarks/openrouter_eval.py](../benchmarks/openrouter_eval.py) and [benchmarks/gemini_api_eval.py](../benchmarks/gemini_api_eval.py): completed all `16/16` clean SandBase cells for `DeepSeek-V4.1-Flash 552B-A16B` (`19/19 = 100.0%` at `SKILL.md | T=1 | Think | Max4096` vs `11–13/19` on `Pure`/`NoThink`), `2` clean OpenRouter `Nemotron 3.5 Super VL 120B-A12B` cells (`SKILL.md` `18/19`, `Pure` `12/19`, plus `16/17` partial `NoThink` halted by quota/401), and `4` clean Gemini API `Gemma 4 31B IT` `Pure` cells (`12/19–13/19`, `SKILL.md` halted at `4/19` by upstream HTTP 503s); documented account-level blocks on TeamoRouter (`$0` balance) and OrcaRouter (`err_free_access_denied` / exhausted `orcarouter/free`).
- [x] **Evaluate external public benchmark (`BFCL v3`, `N = 120`) + ReAct baseline + 95% Wilson CIs:**
  - Implemented [benchmarks/external_bfcl.py](../benchmarks/external_bfcl.py) (40 `simple`, 40 `multiple`, 40 `irrelevance` from `gorilla-llm/Berkeley-Function-Calling-Leaderboard`), zero-oracle `build_react_feedback` in [evaluators.py](../benchmarks/evaluators.py), and unique-task Wilson CI aggregation in [visualize_results.py](../benchmarks/visualize_results.py).
  - Ran 6 local arms on BFCL v3 `N = 120`: `STRATA v2 Profile C (0.8B)` `42/120` (`40/40 = 100.0%` irrelevance refusal vs `9/40 = 22.5%` on matched `0.8B` `SKILL.md`/`ReAct-3T`), `Profile O [No-RR]` `18/120`, and `Qwen2.5-Coder-7B` `104/120` (`86.7%`), documenting honest multi-step vs. 1-step boundary tradeoffs.
- [x] **Update `docs/benchmarks/*` & `README.md` with local + frontier + BFCL v3 results:**
  - Archived historical v1 assets into [docs/benchmarks/v1_archive/](benchmarks/v1_archive/), generated matched-denominator `v2_expanded` (`docs/benchmarks/`), `frontier_ablations` (`docs/benchmarks/frontier_ablations/`), and `BFCL v3` (`docs/benchmarks/bfcl_v3/`) CSV/MD tables and collision-free figures via [benchmarks/visualize_results.py](../benchmarks/visualize_results.py), and updated [README.md](../README.md) & [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
