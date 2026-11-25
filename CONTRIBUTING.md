# Contributing to STRATA

Thank you for contributing to **STRATA** (*Symbolic Topological Routing Architecture for Typed Automata*).

## Architectural Rules

1. **Zero Domain Hardcodes in the Core Engine**:
   - The synthesis kernel (`src/type_system.py`, `src/typed_ir.py`, `src/typed_planner.py`, `src/bridge_graph.py`, `src/strata_pipeline.py`, `src/port_binder.py`, `src/subgoal_retrieval.py`, `src/language_backend.py`, `src/frontends/*`) MUST remain 100% library-agnostic.
   - All domain libraries, types, typestates, `runtime_setup` hooks, `bridge_profile` conversions, and `aliases` live strictly in JSON manifests under `trees/*.json`.
   - `src/gevr_sandbox.py` enforces process isolation cleanly without hardcoded third-party imports or duck-typed attributes.
2. **Enforced Static Invariants**:
   - Do not import `re` anywhere in `src/` except `src/frontends/formal_regex.py`.
   - Do not use `frozenset` or `allow_fallback` flags in engine code.
   - Verify via `python3 scripts/check_static.py`.
3. **Domain Manifest Neutrality & Schema Compliance**:
   - Domain manifests use canonical `TreeSchema` / `CellSchema` / `PortSchema` (`extra="forbid"`, see [tools/nstl_node_spec.json](tools/nstl_node_spec.json)).
   - Never inject benchmark-specific phrasing, test filenames, or unprincipled coercion shortcuts into domain trees.

## Adding or Updating a Domain Plugin

1. Read [tools/ai_node_generation_prompt.md](tools/ai_node_generation_prompt.md) and [tools/nstl_node_spec.json](tools/nstl_node_spec.json).
2. Add or edit `trees/<domain>.json`. Set `"is_core": false` for swappable domain libraries and declare natural-language `"aliases"` if users may refer to the library differently (e.g., `["opencv", "cv"]` for `cv2`).
3. Validate syntax, 5-axis contracts, neutrality, SQLite v4 compilation, and static gates:
   ```bash
   python3 scripts/check_static.py
   python3 scripts/lint_trees.py --include-fixtures
   python3 scripts/audit_trees.py --strict-five-axis --strict-neutral --include-fixtures
   python3 strata_cli.py compile --clean && python3 strata_cli.py validate && python3 strata_cli.py audit
   ```

## Local Testing Before Opening a PR

All changes should be verified on `dev` with:

```bash
python3 scripts/check_static.py
python3 -m pytest tests/
python3 test.py --profiles 0 --early-stop
```

When local GGUF / embedding models are installed, also run the full 7-profile matrix:

```bash
python3 test.py --early-stop
```
