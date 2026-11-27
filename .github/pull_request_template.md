## Pull Request Summary
- Describe the architectural or domain-tree change cleanly and explain **why** it preserves STRATA's type soundness and domain-neutrality invariants.

## Domain & Kernel Checklist
- [ ] Core modules (`CLEAN_CORE_MODULES` in `scripts/check_static.py`) contain zero hardcoded domain/library names, zero `re` outside `src/frontends/formal_regex.py`, and zero `frozenset`.
- [ ] `src/gevr_sandbox.py` contains zero third-party library imports or duck-typed library attributes.
- [ ] Any domain tree changes conform strictly to canonical `TreeSchema` / `CellSchema` / `PortSchema` (`extra="forbid"`) without leaked benchmark strings.
- [ ] New files/fixtures do not clutter repo root (media/evaluation fixtures belong under `benchmarks/fixtures/`).

## Verification Run Locally
```bash
python3 scripts/check_static.py
python3 scripts/lint_trees.py --include-fixtures
python3 scripts/audit_trees.py --strict-five-axis --strict-neutral --include-fixtures
python3 strata_cli.py compile --clean && python3 strata_cli.py validate && python3 strata_cli.py audit
python3 test.py --profiles 0 --early-stop
```
- Paste exact pass counts observed after the final edit.
