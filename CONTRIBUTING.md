# Contributing to STRATA

Welcome to the **STRATA** (**S**ymbolic **T**opological **R**outing **A**rchitecture for **T**yped **A**utomata) project! We follow a disciplined, lightweight Git workflow to ensure reliability, mathematical correctness, and seamless collaboration.

---

## Code of Collaboration

1. **Every task starts with an Issue**:
   - Open a GitHub issue or pick one from the project board.
   - Assign the issue to yourself before creating a branch.
2. **Branches represent tasks, not people**:
   - Use `feat/*`, `fix/*`, `tree/*`, or `docs/*`.
   - Never push directly to `main` or `dev`.
3. **Pull Requests always target `dev`**:
   - Only release PRs target `main` (from `dev`).
   - All PRs must have passing automated CI checks and at least one team member approval.
4. **Clean Git History**:
   - Write clear, conventional commit messages: `feat(planner): ...`, `fix(unification): ...`, `docs: ...`.
   - Feature PRs are **Squash and merged** into `dev`.

---

## Pre-PR Verification Checklist

Before opening or requesting review on a Pull Request, ensure that:

- [ ] All 18 domain trees pass schema validation:
  ```bash
  python scripts/lint_trees.py
  ```
- [ ] No undefined variables or static syntax errors:
  ```bash
  python scripts/check_static.py
  ```
- [ ] The SQLite lattice compiles and passes AST dry-run validation:
  ```bash
  python nstl_cli.py compile
  python nstl_cli.py validate
  ```
- [ ] Relevant tests pass cleanly:
  ```bash
  pytest tests/test_soundness_topology.py
  ```
- [ ] No build artifacts, logs, or temporary files (`.diff`, `*.db`, `*.log`, `__pycache__`) are staged in git.

---

## Opening a Pull Request

1. Push your branch to `origin`:
   ```bash
   git push -u origin feat/<task-name>
   ```
2. Navigate to GitHub and click **Compare & pull request**.
3. Set base branch to **`dev`**.
4. Complete the PR template with a summary, testing evidence, and linked issue number.
5. Request review from a teammate.
