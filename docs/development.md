# STRATA (Symbolic Topological Routing Architecture for Typed Automata) - Team Development Guide

This guide outlines the development environment setup, coding conventions, testing protocols, and Git collaboration workflows for the STRATA project team.

---

## 1. Local Environment Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git
- Recommended: `venv` or `pyenv`

### Quickstart Setup
```bash
# 1. Clone repository
git clone https://github.com/MatthewNader2/STRATA.git
cd STRATA

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install core and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Provision default lightweight models (~750 MB)
python scripts/download_model.py --tier default

# 5. Compile canonical domain trees into the SQLite lattice
python nstl_cli.py compile
python nstl_cli.py validate
```

---

## 2. Git & Branching Strategy

Our team uses a **Protected `main` + `dev` + Feature Branches** workflow:

```
main (Stable Releases only)
  ▲
  │ (Release PR: Reviewed & Merged by Project Lead)
dev (Team Integration & Testing)
  ▲
  ├── feat/semantic-routing-enhancement (PR by Dev 1)
  ├── feat/pandas-tree-expansion (PR by Dev 2)
  └── fix/port-unification-edge-case (PR by Dev 3)
```

### Branch Naming Conventions
Always name branches after the **task/feature**, never after individuals:
- `feat/<feature-name>` : New functionality or engine components (e.g. `feat/mcts-expansion`)
- `fix/<bug-description>` : Bug fixes or edge-case resolutions (e.g. `fix/unification-null-leak`)
- `docs/<topic>` : Documentation updates (e.g. `docs/api-reference`)
- `tree/<domain>` : Domain tree additions or modifications (e.g. `tree/opencv-filters`)
- `test/<suite>` : Test additions or benchmarks (e.g. `test/topology-coverage`)

### Daily Workflow

1. **Start from `dev`**:
   ```bash
   git switch dev
   git pull --ff-only origin dev
   git switch -c feat/my-new-feature
   ```

2. **Develop and commit in small, logical units**:
   ```bash
   git add src/router.py tests/
   git commit -m "feat(router): improve candidate ranking for multi-clause queries"
   ```

3. **Pre-flight verification before pushing**:
   Always verify locally before opening a PR:
   ```bash
   # Lint domain trees
   python scripts/lint_trees.py

   # Check undefined names
   python scripts/check_static.py

   # Validate compiled lattice
   python nstl_cli.py compile
   python nstl_cli.py validate
   ```

4. **Keep your branch rebased or synced with `dev`**:
   ```bash
   git fetch origin
   git merge origin/dev
   ```

5. **Push and open a Pull Request**:
   ```bash
   git push -u origin feat/my-new-feature
   ```
   - Target base: **`dev`** (NOT `main`).
   - Fill in the Pull Request template.
   - Request review from at least one teammate.

6. **Merging**:
   - Feature branches are **Squash and merged** into `dev` after approval and CI passes.
   - Delete the feature branch after merge.

---

## 3. Releases (`dev` → `main`)

Only release-ready milestones are merged from `dev` into `main`:
1. Open a PR from `dev` into `main`.
2. Full test suite and benchmarks are run and verified.
3. Project Lead (@MatthewNader2) approves and merges using a **merge commit** (preserving branch history).
4. Tag the release:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0: Multi-domain topological routing and AST synthesis"
   git push origin v0.1.0
   ```
