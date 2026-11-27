#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "  STRATA v2 Automated Setup"
echo "=================================================="
echo ""

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or higher is required.")
PY

echo "[1/4] Installing STRATA package and dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[domains]" -r requirements.txt

echo ""
echo "[2/4] Verifying Python environment..."
python3 tools/check_env.py --check || true

echo ""
echo "[3/4] Downloading default local models (~880 MB)..."
python3 scripts/download_model.py --tier default

echo ""
echo "[4/4] Compiling & auditing STRATA SQLite v4 domain lattice..."
python3 strata_cli.py compile --clean
python3 strata_cli.py validate
python3 strata_cli.py audit

echo ""
echo "=================================================="
echo "  STRATA installation complete!"
echo "=================================================="
echo "Run interactive CLI studio : python3 strata_cli.py shell"
echo "Or execute via wrapper     : ./run_strata.sh"
echo ""
