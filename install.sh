#!/usr/bin/env bash
# ============================================================
#  STRATA - Installation & Setup Script
#  Installs dependencies, provisions default local AI models,
#  and compiles the domain knowledge lattice.
# ============================================================

set -e

echo ""
echo "======================================"
echo "        STRATA Setup & Install        "
echo "======================================"
echo ""

# --- 1. Python dependencies ---
echo "[1/3] Installing core Python dependencies..."
pip install -r requirements.txt

# --- 2. Download default models ---
echo "[2/3] Downloading default models (Jina Embedder Nano + Qwen 0.5B Coder GGUF)..."
python scripts/download_model.py --tier default

# --- 3. Compile domain lattice ---
echo "[3/3] Compiling canonical domain trees into SQLite lattice..."
python nstl_cli.py compile
python nstl_cli.py validate

echo ""
echo "======================================"
echo "   Installation & Setup Complete!"
echo "======================================"
echo "   To start the REST API:"
echo "     python src/main.py"
echo "   To launch the interactive TUI shell:"
echo "     python nstl_cli.py shell"
echo "   To synthesize code directly:"
echo "     python nstl_cli.py run \"load a csv file named input.csv and compute mean\""
echo "======================================"
echo ""
