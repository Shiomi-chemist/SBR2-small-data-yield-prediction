#!/usr/bin/env bash
# Create the Python environment for the standalone MFF pipeline (macOS, Apple silicon or Intel).
# Requires python3 >= 3.10 (tested with python.org 3.13 on macOS 26 / arm64).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import rdkit, sklearn; print('rdkit', rdkit.__version__, '| scikit-learn', sklearn.__version__)"
echo "OK. Activate with: source .venv/bin/activate"
