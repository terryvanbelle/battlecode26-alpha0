#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'numpy==2.3.5' 'matplotlib==3.10.8' 'pytest==9.2.0'
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/setup_engine.py
.venv/bin/python scripts/fetch_benchmarks.py
.venv/bin/python -m alpha0.train compile-benchmarks
