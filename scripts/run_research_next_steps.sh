#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="./.venv/bin/python"

echo "[1/6] Validating AVB-lite dataset..."
"$PY" scripts/validate_avb_lite.py

echo "[2/6] Running Exp001 activation patching..."
"$PY" experiments/001_activation_patching_gpt2/run.py

echo "[3/6] Running Exp002 watched vs unwatched probe..."
"$PY" experiments/002_watched_vs_unwatched_probe/run.py

echo "[4/6] Fetching real-world alignment signals..."
"$PY" -u scripts/fetch_real_world_signals.py --arxiv-max-results 40

echo "[5/6] Running Exp003 mechanistic stress audit..."
"$PY" experiments/003_mechanistic_stress_audit/run.py

echo "[6/6] Building research dashboard..."
"$PY" scripts/build_results_dashboard.py

echo "Done. Artifacts updated under experiments/results/, data/real_world/, and reports/."
