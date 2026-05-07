#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="./.venv/bin/python"

echo "[1/3] Running Exp001 activation patching..."
"$PY" experiments/001_activation_patching_gpt2/run.py

echo "[2/3] Running Exp002 watched vs unwatched probe..."
"$PY" experiments/002_watched_vs_unwatched_probe/run.py

echo "[3/3] Fetching real-world alignment signals..."
"$PY" scripts/fetch_real_world_signals.py --arxiv-max-results 80

echo "Done. Artifacts updated under experiments/results/ and data/real_world/."
