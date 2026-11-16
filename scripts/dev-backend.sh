#!/usr/bin/env bash
# Start the FastAPI backend for local / VS Code development.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

cd "${ROOT}/backend"
export PYTHONPATH="${ROOT}/backend${PYTHONPATH:+:$PYTHONPATH}"
# Repo root must stay discoverable for data/sample and artifacts paths.
export INSULIN_MATRIX_HOME="${INSULIN_MATRIX_HOME:-$ROOT}"
echo "Starting Insulin Matrix API on http://127.0.0.1:8000 ..."
exec python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
