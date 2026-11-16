#!/usr/bin/env bash
# Start the Vite React frontend for local / VS Code development.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

if [[ ! -d node_modules ]]; then
  npm install
fi

echo "Starting Insulin Matrix UI on http://127.0.0.1:5173 ..."
echo "API proxy: /api -> http://127.0.0.1:8000"
exec npm run dev -- --host 127.0.0.1 --port 5173
