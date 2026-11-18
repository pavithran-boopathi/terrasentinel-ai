#!/usr/bin/env bash
# Start API (:8000) and web (:5173) locally without Docker. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
alembic upgrade head
uvicorn backend.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
(cd frontend && npm run dev)
