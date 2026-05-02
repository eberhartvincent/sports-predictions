#!/bin/bash
# .devcontainer/on_start.sh
# Runs every time the Codespace starts.

set -e

echo "[startup] Pulling latest code and caches..."
git pull --ff-only 2>/dev/null || echo "[startup] Git pull skipped"

mkdir -p data/cache/nhl data/cache/mlb data/cache/nba data/cache/model logs

echo "[startup] Starting Streamlit on port 8501..."
nohup streamlit run main.py \
  --server.port 8501 \
  --server.headless true \
  --server.runOnSave false \
  >> /tmp/streamlit.log 2>&1 &

echo "[startup] Done. App at http://localhost:8501"
echo "[startup] Logs: tail -f /tmp/streamlit.log"
