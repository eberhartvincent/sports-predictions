#!/bin/bash
# Runs every time the Codespace starts

echo "[startup] Pulling latest caches from repo..."
git pull --ff-only 2>/dev/null || echo "[startup] Git pull skipped"

mkdir -p data/cache/nhl data/cache/mlb data/cache/nba data/cache/model logs

echo "[startup] Installing/verifying dependencies..."
pip install -r requirements.txt -q

echo "[startup] Starting Streamlit on port 8501..."
streamlit run main.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.runOnSave false \
  --server.fileWatcherType none &

STREAMLIT_PID=$!
echo "[startup] Streamlit PID: $STREAMLIT_PID"

# Wait until Streamlit is actually responding before exiting
echo "[startup] Waiting for Streamlit to be ready..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "[startup] Streamlit is ready on port 8501"
    break
  fi
  sleep 1
done

echo "[startup] Done. App running at http://localhost:8501"
echo "[startup] Logs: tail -f /tmp/streamlit.log"
