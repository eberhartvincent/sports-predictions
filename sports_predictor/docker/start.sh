#!/bin/bash
set -e

echo "[startup] Sports Predictor container starting..."
echo "[startup] Timezone: $(cat /etc/timezone)"
echo "[startup] Python: $(python --version)"

# Start cron daemon in background
echo "[startup] Starting cron daemon..."
service cron start

# Run an initial cache warm on first boot if caches are empty
CACHE_EMPTY=false
if [ ! "$(ls -A /app/mlb_cache 2>/dev/null)" ]; then
    CACHE_EMPTY=true
fi

if [ "$CACHE_EMPTY" = true ] || [ "$WARM_ON_START" = "true" ]; then
    echo "[startup] Cache is empty — running initial warm (this takes ~5 minutes)..."
    cd /app && python warm_cache.py >> /var/log/warm_cache.log 2>&1 &
    echo "[startup] Cache warmer running in background (PID $!)"
    echo "[startup] Tail logs: docker exec <container> tail -f /var/log/warm_cache.log"
else
    echo "[startup] Cache exists — skipping initial warm (next auto-warm at 12:00 PM ET)"
fi

# Start Streamlit — this is the main process, keeps container alive
echo "[startup] Starting Streamlit on port 8501..."
cd /app && exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=false \
    --server.fileWatcherType=none
