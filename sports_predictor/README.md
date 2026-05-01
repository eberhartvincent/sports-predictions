# 🏆 Sports Predictor

ML-powered daily predictions for NHL, MLB, and NBA.

## Quick Start (Docker)

```bash
# 1. Set your admin password in app/auth.py
# 2. Build and run
docker-compose up -d --build

# Access at http://localhost:8501
# On your network: http://<your-ip>:8501
```

## Development

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Structure

```
sports_predictor/
├── main.py                    # Entry point
├── compat.py                  # Import aliasing shim
├── app/
│   ├── auth.py                # Login + role system
│   └── tabs/                  # NHL, MLB, NBA UI tabs
├── core/
│   ├── pipelines/             # NHL, MLB, NBA pipelines
│   ├── models/                # XGBoost wrappers
│   └── features/              # Feature engineering
├── data/
│   ├── api/                   # NHL, MLB, NBA API clients
│   └── cache/                 # Auto-populated at runtime
├── config/settings.py         # All constants
├── scripts/warm_cache.py      # Daily cache warmer
└── docker/                    # Container config
```

## Credentials

Edit `app/auth.py`. Generate password hash:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```

## Cache Warming

Runs automatically daily at 12:00 PM ET (via Docker cron).
Manual trigger:
```bash
docker exec sports-predictor python scripts/warm_cache.py
```
