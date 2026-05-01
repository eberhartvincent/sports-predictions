# 🏆 Sports Predictor

Machine-learning daily predictions for **NHL**, **MLB**, and **NBA** — goalscorer probabilities, batter H+R+RBI projections, pitcher quality, player props, and game totals. Powered by XGBoost trained on rolling stats, pitcher matchups, park factors, weather, and lineup context.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [First-Time Setup](#first-time-setup)
4. [Running Locally (Development)](#running-locally-development)
5. [Running with Docker (Recommended)](#running-with-docker-recommended)
6. [Accessing on Your Network](#accessing-on-your-network)
7. [User Accounts & Roles](#user-accounts--roles)
8. [GitHub Actions: Automated Daily Updates](#github-actions-automated-daily-updates)
9. [Cache & Model Management](#cache--model-management)
10. [Configuration Reference](#configuration-reference)
11. [Season Updates](#season-updates)
12. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
sports_predictor/
│
├── main.py                          # App entry point — run this
├── compat.py                        # Module aliasing shim (do not edit)
│
├── app/
│   ├── auth.py                      # ⚠️  LOGIN CREDENTIALS — edit this
│   ├── tabs/
│   │   ├── nhl_tab.py               # NHL predictions UI
│   │   ├── mlb_tab.py               # MLB predictions UI (batters + pitchers)
│   │   └── nba_tab.py               # NBA predictions UI
│   ├── components/                  # Shared UI components
│   └── pages/                       # Reserved for future pages
│
├── core/
│   ├── pipelines/
│   │   ├── nhl_pipeline.py          # NHL data → features → predictions
│   │   ├── mlb_pipeline.py          # MLB data → features → predictions
│   │   ├── nba_pipeline.py          # NBA data → features → predictions
│   │   └── nhl_betting.py           # NHL Poisson betting projections
│   ├── models/
│   │   ├── sport_model.py           # Shared XGBoost wrapper (classify + regress)
│   │   └── nhl_model.py             # NHL-specific model interface (wraps sport_model)
│   └── features/
│       └── nhl_features.py          # NHL feature engineering
│
├── data/
│   ├── api/
│   │   ├── nhl_api.py               # NHL Stats API client
│   │   ├── mlb_api.py               # MLB Stats API + weather client
│   │   ├── nba_client.py            # ESPN API client (NBA)
│   │   └── nst_scraper.py           # Natural Stat Trick advanced stats
│   └── cache/
│       ├── nhl/                     # Auto-populated — do not edit
│       ├── mlb/                     # Auto-populated — do not edit
│       ├── nba/                     # Auto-populated — do not edit
│       └── model/                   # Trained model files
│
├── config/
│   └── settings.py                  # ⚙️  ALL constants — seasons, thresholds, etc.
│
├── scripts/
│   └── warm_cache.py                # Headless daily cache warmer
│
├── docker/
│   ├── start.sh                     # Container startup script
│   ├── crontab                      # Cron schedule (noon ET daily)
│   └── streamlit_config.toml        # Streamlit server settings
│
├── .github/
│   └── workflows/
│       └── daily_cache.yml          # GitHub Actions workflow (see below)
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## How It Works

Each sport runs an independent pipeline:

```
Schedule API → Rosters → Game Logs → Feature Engineering → XGBoost → Predictions
```

**NHL** — Predicts anytime goalscorer probability using rolling shot/TOI stats,
confirmed starting goalie save%, opponent quality, back-to-back flag, and NST
advanced stats (xG, Corsi).

**MLB** — Predicts H+R+RBI, HR probability, hits, RBI, runs, strikeouts, and total
bases per batter. Uses per-AB rates (not raw counts), park factors, platoon splits,
weather (Open-Meteo API), batter-vs-pitcher career history, and today's confirmed
starter's ERA/WHIP/K9. Also projects pitcher IP, K, and ER per start.

**NBA** — Predicts points, rebounds, assists, 3-pointers, steals+blocks, and
double-double probability using rolling per-game stats, opponent defensive rating,
rest days, and back-to-back detection.

**Predictions are blended:** early in the season the model output is weighted against
the player's direct season stats (regression to mean). As the season progresses the
ML model's weight increases up to 100% at 60 games.

---

## First-Time Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/sports-predictor.git
cd sports-predictor
```

### 2. Set your admin password

Open `app/auth.py`. Find the `USERS` dictionary near the top.

Generate a password hash:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD_HERE').hexdigest())"
```

Paste the output into the `password_hash` field:

```python
USERS = {
    "admin": {
        "password_hash": "paste_your_hash_here",
        "role":          "admin",
        "display_name":  "Admin",
    },
    # Add viewer accounts the same way:
    # "alice": {
    #     "password_hash": "alices_hash_here",
    #     "role":          "viewer",
    #     "display_name":  "Alice",
    # },
}
```

> ⚠️ **Never commit plaintext passwords.** The hash is safe to commit.

### 3. Verify the current season settings

Open `config/settings.py` and confirm these match the current seasons:

```python
NHL_CURRENT_SEASON = "20252026"   # update each October
MLB_SEASON         = "2026"       # update each March
NBA_SEASON         = "2025-26"    # update each October
```

---

## Running Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

Open `http://localhost:8501` in your browser. Log in with your admin credentials.

---

## Running with Docker (Recommended)

Docker runs the app persistently — it auto-restarts on crash and on machine reboot,
and runs the daily cache warm at noon ET automatically.

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows)
  or Docker Engine (Linux)
- Docker Compose (included with Docker Desktop)

### Build and start

```bash
docker-compose up -d --build
```

The first boot will automatically warm the caches (~5 minutes). After that, data
loads instantly from cache.

### Useful commands

```bash
# Check status
docker-compose ps

# View live logs
docker-compose logs -f

# Watch the cache warmer
docker exec sports-predictor tail -f /app/logs/warm_cache.log

# Force a cache warm right now
docker exec sports-predictor python scripts/warm_cache.py

# Restart
docker-compose restart

# Stop
docker-compose down

# Update code and rebuild (caches are preserved)
git pull
docker-compose up -d --build
```

### Updating the app

```bash
git pull                          # get latest code
docker-compose up -d --build      # rebuild image (caches survive)
```

---

## Accessing on Your Network

Find your machine's local IP:

```bash
# Mac
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'

# Windows (in Command Prompt)
ipconfig | findstr "IPv4"
```

Then any device on the same WiFi can open:
```
http://192.168.x.x:8501
```

> If you want to access it from outside your home network you'll need to configure
> port forwarding on your router (port 8501 → your machine's IP) and use your
> public IP or a domain name. Consider putting it behind a reverse proxy like
> Nginx or Caddy with HTTPS for security.

---

## User Accounts & Roles

There are two roles:

| Role   | Can view predictions | Can force retrain | Can warm cache | Can add users |
|--------|---------------------|-------------------|----------------|---------------|
| Admin  | ✅                   | ✅                 | ✅              | ✅ (edit auth.py) |
| Viewer | ✅                   | ❌                 | ❌              | ❌             |

### Adding a viewer

1. Generate their password hash:
   ```bash
   python3 -c "import hashlib; print(hashlib.sha256(b'theirpassword').hexdigest())"
   ```
2. Add them to `USERS` in `app/auth.py`:
   ```python
   "alice": {
       "password_hash": "their_hash",
       "role":          "viewer",
       "display_name":  "Alice",
   },
   ```
3. Rebuild the container: `docker-compose up -d --build`

### Lockout policy

After 5 failed login attempts, the account is locked for 5 minutes. This is
configurable at the top of `app/auth.py`:

```python
MAX_ATTEMPTS = 5
LOCKOUT_SECS = 300
```

---

## GitHub Actions: Automated Daily Updates

This is the recommended setup for Codespaces users. GitHub runs the cache warm
on a cloud VM every day at a scheduled time, commits the results to your repo,
and when you open Codespaces the fresh data is already there waiting.

### Step 1 — Create the workflow file

Create `.github/workflows/daily_cache.yml` in your repo:

```yaml
name: Daily Cache Warm

on:
  schedule:
    # 11:00 AM ET = 15:00 UTC (adjust for daylight saving as needed)
    - cron: "0 15 * * *"
  workflow_dispatch:          # also allows manual trigger from GitHub UI
    inputs:
      sport:
        description: "Sport to warm (all / mlb / nhl / nba)"
        required: false
        default: "all"

jobs:
  warm-cache:
    runs-on: ubuntu-latest
    timeout-minutes: 45

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Restore data caches
        uses: actions/cache@v4
        with:
          path: |
            data/cache/nhl/
            data/cache/mlb/
            data/cache/nba/
            data/cache/model/
          key: sport-caches-${{ github.run_id }}
          restore-keys: sport-caches-

      - name: Run cache warmer
        run: |
          SPORT="${{ github.event.inputs.sport || 'all' }}"
          DATE=$(TZ="America/New_York" date +%Y-%m-%d)
          echo "Warming cache: sport=$SPORT date=$DATE"
          python scripts/warm_cache.py --sport "$SPORT" --date "$DATE"

      - name: Save data caches
        uses: actions/cache@v4
        with:
          path: |
            data/cache/nhl/
            data/cache/mlb/
            data/cache/nba/
            data/cache/model/
          key: sport-caches-${{ github.run_id }}

      - name: Commit updated caches to repo
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -f data/cache/ 2>/dev/null || true
          if git diff --staged --quiet; then
            echo "No cache changes to commit"
          else
            DATE=$(TZ="America/New_York" date +%Y-%m-%d)
            git commit -m "chore: daily cache warm ${DATE} [skip ci]"
            git push
          fi
```

### Step 2 — Enable write permissions

In your GitHub repo:

```
Settings → Actions → General → Workflow permissions
→ Select "Read and write permissions"
→ Save
```

Without this the commit step will fail with a 403 error.

### Step 3 — Push and test

```bash
git add .github/workflows/daily_cache.yml
git commit -m "feat: add daily cache warm workflow"
git push
```

Then manually trigger it to verify everything works:

```
GitHub repo → Actions tab → "Daily Cache Warm" → Run workflow → Run workflow
```

Watch the logs. A successful run looks like:

```
Warming cache: sport=all date=2026-04-30
  [MLB] Found 15 games on 2026-04-30
  [MLB] Rosters: 780 active players
  ...
  [NHL] Predictions: 142 players
  [NBA] Predictions: 89 players
```

### Step 4 — Codespaces auto-pull on startup

If you use GitHub Codespaces, add a `.devcontainer/devcontainer.json` so it
automatically pulls fresh caches and starts the app when the Codespace wakes up:

```json
{
  "name": "Sports Predictor",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt",
  "postStartCommand": "bash .devcontainer/on_start.sh",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-python.vscode-pylance"]
    }
  }
}
```

Create `.devcontainer/on_start.sh`:

```bash
#!/bin/bash
echo "[startup] Pulling latest caches from repo..."
git pull --ff-only 2>/dev/null || true

echo "[startup] Starting Streamlit..."
nohup streamlit run main.py \
  --server.port 8501 \
  --server.headless true \
  >> /tmp/streamlit.log 2>&1 &

echo "[startup] App running at http://localhost:8501"
```

Make it executable:
```bash
chmod +x .devcontainer/on_start.sh
git add .devcontainer/
git commit -m "feat: devcontainer auto-startup"
git push
```

Now every time you open Codespaces it pulls the latest committed caches and
starts the app automatically — no waiting for the pipeline to run.

### How the full automated flow works

```
12:00 PM ET every day
       │
       ▼
GitHub Actions (free cloud VM)
  • Fetches schedules, rosters, game logs from all sport APIs
  • Fetches pitcher stats, weather, BvP history
  • Trains / refreshes ML models
  • Commits updated cache files to repo
       │
       ▼ (caches committed to repo)
       │
You open GitHub Codespaces (or pull on Docker machine)
  • git pull gets the pre-built caches
  • App loads instantly — no waiting
  • Click Load on any sport tab → predictions display immediately
```

---

## Cache & Model Management

### Cache locations

| Cache | Directory | TTL |
|-------|-----------|-----|
| Schedules | `data/cache/nhl/`, `mlb/`, `nba/` | 30–60 min |
| Rosters | `data/cache/mlb/` | 30 min |
| Game logs | Per-player JSON files | 60–240 min |
| Models | `data/cache/model/` | Persistent (retrained on demand) |
| Weather | `data/cache/mlb/` | 60 min |
| BvP history | `data/cache/mlb/` | 24 hours |

### Clearing caches

```bash
# Clear everything (forces full re-fetch on next load)
rm -rf data/cache/nhl/* data/cache/mlb/* data/cache/nba/* data/cache/model/*

# Clear only models (forces retrain)
rm -rf data/cache/model/*

# Clear only MLB (forces fresh MLB data)
rm -rf data/cache/mlb/*
```

In Docker:
```bash
docker exec sports-predictor rm -rf /app/data/cache/model/*
docker-compose restart
```

### Force model retrain

Log in as admin and check "Force model retrain" in the sidebar before clicking
Load, or click "Warm Cache Now" which also retrains.

---

## Configuration Reference

All constants are in `config/settings.py`. Key ones to know:

| Setting | Default | Description |
|---------|---------|-------------|
| `NHL_CURRENT_SEASON` | `"20252026"` | NHL season string for API calls |
| `MLB_SEASON` | `"2026"` | MLB season year |
| `NBA_SEASON` | `"2025-26"` | NBA season string |
| `MIN_GP` | `10` | Minimum games played to appear in NHL predictions |
| `MLB_MIN_GP` | `10` | Minimum games for MLB batters |
| `NHL_INACTIVITY_DAYS` | `30` | Days since last game before player is excluded |
| `MLB_REQUEST_DELAY` | `0.2` | Seconds between MLB API calls (avoid rate limits) |

---

## Season Updates

At the start of each new season, update `config/settings.py`:

```python
# October (NHL + NBA new seasons)
NHL_CURRENT_SEASON = "20262027"
NBA_SEASON         = "2026-27"

# March (MLB new season)
MLB_SEASON = "2027"
```

Then clear all caches and retrain:

```bash
rm -rf data/cache/nhl/* data/cache/mlb/* data/cache/nba/* data/cache/model/*
```

Or in Docker:
```bash
docker exec sports-predictor rm -rf /app/data/cache/**/*
docker-compose restart
```

---

## Troubleshooting

### App won't start

```bash
# Check what's wrong
docker-compose logs

# Verify Python can import everything
docker exec sports-predictor python -c "import compat; from core.pipelines.mlb_pipeline import MLBPipeline; print('OK')"
```

### Predictions are empty

1. Check the console output in the app — it prints every pipeline step
2. Verify the season strings in `config/settings.py` are correct
3. Check that today's games haven't already started (schedule cache may be stale)
4. Clear the relevant cache and reload: `rm -rf data/cache/mlb/*`

### GitHub Actions failing

- **403 on push**: Go to Settings → Actions → General → enable "Read and write permissions"
- **Timeout**: The full warm takes ~15 min. Increase `timeout-minutes: 45` if needed
- **Import errors**: Make sure `requirements.txt` is up to date in the repo

### Player showing with wrong stats

Their game log cache is stale. Clear it:
```bash
# Find their cache file (keyed by player ID)
ls data/cache/mlb/ | grep "mlb_log"
# Clear all MLB logs
rm data/cache/mlb/mlb_log_*
```

### Port 8501 already in use

```bash
# Find what's using it
lsof -i :8501

# Use a different port in docker-compose.yml
ports:
  - "8502:8501"
```

---

## Data Sources

All free, no API keys required:

| Sport | Source | Notes |
|-------|--------|-------|
| NHL | `api-web.nhle.com/v1` | Official NHL Stats API |
| NHL advanced | `naturalstattrick.com` | xG, Corsi, Fenwick |
| MLB | `statsapi.mlb.com` | Official MLB Stats API |
| Weather | `api.open-meteo.com` | Free, no key |
| NBA | `site.web.api.espn.com` | ESPN public API (stats.nba.com blocks servers) |

---

## License

For personal use only. Predictions are probabilistic estimates for entertainment
purposes. Not financial or gambling advice.
