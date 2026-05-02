# 🏆 Sports Predictor

ML-powered daily predictions for **NHL**, **MLB**, and **NBA**.

---

## How it works

```
You push code → GitHub Actions builds Docker image → pushes to ghcr.io
                                                              │
                                          Watchtower on your home machine
                                          sees new image → pulls → restarts
                                                              │
                                              App live at http://your-ip:8501
```

GitHub builds and publishes a fresh Docker image every day at 11 AM ET with
today's predictions baked in. Watchtower on your home machine picks it up
automatically within 5 minutes. You never have to touch anything.

---

## First-time setup

### Step 1 — Set your password

Open `app/auth.py`. Generate a hash for your password:

```bash
python3 -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD').hexdigest())"
```

Paste it into `USERS["admin"]["password_hash"]`. Add viewers the same way.

### Step 2 — Push to GitHub

Create a **private** repo on GitHub, then:

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/sports-predictor.git
git push -u origin main
```

### Step 3 — Enable GitHub Actions permissions

In your repo:
```
Settings → Actions → General → Workflow permissions
→ Read and write permissions ✓
→ Save
```

Also make your container package public (or Watchtower needs credentials):
```
GitHub → your profile → Packages → sports-predictor → Package settings
→ Change visibility → Public
```

Or keep it private — the setup script handles the credentials.

### Step 4 — Trigger the first build

```
GitHub → your repo → Actions → "Build, Warm & Publish" → Run workflow
```

This takes ~15 minutes. It fetches all the data, trains the models, bakes
everything into a Docker image, and pushes it to `ghcr.io/YOUR_USERNAME/sports-predictor`.

### Step 5 — Set up your home machine

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then:

```bash
# Download just the docker-compose files (or clone the repo)
bash home_machine_setup.sh
```

The script logs you into ghcr.io, pulls the image, and starts everything.
After that it's fully automatic.

---

## Accessing the app

| Device | URL |
|--------|-----|
| Same machine | `http://localhost:8501` |
| Any device on your WiFi | `http://YOUR-MACHINE-IP:8501` |

Find your machine's IP:
```bash
# Mac
ipconfig getifaddr en0

# Linux
hostname -I | awk '{print $1}'

# Windows
ipconfig | findstr "IPv4"
```

---

## User accounts & roles

Edit `app/auth.py`:

```python
USERS = {
    "admin": {
        "password_hash": "your_hash_here",   # generate with hashlib.sha256
        "role":          "admin",
        "display_name":  "Admin",
    },
    "alice": {
        "password_hash": "alice_hash_here",
        "role":          "viewer",             # viewers can see but not retrain
        "display_name":  "Alice",
    },
}
```

Generate a hash: `python3 -c "import hashlib; print(hashlib.sha256(b'password').hexdigest())"`

Commit and push — the new image will deploy automatically within minutes.

---

## Automatic update schedule

| Time | What happens |
|------|-------------|
| 11:00 AM ET | GitHub Actions starts |
| 11:05 AM | All APIs fetched (schedules, rosters, logs, weather, BvP) |
| 11:15 AM | Models trained/refreshed |
| 11:20 AM | Docker image built with fresh data baked in |
| 11:25 AM | Image pushed to ghcr.io |
| ~11:30 AM | Watchtower detects new image, pulls it, restarts container |
| 11:31 AM | App shows today's predictions |

---

## Useful commands

```bash
# Check what's running
docker-compose ps

# Live logs
docker-compose logs -f sports-predictor

# Manual update (normally happens automatically)
docker-compose pull && docker-compose up -d

# Force a fresh cache warm right now
docker exec sports-predictor python scripts/warm_cache.py

# Restart
docker-compose restart

# Stop everything
docker-compose down
```

---

## Project structure

```
sports_predictor/
├── main.py                    # Entry point: streamlit run main.py
├── compat.py                  # Module aliasing shim
├── app/
│   ├── auth.py                # ⚠️  EDIT THIS — passwords and users
│   └── tabs/                  # NHL, MLB, NBA UI tabs
├── core/
│   ├── pipelines/             # NHL, MLB, NBA prediction pipelines
│   ├── models/                # XGBoost wrappers
│   └── features/              # Feature engineering
├── data/
│   ├── api/                   # API clients (NHL, MLB, NBA, weather)
│   └── cache/                 # Auto-populated — do not edit
├── config/settings.py         # ⚙️  All constants — update seasons here
├── scripts/warm_cache.py      # Daily cache warmer
├── docker/                    # Container config files
├── .github/workflows/         # GitHub Actions — builds + publishes daily
├── Dockerfile
├── docker-compose.yml         # Run on your home machine
└── home_machine_setup.sh      # One-command home machine setup
```

---

## Season updates

Edit `config/settings.py`:

```python
NHL_CURRENT_SEASON = "20262027"   # update each October
NBA_SEASON         = "2026-27"    # update each October
MLB_SEASON         = "2027"       # update each March
```

Commit and push. The new image builds and deploys automatically.

---

## Configuration

All tuning constants are in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `NHL_CURRENT_SEASON` | `"20252026"` | NHL API season string |
| `MLB_SEASON` | `"2026"` | MLB season year |
| `NBA_SEASON` | `"2025-26"` | NBA season string |
| `MLB_MIN_GP` | `10` | Min games for batter to appear |
| `NHL_INACTIVITY_DAYS` | `30` | Days before player excluded |
| `MLB_REQUEST_DELAY` | `0.2` | Seconds between MLB API calls |

---

## Data sources (all free, no API keys)

| Sport | API |
|-------|-----|
| NHL stats | `api-web.nhle.com/v1` |
| NHL advanced | `naturalstattrick.com` |
| MLB stats | `statsapi.mlb.com` |
| Weather | `api.open-meteo.com` |
| NBA | `site.web.api.espn.com` |

---

## Troubleshooting

**GitHub Actions failing on push step**
→ Settings → Actions → General → enable "Read and write permissions"

**Watchtower not pulling new image**
→ `docker logs watchtower` — check for auth errors
→ Make the package public or re-run `home_machine_setup.sh` to refresh credentials

**App shows stale predictions**
→ `docker exec sports-predictor python scripts/warm_cache.py`

**Port 8501 in use**
→ Change `"8501:8501"` to `"8502:8501"` in `docker-compose.yml`

**Season data missing (new season)**
→ Update season strings in `config/settings.py`, commit and push

---

*For personal use. Predictions are probabilistic estimates for entertainment. Not financial advice.*
