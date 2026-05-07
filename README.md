# Sports Predictor

A multi-sport daily prediction platform covering **NHL**, **MLB**, and **NBA**. Built with Python, XGBoost, and Streamlit. Predictions are generated automatically every morning via GitHub Actions and served instantly to users — no waiting, no pipeline runs on page load.

Designed with the same statistical rigour a professional quant would apply: Bayesian shooting percentage regression, Statcast xStats integration, isotonic probability calibration, physics-based probability ceilings, and date-based train/test splits to prevent data leakage.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GITHUB ACTIONS                              │
│                    Daily at 11:00 AM ET                             │
│                                                                     │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                 │
│  │ NHL        │   │ MLB        │   │ NBA        │                 │
│  │ Pipeline   │   │ Pipeline   │   │ Pipeline   │                 │
│  │            │   │ + Statcast │   │ + Pace Adj │                 │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                 │
│        │                │                 │                         │
│        └────────────────┴─────────────────┘                         │
│                         │                                           │
│              ┌───────────▼───────────┐                             │
│              │  warm_cache.py        │                             │
│              │  Saves predictions    │                             │
│              │  to parquet files     │                             │
│              └───────────┬───────────┘                             │
│                          │                                          │
│        ┌─────────────────┼─────────────────┐                       │
│        ▼                 ▼                 ▼                       │
│  nhl_predictions   mlb_predictions   nba_predictions               │
│  .parquet          .parquet          .parquet                      │
│                                                                     │
│  history/nhl_YYYY-MM-DD.parquet  ← daily snapshot for backtest     │
│                                                                     │
│        ┌─────────────────────────────────┐                         │
│        │  Commit to GitHub repo          │                         │
│        │  Send email digest              │                         │
│        └─────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          │  git pull (on Codespace start)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GITHUB CODESPACES                             │
│                    streamlit run main.py                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Streamlit App (main.py)                   │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ NHL Tab  │  │ MLB Tab  │  │ NBA Tab  │  │Backtest  │   │   │
│  │  │          │  │          │  │          │  │(admin)   │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘   │   │
│  │       │              │              │                        │   │
│  │       └──────────────┴──────────────┘                       │   │
│  │                      │                                       │   │
│  │            ┌──────────▼──────────┐                          │   │
│  │            │  prediction_store   │                          │   │
│  │            │  Reads parquet →    │                          │   │
│  │            │  instant display    │                          │   │
│  │            └─────────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Daily Sequence Diagram

```
GitHub Actions          NHL API    MLB APIs    ESPN API    Statcast    GitHub Repo    Email
      │                    │           │            │           │           │            │
      │── 11:00 AM ET ─────│           │            │           │           │            │
      │                    │           │            │           │           │            │
      │── fetch schedule ──►           │            │           │           │            │
      │◄─ games today ─────│           │            │           │           │            │
      │                    │           │            │           │           │            │
      │── fetch rosters ───►           │            │           │           │            │
      │── fetch game logs ─►           │            │           │           │            │
      │── fetch NST stats ─►           │            │           │           │            │
      │── fetch goalie quality ────────►            │           │           │            │
      │── fetch injury reports ────────►            │           │           │            │
      │                    │           │            │           │           │            │
      │────── MLB: fetch schedule ─────►            │           │           │            │
      │────── MLB: fetch rosters/logs ─►            │           │           │            │
      │────── MLB: fetch Statcast ─────────────────────────────►│           │            │
      │────── MLB: fetch umpire data ──────────────────────────►│           │            │
      │────── MLB: fetch BvP ──────────►            │           │           │            │
      │────── MLB: fetch weather ───────►            │           │           │            │
      │                    │           │            │           │           │            │
      │────────────── NBA: fetch schedule ──────────►           │           │            │
      │────────────── NBA: fetch rosters ───────────►           │           │            │
      │────────────── NBA: fetch game logs ─────────►           │           │            │
      │────────────── NBA: fetch team defense ───────►          │           │            │
      │                    │           │            │           │           │            │
      │── train/load models (XGBoost + isotonic calibration)    │           │            │
      │── build predictions (all sports)            │           │           │            │
      │── apply probability ceilings (NHL)          │           │           │            │
      │── save parquets + history snapshots         │           │           │            │
      │                    │           │            │           │           │            │
      │──────────────────────────────────────────────────── commit ────────►│            │
      │                    │           │            │           │           │            │
      │──────────────────────────────────────────────────────────────── send email ─────►
      │                    │           │            │           │           │            │

User opens Codespaces
      │                    │           │            │           │           │            │
      │── git pull ────────────────────────────────────────────────────────►│            │
      │◄─ fresh parquet files ─────────────────────────────────────────────│            │
      │── prediction_store.load() (< 100ms)         │           │           │            │
      │── display instantly ────────────────────────►           │           │            │
```

---

## File Structure

```
sports-predictions/
│
├── main.py                          # Streamlit entry point, auth, tab routing
├── compat.py                        # sys.path + module alias registration
├── requirements.txt
│
├── config/
│   └── settings.py                  # All constants: seasons, cache TTLs,
│                                    # confidence thresholds, API base URLs
│
├── core/
│   ├── models/
│   │   ├── sport_model.py           # Shared XGBoost wrapper
│   │   │                            # ├── Date-based 80/20 train/test split
│   │   │                            # ├── Isotonic probability calibration
│   │   │                            # ├── Adaptive complexity (n_estimators
│   │   │                            # │   scales with training sample size)
│   │   │                            # └── Saves/loads calibrator alongside model
│   │   └── nhl_model.py             # NHL goal classifier wrapper
│   │                                # └── Physics-based probability ceiling
│   │                                #     (shots/game × regressed sh% × save%)
│   │
│   ├── features/
│   │   └── nhl_features.py          # NHL feature engineering
│   │                                # ├── Rolling windows (3/5/10 games)
│   │                                # ├── Bayesian shooting% regression (K=150)
│   │                                # ├── Shot momentum + trend features
│   │                                # ├── NST situation splits (5v5 vs PP)
│   │                                # └── High-danger chance rates
│   │
│   └── pipelines/
│       ├── nhl_pipeline.py          # NHL end-to-end pipeline
│       │                            # ├── Two-level injury filter
│       │                            # │   (hard: 30d, soft: 7d vs team activity)
│       │                            # ├── Minimum shot threshold filter
│       │                            # └── GSax-adjusted goalie quality
│       ├── mlb_pipeline.py          # MLB end-to-end pipeline
│       │                            # ├── Statcast xStats (xBA/xSLG/xwOBA)
│       │                            # ├── Luck indicators (actual - expected)
│       │                            # ├── Umpire tendency adjustment
│       │                            # ├── Park factors + weather
│       │                            # ├── Platoon splits + BvP history
│       │                            # └── Probable pitcher quality
│       ├── nba_pipeline.py          # NBA end-to-end pipeline
│       │                            # ├── Pace-adjusted points (per 100 poss)
│       │                            # ├── Non-linear rest curve
│       │                            # │   (B2B=0.94, 1d=1.02, 2d=1.01, 3d=1.00)
│       │                            # ├── Usage% estimation
│       │                            # └── Opponent defensive rating
│       └── nhl_betting.py           # NHL game projections (Poisson model)
│                                    # Moneyline, puck line, O/U at 5.5/6.0/6.5
│
├── data/
│   └── api/
│       ├── nhl_api.py               # NHL API client (api-web.nhle.com/v1)
│       ├── nst_scraper.py           # Natural Stat Trick scraper
│       │                            # 5v5 xG, high-danger chances, Corsi%
│       ├── mlb_api.py               # MLB Stats API client
│       ├── statcast_client.py       # Baseball Savant client (NEW)
│       │                            # ├── Exit velocity, barrel rate, hard hit%
│       │                            # ├── xBA, xSLG, xwOBA, sweet spot%
│       │                            # ├── Luck indicators (actual - expected)
│       │                            # ├── Umpire tendency (K/BB/run factors)
│       │                            # └── Sprint speed
│       └── nba_client.py            # ESPN API client (stats.nba.com blocked
│                                    # on cloud — ESPN is the fallback)
│
├── app/
│   ├── auth.py                      # SHA256 hashed passwords, lockout logic
│   ├── prediction_store.py          # Parquet reader with staleness detection
│   │                                # Auto-reloads when workflow commits new files
│   ├── tabs/
│   │   ├── nhl_tab.py               # NHL UI — applies prob ceiling on load
│   │   ├── mlb_tab.py               # MLB UI
│   │   └── nba_tab.py               # NBA UI
│   └── pages/
│       └── backtest_page.py         # Admin-only backtest report
│                                    # All three sports, calibration curves,
│                                    # ECE, Brier, AUC, ROI estimates
│
├── scripts/
│   ├── warm_cache.py                # Daily pipeline runner
│   │                                # Saves predictions + history snapshots
│   ├── backtest.py                  # Backtest engine
│   │                                # Fetches actual results from APIs,
│   │                                # computes ECE/Brier/AUC/ROI by tier
│   └── send_predictions_email.py    # HTML email digest (Elite picks only)
│                                    # Top 10 per category, all sports
│
├── data/cache/
│   ├── predictions/
│   │   ├── nhl_predictions.parquet  # ← committed to repo daily
│   │   ├── mlb_predictions.parquet
│   │   ├── nba_predictions.parquet
│   │   ├── *_game_projections.json
│   │   ├── *_meta.json
│   │   └── history/
│   │       └── nhl_YYYY-MM-DD.parquet  # ← history for backtest
│   ├── nhl/                         # API response cache (GitHub Actions cache)
│   ├── mlb/                         # API response cache
│   ├── nba/                         # API response cache
│   └── model/                       # Trained XGBoost models + calibrators
│
└── .github/workflows/
    ├── daily_cache.yml              # Main workflow — 11 AM ET daily
    └── keep_alive.yml               # Weekly commit to prevent schedule deactivation
```

---

## ML Model Design

### Shared Infrastructure (`sport_model.py`)

All three sports use the same XGBoost wrapper with identical training conventions:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Train/test split | 80/20 by date | Prevents data leakage — train on past, test on recent |
| Calibration | Isotonic regression | Corrects XGBoost's tendency to push probabilities to extremes |
| Regularisation | gamma=0.2, reg_alpha scales with n | Harder splits require more evidence — reduces overfitting |
| Subsample | 0.75 | Random sampling reduces variance |
| Complexity | Adaptive | n_estimators/depth scale with training set size |
| n_jobs | 1 | Required for GitHub Codespaces compatibility |

### NHL — Goal Probability Model

**Philosophy:** Shot generation is far more repeatable than goal scoring. A player generating 4 shots/game will outscore one at 1 shot/game regardless of the context. The model is anchored to this physical reality.

**Key features:**
- `season_shooting_pct` — Bayesian-regressed toward 10.4% league average (K=150 shots required to trust individual rate)
- `rolling_Ng_shooting_pct_reg` — Rolling regressed shooting% at 3/5/10 game windows
- `shot_momentum_3g` / `shot_trend` — Recent shots vs season baseline (positive = heating up)
- `nst_ixg` / `nst_ixg_per60` — Individual expected goals from Natural Stat Trick
- `nst_ihdcf` / `nst_hdcf_pct` — High-danger chance rate (best shot quality proxy)
- `nst_ev_ixg` vs `nst_pp_ixg` — Even-strength xG separated from powerplay (5v5 is more repeatable)
- `opp_save_pct` — GSax-adjusted goalie quality (not raw save%)

**Probability ceiling (post-processing):**
```
ceiling = shots_per_game × regressed_sh% × (1 / avg_save_pct_faced)
final   = 0.90 × min(model_prob, ceiling) + 0.10 × model_prob
```
A 0-goal defender with 0.4 shots/game has a ceiling of ~0.048. A star forward with 4.0 shots/game and 15% sh% has a ceiling of ~0.52. This prevents contextual features (weak goalie, home ice) from inflating predictions for statistical non-threats.

**Confidence tiers:** Elite ≥ 0.32 · High ≥ 0.22 · Medium ≥ 0.14 · Low < 0.14

### MLB — Batter & Pitcher Projection Model

**Philosophy:** Statcast expected statistics (xBA, xSLG, xwOBA) are significantly stronger predictors than actual batting stats. A player with .230 BA but .310 xBA is underperforming their true talent and is a regression buy. Luck indicators (`actual - expected`) identify these opportunities.

**Key Statcast features (Baseball Savant — free):**
- `exit_velocity_avg` — Raw power indicator
- `barrel_pct` — Optimal launch angle + velocity (strongest HR predictor)
- `xwoba` — Expected wOBA (best single hitting metric)
- `woba_luck` — Actual wOBA minus expected (negative = unlucky = buy)
- `umpire_k_factor` / `umpire_run_factor` — Today's HP umpire tendency
- `sprint_speed` — Run-scoring proxy for fast players

**Additional features:** Park factors, weather (temperature, wind, roof), platoon splits, batter vs pitcher history weighted by sample size, regressed batting stats (K=150 AB).

**Confidence tiers:** Elite ≥ 0.80 hit prob · High ≥ 0.68 · Medium ≥ 0.55 · Low < 0.55

### NBA — Player Props Model

**Philosophy:** Pace-adjusted statistics are far more predictive than raw counts. A player scoring 22 points in a 95-possession game is performing equivalently to scoring 22.8 in a neutral-pace game. Rest matters non-linearly — 1-2 days is optimal, B2B is worst, 4+ days has a slight staleness effect.

**Key features:**
- `pace_adj_pts_3g` — Points per 100 possessions, last 3 games
- `rest_curve` — Non-linear rest factor (B2B=0.94, 1d=1.02, 2d=1.01, 3d=1.00, 4+=0.99)
- `usage_pct_3g` — Estimated usage % (how many possessions end with this player)
- `opp_def_rating` — Opponent points allowed per 100 possessions
- `rolling_Ng_pts/reb/ast/fg3m` — Rolling averages at 3/5/10 game windows

**Confidence tiers:** Elite ≥ 30 proj pts · High ≥ 22 · Medium ≥ 15 · Low < 15

---

## Data Sources

All free — no API keys required.

| Source | Data | Sport |
|--------|------|-------|
| `api-web.nhle.com/v1` | Schedule, rosters, game logs, goalies | NHL |
| `naturalstattrick.com` | Corsi%, xG, high-danger chances, situation splits | NHL |
| `statsapi.mlb.com/api/v1` | Schedule, rosters, game logs, pitchers, BvP | MLB |
| `baseballsavant.mlb.com` | xBA, xSLG, xwOBA, barrel rate, exit velo, umpires, sprint speed | MLB |
| `api.open-meteo.com` | Weather (temperature, wind, precipitation) | MLB |
| `site.api.espn.com` | Schedule, rosters, game logs, team defense | NBA |

> **Note:** `stats.nba.com` blocks cloud IPs — ESPN's public API is used as a reliable fallback and provides equivalent data.

---

## Deployment

### GitHub Actions (automatic, no action required)

The `daily_cache.yml` workflow runs every day at **11:00 AM ET**:

1. Restores API/model caches from previous run (avoids re-fetching all season data)
2. Runs `warm_cache.py` for all three sports
3. Saves prediction parquets + daily history snapshots
4. Sends email digest to configured recipients
5. Commits prediction files back to the repo
6. Saves API/model caches for next run

A `keep_alive.yml` workflow runs every Sunday to prevent GitHub from disabling the schedule on inactive repos.

### GitHub Codespaces (viewer interface)

```bash
# First time setup
git clone https://github.com/eberhartvincent/sports-predictions
cd sports-predictions

# Every subsequent start (or automatic via postStartCommand)
git pull          # gets fresh predictions committed by last workflow run
streamlit run main.py --server.port 8501
```

Predictions load in **< 100ms** from the committed parquet files — no API calls, no model training on page load.

### Rebuilding models

Force a full retrain by clearing the model cache and triggering the workflow:

```bash
rm -rf data/cache/model/
# Then: GitHub → Actions → Daily Cache Warm → Run workflow
```

### Updating seasons

Edit `config/settings.py` only — change `CURRENT_SEASON`, `MLB_SEASON`, `NBA_SEASON`. Then clear the model cache and re-run.

---

## User Roles

| Feature | Admin | Viewer |
|---------|-------|--------|
| View predictions | ✅ | ✅ |
| Date picker (historical runs) | ✅ | ❌ |
| Refresh / force pipeline | ✅ | ❌ |
| Backtest tab | ✅ | ❌ |
| Force model retrain | ✅ | ❌ |

Passwords are SHA256-hashed in `app/auth.py`. To change a password:
```bash
python3 -c "import hashlib; print(hashlib.sha256(b'yourpassword').hexdigest())"
```
Paste the output into `app/auth.py`.

---

## Backtesting

The backtest tab (admin only) compares past predictions against actual game results fetched from the respective APIs.

**Metrics computed:**

| Metric | Description | Good value |
|--------|-------------|-----------|
| **ECE** | Expected calibration error — how far off are the probabilities | < 0.05 |
| **Brier score** | Mean squared probability error | < 0.15 |
| **AUC** | Model discrimination quality | > 0.60 |
| **Elite ROI** | Simulated return at +150/+130/+110 odds | > 0% |
| **MAE (MLB)** | Mean absolute error on H/HR/RBI/R/TB projections | H < 0.35 |
| **HR direction** | % correct on HR vs no-HR binary call | > 60% |
| **DD accuracy** | Double-double probability calibration | > 65% |

History accumulates daily in `data/cache/predictions/history/`. Meaningful results require 7+ days of snapshots.

---

## Key Design Decisions

**Pre-computed predictions over live pipeline:**
Viewers see results instantly without running any ML code. The pipeline runs once per day in GitHub Actions. This makes the app responsive for all users regardless of hardware.

**Parquet over CSV:**
Binary format is ~10x smaller and loads in milliseconds. A full season of NHL predictions for 200+ players is ~200KB.

**Bayesian stat anchoring:**
Raw stats are noisy with small samples. All rate stats (shooting%, batting average equivalents) are regressed toward league averages using prior counts (K values). This prevents a player with 3 goals in 5 games from projecting at 60% shooting.

**Physics-based probability ceiling (NHL):**
The XGBoost model can be misled by contextual features (weak goalie, home ice) to inflate probabilities for statistical non-threats. A hard ceiling derived from shot volume × regressed shooting% ensures predictions are physically plausible.

**ESPN over stats.nba.com:**
The official NBA stats API blocks cloud IP ranges used by GitHub Actions and Codespaces. ESPN's equivalent public API provides the same data without IP restrictions.
