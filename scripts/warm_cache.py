"""
warm_cache.py — Daily cache warmer + prediction saver.

Runs via GitHub Actions every day at 11 AM ET.
Saves final predictions to data/cache/predictions/ so the app
loads instantly for all users with zero pipeline execution on page load.
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure repo root is on sys.path so all modules resolve correctly
# Works whether called as: python scripts/warm_cache.py  OR  python warm_cache.py
_SCRIPT_DIR = Path(__file__).resolve().parent      # .../scripts/
_ROOT = _SCRIPT_DIR.parent                         # .../sports-predictions/
for _p in [
    _ROOT,
    _ROOT / "config",
    _ROOT / "core" / "pipelines",
    _ROOT / "core" / "models",
    _ROOT / "core" / "features",
    _ROOT / "data" / "api",
    _ROOT / "app",
    _ROOT / "scripts",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Also run compat to register remaining aliases
try:
    import compat  # noqa
except ImportError:
    pass  # paths already set above

ET = ZoneInfo("America/New_York")
PRED_DIR = Path("data/cache/predictions")


def log(msg: str):
    ts = datetime.now(ET).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_predictions(sport: str, pipe, date: str):
    """Save pipeline predictions + metadata to disk for instant app loading."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    try:
        # Predictions DataFrame
        preds = getattr(pipe, "predictions", pd.DataFrame())
        if not preds.empty:
            preds.to_parquet(PRED_DIR / f"{sport}_predictions.parquet", index=False)
            log(f"  Saved {len(preds)} {sport.upper()} predictions")

        # Pitcher predictions (MLB only)
        pitcher_preds = getattr(pipe, "pitcher_predictions", pd.DataFrame())
        if not pitcher_preds.empty:
            pitcher_preds.to_parquet(PRED_DIR / f"{sport}_pitcher_predictions.parquet", index=False)

        # Game projections
        game_proj = getattr(pipe, "game_projections",
                    getattr(pipe, "game_proj", []))
        if game_proj:
            with open(PRED_DIR / f"{sport}_game_projections.json", "w") as f:
                json.dump(game_proj, f)

        # Games list
        games = getattr(pipe, "todays_games",
                getattr(pipe, "games", []))
        if games:
            with open(PRED_DIR / f"{sport}_games.json", "w") as f:
                json.dump(games, f)

        # Model metrics
        metrics = getattr(pipe, "model_metrics",
                  getattr(pipe, "metrics", {}))
        if metrics:
            with open(PRED_DIR / f"{sport}_metrics.json", "w") as f:
                json.dump(metrics, f)

        # Metadata
        meta = {
            "date":       date,
            "updated_at": datetime.now(ET).isoformat(),
            "n_players":  len(preds),
            "n_games":    len(game_proj),
        }
        with open(PRED_DIR / f"{sport}_meta.json", "w") as f:
            json.dump(meta, f)

        log(f"  ✅ {sport.upper()} predictions saved to {PRED_DIR}/")
    except Exception as e:
        log(f"  ⚠️  Could not save {sport} predictions: {e}")
        traceback.print_exc()


def warm_mlb(date: str, train: bool = True):
    log("=== MLB ===")
    try:
        from core.pipelines.mlb_pipeline import MLBPipeline
        pipe = MLBPipeline()
        log("  Fetching schedule …");      pipe.fetch_schedule(date)
        if not pipe.games:
            log("  No MLB games today — skipping"); return
        log(f"  {len(pipe.games)} games found")
        log("  Fetching rosters …");           pipe.fetch_rosters()
        log("  Fetching IL players …");         pipe.fetch_il_players()
        log("  Fetching game logs …");          pipe.fetch_game_logs()
        log("  Fetching pitcher stats …");      pipe.fetch_pitcher_stats()
        log("  Fetching team pitching …");      pipe.fetch_team_pitching_stats()
        log("  Fetching batter handedness …");  pipe.fetch_batter_handedness()
        log("  Fetching batter vs pitcher …");  pipe.fetch_batter_vs_pitcher()
        log("  Fetching weather …");            pipe.fetch_weather()
        if train:
            log("  Training / loading models …"); pipe.train_models(force=False)
        log("  Building predictions …");        pipe.build_predictions()
        log("  Building pitcher projections …"); pipe.build_pitcher_predictions()
        log("  Building game projections …");   pipe.build_game_projections()
        save_predictions("mlb", pipe, date)
        log(f"  ✅ MLB done — {len(pipe.predictions)} players")
    except Exception as e:
        log(f"  ❌ MLB error: {e}"); traceback.print_exc()


def warm_nhl(date: str, train: bool = True):
    log("=== NHL ===")
    try:
        from core.pipelines.nhl_pipeline import NHLPipeline
        pipe = NHLPipeline()
        log("  Fetching schedule …");           pipe.fetch_schedule(date)
        if not pipe.todays_games:
            log("  No NHL games today — skipping"); return
        log(f"  {len(pipe.todays_games)} games found")
        log("  Fetching rosters …");            pipe.fetch_rosters()
        log("  Fetching game logs …");          pipe.fetch_game_logs()
        log("  Fetching NST stats …");          pipe.fetch_nst_stats()
        log("  Fetching goalie quality …");     pipe.fetch_goalie_quality()
        log("  Fetching unavailable players …");pipe.fetch_unavailable_players()
        if train:
            log("  Training / loading model …"); pipe.train_model(force_retrain=False)
        log("  Building predictions …");        pipe.build_predictions()
        log("  Building betting projections …");pipe.build_betting_projections()
        save_predictions("nhl", pipe, date)
        log(f"  ✅ NHL done — {len(pipe.predictions)} players")
    except Exception as e:
        log(f"  ❌ NHL error: {e}"); traceback.print_exc()


def warm_nba(date: str, train: bool = True):
    log("=== NBA ===")
    try:
        from core.pipelines.nba_pipeline import NBAPipeline
        pipe = NBAPipeline()
        log("  Fetching schedule …");       pipe.fetch_schedule(date)
        if not pipe.games:
            log("  No NBA games today — skipping"); return
        log(f"  {len(pipe.games)} games found")
        log("  Fetching rosters …");        pipe.fetch_rosters()
        log("  Fetching game logs …");      pipe.fetch_game_logs()
        log("  Fetching team defense …");   pipe.fetch_team_defense()
        if train:
            log("  Training / loading models …"); pipe.train_models(force=False)
        log("  Building predictions …");    pipe.build_predictions()
        log("  Building game projections …");pipe.build_game_projections()
        save_predictions("nba", pipe, date)
        log(f"  ✅ NBA done — {len(pipe.predictions)} players")
    except Exception as e:
        log(f"  ❌ NBA error: {e}"); traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",     default=None)
    parser.add_argument("--sport",    default="all", choices=["all","mlb","nhl","nba"])
    parser.add_argument("--no-train", action="store_true")
    args   = parser.parse_args()
    date   = args.date or datetime.now(ET).strftime("%Y-%m-%d")
    train  = not args.no_train

    log(f"Warming caches for {date} | sport={args.sport} | train={train}")
    if args.sport in ("all","mlb"): warm_mlb(date, train)
    if args.sport in ("all","nhl"): warm_nhl(date, train)
    if args.sport in ("all","nba"): warm_nba(date, train)
    log("All done.")
