"""
warm_cache.py — Pre-warm all sport caches without starting Streamlit.

Run this daily (via cron or GitHub Actions) so the app loads instantly:
    python warm_cache.py

Options:
    python warm_cache.py --date 2026-04-20   # specific date
    python warm_cache.py --sport mlb         # single sport only
    python warm_cache.py --no-train          # skip model training (cache-only)
"""

import argparse
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def log(msg: str):
    ts = datetime.now(ET).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def warm_mlb(date: str, train: bool = True):
    log("=== MLB ===")
    try:
        from mlb_pipeline import MLBPipeline
        pipe = MLBPipeline()

        log("  Fetching schedule …")
        pipe.fetch_schedule(date)

        if not pipe.games:
            log("  No MLB games today — skipping")
            return

        log(f"  {len(pipe.games)} games found")
        log("  Fetching rosters …")
        pipe.fetch_rosters()

        log("  Fetching IL players …")
        pipe.fetch_il_players()

        log("  Fetching game logs …")
        pipe.fetch_game_logs()

        log("  Fetching pitcher stats …")
        pipe.fetch_pitcher_stats()

        log("  Fetching team pitching stats …")
        pipe.fetch_team_pitching_stats()

        log("  Fetching batter handedness …")
        pipe.fetch_batter_handedness()

        log("  Fetching batter vs pitcher history …")
        pipe.fetch_batter_vs_pitcher()

        log("  Fetching weather …")
        pipe.fetch_weather()

        if train:
            log("  Training / loading models …")
            pipe.train_models(force=False)

        log("  Building predictions …")
        pipe.build_predictions()

        log("  Building game projections …")
        pipe.build_game_projections()

        log(f"  ✅ MLB done — {len(pipe.predictions)} player predictions")
    except Exception as e:
        log(f"  ❌ MLB error: {e}")
        traceback.print_exc()


def warm_nhl(date: str, train: bool = True):
    log("=== NHL ===")
    try:
        from data_pipeline import NHLPipeline
        pipe = NHLPipeline()

        log("  Fetching schedule …")
        pipe.fetch_schedule(date)

        if not pipe.todays_games:
            log("  No NHL games today — skipping")
            return

        log(f"  {len(pipe.todays_games)} games found")
        log("  Fetching rosters …")
        pipe.fetch_rosters()

        log("  Fetching game logs …")
        pipe.fetch_game_logs()

        log("  Fetching NST stats …")
        pipe.fetch_nst_stats()

        log("  Fetching goalie quality …")
        pipe.fetch_goalie_quality()

        log("  Fetching unavailable players …")
        pipe.fetch_unavailable_players()

        if train:
            log("  Training / loading model …")
            pipe.train_model(force_retrain=False)

        log("  Building predictions …")
        pipe.build_predictions()

        log("  Building betting projections …")
        pipe.build_betting_projections()

        log(f"  ✅ NHL done — {len(pipe.predictions)} player predictions")
    except Exception as e:
        log(f"  ❌ NHL error: {e}")
        traceback.print_exc()


def warm_nba(date: str, train: bool = True):
    log("=== NBA ===")
    try:
        from nba_pipeline import NBAPipeline
        pipe = NBAPipeline()

        log("  Fetching schedule …")
        pipe.fetch_schedule(date)

        if not pipe.games:
            log("  No NBA games today — skipping")
            return

        log(f"  {len(pipe.games)} games found")
        log("  Fetching rosters …")
        pipe.fetch_rosters()

        log("  Fetching game logs …")
        pipe.fetch_game_logs()

        log("  Fetching team defense …")
        pipe.fetch_team_defense()

        if train:
            log("  Training / loading models …")
            pipe.train_models(force=False)

        log("  Building predictions …")
        pipe.build_predictions()

        log("  Building game projections …")
        pipe.build_game_projections()

        log(f"  ✅ NBA done — {len(pipe.predictions)} player predictions")
    except Exception as e:
        log(f"  ❌ NBA error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-warm sport prediction caches")
    parser.add_argument("--date",     default=None,  help="YYYY-MM-DD (default: today ET)")
    parser.add_argument("--sport",    default="all", choices=["all","mlb","nhl","nba"])
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    args = parser.parse_args()

    date  = args.date or datetime.now(ET).strftime("%Y-%m-%d")
    train = not args.no_train

    log(f"Warming caches for {date} | sports={args.sport} | train={train}")

    if args.sport in ("all", "mlb"): warm_mlb(date, train)
    if args.sport in ("all", "nhl"): warm_nhl(date, train)
    if args.sport in ("all", "nba"): warm_nba(date, train)

    log("Cache warming complete.")
