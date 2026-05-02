"""
app/prediction_store.py — Load pre-computed predictions saved by warm_cache.py.

All three tabs call load_predictions(sport) on startup.
Returns instantly from disk — no API calls, no model runs, no waiting.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd

PRED_DIR = Path("data/cache/predictions")
ET = ZoneInfo("America/New_York")


def predictions_exist(sport: str) -> bool:
    """True if warm_cache.py has saved predictions for this sport today."""
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
        saved_date = meta.get("date", "")
        today = datetime.now(ET).strftime("%Y-%m-%d")
        return saved_date == today
    except Exception:
        return False


def load_predictions(sport: str) -> dict:
    """
    Load pre-computed predictions from disk.
    Returns a dict with keys: predictions, game_projections, games, metrics,
                               pitcher_predictions, meta, updated_at
    Returns empty dict if no predictions saved yet.
    """
    result = {
        "predictions":        pd.DataFrame(),
        "pitcher_predictions": pd.DataFrame(),
        "game_projections":   [],
        "games":              [],
        "metrics":            {},
        "meta":               {},
        "updated_at":         None,
    }

    if not PRED_DIR.exists():
        return result

    try:
        # Predictions
        pred_file = PRED_DIR / f"{sport}_predictions.parquet"
        if pred_file.exists():
            result["predictions"] = pd.read_parquet(pred_file)

        # Pitcher predictions (MLB only)
        pitcher_file = PRED_DIR / f"{sport}_pitcher_predictions.parquet"
        if pitcher_file.exists():
            result["pitcher_predictions"] = pd.read_parquet(pitcher_file)

        # Game projections
        proj_file = PRED_DIR / f"{sport}_game_projections.json"
        if proj_file.exists():
            result["game_projections"] = json.loads(proj_file.read_text())

        # Games list
        games_file = PRED_DIR / f"{sport}_games.json"
        if games_file.exists():
            result["games"] = json.loads(games_file.read_text())

        # Model metrics
        metrics_file = PRED_DIR / f"{sport}_metrics.json"
        if metrics_file.exists():
            result["metrics"] = json.loads(metrics_file.read_text())

        # Metadata
        meta_file = PRED_DIR / f"{sport}_meta.json"
        if meta_file.exists():
            result["meta"] = json.loads(meta_file.read_text())
            result["updated_at"] = result["meta"].get("updated_at")

    except Exception as e:
        print(f"[prediction_store] Error loading {sport}: {e}")

    return result


def last_updated(sport: str) -> Optional[str]:
    """Human-readable string of when predictions were last saved."""
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return None
    try:
        meta  = json.loads(meta_file.read_text())
        ts    = meta.get("updated_at")
        if ts:
            dt = datetime.fromisoformat(ts).astimezone(ET)
            return dt.strftime("%I:%M %p ET on %b %d")
    except Exception:
        pass
    return None
