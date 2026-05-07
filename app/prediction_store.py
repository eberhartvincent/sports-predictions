"""
app/prediction_store.py — Load pre-computed predictions saved by warm_cache.py.

Staleness detection uses file modification time so updates are caught
even when the date hasn't changed (e.g. admin re-runs workflow same day).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

PRED_DIR = Path("data/cache/predictions")
ET       = ZoneInfo("America/New_York")


def predictions_exist(sport: str) -> bool:
    return (PRED_DIR / f"{sport}_predictions.parquet").exists()


def predictions_date(sport: str) -> Optional[str]:
    """Returns the date string the saved predictions are for."""
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text()).get("date")
    except Exception:
        return None


def predictions_mtime(sport: str) -> Optional[float]:
    """
    Returns file modification timestamp of the predictions parquet.
    Used for staleness detection — newer file on disk = reload.
    This catches updates even within the same day.
    """
    pred_file = PRED_DIR / f"{sport}_predictions.parquet"
    if not pred_file.exists():
        return None
    try:
        return pred_file.stat().st_mtime
    except Exception:
        return None


def load_predictions(sport: str) -> dict:
    """Load pre-computed predictions from disk. Returns instantly."""
    result = {
        "predictions":         pd.DataFrame(),
        "pitcher_predictions": pd.DataFrame(),
        "game_projections":    [],
        "games":               [],
        "metrics":             {},
        "meta":                {},
        "updated_at":          None,
        "mtime":               None,
    }

    if not PRED_DIR.exists():
        return result

    try:
        pred_file = PRED_DIR / f"{sport}_predictions.parquet"
        if pred_file.exists():
            result["predictions"] = pd.read_parquet(pred_file)
            result["mtime"]       = pred_file.stat().st_mtime

        pitcher_file = PRED_DIR / f"{sport}_pitcher_predictions.parquet"
        if pitcher_file.exists():
            result["pitcher_predictions"] = pd.read_parquet(pitcher_file)

        for key, fname in [
            ("game_projections", f"{sport}_game_projections.json"),
            ("games",            f"{sport}_games.json"),
            ("metrics",          f"{sport}_metrics.json"),
            ("meta",             f"{sport}_meta.json"),
        ]:
            f = PRED_DIR / fname
            if f.exists():
                result[key] = json.loads(f.read_text())

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
        meta = json.loads(meta_file.read_text())
        ts   = meta.get("updated_at")
        if ts:
            dt = datetime.fromisoformat(ts).astimezone(ET)
            return dt.strftime("%I:%M %p ET on %b %d")
    except Exception:
        pass
    return None
