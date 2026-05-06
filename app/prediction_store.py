"""app/prediction_store.py — Load pre-computed predictions saved by warm_cache.py."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

PRED_DIR = Path("data/cache/predictions")
ET       = ZoneInfo("America/New_York")


def predictions_exist(sport: str) -> bool:
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return False
    try:
        meta  = json.loads(meta_file.read_text())
        today = datetime.now(ET).strftime("%Y-%m-%d")
        return meta.get("date","") == today
    except Exception:
        return False


def load_predictions(sport: str) -> dict:
    result = {
        "predictions":         pd.DataFrame(),
        "pitcher_predictions": pd.DataFrame(),
        "game_projections":    [],
        "games":               [],
        "metrics":             {},
        "meta":                {},
        "updated_at":          None,
    }
    if not PRED_DIR.exists():
        return result
    try:
        for key, fname, loader in [
            ("predictions",         f"{sport}_predictions.parquet",        pd.read_parquet),
            ("pitcher_predictions", f"{sport}_pitcher_predictions.parquet", pd.read_parquet),
        ]:
            f = PRED_DIR / fname
            if f.exists():
                result[key] = loader(f)
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


def is_stale(sport: str) -> bool:
    """True if the file on disk is newer/different date than what may be in session state."""
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return False
    try:
        meta  = json.loads(meta_file.read_text())
        saved = meta.get("date","")
        today = datetime.now(ET).strftime("%Y-%m-%d")
        return saved == today
    except Exception:
        return False


def predictions_date(sport: str) -> Optional[str]:
    """Returns the date string the saved predictions are for."""
    meta_file = PRED_DIR / f"{sport}_meta.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text()).get("date")
    except Exception:
        return None


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
