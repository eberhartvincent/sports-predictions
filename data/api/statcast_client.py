"""
data/api/statcast_client.py — Baseball Savant / Statcast API client

Fetches:
  - Exit velocity, barrel rate, hard hit% per batter (xStats)
  - Pitcher stuff grades, whiff rates
  - Umpire tendency data
  - Park-adjusted expected stats (xBA, xSLG, xwOBA)

All data is free from baseballsavant.mlb.com — no API key required.
Cache TTL: 4 hours (stats update after each game).
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd

ET        = ZoneInfo("America/New_York")
CACHE_DIR = Path("data/cache/mlb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; SportsPredictorBot/1.0)"}
DELAY     = 0.5   # seconds between requests


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"statcast_{key}.json"


def _load(key: str, ttl_minutes: int = 240):
    p = _cache_path(key)
    if not p.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).seconds / 60
    if age > ttl_minutes:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save(key: str, data):
    try:
        _cache_path(key).write_text(json.dumps(data))
    except Exception:
        pass


def get_statcast_batter_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-batter Statcast metrics for the season from Baseball Savant.
    Returns DataFrame with: player_id, player_name, exit_velocity_avg,
    barrel_pct, hard_hit_pct, xba, xslg, xwoba, whiff_pct, k_pct, bb_pct

    These are the single strongest individual predictors available — a player
    with .230 BA but .310 xBA is being unlucky and is a buy.
    """
    key = f"batter_statcast_{season}"
    cached = _load(key, ttl_minutes=240)
    if cached:
        return pd.DataFrame(cached)

    url = (
        "https://baseballsavant.mlb.com/leaderboards/expected_statistics"
        f"?type=batter&year={season}&position=&team=&min=25&csv=true"
    )
    try:
        time.sleep(DELAY)
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"  [Statcast] Batter stats HTTP {r.status_code}")
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))

        # Normalise columns
        rename = {
            "player_id":           "player_id",
            "last_name, first_name":"player_name_raw",
            "ba":                  "avg",
            "xba":                 "xba",
            "slg":                 "slg",
            "xslg":                "xslg",
            "woba":                "woba",
            "xwoba":               "xwoba",
            "xobp":                "xobp",
            "exit_velocity_avg":   "exit_velocity_avg",
            "launch_angle_avg":    "launch_angle_avg",
            "sweet_spot_percent":  "sweet_spot_pct",
            "barrel_batted_rate":  "barrel_pct",
            "hard_hit_percent":    "hard_hit_pct",
            "k_percent":           "k_pct",
            "bb_percent":          "bb_pct",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Parse player name
        if "player_name_raw" in df.columns:
            def _flip(name):
                try:
                    parts = str(name).split(", ")
                    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
                except Exception:
                    return name
            df["player_name"] = df["player_name_raw"].apply(_flip)
            df = df.drop(columns=["player_name_raw"], errors="ignore")

        # Compute luck indicators
        if "xba" in df.columns and "avg" in df.columns:
            df["ba_luck"]  = pd.to_numeric(df["avg"],  errors="coerce") - \
                             pd.to_numeric(df["xba"],  errors="coerce")
        if "xslg" in df.columns and "slg" in df.columns:
            df["slg_luck"] = pd.to_numeric(df["slg"],  errors="coerce") - \
                             pd.to_numeric(df["xslg"], errors="coerce")
        if "xwoba" in df.columns and "woba" in df.columns:
            df["woba_luck"]= pd.to_numeric(df["woba"], errors="coerce") - \
                             pd.to_numeric(df["xwoba"],errors="coerce")
            # Negative woba_luck = unlucky (performing below expected) = buy signal

        # Numeric coerce
        for col in ["exit_velocity_avg","barrel_pct","hard_hit_pct","xba","xslg",
                    "xwoba","xobp","k_pct","bb_pct","sweet_spot_pct","ba_luck",
                    "slg_luck","woba_luck","launch_angle_avg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _save(key, df.to_dict("records"))
        print(f"  [Statcast] Batter stats: {len(df)} players")
        return df

    except Exception as e:
        print(f"  [Statcast] Batter stats error: {e}")
        return pd.DataFrame()


def get_statcast_pitcher_stats(season: str) -> pd.DataFrame:
    """
    Fetch per-pitcher Statcast metrics.
    Returns: player_id, player_name, stuff_plus, location_plus,
             pitching_plus, whiff_pct, k_pct, xera, barrel_pct_against
    """
    key = f"pitcher_statcast_{season}"
    cached = _load(key, ttl_minutes=240)
    if cached:
        return pd.DataFrame(cached)

    # Stuff+ from baseball savant
    url = (
        "https://baseballsavant.mlb.com/leaderboards/pitcher-predicted-stats"
        f"?type=pitcher&year={season}&min=5&csv=true"
    )
    try:
        time.sleep(DELAY)
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(r.text))

        rename = {
            "player_id":           "player_id",
            "last_name, first_name":"player_name_raw",
            "p_era":               "era",
            "p_xera":              "xera",
            "p_k_percent":         "k_pct",
            "p_bb_percent":        "bb_pct",
            "p_whiff_percent":     "whiff_pct",
            "barrel_batted_rate":  "barrel_pct_against",
            "hard_hit_percent":    "hard_hit_pct_against",
            "exit_velocity_avg":   "exit_velocity_against",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "player_name_raw" in df.columns:
            def _flip(name):
                try:
                    parts = str(name).split(", ")
                    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
                except Exception:
                    return name
            df["pitcher_name"] = df["player_name_raw"].apply(_flip)
            df = df.drop(columns=["player_name_raw"], errors="ignore")

        for col in ["era","xera","k_pct","bb_pct","whiff_pct",
                    "barrel_pct_against","hard_hit_pct_against","exit_velocity_against"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _save(key, df.to_dict("records"))
        print(f"  [Statcast] Pitcher stats: {len(df)} pitchers")
        return df

    except Exception as e:
        print(f"  [Statcast] Pitcher stats error: {e}")
        return pd.DataFrame()


def get_umpire_tendency(game_date: str) -> dict:
    """
    Fetch home plate umpire for today's games and their tendency stats.
    Returns: {game_pk: {"umpire": name, "k_rate_factor": float,
                        "bb_rate_factor": float, "run_factor": float}}

    Umpires with small strike zones inflate walks and run scoring by ~0.5 R/game.
    This is a free edge most casual models completely miss.
    """
    key = f"umpire_{game_date}"
    cached = _load(key, ttl_minutes=60)
    if cached:
        return cached

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}&hydrate=officials"
    try:
        time.sleep(DELAY)
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {}

        result = {}
        for date_data in r.json().get("dates", []):
            for game in date_data.get("games", []):
                gid = game.get("gamePk")
                officials = game.get("officials", [])
                hp_ump = next((o["official"]["fullName"]
                               for o in officials
                               if o.get("officialType") == "Home Plate"), None)
                if hp_ump:
                    # Apply known tight/liberal umpire factors
                    # Positive run_factor = umpire inflates run scoring
                    tendency = _umpire_factors(hp_ump)
                    result[str(gid)] = {"umpire": hp_ump, **tendency}

        _save(key, result)
        return result

    except Exception as e:
        print(f"  [Statcast] Umpire data error: {e}")
        return {}


def _umpire_factors(name: str) -> dict:
    """
    Known umpire tendencies. Run factor > 1.0 = inflates run scoring.
    These are approximations based on historical called strike rates.
    A full implementation would query Baseball Savant's umpire leaderboard.
    """
    # Umpires known for tight zones (suppress runs ~0.3-0.5 per game)
    TIGHT = {"Angel Hernandez", "Joe West", "CB Bucknor", "Tom Hallion",
             "Rob Drake", "Marvin Hudson"}
    # Umpires known for liberal zones (inflate runs ~0.3-0.5 per game)
    LIBERAL = {"Ted Barrett", "Bill Miller", "Dan Bellino", "Alfonso Marquez",
               "David Rackley", "Phil Cuzzi"}

    if name in TIGHT:
        return {"k_rate_factor": 0.94, "bb_rate_factor": 1.06, "run_factor": 0.95}
    elif name in LIBERAL:
        return {"k_rate_factor": 1.06, "bb_rate_factor": 0.94, "run_factor": 1.05}
    else:
        return {"k_rate_factor": 1.00, "bb_rate_factor": 1.00, "run_factor": 1.00}


def get_sprint_speed(season: str) -> pd.DataFrame:
    """
    Fetch sprint speed data — useful for stolen base / run scoring models.
    Fast runners in high-stolen-base situations score more runs.
    """
    key = f"sprint_speed_{season}"
    cached = _load(key, ttl_minutes=240 * 7)  # weekly
    if cached:
        return pd.DataFrame(cached)

    url = (f"https://baseballsavant.mlb.com/leaderboards/sprint_speed"
           f"?year={season}&position=&team=&min=10&csv=true")
    try:
        time.sleep(DELAY)
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        rename = {"player_id":"player_id","hp_to_1b":"sprint_speed",
                  "last_name, first_name":"player_name_raw"}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        if "player_name_raw" in df.columns:
            df["player_name"] = df["player_name_raw"].apply(
                lambda n: f"{n.split(', ')[1]} {n.split(', ')[0]}" if ", " in str(n) else n)
        df["sprint_speed"] = pd.to_numeric(df.get("sprint_speed",0), errors="coerce")
        _save(key, df.to_dict("records"))
        return df
    except Exception as e:
        print(f"  [Statcast] Sprint speed error: {e}")
        return pd.DataFrame()
