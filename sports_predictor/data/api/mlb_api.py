"""
mlb_api.py — MLB Stats API wrapper (statsapi.mlb.com, free, no key required)
"""

import json, os, time, requests
from datetime import datetime
from typing import Optional
import pandas as pd

from config import (
    MLB_API_BASE as MLB_API, MLB_CACHE_DIR as MLB_CACHE,
    MLB_TEAMS, MLB_TEAM_NAMES,
    MLB_TTL_SCHEDULE, MLB_TTL_ROSTER, MLB_TTL_GAME_LOGS, MLB_TTL_PITCHERS,
    REQUEST_HEADERS as HEADERS, API_TIMEOUT_SECONDS, API_RETRIES,
)
os.makedirs(MLB_CACHE, exist_ok=True)


def _cache(name): return os.path.join(MLB_CACHE, f"{name}.json")

def _save(name, data):
    with open(_cache(name), "w") as f: json.dump(data, f)

def _load(name, ttl=60):
    p = _cache(name)
    if not os.path.exists(p): return None
    if (time.time() - os.path.getmtime(p)) / 60 > ttl: return None
    with open(p) as f: return json.load(f)

def _get(path, params=None):
    try:
        r = requests.get(f"{MLB_API}{path}", params=params,
                         headers=HEADERS, timeout=API_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[MLB API] {path} → {e}")
        return {}

def get_mlb_schedule(date_str: str) -> list:
    key = f"mlb_sched_{date_str}"
    cached = _load(key, ttl=MLB_TTL_SCHEDULE)
    if cached: return cached

    data = _get("/schedule", {"date": date_str, "sportId": 1,
                               "hydrate": "team,linescore,probablePitcher"})
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away_id = g.get("teams",{}).get("away",{}).get("team",{}).get("id")
            home_id = g.get("teams",{}).get("home",{}).get("team",{}).get("id")

            # Parse probable pitchers directly from the schedule hydration
            away_pitcher = g.get("teams",{}).get("away",{}).get("probablePitcher",{})
            home_pitcher = g.get("teams",{}).get("home",{}).get("probablePitcher",{})

            games.append({
                "game_id":              g.get("gamePk"),
                "date":                 date_str,
                "away_team":            MLB_TEAMS.get(away_id, str(away_id)),
                "home_team":            MLB_TEAMS.get(home_id, str(home_id)),
                "away_team_id":         away_id,
                "home_team_id":         home_id,
                "status":               g.get("status",{}).get("abstractGameState",""),
                "start_time":           g.get("gameDate",""),
                "away_pitcher_id":      away_pitcher.get("id"),
                "away_pitcher_name":    away_pitcher.get("fullName",""),
                "home_pitcher_id":      home_pitcher.get("id"),
                "home_pitcher_name":    home_pitcher.get("fullName",""),
            })
    print(f"[MLB] {len(games)} games on {date_str}")
    for g in games:
        print(f"  {g['away_team']} ({g.get('away_pitcher_name','TBD')}) "
              f"@ {g['home_team']} ({g.get('home_pitcher_name','TBD')})")
    _save(key, games)
    return games

def get_mlb_roster(team_id: int, season: str = "2025") -> pd.DataFrame:
    key = f"mlb_roster_{team_id}_{season}"
    cached = _load(key, ttl=MLB_TTL_ROSTER)
    if cached: return pd.DataFrame(cached)

    data = _get(f"/teams/{team_id}/roster",
                {"rosterType": "active", "season": season})
    rows = []
    for p in data.get("roster", []):
        pi   = p.get("person", {})
        pos  = p.get("position", {}).get("abbreviation", "")
        rows.append({
            "player_id":   pi.get("id"),
            "player_name": pi.get("fullName", ""),
            "position":    pos,
            "team_id":     team_id,
            "team":        MLB_TEAMS.get(team_id, str(team_id)),
        })
    _save(key, rows)
    return pd.DataFrame(rows)

def get_mlb_player_gamelog(player_id: int, season: str = "2025") -> pd.DataFrame:
    key = f"mlb_log_{player_id}_{season}"
    cached = _load(key, ttl=MLB_TTL_ROSTER)
    if cached: return pd.DataFrame(cached)

    data = _get(f"/people/{player_id}/stats",
                {"stats": "gameLog", "season": season,
                 "group": "hitting", "gameType": "R"})
    rows = []
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        rows.append({
            "player_id":    player_id,
            "game_date":    split.get("date",""),
            "team":         split.get("team",{}).get("abbreviation",""),
            "opponent":     split.get("opponent",{}).get("abbreviation",""),
            "home_away":    split.get("isHome", True),
            "ab":           int(s.get("atBats", 0)),
            "hits":         int(s.get("hits", 0)),
            "doubles":      int(s.get("doubles", 0)),
            "triples":      int(s.get("triples", 0)),
            "hr":           int(s.get("homeRuns", 0)),
            "rbi":          int(s.get("rbi", 0)),
            "runs":         int(s.get("runs", 0)),
            "bb":           int(s.get("baseOnBalls", 0)),
            "k":            int(s.get("strikeOuts", 0)),
            "sb":           int(s.get("stolenBases", 0)),
            "tb":           int(s.get("totalBases", 0)),
            "avg":          float(s.get("avg", 0) or 0),
            "obp":          float(s.get("obp", 0) or 0),
            "slg":          float(s.get("slg", 0) or 0),
            "hit_flag":     1 if int(s.get("hits", 0)) > 0 else 0,
            "hr_flag":      1 if int(s.get("homeRuns", 0)) > 0 else 0,
        })
    _save(key, rows)
    return pd.DataFrame(rows)

def get_mlb_pitcher_gamelog(player_id: int, season: str = "2025") -> pd.DataFrame:
    key = f"mlb_pitcher_{player_id}_{season}"
    cached = _load(key, ttl=MLB_TTL_ROSTER)
    if cached: return pd.DataFrame(cached)

    data = _get(f"/people/{player_id}/stats",
                {"stats": "gameLog", "season": season,
                 "group": "pitching", "gameType": "R"})
    rows = []
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        try:
            era = float(s.get("era", 4.5) or 4.5)
        except: era = 4.5
        rows.append({
            "player_id":   player_id,
            "game_date":   split.get("date",""),
            "team":        split.get("team",{}).get("abbreviation",""),
            "era":         era,
            "whip":        float(s.get("whip", 1.3) or 1.3),
            "k9":          float(s.get("strikeoutsPer9Inn", 8.0) or 8.0),
        })
    _save(key, rows)
    return pd.DataFrame(rows)

def get_team_pitching_stats(season: str = "2026") -> dict:
    """
    Fetch season pitching stats for every team.
    Returns {team_abbrev: {era, whip, k9}} using MLB Stats API team stats endpoint.
    Used to enrich training data with real opponent pitcher quality.
    """
    key = f"mlb_team_pitching_{season}"
    cached = _load(key, ttl=240)
    if cached:
        return cached

    data = _get("/teams/stats", {
        "stats":     "season",
        "group":     "pitching",
        "season":    season,
        "sportId":   1,
        "gameType":  "R",
    })

    result = {}
    for rec in data.get("stats", [{}])[0].get("splits", []):
        team   = rec.get("team", {})
        abbrev = team.get("abbreviation", "")
        s      = rec.get("stat", {})
        if not abbrev:
            continue
        try:
            era  = float(s.get("era",  4.20) or 4.20)
            whip = float(s.get("whip", 1.30) or 1.30)
            # k9 = strikeouts / innings * 9
            k    = float(s.get("strikeOuts", 0) or 0)
            ip   = float(s.get("inningsPitched", 1) or 1)
            k9   = round(k / ip * 9, 2) if ip > 0 else 8.5
        except Exception:
            era, whip, k9 = 4.20, 1.30, 8.5
        result[abbrev] = {"era": era, "whip": whip, "k9": k9}

    _save(key, result)
    return result


# ── Weather ───────────────────────────────────────────────────────────────────

# Ballpark coordinates for weather lookup
BALLPARK_COORDS = {
    "ARI":(33.4453,-112.0667), "ATL":(33.8907,-84.4677),  "BAL":(39.2838,-76.6216),
    "BOS":(42.3467,-71.0972),  "CHC":(41.9484,-87.6553),  "CWS":(41.8299,-87.6338),
    "CIN":(39.0974,-84.5061),  "CLE":(41.4962,-81.6852),  "COL":(39.7559,-104.9942),
    "DET":(42.3390,-83.0485),  "HOU":(29.7573,-95.3555),  "KC": (39.0517,-94.4803),
    "LAA":(33.8003,-117.8827), "LAD":(34.0739,-118.2400), "MIA":(25.7781,-80.2197),
    "MIL":(43.0280,-87.9712),  "MIN":(44.9817,-93.2778),  "NYM":(40.7571,-73.8458),
    "NYY":(40.8296,-73.9262),  "OAK":(37.7516,-122.2005), "PHI":(39.9061,-75.1665),
    "PIT":(40.4469,-80.0057),  "SD": (32.7073,-117.1566), "SF": (37.7786,-122.3893),
    "SEA":(47.5914,-122.3325), "STL":(38.6226,-90.1928),  "TB": (27.7682,-82.6534),
    "TEX":(32.7512,-97.0832),  "TOR":(43.6414,-79.3894),  "WSH":(38.8730,-77.0074),
}

# Stadiums with roofs (weather has minimal impact — still fetch for completeness)
ROOFED_STADIUMS = {"ARI","HOU","MIA","MIL","MIN","SEA","TB","TOR"}


def get_game_weather(home_team: str, game_date: str, start_time_utc: str = "") -> dict:
    """
    Fetch weather conditions at game time using Open-Meteo (free, no API key).

    Returns dict with:
      temp_c, temp_f        — temperature
      wind_speed_mph        — wind speed
      wind_dir_deg          — wind direction (0=N, 90=E, 180=S, 270=W)
      wind_component        — +ve = blowing out (helps HRs), -ve = blowing in
      precip_mm             — precipitation
      is_dome               — True if roofed stadium
      weather_hr_factor     — multiplicative HR adjustment (e.g. 1.08 = 8% boost)
      weather_desc          — human-readable description
    """
    key = f"weather_{home_team}_{game_date}"
    cached = _load(key, ttl=60)
    if cached:
        return cached

    coords = BALLPARK_COORDS.get(home_team)
    is_dome = home_team in ROOFED_STADIUMS

    default = {
        "temp_f": 72, "temp_c": 22, "wind_speed_mph": 0, "wind_dir_deg": 0,
        "wind_component": 0.0, "precip_mm": 0.0, "is_dome": is_dome,
        "weather_hr_factor": 1.0, "weather_desc": "Unknown",
    }

    if is_dome or not coords:
        _save(key, default)
        return default

    lat, lon = coords

    # Parse game start hour (UTC → use as-is for the API)
    try:
        from datetime import datetime as dt
        hour = int(dt.strptime(start_time_utc[:16], "%Y-%m-%dT%H:%M").hour) if start_time_utc else 19
    except Exception:
        hour = 19

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":       lat,
            "longitude":      lon,
            "hourly":         "temperature_2m,windspeed_10m,winddirection_10m,precipitation",
            "windspeed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "forecast_days":  2,
            "timezone":       "UTC",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        hourly     = data.get("hourly", {})
        times      = hourly.get("time", [])
        target_dt  = f"{game_date}T{hour:02d}:00"

        # Find the closest hour to game time
        idx = 0
        for i, t in enumerate(times):
            if t >= target_dt:
                idx = i
                break

        temp_f      = float(hourly.get("temperature_2m",    [72])[idx])
        wind_mph    = float(hourly.get("windspeed_10m",     [0])[idx])
        wind_deg    = float(hourly.get("winddirection_10m", [0])[idx])
        precip_mm   = float(hourly.get("precipitation",     [0])[idx])

        # Wind component: how much blows toward CF (roughly 0° from home plate)
        # Wind blowing FROM 180° (S) = blowing OUT to CF = positive (helps HRs)
        # Wind blowing FROM 0° (N)   = blowing IN from CF = negative (hurts HRs)
        import math
        wind_component = wind_mph * math.cos(math.radians(wind_deg - 180))

        # HR factor: each 1 mph of outward wind ≈ +0.5% HR rate
        # Temperature: each 10°F above 72°F ≈ +1% HR rate (ball travels farther)
        temp_factor = 1.0 + max(0, (temp_f - 72) / 10) * 0.01
        wind_factor = 1.0 + (wind_component / 10) * 0.05
        hr_factor   = round(temp_factor * wind_factor, 4)

        # Human-readable description
        if wind_mph < 5:
            wind_desc = "calm"
        elif wind_component > 5:
            wind_desc = f"{wind_mph:.0f} mph out"
        elif wind_component < -5:
            wind_desc = f"{wind_mph:.0f} mph in"
        else:
            wind_desc = f"{wind_mph:.0f} mph crosswind"
        weather_desc = f"{temp_f:.0f}°F, {wind_desc}"
        if precip_mm > 0.5:
            weather_desc += f", {precip_mm:.1f}mm precip"

        result = {
            "temp_f":            temp_f,
            "temp_c":            round((temp_f - 32) * 5/9, 1),
            "wind_speed_mph":    wind_mph,
            "wind_dir_deg":      wind_deg,
            "wind_component":    round(wind_component, 2),
            "precip_mm":         precip_mm,
            "is_dome":           False,
            "weather_hr_factor": hr_factor,
            "weather_desc":      weather_desc,
        }
        _save(key, result)
        return result

    except Exception as e:
        print(f"[Weather] {home_team} {game_date}: {e}")
        _save(key, default)
        return default


# ── Batter vs Pitcher history ─────────────────────────────────────────────────

def get_batter_vs_pitcher(batter_id: int, pitcher_id: int) -> dict:
    """
    Fetch career batter vs. pitcher splits from MLB Stats API.

    Returns dict with:
      bvp_ab, bvp_hits, bvp_hr, bvp_avg, bvp_slg, bvp_ops
      bvp_sample_weight  — reliability weight (0–1), low when <10 AB
    """
    key = f"bvp_{batter_id}_{pitcher_id}"
    cached = _load(key, ttl=1440)   # 24h — career stats don't change much
    if cached:
        return cached

    default = {
        "bvp_ab": 0, "bvp_hits": 0, "bvp_hr": 0,
        "bvp_avg": 0.0, "bvp_slg": 0.0, "bvp_ops": 0.0,
        "bvp_sample_weight": 0.0,
    }

    try:
        data = _get(f"/people/{batter_id}/stats", {
            "stats":    "vsPlayer",
            "opposingPlayerId": pitcher_id,
            "group":    "hitting",
            "gameType": "R",
        })

        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            _save(key, default)
            return default

        s   = splits[0].get("stat", {})
        ab  = int(s.get("atBats", 0))
        h   = int(s.get("hits", 0))
        hr  = int(s.get("homeRuns", 0))
        avg = float(s.get("avg", 0) or 0)
        slg = float(s.get("slg", 0) or 0)
        obp = float(s.get("obp", 0) or 0)
        ops = round(obp + slg, 4)

        # Sample weight: low confidence below 10 AB, full confidence at 50+ AB
        sample_weight = min(1.0, ab / 50)

        result = {
            "bvp_ab":             ab,
            "bvp_hits":           h,
            "bvp_hr":             hr,
            "bvp_avg":            avg,
            "bvp_slg":            slg,
            "bvp_ops":            ops,
            "bvp_sample_weight":  round(sample_weight, 3),
        }
        _save(key, result)
        return result

    except Exception as e:
        print(f"[BvP] {batter_id} vs {pitcher_id}: {e}")
        _save(key, default)
        return default
