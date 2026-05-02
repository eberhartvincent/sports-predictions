"""
nba_client.py — NBA data via ESPN public API (no auth, works from cloud IPs)

Endpoints used:
  Schedule:  site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  Roster:    site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster
  Game log:  site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog
"""

import json, os, time, requests
import pandas as pd

from config import (
    NBA_ESPN_BASE as ESPN_BASE, NBA_CACHE_DIR as NBA_CACHE,
    NBA_SEASON as CURRENT_SEASON, NBA_ESPN_TEAMS as ESPN_TEAMS,
    NBA_TEAM_NAMES, REQUEST_HEADERS as HEADERS,
    API_TIMEOUT_SECONDS, API_RETRIES,
    NBA_TTL_SCHEDULE, NBA_TTL_ROSTER, NBA_TTL_GAME_LOGS,
)

NBA_TEAMS   = ESPN_TEAMS
TEAM_ID_MAP = {v: k for k, v in ESPN_TEAMS.items()}

os.makedirs(NBA_CACHE, exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(name): return os.path.join(NBA_CACHE, f"{name}.json")

def _save(name, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)

def _load(name, ttl_minutes=60):
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 60 > ttl_minutes:
        return None
    with open(path) as f:
        return json.load(f)

def _get(url, params=None, silent_404=False):
    """GET with retry; silences 404s when expected (player has no log data)."""
    for attempt in range(API_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT_SECONDS)
            if r.status_code == 404:
                if not silent_404:
                    print(f"[NBA] 404: {url.split('/')[-1]}")
                return {}
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            return {}
        except Exception as e:
            if attempt == API_RETRIES - 1:
                print(f"[NBA] GET failed: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


# ── Public API functions ──────────────────────────────────────────────────────

def get_nba_schedule(date_str: str) -> list:
    """Return list of games for YYYY-MM-DD."""
    cached = _load(f"nba_sched_{date_str}", ttl_minutes=NBA_TTL_SCHEDULE)
    if cached is not None:
        return cached

    data  = _get(f"{ESPN_BASE}/scoreboard",
                 {"dates": date_str.replace("-", ""), "limit": 20})
    games = []
    for event in data.get("events", []):
        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_id = int(home.get("team", {}).get("id", 0))
        away_id = int(away.get("team", {}).get("id", 0))
        games.append({
            "game_id":      str(event.get("id", "")),
            "date":         date_str,
            "away_team":    ESPN_TEAMS.get(away_id, away.get("team", {}).get("abbreviation", "")),
            "home_team":    ESPN_TEAMS.get(home_id, home.get("team", {}).get("abbreviation", "")),
            "away_team_id": away_id,
            "home_team_id": home_id,
            "status":       event.get("status", {}).get("type", {}).get("description", ""),
        })

    print(f"[NBA] {len(games)} games on {date_str}")
    _save(f"nba_sched_{date_str}", games)
    return games


def get_nba_roster(team_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Return active roster for an ESPN team ID."""
    key    = f"nba_roster_{team_id}_{season}"
    cached = _load(key, ttl_minutes=NBA_TTL_ROSTER)
    if cached is not None:
        return pd.DataFrame(cached)

    data = _get(f"{ESPN_BASE}/teams/{team_id}/roster")
    rows = []
    abbrev = ESPN_TEAMS.get(team_id, str(team_id))
    for athlete in data.get("athletes", []):
        rows.append({
            "player_id":   int(athlete.get("id", 0)),
            "player_name": athlete.get("displayName", athlete.get("fullName", "")),
            "position":    athlete.get("position", {}).get("abbreviation", ""),
            "team_id":     team_id,
            "team":        abbrev,
        })

    _save(key, rows)
    return pd.DataFrame(rows)


def get_nba_player_gamelog(player_id: int,
                            season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Fetch per-game stats for one player. Tries primary URL then fallbacks."""
    key    = f"nba_log_{player_id}_{season}"
    cached = _load(key, ttl_minutes=NBA_TTL_GAME_LOGS)
    if cached is not None:
        return pd.DataFrame(cached)

    end_year = int(season.split("-")[0]) + 1   # 2026 for "2025-26"

    # Try URLs in priority order; stop at first successful response
    attempts = [
        (f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
         f"/athletes/{player_id}/gamelog", {"season": end_year}, False),
        (f"{ESPN_BASE}/athletes/{player_id}/gamelog", {"season": end_year}, True),
        (f"{ESPN_BASE}/athletes/{player_id}/gamelog", {}, True),
    ]
    data = {}
    for url, params, silent in attempts:
        data = _get(url, params, silent_404=silent)
        if data.get("seasonTypes") or data.get("entries"):
            break
        time.sleep(0.3)

    rows = _parse_gamelog(data, player_id)
    rows.sort(key=lambda r: r["game_date"])
    _save(key, rows)
    return pd.DataFrame(rows)


def _parse_gamelog(data: dict, player_id: int) -> list:
    """Parse ESPN gamelog response into a list of row dicts."""
    rows = []
    if not data:
        return rows

    if "seasonTypes" in data:
        events_map  = data.get("events", {})
        stat_labels = [c.lower() for c in data.get("labels", [])]

        def _ma(stats, label):
            """Parse 'M-A' string or plain float from stat list."""
            try:
                val = stats[stat_labels.index(label)]
                if isinstance(val, str) and "-" in val:
                    return float(val.split("-")[0]), float(val.split("-")[1])
                return float(val), 0.0
            except (ValueError, IndexError, TypeError):
                return 0.0, 0.0

        def _stat(stats, label):
            try:
                val = stats[stat_labels.index(label)]
                if isinstance(val, str) and "-" in val:
                    return float(val.split("-")[0])
                return float(val)
            except (ValueError, IndexError, TypeError):
                return 0.0

        for season_type in data.get("seasonTypes", []):
            for category in season_type.get("categories", []):
                for ev in category.get("events", []):
                    eid        = str(ev.get("eventId", ""))
                    stats_list = ev.get("stats", [])
                    info       = events_map.get(eid, {})
                    game_date  = str(info.get("gameDate", ""))
                    if not game_date or not stats_list:
                        continue

                    fgm, fga = _ma(stats_list, "fg")
                    fg3m, _  = _ma(stats_list, "3pt")
                    ftm, fta = _ma(stats_list, "ft")
                    pts      = fgm * 2 + fg3m + ftm

                    rows.append({
                        "player_id":  player_id,
                        "game_id":    eid,
                        "game_date":  game_date,
                        "is_home":    "vs" in str(info.get("atVs", "vs")).lower(),
                        "opponent":   str(info.get("opponent", {}).get("abbreviation", "")),
                        "min":        _stat(stats_list, "min"),
                        "pts":        pts,
                        "reb":        _stat(stats_list, "reb"),
                        "ast":        _stat(stats_list, "ast"),
                        "stl":        _stat(stats_list, "stl"),
                        "blk":        _stat(stats_list, "blk"),
                        "tov":        _stat(stats_list, "to"),
                        "fg3m":       fg3m,
                        "fgm":        fgm,
                        "fga":        fga,
                        "ftm":        ftm,
                        "fta":        fta,
                        "plus_minus": _stat(stats_list, "+/-"),
                    })

    elif "entries" in data:
        for entry in data.get("entries", []):
            s = {x.get("name","").lower(): float(x.get("value", 0) or 0)
                 for x in entry.get("statistics", [])}
            game_date = str(entry.get("date", ""))[:10]
            if not game_date:
                continue
            rows.append({
                "player_id":  player_id,
                "game_id":    str(entry.get("id", "")),
                "game_date":  game_date,
                "is_home":    True,
                "opponent":   "",
                "min":        s.get("minutes", 0),
                "pts":        s.get("points", 0),
                "reb":        s.get("rebounds", 0),
                "ast":        s.get("assists", 0),
                "stl":        s.get("steals", 0),
                "blk":        s.get("blocks", 0),
                "tov":        s.get("turnovers", 0),
                "fg3m":       s.get("threePointFieldGoalsMade", 0),
                "fgm":        s.get("fieldGoalsMade", 0),
                "fga":        s.get("fieldGoalsAttempted", 0),
                "ftm":        s.get("freeThrowsMade", 0),
                "fta":        s.get("freeThrowsAttempted", 0),
                "plus_minus": s.get("plusminus", 0),
            })

    return rows
