"""
data/api/nhl_api.py — NHL data via ESPN public API.

Uses the same ESPN endpoint pattern as nba_client.py since
the official NHL API (api-web.nhle.com) blocks cloud/datacenter IPs.

ESPN NHL endpoints (all public, no auth required):
  Scoreboard:  site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard
  Summary:     site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={id}
  Roster:      site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{id}/roster
  Teams:       site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams

Fallback: if ESPN returns 403, tries the NHL CDN (content.nhl.com) which
serves static JSON files and is less aggressively rate-limited.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    NHL_CACHE_DIR, REQUEST_HEADERS, CURRENT_SEASON, MIN_GP,
    NHL_CONF_ELITE, NHL_CONF_HIGH, NHL_CONF_MEDIUM,
)

ET         = ZoneInfo("America/New_York")
CACHE_DIR  = Path(NHL_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_NHL   = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
CDN_NHL    = "https://api-web.nhle.com/v1"   # kept as fallback where not blocked

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.espn.com/nhl/",
    "Origin":          "https://www.espn.com",
}
DELAY = 0.3


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, max_age_minutes: int = 60):
    p = _cache_path(key)
    if not p.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).seconds / 60
    if age > max_age_minutes:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_cache(key: str, data):
    try:
        _cache_path(key).write_text(json.dumps(data, default=str))
    except Exception:
        pass


def _get_espn(endpoint: str, params: dict = None) -> dict:
    """GET from ESPN NHL API with caching."""
    cache_key = f"espn_{endpoint.replace('/', '_')}_{json.dumps(params or {}, sort_keys=True)}"
    cached = _load_cache(cache_key, max_age_minutes=30)
    if cached is not None:
        return cached

    url = f"{ESPN_NHL}/{endpoint.lstrip('/')}"
    try:
        time.sleep(DELAY)
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            _save_cache(cache_key, data)
            return data
        print(f"  [NHL/ESPN] {r.status_code}: {url}")
        return {}
    except Exception as e:
        print(f"  [NHL/ESPN] Error: {e}")
        return {}


def _get_cdn(endpoint: str) -> dict:
    """GET from NHL CDN (fallback, not always blocked)."""
    cache_key = f"cdn_{endpoint.replace('/', '_')}"
    cached = _load_cache(cache_key, max_age_minutes=30)
    if cached is not None:
        return cached

    url = f"{CDN_NHL}/{endpoint.lstrip('/')}"
    try:
        time.sleep(DELAY)
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            _save_cache(cache_key, data)
            return data
        return {}
    except Exception:
        return {}


# ── Schedule ──────────────────────────────────────────────────────────────────

def get_nhl_schedule(date_str: str) -> list:
    """
    Return list of games for a given date.
    Each game: {game_id, home_team, away_team, home_abbrev, away_abbrev,
                game_time, status, home_score, away_score}
    """
    cached = _load_cache(f"schedule_{date_str}", 60)
    if cached is not None:
        return cached

    # ESPN scoreboard
    date_fmt = date_str.replace("-", "")
    data = _get_espn("scoreboard", {"dates": date_fmt, "limit": 20})
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway")=="home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway")=="away"), competitors[1])
        games.append({
            "game_id":     event.get("id"),
            "home_team":   home.get("team",{}).get("displayName",""),
            "away_team":   away.get("team",{}).get("displayName",""),
            "home_abbrev": home.get("team",{}).get("abbreviation",""),
            "away_abbrev": away.get("team",{}).get("abbreviation",""),
            "home_id":     home.get("team",{}).get("id",""),
            "away_id":     away.get("team",{}).get("id",""),
            "game_time":   event.get("date",""),
            "status":      event.get("status",{}).get("type",{}).get("name",""),
            "home_score":  int(home.get("score",0) or 0),
            "away_score":  int(away.get("score",0) or 0),
        })

    _save_cache(f"schedule_{date_str}", games)
    return games


# ── Rosters ───────────────────────────────────────────────────────────────────

def _get_espn_teams() -> list:
    """Return list of {id, abbreviation, displayName} for all NHL teams."""
    cached = _load_cache("nhl_teams", 60 * 24)
    if cached:
        return cached
    data = _get_espn("teams", {"limit": 40})
    teams = []
    for t in data.get("sports",[{}])[0].get("leagues",[{}])[0].get("teams",[]):
        team = t.get("team", {})
        teams.append({
            "id":           team.get("id"),
            "abbreviation": team.get("abbreviation"),
            "displayName":  team.get("displayName"),
        })
    _save_cache("nhl_teams", teams)
    return teams


def get_nhl_roster(team_abbrev: str) -> pd.DataFrame:
    """
    Fetch roster for a team by abbreviation.
    Returns DataFrame with player_id, player_name, position, team.
    """
    cached = _load_cache(f"roster_{team_abbrev}", 60 * 6)
    if cached:
        return pd.DataFrame(cached)

    # Find team ESPN ID
    teams = _get_espn_teams()
    team_info = next((t for t in teams
                      if t.get("abbreviation","").upper() == team_abbrev.upper()), None)
    if not team_info:
        return pd.DataFrame()

    team_id = team_info["id"]
    data = _get_espn(f"teams/{team_id}/roster")
    rows = []
    for group in data.get("athletes", []):
        for player in (group.get("items", []) if isinstance(group, dict) else [group]):
            pid  = player.get("id")
            name = player.get("fullName", player.get("displayName",""))
            pos  = player.get("position",{}).get("abbreviation","")
            if pid and name:
                rows.append({
                    "player_id":   int(pid),
                    "player_name": name,
                    "position":    pos,
                    "team":        team_abbrev.upper(),
                })

    _save_cache(f"roster_{team_abbrev}", rows)
    return pd.DataFrame(rows)


def get_all_rosters(teams: list = None) -> pd.DataFrame:
    """Fetch rosters for all teams playing today."""
    if not teams:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        games = get_nhl_schedule(today)
        teams = list({g["home_abbrev"] for g in games} |
                     {g["away_abbrev"] for g in games})

    frames = []
    for abbrev in teams:
        df = get_nhl_roster(abbrev)
        if not df.empty:
            frames.append(df)
        time.sleep(DELAY)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Game logs ─────────────────────────────────────────────────────────────────

def _espn_player_gamelog(player_id: int, season_year: int) -> pd.DataFrame:
    """
    Fetch per-game stats for one player via ESPN.
    season_year = the year the season ENDS (e.g. 2026 for 2025-26).
    """
    cache_key = f"gamelog_espn_{player_id}_{season_year}"
    cached = _load_cache(cache_key, 60 * 4)
    if cached:
        return pd.DataFrame(cached)

    url = (f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
           f"/athletes/{player_id}/gamelog")
    try:
        time.sleep(DELAY)
        r = requests.get(url, params={"season": season_year},
                         headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        rows = []
        events   = data.get("events",{})
        stat_map = {e["id"]: e for e in events.get("previousEvents",
                    events.get("events", []))}

        for cat in data.get("seasonTypes", []):
            for entry in cat.get("categories", []):
                for ev in entry.get("events", []):
                    eid   = ev.get("eventId","")
                    stats = ev.get("stats",[])
                    info  = stat_map.get(eid,{})
                    date  = info.get("date","")[:10] if info else ""
                    if not date or not stats:
                        continue
                    # ESPN NHL gamelog stat order:
                    # G, A, PTS, +/-, PIM, EVG, PPG, SHG, GWG, EVA, PPA, SHA, S, S%
                    try:
                        rows.append({
                            "player_id":  player_id,
                            "game_date":  date,
                            "goals":      int(float(stats[0])) if len(stats)>0 else 0,
                            "assists":    int(float(stats[1])) if len(stats)>1 else 0,
                            "points":     int(float(stats[2])) if len(stats)>2 else 0,
                            "plus_minus": int(float(stats[3])) if len(stats)>3 else 0,
                            "shots":      int(float(stats[12])) if len(stats)>12 else 0,
                            "toi_seconds":0,  # ESPN doesn't give TOI in gamelog
                            "home_road":  "H",
                            "opponent":   "",
                        })
                    except (ValueError, IndexError):
                        continue

        _save_cache(cache_key, rows)
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [NHL/ESPN gamelog] player {player_id}: {e}")
        return pd.DataFrame()


def get_player_game_log(player_id: int,
                         season: str = CURRENT_SEASON,
                         game_type: int = 2) -> pd.DataFrame:
    """
    Fetch per-game stats for a player.
    season: NHL format "20252026" or ESPN year "2026"
    Tries ESPN first, falls back to NHL CDN.
    """
    # Convert NHL season format to ESPN year
    if len(season) == 8:  # "20252026"
        espn_year = int(season[4:])
    else:
        espn_year = int(season)

    df = _espn_player_gamelog(player_id, espn_year)
    if not df.empty:
        return df

    # Fallback: NHL CDN (sometimes works)
    cache_key = f"gamelog_cdn_{player_id}_{season}_{game_type}"
    cached = _load_cache(cache_key, 60 * 4)
    if cached:
        return pd.DataFrame(cached)

    data = _get_cdn(f"/player/{player_id}/game-log/{season}/{game_type}")
    rows = []
    for g in data.get("gameLog", []):
        try:
            toi = g.get("toi","0:00")
            mins, secs = (toi.split(":") + ["0"])[:2]
            toi_sec = int(mins)*60 + int(secs)
        except Exception:
            toi_sec = 0
        rows.append({
            "player_id":  player_id,
            "game_date":  g.get("gameDate",""),
            "goals":      int(g.get("goals",0)),
            "assists":    int(g.get("assists",0)),
            "points":     int(g.get("points",0)),
            "plus_minus": int(g.get("plusMinus",0)),
            "shots":      int(g.get("shots",0)),
            "toi_seconds":toi_sec,
            "home_road":  g.get("homeRoadFlag","H"),
            "opponent":   g.get("opponentAbbrev",""),
        })
    _save_cache(cache_key, rows)
    return pd.DataFrame(rows)


def fetch_all_game_logs(player_ids: list,
                         season: str = CURRENT_SEASON,
                         prior_seasons: int = 2,
                         delay: float = 0.3) -> pd.DataFrame:
    """Fetch game logs for all players, current + prior seasons."""
    def _prev(s: str) -> str:
        start = int(s[:4]) - 1
        return f"{start}{start+1}"

    seasons = [season]
    s = season
    for _ in range(prior_seasons):
        s = _prev(s)
        seasons.append(s)

    frames = []
    for pid in player_ids:
        for szn in seasons:
            try:
                df = get_player_game_log(pid, szn)
                if not df.empty:
                    df["season"] = szn
                    frames.append(df)
            except Exception as e:
                print(f"  [NHL] Player {pid} season {szn}: {e}")
            time.sleep(delay)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined = combined.dropna(subset=["game_date"])
    return combined.sort_values(["player_id","game_date"]).reset_index(drop=True)


# ── Goalies ───────────────────────────────────────────────────────────────────

def get_goalie_stats() -> dict:
    """
    Returns {team_abbrev: save_pct} for each team's likely starter.
    Uses ESPN team stats endpoint.
    """
    cached = _load_cache("goalie_stats", 60 * 6)
    if cached:
        return cached

    result = {}
    teams = _get_espn_teams()
    for team in teams:
        tid   = team.get("id")
        abbr  = team.get("abbreviation","")
        data  = _get_espn(f"teams/{tid}", {"enable": "roster"})
        # Look for goalies in roster
        for group in data.get("athletes",[]):
            items = group.get("items",[]) if isinstance(group,dict) else []
            for p in items:
                if p.get("position",{}).get("abbreviation","") == "G":
                    stats = p.get("statistics",{}).get("splits",{})
                    svpct = stats.get("categories",[{}])[0].get("stats",[{}])
                    # Try to extract save percentage
                    for s in svpct:
                        if s.get("name","").lower() in ("savepctg","savepct","sv%"):
                            result[abbr] = float(s.get("value", 0.910))
                            break
                    break
        time.sleep(DELAY)

    if not result:
        # Fall back to league average
        result = {}

    _save_cache("goalie_stats", result)
    return result


def get_goalie_matchup(home_abbrev: str,
                        away_abbrev: str,
                        game_id: str = None) -> dict:
    """
    Returns {home_save_pct, away_save_pct, home_starter, away_starter}.
    Falls back to league average (0.910) if data unavailable.
    """
    LEAGUE_AVG = 0.910
    goalie_stats = get_goalie_stats()
    return {
        "home_save_pct": goalie_stats.get(home_abbrev, LEAGUE_AVG),
        "away_save_pct": goalie_stats.get(away_abbrev, LEAGUE_AVG),
        "home_starter":  "TBD",
        "away_starter":  "TBD",
    }


# ── Game results (for backtest) ───────────────────────────────────────────────

def get_game_results(date_str: str) -> dict:
    """
    Returns {player_id: {goals, shots, assists}} for a past date.
    Used by backtest — tries ESPN summary endpoint per game.
    """
    games = get_nhl_schedule(date_str)
    results = {}
    for game in games:
        gid = game.get("game_id")
        if not gid:
            continue
        data = _get_espn("summary", {"event": gid})
        for team_data in data.get("boxscore",{}).get("players",[]):
            for grp in team_data.get("statistics",[]):
                labels = [k.lower() for k in grp.get("keys",
                          grp.get("labels", grp.get("names",[])))]
                def _idx(name, default):
                    try: return labels.index(name)
                    except ValueError: return default
                i_g = _idx("g",  _idx("goals",  0))
                i_a = _idx("a",  _idx("assists", 1))
                i_s = _idx("sog",_idx("shots",   12))
                for athlete in grp.get("athletes",[]):
                    pid   = athlete.get("athlete",{}).get("id")
                    stats = athlete.get("stats",[])
                    if not pid or not stats:
                        continue
                    try:
                        results[int(pid)] = {
                            "goals":   int(float(stats[i_g])) if i_g<len(stats) else 0,
                            "assists": int(float(stats[i_a])) if i_a<len(stats) else 0,
                            "shots":   int(float(stats[i_s])) if i_s<len(stats) else 0,
                        }
                    except (ValueError, IndexError):
                        pass
        time.sleep(DELAY)
    return results


# ── Injuries / unavailable ────────────────────────────────────────────────────

def get_team_injuries(team_abbrev: str) -> list:
    """Return list of injured player names for a team."""
    cached = _load_cache(f"injuries_{team_abbrev}", max_age_minutes=15)
    if cached is not None:
        return cached

    teams = _get_espn_teams()
    team  = next((t for t in teams
                  if t.get("abbreviation","").upper()==team_abbrev.upper()), None)
    if not team:
        return []

    data  = _get_espn(f"teams/{team['id']}/injuries")
    names = []
    for inj in data.get("injuries", []):
        name = inj.get("athlete",{}).get("fullName","")
        if name:
            names.append(name)

    _save_cache(f"injuries_{team_abbrev}", names)
    return names


def get_unavailable_players(team_abbrev: str) -> set:
    """Return set of player names unavailable tonight."""
    try:
        return set(get_team_injuries(team_abbrev))
    except Exception:
        return set()


# ── NST / Natural Stat Trick (advanced stats) ─────────────────────────────────
# These endpoints are fetched by nst_scraper.py separately.
# Kept here as stubs so imports don't break.

def get_all_situations_stats(season: str) -> pd.DataFrame:
    try:
        from nst_scraper import get_all_situations_stats as _f
        return _f(season)
    except Exception:
        return pd.DataFrame()


# ── Legacy aliases (nhl_pipeline.py compatibility) ────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict:
    """Legacy shim — try ESPN first, fall back to CDN."""
    data = _get_espn(endpoint, params)
    if data:
        return data
    return _get_cdn(endpoint)
