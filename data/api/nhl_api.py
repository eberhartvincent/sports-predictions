"""
nhl_api.py — Fetches data from the free NHL API (api-web.nhle.com)

Endpoints used:
  /v1/schedule/today              → today's games
  /v1/schedule/{date}             → games on a specific date
  /v1/roster/{team}/current       → team rosters
  /v1/player/{id}/game-log/{season}/{gameType}
                                  → per-game stats for a player
  /v1/skater-stats-leaders/current → league stat leaders
"""

import requests
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import NHL_API_BASE, REQUEST_HEADERS, CACHE_DIR, CURRENT_SEASON, SEASON_TYPE

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict = None, retries: int = 3) -> dict:
    """GET wrapper with basic retry logic."""
    url = f"{NHL_API_BASE}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[NHL API] Failed: {url} → {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


def _load_cache(name: str, max_age_minutes: int = 30) -> Optional[dict]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 60
    if age > max_age_minutes:
        return None
    with open(path) as f:
        return json.load(f)


# ── Schedule ──────────────────────────────────────────────────────────────────

def get_today_schedule() -> list[dict]:
    """Return list of games scheduled for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"schedule_{today}"
    cached = _load_cache(cache_key, max_age_minutes=60)
    if cached:
        return cached

    # Fetch by explicit date — more reliable than /schedule/now
    data = _get(f"/schedule/{today}")
    games = _parse_schedule(data)

    # If nothing returned, also try /schedule/now as fallback
    if not games:
        data = _get("/schedule/now")
        games = _parse_schedule(data)

    print(f"  [Schedule] Fetched {len(games)} games for {today}")
    for g in games:
        print(f"    {g['date']} | {g['away_team']} @ {g['home_team']} | status={g['status']}")

    _save_cache(cache_key, games)
    return games


def get_schedule_for_date(date_str: str) -> list[dict]:
    """Return list of games for a specific date (YYYY-MM-DD)."""
    cache_key = f"schedule_{date_str}"
    cached = _load_cache(cache_key, max_age_minutes=1440)
    if cached:
        return cached

    data = _get(f"/schedule/{date_str}")
    games = _parse_schedule(data)
    _save_cache(cache_key, games)
    return games


def _parse_schedule(data: dict) -> list[dict]:
    """Parse the raw schedule payload into a flat list of game dicts."""
    games = []
    for week in data.get("gameWeek", []):
        week_date = week.get("date", "")
        for game in week.get("games", []):
            away = game.get("awayTeam", {})
            home = game.get("homeTeam", {})

            # Try every possible location for the date
            game_date = (
                game.get("gameDate")
                or week_date
                or str(game.get("startTimeUTC", ""))[:10]
                or ""
            )
            # startTimeUTC looks like "2026-03-07T23:00:00Z" — grab first 10 chars
            if not game_date or game_date == "None":
                utc = str(game.get("startTimeUTC", ""))
                game_date = utc[:10] if len(utc) >= 10 else ""

            games.append({
                "game_id":        game.get("id"),
                "date":           game_date,
                "status":         game.get("gameState"),
                "away_team":      away.get("abbrev"),
                "away_team_name": away.get("commonName", {}).get("default", ""),
                "home_team":      home.get("abbrev"),
                "home_team_name": home.get("commonName", {}).get("default", ""),
                "venue":          game.get("venue", {}).get("default", ""),
                "start_time_utc": game.get("startTimeUTC", ""),
            })
    return games


# ── Rosters ───────────────────────────────────────────────────────────────────

def get_team_roster(team_abbrev: str) -> pd.DataFrame:
    """Return current roster for a team as a DataFrame."""
    cache_key = f"roster_{team_abbrev}"
    cached = _load_cache(cache_key, max_age_minutes=120)
    if cached:
        return pd.DataFrame(cached)

    data = _get(f"/roster/{team_abbrev}/current")
    players = []
    for group in ("forwards", "defensemen"):
        for p in data.get(group, []):
            players.append({
                "player_id":   p.get("id"),
                "player_name": f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}",
                "position":    p.get("positionCode"),
                "sweater_number": p.get("sweaterNumber"),
                "team":        team_abbrev,
            })

    _save_cache(cache_key, players)
    return pd.DataFrame(players)


def get_all_rosters(team_list: list[str]) -> pd.DataFrame:
    """Fetch rosters for all given teams and concatenate."""
    frames = []
    for team in team_list:
        try:
            df = get_team_roster(team)
            if not df.empty:
                frames.append(df)
            time.sleep(0.3)
        except Exception as e:
            print(f"[Roster] Could not fetch {team}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ── Player Game Logs ──────────────────────────────────────────────────────────

def get_player_game_log(player_id: int, season: str = CURRENT_SEASON,
                         game_type: int = SEASON_TYPE) -> pd.DataFrame:
    """
    Fetch per-game stats for a player for the given season.
    Returns a DataFrame with one row per game.
    """
    cache_key = f"gamelog_{player_id}_{season}_{game_type}"
    cached = _load_cache(cache_key, max_age_minutes=120)
    if cached:
        return pd.DataFrame(cached)

    data = _get(f"/player/{player_id}/game-log/{season}/{game_type}")
    rows = []
    for g in data.get("gameLog", []):
        rows.append({
            "player_id":    player_id,
            "game_id":      g.get("gameId"),
            "game_date":    g.get("gameDate"),
            "team":         g.get("teamAbbrev"),
            "opponent":     g.get("opponentAbbrev"),
            "home_road":    g.get("homeRoadFlag"),   # "H" or "R"
            "goals":        g.get("goals", 0),
            "assists":      g.get("assists", 0),
            "points":       g.get("points", 0),
            "shots":        g.get("shots", 0),
            "toi_seconds":  _parse_toi(g.get("toi", "0:00")),
            "pim":          g.get("pim", 0),
            "plus_minus":   g.get("plusMinus", 0),
            "pp_goals":     g.get("powerPlayGoals", 0),
            "pp_points":    g.get("powerPlayPoints", 0),
            "sh_goals":     g.get("shortHandedGoals", 0),
            "scored_goal":  1 if g.get("goals", 0) > 0 else 0,
        })

    _save_cache(cache_key, rows)
    return pd.DataFrame(rows)


def _parse_toi(toi_str: str) -> float:
    """Convert 'MM:SS' to seconds (float)."""
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


# ── Bulk Game Log Fetching ────────────────────────────────────────────────────

def fetch_all_game_logs(player_ids: list[int],
                         season: str = CURRENT_SEASON,
                         delay: float = 0.25) -> pd.DataFrame:
    """
    Fetch game logs for a list of player IDs.
    Returns combined DataFrame sorted by player and date.
    """
    frames = []
    for pid in player_ids:
        try:
            df = get_player_game_log(pid, season)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[GameLog] Player {pid} error: {e}")
        time.sleep(delay)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        combined = combined.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        return combined
    return pd.DataFrame()


# ── Season-level skater stats (for quick reference) ───────────────────────────

def get_skater_stats_leaders(category: str = "goals", limit: int = 100) -> list[dict]:
    """Fetch current season stat leaders (goals, assists, points, etc.)."""
    cache_key = f"leaders_{category}_{CURRENT_SEASON}"
    cached = _load_cache(cache_key, max_age_minutes=240)
    if cached:
        return cached

    data = _get(f"/skater-stats-leaders/{CURRENT_SEASON}/{SEASON_TYPE}",
                params={"categories": category, "limit": limit})
    leaders = []
    for item in data.get(category, []):
        p = item.get("player", {})
        leaders.append({
            "player_id":   p.get("id"),
            "player_name": f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}",
            "team":        p.get("teamAbbrevs", ""),
            "position":    p.get("position", ""),
            category:      item.get("value"),
        })

    _save_cache(cache_key, leaders)
    return leaders


# ── Goalie stats ──────────────────────────────────────────────────────────────

def get_goalie_stats() -> pd.DataFrame:
    """Fetch save percentage and GAA for all current-season goalies."""
    cache_key = f"goalies_{CURRENT_SEASON}"
    cached = _load_cache(cache_key, max_age_minutes=240)
    if cached:
        return pd.DataFrame(cached)

    data = _get(f"/goalie-stats-leaders/{CURRENT_SEASON}/{SEASON_TYPE}",
                params={"categories": "savePctg", "limit": 60})

    rows = []
    for item in data.get("savePctg", []):
        p = item.get("player", {})
        rows.append({
            "player_id":   p.get("id"),
            "goalie_name": f"{p.get('firstName',{}).get('default','')} {p.get('lastName',{}).get('default','')}",
            "team":        p.get("teamAbbrevs", ""),
            "save_pct":    item.get("value", 0.900),
        })

    # Also fetch GAA
    data2 = _get(f"/goalie-stats-leaders/{CURRENT_SEASON}/{SEASON_TYPE}",
                 params={"categories": "goalsAgainstAverage", "limit": 60})
    gaa_map = {}
    for item in data2.get("goalsAgainstAverage", []):
        p = item.get("player", {})
        pid = p.get("id")
        gaa_map[pid] = item.get("value", 3.0)

    for row in rows:
        row["gaa"] = gaa_map.get(row["player_id"], 3.0)

    _save_cache(cache_key, rows)
    return pd.DataFrame(rows)


def get_team_goalie_quality() -> dict:
    """
    Returns a dict mapping team_abbrev → avg_save_pct for all goalies on that team.
    Used as a feature: facing a bad goalie = higher goal probability.
    """
    df = get_goalie_stats()
    if df.empty:
        return {}
    # Normalise team column (may be comma-separated if traded)
    team_sv = {}
    for _, row in df.iterrows():
        for team in str(row["team"]).split(","):
            team = team.strip()
            if team not in team_sv:
                team_sv[team] = []
            team_sv[team].append(row["save_pct"])
    return {t: sum(v) / len(v) for t, v in team_sv.items()}


# ── Injuries & scratches ──────────────────────────────────────────────────────

def get_team_injuries(team_abbrev: str) -> set:
    """
    Returns player_ids to exclude: injured, IR, LTIR, or inactive.
    Uses the NHL roster API + checks both injuryStatus and rosterStatus fields.
    Cache TTL is 30 min so moves within a day are caught quickly.
    """
    cache_key = f"injuries_{team_abbrev}"
    cached = _load_cache(cache_key, max_age_minutes=30)
    if cached is not None:
        return set(cached)

    injured_ids = set()
    data = _get(f"/roster/{team_abbrev}/current")

    # Terms that indicate a player is unavailable
    UNAVAILABLE_TERMS = {
        "IR", "LTIR", "IR-NR", "10-DAY-DL", "60-DAY-DL",
        "INJURED", "DTD", "DAY-TO-DAY", "OUT", "SUSPENDED",
        "CONDITIONING", "RECALL", "REASSIGNED",
    }
    # Terms that explicitly mean active — anything else is flagged
    ACTIVE_TERMS = {"", "ACTIVE", "OK", "NONE", "AVAILABLE", "ACT", "NONE"}

    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            pid   = p.get("id")
            fname = p.get("firstName", {}).get("default", "")
            lname = p.get("lastName", {}).get("default", "")

            is_injured = False

            # active=False is the most reliable signal
            if p.get("active") is False:
                is_injured = True

            # Check all status fields
            for field in ("injuryStatus", "status", "statusCode", "rosterStatus"):
                val = str(p.get(field, "") or "").upper().strip()
                # Flag if explicitly unavailable
                if val in UNAVAILABLE_TERMS:
                    is_injured = True
                    break
                # Flag if non-empty and not an active term
                if val and val not in ACTIVE_TERMS:
                    is_injured = True
                    break

            if is_injured:
                injured_ids.add(pid)
                print(f"[Injuries] Excluding {fname} {lname} ({team_abbrev})")

    _save_cache(cache_key, list(injured_ids))
    return injured_ids


def get_confirmed_starting_goalies(games: list[dict]) -> dict:
    """
    Fetch the confirmed starting goalie for each game from the NHL gamecenter API.
    Returns {team_abbrev: save_pct} for every team playing today.

    The landing endpoint exposes the starting goalie under teamGameStats.
    Falls back to season team average if the starter isn't yet confirmed
    (typically confirmed 1-2 hours before puck drop).

    This is the most important goalie feature — facing a backup (.890)
    vs a starter (.920) is a 3x difference in expected extra goals conceded.
    """
    cache_key = f"starting_goalies_{'_'.join(str(g.get('game_id','')) for g in games)}"
    cached = _load_cache(cache_key, max_age_minutes=30)
    if cached:
        return cached

    # First get season save% for every goalie so we can look up the starter's rate
    goalie_df = get_goalie_stats()
    goalie_sv = {}   # player_id → save_pct
    goalie_names = {}
    if not goalie_df.empty and "player_id" in goalie_df.columns:
        for _, row in goalie_df.iterrows():
            goalie_sv[row["player_id"]]    = float(row.get("save_pct", 0.910))
            goalie_names[row["player_id"]] = str(row.get("goalie_name", ""))

    team_avg = get_team_goalie_quality()  # fallback
    result   = {}

    for game in games:
        gid  = game.get("game_id")
        away = game.get("away_team", "")
        home = game.get("home_team", "")
        if not gid:
            continue

        try:
            data = _get(f"/gamecenter/{gid}/landing")
            if not data:
                raise ValueError("Empty response")

            for side_key, team_abbrev in [("homeTeam", home), ("awayTeam", away)]:
                team_data = data.get(side_key, {})

                # Primary: look for startingGoalie in teamGameStats
                starter_id = None
                starter_sv = None

                # The landing page puts the starting goalie in the game header
                # under teamGameStats or directly in the team block
                tgs = team_data.get("teamGameStats", {})

                # Check for goalie info in the starting lineup
                for goalie in team_data.get("goalies", []):
                    if goalie.get("starter", False) or goalie.get("starting", False):
                        starter_id = goalie.get("playerId") or goalie.get("id")
                        break

                # If not explicitly marked, take the first goalie listed
                if not starter_id:
                    goalies = team_data.get("goalies", [])
                    if goalies:
                        starter_id = goalies[0].get("playerId") or goalies[0].get("id")

                if starter_id and starter_id in goalie_sv:
                    starter_sv = goalie_sv[starter_id]
                    starter_nm = goalie_names.get(starter_id, f"ID:{starter_id}")
                    print(f"  [NHL] Starting goalie {team_abbrev}: {starter_nm} "
                          f"(sv%={starter_sv:.3f})")
                    result[team_abbrev] = starter_sv
                else:
                    # Fall back to team season average
                    result[team_abbrev] = team_avg.get(team_abbrev, 0.910)

        except Exception as e:
            # On error, use team season average for both teams in this game
            result[away] = team_avg.get(away, 0.910)
            result[home] = team_avg.get(home, 0.910)

    # Fill any teams not yet confirmed with their season average
    for game in games:
        for team in (game.get("away_team",""), game.get("home_team","")):
            if team and team not in result:
                result[team] = team_avg.get(team, 0.910)

    if result:
        _save_cache(cache_key, result)
    return result



    """
    Fetch the confirmed lineup (scratches + healthy scratches) for a specific game.
    Uses /v1/gamecenter/{game_id}/play-by-play which includes lineup info,
    or falls back to /v1/gamecenter/{game_id}/landing for pre-game lineups.

    Returns a set of player_ids confirmed as ACTIVE (in lineup).
    Empty set means lineup not yet available — don't filter in that case.
    """
    cache_key = f"lineup_{game_id}_{team_abbrev}"
    cached = _load_cache(cache_key, max_age_minutes=30)
    if cached is not None:
        return set(cached) if cached else set()

    active_ids = set()

    # Try landing page first (pre-game and live lineups)
    data = _get(f"/gamecenter/{game_id}/landing")
    if not data:
        return set()

    # matchup → teamGameStats has skaters
    for side in ("homeTeam", "awayTeam"):
        team = data.get(side, {})
        if team.get("abbrev", "") != team_abbrev:
            continue
        for skater in team.get("skaters", []):
            active_ids.add(skater.get("playerId") or skater.get("id"))
        for scratch in team.get("scratches", []):
            # Remove scratches from active
            sid = scratch.get("playerId") or scratch.get("id")
            active_ids.discard(sid)

    if active_ids:
        print(f"[Lineup] {team_abbrev} game {game_id}: {len(active_ids)} active skaters confirmed")
        _save_cache(cache_key, list(active_ids))

    return active_ids


def get_unavailable_players(teams_and_games: list[dict]) -> set:
    """
    Return player_ids to exclude: on IR/LTIR by roster API, or absent from
    the active roster (players not listed at all are already excluded by
    fetch_rosters using the active roster endpoint).
    """
    unavailable = set()
    all_teams   = set()
    for g in teams_and_games:
        all_teams.add(g.get("away_team", ""))
        all_teams.add(g.get("home_team", ""))

    for team in all_teams:
        if not team:
            continue
        try:
            injured = get_team_injuries(team)
            unavailable.update(injured)
        except Exception as e:
            print(f"[Injuries] Could not check {team}: {e}")

    # Also check LTIR roster explicitly — some APIs have a separate injured list
    try:
        for team in all_teams:
            if not team:
                continue
            data = _get(f"/roster/{team}/current")
            # LTIR players sometimes appear in a separate 'injured' group
            for p in data.get("injured", []):
                pid = p.get("id")
                if pid:
                    unavailable.add(pid)
                    fname = p.get("firstName", {}).get("default", "")
                    lname = p.get("lastName", {}).get("default", "")
                    print(f"[Injuries] LTIR: {fname} {lname} ({team})")
    except Exception:
        pass

    print(f"[Injuries] Total excluded: {len(unavailable)} players")
    return unavailable
