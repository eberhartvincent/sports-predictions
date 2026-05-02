"""
betting_projections.py — Team-level moneyline, spread, and total goals projections

Uses aggregated player-level predictions + team season stats to project:
  - Moneyline win probability for each team
  - Spread projection (expected goal differential)
  - Total goals projection (over/under)

All projections are model-based estimates, NOT official betting lines.
"""

from math import exp, factorial
import numpy as np
import pandas as pd
from typing import Optional

from nhl_api import _get, _load_cache, _save_cache
from config import (
    CURRENT_SEASON, SEASON_TYPE,
    NHL_LEAGUE_AVG_GOALS, NHL_HOME_ADVANTAGE_GOALS,
    NHL_REGRESSION_WEIGHT, NHL_PLAYER_SIGNAL_WEIGHT,
    NHL_XG_MIN, NHL_XG_MAX,
)


# ── Team season stats ─────────────────────────────────────────────────────────

def get_team_stats() -> pd.DataFrame:
    """
    Fetch current season standings/stats for all teams.
    Returns DataFrame with goals-for, goals-against, win%, home/away splits.
    """
    cache_key = f"team_stats_{CURRENT_SEASON}"
    cached = _load_cache(cache_key, max_age_minutes=240)
    if cached:
        return pd.DataFrame(cached)

    # Try the current standings endpoint
    data = _get(f"/standings/now")
    if not data:
        data = _get(f"/standings")


    rows = []
    # The NHL API nests standings under 'standings' key
    standing_list = data.get("standings", [])

    for entry in standing_list:
        abbrev = entry.get("teamAbbrev", {})
        if isinstance(abbrev, dict):
            abbrev = abbrev.get("default", "")

        gp = entry.get("gamesPlayed", 0)
        if gp == 0:
            continue

        rows.append({
            "team":          abbrev,
            "wins":          entry.get("wins", 0),
            "losses":        entry.get("losses", 0),
            "ot_losses":     entry.get("otLosses", 0),
            "points":        entry.get("points", 0),
            "gp":            gp,
            "gf":            entry.get("goalFor", 0),
            "ga":            entry.get("goalAgainst", 0),
            "home_wins":     entry.get("homeWins", 0),
            "home_losses":   entry.get("homeLosses", 0),
            "road_wins":     entry.get("roadWins", 0),
            "road_losses":   entry.get("roadLosses", 0),
            "l10_wins":      entry.get("l10Wins", 0),
            "l10_losses":    entry.get("l10Losses", 0),
        })

    if not rows:
        print("[Betting] No rows parsed from standings — check API response structure")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["gf_pg"]        = df["gf"]         / df["gp"].replace(0, np.nan)
    df["ga_pg"]        = df["ga"]         / df["gp"].replace(0, np.nan)
    df["win_pct"]      = df["wins"]       / df["gp"].replace(0, np.nan)
    df["home_win_pct"] = df["home_wins"]  / (df["home_wins"] + df["home_losses"]).replace(0, np.nan)
    df["road_win_pct"] = df["road_wins"]  / (df["road_wins"] + df["road_losses"]).replace(0, np.nan)
    df["l10_win_pct"]  = df["l10_wins"]   / 10.0
    df = df.fillna(0)


    _save_cache(cache_key, df.to_dict("records"))
    return df


# ── Core projection engine ────────────────────────────────────────────────────

class BettingProjector:
    """
    Generates moneyline, spread, and total projections for each game.
    """

    # League averages — 2024-25 NHL season (~5.7 combined goals/game)
    LEAGUE_AVG_GOALS_PER_GAME = NHL_LEAGUE_AVG_GOALS  # per team (5.70 total)
    HOME_ADVANTAGE_GOALS      = NHL_HOME_ADVANTAGE_GOALS  # home teams score modestly more
    HOME_WIN_BOOST            = 0.03

    # Regression-to-mean weight — pulls extreme team ratings toward league avg
    # Higher = more regression (more conservative totals)
    REGRESSION_WEIGHT         = NHL_REGRESSION_WEIGHT

    def __init__(self):
        self.team_stats: pd.DataFrame = pd.DataFrame()
        self._stats_map: dict         = {}

    def load(self) -> None:
        """Load team stats from the NHL API."""
        self.team_stats = get_team_stats()
        if not self.team_stats.empty:
            self._stats_map = {
                row["team"]: row
                for _, row in self.team_stats.iterrows()
            }
        else:
            print(f"[Betting] Loaded stats for {len(self._stats_map)} teams")

    def _get_team(self, team: str) -> dict:
        """Return stats dict for a team, with safe defaults."""
        return self._stats_map.get(team, {
            "gf_pg":        NHL_LEAGUE_AVG_GOALS,
            "ga_pg":        NHL_LEAGUE_AVG_GOALS,
            "win_pct":      0.500,
            "home_win_pct": 0.530,
            "road_win_pct": 0.470,
            "l10_win_pct":  0.500,
        })

    # ── Goals projection ──────────────────────────────────────────────────────

    def project_goals(self, away_team: str, home_team: str,
                       player_preds: Optional[pd.DataFrame] = None) -> dict:
        """
        Project expected goals for each team using:
          - Team's GF/game (offensive strength)
          - Opponent's GA/game (defensive weakness)
          - Home ice advantage
          - Regression to league mean
          - Player-level goal projections (if available)
        """
        away = self._get_team(away_team)
        home = self._get_team(home_team)

        league_avg = self.LEAGUE_AVG_GOALS_PER_GAME
        r          = self.REGRESSION_WEIGHT   # regression-to-mean factor

        # Regress team ratings toward league average
        away_off = away.get("gf_pg", league_avg) * (1 - r) + league_avg * r
        away_def = away.get("ga_pg", league_avg) * (1 - r) + league_avg * r
        home_off = home.get("gf_pg", league_avg) * (1 - r) + league_avg * r
        home_def = home.get("ga_pg", league_avg) * (1 - r) + league_avg * r

        # Attack × Defence model (each team's offence vs opponent defence)
        away_xg_model = (away_off / league_avg) * (home_def / league_avg) * league_avg
        home_xg_model = ((home_off / league_avg) * (away_def / league_avg) * league_avg
                         + self.HOME_ADVANTAGE_GOALS)

        # Blend with player-level predictions if available
        if player_preds is not None and not player_preds.empty:
            away_players    = player_preds[player_preds["team"] == away_team]
            home_players    = player_preds[player_preds["team"] == home_team]

            # Use only top 8 forwards by probability — full roster sum inflates totals
            def top_player_xg(players_df):
                if players_df.empty:
                    return None
                top = players_df.nlargest(8, "goal_probability")
                raw = float(top["goal_probability"].sum())
                # Normalize: league avg team scores ~2.85/game, cap contribution
                return float(np.clip(raw, 1.0, 4.0))

            away_xg_players = top_player_xg(away_players)
            home_xg_players = top_player_xg(home_players)

            # Blend 80% team model, 20% player signal
            if away_xg_players is not None:
                away_xg = (away_xg_model * (1-NHL_PLAYER_SIGNAL_WEIGHT)) + (away_xg_players * NHL_PLAYER_SIGNAL_WEIGHT)
            else:
                away_xg = away_xg_model

            if home_xg_players is not None:
                home_xg = (home_xg_model * (1-NHL_PLAYER_SIGNAL_WEIGHT)) + (home_xg_players * NHL_PLAYER_SIGNAL_WEIGHT)
            else:
                home_xg = home_xg_model
        else:
            away_xg = away_xg_model
            home_xg = home_xg_model

        # Clamp to realistic range (NHL games rarely see <1.5 or >5 per team)
        away_xg = float(np.clip(away_xg, NHL_XG_MIN, NHL_XG_MAX))
        home_xg = float(np.clip(home_xg, NHL_XG_MIN, NHL_XG_MAX))
        total   = round(away_xg + home_xg, 2)

        return {
            "away_proj_goals":  round(away_xg, 2),
            "home_proj_goals":  round(home_xg, 2),
            "total_proj_goals": total,
        }

    # ── Win probability / moneyline ───────────────────────────────────────────

    def project_moneyline(self, away_team: str, home_team: str,
                           goals: dict) -> dict:
        """
        Convert projected goals into win probabilities using a
        Poisson-based model, then convert to American moneyline odds.
        """
        away_xg = goals["away_proj_goals"]
        home_xg = goals["home_proj_goals"]

        # Poisson win probability simulation
        away_win_prob, home_win_prob, tie_prob = _poisson_win_prob(away_xg, home_xg)

        # In NHL, ~25% of games go to OT/SO — redistribute tie probability
        # Home team wins ~54% of OT games historically
        home_win_prob += tie_prob * 0.54
        away_win_prob += tie_prob * 0.46

        # Blend with season win% for stability
        away_season = self._get_team(away_team)
        home_season = self._get_team(home_team)

        away_form = (
            away_season.get("l10_win_pct", 0.5) * 0.3 +
            away_season.get("road_win_pct", 0.47) * 0.3 +
            away_season.get("win_pct", 0.5) * 0.4
        )
        home_form = (
            home_season.get("l10_win_pct", 0.5) * 0.3 +
            home_season.get("home_win_pct", 0.53) * 0.3 +
            home_season.get("win_pct", 0.5) * 0.4
        )

        # Normalise form
        total_form    = away_form + home_form
        away_form_pct = away_form / total_form if total_form > 0 else 0.5
        home_form_pct = home_form / total_form if total_form > 0 else 0.5

        # Final blend: 65% Poisson model, 35% season form
        final_away = (away_win_prob * 0.65) + (away_form_pct * 0.35)
        final_home = (home_win_prob * 0.65) + (home_form_pct * 0.35)

        # Normalise
        total         = final_away + final_home
        final_away   /= total
        final_home   /= total

        return {
            "away_win_prob":   round(final_away, 3),
            "home_win_prob":   round(final_home, 3),
            "away_moneyline":  _prob_to_moneyline(final_away),
            "home_moneyline":  _prob_to_moneyline(final_home),
            "away_ml_display": _format_moneyline(_prob_to_moneyline(final_away)),
            "home_ml_display": _format_moneyline(_prob_to_moneyline(final_home)),
            "favourite":       home_team if final_home > final_away else away_team,
        }

    # ── Spread projection ─────────────────────────────────────────────────────

    def project_spread(self, away_team: str, home_team: str, goals: dict) -> dict:
        """
        Project the goal spread (puck line equivalent).
        Standard NHL puck line is -1.5 / +1.5.
        The favoured team (higher projected goals) gets -1.5.
        """
        away_xg = goals["away_proj_goals"]
        home_xg = goals["home_proj_goals"]
        spread  = round(home_xg - away_xg, 2)  # positive = home favoured

        if spread >= 0:
            # Home team favoured — home covers -1.5
            fav_team    = home_team
            dog_team    = away_team
            fav_xg      = home_xg
            dog_xg      = away_xg
            puck_home   = "-1.5"
            puck_away   = "+1.5"
            fav_cover   = _spread_cover_prob(home_xg, away_xg, spread_line=1.5)
            dog_cover   = round(1 - fav_cover, 3)
            home_cover  = fav_cover
            away_cover  = dog_cover
        else:
            # Away team favoured — away covers -1.5
            fav_team    = away_team
            dog_team    = home_team
            fav_xg      = away_xg
            dog_xg      = home_xg
            puck_home   = "+1.5"
            puck_away   = "-1.5"
            # Recalculate with away as the "home" in the formula
            fav_cover   = _spread_cover_prob(away_xg, home_xg, spread_line=1.5)
            dog_cover   = round(1 - fav_cover, 3)
            home_cover  = dog_cover   # home team is the underdog here
            away_cover  = fav_cover

        return {
            "proj_spread":      spread,
            "spread_display":   f"{'−' if spread >= 0 else '+'}{abs(spread):.1f}",
            "home_cover_prob":  round(home_cover, 3),
            "away_cover_prob":  round(away_cover, 3),
            "puck_line_home":   puck_home,
            "puck_line_away":   puck_away,
            "fav_team":         fav_team,
            "fav_cover_prob":   round(fav_cover, 3),
            "dog_cover_prob":   round(dog_cover, 3),
        }

    # ── Total goals (over/under) ──────────────────────────────────────────────

    def project_total(self, goals: dict) -> dict:
        """
        Project the over/under total and probability of going over
        common betting lines (5.5, 6.0, 6.5).
        """
        total   = goals["total_proj_goals"]
        away_xg = goals["away_proj_goals"]
        home_xg = goals["home_proj_goals"]

        over_5_5 = _poisson_over_prob(away_xg + home_xg, 5.5)
        over_6_0 = _poisson_over_prob(away_xg + home_xg, 6.0)
        over_6_5 = _poisson_over_prob(away_xg + home_xg, 6.5)

        lines = {5.5: over_5_5, 6.0: over_6_0, 6.5: over_6_5}

        # Pick the line with the strongest edge (furthest from 50%)
        # This avoids recommending a near-coinflip line
        def edge(prob):
            return abs(prob - 0.5)

        best_line = max(lines, key=lambda l: edge(lines[l]))
        best_over_prob = lines[best_line]
        recommendation = "OVER" if best_over_prob > 0.5 else "UNDER"
        best_prob = best_over_prob if best_over_prob > 0.5 else (1 - best_over_prob)

        return {
            "proj_total":       total,
            "over_5_5_prob":    round(over_5_5, 3),
            "over_6_0_prob":    round(over_6_0, 3),
            "over_6_5_prob":    round(over_6_5, 3),
            "best_ou_line":     best_line,
            "best_ou_prob":     round(best_prob, 3),
            "recommendation":   recommendation,
        }

    # ── Full game projection ──────────────────────────────────────────────────

    def project_game(self, away_team: str, home_team: str,
                      player_preds: Optional[pd.DataFrame] = None) -> dict:
        """
        Run all three projections for a single game.
        Returns a combined dict with all projection data.
        """
        goals    = self.project_goals(away_team, home_team, player_preds)
        ml       = self.project_moneyline(away_team, home_team, goals)
        spread   = self.project_spread(away_team, home_team, goals)
        total    = self.project_total(goals)

        return {
            "away_team": away_team,
            "home_team": home_team,
            **goals,
            **ml,
            **spread,
            **total,
        }

    def project_all_games(self, games: list,
                           player_preds: Optional[pd.DataFrame] = None) -> list[dict]:
        """Project all games in today's schedule."""
        results = []
        for game in games:
            try:
                proj = self.project_game(
                    game["away_team"], game["home_team"], player_preds
                )
                proj["game_id"]        = game.get("game_id")
                proj["start_time_utc"] = game.get("start_time_utc", "")
                results.append(proj)
            except Exception as e:
                print(f"[Betting] Error projecting {game['away_team']} @ {game['home_team']}: {e}")
        return results


# ── Math helpers ──────────────────────────────────────────────────────────────

def _poisson_win_prob(lambda_a: float, lambda_h: float,
                       max_goals: int = 10) -> tuple[float, float, float]:
    """
    Compute win/draw/loss probabilities using independent Poisson distributions.
    Returns (away_win, home_win, tie).
    """
    
    def poisson_pmf(k, lam):
        return (lam ** k) * exp(-lam) / factorial(k)

    away_win = home_win = tie = 0.0
    for a in range(max_goals + 1):
        for h in range(max_goals + 1):
            p = poisson_pmf(a, lambda_a) * poisson_pmf(h, lambda_h)
            if a > h:
                away_win += p
            elif h > a:
                home_win += p
            else:
                tie += p

    return away_win, home_win, tie


def _poisson_over_prob(total_lambda: float, line: float) -> float:
    """Probability that total goals > line using Poisson distribution."""
    
    under_prob = 0.0
    k = 0
    while k <= int(line):
        under_prob += (total_lambda ** k) * exp(-total_lambda) / factorial(k)
        k += 1

    # Handle half-point lines
    if line == int(line):
        # Push at exactly the line — split evenly
        exact = (total_lambda ** int(line)) * exp(-total_lambda) / factorial(int(line))
        under_prob -= exact / 2

    return float(np.clip(1 - under_prob, 0.01, 0.99))


def _spread_cover_prob(fav_xg: float, dog_xg: float,
                        spread_line: float = 1.5) -> float:
    """
    Probability that the favoured team wins by more than spread_line goals.
    """
    
    def poisson_pmf(k, lam):
        return (lam ** k) * exp(-lam) / factorial(k)

    cover_prob = 0.0
    for f in range(15):
        for d in range(15):
            if (f - d) > spread_line:
                cover_prob += poisson_pmf(f, fav_xg) * poisson_pmf(d, dog_xg)

    return float(np.clip(cover_prob, 0.01, 0.99))


def _prob_to_moneyline(prob: float) -> int:
    """Convert win probability to American moneyline odds."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return int(-(prob / (1 - prob)) * 100)
    else:
        return int(((1 - prob) / prob) * 100)


def _format_moneyline(ml: int) -> str:
    """Format moneyline as string with sign."""
    return f"+{ml}" if ml > 0 else str(ml)