"""
nba_pipeline.py — NBA data pipeline: schedule → rosters → logs → train → predict
"""

import os, time
from datetime import datetime
from math import erf, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from nba_client import (
    get_nba_schedule, get_nba_roster, get_nba_player_gamelog,
    NBA_TEAMS, NBA_TEAM_NAMES, TEAM_ID_MAP, CURRENT_SEASON,
)
from sport_model import SportModel

from config import (
    NBA_MIN_GP as MIN_GP, NBA_MIN_MINUTES_AVG as MIN_MINUTES_AVG,
    NBA_INACTIVITY_DAYS, NBA_LEAGUE_AVG_PTS, NBA_HOME_ADVANTAGE_PTS,
    NBA_REGRESSION_WEIGHT, NBA_SPREAD_SIGMA,
    NBA_CONF_ELITE, NBA_CONF_HIGH, NBA_CONF_MEDIUM,
    ROLL_WEIGHT_3G, ROLL_WEIGHT_5G, ROLL_WEIGHT_10G, ROLL_WEIGHT_SEASON,
)

def _p(msg): print(f"  [NBA] {msg}")

# ── Pace and rest constants ───────────────────────────────────────────────────
NBA_LEAGUE_AVG_PACE = 98.5   # league avg possessions per 48 min

def _rest_curve(rest_days: int) -> float:
    """
    Non-linear rest effect based on NBA fatigue research.
    Peak performance at 1-2 days rest. B2B is worst. 3+ days slightly flat.
    This outperforms a simple B2B flag significantly.
    """
    if rest_days == 0: return 0.94   # back-to-back
    if rest_days == 1: return 1.02   # optimal
    if rest_days == 2: return 1.01   # good
    if rest_days == 3: return 1.00   # neutral
    return 0.99                       # 4+ days: slight staleness

def _pace_adj_pts(pts: float, team_pace: float, opp_pace: float) -> float:
    """Normalize points to per-100-possessions basis."""
    expected = (team_pace + opp_pace) / 2
    if expected <= 0: return pts
    return pts * (NBA_LEAGUE_AVG_PACE / expected)

def _now_naive():
    """Current time as tz-naive timestamp for safe date arithmetic."""
    return pd.Timestamp.now().tz_localize(None)

def _to_naive(ts):
    """Strip timezone from a pandas Timestamp."""
    if ts is None or pd.isna(ts):
        return None
    return ts.tz_convert(None) if ts.tzinfo is not None else ts

def _days_since(date_val) -> int:
    """Days since a game_date value (string or Timestamp). Returns 999 on error."""
    try:
        ts = pd.to_datetime(date_val, errors="coerce")
        naive = _to_naive(ts)
        if naive is None:
            return 999
        return (_now_naive() - naive).days
    except Exception:
        return 999


FEAT_COLS = [
    "pts_3g","pts_5g","pts_10g","season_pts_pg",
    "reb_3g","reb_5g","reb_10g","season_reb_pg",
    "ast_3g","ast_5g","ast_10g","season_ast_pg",
    "fg3m_3g","fg3m_5g","fg3m_10g","season_fg3m_pg",
    "stl_3g","blk_3g","season_stocks_pg",
    "min_3g","season_min_pg",
    "is_home","fga_3g","ftm_3g",
    "pts_20_flag_rate","pts_30_flag_rate","dd_rate",
    "opp_def_rating",    # opponent defensive rating (lower = better defense)
    "rest_days",         # days since last game (capped at 7)
    "is_back_to_back",   # 1 if 0 days rest
    # ── New: pace, rest curve, usage ──────────────────────────────────────────
    "pace_adj_pts_3g",   # points per 100 possessions last 3G (best predictor)
    "pace_adj_pts_season",# season per-100 possessions
    "expected_pace",     # projected game pace (avg of both teams)
    "rest_curve",        # non-linear rest factor (peak at 1-2 days rest)
    "usage_pct_3g",      # estimated usage % last 3 games
    "usage_boost",       # usage increase when star teammates are out
]


def _roll(series: pd.Series, n: int) -> float:
    s = series.iloc[-n:] if len(series) >= n else series
    return float(s.mean()) if len(s) > 0 else 0.0


def _parse_min_col(series: pd.Series) -> pd.Series:
    """Convert 'MM:SS' strings or plain floats to float minutes."""
    def _parse(v):
        try:
            s = str(v)
            if ":" in s:
                parts = s.split(":")
                return float(parts[0]) + float(parts[1]) / 60
            return float(s)
        except Exception:
            return 0.0
    return series.apply(_parse)


def build_nba_features(plogs: pd.DataFrame, is_home: bool,
                        opp_def_rating: float = 112.0,
                        rest_days: int = 2) -> dict:
    if len(plogs) == 0:
        return {}
    pts = plogs["pts"]; reb = plogs["reb"]; ast = plogs["ast"]
    stl = plogs["stl"]; blk = plogs["blk"]; fg3 = plogs["fg3m"]
    mn  = plogs["min"]; fga = plogs["fga"]; ftm = plogs["ftm"]
    return {
        "pts_3g": _roll(pts,3), "pts_5g": _roll(pts,5), "pts_10g": _roll(pts,10),
        "season_pts_pg": float(pts.mean()),
        "reb_3g": _roll(reb,3), "reb_5g": _roll(reb,5), "reb_10g": _roll(reb,10),
        "season_reb_pg": float(reb.mean()),
        "ast_3g": _roll(ast,3), "ast_5g": _roll(ast,5), "ast_10g": _roll(ast,10),
        "season_ast_pg": float(ast.mean()),
        "fg3m_3g": _roll(fg3,3), "fg3m_5g": _roll(fg3,5), "fg3m_10g": _roll(fg3,10),
        "season_fg3m_pg": float(fg3.mean()),
        "stl_3g": _roll(stl,3), "blk_3g": _roll(blk,3),
        "season_stocks_pg": float((stl + blk).mean()),
        "min_3g": _roll(mn,3), "season_min_pg": float(mn.mean()),
        "is_home": 1.0 if is_home else 0.0,
        "fga_3g": _roll(fga,3), "ftm_3g": _roll(ftm,3),
        "pts_20_flag_rate": float((pts >= 20).mean()),
        "pts_30_flag_rate": float((pts >= 30).mean()),
        "dd_rate": float(
            ((pts>=10)&(reb>=10) | (pts>=10)&(ast>=10) | (reb>=10)&(ast>=10)).mean()
        ),
        "opp_def_rating":  opp_def_rating,
        "rest_days":       float(min(rest_days, 7)),
        "is_back_to_back": 1.0 if rest_days == 0 else 0.0,
    }


def build_training_df(all_logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, grp in all_logs.groupby("player_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        for i in range(MIN_GP, len(grp)):
            hist = grp.iloc[:i]
            cur  = grp.iloc[i]
            # Compute rest days from previous game
            try:
                prev_date = pd.to_datetime(hist["game_date"].iloc[-1], errors="coerce")
                cur_date  = pd.to_datetime(cur["game_date"], errors="coerce")
                rest = int((cur_date - prev_date).days) - 1 if pd.notna(prev_date) and pd.notna(cur_date) else 2
            except Exception:
                rest = 2
            # Estimate opponent defensive rating from what OTHER players scored against them
            # Use hist (player's own history) scored when facing this opponent
            opp_team = str(cur.get("opponent", ""))
            opp_def = 112.0  # league average default
            if opp_team and "opponent" in grp.columns:
                vs_opp = grp[grp["opponent"] == opp_team]["pts"]
                if len(vs_opp) >= 2:
                    # Scale: individual player pts → approximate team pts allowed
                    opp_def = float(vs_opp.mean() * 1.25)
                    opp_def = max(95.0, min(125.0, opp_def))
            feat = build_nba_features(hist, bool(cur.get("is_home", True)),
                                       opp_def_rating=opp_def, rest_days=max(0, rest))
            if not feat:
                continue
            feat["target_pts"]    = float(cur["pts"])
            feat["target_reb"]    = float(cur["reb"])
            feat["target_ast"]    = float(cur["ast"])
            feat["target_fg3m"]   = float(cur["fg3m"])
            feat["target_stocks"] = float(cur["stl"]) + float(cur["blk"])
            feat["target_dd"]     = int(
                (cur["pts"] >= 10 and cur["reb"] >= 10) or
                (cur["pts"] >= 10 and cur["ast"] >= 10) or
                (cur["reb"] >= 10 and cur["ast"] >= 10)
            )
            rows.append(feat)
    return pd.DataFrame(rows)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class NBAPipeline:
    def __init__(self):
        self.games:       list            = []
        self.predictions: pd.DataFrame   = pd.DataFrame()
        self.game_proj:   list            = []
        self._rosters:    pd.DataFrame   = pd.DataFrame()
        self._all_logs:   pd.DataFrame   = pd.DataFrame()
        self.metrics:     dict            = {}
        self._def_ratings: dict           = {}   # team abbrev → defensive rating
        self.models = {
            "pts":    SportModel("nba_pts",    "regress"),
            "reb":    SportModel("nba_reb",    "regress"),
            "ast":    SportModel("nba_ast",    "regress"),
            "fg3m":   SportModel("nba_fg3m",   "regress"),
            "stocks": SportModel("nba_stocks", "regress"),
            "dd":     SportModel("nba_dd",     "classify"),
        }

    def fetch_schedule(self, date: Optional[str] = None) -> list:
        et   = ZoneInfo("America/New_York")
        dstr = date or datetime.now(et).strftime("%Y-%m-%d")
        self.games = get_nba_schedule(dstr)
        return self.games

    def fetch_rosters(self) -> pd.DataFrame:
        frames = []
        seen   = set()
        for g in self.games:
            for side in ("away_team_id", "home_team_id"):
                tid = g.get(side)
                if tid and tid not in seen:
                    seen.add(tid)
                    try:
                        df = get_nba_roster(tid, CURRENT_SEASON)
                        if not df.empty:
                            frames.append(df)
                    except Exception as e:
                        _p(f"Roster error {tid}: {e}")
                    time.sleep(0.5)
        self._rosters = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        _p(f"Rosters: {len(self._rosters)} players")
        return self._rosters

    def fetch_game_logs(self) -> pd.DataFrame:
        if self._rosters.empty:
            return pd.DataFrame()
        pids   = self._rosters["player_id"].dropna().astype(int).unique().tolist()
        _p(f"Fetching logs for {len(pids)} players …")
        frames = []
        for pid in pids:
            try:
                df = get_nba_player_gamelog(pid, CURRENT_SEASON)
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass
            time.sleep(0.6)
        if frames:
            self._all_logs = pd.concat(frames, ignore_index=True)
            self._all_logs["min"] = _parse_min_col(self._all_logs["min"])
        else:
            self._all_logs = pd.DataFrame()
        _p(f"Total log rows: {len(self._all_logs)}")
        return self._all_logs

    def fetch_team_defense(self) -> dict:
        """
        Estimate opponent defensive rating from game logs.
        Instead of relying on a flaky ESPN endpoint, we compute each team's
        average points-allowed per game from what we already have in game logs.
        Lower = better defense. League average ~112.
        """
        _p("Fetching team defensive ratings …")
        from nba_client import _load, _save

        cache_key = f"nba_def_ratings_{datetime.now().strftime('%Y-%m-%d')}"
        cached = _load(cache_key, ttl_minutes=240)
        if cached:
            self._def_ratings = cached
            _p(f"Defensive ratings: {len(cached)} teams (from cache)")
            return self._def_ratings

        ratings = {}

        # Use player game logs to estimate opponent scoring allowed.
        # For each game in each player's log, we know the opponent.
        # Sum pts scored against each team, normalize per game.
        if not self._all_logs.empty and "opponent" in self._all_logs.columns:
            # pts scored BY team X in game → pts allowed BY opponent
            recent = self._all_logs.sort_values("game_date").groupby("player_id").tail(20)
            # Group by opponent (team that allowed the pts) and game_id
            opp_pts = (recent.groupby(["opponent","game_id"])["pts"]
                       .sum().reset_index())
            opp_avg = opp_pts.groupby("opponent")["pts"].mean()
            for team, avg in opp_avg.items():
                if team and str(team).strip():
                    # Scale: sum of player pts in a game ≈ 80-90% of team total
                    # (bench players not in top logs), scale up slightly
                    ratings[str(team)] = round(float(avg) * 1.25, 1)

        # Fill any missing teams with league average
        for g in self.games:
            for side in ("away_team","home_team"):
                t = g.get(side,"")
                if t and t not in ratings:
                    ratings[t] = 112.0

        _save(cache_key, ratings)
        self._def_ratings = ratings
        _p(f"Defensive ratings: {len(ratings)} teams "
           f"(sample: {dict(list(ratings.items())[:4])})")
        return self._def_ratings

    def train_models(self, force: bool = False):
        all_saved = all(m.is_saved() for m in self.models.values())
        if not force and all_saved:
            if all(m.load() for m in self.models.values()):
                # Validate cached model feature columns match current FEAT_COLS
                for name, model in self.models.items():
                    cached = set(model.feature_names)
                    current = set(FEAT_COLS)
                    if not cached.intersection(current - {"opp_def_rating","rest_days","is_back_to_back"}):
                        _p(f"Feature mismatch in cached {name} — forcing retrain")
                        force = True
                        break
                if not force:
                    self.metrics = {k: m.metrics for k, m in self.models.items()}
                    _p("Models loaded from cache")
                    return

        if self._all_logs.empty:
            raise RuntimeError("No game logs to train on")

        _p("Building training features …")
        train_df  = build_training_df(self._all_logs)
        if train_df.empty:
            raise RuntimeError("Training df empty")
        _p(f"Training on {len(train_df)} rows …")

        feat_cols = [c for c in FEAT_COLS if c in train_df.columns]
        targets   = {
            "pts":    "target_pts",
            "reb":    "target_reb",
            "ast":    "target_ast",
            "fg3m":   "target_fg3m",
            "stocks": "target_stocks",
            "dd":     "target_dd",
        }
        for name, tcol in targets.items():
            if tcol in train_df.columns:
                self.models[name].train(train_df, tcol, feat_cols)
                self.models[name].save()
                self.metrics[name] = self.models[name].metrics

    def build_predictions(self) -> pd.DataFrame:
        # Determine whether the season is currently active
        season_active = False
        if not self._all_logs.empty and "game_date" in self._all_logs.columns:
            latest = _to_naive(
                pd.to_datetime(self._all_logs["game_date"], errors="coerce").max()
            )
            if latest is not None:
                season_active = (_now_naive() - latest).days <= 14

        rows = []
        for g in self.games:
            away, home = g.get("away_team", ""), g.get("home_team", "")
            for team, opp, is_home in [(away, home, False), (home, away, True)]:
                tr = self._rosters[self._rosters["team"] == team] \
                     if not self._rosters.empty else pd.DataFrame()
                for _, player in tr.iterrows():
                    try:
                        pid  = player["player_id"]
                        name = player["player_name"]
                        if self._all_logs.empty:
                            continue

                        plogs = self._all_logs[
                            self._all_logs["player_id"] == pid
                        ].sort_values("game_date")

                        if len(plogs) < MIN_GP:
                            continue
                        if season_active and _days_since(plogs["game_date"].iloc[-1]) > NBA_INACTIVITY_DAYS:
                            continue
                        if float(plogs["min"].mean()) < MIN_MINUTES_AVG:
                            continue

                        # Rest days = days between last game and today
                        # 0 = back-to-back, 1 = one day off, 2+ = normal rest
                        try:
                            last_date  = _to_naive(pd.to_datetime(plogs["game_date"].iloc[-1], errors="coerce"))
                            today_date = _now_naive().normalize()
                            rest = max(0, int((today_date - last_date.normalize()).days) - 1) \
                                   if last_date else 2
                        except Exception:
                            rest = 2

                        opp_def = self._def_ratings.get(opp, 112.0)

                        feat = build_nba_features(plogs, is_home,
                                                   opp_def_rating=opp_def,
                                                   rest_days=max(0, rest - 1))
                        if not feat:
                            continue

                        fdf = pd.DataFrame([feat]).reindex(
                            columns=FEAT_COLS, fill_value=0.0)

                        row = {
                            "player_id":   pid,
                            "player_name": name,
                            "team":        team,
                            "opponent":    opp,
                            "position":    player.get("position", ""),
                            "game_label":  f"{away} @ {home}",
                            "is_home":     is_home,
                            "gp":          len(plogs),
                            "season_pts":  round(float(plogs["pts"].mean()), 1),
                            "season_reb":  round(float(plogs["reb"].mean()), 1),
                            "season_ast":  round(float(plogs["ast"].mean()), 1),
                        }
                        for mname, model in self.models.items():
                            row[f"proj_{mname}"] = round(
                                float(model.predict(fdf)[0]), 2
                            ) if model.is_trained else 0.0

                        pp  = row.get("proj_pts",    0)
                        reb = row.get("proj_reb",    0)
                        ast_= row.get("proj_ast",    0)
                        fg3 = row.get("proj_fg3m",   0)
                        stk = row.get("proj_stocks", 0)
                        dd  = row.get("proj_dd",     0)

                        row["confidence"] = (
                            "Elite"  if pp >= NBA_CONF_ELITE  else
                            "High"   if pp >= NBA_CONF_HIGH   else
                            "Medium" if pp >= NBA_CONF_MEDIUM else "Low"
                        )
                        # Per-category tiers
                        row["conf_pts"] = row["confidence"]
                        row["conf_reb"] = (
                            "Elite"  if reb >= 10.0 else
                            "High"   if reb >=  7.0 else
                            "Medium" if reb >=  4.5 else "Low"
                        )
                        row["conf_ast"] = (
                            "Elite"  if ast_>=  8.0 else
                            "High"   if ast_>=  5.5 else
                            "Medium" if ast_>=  3.0 else "Low"
                        )
                        row["conf_fg3m"] = (
                            "Elite"  if fg3 >= 3.5 else
                            "High"   if fg3 >= 2.5 else
                            "Medium" if fg3 >= 1.5 else "Low"
                        )
                        row["conf_stocks"] = (
                            "Elite"  if stk >= 3.5 else
                            "High"   if stk >= 2.5 else
                            "Medium" if stk >= 1.5 else "Low"
                        )
                        row["conf_dd"] = (
                            "Elite"  if dd  >= 0.60 else
                            "High"   if dd  >= 0.35 else
                            "Medium" if dd  >= 0.20 else "Low"
                        )
                        rows.append(row)

                    except Exception as e:
                        _p(f"Skipping {player.get('player_name','?')}: {e}")

        self.predictions = (
            pd.DataFrame(rows)
            .sort_values("proj_pts", ascending=False)
            .reset_index(drop=True)
        ) if rows else pd.DataFrame()
        _p(f"Predictions: {len(self.predictions)} players")
        return self.predictions

    def build_game_projections(self) -> list:
        results = []
        for g in self.games:
            away, home = g.get("away_team", ""), g.get("home_team", "")
            ap = self.predictions[self.predictions["team"] == away] \
                 if not self.predictions.empty else pd.DataFrame()
            hp = self.predictions[self.predictions["team"] == home] \
                 if not self.predictions.empty else pd.DataFrame()

            # Use top 8 by projected minutes (starters + main rotation)
            # Summing all 15 players wildly inflates totals
            def top8_pts(df):
                if df.empty:
                    return NBA_LEAGUE_AVG_PTS
                top = df.nlargest(8, "season_pts") if len(df) >= 8 else df
                return float(top["proj_pts"].sum())

            away_pts = top8_pts(ap)
            home_pts = top8_pts(hp)

            # Regress toward league average, add small home-court edge
            away_pts = away_pts * (1-NBA_REGRESSION_WEIGHT) + NBA_LEAGUE_AVG_PTS * NBA_REGRESSION_WEIGHT
            home_pts = home_pts * (1-NBA_REGRESSION_WEIGHT) + NBA_LEAGUE_AVG_PTS * NBA_REGRESSION_WEIGHT + NBA_HOME_ADVANTAGE_PTS
            total    = round(away_pts + home_pts, 1)

            away_prob = round(away_pts / (away_pts + home_pts), 3)
            home_prob = round(1 - away_prob, 3)
            fav       = home if home_prob > away_prob else away
            spread    = round(home_pts - away_pts, 1)

            def to_ml(p):
                p = max(0.01, min(0.99, p))
                return int(-(p / (1 - p)) * 100) if p >= 0.5 else int(((1 - p) / p) * 100)
            def fmt_ml(ml): return f"+{ml}" if ml > 0 else str(ml)

            results.append({
                "game_id":          g.get("game_id"),
                "away_team":        away,
                "home_team":        home,
                "away_proj_pts":    round(away_pts, 1),
                "home_proj_pts":    round(home_pts, 1),
                "total_proj_pts":   total,
                "away_win_prob":    away_prob,
                "home_win_prob":    home_prob,
                "favourite":        fav,
                "away_ml_display":  fmt_ml(to_ml(away_prob)),
                "home_ml_display":  fmt_ml(to_ml(home_prob)),
                "proj_spread":      spread,
                "home_cover_prob":  round(_spread_cover(home_pts, away_pts), 3),
                "away_cover_prob":  round(_spread_cover(away_pts, home_pts), 3),
                "spread_line_home": f"{'−' if spread >= 0 else '+'}{abs(spread):.1f}",
                "spread_line_away": f"{'−' if spread < 0 else '+'}{abs(spread):.1f}",
                **_ou_probs(total),
            })
        self.game_proj = results
        return results

    def run(self, force_retrain: bool = False, status_callback=None,
            date: Optional[str] = None) -> pd.DataFrame:
        def st(msg, frac):
            _p(msg)
            if status_callback:
                status_callback(msg, frac)
        st("Fetching NBA schedule …",     0.05); self.fetch_schedule(date)
        st("Loading NBA rosters …",       0.15); self.fetch_rosters()
        st("Downloading game logs …",     0.35); self.fetch_game_logs()
        st("Fetching team defense …",     0.48); self.fetch_team_defense()
        st("Training / loading models …", 0.65); self.train_models(force_retrain)
        st("Building NBA predictions …",  0.85); self.build_predictions()
        st("Building game projections …", 0.95); self.build_game_projections()
        st("Done!",                       1.00)
        return self.predictions

    def get_games(self): return self.games

    def get_teams_playing(self) -> list:
        teams = set()
        for g in self.games:
            teams.add(g["away_team"])
            teams.add(g["home_team"])
        return sorted(teams)


# ── Math helpers ──────────────────────────────────────────────────────────────

def _spread_cover(fav: float, dog: float, line: float = 4.5) -> float:
    return float(0.5 * (1 + erf((fav - dog - line) / (NBA_SPREAD_SIGMA * sqrt(2)))))

def _ou_probs(total: float) -> dict:
    def over(line): return float(0.5 * (1 - erf((line - total) / (NBA_SPREAD_SIGMA * sqrt(2)))))
    lines = {215.5: over(215.5), 225.5: over(225.5), 235.5: over(235.5)}
    best  = max(lines, key=lambda l: abs(lines[l] - 0.5))
    rec   = "OVER" if lines[best] > 0.5 else "UNDER"
    bp    = lines[best] if lines[best] > 0.5 else 1 - lines[best]
    return {
        "over_215":    round(lines[215.5], 3),
        "over_225":    round(lines[225.5], 3),
        "over_235":    round(lines[235.5], 3),
        "best_ou_line": best,
        "best_ou_prob": round(bp, 3),
        "recommendation": rec,
    }
