"""
nhl_pipeline.py — NHL prediction pipeline

Steps:
  1. Fetch schedule
  2. Fetch rosters
  3. Fetch game logs
  4. Fetch NST advanced stats (best-effort)
  5. Fetch goalie quality
  6. Filter unavailable players (injury/inactivity)
  7. Train or load model
  8. Build predictions
  9. Build betting projections
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CURRENT_SEASON, CACHE_DIR, NHL_TEAMS, MIN_GP,
    NHL_INACTIVITY_DAYS, ROLL_WEIGHT_3G, ROLL_WEIGHT_5G,
    ROLL_WEIGHT_10G, ROLL_WEIGHT_SEASON,
)
from nhl_api import (
    get_all_rosters, fetch_all_game_logs,
    get_team_goalie_quality, get_confirmed_starting_goalies,
    get_unavailable_players,
)
from nst_scraper import get_all_situations_stats
from feature_engineering import build_training_features, build_prediction_row
from model_trainer import GoalscorerModel
from betting_projections import BettingProjector

os.makedirs(CACHE_DIR, exist_ok=True)


def _p(msg): print(f"  {msg}")


def _days_since(date_val) -> int:
    try:
        ts = pd.to_datetime(date_val, errors="coerce")
        if pd.isna(ts):
            return 999
        ts_naive = ts.tz_convert(None) if ts.tzinfo is not None else ts
        return (pd.Timestamp.now() - ts_naive).days
    except Exception:
        return 999


def _weighted_roll(series: pd.Series) -> float:
    """0.40/0.30/0.20/0.10 weighted blend of last 3/5/10 and season avg."""
    def _avg(n): return float(series.iloc[-n:].mean() if len(series) >= n else series.mean())
    return round(
        _avg(3) * ROLL_WEIGHT_3G + _avg(5) * ROLL_WEIGHT_5G +
        _avg(10) * ROLL_WEIGHT_10G + float(series.mean()) * ROLL_WEIGHT_SEASON, 2
    )


class NHLPipeline:
    """
    Stateful pipeline — call run() for a full refresh.
    After run(): .todays_games, .predictions, .game_projections, .model_metrics
    """

    def __init__(self):
        self.todays_games:     list           = []
        self.predictions:      pd.DataFrame   = pd.DataFrame()
        self.game_projections: list           = []
        self.model:            GoalscorerModel = GoalscorerModel()
        self.model_metrics:    dict           = {}
        self._all_game_logs:   pd.DataFrame   = pd.DataFrame()
        self._nst_df:          pd.DataFrame   = pd.DataFrame()
        self._goalie_map:      dict           = {}
        self._rosters:         pd.DataFrame   = pd.DataFrame()
        self._projector:       BettingProjector = BettingProjector()
        self._unavailable_ids: set            = set()

    # ── Steps ─────────────────────────────────────────────────────────────────

    def fetch_schedule(self, date: Optional[str] = None) -> list:
        from nhl_api import get_schedule_for_date
        target = date or datetime.now().strftime("%Y-%m-%d")
        games  = get_schedule_for_date(target)
        self.todays_games = [g for g in games if str(g.get("date",""))[:10] == target] or games
        _p(f"Schedule: {len(self.todays_games)} game(s) on {target}")
        return self.todays_games

    def fetch_rosters(self) -> pd.DataFrame:
        teams = {g["away_team"] for g in self.todays_games} | \
                {g["home_team"] for g in self.todays_games}
        self._rosters = get_all_rosters(list(teams))
        _p(f"Rosters: {len(self._rosters)} players across {len(teams)} teams")
        return self._rosters

    def fetch_game_logs(self) -> pd.DataFrame:
        if self._rosters.empty:
            return pd.DataFrame()
        pids = self._rosters["player_id"].dropna().astype(int).unique().tolist()
        _p(f"Fetching game logs for {len(pids)} players …")
        self._all_game_logs = fetch_all_game_logs(pids)
        _p(f"Game log rows: {len(self._all_game_logs)}")
        return self._all_game_logs

    def fetch_nst_stats(self) -> pd.DataFrame:
        try:
            self._nst_df = get_all_situations_stats(CURRENT_SEASON)
            _p(f"NST advanced stats: {len(self._nst_df)} rows")
        except Exception as e:
            _p(f"NST unavailable ({e}) — continuing without advanced stats")
            self._nst_df = pd.DataFrame()
        return self._nst_df

    def fetch_goalie_quality(self) -> dict:
        """
        Fetch the confirmed starting goalie for each game.
        Falls back to team season average if starter not yet announced.
        Short TTL (30 min) so late scratches are caught.
        """
        try:
            # Try confirmed starters first — most accurate signal
            self._goalie_map = get_confirmed_starting_goalies(self.todays_games)
            confirmed = sum(1 for v in self._goalie_map.values() if v != 0.910)
            _p(f"Goalie quality: {len(self._goalie_map)} teams "
               f"({confirmed} confirmed starters, rest use team average)")
        except Exception as e:
            _p(f"Starting goalie fetch failed ({e}) — using team averages")
            try:
                self._goalie_map = get_team_goalie_quality()
            except Exception:
                self._goalie_map = {}
        return self._goalie_map

    def fetch_unavailable_players(self) -> set:
        try:
            self._unavailable_ids = get_unavailable_players(self.todays_games)
            _p(f"Unavailable players: {len(self._unavailable_ids)}")
        except Exception as e:
            _p(f"Injury check failed ({e}) — skipping filter")
            self._unavailable_ids = set()
        return self._unavailable_ids

    def train_model(self, force_retrain: bool = False) -> GoalscorerModel:
        if not force_retrain and self.model.is_saved():
            if self.model.load():
                _p("Model loaded from cache")
                return self.model

        if self._all_game_logs.empty:
            raise RuntimeError("Game logs empty — cannot train")

        feature_df = build_training_features(
            self._all_game_logs,
            nst_df=self._nst_df if not self._nst_df.empty else None,
            goalie_map=self._goalie_map or None,
        )

        # Date-based train/test split — prevents temporal data leakage.
        # Train on oldest 80% of games, report AUC on most recent 20%.
        if "game_date" in feature_df.columns and len(feature_df) > 200:
            cutoff = pd.to_datetime(
                feature_df["game_date"], errors="coerce"
            ).quantile(0.80)
            train_df = feature_df[pd.to_datetime(feature_df["game_date"], errors="coerce") <= cutoff]
            _p(f"Train/test split at {cutoff.date()} — "
               f"{len(train_df)} train rows, {len(feature_df)-len(train_df)} test rows")
        else:
            train_df = feature_df

        _p(f"Training on {train_df.shape[0]} rows, {train_df.shape[1]} features …")
        self.model_metrics = self.model.train(train_df, verbose=True)
        self.model.save()
        return self.model

    def build_predictions(self) -> pd.DataFrame:
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before predictions")

        # Normalise player_id types for reliable joins
        if not self._all_game_logs.empty:
            self._all_game_logs["player_id"] = pd.to_numeric(
                self._all_game_logs["player_id"], errors="coerce").astype("Int64")
            self._rosters["player_id"] = pd.to_numeric(
                self._rosters["player_id"], errors="coerce").astype("Int64")

        # NST lookup by lowercase name
        nst_lookup = {}
        if not self._nst_df.empty and "player_name" in self._nst_df.columns:
            nst_lookup = {str(r["player_name"]).lower().strip(): r
                          for _, r in self._nst_df.iterrows()}

        # Pre-compute TOI ranks per team from recent logs (last 5 games)
        toi_ranks = {}
        if not self._all_game_logs.empty and "toi_seconds" in self._all_game_logs.columns:
            recent = self._all_game_logs.sort_values("game_date").groupby("player_id").tail(5)
            avg_toi = recent.groupby("player_id")["toi_seconds"].mean()
            team_map = self._rosters.set_index("player_id")["team"].to_dict() \
                       if not self._rosters.empty else {}
            toi_df   = avg_toi.reset_index()
            toi_df["team"] = toi_df["player_id"].map(team_map)
            toi_df["toi_rank"] = toi_df.groupby("team")["toi_seconds"].rank(
                ascending=False, method="dense")
            toi_ranks = toi_df.set_index("player_id")["toi_rank"].to_dict()

        rows = []
        for game in self.todays_games:
            away, home = game["away_team"], game["home_team"]
            for team, opponent, is_home in [(away, home, False), (home, away, True)]:
                for _, player in self._rosters[self._rosters["team"] == team].iterrows():
                    pid  = player["player_id"]
                    name = player["player_name"]
                    pos  = str(player.get("position", "F") or "F").upper()

                    if pos in ("G", "GK"):
                        continue

                    try:
                        pid_int = int(pid) if pd.notna(pid) else None
                    except (ValueError, TypeError):
                        pid_int = None
                    if pid_int and pid_int in self._unavailable_ids:
                        continue

                    plogs = (self._all_game_logs[self._all_game_logs["player_id"] == pid]
                             .sort_values("game_date")
                             if not self._all_game_logs.empty else pd.DataFrame())

                    if len(plogs) < MIN_GP:
                        continue
                    if _days_since(plogs["game_date"].iloc[-1]) > NHL_INACTIVITY_DAYS:
                        continue

                    feat = build_prediction_row(
                        player_id=pid_int or 0,
                        player_name=name,
                        team=team,
                        opponent=opponent,
                        is_home=is_home,
                        recent_logs=plogs,
                        nst_row=nst_lookup.get(name.lower().strip()),
                        goalie_map=self._goalie_map,
                        feature_cols=self.model.feature_names,
                    )

                    # Use pre-computed TOI rank
                    feat["toi_rank"] = toi_ranks.get(pid, 0)

                    feat["game_label"]        = f"{away} @ {home}"
                    feat["away_team"]         = away
                    feat["home_team"]         = home
                    feat["position"]          = pos
                    feat["gp_season"]         = len(plogs)
                    feat["season_goals"]      = int(plogs["goals"].sum())
                    feat["season_assists"]    = int(plogs["assists"].sum()) if "assists" in plogs.columns else 0
                    feat["season_shots"]      = int(plogs["shots"].sum())
                    feat["projected_sog"]     = round(
                        feat.get("rolling_3g_shots", 0) * ROLL_WEIGHT_3G +
                        feat.get("rolling_5g_shots", 0) * ROLL_WEIGHT_5G +
                        feat.get("rolling_10g_shots", 0) * ROLL_WEIGHT_10G +
                        feat.get("season_shots_pg", 0) * ROLL_WEIGHT_SEASON, 1)
                    feat["projected_points"]  = _weighted_roll(plogs["points"])
                    feat["projected_assists"] = _weighted_roll(plogs["assists"]) \
                                                if "assists" in plogs.columns else 0.0
                    rows.append(feat)

        if not rows:
            _p("No prediction rows built")
            self.predictions = pd.DataFrame()
            return self.predictions

        pred_df = pd.DataFrame(rows)
        self.predictions = self.model.predict_df(pred_df)
        _p(f"Predictions: {len(self.predictions)} players")
        return self.predictions

    def build_betting_projections(self) -> list:
        try:
            self._projector.load()
            self.game_projections = self._projector.project_all_games(
                self.todays_games,
                player_preds=self.predictions if not self.predictions.empty else None,
            )
        except Exception as e:
            _p(f"Betting projections failed ({e})")
            self.game_projections = []
        return self.game_projections

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self, force_retrain: bool = False, status_callback=None,
            date: Optional[str] = None) -> pd.DataFrame:
        def st(msg, frac):
            _p(msg)
            if status_callback:
                status_callback(msg, frac)

        st("Fetching schedule …",              0.05); self.fetch_schedule(date)
        st("Loading rosters …",                0.15); self.fetch_rosters()
        st("Downloading game logs …",          0.30); self.fetch_game_logs()
        st("Fetching advanced stats …",        0.50); self.fetch_nst_stats()
        st("Fetching goalie stats …",          0.60); self.fetch_goalie_quality()
        st("Checking injuries & scratches …",  0.65); self.fetch_unavailable_players()
        st("Training / loading model …",       0.70); self.train_model(force_retrain)
        st("Building predictions …",           0.90); self.build_predictions()
        st("Calculating game projections …",   0.95); self.build_betting_projections()
        st("Done!",                            1.00)
        return self.predictions

    # ── Convenience getters ───────────────────────────────────────────────────

    def get_games_today(self) -> list:    return self.todays_games
    def get_teams_playing(self) -> list:
        teams = set()
        for g in self.todays_games:
            teams.update([g["away_team"], g["home_team"]])
        return sorted(teams)
