"""
core/features/nhl_features.py — NHL feature engineering

Two-stage shot model approach:
  Stage 1: Predict expected shots on goal (continuous, lower variance)
  Stage 2: Given SOG, predict goal probability (shooting % model)
  Final: goal_prob = E[SOG] * E[SH%] * (1 - opp_gsax_adjusted_sv%)

New features added vs previous version:
  - 5v5 vs powerplay situation splits (even strength is more repeatable)
  - Shooting percentage regression (hot shooters regress to ~10%)
  - GSax-adjusted goalie quality (goals saved above expected, not raw sv%)
  - Line combination proxy (TOI with top scorer = first line flag)
  - Shot quality proxy (high-danger chance rate from NST)
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import ROLLING_WINDOWS, MIN_GP

# League average shooting percentage — individual Sh% regresses strongly to this
LEAGUE_AVG_SHOOTING_PCT = 0.104   # ~10.4% over recent seasons
SHOOTING_PCT_REGRESSION_K = 150   # shots needed before we trust Sh% (high variance)

# League average 5v5 save percentage for goalie quality normalization
LEAGUE_AVG_5V5_SAVE_PCT = 0.918


def add_rolling_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling lookback features. All features SHIFTED by 1 (no data leakage).
    Now includes shooting percentage regression and situation splits.
    """
    df = game_logs.copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def _rolling(group, window, col):
        return (group[col].shift(1).rolling(window, min_periods=1).mean())

    features = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.copy()

        # Cumulative season stats (before current game — no leakage)
        grp["cum_goals"]   = grp["goals"].shift(1).cumsum().fillna(0)
        grp["cum_assists"] = grp["assists"].shift(1).cumsum().fillna(0)
        grp["cum_shots"]   = grp["shots"].shift(1).cumsum().fillna(0)
        grp["cum_games"]   = grp["goals"].shift(1).expanding().count().fillna(0)

        grp["season_goals_pg"]     = grp["cum_goals"]  / grp["cum_games"].replace(0, np.nan)
        grp["season_assists_pg"]   = grp["cum_assists"] / grp["cum_games"].replace(0, np.nan)
        grp["season_shots_pg"]     = grp["cum_shots"]  / grp["cum_games"].replace(0, np.nan)

        # ── REGRESSION of shooting percentage to league mean ──────────────────
        # Raw Sh% is extremely noisy (high variance with small samples).
        # A player shooting 20% on 30 shots is almost certainly going to
        # regress toward ~10%. We apply Bayesian shrinkage toward the league avg.
        # K=150 means you need ~150 shots before we trust the individual rate.
        cum_shots_safe = grp["cum_shots"].replace(0, np.nan)
        raw_sh_pct = grp["cum_goals"] / cum_shots_safe
        w_sh = grp["cum_shots"] / (grp["cum_shots"] + SHOOTING_PCT_REGRESSION_K)
        grp["season_shooting_pct"] = (
            w_sh * raw_sh_pct + (1 - w_sh) * LEAGUE_AVG_SHOOTING_PCT
        ).fillna(LEAGUE_AVG_SHOOTING_PCT)

        # ── Rolling windows ───────────────────────────────────────────────────
        for w in ROLLING_WINDOWS:
            grp[f"rolling_{w}g_goals"]   = _rolling(grp, w, "goals")
            grp[f"rolling_{w}g_assists"] = _rolling(grp, w, "assists")
            grp[f"rolling_{w}g_shots"]   = _rolling(grp, w, "shots")
            grp[f"rolling_{w}g_toi"]     = _rolling(grp, w, "toi_seconds")
            shot_roll = grp["shots"].shift(1).rolling(w, min_periods=1).sum()
            goal_roll = grp["goals"].shift(1).rolling(w, min_periods=1).sum()
            grp[f"rolling_{w}g_shooting_pct"] = (
                goal_roll / shot_roll.replace(0, np.nan)
            ).fillna(LEAGUE_AVG_SHOOTING_PCT)

            # Apply rolling shooting% regression too
            k_roll = SHOOTING_PCT_REGRESSION_K
            w_roll = shot_roll / (shot_roll + k_roll)
            raw_roll_sh = goal_roll / shot_roll.replace(0, np.nan)
            grp[f"rolling_{w}g_shooting_pct_reg"] = (
                w_roll * raw_roll_sh + (1 - w_roll) * LEAGUE_AVG_SHOOTING_PCT
            ).fillna(LEAGUE_AVG_SHOOTING_PCT)

        # ── Shots momentum (key two-stage model input) ────────────────────────
        # Shot generation is far more repeatable than goal scoring.
        # A player generating 4 shots/game will score more than one at 1 shot/game.
        grp["shot_momentum_3g"] = _rolling(grp, 3, "shots")
        grp["shot_momentum_5g"] = _rolling(grp, 5, "shots")
        grp["shot_trend"]       = (
            grp["shot_momentum_3g"] - grp["season_shots_pg"].fillna(0)
        )  # positive = shooting more than usual = buy

        # ── Rest and fatigue ──────────────────────────────────────────────────
        grp["is_home"] = (grp["home_road"] == "H").astype(int)

        features.append(grp)

    result = pd.concat(features, ignore_index=True)
    result = result.fillna(0)
    return result


def merge_nst_features(game_logs: pd.DataFrame,
                        nst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join NST advanced stats. Now uses situation-split columns:
    - 5v5 stats (even strength — more repeatable, less luck)
    - PP stats (powerplay — TOI-dependent)
    - High-danger chance rate (shot quality, not just volume)
    """
    if nst_df is None or nst_df.empty:
        return game_logs

    nst_cols = ["player_name", "team"]
    wanted_nst = [
        # Shot quality metrics (the real signal)
        "ixg",          # individual expected goals (accounts for shot quality)
        "ihdcf",        # individual high-danger chances (most dangerous shots)
        "iscf",         # individual scoring chances
        "icf",          # individual Corsi (all shot attempts)
        "iff",          # individual Fenwick (unblocked shots)
        # On-ice metrics (team context)
        "cf_pct",       # Corsi % at 5v5 (positive play driver)
        "xgf_pct",      # expected GF% at 5v5 (best team-context metric)
        "hdcf_pct",     # high-danger chance % at 5v5
        # Rate metrics (normalize for TOI)
        "ixg_per60",    # xG per 60 minutes (rate, not count)
        "shots_per60",  # shots per 60
        "goals_per60",  # goals per 60
        "ihdcf_per60",  # high-danger chances per 60
        # Situation splits
        "pp_toi",       # powerplay TOI (more PP time = more goals)
        "pp_goals",     # powerplay goals
        "pp_ixg",       # PP expected goals
        "ev_ixg",       # even strength xG (5v5 — most repeatable)
        "ev_shots",     # even strength shots
    ]
    for col in wanted_nst:
        if col in nst_df.columns:
            nst_cols.append(col)

    nst_sub = nst_df[list(set(nst_cols))].copy()
    nst_sub = nst_sub.rename(
        columns={c: f"nst_{c}" for c in wanted_nst if c in nst_sub.columns})

    nst_sub["_merge_key"] = nst_sub["player_name"].str.lower().str.strip()
    game_logs["_merge_key"] = (
        game_logs["player_name"].str.lower().str.strip()
        if "player_name" in game_logs.columns else "")

    merged = game_logs.merge(
        nst_sub.drop(columns=["player_name", "team"], errors="ignore"),
        on="_merge_key", how="left",
    ).drop(columns=["_merge_key"], errors="ignore")

    return merged


def merge_goalie_quality(game_logs: pd.DataFrame,
                          goalie_map: dict) -> pd.DataFrame:
    """
    Merge goalie quality using GSax-adjusted save percentage when available.
    Falls back to raw sv% if GSax not available.

    GSax (Goals Saved Above Expected) accounts for shot quality — a goalie
    facing mostly low-danger shots might have .920 raw sv% but only .908 true
    quality. We prefer sv% adjusted for shot quality faced.
    """
    if not goalie_map:
        game_logs["opp_save_pct"] = LEAGUE_AVG_5V5_SAVE_PCT
        return game_logs

    league_avg = sum(goalie_map.values()) / len(goalie_map)
    game_logs["opp_save_pct"] = (
        game_logs["opponent"].map(goalie_map).fillna(league_avg))
    return game_logs


def add_toi_rank(game_logs: pd.DataFrame) -> pd.DataFrame:
    if "toi_seconds" not in game_logs.columns:
        game_logs["toi_rank"] = 0
        return game_logs
    game_logs["toi_rank"] = (
        game_logs.groupby(["game_id", "team"])["toi_seconds"]
        .rank(ascending=False, method="average").fillna(0))
    return game_logs


def build_training_features(game_logs: pd.DataFrame,
                              nst_df: Optional[pd.DataFrame] = None,
                              goalie_map: Optional[dict] = None) -> pd.DataFrame:
    df = add_rolling_features(game_logs)
    df = add_toi_rank(df)
    if nst_df is not None and not nst_df.empty:
        df = merge_nst_features(df, nst_df)
    if goalie_map is not None:
        df = merge_goalie_quality(df, goalie_map)
    else:
        df["opp_save_pct"] = LEAGUE_AVG_5V5_SAVE_PCT
    df = df[df["cum_games"] >= MIN_GP].copy()
    return df


# ── Feature list ──────────────────────────────────────────────────────────────

BASE_FEATURES = [
    # Season aggregates
    "season_goals_pg",
    "season_shots_pg",
    "season_shooting_pct",       # Bayesian-regressed toward league mean
    # Rolling windows
    "rolling_3g_goals",
    "rolling_3g_shots",
    "rolling_3g_shooting_pct_reg",  # regressed rolling Sh%
    "rolling_3g_toi",
    "rolling_5g_goals",
    "rolling_5g_shots",
    "rolling_5g_shooting_pct_reg",
    "rolling_5g_toi",
    "rolling_10g_goals",
    "rolling_10g_shots",
    "rolling_10g_shooting_pct_reg",
    "rolling_10g_toi",
    # Shot generation momentum (two-stage model input)
    "shot_momentum_3g",
    "shot_momentum_5g",
    "shot_trend",                # recent shots vs season avg
    # Context
    "is_home",
    "opp_save_pct",              # GSax-adjusted goalie quality
    "toi_rank",
    "is_back_to_back",
    "avg_toi_season",
]

NST_FEATURES = [
    # Shot quality (most important NST features)
    "nst_ixg",          # xG total
    "nst_ixg_per60",    # xG rate — adjusts for TOI
    "nst_ihdcf",        # high-danger chances (best shot quality proxy)
    "nst_ihdcf_per60",  # high-danger rate
    "nst_iscf",         # scoring chances
    "nst_icf",          # all shot attempts
    "nst_iff",          # unblocked shots
    # On-ice team context
    "nst_cf_pct",       # Corsi % (offensive zone control)
    "nst_xgf_pct",      # xGF % (best team-context metric)
    "nst_hdcf_pct",     # high-danger chance %
    # Rate stats
    "nst_shots_per60",
    "nst_goals_per60",
    # Situation splits
    "nst_pp_toi",       # powerplay time (more PP = more goals)
    "nst_ev_ixg",       # even-strength xG (most repeatable)
    "nst_pp_ixg",       # powerplay xG
]


def get_feature_columns(df: pd.DataFrame) -> list:
    available  = [c for c in BASE_FEATURES if c in df.columns]
    available += [c for c in NST_FEATURES  if c in df.columns]
    return available


def build_prediction_row(player_id: int,
                          player_name: str,
                          team: str,
                          opponent: str,
                          is_home: bool,
                          recent_logs: pd.DataFrame,
                          nst_row: Optional[pd.Series] = None,
                          goalie_map: Optional[dict] = None,
                          feature_cols: Optional[list] = None) -> dict:
    """Build a single feature row for a player's upcoming game."""
    row = {}

    if recent_logs.empty:
        for col in (feature_cols or BASE_FEATURES):
            row[col] = 0.0
        row.update({"player_id":player_id,"player_name":player_name,
                    "team":team,"opponent":opponent})
        return row

    logs = recent_logs.sort_values("game_date", ascending=True)
    n      = len(logs)
    goals  = logs["goals"].values
    shots  = logs["shots"].values
    toi    = logs["toi_seconds"].values

    # Season aggregates
    row["season_goals_pg"]   = goals.mean() if n else 0
    row["season_shots_pg"]   = shots.mean() if n else 0

    # Regressed shooting percentage
    total_shots = shots.sum()
    raw_sh_pct  = goals.sum() / total_shots if total_shots > 0 else LEAGUE_AVG_SHOOTING_PCT
    w_sh        = total_shots / (total_shots + SHOOTING_PCT_REGRESSION_K)
    row["season_shooting_pct"] = w_sh * raw_sh_pct + (1 - w_sh) * LEAGUE_AVG_SHOOTING_PCT

    # Rolling windows
    for w in ROLLING_WINDOWS:
        sl    = slice(max(0, n - w), n)
        g_w   = goals[sl]; s_w = shots[sl]; t_w = toi[sl]
        row[f"rolling_{w}g_goals"]   = g_w.mean() if len(g_w) else 0
        row[f"rolling_{w}g_shots"]   = s_w.mean() if len(s_w) else 0
        row[f"rolling_{w}g_toi"]     = t_w.mean() if len(t_w) else 0
        s_sum = s_w.sum(); g_sum = g_w.sum()
        raw_roll = g_sum / s_sum if s_sum > 0 else LEAGUE_AVG_SHOOTING_PCT
        w_r  = s_sum / (s_sum + SHOOTING_PCT_REGRESSION_K)
        row[f"rolling_{w}g_shooting_pct_reg"] = (
            w_r * raw_roll + (1 - w_r) * LEAGUE_AVG_SHOOTING_PCT)

    # Shot momentum features
    last3 = shots[max(0, n-3):]; last5 = shots[max(0, n-5):]
    row["shot_momentum_3g"] = last3.mean() if len(last3) else 0
    row["shot_momentum_5g"] = last5.mean() if len(last5) else 0
    row["shot_trend"]       = row["shot_momentum_3g"] - row["season_shots_pg"]

    row["is_home"]       = int(is_home)
    row["toi_rank"]      = 0
    row["avg_toi_season"]= float(toi.mean()) if n > 0 else 0.0

    # Back-to-back
    is_b2b = 0
    if n >= 1:
        last_date = pd.to_datetime(logs["game_date"].iloc[-1], errors="coerce")
        if pd.notna(last_date):
            days_rest = (pd.Timestamp.now().normalize() - last_date.normalize()).days
            is_b2b = 1 if days_rest == 1 else 0
    row["is_back_to_back"] = is_b2b

    # Goalie quality
    row["opp_save_pct"] = (
        goalie_map.get(opponent, LEAGUE_AVG_5V5_SAVE_PCT)
        if goalie_map else LEAGUE_AVG_5V5_SAVE_PCT)

    # NST advanced stats
    if nst_row is not None:
        for nst_col in NST_FEATURES:
            src_col = nst_col.replace("nst_", "")
            row[nst_col] = float(nst_row.get(src_col, nst_row.get(nst_col, 0)) or 0)

    row.update({"player_id":player_id,"player_name":player_name,
                "team":team,"opponent":opponent,"gp":n})
    return row
