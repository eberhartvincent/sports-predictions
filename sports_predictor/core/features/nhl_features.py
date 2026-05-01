"""
feature_engineering.py — Build ML features from raw NHL + NST data

Training features (computed per game, looking backwards):
  rolling_{n}g_goals         — goals in last N games
  rolling_{n}g_shots         — shots in last N games
  rolling_{n}g_shooting_pct  — shooting % in last N games
  rolling_{n}g_toi           — avg TOI (seconds) in last N games
  season_goals_pg            — season goals per game so far
  season_shots_pg            — season shots per game so far
  season_shooting_pct        — season shooting %
  career_goals_pg            — career goals per game (proxy: season rate)
  pp_toi_pct                 — power-play TOI fraction (from NST)
  ixg_per_game               — individual xGoals per game (from NST)
  cf_pct                     — on-ice Corsi % (from NST)
  xgf_pct                    — on-ice expected GF % (from NST)
  is_home                    — 1 if home game else 0
  opp_save_pct               — opponent goalie save % (lower = easier to score)
  toi_rank                   — player's TOI rank on their team (1 = most ice)
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import ROLLING_WINDOWS, MIN_GP


# ── Rolling features from game logs ──────────────────────────────────────────

def add_rolling_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Given a game-log DataFrame (one row per player-game), add rolling
    lookback features.  All features are SHIFTED by 1 so they represent
    information available BEFORE the current game (no data leakage).
    """
    df = game_logs.copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def _rolling(group: pd.DataFrame, window: int, col: str) -> pd.Series:
        return (
            group[col]
            .shift(1)                        # no leakage
            .rolling(window, min_periods=1)
            .mean()
        )

    features = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.copy()

        # Cumulative season stats (up to but not including current game)
        grp["cum_goals"]   = grp["goals"].shift(1).cumsum().fillna(0)
        grp["cum_assists"] = grp["assists"].shift(1).cumsum().fillna(0)
        grp["cum_shots"]   = grp["shots"].shift(1).cumsum().fillna(0)
        grp["cum_games"]   = grp["goals"].shift(1).expanding().count().fillna(0)

        grp["season_goals_pg"]       = grp["cum_goals"]   / grp["cum_games"].replace(0, np.nan)
        grp["season_assists_pg"]     = grp["cum_assists"]  / grp["cum_games"].replace(0, np.nan)
        grp["season_shots_pg"]       = grp["cum_shots"]   / grp["cum_games"].replace(0, np.nan)
        grp["season_shooting_pct"]   = (
            grp["cum_goals"] / grp["cum_shots"].replace(0, np.nan)
        ).fillna(0)

        # Rolling windows
        for w in ROLLING_WINDOWS:
            grp[f"rolling_{w}g_goals"]   = _rolling(grp, w, "goals")
            grp[f"rolling_{w}g_assists"] = _rolling(grp, w, "assists")
            grp[f"rolling_{w}g_shots"]   = _rolling(grp, w, "shots")
            grp[f"rolling_{w}g_toi"]     = _rolling(grp, w, "toi_seconds")
            shot_roll  = grp["shots"].shift(1).rolling(w, min_periods=1).sum()
            goal_roll  = grp["goals"].shift(1).rolling(w, min_periods=1).sum()
            grp[f"rolling_{w}g_shooting_pct"] = (
                goal_roll / shot_roll.replace(0, np.nan)
            ).fillna(0)

        # Is home game
        grp["is_home"] = (grp["home_road"] == "H").astype(int)

        features.append(grp)

    result = pd.concat(features, ignore_index=True)
    result = result.fillna(0)
    return result


# ── Merge NST advanced stats ──────────────────────────────────────────────────

def merge_nst_features(game_logs: pd.DataFrame,
                        nst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join NST season-level advanced stats onto the game log DataFrame.
    NST stats are season averages, so they're the same for every game —
    but they still provide useful context (playing style, xG profile, etc.)
    """
    if nst_df is None or nst_df.empty:
        return game_logs

    nst_cols = ["player_name", "team"]
    wanted_nst = [
        "ixg", "icf", "iff", "iscf", "ihdcf",
        "cf_pct", "xgf_pct", "hdcf_pct",
        "pp_toi", "pp_goals",
        "goals_per60", "ixg_per60", "shots_per60",
    ]
    for col in wanted_nst:
        if col in nst_df.columns:
            nst_cols.append(col)

    nst_sub = nst_df[list(set(nst_cols))].copy()
    nst_sub = nst_sub.rename(columns={c: f"nst_{c}" for c in wanted_nst if c in nst_sub.columns})

    # Normalise names for merging
    nst_sub["_merge_key"] = nst_sub["player_name"].str.lower().str.strip()
    game_logs["_merge_key"] = game_logs["player_name"].str.lower().str.strip() \
        if "player_name" in game_logs.columns else ""

    merged = game_logs.merge(
        nst_sub.drop(columns=["player_name", "team"], errors="ignore"),
        on="_merge_key",
        how="left",
    ).drop(columns=["_merge_key"], errors="ignore")

    return merged


# ── Merge goalie quality ──────────────────────────────────────────────────────

def merge_goalie_quality(game_logs: pd.DataFrame,
                          goalie_map: dict) -> pd.DataFrame:
    """
    Add opponent save% as a feature.
    goalie_map: {team_abbrev: avg_save_pct}
    """
    if not goalie_map:
        game_logs["opp_save_pct"] = 0.910   # league average
        return game_logs

    league_avg = sum(goalie_map.values()) / len(goalie_map) if goalie_map else 0.910
    game_logs["opp_save_pct"] = (
        game_logs["opponent"].map(goalie_map).fillna(league_avg)
    )
    return game_logs


# ── TOI rank on team ──────────────────────────────────────────────────────────

def add_toi_rank(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, rank players on the same team by TOI.
    Lower rank = more ice time = better scoring opportunity.
    """
    if "toi_seconds" not in game_logs.columns:
        game_logs["toi_rank"] = 0
        return game_logs

    game_logs["toi_rank"] = (
        game_logs.groupby(["game_id", "team"])["toi_seconds"]
        .rank(ascending=False, method="average")
        .fillna(0)
    )
    return game_logs


# ── Full feature pipeline for training ───────────────────────────────────────

def build_training_features(game_logs: pd.DataFrame,
                              nst_df: Optional[pd.DataFrame] = None,
                              goalie_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Full pipeline: rolling features → merge NST → goalie quality → TOI rank.
    Returns DataFrame with target column 'scored_goal' and all feature columns.
    """
    df = add_rolling_features(game_logs)
    df = add_toi_rank(df)

    if nst_df is not None and not nst_df.empty:
        df = merge_nst_features(df, nst_df)

    if goalie_map is not None:
        df = merge_goalie_quality(df, goalie_map)
    else:
        df["opp_save_pct"] = 0.910

    # Drop rows with no prior games (can't make rolling features)
    df = df[df["cum_games"] >= MIN_GP].copy()

    return df


# ── Feature list for the model ────────────────────────────────────────────────

BASE_FEATURES = [
    "season_goals_pg",
    "season_shots_pg",
    "season_shooting_pct",
    "rolling_3g_goals",
    "rolling_3g_shots",
    "rolling_3g_shooting_pct",
    "rolling_3g_toi",
    "rolling_5g_goals",
    "rolling_5g_shots",
    "rolling_5g_shooting_pct",
    "rolling_5g_toi",
    "rolling_10g_goals",
    "rolling_10g_shots",
    "rolling_10g_shooting_pct",
    "rolling_10g_toi",
    "is_home",
    "opp_save_pct",       # opponent goalie save% (confirmed starter if available)
    "toi_rank",
    "is_back_to_back",
    "avg_toi_season",
]

NST_FEATURES = [
    "nst_ixg",
    "nst_icf",
    "nst_iff",
    "nst_iscf",
    "nst_ihdcf",
    "nst_cf_pct",
    "nst_xgf_pct",
    "nst_ixg_per60",
    "nst_shots_per60",
    "nst_goals_per60",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature columns available in df (BASE + NST if present)."""
    available = [c for c in BASE_FEATURES if c in df.columns]
    available += [c for c in NST_FEATURES if c in df.columns]
    return available


# ── Build prediction features for today ──────────────────────────────────────

def build_prediction_row(player_id: int,
                          player_name: str,
                          team: str,
                          opponent: str,
                          is_home: bool,
                          recent_logs: pd.DataFrame,
                          nst_row: Optional[pd.Series] = None,
                          goalie_map: Optional[dict] = None,
                          feature_cols: Optional[list] = None) -> dict:
    """
    Build a single feature row for a player's upcoming game.
    recent_logs: the player's game logs sorted by date (most recent last).
    """
    row = {}

    if recent_logs.empty:
        # Fill zeros for unknown players
        for col in (feature_cols or BASE_FEATURES):
            row[col] = 0.0
        row["player_id"]   = player_id
        row["player_name"] = player_name
        row["team"]        = team
        row["opponent"]    = opponent
        return row

    # Sort newest first
    logs = recent_logs.sort_values("game_date", ascending=True)

    n = len(logs)
    goals   = logs["goals"].values
    shots   = logs["shots"].values
    toi     = logs["toi_seconds"].values

    # Season aggregates
    row["season_goals_pg"]     = goals.mean() if n else 0
    row["season_shots_pg"]     = shots.mean() if n else 0
    row["season_shooting_pct"] = (goals.sum() / shots.sum()) if shots.sum() > 0 else 0

    # Rolling windows
    for w in ROLLING_WINDOWS:
        sl = slice(max(0, n - w), n)
        g_w = goals[sl]
        s_w = shots[sl]
        t_w = toi[sl]
        row[f"rolling_{w}g_goals"]        = g_w.mean()  if len(g_w) else 0
        row[f"rolling_{w}g_shots"]        = s_w.mean()  if len(s_w) else 0
        row[f"rolling_{w}g_toi"]          = t_w.mean()  if len(t_w) else 0
        row[f"rolling_{w}g_shooting_pct"] = (
            g_w.sum() / s_w.sum() if s_w.sum() > 0 else 0
        )

    row["is_home"]     = int(is_home)
    row["toi_rank"]    = 0   # unknown before the game; will be set later

    # Back-to-back: did this player play yesterday?
    is_b2b = 0
    if n >= 1:
        last_game_date = pd.to_datetime(logs["game_date"].iloc[-1], errors="coerce")
        if pd.notna(last_game_date):
            days_rest = (pd.Timestamp.now().normalize() - last_game_date.normalize()).days
            is_b2b = 1 if days_rest == 1 else 0
    row["is_back_to_back"] = is_b2b

    # Season average TOI — higher TOI = more shots = more goals
    row["avg_toi_season"] = float(toi.mean()) if n > 0 else 0.0

    league_avg_sv = 0.910
    if goalie_map:
        row["opp_save_pct"] = goalie_map.get(opponent, league_avg_sv)
    else:
        row["opp_save_pct"] = league_avg_sv

    # NST advanced stats (season averages)
    if nst_row is not None:
        for nst_col in NST_FEATURES:
            src_col = nst_col.replace("nst_", "")
            row[nst_col] = float(nst_row.get(src_col, nst_row.get(nst_col, 0)) or 0)

    row["player_id"]   = player_id
    row["player_name"] = player_name
    row["team"]        = team
    row["opponent"]    = opponent
    row["gp"]          = n

    return row
