"""
core/models/nhl_model.py — NHL goalscorer model.

GoalscorerModel is a thin wrapper around SportModel that preserves
the interface expected by nhl_pipeline.py while reusing the shared
XGBoost infrastructure. There is no longer a separate implementation.
"""

import os
import numpy as np
import pandas as pd

from core.models.sport_model import SportModel
from config.settings import MODEL_CACHE_DIR

# Feature column helper still lives in nhl_features
from core.features.nhl_features import get_feature_columns


class GoalscorerModel:
    """
    NHL goal probability classifier.
    Delegates all ML work to SportModel("nhl_goals", "classify").
    """

    def __init__(self):
        self._model     = SportModel("nhl_goals", "classify")
        self.metrics:   dict  = {}

    # ── Proxy properties ──────────────────────────────────────────────────────
    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def feature_names(self) -> list:
        return self._model.feature_names

    # ── Public interface (matches what nhl_pipeline.py calls) ─────────────────
    def train(self, feature_df: pd.DataFrame,
              target_col: str = "scored_goal",
              min_samples: int = 200,
              verbose: bool = True) -> dict:
        feat_cols = get_feature_columns(feature_df)
        self._model.train(feature_df, target_col, feat_cols)
        self.metrics = self._model.metrics
        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_df(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add goal_probability and confidence columns to prediction_df.
        Returns the enriched DataFrame sorted by goal_probability desc.

        IMPORTANT: Model output is clipped to a physics-based ceiling derived
        from the player's own shot volume and shooting percentage. This prevents
        contextual features (weak goalie, home ice) from inflating predictions
        for players who barely shoot (e.g. 4th-line D with 0 goals).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        feat_cols = [c for c in self.feature_names if c in prediction_df.columns]
        X = prediction_df[feat_cols].fillna(0)
        probs = self._model.predict(X)
        prediction_df = prediction_df.copy()

        # ── Physics-based probability ceiling ────────────────────────────────
        # A player cannot score more goals than their shot volume × sh% allows.
        # Ceiling = shots_per_game × regressed_sh% × (1 / avg_save_pct_faced)
        # This ensures a 0-goal/0.4-shot defender never projects at 0.60.
        LEAGUE_AVG_SH_PCT  = 0.104   # league avg shooting pct
        BEST_CASE_GOALS_PER_SHOT = 0.130  # facing a .870 sv% goalie
        K_SHOTS = 150   # shots before we fully trust individual sh%

        ceilings = []
        for _, row in prediction_df.iterrows():
            shots_pg  = float(row.get("season_shots_pg",
                         row.get("rolling_5g_shots", 0)) or 0)
            gp        = max(int(row.get("gp_season", row.get("gp", 1))), 1)
            s_goals   = int(row.get("season_goals", 0))
            total_shots = shots_pg * gp
            raw_sh    = s_goals / max(total_shots, 1)

            # Regress shooting pct toward league avg
            w         = total_shots / (total_shots + K_SHOTS)
            reg_sh    = w * raw_sh + (1 - w) * LEAGUE_AVG_SH_PCT

            # Ceiling: shots/game × best-case shooting pct
            ceiling   = shots_pg * min(reg_sh * 1.5, BEST_CASE_GOALS_PER_SHOT)
            ceiling   = max(0.02, min(ceiling, 0.65))  # bounds
            ceilings.append(ceiling)

        # Apply ceiling — model cannot exceed what the player's stats support
        raw_probs    = np.array(probs)
        ceiling_arr  = np.array(ceilings)
        clipped      = np.minimum(raw_probs, ceiling_arr)

        # Soft blend so we don't hard-clip stars (they should be near ceiling anyway)
        # 90% ceiling-clipped, 10% raw — preserves ranking, prevents absurd outliers
        final_probs  = 0.90 * clipped + 0.10 * np.minimum(raw_probs, 0.65)
        prediction_df["goal_probability"] = np.round(final_probs, 4)
        prediction_df["prob_ceiling"]     = np.round(ceiling_arr, 4)

        def _conf(p):
            if p >= 0.32: return "Elite"
            if p >= 0.22: return "High"
            if p >= 0.14: return "Medium"
            return "Low"

        prediction_df["confidence"]  = prediction_df["goal_probability"].apply(_conf)
        prediction_df["conf_goals"]  = prediction_df["confidence"]  # alias

        # SOG per-category confidence
        def _conf_sog(s):
            if s >= 4.0: return "Elite"
            if s >= 3.0: return "High"
            if s >= 2.0: return "Medium"
            return "Low"
        if "projected_sog" in prediction_df.columns:
            prediction_df["conf_sog"] = prediction_df["projected_sog"].apply(_conf_sog)
        else:
            prediction_df["conf_sog"] = "Low"

        return prediction_df.sort_values("goal_probability", ascending=False).reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        fi = self._model.feature_importance()
        if not fi.empty and "importance" in fi.columns:
            fi["importance_pct"] = fi["importance"] / fi["importance"].sum() * 100
        return fi

    def save(self) -> None:
        self._model.save()

    def load(self) -> bool:
        ok = self._model.load()
        if ok:
            self.metrics = self._model.metrics
        return ok

    def is_saved(self) -> bool:
        return self._model.is_saved()
