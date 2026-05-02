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
    NHL anytime goalscorer classifier.
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
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        feat_cols = [c for c in self.feature_names if c in prediction_df.columns]
        X = prediction_df[feat_cols].fillna(0)
        probs = self._model.predict(X)
        prediction_df = prediction_df.copy()
        prediction_df["goal_probability"] = probs

        def _conf(p):
            if p >= 0.35: return "Elite"
            if p >= 0.25: return "High"
            if p >= 0.15: return "Medium"
            return "Low"

        prediction_df["confidence"] = prediction_df["goal_probability"].apply(_conf)
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
