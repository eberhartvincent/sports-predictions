"""
sport_model.py — Generic XGBoost model reused across NHL, MLB, NBA pipelines.
Supports both classification (scored/didn't) and regression (predict value).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import (
    MODEL_CACHE_DIR, MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH,
    MODEL_LEARNING_RATE, MODEL_SUBSAMPLE, MODEL_COLSAMPLE_BYTREE,
    MODEL_RANDOM_STATE, MODEL_MIN_SAMPLES,
)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class SportModel:
    """
    Generic XGBoost model for any sport projection.
    mode='classify'  → predicts probability of binary outcome (e.g. scored a goal)
    mode='regress'   → predicts continuous value (e.g. expected points)
    """

    def __init__(self, name: str, mode: str = "classify"):
        self.name          = name          # e.g. "mlb_hits", "nba_points"
        self.mode          = mode          # "classify" or "regress"
        self.model         = None
        self.scaler        = StandardScaler()
        self.feature_names = []
        self.is_trained    = False
        self.metrics       = {}

        self._model_path    = f"{MODEL_CACHE_DIR}/{name}_model.joblib"
        self._scaler_path   = f"{MODEL_CACHE_DIR}/{name}_scaler.joblib"
        self._features_path = f"{MODEL_CACHE_DIR}/{name}_features.joblib"
        self._metrics_path  = f"{MODEL_CACHE_DIR}/{name}_metrics.joblib"

    def _build_clf(self, scale_pos_weight=4.0, n_samples=1000):
        # Scale complexity with data size — prevents overfitting on small datasets
        if n_samples < 300:
            n_est, depth, lr, alpha, lam = 50,  2, 0.10, 1.0, 2.0
        elif n_samples < 1000:
            n_est, depth, lr, alpha, lam = 100, 3, 0.08, 0.5, 1.0
        else:
            n_est, depth, lr, alpha, lam = MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_LEARNING_RATE, 0.1, 1.0

        if XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=MODEL_SUBSAMPLE, colsample_bytree=MODEL_COLSAMPLE_BYTREE,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=alpha, reg_lambda=lam,
                min_child_weight=5,   # prevents splitting on tiny leaf nodes
                gamma=0.1,            # minimum gain to make a split
                eval_metric="logloss", random_state=MODEL_RANDOM_STATE,
                verbosity=0, n_jobs=1,
            )
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=MODEL_SUBSAMPLE, random_state=MODEL_RANDOM_STATE,
            min_samples_leaf=10)

    def _build_reg(self, n_samples=1000):
        if n_samples < 300:
            n_est, depth, lr, alpha, lam = 50,  2, 0.10, 1.0, 2.0
        elif n_samples < 1000:
            n_est, depth, lr, alpha, lam = 100, 3, 0.08, 0.5, 1.0
        else:
            n_est, depth, lr, alpha, lam = MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_LEARNING_RATE, 0.1, 1.0

        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=MODEL_SUBSAMPLE, colsample_bytree=MODEL_COLSAMPLE_BYTREE,
                reg_alpha=alpha, reg_lambda=lam,
                min_child_weight=5,
                gamma=0.1,
                eval_metric="mae", random_state=MODEL_RANDOM_STATE,
                verbosity=0, n_jobs=1,
            )
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=MODEL_SUBSAMPLE, random_state=MODEL_RANDOM_STATE,
            min_samples_leaf=10)

    def train(self, df: pd.DataFrame, target_col: str,
              feat_cols: list, min_samples: int = 100) -> dict:
        df = df.dropna(subset=[target_col])
        X  = df[feat_cols].fillna(0).values.astype(np.float32)
        y  = df[target_col].values

        min_samples = min_samples or MODEL_MIN_SAMPLES
        if len(y) < 30:
            raise ValueError(f"Need ≥30 samples; got {len(y)}")

        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feat_cols
        n = len(y)

        if self.mode == "classify":
            y = y.astype(int)
            n_neg = int((y==0).sum()); n_pos = int((y==1).sum())
            spw   = n_neg / max(n_pos, 1)
            self.model = self._build_clf(spw, n_samples=n)
            self.model.fit(X_scaled, y)

            proba = self.model.predict_proba(X_scaled)[:, 1]
            # Cross-val AUC only when n is large enough to be reliable and fast
            if n >= 500:
                from sklearn.model_selection import cross_val_score
                cv_folds = min(5, max(2, n // 200))
                try:
                    cv_scores = cross_val_score(
                        self._build_clf(spw, n_samples=n), X_scaled, y,
                        cv=cv_folds, scoring="roc_auc", n_jobs=1)
                    auc = float(cv_scores.mean())
                except Exception:
                    auc = float(roc_auc_score(y, proba))
            else:
                auc = float(roc_auc_score(y, proba))

            self.metrics = {
                "train_auc":   auc,
                "brier_score": float(brier_score_loss(y, proba)),
                "n_samples":   int(n),
                "n_features":  int(len(feat_cols)),
                "pos_rate":    float(y.mean()),
            }
            cv_label = "CV-AUC" if n >= 500 else "AUC"
            print(f"[{self.name}] {cv_label}={auc:.3f} n={n} pos_rate={y.mean():.1%}")
        else:
            self.model = self._build_reg(n_samples=n)
            self.model.fit(X_scaled, y)
            preds = self.model.predict(X_scaled)
            self.metrics = {
                "train_mae":  float(mean_absolute_error(y, preds)),
                "n_samples":  int(len(y)),
                "n_features": int(len(feat_cols)),
                "target_mean": float(y.mean()),
            }
            print(f"[{self.name}] MAE={self.metrics['train_mae']:.3f} "
                  f"mean={y.mean():.2f} n={len(y)}")

        self.is_trained = True
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"{self.name} not trained")
        aligned = pd.DataFrame(0.0, index=X.index, columns=self.feature_names)
        for c in self.feature_names:
            if c in X.columns:
                aligned[c] = X[c].fillna(0)
        Xs = self.scaler.transform(aligned.values.astype(np.float32))
        if self.mode == "classify":
            return self.model.predict_proba(Xs)[:, 1]
        return self.model.predict(Xs)

    def save(self):
        joblib.dump(self.model,         self._model_path)
        joblib.dump(self.scaler,        self._scaler_path)
        joblib.dump(self.feature_names, self._features_path)
        joblib.dump(self.metrics,       self._metrics_path)

    def load(self) -> bool:
        try:
            self.model         = joblib.load(self._model_path)
            self.scaler        = joblib.load(self._scaler_path)
            self.feature_names = joblib.load(self._features_path)
            self.metrics       = joblib.load(self._metrics_path)
            self.is_trained    = True
            return True
        except FileNotFoundError:
            return False

    def is_saved(self) -> bool:
        return all(os.path.exists(p) for p in
                   [self._model_path, self._scaler_path, self._features_path])

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        fi = pd.DataFrame({"feature": self.feature_names,
                           "importance": self.model.feature_importances_})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi["pct"] = fi["importance"] / fi["importance"].sum() * 100
        return fi
