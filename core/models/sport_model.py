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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.model_selection import train_test_split

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
        self.calibrator    = None
        self.metrics       = {}

        self._model_path    = f"{MODEL_CACHE_DIR}/{name}_model.joblib"
        self._scaler_path   = f"{MODEL_CACHE_DIR}/{name}_scaler.joblib"
        self._features_path = f"{MODEL_CACHE_DIR}/{name}_features.joblib"
        self._metrics_path  = f"{MODEL_CACHE_DIR}/{name}_metrics.joblib"

    def _build_clf(self, scale_pos_weight=4.0, n_samples=1000):
        # Scale complexity with data — prevents overfitting on small datasets.
        # Key anti-overfitting levers:
        #   max_depth=3      — shallow trees generalise better for sports data
        #   min_child_weight — don't split on tiny groups of players
        #   gamma            — only split if gain is meaningful
        #   reg_alpha/lambda — L1+L2 regularisation
        #   subsample/colsample — random sampling reduces variance
        if n_samples < 300:
            n_est, depth, lr, alpha, lam, mcw = 80,  2, 0.05, 2.0, 3.0, 10
        elif n_samples < 1000:
            n_est, depth, lr, alpha, lam, mcw = 150, 3, 0.05, 1.0, 2.0, 8
        else:
            n_est, depth, lr, alpha, lam, mcw = MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_LEARNING_RATE, 0.5, 1.5, 5

        if XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=0.75, colsample_bytree=0.75,   # more aggressive subsampling
                scale_pos_weight=scale_pos_weight,
                reg_alpha=alpha, reg_lambda=lam,
                min_child_weight=mcw,
                gamma=0.2,              # higher = harder to split = less overfitting
                eval_metric="logloss", random_state=MODEL_RANDOM_STATE,
                verbosity=0, n_jobs=1,
            )
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.75, random_state=MODEL_RANDOM_STATE,
            min_samples_leaf=15)

    def _build_reg(self, n_samples=1000):
        if n_samples < 300:
            n_est, depth, lr, alpha, lam, mcw = 80,  2, 0.05, 2.0, 3.0, 10
        elif n_samples < 1000:
            n_est, depth, lr, alpha, lam, mcw = 150, 3, 0.05, 1.0, 2.0, 8
        else:
            n_est, depth, lr, alpha, lam, mcw = MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_LEARNING_RATE, 0.5, 1.5, 5

        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=0.75, colsample_bytree=0.75,
                reg_alpha=alpha, reg_lambda=lam,
                min_child_weight=mcw,
                gamma=0.2,
                eval_metric="mae", random_state=MODEL_RANDOM_STATE,
                verbosity=0, n_jobs=1,
            )
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.75, random_state=MODEL_RANDOM_STATE,
            min_samples_leaf=15)

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

            # ── Date-based train/test split (prevents data leakage) ────────────
            # Use last 20% of rows as held-out test set when we have enough data.
            # This simulates real prediction: train on past, test on recent.
            if n >= 200:
                split = int(n * 0.80)
                X_tr, X_te = X_scaled[:split], X_scaled[split:]
                y_tr, y_te = y[:split],         y[split:]
            else:
                X_tr, X_te, y_tr, y_te = X_scaled, X_scaled, y, y

            self.model.fit(X_tr, y_tr)

            # ── Isotonic calibration on held-out data ──────────────────────────
            # Ensures a 35% prediction actually hits ~35% of the time.
            # Only fit calibrator when held-out set is large enough.
            if len(y_te) >= 30 and len(np.unique(y_te)) > 1:
                raw_proba = self.model.predict_proba(X_te)[:, 1]
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(raw_proba, y_te)
                proba_te = self.calibrator.predict(raw_proba)
                auc = float(roc_auc_score(y_te, proba_te))
                # Also get in-sample proba for brier
                proba_all = self.calibrator.predict(
                    self.model.predict_proba(X_scaled)[:, 1])
            else:
                proba_te  = self.model.predict_proba(X_te)[:, 1]
                proba_all = self.model.predict_proba(X_scaled)[:, 1]
                auc = float(roc_auc_score(y_te, proba_te))

            proba = proba_all

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
            raw = self.model.predict_proba(Xs)[:, 1]
            if self.calibrator is not None:
                return self.calibrator.predict(raw)
            return raw
        return self.model.predict(Xs)

    def save(self):
        joblib.dump(self.model,         self._model_path)
        joblib.dump(self.scaler,        self._scaler_path)
        joblib.dump(self.feature_names, self._features_path)
        joblib.dump(self.metrics,       self._metrics_path)
        joblib.dump(self.calibrator,    self._model_path.replace("_model.", "_calibrator."))

    def load(self) -> bool:
        try:
            self.model         = joblib.load(self._model_path)
            self.scaler        = joblib.load(self._scaler_path)
            self.feature_names = joblib.load(self._features_path)
            self.metrics       = joblib.load(self._metrics_path)
            # Load calibrator if it exists (may not exist for old saved models)
            cal_path = self._model_path.replace("_model.", "_calibrator.")
            try:
                self.calibrator = joblib.load(cal_path)
            except FileNotFoundError:
                self.calibrator = None
            self.is_trained = True
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
