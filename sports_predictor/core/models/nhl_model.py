"""
model_trainer.py — Train, save, load, and evaluate the goalscorer model
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not installed; falling back to GradientBoostingClassifier")

from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH, MODEL_DIR
from feature_engineering import get_feature_columns

os.makedirs(MODEL_DIR, exist_ok=True)


class GoalscorerModel:

    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.is_trained    = False
        self.metrics       = {}

    def _build_estimator(self, scale_pos_weight: float = 4.0):
        if XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8, random_state=42,
            )

    def train(self, feature_df: pd.DataFrame,
              target_col: str = "scored_goal",
              min_samples: int = 200,
              verbose: bool = True) -> dict:

        if feature_df.empty or target_col not in feature_df.columns:
            raise ValueError("feature_df must be non-empty and contain 'scored_goal'.")

        feat_cols = get_feature_columns(feature_df)
        if not feat_cols:
            raise ValueError("No valid feature columns found.")

        df = feature_df[feat_cols + [target_col]].dropna(subset=[target_col])
        X  = df[feat_cols].fillna(0).values.astype(np.float32)
        y  = df[target_col].astype(int).values

        if len(y) < min_samples:
            raise ValueError(f"Need at least {min_samples} samples; got {len(y)}.")

        if verbose:
            print(f"[Model] Training on {len(y)} samples | "
                  f"Goal rate: {y.mean():.1%} | Features: {len(feat_cols)}")

        self.scaler        = StandardScaler()
        X_scaled           = self.scaler.fit_transform(X)
        self.feature_names = feat_cols

        n_neg            = int((y == 0).sum())
        n_pos            = int((y == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        if verbose:
            print(f"[Model] Class balance: {n_neg} neg / {n_pos} pos | "
                  f"weight={scale_pos_weight:.2f}")

        self.model = self._build_estimator(scale_pos_weight)
        self.model.fit(X_scaled, y)

        # Cross-validated AUC — honest out-of-sample estimate
        from sklearn.model_selection import cross_val_score
        n_folds = min(5, max(2, len(y) // 200))
        try:
            cv_scores = cross_val_score(
                self._build_estimator(scale_pos_weight),
                X_scaled, y, cv=n_folds, scoring="roc_auc", n_jobs=1)
            cv_auc = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_auc = 0.0
            cv_std = 0.0

        proba     = self.model.predict_proba(X_scaled)[:, 1]
        train_auc = float(roc_auc_score(y, proba))
        brier     = float(brier_score_loss(y, proba))

        self.metrics = {
            "cv_auc_mean": cv_auc,
            "cv_auc_std":  cv_std,
            "train_auc":   cv_auc,    # show CV AUC in UI — more honest
            "brier_score": brier,
            "n_samples":   int(len(y)),
            "n_features":  int(len(feat_cols)),
            "goal_rate":   float(y.mean()),
        }
        self.is_trained = True

        if verbose:
            print(f"[Model] CV-AUC: {cv_auc:.3f} ±{cv_std:.3f} | "
                  f"Train-AUC: {train_auc:.3f} | Brier: {brier:.4f}")

        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X_aligned = pd.DataFrame(0.0, index=X.index, columns=self.feature_names)
        for col in self.feature_names:
            if col in X.columns:
                X_aligned[col] = X[col].fillna(0)

        X_scaled = self.scaler.transform(X_aligned.values.astype(np.float32))
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_df(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(prediction_df)
        out   = prediction_df.copy()
        out["goal_probability"] = probs
        out["confidence"] = pd.cut(
            out["goal_probability"],
            bins=[0, 0.15, 0.25, 0.35, 1.0],
            labels=["Low", "Medium", "High", "Elite"],
        )
        out["expected_goals"] = out["goal_probability"]
        return out.sort_values("goal_probability", ascending=False).reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        fi = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi["importance_pct"] = fi["importance"] / fi["importance"].sum() * 100
        return fi

    def save(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Nothing to save.")
        joblib.dump(self.model,         MODEL_PATH)
        joblib.dump(self.scaler,        SCALER_PATH)
        joblib.dump(self.feature_names, FEATURES_PATH)
        joblib.dump(self.metrics,       METRICS_PATH)
        print(f"[Model] Saved to {MODEL_DIR}/")

    def load(self) -> bool:
        try:
            self.model         = joblib.load(MODEL_PATH)
            self.scaler        = joblib.load(SCALER_PATH)
            self.feature_names = joblib.load(FEATURES_PATH)
            self.is_trained    = True
            try:
                self.metrics = joblib.load(METRICS_PATH)
                # If metrics show AUC=0, the cache is stale — signal for retrain
                if self.metrics.get("train_auc", 0) == 0.0:
                    print("[Model] Cached metrics show AUC=0 — will retrain")
                    self.is_trained = False
                    return False
            except FileNotFoundError:
                print("[Model] Metrics file missing — will retrain")
                self.is_trained = False
                return False
            print(f"[Model] Loaded from cache. CV-AUC={self.metrics.get('train_auc',0):.3f}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[Model] Load error: {e}")
            return False

    def is_saved(self) -> bool:
        return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH])
