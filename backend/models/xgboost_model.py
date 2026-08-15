"""
XGBoost Traffic Forecasting Model — Classical ML Baseline.

Fast, interpretable gradient-boosted tree model for traffic speed prediction.
Uses tabular features (lag, temporal, rolling, spatial).
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import xgboost as xgb

from backend.models.base_model import BaseTrafficModel
from backend.config import XGBOOST_PARAMS

logger = logging.getLogger(__name__)


class XGBoostTrafficModel(BaseTrafficModel):
    """XGBoost regression model for traffic speed forecasting."""

    def __init__(self, forecast_horizon: int = 2, params: Optional[Dict] = None):
        super().__init__("xgboost", forecast_horizon)
        self.params = params or XGBOOST_PARAMS.copy()
        self.config = {"params": self.params}
        self.model = None
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X_train: Feature matrix (n_samples, n_features) — numpy or DataFrame
            y_train: Target vector (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training XGBoost: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Store feature names if DataFrame
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)

        start = time.time()

        self.model = xgb.XGBRegressor(**self.params)

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            verbose=kwargs.get("verbose", False),
        )

        self.training_time = round(time.time() - start, 2)
        self.is_trained = True

        # Get training history
        results = {}
        try:
            evals = self.model.evals_result()
            if 'validation_0' in evals:
                results["val_rmse"] = evals['validation_0'].get('rmse', [])
        except Exception:
            pass  # No eval_set was used during training

        results["training_time_seconds"] = self.training_time
        results["n_features"] = X_train.shape[1]
        results["n_samples"] = X_train.shape[0]

        logger.info(f"XGBoost training complete in {self.training_time}s")
        return results

    def predict(self, X) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_multi_step(self, X_initial, n_steps: int,
                           update_func=None) -> np.ndarray:
        """
        Multi-step recursive forecasting.

        For each step, predict the next value and update lag features.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        predictions = []
        current_X = X_initial.copy()

        for step in range(n_steps):
            pred = self.model.predict(current_X.reshape(1, -1))[0]
            predictions.append(pred)

            # Update features for next step (shift lags)
            if update_func:
                current_X = update_func(current_X, pred)

        return np.array(predictions)

    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        if self.feature_names:
            imp_dict = dict(zip(self.feature_names, importance))
        else:
            imp_dict = {f"f_{i}": v for i, v in enumerate(importance)}

        # Sort and take top N
        sorted_imp = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return sorted_imp

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to pickle."""
        path = path or self.get_default_save_path()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "config": self.config,
            "forecast_horizon": self.forecast_horizon,
            "training_time": self.training_time,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"XGBoost model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load model from pickle."""
        path = path or self.get_default_save_path()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.feature_names = state.get("feature_names")
        self.config = state.get("config", {})
        self.forecast_horizon = state.get("forecast_horizon", 2)
        self.training_time = state.get("training_time")
        self.is_trained = True

        logger.info(f"XGBoost model loaded from {path}")
