"""
Abstract base class for all traffic forecasting models.
Defines the common interface for training, prediction, evaluation,
and persistence (save/load).
"""

import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

from backend.config import MODELS_DIR

logger = logging.getLogger(__name__)


class BaseTrafficModel(ABC):
    """Abstract base for XGBoost, LSTM, and STGCN models."""

    def __init__(self, model_name: str, forecast_horizon: int = 2):
        self.model_name = model_name
        self.forecast_horizon = forecast_horizon
        self.is_trained = False
        self.training_time = None  # seconds
        self.training_history: list = []
        self.config: Dict[str, Any] = {}

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Dict with training metadata (loss history, time, etc.)
        """
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model weights/state to disk."""
        pass

    @abstractmethod
    def load(self, path: Optional[Path] = None) -> None:
        """Load model weights/state from disk."""
        pass

    def get_default_save_path(self) -> Path:
        """Get default save path for this model."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return MODELS_DIR / f"{self.model_name}_h{self.forecast_horizon}.pkl"

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "forecast_horizon": self.forecast_horizon,
            "is_trained": self.is_trained,
            "training_time_seconds": self.training_time,
            "config": self.config,
        }

    def _time_training(self, func, *args, **kwargs):
        """Utility to time training execution."""
        start = time.time()
        result = func(*args, **kwargs)
        self.training_time = round(time.time() - start, 2)
        self.is_trained = True
        return result
