"""
Evaluation metrics for traffic forecasting models.

Metrics:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
"""

import time
import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error.
    Measures the average magnitude of prediction errors.
    """
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    Penalizes larger errors more heavily than MAE.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Expresses error as a percentage of actual values.

    Uses epsilon to avoid division by zero.
    """
    # Filter out near-zero actual values to avoid inflated MAPE
    mask = np.abs(actual) > epsilon
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Compute all three metrics at once.

    Returns:
        Dict with 'mae', 'rmse', 'mape' keys.
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    return {
        "mae": round(mae(actual, predicted), 4),
        "rmse": round(rmse(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 2),
    }


def evaluate_model(model, X_test, y_test, **predict_kwargs) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a model.

    Args:
        model: Trained model with predict() method.
        X_test: Test input data.
        y_test: Test ground truth.

    Returns:
        Dict with metrics + inference time.
    """
    # Measure inference time
    start = time.time()
    predictions = model.predict(X_test, **predict_kwargs)
    inference_time_ms = (time.time() - start) * 1000

    # Flatten arrays for metric computation
    actual = np.array(y_test).flatten()
    predicted = np.array(predictions).flatten()

    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    metrics = evaluate_predictions(actual, predicted)
    metrics["inference_time_ms"] = round(inference_time_ms, 2)
    metrics["num_test_samples"] = len(actual)
    metrics["model_name"] = model.model_name

    logger.info(
        f"Evaluation [{model.model_name}]: "
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"MAPE={metrics['mape']:.2f}%, Inference={metrics['inference_time_ms']:.1f}ms"
    )

    return metrics


def naive_baseline(y_test: np.ndarray) -> Dict[str, float]:
    """
    Naive 'last known value' baseline.
    Predicts the previous value for each time step.
    Useful as a sanity-check baseline.
    """
    actual = y_test.flatten()
    predicted = np.roll(actual, 1)
    predicted[0] = actual[0]

    return {
        "model_name": "naive_baseline",
        **evaluate_predictions(actual, predicted),
    }
