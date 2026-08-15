"""Tests for evaluation metrics."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.evaluation.metrics import mae, rmse, mape, evaluate_predictions, naive_baseline


class TestMAE:
    def test_perfect_prediction(self):
        actual = np.array([1, 2, 3, 4, 5])
        assert mae(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([1, 2, 3])
        predicted = np.array([2, 3, 4])
        assert abs(mae(actual, predicted) - 1.0) < 1e-6

    def test_symmetric(self):
        actual = np.array([1, 2, 3])
        predicted = np.array([3, 4, 5])
        assert mae(actual, predicted) == mae(predicted, actual)


class TestRMSE:
    def test_perfect_prediction(self):
        actual = np.array([1, 2, 3])
        assert rmse(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([1, 2, 3])
        predicted = np.array([2, 3, 4])
        assert abs(rmse(actual, predicted) - 1.0) < 1e-6

    def test_rmse_geq_mae(self):
        actual = np.array([1, 5, 10, 15, 20])
        predicted = np.array([2, 7, 8, 18, 22])
        assert rmse(actual, predicted) >= mae(actual, predicted)


class TestMAPE:
    def test_perfect_prediction(self):
        actual = np.array([10, 20, 30])
        assert mape(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([90.0, 180.0])
        assert abs(mape(actual, predicted) - 10.0) < 1e-6

    def test_zero_protection(self):
        actual = np.array([0.0, 0.0, 10.0])
        predicted = np.array([1.0, 1.0, 12.0])
        # Should not crash with zeros
        result = mape(actual, predicted)
        assert result >= 0


class TestEvaluatePredictions:
    def test_returns_all_metrics(self):
        actual = np.array([10, 20, 30, 40, 50])
        predicted = np.array([12, 22, 28, 38, 52])
        result = evaluate_predictions(actual, predicted)
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert all(v >= 0 for v in result.values())


class TestNaiveBaseline:
    def test_returns_metrics(self):
        y_test = np.array([10, 11, 12, 13, 14])
        result = naive_baseline(y_test)
        assert "mae" in result
        assert "model_name" in result
        assert result["model_name"] == "naive_baseline"
