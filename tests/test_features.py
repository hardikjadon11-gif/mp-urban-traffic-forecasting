"""Tests for feature engineering."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.features.engineering import (
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    prepare_xgboost_features,
    prepare_sequence_data,
    prepare_stgcn_data,
)


def _make_data(n=200, n_sensors=5):
    np.random.seed(42)
    ts = pd.date_range("2024-01-01 08:00", periods=n, freq="15min")
    data = np.random.uniform(20, 70, (n, n_sensors))
    return pd.DataFrame(data, index=ts, columns=[f"S{i}" for i in range(n_sensors)])


class TestTemporalFeatures:
    def test_output_columns(self):
        df = _make_data()
        features = create_temporal_features(df)
        expected = ["hour", "minute", "day_of_week", "is_weekend", "month",
                     "is_peak", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        for col in expected:
            assert col in features.columns

    def test_peak_hours(self):
        df = _make_data()
        features = create_temporal_features(df)
        # 8 AM should be peak
        assert features.iloc[0]["is_peak"] == 1

    def test_cyclical_range(self):
        df = _make_data()
        features = create_temporal_features(df)
        assert features["hour_sin"].min() >= -1.01
        assert features["hour_sin"].max() <= 1.01


class TestLagFeatures:
    def test_default_lags(self):
        series = _make_data()["S0"]
        lags = create_lag_features(series)
        assert "lag_1" in lags.columns
        assert "lag_12" in lags.columns
        assert len(lags) == len(series)


class TestRollingFeatures:
    def test_default_windows(self):
        series = _make_data()["S0"]
        rolling = create_rolling_features(series)
        assert "rolling_mean_4" in rolling.columns
        assert "rolling_std_12" in rolling.columns


class TestXGBoostFeatures:
    def test_output_shape(self):
        df = _make_data()
        X, y = prepare_xgboost_features(df, target_sensor="S0")
        assert len(X) == len(y)
        assert X.shape[1] > 5  # Should have multiple features
        assert not X.isnull().any().any()


class TestSequenceData:
    def test_output_shapes(self):
        df = _make_data(n=100, n_sensors=3)
        X, y = prepare_sequence_data(df, seq_length=12, forecast_horizon=2, target_sensor="S0")
        assert X.ndim == 3  # (samples, seq_len, features)
        assert X.shape[1] == 12
        assert y.shape[1] == 2


class TestSTGCNData:
    def test_output_shapes(self):
        df = _make_data(n=100, n_sensors=5)
        X, y = prepare_stgcn_data(df, seq_length=12, forecast_horizon=2)
        assert X.ndim == 4  # (samples, nodes, 1, seq_len)
        assert X.shape[1] == 5  # n_sensors
        assert X.shape[2] == 1  # features
        assert X.shape[3] == 12  # seq_len
        assert y.shape[1] == 5  # n_sensors
        assert y.shape[2] == 2  # horizon
