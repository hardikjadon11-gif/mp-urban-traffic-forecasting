"""Tests for data preprocessing pipeline."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.preprocessing import TrafficPreprocessor


def _make_sample_data(n_timestamps=200, n_sensors=5, seed=42):
    """Create sample traffic speed data."""
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_timestamps, freq="5min")
    data = np.random.uniform(20, 70, (n_timestamps, n_sensors))
    columns = [f"S_{i}" for i in range(n_sensors)]
    return pd.DataFrame(data, index=timestamps, columns=columns)


class TestPreprocessor:
    def test_full_pipeline_returns_three_splits(self):
        df = _make_sample_data()
        pp = TrafficPreprocessor(resample_minutes=15)
        train, val, test = pp.full_pipeline(df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) > 0

    def test_chronological_order_preserved(self):
        df = _make_sample_data()
        pp = TrafficPreprocessor(resample_minutes=15)
        train, val, test = pp.full_pipeline(df)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_no_nans_after_preprocessing(self):
        df = _make_sample_data()
        # Inject NaNs
        df.iloc[10:15, 0] = np.nan
        df.iloc[50, :] = np.nan
        pp = TrafficPreprocessor(resample_minutes=15)
        train, val, test = pp.full_pipeline(df)
        assert train.isnull().sum().sum() == 0
        assert val.isnull().sum().sum() == 0
        assert test.isnull().sum().sum() == 0

    def test_normalization_range(self):
        df = _make_sample_data()
        pp = TrafficPreprocessor(resample_minutes=15, normalization="minmax")
        train, val, test = pp.full_pipeline(df)
        # Train data should be in [0, 1] range (MinMax)
        assert train.values.min() >= -0.01
        assert train.values.max() <= 1.01

    def test_inverse_transform(self):
        df = _make_sample_data(n_timestamps=100, n_sensors=3)
        pp = TrafficPreprocessor(resample_minutes=15)
        train, val, test = pp.full_pipeline(df)
        restored = pp.inverse_transform(train)
        # Should be back in original-scale range
        assert restored.values.max() > 1.0

    def test_duplicate_removal(self):
        df = _make_sample_data(n_timestamps=100)
        # Add duplicates
        dup = df.iloc[:5].copy()
        df = pd.concat([df, dup])
        pp = TrafficPreprocessor(resample_minutes=15)
        result = pp.preprocess_without_split(df)
        assert not result.index.duplicated().any()

    def test_resampling(self):
        df = _make_sample_data(n_timestamps=100)  # 5-min intervals
        pp = TrafficPreprocessor(resample_minutes=15)
        result = pp.preprocess_without_split(df)
        diffs = pd.Series(result.index).diff().dropna()
        median_minutes = diffs.median().total_seconds() / 60
        assert abs(median_minutes - 15) < 1
