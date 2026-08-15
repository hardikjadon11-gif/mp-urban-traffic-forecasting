"""
Feature engineering for traffic forecasting.

Generates features including:
  - Temporal: hour, minute, day_of_week, is_weekend, month, is_peak
  - Lag: lag_1..lag_12 (historical values at prior time steps)
  - Rolling: rolling_mean, rolling_std (over configurable windows)
  - Spatial: neighbor_mean_speed (from adjacency matrix)
"""

import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from backend.config import SEQUENCE_LENGTH

logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the DatetimeIndex.

    Features: hour, minute, day_of_week, is_weekend, month, is_peak_hour
    """
    features = pd.DataFrame(index=df.index)
    features["hour"] = df.index.hour
    features["minute"] = df.index.minute
    features["day_of_week"] = df.index.dayofweek
    features["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    features["month"] = df.index.month

    # Peak hours: 7-9 AM and 5-7 PM (configurable)
    hour = df.index.hour
    features["is_peak"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(int)

    # Cyclical encoding for hour and day (helps models capture periodicity)
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    return features


def create_lag_features(series: pd.Series, lags: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Create lag features for a single sensor's time series.

    Args:
        series: Traffic speed series for one sensor.
        lags: List of lag steps. Default: [1,2,3,4,6,8,12]
              At 15-min intervals: lag_12 = 3 hours lookback.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 6, 8, 12]

    lag_df = pd.DataFrame(index=series.index)
    for lag in lags:
        lag_df[f"lag_{lag}"] = series.shift(lag)

    return lag_df


def create_rolling_features(series: pd.Series,
                            windows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Create rolling statistics features.

    Args:
        series: Traffic speed series for one sensor.
        windows: Rolling window sizes. Default: [4, 8, 12]
                 At 15-min intervals: window_4 = 1 hour.
    """
    if windows is None:
        windows = [4, 8, 12]

    rolling_df = pd.DataFrame(index=series.index)
    for w in windows:
        rolling_df[f"rolling_mean_{w}"] = series.rolling(window=w, min_periods=1).mean()
        rolling_df[f"rolling_std_{w}"] = series.rolling(window=w, min_periods=1).std().fillna(0)

    return rolling_df


def create_spatial_features(speed_df: pd.DataFrame,
                            adj_matrix: np.ndarray) -> pd.DataFrame:
    """
    Create spatial features: mean speed of neighboring sensors.

    Args:
        speed_df: Full sensor speed DataFrame.
        adj_matrix: Adjacency matrix (N_sensors × N_sensors).

    Returns:
        DataFrame with neighbor_mean columns for each sensor.
    """
    n_sensors = len(speed_df.columns)
    if adj_matrix.shape[0] != n_sensors:
        logger.warning("Adjacency matrix size mismatch. Skipping spatial features.")
        return pd.DataFrame(index=speed_df.index)

    spatial_df = pd.DataFrame(index=speed_df.index)
    values = speed_df.values  # (T, N)

    for i in range(n_sensors):
        neighbors = adj_matrix[i] > 0
        if neighbors.any():
            neighbor_mean = values[:, neighbors].mean(axis=1)
        else:
            neighbor_mean = values[:, i]  # Use own value if no neighbors
        spatial_df[f"{speed_df.columns[i]}_neighbor_mean"] = neighbor_mean

    return spatial_df


def prepare_xgboost_features(speed_df: pd.DataFrame,
                             adj_matrix: Optional[np.ndarray] = None,
                             target_sensor: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a flat feature matrix for XGBoost.

    For a single sensor or aggregated across sensors:
      - Temporal features
      - Lag features
      - Rolling features
      - Spatial features (neighbor mean)

    Returns: (X, y) ready for XGBoost training.
    """
    if target_sensor and target_sensor in speed_df.columns:
        target_series = speed_df[target_sensor]
    else:
        # Use mean across all sensors
        target_series = speed_df.mean(axis=1)

    # Temporal
    temporal = create_temporal_features(speed_df)

    # Lag
    lags = create_lag_features(target_series)

    # Rolling
    rolling = create_rolling_features(target_series)

    # Spatial (neighbor mean for target sensor)
    spatial = pd.DataFrame(index=speed_df.index)
    if adj_matrix is not None and target_sensor in speed_df.columns:
        sensor_idx = list(speed_df.columns).index(target_sensor)
        neighbors = adj_matrix[sensor_idx] > 0
        if neighbors.any():
            spatial["neighbor_mean"] = speed_df.values[:, neighbors].mean(axis=1)
            spatial["neighbor_std"] = speed_df.values[:, neighbors].std(axis=1)

    # Combine
    X = pd.concat([temporal, lags, rolling, spatial], axis=1)
    y = target_series.copy()
    y.name = "target_speed"

    # Drop rows with NaN from lag features
    valid = X.dropna().index
    X = X.loc[valid]
    y = y.loc[valid]

    logger.info(f"XGBoost features: {X.shape[1]} features, {len(X)} samples")
    return X, y


def prepare_sequence_data(speed_df: pd.DataFrame,
                          seq_length: int = SEQUENCE_LENGTH,
                          forecast_horizon: int = 2,
                          target_sensor: Optional[str] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data for LSTM/STGCN.

    Creates sliding-window sequences:
      X: (num_samples, seq_length, n_features)
      y: (num_samples, forecast_horizon)

    Args:
        speed_df: Preprocessed speed DataFrame.
        seq_length: Number of historical time steps as input.
        forecast_horizon: Number of future time steps to predict.
        target_sensor: If specified, predict this sensor only.
    """
    if target_sensor and target_sensor in speed_df.columns:
        values = speed_df[target_sensor].values.reshape(-1, 1)
    else:
        values = speed_df.values  # (T, N_sensors)

    n_timestamps = len(values)
    n_features = values.shape[1]

    X, y = [], []
    for i in range(n_timestamps - seq_length - forecast_horizon + 1):
        X.append(values[i:i + seq_length])
        if target_sensor:
            y.append(values[i + seq_length:i + seq_length + forecast_horizon, 0])
        else:
            # For STGCN: predict all sensors
            y.append(values[i + seq_length:i + seq_length + forecast_horizon])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Sequence data: X={X.shape}, y={y.shape} (seq_len={seq_length}, horizon={forecast_horizon})")
    return X, y


def prepare_stgcn_data(speed_df: pd.DataFrame,
                       seq_length: int = SEQUENCE_LENGTH,
                       forecast_horizon: int = 2
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data specifically for STGCN.

    Output shapes:
      X: (num_samples, n_sensors, 1, seq_length)  — [batch, nodes, features, time]
      y: (num_samples, n_sensors, forecast_horizon)
    """
    values = speed_df.values  # (T, N)
    n_timestamps, n_sensors = values.shape

    X, y = [], []
    for i in range(n_timestamps - seq_length - forecast_horizon + 1):
        # Input: (N, 1, seq_length)
        x_i = values[i:i + seq_length].T  # (N, seq_length)
        x_i = x_i[:, np.newaxis, :]  # (N, 1, seq_length)
        X.append(x_i)

        # Target: (N, horizon)
        y_i = values[i + seq_length:i + seq_length + forecast_horizon].T  # (N, horizon)
        y.append(y_i)

    X = np.array(X)  # (samples, N, 1, seq_length)
    y = np.array(y)  # (samples, N, horizon)

    logger.info(f"STGCN data: X={X.shape}, y={y.shape}")
    return X, y
