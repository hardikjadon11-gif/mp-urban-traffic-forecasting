"""
Data preprocessing pipeline for traffic speed time-series.

Pipeline steps:
  1. Missing-value handling (forward-fill → backward-fill → median)
  2. Duplicate timestamp removal
  3. Timestamp parsing & sorting
  4. Resampling to target interval
  5. Outlier handling (IQR-based capping)
  6. Normalization (fit on training data only to prevent leakage)
  7. Chronological train/val/test splitting
"""

import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from backend.config import (
    RESAMPLE_INTERVAL_MINUTES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

logger = logging.getLogger(__name__)


class TrafficPreprocessor:
    """
    Stateful preprocessor for traffic speed data.
    Stores normalization parameters fitted on training data.
    """

    def __init__(self, resample_minutes: int = RESAMPLE_INTERVAL_MINUTES,
                 normalization: str = "minmax"):
        self.resample_minutes = resample_minutes
        self.normalization = normalization  # "minmax" or "standard"
        self.scaler = None
        self.is_fitted = False
        self.stats: Dict[str, Any] = {}

    def full_pipeline(self, speed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.
        Returns: (train_df, val_df, test_df) — all normalized.
        """
        logger.info(f"Starting preprocessing: {speed_df.shape[0]} timestamps × {speed_df.shape[1]} sensors")

        # Step 1: Parse timestamps & sort
        df = self._parse_and_sort(speed_df)

        # Step 2: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 3: Handle missing values
        df = self._handle_missing(df)

        # Step 4: Resample
        df = self._resample(df)

        # Step 5: Handle outliers
        df = self._handle_outliers(df)

        # Step 6: Chronological split (before normalization!)
        train_df, val_df, test_df = self._chronological_split(df)

        # Step 7: Normalize (fit on train only)
        train_df, val_df, test_df = self._normalize(train_df, val_df, test_df)

        self._log_stats(df, train_df, val_df, test_df)
        return train_df, val_df, test_df

    def preprocess_without_split(self, speed_df: pd.DataFrame) -> pd.DataFrame:
        """Run preprocessing without splitting (for inference / demo)."""
        df = self._parse_and_sort(speed_df)
        df = self._remove_duplicates(df)
        df = self._handle_missing(df)
        df = self._resample(df)
        df = self._handle_outliers(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted normalization to new data."""
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted. Returning data as-is.")
            return df
        values = self.scaler.transform(df.values)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse normalization to get original-scale values."""
        if not self.is_fitted:
            return df
        values = self.scaler.inverse_transform(df.values)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def inverse_transform_array(self, arr: np.ndarray, n_sensors: int) -> np.ndarray:
        """Reverse normalization for a numpy array."""
        if not self.is_fitted:
            return arr
        # Handle different shapes
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != n_sensors:
            # Pad or truncate to match scaler
            padded = np.zeros((arr.shape[0], n_sensors))
            padded[:, :arr.shape[1]] = arr[:, :n_sensors]
            result = self.scaler.inverse_transform(padded)
            return result[:, :original_shape[-1]] if len(original_shape) == 1 else result
        return self.scaler.inverse_transform(arr).reshape(original_shape)

    # ── Pipeline Steps ──────────────────────────────────────────────────

    def _parse_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex and sort chronologically."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        logger.info(f"  Parsed & sorted: {df.index[0]} → {df.index[-1]}")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the first occurrence."""
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"  Removed {n_removed} duplicate timestamps")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values:
          1. Forward-fill (propagate last known value)
          2. Backward-fill (for leading NaN gaps)
          3. Fill remaining with column median
        """
        missing_before = df.isnull().sum().sum()
        df = df.ffill()
        df = df.bfill()
        # Any remaining NaNs get column median
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
        missing_after = df.isnull().sum().sum()
        logger.info(f"  Missing values: {missing_before} → {missing_after}")
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample to target interval using mean aggregation."""
        original_len = len(df)
        df = df.resample(f"{self.resample_minutes}min").mean()
        # Handle any NaNs from resampling
        df = df.ffill().bfill()
        logger.info(f"  Resampled {original_len} → {len(df)} rows ({self.resample_minutes}min interval)")
        return df

    def _handle_outliers(self, df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
        """
        Cap outliers using the IQR method.
        Values beyond Q1 - factor*IQR or Q3 + factor*IQR are clipped.
        """
        n_capped = 0
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            n_capped += mask.sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped > 0:
            logger.info(f"  Capped {n_capped} outlier values (IQR×{iqr_factor})")
        return df

    def _chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no shuffling!).
        Earlier period → Training
        Middle period  → Validation
        Later period   → Testing
        """
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(f"  Chronological split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    def _normalize(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize data. Scaler is fit ONLY on training data to prevent leakage.
        """
        if self.normalization == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit on training data only
        self.scaler.fit(train_df.values)
        self.is_fitted = True

        # Transform all splits
        train_norm = pd.DataFrame(
            self.scaler.transform(train_df.values),
            index=train_df.index, columns=train_df.columns
        )
        val_norm = pd.DataFrame(
            self.scaler.transform(val_df.values),
            index=val_df.index, columns=val_df.columns
        )
        test_norm = pd.DataFrame(
            self.scaler.transform(test_df.values),
            index=test_df.index, columns=test_df.columns
        )

        logger.info(f"  Normalized using {self.normalization} (fitted on training data only)")
        return train_norm, val_norm, test_norm

    def _log_stats(self, original_df, train_df, val_df, test_df):
        """Store preprocessing statistics."""
        self.stats = {
            "original_shape": list(original_df.shape),
            "train_shape": list(train_df.shape),
            "val_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "resample_interval": self.resample_minutes,
            "normalization": self.normalization,
            "train_range": [str(train_df.index[0]), str(train_df.index[-1])],
            "val_range": [str(val_df.index[0]), str(val_df.index[-1])],
            "test_range": [str(test_df.index[0]), str(test_df.index[-1])],
        }
