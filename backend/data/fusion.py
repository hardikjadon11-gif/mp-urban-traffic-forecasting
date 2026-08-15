"""
Multi-source data fusion pipeline.

Combines data from METR-LA, PeMS-BAY, and Uber Movement into a
unified feature matrix following the conceptual pipeline:

  STEP 1 — Temporal Alignment  (resample to common 15-min interval)
  STEP 2 — Spatial Alignment   (map sensors to graph nodes via coordinates)
  STEP 3 — Normalization       (scale each source independently)
  STEP 4 — Feature Concatenation
  STEP 5 — Weighted Combination (configurable source weights)
"""

import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.config import (
    METR_LA_WEIGHT, PEMS_BAY_WEIGHT, UBER_WEIGHT,
    RESAMPLE_INTERVAL_MINUTES
)

logger = logging.getLogger(__name__)


class DataFusionPipeline:
    """
    Multi-source data fusion pipeline.
    Produces a unified feature matrix from heterogeneous traffic sources.
    """

    def __init__(self, metr_weight: float = METR_LA_WEIGHT,
                 pems_weight: float = PEMS_BAY_WEIGHT,
                 uber_weight: float = UBER_WEIGHT,
                 resample_minutes: int = RESAMPLE_INTERVAL_MINUTES):
        self.metr_weight = metr_weight
        self.pems_weight = pems_weight
        self.uber_weight = uber_weight
        self.resample_minutes = resample_minutes
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.fusion_status: Dict[str, Any] = {
            "step_1_temporal": "pending",
            "step_2_spatial": "pending",
            "step_3_normalization": "pending",
            "step_4_concatenation": "pending",
            "step_5_weighted": "pending",
            "sources_used": [],
        }

    def fuse(self,
             metr_la_df: Optional[pd.DataFrame] = None,
             pems_bay_df: Optional[pd.DataFrame] = None,
             uber_df: Optional[pd.DataFrame] = None,
             metr_sensors: Optional[pd.DataFrame] = None,
             pems_sensors: Optional[pd.DataFrame] = None,
             ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Execute the full fusion pipeline.

        Returns:
            unified_df: Unified speed/feature DataFrame
            all_sensors: Combined sensor metadata
            combined_adj: Combined adjacency matrix
        """
        sources = {}

        # ── STEP 1: Temporal Alignment ──────────────────────────────────
        logger.info("Fusion Step 1: Temporal alignment")
        if metr_la_df is not None and len(metr_la_df) > 0:
            sources["metr_la"] = self._temporal_align(metr_la_df, "METR-LA")
        if pems_bay_df is not None and len(pems_bay_df) > 0:
            sources["pems_bay"] = self._temporal_align(pems_bay_df, "PeMS-BAY")

        # Uber Movement is at hourly level — aggregate to zone-level features
        uber_features = None
        if uber_df is not None and len(uber_df) > 0:
            uber_features = self._process_uber(uber_df)

        self.fusion_status["step_1_temporal"] = "completed"
        self.fusion_status["sources_used"] = list(sources.keys())
        if uber_features is not None:
            self.fusion_status["sources_used"].append("uber")

        if not sources:
            raise ValueError("No traffic speed data available for fusion")

        # ── STEP 2: Spatial Alignment ───────────────────────────────────
        logger.info("Fusion Step 2: Spatial alignment")
        all_sensors = self._spatial_align(metr_sensors, pems_sensors)
        self.fusion_status["step_2_spatial"] = "completed"

        # ── STEP 3: Normalization ───────────────────────────────────────
        logger.info("Fusion Step 3: Normalization")
        normalized_sources = {}
        for name, df in sources.items():
            normalized_sources[name] = self._normalize_source(df, name)
        self.fusion_status["step_3_normalization"] = "completed"

        # ── STEP 4: Feature Concatenation ───────────────────────────────
        logger.info("Fusion Step 4: Feature concatenation")
        concatenated = self._concatenate(normalized_sources, uber_features)
        self.fusion_status["step_4_concatenation"] = "completed"

        # ── STEP 5: Weighted Combination ────────────────────────────────
        logger.info("Fusion Step 5: Weighted combination")
        unified = self._weighted_combine(concatenated, sources)
        self.fusion_status["step_5_weighted"] = "completed"

        # Build combined adjacency matrix
        n_total = len(unified.columns)
        combined_adj = self._build_combined_adjacency(n_total, all_sensors)

        logger.info(f"Fusion complete: {unified.shape[0]} timestamps × {unified.shape[1]} sensors")
        return unified, all_sensors, combined_adj

    def get_status(self) -> Dict[str, Any]:
        """Return current fusion pipeline status."""
        return {
            **self.fusion_status,
            "weights": {
                "metr_la": self.metr_weight,
                "pems_bay": self.pems_weight,
                "uber": self.uber_weight,
            },
            "resample_minutes": self.resample_minutes,
        }

    # ── Step Implementations ────────────────────────────────────────────

    def _temporal_align(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Resample to common interval."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        resampled = df.resample(f"{self.resample_minutes}min").mean()
        resampled = resampled.ffill().bfill()
        logger.info(f"  {source_name}: {len(df)} → {len(resampled)} rows ({self.resample_minutes}min)")
        return resampled

    def _process_uber(self, uber_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process Uber Movement data into zone-level travel time features.
        Aggregate to common time intervals.
        """
        if uber_df is None or len(uber_df) == 0:
            return None

        try:
            if "timestamp" in uber_df.columns:
                uber_df["timestamp"] = pd.to_datetime(uber_df["timestamp"])
                # Aggregate: mean travel time per timestamp
                agg = uber_df.groupby("timestamp")["mean_travel_time_minutes"].mean()
                agg = agg.resample(f"{self.resample_minutes}min").mean().ffill().bfill()
                return agg.to_frame(name="uber_travel_time")
        except Exception as e:
            logger.warning(f"  Failed to process Uber data: {e}")
        return None

    def _spatial_align(self, metr_sensors: Optional[pd.DataFrame],
                       pems_sensors: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Combine sensor metadata from multiple sources."""
        dfs = []
        if metr_sensors is not None and len(metr_sensors) > 0:
            metr_sensors = metr_sensors.copy()
            if "source" not in metr_sensors.columns:
                metr_sensors["source"] = "metr-la"
            dfs.append(metr_sensors)

        if pems_sensors is not None and len(pems_sensors) > 0:
            pems_sensors = pems_sensors.copy()
            if "source" not in pems_sensors.columns:
                pems_sensors["source"] = "pems-bay"
            dfs.append(pems_sensors)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=["sensor_id", "latitude", "longitude", "source"])

        logger.info(f"  Combined {len(combined)} sensor locations")
        return combined

    def _normalize_source(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Normalize a source independently (MinMax to [0,1])."""
        scaler = MinMaxScaler()
        values = scaler.fit_transform(df.values)
        self.scalers[source_name] = scaler
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def _concatenate(self, normalized_sources: Dict[str, pd.DataFrame],
                     uber_features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate all sources by columns (sensors), aligning on time index."""
        dfs = list(normalized_sources.values())

        if uber_features is not None:
            # Align Uber features to the same time index
            common_index = dfs[0].index if dfs else uber_features.index
            uber_aligned = uber_features.reindex(common_index, method="nearest")
            dfs.append(uber_aligned)

        # Find common time range
        if len(dfs) > 1:
            common_start = max(df.index.min() for df in dfs)
            common_end = min(df.index.max() for df in dfs)
            dfs = [df.loc[common_start:common_end] for df in dfs]

        concatenated = pd.concat(dfs, axis=1)
        concatenated = concatenated.ffill().bfill()
        logger.info(f"  Concatenated shape: {concatenated.shape}")
        return concatenated

    def _weighted_combine(self, concatenated: pd.DataFrame,
                          sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply configurable weights to each source's columns.
        This allows tuning the relative importance of each data source.
        """
        # For now, since sources have different sensor columns,
        # weighting is applied per-source-block
        result = concatenated.copy()

        col_idx = 0
        weights_map = {
            "metr_la": self.metr_weight,
            "pems_bay": self.pems_weight,
        }

        for name, df in sources.items():
            n_cols = len(df.columns)
            weight = weights_map.get(name, 1.0)
            end_idx = col_idx + n_cols
            if end_idx <= len(result.columns):
                result.iloc[:, col_idx:end_idx] *= weight
            col_idx = end_idx

        # Uber columns (if present)
        if col_idx < len(result.columns):
            remaining = len(result.columns) - col_idx
            result.iloc[:, col_idx:] *= self.uber_weight

        return result

    def _build_combined_adjacency(self, n_total: int,
                                  all_sensors: pd.DataFrame) -> np.ndarray:
        """Build a combined adjacency matrix for all sensors."""
        from backend.utils.graph import build_adjacency_from_coordinates

        if len(all_sensors) == n_total and "latitude" in all_sensors.columns:
            return build_adjacency_from_coordinates(
                all_sensors["latitude"].values,
                all_sensors["longitude"].values,
                threshold_km=10.0
            )
        return np.eye(n_total)
