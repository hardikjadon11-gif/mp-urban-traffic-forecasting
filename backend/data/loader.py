"""
Dataset loaders for METR-LA, PeMS-BAY, Uber Movement, and demo data.

Each loader returns data in a consistent format:
  - Speed data: pd.DataFrame with DatetimeIndex and sensor columns
  - Sensor metadata: pd.DataFrame with sensor_id, latitude, longitude
  - Adjacency: np.ndarray

Auto-falls back to demo data when real datasets are unavailable.
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from backend.config import (
    METR_LA_PATH, PEMS_BAY_PATH, UBER_MOVEMENT_PATH, DEMO_DATA_DIR
)

logger = logging.getLogger(__name__)


class DatasetInfo:
    """Metadata about a loaded dataset."""
    def __init__(self, name: str, source: str, n_sensors: int, n_records: int,
                 time_range: Tuple[str, str], interval_minutes: int,
                 missing_pct: float, is_demo: bool):
        self.name = name
        self.source = source
        self.n_sensors = n_sensors
        self.n_records = n_records
        self.time_range = time_range
        self.interval_minutes = interval_minutes
        self.missing_pct = missing_pct
        self.is_demo = is_demo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "n_sensors": self.n_sensors,
            "n_records": self.n_records,
            "time_range_start": self.time_range[0],
            "time_range_end": self.time_range[1],
            "interval_minutes": self.interval_minutes,
            "missing_pct": round(self.missing_pct, 2),
            "is_demo": self.is_demo,
        }


def _compute_dataset_info(name: str, source: str, df: pd.DataFrame,
                          is_demo: bool) -> DatasetInfo:
    """Compute metadata from a speed DataFrame."""
    if len(df) < 2:
        interval = 5
    else:
        diffs = pd.Series(df.index).diff().dropna()
        interval = int(diffs.median().total_seconds() / 60)

    missing_pct = (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0

    return DatasetInfo(
        name=name,
        source=source,
        n_sensors=len(df.columns),
        n_records=len(df),
        time_range=(str(df.index.min()), str(df.index.max())),
        interval_minutes=interval,
        missing_pct=missing_pct,
        is_demo=is_demo,
    )


# ── METR-LA ─────────────────────────────────────────────────────────────

def load_metr_la(path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """
    Load METR-LA dataset.
    Returns: (speed_df, sensor_locations, adjacency_matrix, info)
    Falls back to demo data if real data is unavailable.
    """
    path = Path(path) if path else METR_LA_PATH

    if path.exists():
        logger.info(f"Loading real METR-LA data from {path}")
        try:
            speed_df = pd.read_hdf(path, key="df")
            if not isinstance(speed_df.index, pd.DatetimeIndex):
                speed_df.index = pd.to_datetime(speed_df.index)
            # Try loading associated sensor/adj files
            data_dir = path.parent
            sensors = _load_sensors_csv(data_dir, "metr_la_sensors.csv", speed_df, "metr-la")
            adj = _load_adjacency(data_dir, "metr_la_adj.pkl", len(speed_df.columns))
            info = _compute_dataset_info("METR-LA", "metr-la", speed_df, is_demo=False)
            return speed_df, sensors, adj, info
        except Exception as e:
            logger.warning(f"Failed to load real METR-LA: {e}. Falling back to demo.")

    return _load_demo_metr_la()


def _load_demo_metr_la() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load demo METR-LA data."""
    demo_speed_path = DEMO_DATA_DIR / "demo_metr_la.h5"
    demo_sensor_path = DEMO_DATA_DIR / "metr_la_sensors.csv"
    demo_adj_path = DEMO_DATA_DIR / "metr_la_adj.pkl"

    if not demo_speed_path.exists():
        raise FileNotFoundError(
            f"Demo data not found at {demo_speed_path}. "
            "Run 'python scripts/generate_demo_data.py' to generate it."
        )

    speed_df = pd.read_hdf(demo_speed_path, key="df")
    if not isinstance(speed_df.index, pd.DatetimeIndex):
        speed_df.index = pd.to_datetime(speed_df.index)

    sensors = pd.read_csv(demo_sensor_path) if demo_sensor_path.exists() else \
        _generate_fallback_sensors(speed_df, "metr-la")

    if demo_adj_path.exists():
        with open(demo_adj_path, "rb") as f:
            adj = pickle.load(f)
    else:
        adj = np.eye(len(speed_df.columns))

    info = _compute_dataset_info("METR-LA (Demo)", "metr-la", speed_df, is_demo=True)
    logger.info(f"Loaded demo METR-LA: {info.n_sensors} sensors, {info.n_records} records")
    return speed_df, sensors, adj, info


# ── PeMS-BAY ────────────────────────────────────────────────────────────

def load_pems_bay(path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load PeMS-BAY dataset. Falls back to demo data."""
    path = Path(path) if path else PEMS_BAY_PATH

    if path.exists():
        logger.info(f"Loading real PeMS-BAY data from {path}")
        try:
            speed_df = pd.read_hdf(path, key="df")
            if not isinstance(speed_df.index, pd.DatetimeIndex):
                speed_df.index = pd.to_datetime(speed_df.index)
            data_dir = path.parent
            sensors = _load_sensors_csv(data_dir, "pems_bay_sensors.csv", speed_df, "pems-bay")
            adj = _load_adjacency(data_dir, "pems_bay_adj.pkl", len(speed_df.columns))
            info = _compute_dataset_info("PeMS-BAY", "pems-bay", speed_df, is_demo=False)
            return speed_df, sensors, adj, info
        except Exception as e:
            logger.warning(f"Failed to load real PeMS-BAY: {e}. Falling back to demo.")

    return _load_demo_pems_bay()


def _load_demo_pems_bay() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load demo PeMS-BAY data."""
    demo_speed_path = DEMO_DATA_DIR / "demo_pems_bay.h5"
    demo_sensor_path = DEMO_DATA_DIR / "pems_bay_sensors.csv"

    if not demo_speed_path.exists():
        raise FileNotFoundError(
            f"Demo data not found at {demo_speed_path}. "
            "Run 'python scripts/generate_demo_data.py' to generate it."
        )

    speed_df = pd.read_hdf(demo_speed_path, key="df")
    if not isinstance(speed_df.index, pd.DatetimeIndex):
        speed_df.index = pd.to_datetime(speed_df.index)

    sensors = pd.read_csv(demo_sensor_path) if demo_sensor_path.exists() else \
        _generate_fallback_sensors(speed_df, "pems-bay")

    adj = np.eye(len(speed_df.columns))  # Identity fallback
    info = _compute_dataset_info("PeMS-BAY (Demo)", "pems-bay", speed_df, is_demo=True)
    logger.info(f"Loaded demo PeMS-BAY: {info.n_sensors} sensors, {info.n_records} records")
    return speed_df, sensors, adj, info


# ── India MP (Madhya Pradesh) ──────────────────────────────────────────

def load_india_mp() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load India MP (Madhya Pradesh) traffic dataset."""
    demo_speed_path = DEMO_DATA_DIR / "demo_india_mp.h5"
    demo_sensor_path = DEMO_DATA_DIR / "india_mp_sensors.csv"
    demo_adj_path = DEMO_DATA_DIR / "india_mp_adj.pkl"

    if not demo_speed_path.exists():
        raise FileNotFoundError(f"India MP data not found at {demo_speed_path}")

    speed_df = pd.read_hdf(demo_speed_path, key="df")
    if not isinstance(speed_df.index, pd.DatetimeIndex):
        speed_df.index = pd.to_datetime(speed_df.index)

    sensors = pd.read_csv(demo_sensor_path) if demo_sensor_path.exists() else \
        _generate_fallback_sensors(speed_df, "india-mp")

    if demo_adj_path.exists():
        with open(demo_adj_path, "rb") as f:
            adj = pickle.load(f)
    else:
        adj = np.eye(len(speed_df.columns))

    info = _compute_dataset_info("India MP Traffic Network", "india-mp", speed_df, is_demo=True)
    logger.info(f"Loaded India MP dataset: {info.n_sensors} sensors (Bhopal, Indore, Gwalior, Jabalpur, Ujjain)")
    return speed_df, sensors, adj, info


# ── Uber Movement ───────────────────────────────────────────────────────

def load_uber_movement(path: Optional[Path] = None) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load Uber Movement travel-time data.
    Returns: (travel_times_df, info)
    """
    path = Path(path) if path else UBER_MOVEMENT_PATH

    if path.exists():
        logger.info(f"Loading real Uber Movement data from {path}")
        try:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            info = DatasetInfo(
                name="Uber Movement",
                source="uber",
                n_sensors=df["source_zone"].nunique() if "source_zone" in df.columns else 0,
                n_records=len(df),
                time_range=(str(df["timestamp"].min()), str(df["timestamp"].max())) if "timestamp" in df.columns else ("N/A", "N/A"),
                interval_minutes=60,
                missing_pct=0,
                is_demo=False,
            )
            return df, info
        except Exception as e:
            logger.warning(f"Failed to load Uber Movement: {e}. Falling back to demo.")

    return _load_demo_uber()


def _load_demo_uber() -> Tuple[pd.DataFrame, DatasetInfo]:
    """Load demo Uber Movement data."""
    demo_path = DEMO_DATA_DIR / "demo_uber_movement.csv"
    if not demo_path.exists():
        # Return empty DataFrame rather than crashing
        logger.warning("Demo Uber Movement data not found. Returning empty dataset.")
        df = pd.DataFrame(columns=["timestamp", "source_zone", "destination_zone", "mean_travel_time_minutes"])
        info = DatasetInfo("Uber Movement (Unavailable)", "uber", 0, 0, ("N/A", "N/A"), 60, 0, True)
        return df, info

    df = pd.read_csv(demo_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    info = DatasetInfo(
        name="Uber Movement (Demo)",
        source="uber",
        n_sensors=df["source_zone"].nunique() if "source_zone" in df.columns else 0,
        n_records=len(df),
        time_range=(str(df["timestamp"].min()), str(df["timestamp"].max())) if "timestamp" in df.columns else ("N/A", "N/A"),
        interval_minutes=60,
        missing_pct=0,
        is_demo=True,
    )
    return df, info


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_sensors_csv(data_dir: Path, filename: str,
                      speed_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Try loading a sensor CSV; generate fallback if missing."""
    path = data_dir / filename
    if path.exists():
        return pd.read_csv(path)
    return _generate_fallback_sensors(speed_df, source)


def _generate_fallback_sensors(speed_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Generate placeholder sensor metadata when CSV is missing."""
    n = len(speed_df.columns)
    return pd.DataFrame({
        "sensor_id": speed_df.columns.tolist(),
        "latitude": np.linspace(34.0, 34.15, n),
        "longitude": np.linspace(-118.4, -118.2, n),
        "source": source,
    })


def _load_adjacency(data_dir: Path, filename: str, n_sensors: int) -> np.ndarray:
    """Try loading adjacency; return identity if missing."""
    path = data_dir / filename
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    logger.warning(f"Adjacency matrix not found at {path}. Using identity.")
    return np.eye(n_sensors)


def get_all_dataset_info() -> list:
    """Get metadata for all available datasets (quick check, no full loading)."""
    infos = []

    # METR-LA
    try:
        _, _, _, info = load_metr_la()
        infos.append(info.to_dict())
    except Exception as e:
        infos.append({"name": "METR-LA", "source": "metr-la", "status": "unavailable", "error": str(e)})

    # PeMS-BAY
    try:
        _, _, _, info = load_pems_bay()
        infos.append(info.to_dict())
    except Exception as e:
        infos.append({"name": "PeMS-BAY", "source": "pems-bay", "status": "unavailable", "error": str(e)})

    # Uber Movement
    try:
        _, info = load_uber_movement()
        infos.append(info.to_dict())
    except Exception as e:
        infos.append({"name": "Uber Movement", "source": "uber", "status": "unavailable", "error": str(e)})

    # India MP
    try:
        _, _, _, info = load_india_mp()
        infos.append(info.to_dict())
    except Exception as e:
        infos.append({"name": "India MP Traffic Network", "source": "india-mp", "status": "unavailable", "error": str(e)})

    return infos
