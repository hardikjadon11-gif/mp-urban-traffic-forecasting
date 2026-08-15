"""
Smart Traffic Congestion Forecasting System — Central Configuration

All configurable parameters for the system. Values can be overridden
via environment variables or a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── Base Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

# ── Dataset Paths ───────────────────────────────────────────────────────
METR_LA_PATH = Path(os.getenv("METR_LA_PATH", str(DATA_DIR / "metr-la" / "metr-la.h5")))
PEMS_BAY_PATH = Path(os.getenv("PEMS_BAY_PATH", str(DATA_DIR / "pems-bay" / "pems-bay.h5")))
UBER_MOVEMENT_PATH = Path(os.getenv("UBER_MOVEMENT_PATH", str(DATA_DIR / "uber-movement" / "uber-zones.csv")))
DEMO_DATA_DIR = Path(os.getenv("DEMO_DATA_DIR", str(DATA_DIR / "demo")))

# ── Database ────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'traffic_forecasting.db'}")

# ── Server ──────────────────────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        f"{FRONTEND_URL},http://localhost:5173,http://localhost:3000,http://localhost:8000,http://localhost,https://localhost,capacitor://localhost,*",
    ).split(",")
    if o.strip()
]


# ── Model Configuration ────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "stgcn")
FORECAST_HORIZONS = [int(h) for h in os.getenv("FORECAST_HORIZONS", "30,60").split(",")]

# Model storage
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Training Defaults ──────────────────────────────────────────────────
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "50"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
DEFAULT_LEARNING_RATE = float(os.getenv("DEFAULT_LEARNING_RATE", "0.001"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "12"))  # 12 time steps lookback

# ── Data Fusion Weights ────────────────────────────────────────────────
# Weights for combining multi-source data. Must sum to 1.0.
METR_LA_WEIGHT = float(os.getenv("METR_LA_WEIGHT", "0.5"))
PEMS_BAY_WEIGHT = float(os.getenv("PEMS_BAY_WEIGHT", "0.3"))
UBER_WEIGHT = float(os.getenv("UBER_WEIGHT", "0.2"))

# ── Congestion Thresholds (mph) ────────────────────────────────────────
# These are project-configurable thresholds, not scientifically validated.
# FREE FLOW: speed >= CONGESTION_FREE_THRESHOLD
# MODERATE:  CONGESTION_MODERATE_THRESHOLD <= speed < CONGESTION_FREE_THRESHOLD
# CONGESTED: speed < CONGESTION_MODERATE_THRESHOLD
CONGESTION_FREE_THRESHOLD = float(os.getenv("CONGESTION_FREE_THRESHOLD", "45"))
CONGESTION_MODERATE_THRESHOLD = float(os.getenv("CONGESTION_MODERATE_THRESHOLD", "25"))

# ── Sampling ───────────────────────────────────────────────────────────
RESAMPLE_INTERVAL_MINUTES = int(os.getenv("RESAMPLE_INTERVAL_MINUTES", "15"))

# ── Train/Val/Test Split ───────────────────────────────────────────────
# Chronological split — no random shuffling for time-series data.
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.7"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.15"))

# ── Demo Mode ──────────────────────────────────────────────────────────
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")

# ── STGCN Specific ─────────────────────────────────────────────────────
STGCN_CHANNELS = [1, 16, 32, 64]  # Channel dimensions for ST-Conv blocks
STGCN_NUM_LAYERS = 2  # Number of ST-Conv blocks
CHEB_K = 3  # Chebyshev polynomial order

# ── LSTM Specific ──────────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# ── XGBoost Specific ──────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}


def get_congestion_label(speed_mph: float) -> str:
    """Classify traffic speed into congestion category."""
    if speed_mph >= CONGESTION_FREE_THRESHOLD:
        return "Free Flow"
    elif speed_mph >= CONGESTION_MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Congested"


def validate_config():
    """Validate configuration values at startup."""
    errors = []
    weight_sum = METR_LA_WEIGHT + PEMS_BAY_WEIGHT + UBER_WEIGHT
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Fusion weights must sum to 1.0, got {weight_sum}")
    split_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(split_sum - 1.0) > 0.01:
        errors.append(f"Split ratios must sum to 1.0, got {split_sum}")
    if CONGESTION_MODERATE_THRESHOLD >= CONGESTION_FREE_THRESHOLD:
        errors.append("CONGESTION_MODERATE_THRESHOLD must be < CONGESTION_FREE_THRESHOLD")
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
