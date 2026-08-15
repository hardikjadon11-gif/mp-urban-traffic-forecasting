"""Model training endpoint — triggers async training."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import time as time_module
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    model_name: str = "stgcn"  # xgboost, lstm, stgcn
    dataset: str = "metr-la"
    forecast_horizon: int = 30  # minutes
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    n_sensors: Optional[int] = 20  # Limit sensors for faster training


@router.post("/train")
async def start_training(request: TrainingRequest):
    """Start model training as a background task."""
    from backend.main import app_state

    if request.model_name not in ["xgboost", "lstm", "stgcn"]:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")

    # Check if already training
    status = app_state["training_status"].get(request.model_name, {})
    if status.get("status") == "running":
        raise HTTPException(status_code=409, detail=f"{request.model_name} is already training")

    # Set status
    app_state["training_status"][request.model_name] = {
        "status": "starting",
        "progress": 0,
        "model_name": request.model_name,
        "started_at": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "config": request.model_dump(),
    }

    # Launch training in a background thread
    thread = threading.Thread(
        target=_train_model_thread,
        args=(request, app_state),
        daemon=True,
    )
    thread.start()

    return {
        "message": f"Training {request.model_name} started",
        "status": "starting",
        "config": request.model_dump(),
    }


@router.get("/train/status")
async def get_training_status():
    """Get status of all training jobs."""
    from backend.main import app_state
    return {"training_status": app_state["training_status"]}


@router.get("/train/status/{model_name}")
async def get_model_training_status(model_name: str):
    """Get training status for a specific model."""
    from backend.main import app_state
    status = app_state["training_status"].get(model_name, {"status": "not_started"})
    return status


def _train_model_thread(request: TrainingRequest, app_state: dict):
    """Background thread for model training."""
    import numpy as np

    model_name = request.model_name
    try:
        app_state["training_status"][model_name]["status"] = "running"
        app_state["training_status"][model_name]["progress"] = 5
        logger.info(f"Training {model_name} started...")

        # Get data
        if request.dataset == "pems-bay":
            speeds_df = app_state["pems_bay_speeds"]
            adj_matrix = app_state["pems_bay_adj"]
        else:
            speeds_df = app_state["metr_la_speeds"]
            adj_matrix = app_state["metr_la_adj"]

        if speeds_df is None:
            raise ValueError("Dataset not loaded")

        # Limit sensors for manageable training
        n_sensors = min(request.n_sensors or 20, len(speeds_df.columns))
        speeds_subset = speeds_df.iloc[:, :n_sensors]

        app_state["training_status"][model_name]["progress"] = 10

        # Preprocess
        from backend.data.preprocessing import TrafficPreprocessor
        preprocessor = TrafficPreprocessor()
        train_df, val_df, test_df = preprocessor.full_pipeline(speeds_subset)

        app_state["training_status"][model_name]["progress"] = 20

        # Convert forecast horizon (minutes) to time steps
        interval = preprocessor.resample_minutes
        horizon_steps = max(1, request.forecast_horizon // interval)

        if model_name == "xgboost":
            result = _train_xgboost(train_df, val_df, test_df, adj_matrix, n_sensors, app_state)
        elif model_name == "lstm":
            result = _train_lstm(train_df, val_df, test_df, horizon_steps, request, app_state)
        elif model_name == "stgcn":
            result = _train_stgcn(train_df, val_df, test_df, adj_matrix, n_sensors,
                                   horizon_steps, request, app_state)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Evaluate on test set
        app_state["training_status"][model_name]["progress"] = 90
        app_state["training_status"][model_name]["status"] = "completed"
        app_state["training_status"][model_name]["progress"] = 100
        app_state["training_status"][model_name]["result"] = result
        app_state["training_status"][model_name]["completed_at"] = time_module.strftime("%Y-%m-%d %H:%M:%S")

        # Store preprocessor
        app_state["preprocessor"] = preprocessor

        logger.info(f"Training {model_name} completed: {result}")

    except Exception as e:
        logger.error(f"Training {model_name} failed: {e}", exc_info=True)
        app_state["training_status"][model_name]["status"] = "failed"
        app_state["training_status"][model_name]["error"] = str(e)


def _train_xgboost(train_df, val_df, test_df, adj_matrix, n_sensors, app_state):
    """Train XGBoost model."""
    from backend.models.xgboost_model import XGBoostTrafficModel
    from backend.features.engineering import prepare_xgboost_features
    from backend.evaluation.metrics import evaluate_predictions

    model = XGBoostTrafficModel(forecast_horizon=2)

    # Prepare features using first sensor
    target_sensor = train_df.columns[0]
    all_df = __import__('pandas').concat([train_df, val_df, test_df])

    X, y = prepare_xgboost_features(all_df, target_sensor=target_sensor)

    # Chronological split
    n_train = len(train_df)
    n_val = len(val_df)
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]
    X_test = X.iloc[n_train + n_val:]
    y_test = y.iloc[n_train + n_val:]

    # Drop any remaining NaN
    valid_train = X_train.dropna().index
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    app_state["training_status"]["xgboost"]["progress"] = 40

    # Train
    result = model.train(X_train, y_train, X_val, y_val, verbose=False)

    app_state["training_status"]["xgboost"]["progress"] = 70

    # Evaluate
    valid_test = X_test.dropna().index
    if len(valid_test) > 0:
        X_test = X_test.loc[valid_test]
        y_test = y_test.loc[valid_test]
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test.values, predictions)
        result["metrics"] = metrics

    # Save
    model.save()
    app_state["models"]["xgboost"] = model
    return result


def _train_lstm(train_df, val_df, test_df, horizon_steps, request, app_state):
    """Train LSTM model."""
    from backend.models.lstm_model import LSTMTrafficModel
    from backend.features.engineering import prepare_sequence_data
    from backend.evaluation.metrics import evaluate_predictions
    from backend.config import SEQUENCE_LENGTH
    import pandas as pd

    model = LSTMTrafficModel(forecast_horizon=horizon_steps)

    # Use mean across sensors for simplicity
    target_sensor = train_df.columns[0]

    # Prepare sequences
    all_df = pd.concat([train_df, val_df, test_df])
    X, y = prepare_sequence_data(all_df, seq_length=SEQUENCE_LENGTH,
                                  forecast_horizon=horizon_steps,
                                  target_sensor=target_sensor)

    # Split
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    app_state["training_status"]["lstm"]["progress"] = 30

    # Train
    result = model.train(
        X_train, y_train, X_val, y_val,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
    )

    app_state["training_status"]["lstm"]["progress"] = 80

    # Evaluate
    if len(X_test) > 0:
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        result["metrics"] = metrics

    model.save()
    app_state["models"]["lstm"] = model
    return result


def _train_stgcn(train_df, val_df, test_df, adj_matrix, n_sensors,
                  horizon_steps, request, app_state):
    """Train STGCN model."""
    from backend.models.stgcn_model import STGCNTrafficModel
    from backend.features.engineering import prepare_stgcn_data
    from backend.evaluation.metrics import evaluate_predictions
    from backend.config import SEQUENCE_LENGTH
    import pandas as pd

    model = STGCNTrafficModel(forecast_horizon=horizon_steps)

    # Prepare STGCN data
    all_df = pd.concat([train_df, val_df, test_df])
    X, y = prepare_stgcn_data(all_df, seq_length=SEQUENCE_LENGTH,
                               forecast_horizon=horizon_steps)

    # Split
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    app_state["training_status"]["stgcn"]["progress"] = 30

    # Use matching adjacency submatrix
    adj_sub = adj_matrix[:n_sensors, :n_sensors] if adj_matrix is not None else None

    # Train
    result = model.train(
        X_train, y_train, X_val, y_val,
        adj_matrix=adj_sub,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
    )

    app_state["training_status"]["stgcn"]["progress"] = 80

    # Evaluate
    if len(X_test) > 0:
        predictions = model.predict(X_test, adj_matrix=adj_sub)
        metrics = evaluate_predictions(y_test, predictions)
        result["metrics"] = metrics

    model.save()
    app_state["models"]["stgcn"] = model
    return result
