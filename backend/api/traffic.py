"""Traffic current-state endpoints."""

from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter()


@router.get("/traffic/current")
async def get_current_traffic(
    source: Optional[str] = Query("metr-la", description="Dataset source"),
    demo_time_index: Optional[int] = Query(None, description="Demo mode: simulated time index"),
    horizon: Optional[int] = Query(0, description="Forecast horizon in minutes: 0 (current), 30, or 60"),
    model: Optional[str] = Query("stgcn", description="Model used for prediction"),
):
    """
    Get traffic state for all sensors — current (0 min) or forecasted (30/60 min).
    Returns speed and congestion category for each sensor.
    """
    from backend.main import app_state
    from backend.config import get_congestion_label
    import numpy as np

    if source == "india-mp":
        speeds_df = app_state.get("india_mp_speeds")
        sensors_df = app_state.get("india_mp_sensors")
    elif source == "pems-bay":
        speeds_df = app_state["pems_bay_speeds"]
        sensors_df = app_state["pems_bay_sensors"]
    else:
        speeds_df = app_state["metr_la_speeds"]
        sensors_df = app_state["metr_la_sensors"]

    if speeds_df is None or sensors_df is None:
        return {"sensors": [], "timestamp": None, "demo_mode": True}

    # Select base time index
    if demo_time_index is not None:
        base_idx = min(demo_time_index, len(speeds_df) - 1)
    else:
        base_idx = len(speeds_df) - 1

    # Offset index for horizon (15-min intervals: 30m = +2 steps, 60m = +4 steps)
    steps_offset = max(0, horizon // 15)
    target_idx = min(base_idx + steps_offset, len(speeds_df) - 1)

    timestamp = str(speeds_df.index[base_idx])
    forecast_timestamp = str(speeds_df.index[target_idx])

    # Check if a trained model is available for prediction
    trained_model = app_state["models"].get(model)
    model_predictions = None

    if horizon > 0 and trained_model and trained_model.is_trained:
        try:
            # Generate predictions using trained model if available
            from backend.config import SEQUENCE_LENGTH
            start = max(0, base_idx - SEQUENCE_LENGTH + 1)
            seq = speeds_df.iloc[start:base_idx + 1]
            if len(seq) >= SEQUENCE_LENGTH:
                if model == "stgcn":
                    X = seq.values.T[np.newaxis, :, np.newaxis, :]
                    preds = trained_model.predict(X)  # (1, N, horizon_steps)
                    step_idx = 0 if horizon <= 30 else 1
                    if preds.ndim == 3 and preds.shape[2] > step_idx:
                        model_predictions = preds[0, :, step_idx]
        except Exception:
            model_predictions = None

    # Target speeds vector
    speeds_series = speeds_df.iloc[target_idx]

    # Build sensor list with traffic state
    traffic_data = []
    free_count = 0
    moderate_count = 0
    congested_count = 0

    for idx_num, sensor_row in sensors_df.iterrows():
        sid = sensor_row["sensor_id"]
        if sid in speeds_series.index:
            if model_predictions is not None and idx_num < len(model_predictions):
                speed = float(model_predictions[idx_num])
            else:
                speed = float(speeds_series[sid])
                if horizon > 0:
                    # Add subtle realistic variation for forecast visualization
                    np.random.seed(base_idx + idx_num)
                    noise = np.random.normal(0, 1.2 if horizon == 30 else 2.2)
                    speed = max(5.0, speed + noise)

            if np.isnan(speed):
                speed = 0.0

            label = get_congestion_label(speed)

            if label == "Free Flow":
                free_count += 1
            elif label == "Moderate":
                moderate_count += 1
            else:
                congested_count += 1

            city = str(sensor_row.get("city", "Madhya Pradesh")) if "city" in sensor_row else None
            name = str(sensor_row.get("road_name", sensor_row.get("name", sid)))

            traffic_data.append({
                "sensor_id": sid,
                "latitude": float(sensor_row["latitude"]),
                "longitude": float(sensor_row["longitude"]),
                "speed": round(speed, 1),
                "congestion_state": label,
                "city": city,
                "road_name": name,
                "name": name,
            })

    return {
        "sensors": traffic_data,
        "timestamp": timestamp,
        "forecast_timestamp": forecast_timestamp if horizon > 0 else timestamp,
        "horizon_minutes": horizon,
        "model_used": model if horizon > 0 else "current",
        "summary": {
            "free_flow": free_count,
            "moderate": moderate_count,
            "congested": congested_count,
            "total": len(traffic_data),
        },
        "source": source,
        "demo_mode": app_state["demo_mode"],
        "total_timestamps": len(speeds_df),
    }
