"""Sensor metadata endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter()


@router.get("/sensors")
async def get_sensors(source: Optional[str] = Query(None, description="Filter by source: metr-la, pems-bay")):
    """Get all sensor metadata (ID, lat, lon, source)."""
    from backend.main import app_state
    import pandas as pd

    sensors_list = []

    if source is None or source == "india-mp":
        if app_state.get("india_mp_sensors") is not None:
            df = app_state["india_mp_sensors"]
            for _, row in df.iterrows():
                sensors_list.append({
                    "sensor_id": row["sensor_id"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "source": "india-mp",
                    "city": row.get("city", "Madhya Pradesh"),
                })

    if source is None or source == "metr-la":
        if app_state["metr_la_sensors"] is not None:
            df = app_state["metr_la_sensors"]
            for _, row in df.iterrows():
                sensors_list.append({
                    "sensor_id": row["sensor_id"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "source": row.get("source", "metr-la"),
                })

    if source is None or source == "pems-bay":
        if app_state["pems_bay_sensors"] is not None:
            df = app_state["pems_bay_sensors"]
            for _, row in df.iterrows():
                sensors_list.append({
                    "sensor_id": row["sensor_id"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "source": row.get("source", "pems-bay"),
                })

    return {
        "sensors": sensors_list,
        "total": len(sensors_list),
        "demo_mode": app_state["demo_mode"],
    }


@router.get("/sensors/{sensor_id}")
async def get_sensor(sensor_id: str):
    """Get details for a specific sensor."""
    from backend.main import app_state
    import numpy as np

    # Search in India MP
    if app_state.get("india_mp_sensors") is not None:
        df = app_state["india_mp_sensors"]
        match = df[df["sensor_id"] == sensor_id]
        if len(match) > 0:
            row = match.iloc[0]
            speeds = app_state["india_mp_speeds"]
            recent_speed = None
            speed_history = []
            if speeds is not None and sensor_id in speeds.columns:
                recent_speed = float(speeds[sensor_id].iloc[-1])
                history = speeds[sensor_id].iloc[-96:]
                speed_history = [
                    {"timestamp": str(ts), "speed": float(v)}
                    for ts, v in zip(history.index, history.values)
                    if not np.isnan(v)
                ]

            from backend.config import get_congestion_label
            return {
                "sensor_id": sensor_id,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "source": "india-mp",
                "city": row.get("city", "Madhya Pradesh"),
                "road_name": row.get("road_name", sensor_id),
                "current_speed": recent_speed,
                "congestion_state": get_congestion_label(recent_speed) if recent_speed else "Unknown",
                "speed_history": speed_history[-48:],
                "demo_mode": app_state["demo_mode"],
            }
        df = app_state["metr_la_sensors"]
        match = df[df["sensor_id"] == sensor_id]
        if len(match) > 0:
            row = match.iloc[0]
            # Get recent speed data
            speeds = app_state["metr_la_speeds"]
            recent_speed = None
            speed_history = []
            if speeds is not None and sensor_id in speeds.columns:
                recent_speed = float(speeds[sensor_id].iloc[-1])
                # Last 24 entries (6 hours at 15min or 2 hours at 5min)
                history = speeds[sensor_id].iloc[-96:]
                speed_history = [
                    {"timestamp": str(ts), "speed": float(v)}
                    for ts, v in zip(history.index, history.values)
                    if not np.isnan(v)
                ]

            from backend.config import get_congestion_label
            return {
                "sensor_id": sensor_id,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "source": row.get("source", "metr-la"),
                "current_speed": recent_speed,
                "congestion_state": get_congestion_label(recent_speed) if recent_speed else "Unknown",
                "speed_history": speed_history[-48:],  # Last 48 entries
                "demo_mode": app_state["demo_mode"],
            }

    # Search in PeMS-BAY
    if app_state["pems_bay_sensors"] is not None:
        df = app_state["pems_bay_sensors"]
        match = df[df["sensor_id"] == sensor_id]
        if len(match) > 0:
            row = match.iloc[0]
            speeds = app_state["pems_bay_speeds"]
            recent_speed = None
            speed_history = []
            if speeds is not None and sensor_id in speeds.columns:
                recent_speed = float(speeds[sensor_id].iloc[-1])
                history = speeds[sensor_id].iloc[-96:]
                speed_history = [
                    {"timestamp": str(ts), "speed": float(v)}
                    for ts, v in zip(history.index, history.values)
                    if not np.isnan(v)
                ]

            from backend.config import get_congestion_label
            return {
                "sensor_id": sensor_id,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "source": row.get("source", "pems-bay"),
                "current_speed": recent_speed,
                "congestion_state": get_congestion_label(recent_speed) if recent_speed else "Unknown",
                "speed_history": speed_history[-48:],
                "demo_mode": app_state["demo_mode"],
            }

    raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found")
