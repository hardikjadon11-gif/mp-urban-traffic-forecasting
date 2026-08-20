"""Disruption Simulator API — 'What-If' incident propagation."""

from fastapi import APIRouter, Query
from typing import Optional
import numpy as np
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory simulation state (reset on server restart)
_simulation_state = {
    "active": False,
    "source_sensor": None,
    "affected_sensors": [],
    "duration_minutes": 30,
}


@router.post("/simulate/disruption")
async def simulate_disruption(
    sensor_id: str = Query(..., description="Sensor to simulate incident on"),
    duration_minutes: int = Query(30, description="Propagation window: 30 or 60"),
    source: str = Query("india-mp", description="Dataset source"),
    demo_time_index: Optional[int] = Query(None, description="Demo time index"),
):
    """
    Simulate a traffic incident at a sensor and compute congestion
    propagation to neighboring sensors using the STGCN adjacency matrix.
    """
    from backend.main import app_state
    from backend.config import get_congestion_label

    # Resolve dataset
    if source == "india-mp":
        speeds_df = app_state.get("india_mp_speeds")
        sensors_df = app_state.get("india_mp_sensors")
        adj = app_state.get("india_mp_adj")
    elif source == "pems-bay":
        speeds_df = app_state.get("pems_bay_speeds")
        sensors_df = app_state.get("pems_bay_sensors")
        adj = app_state.get("pems_bay_adj")
    else:
        speeds_df = app_state.get("metr_la_speeds")
        sensors_df = app_state.get("metr_la_sensors")
        adj = app_state.get("metr_la_adj")

    if speeds_df is None or sensors_df is None:
        return {"error": "Dataset not loaded", "affected_sensors": []}

    # Find source sensor index
    sensor_ids = list(speeds_df.columns)
    if sensor_id not in sensor_ids:
        return {"error": f"Sensor '{sensor_id}' not found", "affected_sensors": []}

    source_idx = sensor_ids.index(sensor_id)

    # Use adjacency matrix to find neighbors
    if adj is None:
        adj = np.eye(len(sensor_ids))

    # Get current time index
    if demo_time_index is not None:
        current_idx = min(demo_time_index, len(speeds_df) - 1)
    else:
        current_idx = len(speeds_df) - 1

    # Find neighbors: sensors with non-zero adjacency weight
    adj_row = adj[source_idx] if source_idx < adj.shape[0] else np.zeros(len(sensor_ids))

    # Duration factor: 60-min propagation affects more sensors, deeper
    duration_factor = 1.0 if duration_minutes <= 30 else 1.5

    affected = []
    for i, weight in enumerate(adj_row):
        if i == source_idx:
            continue  # Skip self
        if weight > 0:
            # Speed reduction proportional to adjacency weight
            # Higher weight = closer neighbor = more affected
            reduction_pct = min(0.7, weight * duration_factor * 0.5)

            original_speed = float(speeds_df.iloc[current_idx, i])
            reduced_speed = max(5.0, original_speed * (1.0 - reduction_pct))

            # Find sensor metadata
            sensor_row = sensors_df[sensors_df["sensor_id"] == sensor_ids[i]]
            if len(sensor_row) == 0:
                continue

            sensor_row = sensor_row.iloc[0]
            affected.append({
                "sensor_id": sensor_ids[i],
                "latitude": float(sensor_row["latitude"]),
                "longitude": float(sensor_row["longitude"]),
                "original_speed": round(original_speed, 1),
                "reduced_speed": round(reduced_speed, 1),
                "speed_reduction_pct": round(reduction_pct * 100, 1),
                "original_state": get_congestion_label(original_speed),
                "projected_state": get_congestion_label(reduced_speed),
                "adjacency_weight": round(float(weight), 4),
                "road_name": str(sensor_row.get("road_name", sensor_ids[i])),
                "city": str(sensor_row.get("city", "")) if "city" in sensor_row.index else None,
            })

    # Sort by impact (highest speed reduction first)
    affected.sort(key=lambda x: x["speed_reduction_pct"], reverse=True)

    # Get source sensor info
    source_row = sensors_df[sensors_df["sensor_id"] == sensor_id].iloc[0]
    source_speed = float(speeds_df[sensor_id].iloc[current_idx])

    # Update simulation state
    _simulation_state["active"] = True
    _simulation_state["source_sensor"] = sensor_id
    _simulation_state["affected_sensors"] = [a["sensor_id"] for a in affected]
    _simulation_state["duration_minutes"] = duration_minutes

    return {
        "simulation_active": True,
        "source_sensor": {
            "sensor_id": sensor_id,
            "latitude": float(source_row["latitude"]),
            "longitude": float(source_row["longitude"]),
            "original_speed": round(source_speed, 1),
            "simulated_speed": 5.0,  # Full blockage
            "road_name": str(source_row.get("road_name", sensor_id)),
            "city": str(source_row.get("city", "")) if "city" in source_row.index else None,
        },
        "affected_sensors": affected,
        "total_affected": len(affected),
        "duration_minutes": duration_minutes,
        "propagation_note": "Estimated propagation based on STGCN spatial graph adjacency structure",
    }


@router.post("/simulate/reset")
async def reset_simulation():
    """Reset the disruption simulation."""
    _simulation_state["active"] = False
    _simulation_state["source_sensor"] = None
    _simulation_state["affected_sensors"] = []
    return {"simulation_active": False, "message": "Simulation reset"}
