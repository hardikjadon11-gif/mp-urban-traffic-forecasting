"""Data fusion status endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/data-fusion/status")
async def get_fusion_status():
    """Get the current status and configuration of the data fusion pipeline."""
    from backend.main import app_state
    from backend.config import (
        METR_LA_WEIGHT, PEMS_BAY_WEIGHT, UBER_WEIGHT, RESAMPLE_INTERVAL_MINUTES
    )

    pipeline = app_state["fusion_pipeline"]
    status = pipeline.get_status() if pipeline else {}

    return {
        "pipeline_status": status,
        "configuration": {
            "weights": {
                "metr_la": METR_LA_WEIGHT,
                "pems_bay": PEMS_BAY_WEIGHT,
                "uber": UBER_WEIGHT,
            },
            "resample_interval_minutes": RESAMPLE_INTERVAL_MINUTES,
        },
        "steps": [
            {"step": 1, "name": "Temporal Alignment", "description": "Resample all sources to common 15-minute interval"},
            {"step": 2, "name": "Spatial Alignment", "description": "Map sensors to graph nodes via geographic coordinates"},
            {"step": 3, "name": "Normalization", "description": "Scale each source independently (MinMax to [0,1])"},
            {"step": 4, "name": "Feature Concatenation", "description": "Create unified feature matrix from all sources"},
            {"step": 5, "name": "Weighted Combination", "description": "Apply configurable source weights"},
        ],
        "demo_mode": app_state["demo_mode"],
    }
