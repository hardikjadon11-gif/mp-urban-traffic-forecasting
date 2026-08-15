"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """System health check."""
    from backend.main import app_state

    models_status = {}
    for name in ["xgboost", "lstm", "stgcn"]:
        model = app_state["models"].get(name)
        models_status[name] = {
            "loaded": model is not None,
            "trained": model.is_trained if model else False,
        }

    return {
        "status": "healthy",
        "demo_mode": app_state["demo_mode"],
        "datasets": {
            "metr_la": app_state["metr_la_info"].to_dict() if app_state["metr_la_info"] else None,
            "pems_bay": app_state["pems_bay_info"].to_dict() if app_state["pems_bay_info"] else None,
            "uber": app_state["uber_info"].to_dict() if app_state["uber_info"] else None,
        },
        "models": models_status,
        "fusion_ready": app_state["fusion_pipeline"] is not None,
    }
