"""
Smart Traffic Congestion Forecasting System — FastAPI Backend

Main application entry point. Registers all API routers,
configures CORS, initializes database, and loads demo data on startup.
"""

import logging
from contextlib import asynccontextmanager

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import FRONTEND_URL, ALLOWED_ORIGINS, DEMO_MODE, validate_config, PROJECT_ROOT
from backend.db.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Application State ───────────────────────────────────────────────────
# Shared state across requests (loaded at startup)
app_state = {
    "metr_la_speeds": None,
    "metr_la_sensors": None,
    "metr_la_adj": None,
    "metr_la_info": None,
    "pems_bay_speeds": None,
    "pems_bay_sensors": None,
    "pems_bay_adj": None,
    "pems_bay_info": None,
    "uber_data": None,
    "uber_info": None,
    "preprocessor": None,
    "fusion_pipeline": None,
    "models": {},           # {"xgboost": model, "lstm": model, "stgcn": model}
    "demo_mode": DEMO_MODE,
    "training_status": {},  # Current training jobs
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("  Smart Traffic Congestion Forecasting System")
    logger.info("  Starting up...")
    logger.info("=" * 60)

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Load demo data
    try:
        from backend.data.loader import load_metr_la, load_pems_bay, load_uber_movement

        speeds, sensors, adj, info = load_metr_la()
        app_state["metr_la_speeds"] = speeds
        app_state["metr_la_sensors"] = sensors
        app_state["metr_la_adj"] = adj
        app_state["metr_la_info"] = info
        logger.info(f"METR-LA loaded: {info.n_sensors} sensors, {info.n_records} records")

        speeds, sensors, adj, info = load_pems_bay()
        app_state["pems_bay_speeds"] = speeds
        app_state["pems_bay_sensors"] = sensors
        app_state["pems_bay_adj"] = adj
        app_state["pems_bay_info"] = info
        logger.info(f"PeMS-BAY loaded: {info.n_sensors} sensors, {info.n_records} records")

        from backend.data.loader import load_india_mp
        speeds, sensors, adj, info = load_india_mp()
        app_state["india_mp_speeds"] = speeds
        app_state["india_mp_sensors"] = sensors
        app_state["india_mp_adj"] = adj
        app_state["india_mp_info"] = info
        logger.info(f"India MP loaded: {info.n_sensors} sensors across Madhya Pradesh")

        uber_df, uber_info = load_uber_movement()
        app_state["uber_data"] = uber_df
        app_state["uber_info"] = uber_info
        logger.info(f"Uber Movement loaded: {uber_info.n_records} records")

    except Exception as e:
        logger.warning(f"Failed to load datasets: {e}")

    # Initialize preprocessor
    from backend.data.preprocessing import TrafficPreprocessor
    app_state["preprocessor"] = TrafficPreprocessor()

    # Initialize fusion pipeline
    from backend.data.fusion import DataFusionPipeline
    app_state["fusion_pipeline"] = DataFusionPipeline()

    logger.info("System ready!")
    logger.info(f"Demo mode: {'ON' if app_state['demo_mode'] else 'OFF'}")

    yield  # App runs here

    logger.info("Shutting down...")


# ── Create App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Traffic Congestion Forecasting",
    description="AI-powered Urban Mobility Management System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ───────────────────────────────────────────────────
from backend.api.health import router as health_router
from backend.api.datasets import router as datasets_router
from backend.api.sensors import router as sensors_router
from backend.api.traffic import router as traffic_router
from backend.api.forecast import router as forecast_router
from backend.api.training import router as training_router
from backend.api.evaluation import router as evaluation_router
from backend.api.fusion import router as fusion_router

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(datasets_router, prefix="/api", tags=["Datasets"])
app.include_router(sensors_router, prefix="/api", tags=["Sensors"])
app.include_router(traffic_router, prefix="/api", tags=["Traffic"])
app.include_router(forecast_router, prefix="/api", tags=["Forecast"])
app.include_router(training_router, prefix="/api", tags=["Training"])
app.include_router(evaluation_router, prefix="/api", tags=["Evaluation"])
app.include_router(fusion_router, prefix="/api", tags=["Data Fusion"])


# ── Serve Built Frontend SPA (Production Single-Server Packaging) ────────
frontend_dist = PROJECT_ROOT / "frontend" / "dist"

if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API routes, docs, and openapi schema to pass through
        if full_path.startswith("api") or full_path.startswith("docs") or full_path == "openapi.json":
            return
        target_file = frontend_dist / full_path
        if full_path and target_file.exists() and target_file.is_file():
            return FileResponse(str(target_file))
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "name": "Smart Traffic Congestion Forecasting System",
            "version": "1.0.0",
            "docs": "/docs",
            "demo_mode": app_state["demo_mode"],
        }

