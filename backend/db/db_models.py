"""
SQLAlchemy ORM models for persisting metadata, training runs, and evaluations.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from backend.db.database import Base


class SensorMetadata(Base):
    """Stores metadata for each traffic sensor / road segment."""
    __tablename__ = "sensor_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), default="")
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    source = Column(String(50), default="metr-la")  # metr-la, pems-bay, uber
    road_type = Column(String(50), default="highway")
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    """Records each model training session."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)  # xgboost, lstm, stgcn
    dataset = Column(String(100), default="demo")
    forecast_horizon = Column(Integer, default=30)
    epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=32)
    learning_rate = Column(Float, default=0.001)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 100.0
    training_time_seconds = Column(Float, nullable=True)
    config_json = Column(JSON, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)


class EvaluationResult(Base):
    """Stores evaluation metrics for trained models."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, nullable=True)
    model_name = Column(String(50), nullable=False)
    dataset = Column(String(100), default="demo")
    forecast_horizon = Column(Integer, default=30)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    num_test_samples = Column(Integer, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class ModelMetadata(Base):
    """Tracks saved model files and their configurations."""
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    version = Column(String(50), default="1.0")
    file_path = Column(String(500), nullable=True)
    dataset = Column(String(100), default="demo")
    forecast_horizon = Column(Integer, default=30)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = archived
    config_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
