"""Tests for API endpoints."""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "demo_mode" in data

    def test_health_has_model_info(self):
        response = client.get("/api/health")
        data = response.json()
        assert "models" in data
        assert "stgcn" in data["models"]


class TestDatasetsEndpoint:
    def test_datasets_returns_200(self):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data


class TestSensorsEndpoint:
    def test_sensors_returns_200(self):
        response = client.get("/api/sensors?source=metr-la")
        assert response.status_code == 200
        data = response.json()
        assert "sensors" in data
        assert "total" in data

    def test_invalid_sensor_404(self):
        response = client.get("/api/sensors/NONEXISTENT_999")
        assert response.status_code == 404


class TestTrafficEndpoint:
    def test_current_traffic_returns_200(self):
        response = client.get("/api/traffic/current?source=metr-la")
        assert response.status_code == 200
        data = response.json()
        assert "sensors" in data


class TestForecastEndpoint:
    def test_forecast_for_sensor(self):
        sensors = client.get("/api/sensors?source=metr-la").json()
        if sensors["sensors"]:
            sid = sensors["sensors"][0]["sensor_id"]
            response = client.get(f"/api/forecast/{sid}")
            assert response.status_code == 200
            data = response.json()
            assert "forecasts" in data
            assert "30_min" in data["forecasts"]
            assert "60_min" in data["forecasts"]
            assert data["forecasts"]["30_min"]["horizon_minutes"] == 30
            assert data["forecasts"]["60_min"]["horizon_minutes"] == 60

    def test_invalid_sensor_forecast(self):
        response = client.get("/api/forecast/NONEXISTENT_999")
        assert response.status_code == 404


class TestModelPerformance:
    def test_model_performance(self):
        response = client.get("/api/model-performance")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        names = [m["model_name"] for m in data["models"]]
        assert "naive_baseline" in names
        assert "xgboost" in names
        assert "lstm" in names
        assert "stgcn" in names

    def test_naive_baseline_presence(self):
        data = client.get("/api/model-performance").json()
        naive = [m for m in data["models"] if m["model_name"] == "naive_baseline"][0]
        assert naive["role"] == "Benchmark Floor"
        assert naive["is_primary"] is False

    def test_stgcn_is_primary(self):
        data = client.get("/api/model-performance").json()
        stgcn = [m for m in data["models"] if m["model_name"] == "stgcn"][0]
        assert stgcn["is_primary"] is True


class TestFusionEndpoint:
    def test_fusion_status(self):
        response = client.get("/api/data-fusion/status")
        assert response.status_code == 200
        data = response.json()
        assert "steps" in data
        assert len(data["steps"]) == 5
        assert "configuration" in data


class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            assert "<html" in response.text.lower() or "root" in response.text.lower()
        else:
            data = response.json()
            assert "Smart Traffic" in data["name"]

