"""Integration tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if 'src' in sys.modules:
    del sys.modules['src']

from api.app import app


def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "HEALTHY"


def test_api_power_sample_size():
    with TestClient(app) as client:
        payload = {
            "baseline_conversion_rate": 0.12,
            "mde_absolute": 0.005,
            "alpha": 0.05,
            "power": 0.80,
            "daily_traffic": 15000,
            "test_type": "two_sided"
        }
        response = client.post("/power-analysis/sample-size", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "required_sample_per_variant" in data
        assert data["required_sample_per_variant"] > 0


def test_api_frequentist():
    with TestClient(app) as client:
        payload = {
            "control_conversions": 17489,
            "control_total": 145274,
            "treatment_conversions": 17264,
            "treatment_total": 145310,
            "alpha": 0.05,
            "mde": 0.005
        }
        response = client.post("/analyze/frequentist", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "p_value" in data
        assert not data["is_statistically_significant"]
