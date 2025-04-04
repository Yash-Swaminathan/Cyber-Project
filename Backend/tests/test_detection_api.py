import json
import pytest
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from Backend.Deep_Learning.deep_learning import DeepLearningDetector 

def test_health_endpoint(api_client: TestClient):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_config_endpoint(api_client: TestClient):
    response = api_client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "alerting" in data
    # Verify that sensitive fields are masked.
    if "email" in data["alerting"]:
        assert data["alerting"]["email"].get("password") == "********"

def test_test_alert_endpoint(api_client: TestClient):
    response = api_client.post("/api/v1/test-alert")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "alert_id" in data

def test_detect_anomalies(api_client: TestClient, mock_detector: MagicMock):
    # Configure mock to return specific values for this test
    mock_detector.get_anomaly_scores.return_value = np.array([0.3])  # Non-anomalous score as numpy array
    
    # Create a sample network flow for detection
    sample_flow = {
        "flows": [
            {
                "timestamp": datetime.now().isoformat(),
                "src_ip": "192.168.1.1",
                "dst_ip": "10.0.0.1",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": "TCP",
                "bytes_sent": 1024,
                "bytes_received": 2048,
                "packets_sent": 10,
                "packets_received": 15,
                "duration": 1.0
            }
        ]
    }
    response = api_client.post("/api/v1/detect", json=sample_flow)
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "message" in data