import json
import pytest
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from Backend.detection_api import app, detector
from Backend.deep_learning import DeepLearningDetector

@pytest.fixture
def mock_detector():
    """Create a mock detector for testing"""
    mock = MagicMock(spec=DeepLearningDetector)
    # Configure the mock to return numpy array of non-anomalous scores
    mock.get_anomaly_scores.return_value = np.array([0.3])  # Below threshold
    mock.threshold = 0.8
    return mock

@pytest.fixture
def api_client(mock_detector):
    """Create a test client with a mocked detector"""
    # Apply the mock detector to the global detector variable
    with patch('Backend.detection_api.detector', mock_detector):
        client = TestClient(app)
        yield client

def test_health_endpoint(api_client):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_config_endpoint(api_client):
    response = api_client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "alerting" in data
    # Verify that sensitive fields are masked.
    if "email" in data["alerting"]:
        assert data["alerting"]["email"].get("password") == "********"

def test_test_alert_endpoint(api_client):
    response = api_client.post("/api/v1/test-alert")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "alert_id" in data

def test_detect_anomalies(api_client, mock_detector):
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