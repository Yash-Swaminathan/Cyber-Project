import pytest
from fastapi.testclient import TestClient
from Backend.Deep_Learning.deep_learning import DeepLearningDetector
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from Backend.detection_api import app

# =============================================================================
# Fixtures for API (detection_api.py)
# Related test file: test_detection_api.py
# =============================================================================
@pytest.fixture
def api_client():
    """
    Fixture for testing the FastAPI application.
    Related file: detection_api.py
    """
    from Backend.detection_api import app
    return TestClient(app)

# =============================================================================
# Fixtures for Packet Capture Modules
# Related test file: test_packet_capture.py
# =============================================================================
@pytest.fixture
def packet_capture_instance():
    """
    Fixture to create a PacketCapture instance.
    Related file: packet_capture.py
    """
    from Backend.packet_capture import PacketCapture
    return PacketCapture()

# =============================================================================
# Fixtures for Feature Extraction & Engineering
# Related test files: test_feature_extraction.py, test_feature_engineering.py
# =============================================================================
@pytest.fixture
def feature_extractor():
    """
    Fixture to create a FeatureExtractor instance.
    Related file: feature_extraction.py
    """
    from Backend.feature_extraction import FeatureExtractor
    return FeatureExtractor()

@pytest.fixture
def feature_engineer():
    """
    Fixture to create a FeatureEngineer instance.
    Related file: feature_engineering.py
    """
    from Backend.feature_engineering import FeatureEngineer
    return FeatureEngineer()

# =============================================================================
# Fixtures for Deep Learning Models
# Related test file: test_deep_learning.py
# =============================================================================
@pytest.fixture
def deep_learning_detector():
    """
    Fixture to create a DeepLearningDetector instance.
    This builds an autoencoder with a dummy input shape and encoding dimension.
    Related file: deep_learning.py
    """
    # Create a detector instance with an example input shape of 18 features
    detector = DeepLearningDetector(model_type='autoencoder', input_shape=(18,), model_params={'encoding_dim': 10})
    detector.build_model()
    # Optionally, set a threshold (simulate a trained model)
    detector.threshold = 0.8
    return detector

# =============================================================================
# Fixtures for Traditional Anomaly Detection
# Related test file: test_anomaly_detection.py
# =============================================================================
@pytest.fixture
def anomaly_detector_instance():
    """
    Fixture to create an AnomalyDetector instance.
    Related file: anomaly_detection.py
    """
    from Backend.anomaly_detection import AnomalyDetector
    return AnomalyDetector(model_type='isolation_forest')

# =============================================================================
# Fixtures for PyShark Capture
# Related test file: test_pyshark_capture.py
# =============================================================================
@pytest.fixture
def pyshark_capture():
    from Backend.pyshark_capture import PySharkCapture
    return PySharkCapture(interface="eth0", output_dir="./test_captures")

# =============================================================================
# Additional Data Fixtures for Anomaly Detection and Deep Learning
# - synthetic_data is used in test_anomaly_detection.py
# - dummy_data is used in test_deep_learning.py
# =============================================================================
@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=200, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        weights=[0.9, 0.1], 
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    # Convert labels: 0 becomes normal (1), 1 becomes anomaly (-1)
    y_series = pd.Series(np.where(y == 0, 1, -1), name='label')
    return X_df, y_series

@pytest.fixture
def dummy_data():
    # Create dummy data with 18 features (matching our autoencoder configuration)
    return np.random.rand(50, 18)

# =============================================================================
# Fixtures for Mocking Deep Learning Detector for API tests
# Related test file: test_detection_api.py
# =============================================================================
@pytest.fixture
def mock_detector():
    """Create a mock detector for testing"""
    mock = MagicMock(spec=DeepLearningDetector)
    # Configure the mock to return numpy array of non-anomalous scores
    mock.get_anomaly_scores.return_value = np.array([0.3])  # Below threshold
    mock.threshold = 0.8
    return mock

@pytest.fixture
def api_client(mock_detector: MagicMock):
    """Create a test client with a mocked detector"""
    # Apply the mock detector to the global detector variable
    with patch('Backend.detection_api.detector', mock_detector):
        client = TestClient(app)
        yield client
