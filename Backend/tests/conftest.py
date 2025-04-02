import pytest
from fastapi.testclient import TestClient

# =============================
# Fixtures for API (detection_api.py)
# =============================
@pytest.fixture
def api_client():
    """
    Fixture for testing the FastAPI application.
    Related file: detection_api.py
    """
    from detection_api import app
    return TestClient(app)

# =============================
# Fixtures for Packet Capture Modules
# =============================
@pytest.fixture
def packet_capture_instance():
    """
    Fixture to create a PacketCapture instance.
    Related file: packet_capture.py
    """
    from packet_capture import PacketCapture
    return PacketCapture()

@pytest.fixture
def pyshark_capture_instance():
    """
    Fixture to create a PySharkCapture instance.
    Related file: pyshark_capture.py
    """
    from pyshark_capture import PySharkCapture
    return PySharkCapture()

# =============================
# Fixtures for Feature Extraction & Engineering
# =============================
@pytest.fixture
def feature_extractor():
    """
    Fixture to create a FeatureExtractor instance.
    Related file: feature_extraction.py
    """
    from feature_extraction import FeatureExtractor
    return FeatureExtractor()

@pytest.fixture
def feature_engineer():
    """
    Fixture to create a FeatureEngineer instance.
    Related file: feature_engineering.py
    """
    from feature_engineering import FeatureEngineer
    return FeatureEngineer()

# =============================
# Fixtures for Deep Learning Models
# =============================
@pytest.fixture
def deep_learning_detector():
    """
    Fixture to create a DeepLearningDetector instance.
    This builds an autoencoder with a dummy input shape and encoding dimension.
    Related file: deep_learning.py
    """
    from deep_learning import DeepLearningDetector
    # Create a detector instance with an example input shape of 18 features
    detector = DeepLearningDetector(model_type='autoencoder', input_shape=(18,), model_params={'encoding_dim': 10})
    detector.build_model()
    # Optionally, set a threshold (simulate a trained model)
    detector.threshold = 0.8
    return detector

# =============================
# Fixtures for Traditional Anomaly Detection
# =============================
@pytest.fixture
def anomaly_detector_instance():
    """
    Fixture to create an AnomalyDetector instance.
    Related file: anomaly_detection.py
    """
    from anomaly_detection import AnomalyDetector
    return AnomalyDetector(model_type='isolation_forest')

# =============================
# Fixtures for the Sample Client
# =============================
@pytest.fixture
def sample_flows():
    """
    Fixture to generate sample network flows.
    Uses functions from sample_client.py to produce both normal and anomalous flows.
    Related file: sample_client.py
    """
    import sample_client
    flows = []
    flows.append(sample_client.generate_normal_flow())
    flows.append(sample_client.generate_anomalous_flow())
    return flows

# =============================
# General / Unspecified Fixtures (if needed)
# =============================
# Add additional fixtures here for shared resources or configurations that apply
# across multiple test modules.
