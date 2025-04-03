import numpy as np
import pytest
from Backend.deep_learning import DeepLearningDetector

@pytest.fixture
def dummy_data():
    # Create dummy data with 18 features (matching our autoencoder configuration)
    return np.random.rand(50, 18)

def test_build_and_train_deep_learning_detector(deep_learning_detector, dummy_data):
    detector = deep_learning_detector
    # Train on dummy data for one epoch
    history = detector.train(dummy_data, epochs=1, batch_size=8, validation_split=0.2)
    # Get anomaly scores and verify shape
    scores = detector.get_anomaly_scores(dummy_data)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == dummy_data.shape[0]

def test_save_and_load_deep_learning_detector(tmp_path, deep_learning_detector, dummy_data):
    detector = deep_learning_detector
    model_path = str(tmp_path / "test_model.h5")
    detector.save_model(model_path)
    loaded_detector = DeepLearningDetector.load_model(model_path)
    scores_original = detector.get_anomaly_scores(dummy_data)
    scores_loaded = loaded_detector.get_anomaly_scores(dummy_data)
    # Verify that the output shapes match
    assert scores_original.shape == scores_loaded.shape
