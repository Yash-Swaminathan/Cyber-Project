import numpy as np
import pytest
from Backend.anomaly_detection import AnomalyDetector


def test_prepare_and_train_anomaly_detector(anomaly_detector_instance, synthetic_data):
    X_df, y_series = synthetic_data
    X_train, X_test, y_train, y_test = anomaly_detector_instance.prepare_data(X_df, y_series)
    anomaly_detector_instance.train(X_train, y_train)
    predictions = anomaly_detector_instance.predict(X_test)
    assert isinstance(predictions, np.ndarray)

def test_evaluate_anomaly_detector(anomaly_detector_instance, synthetic_data):
    X_df, y_series = synthetic_data
    X_train, X_test, y_train, y_test = anomaly_detector_instance.prepare_data(X_df, y_series)
    anomaly_detector_instance.train(X_train, y_train)
    metrics = anomaly_detector_instance.evaluate(X_test, y_test)
    assert 'f1_score' in metrics

def test_plot_anomaly_scores(anomaly_detector_instance, synthetic_data):
    X_df, y_series = synthetic_data
    # Train using prepared data
    anomaly_detector_instance.train(*anomaly_detector_instance.prepare_data(X_df))
    fig = anomaly_detector_instance.plot_anomaly_scores(X_df, y_true=y_series)
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
