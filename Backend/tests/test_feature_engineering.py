import pandas as pd
import pytest
from Backend.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='T'),
        'src_ip': ['192.168.0.1', '10.0.0.1', '8.8.8.8', '192.168.0.1', '10.0.0.1'],
        'dst_ip': ['10.0.0.2', '192.168.0.1', '192.168.0.2', '10.0.0.2', '192.168.0.1'],
        'src_port': [80, 443, 12345, 80, 443],
        'dst_port': [12345, 80, 53, 12345, 80],
        'packet_size': [100, 150, 200, 250, 300],
        'protocol': ['TCP', 'TCP', 'UDP', 'TCP', 'UDP']
    })

def test_categorize_ip(feature_engineer):
    fe = feature_engineer
    result = fe.categorize_ip("192.168.1.1")
    assert isinstance(result, dict)
    assert 'is_internal' in result

def test_extract_base_features(feature_engineer, sample_data):
    df = feature_engineer.extract_base_features(sample_data.copy())
    # Check that time-based features are added.
    assert 'hour_of_day' in df.columns
    assert 'src_ip_internal' in df.columns

def test_aggregate_by_time_window(feature_engineer, sample_data):
    df = feature_engineer.extract_base_features(sample_data.copy())
    aggregated = feature_engineer.aggregate_by_time_window(df, groupby_cols=['src_ip'])
    assert isinstance(aggregated, pd.DataFrame)
