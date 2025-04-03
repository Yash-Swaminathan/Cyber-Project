import pandas as pd
import pytest
from Backend.feature_extraction import FeatureExtractor

@pytest.fixture
def sample_packets():
    return [
        {
            'timestamp': '2025-03-20T12:00:00',
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 'TCP',
            'length': 120
        },
        {
            'timestamp': '2025-03-20T12:00:05',
            'src_ip': '192.168.1.101',
            'dst_ip': '8.8.4.4',
            'src_port': 12346,
            'dst_port': 443,
            'protocol': 'TCP',
            'length': 150
        }
    ]

def test_extract_basic_features(feature_extractor, sample_packets):
    df = feature_extractor.extract_basic_features(sample_packets)
    # Verify that the DataFrame contains required columns.
    expected_cols = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'length']
    for col in expected_cols:
        assert col in df.columns

def test_extract_flow_features(feature_extractor, sample_packets):
    df = feature_extractor.extract_basic_features(sample_packets)
    flow_df = feature_extractor.extract_flow_features(df, window_size='1min')
    # Check that flow_df is a DataFrame; if flows exist, it should include the flow_id column.
    assert isinstance(flow_df, pd.DataFrame)
    if not flow_df.empty:
        assert 'flow_id' in flow_df.columns
