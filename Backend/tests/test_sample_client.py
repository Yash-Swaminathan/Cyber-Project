from Backend import sample_client

def test_generate_normal_flow():
    flow = sample_client.generate_normal_flow()
    # Check that all expected keys exist
    expected_keys = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "bytes_sent", "bytes_received", "packets_sent", "packets_received", "duration"}
    assert expected_keys.issubset(flow.keys())

def test_generate_anomalous_flow():
    flow = sample_client.generate_anomalous_flow()
    # Verify expected keys exist; anomalous flows may include extra keys based on anomaly type
    expected_keys = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "duration"}
    assert expected_keys.issubset(flow.keys())
    # If anomaly-specific keys exist, check their types
    if "bytes_sent" in flow:
        assert isinstance(flow["bytes_sent"], int)
