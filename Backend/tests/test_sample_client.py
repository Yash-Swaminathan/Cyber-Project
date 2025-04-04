from Backend.sample_client import generate_normal_flow, generate_anomalous_flow


def test_generate_normal_flow():
    flow = generate_normal_flow()
    # Check that all expected keys exist
    expected_keys = {
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", 
        "protocol", "bytes_sent", "bytes_received", 
        "packets_sent", "packets_received", "duration"
    }
    assert expected_keys.issubset(flow.keys())


def test_generate_anomalous_flow():
    flow = generate_anomalous_flow()
    # Verify expected keys exist; anomalous flows may have different specific attributes
    expected_keys = {
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", 
        "protocol", "duration"
    }
    assert expected_keys.issubset(flow.keys())

    # Verify numeric attributes are integers
    numeric_attrs = ["bytes_sent", "bytes_received", "packets_sent", "packets_received"]
    for attr in numeric_attrs:
        if attr in flow:
            assert isinstance(flow[attr], int)
