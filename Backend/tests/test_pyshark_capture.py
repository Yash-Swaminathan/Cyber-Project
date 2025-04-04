from unittest.mock import patch, MagicMock

@patch('Backend.pyshark_capture.os.path.exists', return_value=True)

@patch('Backend.pyshark_capture.pyshark.FileCapture')
def test_read_pcap_file(mock_file_capture, mock_exists, pyshark_capture):
    mock_file_capture_instance = MagicMock()
    mock_file_capture.return_value = mock_file_capture_instance

    # Mock packet clearly as UDP only
    mock_packet = MagicMock()
    mock_packet.sniff_time.isoformat.return_value = "2025-04-03T12:34:56.789Z"
    mock_packet.length = "60"
    mock_packet.highest_layer = "UDP"

    # IP layer mock
    mock_packet.ip.src = "10.0.0.1"
    mock_packet.ip.dst = "10.0.0.2"
    
    # UDP layer mock
    mock_packet.udp.srcport = "53"
    mock_packet.udp.dstport = "33333"
    
    # Ensure TCP is not present
    del mock_packet.tcp

    mock_file_capture_instance.__iter__.return_value = [mock_packet]

    packets_info = pyshark_capture.read_pcap_file("./test_captures/test.pcapng")

    # Assertions
    assert len(packets_info) == 1
    assert packets_info[0]['timestamp'] == "2025-04-03T12:34:56.789Z"
    assert packets_info[0]['length'] == "60"
    assert packets_info[0]['protocol'] == "UDP"
    assert packets_info[0]['src_ip'] == "10.0.0.1"
    assert packets_info[0]['dst_ip'] == "10.0.0.2"
    assert packets_info[0]['src_port'] == "53"
    assert packets_info[0]['dst_port'] == "33333"
    assert packets_info[0]['transport_protocol'] == "UDP"