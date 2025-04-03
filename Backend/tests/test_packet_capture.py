import os
import pytest
from Backend.packet_capture import PacketCapture

def test_packet_capture_initialization(packet_capture_instance):
    # Verify that PacketCapture initializes correctly with no specified interface.
    assert packet_capture_instance.interface is None
    assert os.path.exists(packet_capture_instance.output_dir)

def test_start_capture(packet_capture_instance):
    # Use a short timeout so that sniff stops quickly.
    packets = packet_capture_instance.start_capture(count=0, timeout=1, filter_str="tcp")
    # Instead of checking for isinstance(packets, list),
    # we can check that it's iterable and that its length is 0.
    assert hasattr(packets, '__iter__')
    assert len(list(packets)) == 0

