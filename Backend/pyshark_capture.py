"""
pyshark_capture.py - Network packet capturing module using PyShark

This module provides an alternative to Scapy for packet capture using PyShark,
which is a Python wrapper for tshark (Wireshark CLI). PyShark provides access
to Wireshark's powerful dissection capabilities with Python ease of use.
"""


import pyshark
import os
import logging
from datetime import datetime

# Configure logging
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "pyshark_capture.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PySharkCapture")

class PySharkCapture:
    def __init__(self, interface=None, output_dir="./captured_data"):
        """
        Initialize PyShark packet capture with specified network interface
        
        Args:
            interface (str): Network interface to capture from (None for auto-select)
            output_dir (str): Directory to save captured packet data
        """
        self.interface = interface
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
            
        logger.info(f"PyShark capture initialized on interface: {interface if interface else 'auto'}")
    
    def start_live_capture(self, capture_time=60, bpf_filter=None, save_file=None):
        """
        Start live packet capture using PyShark
        
        Args:
            capture_time (int): Duration of capture in seconds
            bpf_filter (str): BPF filter string
            save_file (str): Filename to save the capture (None for auto-generated)
            
        Returns:
            list: Captured packets information
        """
        # Auto-generate filename if not provided
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"{self.output_dir}/pyshark_{timestamp}.pcapng"
        
        logger.info(f"Starting PyShark live capture with filter: {bpf_filter}")
        logger.info(f"Saving to: {save_file}")
        
        # Packet container
        packets_info = []
        
        try:
            # Create capture object
            if self.interface:
                capture = pyshark.LiveCapture(
                    interface=self.interface,
                    output_file=save_file,
                    bpf_filter=bpf_filter
                )
            else:
                # Auto-select interface
                capture = pyshark.LiveCapture(
                    output_file=save_file,
                    bpf_filter=bpf_filter
                )
            
            # Set a timeout for the capture
            capture.set_debug()
            
            # Capture packets for the specified time
            logger.info(f"Capturing packets for {capture_time} seconds...")
            capture.sniff(timeout=capture_time)
            
            # Process captured packets
            for i, packet in enumerate(capture):
                packet_info = self._extract_basic_info(packet)
                packets_info.append(packet_info)
                if i % 100 == 0:  # Log every 100 packets
                    logger.debug(f"Processed {i} packets")
            
            logger.info(f"Captured and processed {len(packets_info)} packets")
            return packets_info
        
        except KeyboardInterrupt:
            logger.info("PyShark capture stopped by user")
            return packets_info
        except Exception as e:
            logger.error(f"Error during PyShark capture: {str(e)}")
            return packets_info
    
    def _extract_basic_info(self, packet):
        """
        Extract basic information from a packet
        
        Args:
            packet: PyShark packet object
            
        Returns:
            dict: Dictionary with basic packet information
        """
        packet_info = {
            'timestamp': packet.sniff_time.isoformat() if hasattr(packet, 'sniff_time') else None,
            'length': packet.length if hasattr(packet, 'length') else None,
            'protocol': packet.highest_layer if hasattr(packet, 'highest_layer') else None
        }
        
        # Extract IP information if present
        if hasattr(packet, 'ip'):
            packet_info.update({
                'src_ip': packet.ip.src if hasattr(packet.ip, 'src') else None,
                'dst_ip': packet.ip.dst if hasattr(packet.ip, 'dst') else None
            })
        
        # Extract transport layer (TCP/UDP) information if present
        if hasattr(packet, 'tcp'):
            packet_info.update({
                'src_port': packet.tcp.srcport if hasattr(packet.tcp, 'srcport') else None,
                'dst_port': packet.tcp.dstport if hasattr(packet.tcp, 'dstport') else None,
                'transport_protocol': 'TCP'
            })
        elif hasattr(packet, 'udp'):
            packet_info.update({
                'src_port': packet.udp.srcport if hasattr(packet.udp, 'srcport') else None,
                'dst_port': packet.udp.dstport if hasattr(packet.udp, 'dstport') else None,
                'transport_protocol': 'UDP'
            })
        
        return packet_info
    
    def read_pcap_file(self, pcap_file):
        """
        Read packets from a PCAP file
        
        Args:
            pcap_file (str): Path to the PCAP file
            
        Returns:
            list: Extracted packet information
        """
        if not os.path.exists(pcap_file):
            logger.error(f"PCAP file not found: {pcap_file}")
            return []
        
        logger.info(f"Reading PCAP file: {pcap_file}")
        packets_info = []
        
        try:
            # Create file capture object
            file_capture = pyshark.FileCapture(pcap_file)
            
            # Process packets
            for i, packet in enumerate(file_capture):
                packet_info = self._extract_basic_info(packet)
                packets_info.append(packet_info)
                if i % 1000 == 0:  # Log every 1000 packets
                    logger.debug(f"Processed {i} packets from file")
            
            logger.info(f"Read and processed {len(packets_info)} packets from file")
            return packets_info
        
        except Exception as e:
            logger.error(f"Error reading PCAP file: {str(e)}")
            return []

# Example usage if run directly
if __name__ == "__main__":
    # Example usage
    capture = PySharkCapture()
    # Capture HTTP packets for 30 seconds
    capture.start_live_capture(capture_time=30, bpf_filter="tcp port 80 or tcp port 443")