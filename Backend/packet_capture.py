"""
packet_capture.py - Network packet capturing module using Scapy

This module provides functions to capture network packets in real-time
using Scapy. It includes functionality to capture packets, filter them
based on various criteria, and save the captured data.
"""

import logging
from scapy.all import sniff, wrpcap
import time
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("packet_capture.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PacketCapture")

class PacketCapture:
    def __init__(self, interface=None, output_dir="./captured_data"):
        """
        Initialize packet capture with specified network interface
        
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
            
        logger.info(f"Packet capture initialized on interface: {interface if interface else 'auto'}")
    
    def packet_callback(self, packet):
        """
        Callback function to process each captured packet
        
        Args:
            packet: Scapy packet object
        """
        # Basic logging - customize with specific packet info as needed
        if hasattr(packet, 'summary'):
            logger.debug(f"Captured: {packet.summary()}")
        return packet
    
    def start_capture(self, count=None, timeout=None, filter_str=None, save_file=None):
        """
        Start packet capture with the specified parameters
        
        Args:
            count (int): Number of packets to capture (None for indefinite)
            timeout (int): Timeout in seconds (None for no timeout)
            filter_str (str): BPF filter string (e.g., "tcp port 80")
            save_file (str): Filename to save the capture (None for auto-generated)
            
        Returns:
            list: Captured packets
        """
        # Auto-generate filename if not provided
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"{self.output_dir}/capture_{timestamp}.pcap"
        
        logger.info(f"Starting packet capture with filter: {filter_str}")
        logger.info(f"Saving to: {save_file}")
        
        try:
            # Start sniffing
            packets = sniff(
                iface=self.interface,
                prn=self.packet_callback,
                filter=filter_str,
                count=count,
                timeout=timeout,
                store=True
            )
            
            # Save the captured packets
            wrpcap(save_file, packets)
            logger.info(f"Captured {len(packets)} packets and saved to {save_file}")
            
            return packets
        
        except KeyboardInterrupt:
            logger.info("Packet capture stopped by user")
            return []
        except Exception as e:
            logger.error(f"Error during packet capture: {str(e)}")
            return []
    
    def continuous_capture(self, interval=60, filter_str=None, max_files=None):
        """
        Continuously capture packets and save them in interval-based files
        
        Args:
            interval (int): Time interval in seconds for each capture file
            filter_str (str): BPF filter string
            max_files (int): Maximum number of files to create (None for unlimited)
        """
        logger.info(f"Starting continuous capture with {interval}s intervals")
        file_count = 0
        
        try:
            while max_files is None or file_count < max_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_file = f"{self.output_dir}/capture_{timestamp}.pcap"
                
                logger.info(f"Capturing packets for {interval} seconds...")
                packets = sniff(
                    iface=self.interface,
                    filter=filter_str,
                    timeout=interval,
                    store=True
                )
                
                # Save this batch
                wrpcap(save_file, packets)
                logger.info(f"Saved {len(packets)} packets to {save_file}")
                
                file_count += 1
        
        except KeyboardInterrupt:
            logger.info("Continuous capture stopped by user")
        except Exception as e:
            logger.error(f"Error during continuous capture: {str(e)}")

# Example usage if run directly
if __name__ == "__main__":
    # Example usage
    capture = PacketCapture()
    # Capture 100 TCP packets
    capture.start_capture(count=100, filter_str="tcp")