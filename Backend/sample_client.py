"""
sample_client.py - Sample client for sending network flow data to the detection API

This script simulates network flow data and sends it to the API for anomaly detection.
"""


import requests
import json
import time
import random
import uuid
from datetime import datetime, timedelta
import argparse

def generate_random_ip():
    """Generate a random IP address"""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}." \
           f"{random.randint(0, 255)}.{random.randint(1, 255)}"

def generate_normal_flow():
    """Generate a normal network flow"""
    protocols = ["TCP", "UDP", "ICMP"]
    common_ports = [80, 443, 22, 25, 53, 123, 8080]
    
    src_ip = generate_random_ip()
    dst_ip = generate_random_ip()
    
    # For normal flows, use common ports and more predictable patterns
    return {
        "timestamp": datetime.now().isoformat(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.choice([random.randint(49152, 65535)] + common_ports),
        "dst_port": random.choice(common_ports),
        "protocol": random.choice(protocols),
        "bytes_sent": random.randint(100, 10000),
        "bytes_received": random.randint(100, 20000),
        "packets_sent": random.randint(1, 100),
        "packets_received": random.randint(1, 200),
        "duration": round(random.uniform(0.1, 5.0), 3)
    }

def generate_anomalous_flow():
    """Generate an anomalous network flow"""
    protocols = ["TCP", "UDP", "ICMP"]
    uncommon_ports = [31337, 12345, 9999, 6666, 4444, 1337]
    
    src_ip = generate_random_ip()
    dst_ip = generate_random_ip()
    
    # For anomalous flows, use unusual patterns and port numbers
    flow = {
        "timestamp": datetime.now().isoformat(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.choice(uncommon_ports + [random.randint(1024, 65535)]),
        "dst_port": random.choice(uncommon_ports + [random.randint(1, 1023)]),
        "protocol": random.choice(protocols),
        "duration": round(random.uniform(0.1, 10.0), 3)
    }
    
    # Choose one of several anomaly patterns
    anomaly_type = random.choice([
        "high_volume",
        "port_scan",
        "data_exfiltration",
        "unusual_protocol"
    ])
    
    if anomaly_type == "high_volume":
        # High volume data transfer
        flow["bytes_sent"] = random.randint(100000, 1000000)
        flow["bytes_received"] = random.randint(10000, 100000)
        flow["packets_sent"] = random.randint(1000, 10000)
        flow["packets_received"] = random.randint(100, 1000)
    
    elif anomaly_type == "port_scan":
        # Port scanning pattern (many packets, little data)
        flow["bytes_sent"] = random.randint(100, 1000)
        flow["bytes_received"] = random.randint(10, 100)
        flow["packets_sent"] = random.randint(100, 1000)
        flow["packets_received"] = random.randint(0, 10)
    
    elif anomaly_type == "data_exfiltration":
        # Data exfiltration (high outbound, low inbound)
        flow["bytes_sent"] = random.randint(50000, 500000)
        flow["bytes_received"] = random.randint(100, 1000)
        flow["packets_sent"] = random.randint(500, 5000)
        flow["packets_received"] = random.randint(1, 10)
    
    else:  # unusual_protocol
        # Unusual protocol behavior
        flow["bytes_sent"] = random.randint(1, 100)
        flow["bytes_received"] = random.randint(1, 100)
        flow["packets_sent"] = random.randint(1, 5)
        flow["packets_received"] = random.randint(1, 5)
        flow["protocol"] = random.choice(["IGMP", "GRE", "ESP"])
    
    return flow

def send_flows(api_url, flows, verbose=False):
    """Send flows to the API"""
    payload = {
        "flows": flows
    }
    
    try:
        if verbose:
            print(f"Sending {len(flows)} flows to {api_url}")
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if verbose:
            print(f"Response: {result['message']}")
            
            if result['alerts']:
                for alert in result['alerts']:
                    print(f"ALERT [{alert['severity'].upper()}]: {alert['description']}")
                    print(f"  Anomaly Score: {alert['anomaly_score']}")
                    print(f"  Alert ID: {alert['alert_id']}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Sample client for the network anomaly detection API")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/detect", help="API endpoint URL")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of flows per batch")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between batches")
    parser.add_argument("--anomaly-prob", type=float, default=0.1, help="Probability of generating anomalous flows")
    parser.add_argument("--inject-anomaly", action="store_true", help="Force inject an anomaly in the first batch")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to send (if not continuous)")
    
    args = parser.parse_args()
    
    try:
        batch_count = 0
        
        while True:
            batch_count += 1
            flows = []
            
            # Generate normal flows
            for i in range(args.batch_size):
                if args.inject_anomaly and batch_count == 1 and i < 3:
                    # Inject anomalies in the first batch if requested
                    flows.append(generate_anomalous_flow())
                elif random.random() < args.anomaly_prob:
                    # Randomly insert anomalies based on probability
                    flows.append(generate_anomalous_flow())
                else:
                    flows.append(generate_normal_flow())
            
            # Send batch to API
            if args.verbose:
                print(f"\nBatch {batch_count}:")
                
            result = send_flows(args.url, flows, args.verbose)
            
            if not args.continuous and batch_count >= args.batches:
                break
                
            # Wait before sending next batch
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nStopping client...")

if __name__ == "__main__":
    main()