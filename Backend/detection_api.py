"""
detection_api.py - Real-time network traffic anomaly detection API

This module implements a FastAPI backend for real-time network traffic anomaly detection,
ingesting live data, processing it, and generating alerts when anomalies are detected.
"""

##############################################    MADE SAMPLE_CLIENT.PY RUN WHEN THIS IS STARTED
import os
import json
import time
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import asyncio
from contextlib import asynccontextmanager
import runpy


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import our Deep Learning Detector
from Backend.Deep_Learning.deep_learning import DeepLearningDetector  # type: ignore

# Configure logging
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "detection_api.log"), mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger("DetectionAPI")

# Load configuration
CONFIG_PATH = "config.json"

def load_config():
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {CONFIG_PATH} not found. Using default configuration.")
        return {
            "model_path": "models/deep_autoencoder.h5",
            "alert_threshold": 0.8,
            "batch_size": 100,
            "alert_cooldown_seconds": 300,
            "alerting": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your-email@gmail.com",
                    "password": "your-app-password",
                    "sender": "your-email@gmail.com",
                    "recipients": ["admin@example.com"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
                }
            }
        }

# Global variables
config = load_config()
detector = None
last_alert_time = 0

# Pydantic models for API
class NetworkFlowData(BaseModel):
    """Single network flow data point"""
    timestamp: str = Field(..., description="Timestamp of the flow")
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(..., description="Source port")
    dst_port: int = Field(..., description="Destination port")
    protocol: str = Field(..., description="Protocol (TCP, UDP, etc.)")
    bytes_sent: int = Field(..., description="Bytes sent")
    bytes_received: int = Field(..., description="Bytes received")
    packets_sent: int = Field(..., description="Packets sent")
    packets_received: int = Field(..., description="Packets received")
    duration: float = Field(..., description="Flow duration in seconds")
    additional_features: Optional[Dict[str, Any]] = Field(default=None, description="Additional features")

class NetworkFlowBatch(BaseModel):
    """Batch of network flow data points"""
    flows: List[NetworkFlowData] = Field(..., description="List of network flows")

class AnomalyAlert(BaseModel):
    """Anomaly alert response model"""
    alert_id: str = Field(..., description="Unique alert ID")
    timestamp: str = Field(..., description="Alert timestamp")
    severity: str = Field(..., description="Alert severity (low, medium, high)")
    anomaly_score: float = Field(..., description="Anomaly score")
    description: str = Field(..., description="Alert description")
    affected_flows: List[Dict[str, Any]] = Field(..., description="Affected network flows")

class AlertResponse(BaseModel):
    """API response with alerts"""
    alerts: List[AnomalyAlert] = Field(default_factory=list, description="List of anomaly alerts")
    message: str = Field(..., description="Response message")
    timestamp: str = Field(..., description="Response timestamp")
    
# FastAPI startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    try:
        model_path = config["model_path"]
        logger.info(f"Loading model from {model_path}")
        detector = DeepLearningDetector.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Build a dummy model for testing instead of leaving detector as None
        detector = DeepLearningDetector(model_type='autoencoder', input_shape=(18,), model_params={'encoding_dim': 10})
        detector.build_model()
        detector.threshold = config.get("alert_threshold", 0.8)
        logger.info("Dummy model built successfully")
    yield
    logger.info("API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Network Anomaly Detection API",
    description="Real-time API for detecting anomalies in network traffic using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_flow_data(flows: List[NetworkFlowData]) -> pd.DataFrame:
    """
    Preprocess network flow data for the model
    
    Args:
        flows (List[NetworkFlowData]): List of network flow data
        
    Returns:
        pd.DataFrame: Preprocessed data ready for the model, along with the original DataFrame.
    """
    # Convert to DataFrame
    records = []
    for flow in flows:
        record = {
            "timestamp": flow.timestamp,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "bytes_sent": flow.bytes_sent,
            "bytes_received": flow.bytes_received,
            "packets_sent": flow.packets_sent,
            "packets_received": flow.packets_received,
            "duration": flow.duration,
        }
        
        # Add additional features if available
        if flow.additional_features:
            for key, value in flow.additional_features.items():
                record[key] = value
                
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Convert timestamp to datetime with error coercion (invalid formats become NaT)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.fillna(0)
    
    # Extract time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Convert categorical features
    df['is_tcp'] = (df['protocol'] == 'TCP').astype(int)
    df['is_udp'] = (df['protocol'] == 'UDP').astype(int)
    df['is_icmp'] = (df['protocol'] == 'ICMP').astype(int)
    
    # Calculate derived features
    df['bytes_per_packet_sent'] = df['bytes_sent'] / df['packets_sent'].replace(0, 1)
    df['bytes_per_packet_recv'] = df['bytes_received'] / df['packets_received'].replace(0, 1)
    df['total_bytes'] = df['bytes_sent'] + df['bytes_received']
    df['total_packets'] = df['packets_sent'] + df['packets_received']
    df['bytes_ratio'] = df['bytes_sent'] / df['total_bytes'].replace(0, 1)
    df['packets_ratio'] = df['packets_sent'] / df['total_packets'].replace(0, 1)
    
    # Drop non-numeric columns that the model doesn't use
    drop_cols = ['timestamp', 'src_ip', 'dst_ip', 'protocol']
    X = df.drop(columns=drop_cols)
    return X, df


def generate_alert(anomaly_score: float, flow_data: pd.DataFrame, threshold: float) -> AnomalyAlert:
    """
    Generate an anomaly alert based on the anomaly score and flow data
    
    Args:
        anomaly_score (float): Anomaly score from the model
        flow_data (pd.DataFrame): Original flow data
        threshold (float): Threshold for determining severity
        
    Returns:
        AnomalyAlert: Alert object
    """
    # Determine severity
    if anomaly_score > threshold * 1.5:
        severity = "high"
    elif anomaly_score > threshold * 1.2:
        severity = "medium"
    else:
        severity = "low"
    
    # Create alert object
    alert = AnomalyAlert(
        alert_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        severity=severity,
        anomaly_score=float(anomaly_score),
        description=f"Detected network traffic anomaly with score {anomaly_score:.4f}",
        affected_flows=flow_data.to_dict(orient='records')
    )
    
    return alert

async def send_email_alert(alert: AnomalyAlert):
    """
    Send an email alert
    
    Args:
        alert (AnomalyAlert): Alert to send
    """
    if not config["alerting"]["email"]["enabled"]:
        logger.info("Email alerting is disabled")
        return
    
    try:
        smtp_config = config["alerting"]["email"]
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_config["sender"]
        msg['To'] = ', '.join(smtp_config["recipients"])
        msg['Subject'] = f"[{alert.severity.upper()}] Network Anomaly Detected"
        
        # Create message body
        body = f"""
        <html>
        <body>
            <h2>Network Anomaly Alert</h2>
            <p><strong>Alert ID:</strong> {alert.alert_id}</p>
            <p><strong>Timestamp:</strong> {alert.timestamp}</p>
            <p><strong>Severity:</strong> {alert.severity.upper()}</p>
            <p><strong>Anomaly Score:</strong> {alert.anomaly_score:.4f}</p>
            <p><strong>Description:</strong> {alert.description}</p>
            <h3>Affected Flows:</h3>
            <table border="1">
                <tr>
                    <th>Source IP</th>
                    <th>Destination IP</th>
                    <th>Source Port</th>
                    <th>Destination Port</th>
                    <th>Protocol</th>
                </tr>
        """
        
        # Add flow data
        for flow in alert.affected_flows[:5]:  # Limit to first 5 flows
            body += f"""
                <tr>
                    <td>{flow.get('src_ip', 'N/A')}</td>
                    <td>{flow.get('dst_ip', 'N/A')}</td>
                    <td>{flow.get('src_port', 'N/A')}</td>
                    <td>{flow.get('dst_port', 'N/A')}</td>
                    <td>{flow.get('protocol', 'N/A')}</td>
                </tr>
            """
            
        body += """
            </table>
            <p>Please investigate this anomaly immediately.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"])
        server.starttls()
        server.login(smtp_config["username"], smtp_config["password"])
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent for alert ID {alert.alert_id}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")

async def send_slack_alert(alert: AnomalyAlert):
    """
    Send a Slack alert
    
    Args:
        alert (AnomalyAlert): Alert to send
    """
    if not config["alerting"]["slack"]["enabled"]:
        logger.info("Slack alerting is disabled")
        return
    
    try:
        webhook_url = config["alerting"]["slack"]["webhook_url"]
        
        # Create message payload
        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 {alert.severity.upper()} Network Anomaly Detected"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Alert ID:*\n{alert.alert_id}"},
                        {"type": "mrkdwn", "text": f"*Timestamp:*\n{alert.timestamp}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.upper()}"},
                        {"type": "mrkdwn", "text": f"*Anomaly Score:*\n{alert.anomaly_score:.4f}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:*\n{alert.description}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Affected Flows:*"
                    }
                }
            ]
        }
        
        # Add flow data (first 3 flows only to avoid message size limits)
        flow_text = ""
        for i, flow in enumerate(alert.affected_flows[:3]):
            flow_text += f"*Flow {i+1}:* {flow.get('src_ip', 'N/A')}:{flow.get('src_port', 'N/A')} → " \
                         f"{flow.get('dst_ip', 'N/A')}:{flow.get('dst_port', 'N/A')} ({flow.get('protocol', 'N/A')})\n"
        
        payload["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": flow_text
            }
        })
        
        # Send request to Slack webhook
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        
        logger.info(f"Slack alert sent for alert ID {alert.alert_id}")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {str(e)}")

async def send_alerts(alert: AnomalyAlert):
    """
    Send alerts via all configured channels
    
    Args:
        alert (AnomalyAlert): Alert to send
    """
    global last_alert_time
    
    # Check cooldown period
    current_time = time.time()
    cooldown = config.get("alert_cooldown_seconds", 300)  # Default 5 minutes
    
    if current_time - last_alert_time < cooldown:
        logger.info(f"Alert cooldown period active. Skipping alerts for {cooldown - (current_time - last_alert_time):.1f} more seconds")
        return
    
    # Update last alert time
    last_alert_time = current_time
    
    # Log the alert
    logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.description} (Score: {alert.anomaly_score:.4f})")
    
    # Send alerts in parallel
    await asyncio.gather(
        send_email_alert(alert),
        send_slack_alert(alert)
    )

# In detection_api.py, enhance the process_flows function with better error handling:

async def process_flows(flows: List[NetworkFlowData], background_tasks: BackgroundTasks):
    """
    Process network flow data and detect anomalies
    
    Args:
        flows (List[NetworkFlowData]): List of network flow data
        background_tasks (BackgroundTasks): FastAPI background tasks
        
    Returns:
        AlertResponse: Response with alerts
    """
    if detector is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess data
        X, original_df = preprocess_flow_data(flows)
        X_array = X.values.astype('float32')
        logger.info(f"Input shape to model: {X_array.shape}")
        
        # Normalize input using MinMaxScaler so values are in [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler_input = MinMaxScaler()
        X_scaled = scaler_input.fit_transform(X_array)
        
        # Get anomaly scores
        anomaly_scores = detector.get_anomaly_scores(X_scaled)
        
        # Check for anomalies
        alerts = []
        threshold = detector.threshold or config.get("alert_threshold", 0.8)
        
        # Check if any score exceeds threshold
        if np.any(anomaly_scores > threshold):
            # Get indices of anomalous flows
            anomaly_indices = np.where(anomaly_scores > threshold)[0]
            
            # Group anomalies into a single alert
            max_score = anomaly_scores[anomaly_indices].max()
            affected_flows = original_df.iloc[anomaly_indices]
            
            # Generate alert
            alert = generate_alert(max_score, affected_flows, threshold)
            alerts.append(alert)
            
            # Send alerts in background
            background_tasks.add_task(send_alerts, alert)
        
        # Create response
        response = AlertResponse(
            alerts=alerts,
            message=f"Processed {len(flows)} network flows. Detected {len(alerts)} anomalies.",
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error during flow processing: {str(e)}")
        # For debugging purposes, let's see the full traceback
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during flow processing: {str(e)}")


@app.post("/api/v1/detect", response_model=AlertResponse)
async def detect_anomalies(
    batch: NetworkFlowBatch,
    background_tasks: BackgroundTasks
):
    """
    Process a batch of network flows and detect anomalies
    """
    return await process_flows(batch.flows, background_tasks)

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy" if detector is not None else "degraded",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/config")
async def get_config():
    """
    Get current configuration (sanitized)
    """
    # Create a copy and remove sensitive information
    safe_config = config.copy()
    if "alerting" in safe_config:
        if "email" in safe_config["alerting"]:
            if "password" in safe_config["alerting"]["email"]:
                safe_config["alerting"]["email"]["password"] = "********"
    
    return safe_config

@app.post("/api/v1/test-alert")
async def test_alert(background_tasks: BackgroundTasks):
    """
    Send a test alert
    """
    # Create test flow data
    test_flow = NetworkFlowData(
        timestamp=datetime.now().isoformat(),
        src_ip="192.168.1.1",
        dst_ip="10.0.0.1",
        src_port=12345,
        dst_port=443,
        protocol="TCP",
        bytes_sent=1024,
        bytes_received=2048,
        packets_sent=10,
        packets_received=15,
        duration=1.5
    )
    
    # Create test alert
    alert = AnomalyAlert(
        alert_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        severity="medium",
        anomaly_score=0.95,
        description="This is a test alert",
        affected_flows=[test_flow.model_dump()]
    )
    
    # Send alerts in background
    background_tasks.add_task(send_alerts, alert)
    
    return {
        "message": "Test alert sent",
        "alert_id": alert.alert_id
    }

if __name__ == "__main__":
    import subprocess
    import sys
    import time

    logger.info("Starting API server...")

    # Start the FastAPI server as a subprocess
    api_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "Backend.detection_api:app", "--host", "127.0.0.1", "--port", "8000"
    ])

    # Wait a few seconds to allow the server to start
    time.sleep(3)

    logger.info("Starting sample client...")

    # Start the sample client in another subprocess
    client_proc = subprocess.Popen([
        sys.executable, "Backend/sample_client.py", "--inject-anomaly", "-v"
    ])

    # Optional: wait for the API process to end before exiting this script
    try:
        api_proc.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_proc.terminate()
        client_proc.terminate()
