# Cybersecurity Intrusion Detection System - Backend

## Overview
This project develops a real-time backend system for an Intrusion Detection System (IDS) that monitors network traffic and detects anomalies. The backend is responsible for ingesting live data, preprocessing network packets, and applying machine learning models to identify potential cybersecurity threats.

## Features
- **Real-Time Detection Engine:** API endpoints built with Flask/FastAPI for live data ingestion and prediction.
- **Anomaly Detection:** Integration of traditional ML (Isolation Forest, One-Class SVM) and deep learning models (autoencoders, LSTM) to flag deviations from normal network behavior.
- **Alerting & Logging:** Robust logging and alerting mechanisms to notify operators of suspicious activity.
- **Testing:** Comprehensive unit and integration tests using pytest. - pytest Fixtures in Conftest.py files
- **Deployment:** Optional containerization with Docker for consistent deployment across environments.

## System Architecture

### Data Collection & Preprocessing
- **Packet Capture:** Utilize tools like Scapy or PyShark to capture live network traffic.
- **Feature Extraction:** Parse raw packets to extract key features such as IP addresses, ports, protocols, packet sizes, and timestamps. Use Pandas and NumPy for data structuring and normalization.

### Model Integration
- **Traditional Machine Learning:** Start with models like Isolation Forest for quick anomaly detection.
- **Deep Learning:** Develop models using TensorFlow/Keras (e.g., autoencoders or LSTM networks) to learn and identify subtle deviations in network patterns.

### Backend API
- **Endpoints:** Create RESTful endpoints (e.g., `/predict`) that receive preprocessed network data and return anomaly detection results.
- **Data Pipeline:** Ensure that live data undergoes the same preprocessing as the training data to maintain model consistency.

### Alerting & Logging
- **Logging:** Implement Python logging to record events and detected anomalies.
- **Notifications:** Integrate alerting systems (email, dashboard updates, or messaging platforms) to inform operators in real time.

### Must Use Python Version 3.10
