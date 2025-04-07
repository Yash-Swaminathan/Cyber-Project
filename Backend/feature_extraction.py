"""
feature_extraction.py - Extract and process features from network packets

This module handles the extraction of relevant features from packet data
and processes them into a format suitable for analysis or machine learning.
"""

import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import ipaddress
import os

# Configure logging
os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "feature_extraction.log"), mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger("FeatureExtraction")

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor"""
        logger.info("Feature extractor initialized")
        
        # Define common port mappings for service identification
        self.common_ports = {
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            443: 'HTTPS',
            3389: 'RDP',
            445: 'SMB',
            139: 'NetBIOS',
            21: 'FTP',
            110: 'POP3',
            143: 'IMAP',
            3306: 'MySQL',
            1433: 'MSSQL',
            5432: 'PostgreSQL'
        }
    
    def extract_basic_features(self, packets):
        """
        Extract basic features from a list of packet dictionaries
        
        Args:
            packets (list): List of packet dictionaries (from packet capture modules)
            
        Returns:
            pd.DataFrame: DataFrame with basic packet features
        """
        logger.info(f"Extracting basic features from {len(packets)} packets")
        
        # Initialize DataFrame
        df = pd.DataFrame(packets)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert length to numeric if it exists
        if 'length' in df.columns:
            df['length'] = pd.to_numeric(df['length'], errors='coerce')
            
        logger.info(f"Basic feature extraction complete. DataFrame shape: {df.shape}")
        return df
    
    def extract_flow_features(self, df, window_size='1min'):
        """
        Extract flow-based features from packet data using a time window
        
        Args:
            df (pd.DataFrame): DataFrame with basic packet features
            window_size (str): Pandas time window specification
            
        Returns:
            pd.DataFrame: DataFrame with flow features
        """
        logger.info(f"Extracting flow features using {window_size} window")
        
        # Check that required columns exist
        required_cols = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'length']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for flow extraction: {missing_cols}")
            # Add missing columns with None values
            for col in missing_cols:
                df[col] = None
        
        # Ensure timestamp is datetime type
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Create flow identifier (combination of IPs, ports, and protocol)
        df['flow_id'] = df.apply(
            lambda x: f"{x['src_ip']}:{x['src_port']}-{x['dst_ip']}:{x['dst_port']}-{x['protocol']}"
            if pd.notna(x['src_ip']) and pd.notna(x['dst_ip']) and pd.notna(x['src_port']) and pd.notna(x['dst_port']) and pd.notna(x['protocol'])
            else "unknown",
            axis=1
        )
        
        # Group by time window and flow ID
        flow_features = []
        
        try:
            # Set timestamp as index for resampling
            df_temp = df.set_index('timestamp')
            
            # Group by flow_id and resample by time window
            for flow_id, flow_data in df_temp.groupby('flow_id'):
                # Skip "unknown" flows
                if flow_id == "unknown":
                    continue
                    
                # Resample by time window
                for window_start, window_data in flow_data.resample(window_size):
                    if len(window_data) == 0:
                        continue
                        
                    # Extract first row for flow identification
                    first_row = window_data.iloc[0]
                    
                    # Calculate flow features
                    flow_feature = {
                        'window_start': window_start,
                        'flow_id': flow_id,
                        'src_ip': first_row['src_ip'],
                        'dst_ip': first_row['dst_ip'],
                        'src_port': first_row['src_port'],
                        'dst_port': first_row['dst_port'],
                        'protocol': first_row['protocol'],
                        'packet_count': len(window_data),
                        'bytes_total': window_data['length'].sum() if 'length' in window_data else 0,
                        'bytes_mean': window_data['length'].mean() if 'length' in window_data else 0,
                        'bytes_std': window_data['length'].std() if 'length' in window_data else 0,
                        'bytes_min': window_data['length'].min() if 'length' in window_data else 0,
                        'bytes_max': window_data['length'].max() if 'length' in window_data else 0
                    }
                    
                    flow_features.append(flow_feature)
            
            # Create DataFrame from flow features
            flow_df = pd.DataFrame(flow_features)
            logger.info(f"Extracted {len(flow_features)} flow features")
            
            return flow_df
            
        except Exception as e:
            logger.error(f"Error during flow feature extraction: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['window_start', 'flow_id', 'src_ip', 'dst_ip', 
                                        'src_port', 'dst_port', 'protocol', 'packet_count',
                                        'bytes_total', 'bytes_mean', 'bytes_std', 'bytes_min', 'bytes_max'])
    
    def extract_statistical_features(self, df, group_by='src_ip', window_size='5min'):
        """
        Extract statistical features grouped by a specific field (e.g., source IP)
        
        Args:
            df (pd.DataFrame): DataFrame with basic packet features
            group_by (str): Column to group by
            window_size (str): Pandas time window specification
            
        Returns:
            pd.DataFrame: DataFrame with statistical features
        """
        logger.info(f"Extracting statistical features grouped by {group_by} using {window_size} window")
        
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column required for statistical feature extraction")
            return pd.DataFrame()
            
        if group_by not in df.columns:
            logger.error(f"Group by column '{group_by}' not found in DataFrame")
            return pd.DataFrame()
        
        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Set timestamp as index
        df_temp = df.set_index('timestamp')
        
        # Initialize list to store statistical features
        stat_features = []
        
        try:
            # Group by time window and the specified column
            for window_start, window_data in df_temp.resample(window_size):
                if len(window_data) == 0:
                    continue
                    
                # Group by the specified column within this time window
                for group_value, group_data in window_data.groupby(group_by):
                    if pd.isna(group_value) or group_value == "unknown":
                        continue
                        
                    # Basic counts
                    unique_dst_ips = group_data['dst_ip'].nunique() if 'dst_ip' in group_data else 0
                    unique_dst_ports = group_data['dst_port'].nunique() if 'dst_port' in group_data else 0
                    
                    # Calculate entropy of destination IPs and ports if applicable
                    dst_ip_entropy = 0
                    dst_port_entropy = 0
                    
                    if 'dst_ip' in group_data.columns and not group_data['dst_ip'].isna().all():
                        dst_ip_counts = group_data['dst_ip'].value_counts(normalize=True)
                        dst_ip_entropy = -(dst_ip_counts * np.log2(dst_ip_counts)).sum()
                    
                    if 'dst_port' in group_data.columns and not group_data['dst_port'].isna().all():
                        dst_port_counts = group_data['dst_port'].value_counts(normalize=True)
                        dst_port_entropy = -(dst_port_counts * np.log2(dst_port_counts)).sum()
                    
                    # Create feature dictionary
                    feature = {
                        'window_start': window_start,
                        group_by: group_value,
                        'packet_count': len(group_data),
                        'unique_dst_ips': unique_dst_ips,
                        'unique_dst_ports': unique_dst_ports, 
                        'dst_ip_entropy': dst_ip_entropy,
                        'dst_port_entropy': dst_port_entropy
                    }
                    
                    # Add protocol distribution if available
                    if 'protocol' in group_data.columns:
                        protocol_counts = group_data['protocol'].value_counts()
                        for protocol, count in protocol_counts.items():
                            if not pd.isna(protocol):
                                feature[f'proto_{protocol}'] = count
                    
                    # Add packet length statistics if available
                    if 'length' in group_data.columns:
                        feature.update({
                            'bytes_total': group_data['length'].sum(),
                            'bytes_mean': group_data['length'].mean(),
                            'bytes_std': group_data['length'].std(),
                            'pkts_per_second': len(group_data) / (window_size[0] * 60),  # Assuming window in minutes
                            'bytes_per_second': group_data['length'].sum() / (window_size[0] * 60)
                        })
                    
                    stat_features.append(feature)
            
            # Create DataFrame from statistical features
            stat_df = pd.DataFrame(stat_features)
            logger.info(f"Extracted {len(stat_features)} statistical features")
            
            return stat_df
            
        except Exception as e:
            logger.error(f"Error during statistical feature extraction: {str(e)}")
            return pd.DataFrame()
    
    def normalize_features(self, df, columns=None, method='minmax'):
        """
        Normalize selected numeric features
        
        Args:
            df (pd.DataFrame): DataFrame with features
            columns (list): List of columns to normalize (None for all numeric)
            method (str): Normalization method ('minmax' or 'zscore')
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {method} method")
        
        # Create a copy to avoid modifying the original
        df_norm = df.copy()
        
        # Select columns to normalize
        if columns is None:
            # Select numeric columns only
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        try:
            if method == 'minmax':
                # Min-max normalization
                for col in columns:
                    if col in df_norm.columns:
                        min_val = df_norm[col].min()
                        max_val = df_norm[col].max()
                        
                        # Check for a non-zero range to avoid division by zero
                        if max_val > min_val:
                            df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
                        else:
                            df_norm[f'{col}_norm'] = 0
                            
            elif method == 'zscore':
                # Z-score normalization
                for col in columns:
                    if col in df_norm.columns:
                        mean_val = df_norm[col].mean()
                        std_val = df_norm[col].std()
                        
                        # Check for non-zero standard deviation
                        if std_val > 0:
                            df_norm[f'{col}_norm'] = (df_norm[col] - mean_val) / std_val
                        else:
                            df_norm[f'{col}_norm'] = 0
            
            logger.info(f"Normalized {len(columns)} features")
            return df_norm
            
        except Exception as e:
            logger.error(f"Error during feature normalization: {str(e)}")
            return df
    
    def one_hot_encode(self, df, columns):
        """
        Perform one-hot encoding on categorical columns
        
        Args:
            df (pd.DataFrame): DataFrame with features
            columns (list): List of columns to one-hot encode
            
        Returns:
            pd.DataFrame: DataFrame with one-hot encoded features
        """
        logger.info(f"One-hot encoding categorical features: {columns}")
        
        
        # Create a copy to avoid modifying the original
        df_encoded = df.copy()
        
        try:
            for col in columns:
                if col in df_encoded.columns:
                    # Perform one-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    
                    # Add encoded columns to the dataframe
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    
                    # Remove original column
                    df_encoded = df_encoded.drop(col, axis=1)
            
            logger.info(f"One-hot encoded {len(columns)} features")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error during one-hot encoding: {str(e)}")
            return df

# Example usage if run directly
if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Generate sample data
    sample_data = [
        {
            'timestamp': '2025-03-20T12:00:00',
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 45123,
            'dst_port': 443,
            'protocol': 'TCP',
            'length': 120
        },
        {
            'timestamp': '2025-03-20T12:00:01',
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 45123,
            'dst_port': 443,
            'protocol': 'TCP',
            'length': 250
        },
        {
            'timestamp': '2025-03-20T12:00:02',
            'src_ip': '192.168.1.101',
            'dst_ip': '192.168.1.1',
            'src_port': 53123,
            'dst_port': 53,
            'protocol': 'UDP',
            'length': 80
        }
    ]
    
    # Extract features
    df = extractor.extract_basic_features(sample_data)
    flow_df = extractor.extract_flow_features(df, window_size='1min')
    print(flow_df.head())