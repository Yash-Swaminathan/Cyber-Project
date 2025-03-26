"""
feature_engineering.py - Advanced feature engineering for network traffic analysis

This module implements sophisticated feature engineering techniques for network
traffic analysis, focusing on deriving meaningful features from packet data
for intrusion detection purposes.
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import entropy
from datetime import datetime, timedelta
import ipaddress
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("feature_engineering.log"), logging.StreamHandler()]
)
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer"""
        logger.info("Feature engineer initialized")
        
        # Default time windows for aggregation (in seconds)
        self.time_windows = [5, 30, 60, 300, 600]  # 5s, 30s, 1min, 5min, 10min
        
        # Known suspicious port ranges
        self.suspicious_ports = set([
            # Commonly exploited services
            21, 22, 23, 25, 53, 80, 111, 135, 139, 445, 
            3306, 3389, 5432, 5900, 6667, 
            # Ephemeral port ranges often used in scanning
            *range(1024, 1050),  
            # Backdoor ports
            31337, 12345, 54321
        ])
    
    def categorize_ip(self, ip):
        """
        Categorize IP address as internal/external and check if it's in special ranges
        
        Args:
            ip (str): IP address
            
        Returns:
            dict: Dictionary with IP categorization
        """
        if not ip or pd.isna(ip):
            return {
                'is_internal': False,
                'is_multicast': False,
                'is_reserved': False,
                'is_loopback': False
            }
            
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check various IP properties
            is_private = ip_obj.is_private
            is_multicast = ip_obj.is_multicast
            is_reserved = ip_obj.is_reserved
            is_loopback = ip_obj.is_loopback
            
            result = {
                'is_internal': is_private,
                'is_multicast': is_multicast,
                'is_reserved': is_reserved,
                'is_loopback': is_loopback
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error categorizing IP {ip}: {str(e)}")
            return {
                'is_internal': False,
                'is_multicast': False,
                'is_reserved': False,
                'is_loopback': False
            }
    
    def is_suspicious_port(self, port):
        """
        Check if a port is in the suspicious port list
        
        Args:
            port (int): Port number to check
            
        Returns:
            bool: True if port is suspicious, False otherwise
        """
        if not port or pd.isna(port):
            return False
        
        try:
            return int(port) in self.suspicious_ports
        except:
            return False
    
    def extract_base_features(self, df):
        """
        Extract base features from raw packet data
        
        Args:
            df (pandas.DataFrame): DataFrame containing raw packet data
            
        Returns:
            pandas.DataFrame: DataFrame with extracted base features
        """
        logger.info(f"Extracting base features from {len(df)} packets")
        
        # Ensure timestamp is in datetime format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add IP categorization features
        if 'src_ip' in df.columns:
            src_ip_cats = df['src_ip'].apply(self.categorize_ip)
            df['src_ip_internal'] = src_ip_cats.apply(lambda x: x['is_internal'])
            df['src_ip_multicast'] = src_ip_cats.apply(lambda x: x['is_multicast'])
            df['src_ip_reserved'] = src_ip_cats.apply(lambda x: x['is_reserved'])
            df['src_ip_loopback'] = src_ip_cats.apply(lambda x: x['is_loopback'])
            
        if 'dst_ip' in df.columns:
            dst_ip_cats = df['dst_ip'].apply(self.categorize_ip)
            df['dst_ip_internal'] = dst_ip_cats.apply(lambda x: x['is_internal'])
            df['dst_ip_multicast'] = dst_ip_cats.apply(lambda x: x['is_multicast'])
            df['dst_ip_reserved'] = dst_ip_cats.apply(lambda x: x['is_reserved'])
            df['dst_ip_loopback'] = dst_ip_cats.apply(lambda x: x['is_loopback'])
        
        # Add port suspiciousness
        if 'src_port' in df.columns:
            df['src_port_suspicious'] = df['src_port'].apply(self.is_suspicious_port)
            
        if 'dst_port' in df.columns:
            df['dst_port_suspicious'] = df['dst_port'].apply(self.is_suspicious_port)
        
        # Add packet size features if available
        if 'packet_size' in df.columns:
            # Create size categories
            df['size_small'] = df['packet_size'] < 64
            df['size_medium'] = (df['packet_size'] >= 64) & (df['packet_size'] < 1024)
            df['size_large'] = df['packet_size'] >= 1024
        
        # Extract time-based features
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
            df['is_business_hours'] = (df['hour_of_day'] >= 8) & (df['hour_of_day'] < 18)
        
        logger.info(f"Base feature extraction complete. Generated {len(df.columns)} features")
        return df
    
    def aggregate_by_time_window(self, df, groupby_cols, time_windows=None):
        """
        Aggregate packet details over time windows
        
        Args:
            df (pandas.DataFrame): DataFrame with packet data including timestamp
            groupby_cols (list): List of columns to group by besides time window
            time_windows (list, optional): List of time windows in seconds. Default is self.time_windows
            
        Returns:
            pandas.DataFrame: DataFrame with aggregated features
        """
        if time_windows is None:
            time_windows = self.time_windows
        
        logger.info(f"Aggregating data over {len(time_windows)} time windows: {time_windows}")
        
        if 'timestamp' not in df.columns:
            logger.error("Cannot aggregate by time window: 'timestamp' column not found")
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create a list to hold all aggregated DataFrames
        aggregated_dfs = []
        
        for window in time_windows:
            logger.info(f"Processing {window} second window")
            
            # Create window label
            window_label = f"{window}s"
            
            # Create rolling window
            df['window_end'] = df['timestamp']
            window_size = pd.Timedelta(seconds=window)
            
            # Group by window end and other columns
            all_groupby_cols = groupby_cols + ['window_end']
            
            # Find data within window for each group
            df_with_window = df.copy()
            df_with_window['window_start'] = df_with_window['window_end'] - window_size
            
            # Define aggregation functions
            agg_funcs = {
                'packet_size': ['count', 'sum', 'mean', 'min', 'max', 'std'],
                'src_ip': ['nunique'],
                'dst_ip': ['nunique'],
                'src_port': ['nunique'],
                'dst_port': ['nunique'],
                'src_port_suspicious': ['sum'],
                'dst_port_suspicious': ['sum'],
                'src_ip_internal': ['mean'],
                'dst_ip_internal': ['mean'],
            }
            
            # Filter columns that exist in the dataframe
            valid_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() 
                             if col in df_with_window.columns}
            
            # Perform aggregation
            window_df = pd.DataFrame()
            
            try:
                # Group by the specified columns and window end time
                grouped = df_with_window.groupby(all_groupby_cols)
                
                # Apply aggregation functions
                window_df = grouped.agg(valid_agg_funcs)
                
                # Flatten multi-level column names
                window_df.columns = [f"{col}_{func}_{window_label}" 
                                   for col, func in window_df.columns]
                
                # Reset index to get groupby columns back
                window_df = window_df.reset_index()
                
                # Rename window_end back to timestamp
                window_df = window_df.rename(columns={'window_end': 'timestamp'})
                
                # Calculate packet rate
                if ('packet_size_count_' + window_label) in window_df.columns:
                    window_df[f'packet_rate_{window_label}'] = (
                        window_df['packet_size_count_' + window_label] / window
                    )
                
                aggregated_dfs.append(window_df)
                
            except Exception as e:
                logger.error(f"Error during {window}s window aggregation: {str(e)}")
                continue
        
        # Merge all aggregated DataFrames
        if not aggregated_dfs:
            logger.warning("No aggregations were successful")
            return df
        
        # Start with the first DataFrame
        result_df = aggregated_dfs[0]
        
        # Merge with the rest on groupby columns and timestamp
        merge_cols = groupby_cols + ['timestamp']
        for agg_df in aggregated_dfs[1:]:
            result_df = pd.merge(result_df, agg_df, on=merge_cols, how='outer')
        
        logger.info(f"Time window aggregation complete. Generated {len(result_df.columns)} features")
        return result_df
    
    def calculate_entropy_features(self, df, columns, time_windows=None):
        """
        Calculate entropy-based features for specified columns
        
        Args:
            df (pandas.DataFrame): DataFrame with packet data
            columns (list): List of columns to calculate entropy for
            time_windows (list, optional): List of time windows in seconds
            
        Returns:
            pandas.DataFrame: DataFrame with added entropy features
        """
        if time_windows is None:
            time_windows = self.time_windows
            
        logger.info(f"Calculating entropy features for {len(columns)} columns")
        
        result_df = df.copy()
        
        for window in time_windows:
            window_label = f"{window}s"
            window_size = pd.Timedelta(seconds=window)
            
            for column in columns:
                if column not in df.columns:
                    continue
                    
                # Create feature name
                feature_name = f"{column}_entropy_{window_label}"
                
                try:
                    # Group values within time window
                    result_df[feature_name] = df.apply(
                        lambda row: self._calculate_entropy_for_row(
                            df, row['timestamp'], window_size, column
                        ),
                        axis=1
                    )
                except Exception as e:
                    logger.error(f"Error calculating entropy for {column}: {str(e)}")
        
        return result_df
    
    def _calculate_entropy_for_row(self, df, timestamp, window_size, column):
        """Helper method to calculate entropy for values in a time window"""
        start_time = timestamp - window_size
        window_data = df[(df['timestamp'] >= start_time) & 
                        (df['timestamp'] <= timestamp)][column]
        
        if len(window_data) <= 1:
            return 0
            
        # Calculate value frequencies
        value_counts = window_data.value_counts(normalize=True)
        
        # Calculate entropy
        return entropy(value_counts)
    
    def normalize_features(self, df, method='standard', columns=None):
        """
        Normalize numerical features using specified method
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            method (str): Normalization method ('standard', 'minmax')
            columns (list, optional): List of columns to normalize. If None, all numeric columns.
            
        Returns:
            tuple: (Normalized DataFrame, fitted scaler)
        """
        logger.info(f"Normalizing features using {method} method")
        
        # Copy dataframe to avoid modifying original
        result_df = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
            logger.info(f"Auto-selected {len(columns)} numeric columns for normalization")
        
        # Filter out columns that don't exist
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            logger.warning("No valid columns to normalize")
            return result_df, None
        
        # Select scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.error(f"Unknown normalization method: {method}")
            return result_df, None
        
        # Fit and transform
        try:
            normalized_values = scaler.fit_transform(result_df[valid_columns])
            
            # Update DataFrame with normalized values
            for i, col in enumerate(valid_columns):
                result_df[col] = normalized_values[:, i]
                
            logger.info(f"Successfully normalized {len(valid_columns)} columns")
            
        except Exception as e:
            logger.error(f"Error during normalization: {str(e)}")
        
        return result_df, scaler
    
    def encode_categorical_features(self, df, columns=None, method='onehot'):
        """
        Encode categorical features
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            columns (list, optional): List of categorical columns to encode
            method (str): Encoding method ('onehot', 'label', 'binary')
            
        Returns:
            tuple: (Encoded DataFrame, fitted encoder)
        """
        logger.info(f"Encoding categorical features using {method} method")
        
        # Copy dataframe to avoid modifying original
        result_df = df.copy()
        
        # If no columns specified, use all categorical/object columns
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            logger.info(f"Auto-selected {len(columns)} categorical columns for encoding")
        
        # Filter out columns that don't exist
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            logger.warning("No valid columns to encode")
            return result_df, None
        
        # Perform encoding based on method
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            try:
                # Fit encoder
                encoded_values = encoder.fit_transform(result_df[valid_columns])
                
                # Get feature names
                feature_names = encoder.get_feature_names_out(valid_columns)
                
                # Create DataFrame with encoded values
                encoded_df = pd.DataFrame(
                    encoded_values,
                    columns=feature_names,
                    index=result_df.index
                )
                
                # Drop original columns and concatenate encoded ones
                result_df = result_df.drop(columns=valid_columns)
                result_df = pd.concat([result_df, encoded_df], axis=1)
                
                logger.info(f"Successfully one-hot encoded {len(valid_columns)} columns into {len(feature_names)} features")
                
            except Exception as e:
                logger.error(f"Error during one-hot encoding: {str(e)}")
                
        elif method == 'label':
            # Label encoding using pandas factorize
            encoder = {}
            
            for col in valid_columns:
                try:
                    labels, unique = pd.factorize(result_df[col])
                    result_df[f"{col}_encoded"] = labels
                    encoder[col] = dict(zip(unique, range(len(unique))))
                    
                    # Drop original column
                    result_df = result_df.drop(columns=[col])
                    
                except Exception as e:
                    logger.error(f"Error during label encoding of {col}: {str(e)}")
            
            logger.info(f"Successfully label-encoded {len(encoder)} columns")
            
        else:
            logger.error(f"Unknown encoding method: {method}")
            return result_df, None
        
        return result_df, encoder
    
    def create_interaction_features(self, df, feature_pairs):
        """
        Create interaction features between pairs of numerical features
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            feature_pairs (list): List of tuples with feature pairs to interact
            
        Returns:
            pandas.DataFrame: DataFrame with added interaction features
        """
        logger.info(f"Creating interaction features for {len(feature_pairs)} feature pairs")
        
        result_df = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 not in df.columns or feature2 not in df.columns:
                logger.warning(f"Features {feature1} or {feature2} not in DataFrame")
                continue
                
            interaction_name = f"{feature1}_{feature2}_interaction"
            
            try:
                # Multiply features
                result_df[interaction_name] = result_df[feature1] * result_df[feature2]
                logger.info(f"Created interaction feature: {interaction_name}")
                
            except Exception as e:
                logger.error(f"Error creating interaction feature: {str(e)}")
        
        return result_df
    
    def process_features(self, df, group_by_cols=None, agg_time_windows=None, 
                         entropy_columns=None, normalize=True, encode_categorical=True):
        """
        Complete feature processing pipeline
        
        Args:
            df (pandas.DataFrame): Raw packet data
            group_by_cols (list): Columns to group by in aggregation
            agg_time_windows (list): Time windows for aggregation
            entropy_columns (list): Columns to calculate entropy for
            normalize (bool): Whether to normalize numeric features
            encode_categorical (bool): Whether to encode categorical features
            
        Returns:
            pandas.DataFrame: Processed features
        """
        logger.info("Starting feature processing pipeline")
        
        # Extract base features
        processed_df = self.extract_base_features(df)
        
        # Aggregate by time window if specified
        if group_by_cols:
            processed_df = self.aggregate_by_time_window(
                processed_df, 
                group_by_cols, 
                time_windows=agg_time_windows
            )
        
        # Calculate entropy features if specified
        if entropy_columns:
            processed_df = self.calculate_entropy_features(
                processed_df,
                entropy_columns,
                time_windows=agg_time_windows
            )
        
        # Normalize features if requested
        if normalize:
            processed_df, _ = self.normalize_features(processed_df)
        
        # Encode categorical features if requested
        if encode_categorical:
            processed_df, _ = self.encode_categorical_features(processed_df)
        
        logger.info(f"Feature processing complete. Final feature count: {len(processed_df.columns)}")
        return processed_df

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='10s'),
        'src_ip': ['192.168.0.1', '10.0.0.1', '8.8.8.8'] * 33 + ['192.168.0.1'],
        'dst_ip': ['10.0.0.2', '192.168.0.1', '192.168.0.2'] * 33 + ['10.0.0.2'],
        'src_port': [80, 443, 12345] * 33 + [80],
        'dst_port': [12345, 80, 53] * 33 + [12345],
        'packet_size': np.random.randint(40, 1500, 100),
        'protocol': ['TCP', 'UDP', 'HTTP'] * 33 + ['TCP']
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Process features
    processed_data = fe.process_features(
        data,
        group_by_cols=['src_ip', 'dst_ip'],
        entropy_columns=['src_port', 'dst_port']
    )
    
    # Print results
    print(f"Original features: {list(data.columns)}")
    print(f"Processed features: {list(processed_data.columns)}")
    print(f"Number of features generated: {len(processed_data.columns)}")