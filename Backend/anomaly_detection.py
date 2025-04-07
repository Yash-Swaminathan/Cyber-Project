"""
anomaly_detection.py - Traditional machine learning models for network traffic anomaly detection

This module implements anomaly detection using Isolation Forest and One-Class SVM
from scikit-learn, designed to work with features created by the feature engineering module.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler

# Configure logging
os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join("logs", "anomaly_detection.log"), mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger("AnomalyDetection")


class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', model_params=None):
        """
        Initialize the anomaly detector with specified model type
        
        Args:
            model_type (str): Type of model to use ('isolation_forest' or 'one_class_svm')
            model_params (dict, optional): Parameters for the model
        """
        logger.info(f"Initializing anomaly detector with model type: {model_type}")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Set default parameters if none provided
        if model_params is None:
            if model_type == 'isolation_forest':
                self.model_params = {
                    'n_estimators': 100,
                    'max_samples': 'auto',
                    'contamination': 'auto',
                    'random_state': 42
                }
            elif model_type == 'one_class_svm':
                self.model_params = {
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'nu': 0.01  # Expected proportion of outliers
                }
            else:
                logger.error(f"Unknown model type: {model_type}")
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.model_params = model_params
    
    def _initialize_model(self):
        """Initialize the model based on the selected type and parameters"""
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(**self.model_params)
            logger.info(f"Initialized Isolation Forest with parameters: {self.model_params}")
            
        elif self.model_type == 'one_class_svm':
            self.model = OneClassSVM(**self.model_params)
            logger.info(f"Initialized One-Class SVM with parameters: {self.model_params}")
            
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, X, y=None, test_size=0.2, random_state=42):
        """
        Prepare data for training and testing
        
        Args:
            X (pandas.DataFrame): Feature DataFrame
            y (pandas.Series, optional): Labels if available (1 for normal, -1 for anomaly)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) or (X_train, X_test) if y is None
        """
        logger.info(f"Preparing data with test_size={test_size}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Split data with labels: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            logger.info(f"Split data without labels: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test
    
    def train(self, X_train, y_train=None, scale_data=True):
        """
        Train the anomaly detection model
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series, optional): Training labels if available
            scale_data (bool): Whether to scale the data
            
        Returns:
            self: Trained model instance
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        # Initialize model if not already done
        if self.model is None:
            self._initialize_model()
        
        # Scale data if requested
        if scale_data:
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("Data scaled with StandardScaler")
        else:
            X_train_scaled = X_train
        
        # Train model
        start_time = datetime.now()
        
        try:
            if y_train is not None and self.model_type != 'one_class_svm':
                # If labels are available and model supports supervised training
                self.model.fit(X_train_scaled, y_train)
                logger.info("Model trained with labels")
            else:
                # Unsupervised training
                self.model.fit(X_train_scaled)
                logger.info("Model trained without labels (unsupervised)")
                
            training_time = datetime.now() - start_time
            logger.info(f"Training completed in {training_time.total_seconds():.2f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X, scale_data=True):
        """
        Predict anomalies in the data
        
        Args:
            X (pandas.DataFrame): Features to predict
            scale_data (bool): Whether to scale the data
            
        Returns:
            numpy.ndarray: Predictions (-1 for anomaly, 1 for normal)
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Predicting anomalies in {len(X)} samples")
        
        # Scale data if requested
        if scale_data:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        try:
            predictions = self.model.predict(X_scaled)
            logger.info(f"Prediction complete: {np.sum(predictions == -1)} anomalies detected")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def decision_function(self, X, scale_data=True):
        """
        Get anomaly scores for the data
        
        Args:
            X (pandas.DataFrame): Features to score
            scale_data (bool): Whether to scale the data
            
        Returns:
            numpy.ndarray: Anomaly scores (lower values indicate anomalies)
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Calculating anomaly scores for {len(X)} samples")
        
        # Scale data if requested
        if scale_data:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get scores
        try:
            scores = self.model.decision_function(X_scaled)
            
            # For Isolation Forest, scores are negated for consistency with One-Class SVM
            # (lower values indicate anomalies for both)
            if self.model_type == 'isolation_forest':
                scores = -scores
                
            logger.info("Anomaly scoring complete")
            return scores
            
        except Exception as e:
            logger.error(f"Error during anomaly scoring: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_true=None, threshold=None):
        """
        Evaluate the model on test data
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_true (pandas.Series, optional): True labels if available
            threshold (float, optional): Decision threshold for anomaly scores
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Get anomaly scores
        anomaly_scores = self.decision_function(X_test)
        
        # If no true labels, just return statistics on detected anomalies
        if y_true is None:
            predictions = self.predict(X_test)
            anomaly_ratio = np.mean(predictions == -1)
            
            return {
                'anomaly_ratio': anomaly_ratio,
                'anomaly_count': np.sum(predictions == -1),
                'normal_count': np.sum(predictions == 1),
                'mean_score': np.mean(anomaly_scores),
                'min_score': np.min(anomaly_scores),
                'max_score': np.max(anomaly_scores)
            }
        
        # If true labels are available, calculate performance metrics
        # Note: Converting y_true to the same format (-1 for anomaly, 1 for normal)
        y_true_binary = np.where(y_true == -1, 1, 0)  # 1 if anomaly, 0 if normal
        
        # If threshold is not provided, use the default from the model
        if threshold is None:
            predictions = self.predict(X_test)
            y_pred_binary = np.where(predictions == -1, 1, 0)  # 1 if anomaly, 0 if normal
        else:
            # Use the provided threshold
            y_pred_binary = np.where(anomaly_scores > threshold, 1, 0)
        
        # Calculate metrics
        try:
            # Precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true_binary, anomaly_scores)
            pr_auc = auc(recall, precision)
            
            # F1 score
            f1 = f1_score(y_true_binary, y_pred_binary)
            
            # Anomaly detection rate
            detection_rate = recall[1]  # Recall for anomaly class
            
            # False alarm rate
            false_alarm_rate = 1 - precision[1]  # 1 - precision for anomaly class
            
            metrics = {
                'precision_recall_auc': pr_auc,
                'f1_score': f1,
                'detection_rate': detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'anomaly_count': np.sum(y_pred_binary),
                'true_anomaly_count': np.sum(y_true_binary),
                'threshold': threshold if threshold is not None else 'default'
            }
            
            logger.info(f"Evaluation complete: PR-AUC={pr_auc:.4f}, F1={f1:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_model(self, model_path, include_scaler=True):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
            include_scaler (bool): Whether to save the scaler too
        """
        if self.model is None:
            logger.error("Cannot save untrained model")
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        
        try:
            # Save the model
            model_file = f"{model_path}.joblib"
            joblib.dump(self.model, model_file)
            logger.info(f"Model saved to {model_file}")
            
            # Save the scaler if requested
            if include_scaler:
                scaler_file = f"{model_path}_scaler.joblib"
                joblib.dump(self.scaler, scaler_file)
                logger.info(f"Scaler saved to {scaler_file}")
                
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'model_params': self.model_params,
                'date_saved': datetime.now().isoformat(),
                'scaler_included': include_scaler
            }
            
            metadata_file = f"{model_path}_metadata.joblib"
            joblib.dump(metadata, metadata_file)
            logger.info(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path, load_scaler=True):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to load the model from
            load_scaler (bool): Whether to load the scaler too
            
        Returns:
            self: Loaded model instance
        """
        try:
            # Load the model
            model_file = f"{model_path}.joblib"
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
            
            # Load the scaler if requested
            if load_scaler:
                scaler_file = f"{model_path}_scaler.joblib"
                self.scaler = joblib.load(scaler_file)
                logger.info(f"Scaler loaded from {scaler_file}")
            
            # Load metadata
            try:
                metadata_file = f"{model_path}_metadata.joblib"
                metadata = joblib.load(metadata_file)
                self.model_type = metadata['model_type']
                self.model_params = metadata['model_params']
                logger.info(f"Metadata loaded from {metadata_file}")
            except:
                logger.warning("Could not load metadata")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_anomaly_scores(self, X, y_true=None, n_samples=None):
        """
        Plot the distribution of anomaly scores
        
        Args:
            X (pandas.DataFrame): Features to score
            y_true (pandas.Series, optional): True labels if available
            n_samples (int, optional): Number of samples to plot (None for all)
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        # Get anomaly scores
        anomaly_scores = self.decision_function(X)
        
        # Limit number of samples if specified
        if n_samples is not None and n_samples < len(anomaly_scores):
            indices = np.random.choice(len(anomaly_scores), n_samples, replace=False)
            anomaly_scores = anomaly_scores[indices]
            if y_true is not None:
                y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # If true labels are available, create separate distributions
        if y_true is not None:
            # Convert labels to binary format
            y_true_binary = np.where(y_true == -1, 1, 0)  # 1 if anomaly, 0 if normal
            
            # Separate scores by class
            normal_scores = anomaly_scores[y_true_binary == 0]
            anomaly_scores = anomaly_scores[y_true_binary == 1]
            
            # Plot distributions
            sns.histplot(normal_scores, color='blue', label='Normal', alpha=0.5, bins=30)
            sns.histplot(anomaly_scores, color='red', label='Anomaly', alpha=0.5, bins=30)
            plt.legend()
            
        else:
            # Plot single distribution
            sns.histplot(anomaly_scores, bins=30)
            
            # Add vertical line for the decision threshold
            if hasattr(self.model, 'threshold_'):
                plt.axvline(x=self.model.threshold_, color='red', linestyle='--', 
                           label='Decision Threshold')
                plt.legend()
        
        plt.title(f'Distribution of Anomaly Scores ({self.model_type})')
        plt.xlabel('Anomaly Score (higher = more anomalous)')
        plt.ylabel('Count')
        plt.tight_layout()
        
        return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Sample data generation (you would replace this with your actual data)
    from sklearn.datasets import make_classification
    
    # Create a synthetic dataset with 5% anomalies
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        weights=[0.95, 0.05],  # 95% normal, 5% anomalies
        random_state=42
    )
    
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Convert labels: 0 -> 1 (normal), 1 -> -1 (anomaly)
    y_series = pd.Series(np.where(y == 0, 1, -1), name='label')
    
    # Initialize and train an Isolation Forest model
    detector = AnomalyDetector(model_type='isolation_forest')
    
    # Prepare data
    X_train, X_test, y_train, y_test = detector.prepare_data(X_df, y_series)
    
    # Train the model
    detector.train(X_train)
    
    # Make predictions
    predictions = detector.predict(X_test)
    
    # Evaluate the model
    metrics = detector.evaluate(X_test, y_test)
    print("Evaluation metrics:", metrics)
    
    # Save the model
    detector.save_model('models/isolation_forest_model')
    
    # Load the model
    loaded_detector = AnomalyDetector()
    loaded_detector.load_model('models/isolation_forest_model')
    
    # Plot anomaly scores
    fig = detector.plot_anomaly_scores(X_test, y_test)
    plt.show()
    