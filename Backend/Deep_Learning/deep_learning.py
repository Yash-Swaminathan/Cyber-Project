"""
deep_learning.py - Deep learning models for network traffic anomaly detection

This module implements deep learning models such as autoencoders and LSTM networks
using TensorFlow/Keras for network traffic anomaly detection.
"""

import os
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore


@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

# Configure logging
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "deep_learning.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepLearningDetector")

class DeepLearningDetector:
    def __init__(self, model_type='autoencoder', input_shape=None, model_params=None):
        """
        Initialize the deep learning detector.
        
        Args:
            model_type (str): Type of model to use ('autoencoder' or 'lstm')
            input_shape (tuple): Shape of the input data. For autoencoder, use (num_features,);
                                 for LSTM, use (timesteps, num_features).
            model_params (dict, optional): Parameters for the model architecture.
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.model_params = model_params if model_params is not None else {}
        self.model = None
        self.threshold = None  # Optionally set after training for decision-making
        logger.info(f"DeepLearningDetector initialized with model type: {model_type}")

    def build_model(self):
        """
        Build the deep learning model based on the specified type.
        
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        if self.input_shape is None:
            logger.error("Input shape must be defined to build the model.")
            raise ValueError("Input shape must be defined.")
        
        if self.model_type == 'autoencoder':
            self.model = self.build_autoencoder()
        elif self.model_type == 'lstm':
            self.model = self.build_lstm()
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info("Model built successfully.")
        return self.model

    def build_autoencoder(self):
        """
        Build a simple autoencoder model.
        
        Returns:
            tf.keras.Model: Autoencoder model.
        """
        # Determine the encoding dimension (default to 32 if not specified)
        encoding_dim = self.model_params.get('encoding_dim', 32)
        input_layer = layers.Input(shape=self.input_shape)
        
        # Encoder
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        
        # Decoder (reconstructs input)
        decoded = layers.Dense(self.input_shape[0], activation='sigmoid')(encoded)
        
        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        logger.info("Autoencoder model created with encoding dimension %d.", encoding_dim)
        return autoencoder

    def build_lstm(self):
        """
        Build a simple LSTM model for sequence anomaly detection.
        
        Returns:
            tf.keras.Model: LSTM model.
        """
        # Expecting input_shape as (timesteps, features)
        timesteps, features = self.input_shape
        lstm_units = self.model_params.get('lstm_units', 64)
        input_layer = layers.Input(shape=(timesteps, features))
        
        # LSTM layer
        x = layers.LSTM(lstm_units, return_sequences=False)(input_layer)
        # Dense output layer for binary classification (anomaly vs normal)
        output_layer = layers.Dense(1, activation='sigmoid')(x)
        
        lstm_model = models.Model(inputs=input_layer, outputs=output_layer)
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("LSTM model created with %d units.", lstm_units)
        return lstm_model

    def train(self, X_train, y_train=None, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the deep learning model.
        
        Args:
            X_train (np.array): Training data.
            y_train (np.array, optional): Training labels. For autoencoder, this is ignored (X_train is used as target).
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            validation_split (float): Fraction of training data used for validation.
            
        Returns:
            History object from model.fit().
        """
        if self.model is None:
            self.build_model()
        if self.model_type == 'autoencoder':
            # For autoencoders, the target is the input itself.
            y_train = X_train
        logger.info("Training model for %d epochs on %d samples.", epochs, len(X_train))
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 verbose=1)
        logger.info("Model training complete.")
        return history

    def get_anomaly_scores(self, X):
        """
        Compute anomaly scores for the input data.
        
        Args:
            X (np.array): Data for anomaly scoring.
            
        Returns:
            np.array: Anomaly scores.
        """
        if self.model is None:
            logger.error("Model has not been built or trained.")
            raise ValueError("Model not available.")
        
        if self.model_type == 'autoencoder':
            # Reconstruction error as anomaly score
            X_pred = self.model.predict(X)
            mse = np.mean(np.power(X - X_pred, 2), axis=1)
            logger.info("Anomaly scores computed using autoencoder reconstruction error.")
            return mse
        elif self.model_type == 'lstm':
            # Use model's output probability as anomaly score (higher means more anomalous)
            scores = self.model.predict(X).flatten()
            logger.info("Anomaly scores computed using LSTM output.")
            return scores
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")

    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): File path to save the model.
        """
        if self.model is None:
            logger.error("No model to save.")
            raise ValueError("Model not available.")
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.model.save(model_path)
        logger.info("Model saved to %s", model_path)

    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model.
            
        Returns:
            DeepLearningDetector: Instance with the loaded model.
        """
        try:
            custom_objects = {'mse': custom_mse}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            # Infer input shape from loaded model; default to autoencoder type.
            input_shape = model.input_shape[1:]
            instance = cls(model_type='autoencoder', input_shape=input_shape)
            instance.model = model
            logger.info("Model loaded from %s", model_path)
            return instance
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

if __name__ == "__main__":
    # Example usage:
    # For an autoencoder (non-sequential data), assume each sample has 20 features.
    X_dummy = np.random.rand(1000, 18)
    
    # Initialize the detector for an autoencoder with a reduced encoding dimension.
    detector = DeepLearningDetector(model_type='autoencoder', input_shape=(18,), model_params={'encoding_dim': 10})
    
    # Build and train the model on dummy data.
    detector.build_model()
    history = detector.train(X_dummy, epochs=10, batch_size=32)
    
    # Compute anomaly scores (reconstruction errors).
    scores = detector.get_anomaly_scores(X_dummy)
    print("Average anomaly score:", np.mean(scores))
    
    # Save the trained model.
    model_save_path = os.path.join("models", "deep_autoencoder.h5")
    detector.save_model(model_save_path)
    
    # Load the model from disk.
    loaded_detector = DeepLearningDetector.load_model(model_save_path)
    loaded_scores = loaded_detector.get_anomaly_scores(X_dummy)
    print("Average anomaly score from loaded model:", np.mean(loaded_scores))