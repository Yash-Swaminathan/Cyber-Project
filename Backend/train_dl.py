"""
train_dl.py - Script to train the deep learning anomaly detector.

This script demonstrates how to:
  - Generate or load training data.
  - Train a deep learning model (using an autoencoder by default) via the DeepLearningDetector class.
  - Save the trained model for later inference by the backend.

Usage:
  poetry run python train_dl.py
"""

import os
import numpy as np
from Deep_Learning.deep_learning import DeepLearningDetector

def main():
    # For demonstration, generate synthetic training data.
    # For an autoencoder, each sample should be a 1D array with a fixed number of features.
    num_samples = 1000
    num_features = 18  # Ensure this matches your expected feature count.
    X_train = np.random.rand(num_samples, num_features).astype('float32')
    
    # Initialize the deep learning detector as an autoencoder.
    # You can adjust the encoding dimension via model_params.
    detector = DeepLearningDetector(model_type='autoencoder', input_shape=(num_features,), model_params={'encoding_dim': 10})
    
    # Build the model (if not built during training)
    detector.build_model()
    
    # Train the model. For autoencoders, the target is the input data itself.
    print("Training deep learning model...")
    history = detector.train(X_train, epochs=20, batch_size=32, validation_split=0.1)
    
    # Save the trained model to the models directory.
    model_save_path = os.path.join("models", "deep_autoencoder.h5")
    detector.save_model(model_save_path)
    print("Training complete. Model saved to:", model_save_path)

if __name__ == "__main__":
    main()
