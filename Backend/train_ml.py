
import pandas as pd
from anomaly_detection import AnomalyDetector
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    # For demonstration, we create synthetic data.
    # In practice, replace this section with code to load your training dataset.
    X, y = make_classification(
        n_samples=1000,            # Number of samples
        n_features=20,             # Number of features
        n_informative=15,          # Informative features count
        n_redundant=5,             # Redundant features count
        weights=[0.95, 0.05],      # 95% normal, 5% anomalies
        random_state=42
    )
    
    # Create a DataFrame with feature names
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # In unsupervised anomaly detection (Isolation Forest),
    # you typically train on “normal” data.
    # For simplicity, we use the synthetic data as-is.
    # You can filter X_df using the labels y if needed.
    
    # Split the data into training and test sets (if you wish to evaluate later)
    X_train, X_test = train_test_split(X_df, test_size=0.2, random_state=42)
    
    # Initialize the anomaly detector with Isolation Forest
    detector = AnomalyDetector(model_type='isolation_forest')
    
    # Train the model on training data (scaling is enabled by default)
    detector.train(X_train, scale_data=True)
    
    # Save the trained model (model, scaler, metadata) to the models directory
    detector.save_model('models/isolation_forest_model')
    print("Training complete. The model is saved to 'models/isolation_forest_model.joblib'.")

if __name__ == "__main__":
    main()