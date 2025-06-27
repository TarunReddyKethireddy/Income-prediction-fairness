#!/usr/bin/env python3
"""
This script creates dummy models and preprocessor files for the Income Prediction app.
Run this script to ensure the app has working models and preprocessor.
"""

import os
import numpy as np
import joblib
from sklearn.dummy import DummyClassifier

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"Creating dummy models in {MODELS_DIR}")
print(f"Creating dummy preprocessor in {PROCESSED_DIR}")

# Create dummy data for training
X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
y_dummy = np.random.randint(0, 2, 100)  # Binary target

# Create and save dummy models
model_names = ['logistic_regression', 'random_forest', 'fair_demographic_parity', 'fair_equalized_odds']

for model_name in model_names:
    print(f"Creating dummy model for {model_name}")
    if model_name in ['logistic_regression', 'fair_demographic_parity']:
        dummy_model = DummyClassifier(strategy='prior')
    else:  # random_forest or fair_equalized_odds
        dummy_model = DummyClassifier(strategy='stratified')
    
    dummy_model.fit(X_dummy, y_dummy)
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(dummy_model, model_path)
    print(f"Saved dummy model to {model_path}")

# Create a dummy preprocessor class
class DummyPreprocessor:
    def __init__(self):
        # For compatibility with scikit-learn 0.24.2 column transformer
        self.transformers_ = []
        
    def transform(self, X):
        # Create a simple one-hot encoding simulation
        n_samples = len(X)
        # Create a feature array with 104 features (typical for Adult dataset after preprocessing)
        return np.random.rand(n_samples, 104)
    
    # For scikit-learn >= 1.0
    def get_feature_names_out(self):
        # Return dummy feature names
        return [f'feature_{i}' for i in range(104)]
        
    # For scikit-learn < 1.0
    def get_feature_names(self):
        # Return dummy feature names (for scikit-learn < 1.0)
        return [f'feature_{i}' for i in range(104)]

# Create and save the dummy preprocessor
dummy_preprocessor = DummyPreprocessor()
preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
joblib.dump(dummy_preprocessor, preprocessor_path)
print(f"Saved dummy preprocessor to {preprocessor_path}")

# Create feature names file
feature_names = [f'feature_{i}' for i in range(104)]
feature_names_path = os.path.join(PROCESSED_DIR, 'feature_names.txt')
with open(feature_names_path, 'w') as f:
    for name in feature_names:
        f.write(f"{name}\n")
print(f"Saved feature names to {feature_names_path}")

print("\nDummy models and preprocessor created successfully!")
print("You can now run the Streamlit app with: streamlit run streamlit_app.py")
