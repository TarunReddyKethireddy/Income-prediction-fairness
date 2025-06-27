#!/usr/bin/env python3
"""
Test script to verify model loading
"""
import os
import joblib
import numpy as np
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Define DummyPreprocessor class to match the one in streamlit_app.py
# This is needed for loading the preprocessor, even if we don't create new instances
class DummyPreprocessor:
    """Dummy preprocessor class for when the real preprocessor is not available."""
    def transform(self, X):
        """Return random features for the input data."""
        return np.random.rand(len(X), 104)
        
    def get_feature_names_out(self):
        """Return dummy feature names."""
        return [f'feature_{i}' for i in range(104)]

def test_load_model(model_path):
    """Test loading a model from a file"""
    print(f"Testing model loading: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("Model file does not exist.")
        return None
        
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def test_load_preprocessor(preprocessor_path):
    """Test loading a preprocessor from a file"""
    print(f"Testing preprocessor loading: {preprocessor_path}")
    print(f"File exists: {os.path.exists(preprocessor_path)}")
    
    if not os.path.exists(preprocessor_path):
        print("Preprocessor file does not exist.")
        return None
        
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded successfully: {type(preprocessor).__name__}")
        return preprocessor
    except Exception as e:
        print(f"Error loading preprocessor: {str(e)}")
        return None

def test_prediction(model, preprocessor):
    """Test making a prediction with a model and preprocessor"""
    if model is None or preprocessor is None:
        print("Cannot test prediction: model or preprocessor is None")
        return
    
    print("\nTesting prediction with real model and preprocessor")
    
    # Create a simple test input
    test_input = pd.DataFrame({
        'age': [30],
        'workclass': ['Private'],
        'fnlwgt': [200000],
        'education': ['Bachelors'],
        'education-num': [13],
        'marital-status': ['Never-married'],
        'occupation': ['Prof-specialty'],
        'relationship': ['Not-in-family'],
        'race': ['White'],
        'sex': ['Male'],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [40],
        'native-country': ['United-States']
    })
    
    try:
        # Preprocess the input
        X = preprocessor.transform(test_input)
        print(f"Preprocessed input shape: {X.shape}")
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            print("Converted sparse matrix to dense array")
        
        # Make prediction
        prediction = model.predict(X)
        print(f"Prediction result: {prediction}")
        
        # Try predict_proba if available
        if hasattr(model, 'predict_proba'):
            try:
                probability = model.predict_proba(X)
                print(f"Prediction probability: {probability}")
            except Exception as e:
                print(f"Error with predict_proba: {str(e)}")
        return True
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return False

def main():
    """Main function to test model loading and prediction"""
    print("=== Testing Model Loading ===\n")
    
    # Test loading each model
    model_files = {
        'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression.joblib'),
        'random_forest': os.path.join(MODELS_DIR, 'random_forest.joblib'),
        'fair_demographic_parity': os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'),
        'fair_equalized_odds': os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib')
    }
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory does not exist: {MODELS_DIR}")
        print("Please ensure models are downloaded and extracted first.")
        return
    
    # List available files in models directory
    print(f"Files in models directory: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory not found'}")
    
    # Test loading each model
    models = {}
    for name, path in model_files.items():
        print(f"\nTesting {name} model:")
        models[name] = test_load_model(path)
    
    # Test loading preprocessor
    print("\n=== Testing Preprocessor Loading ===\n")
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    
    # Check if processed directory exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"Processed directory does not exist: {PROCESSED_DIR}")
        print("Please ensure preprocessor files are available.")
        return
    
    # List available files in processed directory
    print(f"Files in processed directory: {os.listdir(PROCESSED_DIR) if os.path.exists(PROCESSED_DIR) else 'Directory not found'}")
    
    preprocessor = test_load_preprocessor(preprocessor_path)
    
    if preprocessor is None:
        print("\nCannot proceed with prediction tests: preprocessor is missing.")
        return
    
    # Test prediction with each model
    print("\n=== Testing Predictions ===\n")
    successful_models = []
    
    for name, model in models.items():
        if model is not None:
            print(f"\nTesting prediction with {name} model:")
            success = test_prediction(model, preprocessor)
            if success:
                successful_models.append(name)
    
    # Summary
    print("\n=== Test Summary ===\n")
    print(f"Models successfully loaded and tested: {len(successful_models)} of {len(model_files)}")
    if successful_models:
        print(f"Working models: {', '.join(successful_models)}")
    else:
        print("No models were successfully loaded and tested.")
        print("Please check that model files exist and are compatible.")


if __name__ == "__main__":
    main()
