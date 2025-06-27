#!/usr/bin/env python3
"""
Test script to verify model loading
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def test_load_model(model_path):
    """Test loading a model from a file"""
    print(f"Testing model loading: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
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
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

def main():
    """Main function to test model loading and prediction"""
    print("=== Testing Model Loading ===")
    
    # Test loading each model
    model_files = {
        'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression.joblib'),
        'random_forest': os.path.join(MODELS_DIR, 'random_forest.joblib'),
        'fair_demographic_parity': os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'),
        'fair_equalized_odds': os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib')
    }
    
    models = {}
    for name, path in model_files.items():
        print(f"\nTesting {name} model:")
        models[name] = test_load_model(path)
    
    # Test loading preprocessor
    print("\n=== Testing Preprocessor Loading ===")
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    preprocessor = test_load_preprocessor(preprocessor_path)
    
    # Test prediction with each model
    print("\n=== Testing Predictions ===")
    for name, model in models.items():
        print(f"\nTesting prediction with {name} model:")
        test_prediction(model, preprocessor)

if __name__ == "__main__":
    main()
