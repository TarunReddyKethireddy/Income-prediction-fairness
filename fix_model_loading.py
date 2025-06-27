#!/usr/bin/env python3
"""
This script diagnoses and fixes model loading issues in the Income Prediction app.
It checks for model and preprocessor files, validates their compatibility,
and creates dummy models if needed.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
import shutil
from datetime import datetime

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def check_directory_structure():
    """Check if all required directories exist."""
    print_section("Directory Structure")
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR} (exists: {os.path.exists(DATA_DIR)})")
    print(f"Models directory: {MODELS_DIR} (exists: {os.path.exists(MODELS_DIR)})")
    print(f"Processed directory: {PROCESSED_DIR} (exists: {os.path.exists(PROCESSED_DIR)})")
    
    # List files in each directory
    if os.path.exists(DATA_DIR):
        print(f"\nFiles in data directory: {os.listdir(DATA_DIR)}")
    if os.path.exists(MODELS_DIR):
        print(f"\nFiles in models directory: {os.listdir(MODELS_DIR)}")
    if os.path.exists(PROCESSED_DIR):
        print(f"\nFiles in processed directory: {os.listdir(PROCESSED_DIR)}")

def check_model_files():
    """Check if model files exist and can be loaded."""
    print_section("Model Files")
    
    model_names = ['logistic_regression', 'random_forest', 'fair_demographic_parity', 'fair_equalized_odds']
    all_models_ok = True
    
    for model_name in model_names:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        print(f"\nChecking {model_name}:")
        print(f"  Path: {model_path}")
        print(f"  Exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                # Try to load the model
                model = joblib.load(model_path)
                print(f"  ✅ Successfully loaded {model_name}")
                print(f"  Type: {type(model)}")
            except Exception as e:
                all_models_ok = False
                print(f"  ❌ Error loading {model_name}: {str(e)}")
                
                # Try with pickle as fallback
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"  ✅ Successfully loaded {model_name} with pickle")
                except Exception as e2:
                    print(f"  ❌ Pickle fallback also failed: {str(e2)}")
        else:
            all_models_ok = False
            print(f"  ❌ Model file not found")
    
    return all_models_ok

def check_preprocessor():
    """Check if preprocessor exists and can be loaded."""
    print_section("Preprocessor")
    
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    print(f"Preprocessor path: {preprocessor_path}")
    print(f"Exists: {os.path.exists(preprocessor_path)}")
    
    if os.path.exists(preprocessor_path):
        try:
            # Try to load the preprocessor
            preprocessor = joblib.load(preprocessor_path)
            print(f"✅ Successfully loaded preprocessor")
            print(f"Type: {type(preprocessor)}")
            
            # Check for feature names methods
            if hasattr(preprocessor, 'get_feature_names_out'):
                print("✅ Has get_feature_names_out method")
            elif hasattr(preprocessor, 'get_feature_names'):
                print("✅ Has get_feature_names method")
            else:
                print("❌ No feature names method found")
                
            return True
        except Exception as e:
            print(f"❌ Error loading preprocessor: {str(e)}")
            return False
    else:
        print("❌ Preprocessor file not found")
        return False

def create_dummy_models():
    """Create dummy models if real ones can't be loaded."""
    print_section("Creating Dummy Models")
    
    # Backup existing models directory if it exists
    if os.path.exists(MODELS_DIR):
        backup_dir = f"{MODELS_DIR}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up existing models to {backup_dir}")
        shutil.copytree(MODELS_DIR, backup_dir)
    
    # Create dummy data
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.randint(0, 2, 100)  # Binary target
    
    # Create and save dummy models
    model_names = ['logistic_regression', 'random_forest', 'fair_demographic_parity', 'fair_equalized_odds']
    
    for model_name in model_names:
        print(f"\nCreating dummy model for {model_name}")
        if model_name in ['logistic_regression', 'fair_demographic_parity']:
            dummy_model = DummyClassifier(strategy='prior')
        else:  # random_forest or fair_equalized_odds
            dummy_model = DummyClassifier(strategy='stratified')
        
        dummy_model.fit(X_dummy, y_dummy)
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        try:
            joblib.dump(dummy_model, model_path)
            print(f"✅ Successfully saved dummy model to {model_path}")
        except Exception as e:
            print(f"❌ Error saving dummy model: {str(e)}")

def create_dummy_preprocessor():
    """Create a dummy preprocessor if the real one can't be loaded."""
    print_section("Creating Dummy Preprocessor")
    
    # Backup existing preprocessor if it exists
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        backup_path = f"{preprocessor_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up existing preprocessor to {backup_path}")
        shutil.copy2(preprocessor_path, backup_path)
    
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
    try:
        joblib.dump(dummy_preprocessor, preprocessor_path)
        print(f"✅ Successfully saved dummy preprocessor to {preprocessor_path}")
        
        # Also create feature names file
        feature_names_path = os.path.join(PROCESSED_DIR, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            for i in range(104):
                f.write(f"feature_{i}\n")
        print(f"✅ Successfully created feature names file at {feature_names_path}")
    except Exception as e:
        print(f"❌ Error saving dummy preprocessor: {str(e)}")

def main():
    """Main function to diagnose and fix model loading issues."""
    print_section("Income Prediction App - Model Loading Diagnostic")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check directory structure
    check_directory_structure()
    
    # Check model files
    models_ok = check_model_files()
    
    # Check preprocessor
    preprocessor_ok = check_preprocessor()
    
    # Create dummy models and preprocessor if needed
    if not models_ok or not preprocessor_ok:
        print_section("Issues Detected")
        if not models_ok:
            print("❌ Problems found with model files")
        if not preprocessor_ok:
            print("❌ Problems found with preprocessor")
            
        print("\nWould you like to create dummy models and preprocessor? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            if not models_ok:
                create_dummy_models()
            if not preprocessor_ok:
                create_dummy_preprocessor()
            print_section("Fix Complete")
            print("✅ Dummy models and preprocessor have been created.")
            print("You can now run the Streamlit app again.")
        else:
            print("No changes made. Please fix the issues manually.")
    else:
        print_section("All Good!")
        print("✅ All model files and preprocessor are loading correctly.")

if __name__ == "__main__":
    main()
