#!/usr/bin/env python3
"""
Entry point for the Income Prediction app.
This script ensures models and preprocessor exist before running the Streamlit app.
"""

import os
import sys
import subprocess

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Check if we need to create dummy models
model_files = [
    os.path.join(MODELS_DIR, 'logistic_regression.joblib'),
    os.path.join(MODELS_DIR, 'random_forest.joblib'),
    os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'),
    os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib')
]

preprocessor_file = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')

# Check if any model or preprocessor is missing
if not all(os.path.exists(f) for f in model_files) or not os.path.exists(preprocessor_file):
    print("Creating dummy models and preprocessor...")
    try:
        # Run the script to create dummy models
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'create_dummy_models.py')], check=True)
        print("Dummy models created successfully!")
    except Exception as e:
        print(f"Error creating dummy models: {str(e)}")

# Import and run the Streamlit app
if __name__ == "__main__":
    # Run the Streamlit app
    os.system(f"{sys.executable} -m streamlit run streamlit_app.py --server.headless true")
