"""
Helper functions for the Streamlit application.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import zipfile
import tempfile
import requests
from io import BytesIO

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def create_example_input():
    """
    Create an example input for prediction.
    
    Returns:
        dict: Example input values
    """
    return {
        'age': 38,
        'workclass': 'Private',
        'fnlwgt': 189778,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Divorced',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
    }

def get_column_values():
    """
    Get possible values for categorical columns.
    
    Returns:
        dict: Dictionary mapping column names to possible values
    """
    return {
        'workclass': [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
            'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
        ],
        'education': [
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
            'Doctorate', '5th-6th', '10th', '1st-4th', 'Preschool'
        ],
        'marital-status': [
            'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
            'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
        ],
        'occupation': [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv',
            'Armed-Forces'
        ],
        'relationship': [
            'Wife', 'Own-child', 'Husband', 'Not-in-family',
            'Other-relative', 'Unmarried'
        ],
        'race': [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
            'Other', 'Black'
        ],
        'sex': [
            'Female', 'Male'
        ],
        'native-country': [
            'United-States', 'Cambodia', 'England', 'Puerto-Rico',
            'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
            'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
            'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
            'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
            'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
            'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
            'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
            'Peru', 'Hong', 'Holand-Netherlands'
        ]
    }

def download_and_extract_models(url):
    """
    Download a zip file from a URL and extract model files.
    
    Args:
        url: URL of the zip file
        
    Returns:
        bool: True if download and extraction were successful, False otherwise
    """
    try:
        # Download the zip file
        with st.spinner("Downloading model files..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create a BytesIO object from the response content
            zip_data = BytesIO(response.content)
        
        # Create directories if they don't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Extract files from zip
        with st.spinner("Extracting model files..."):
            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                # Get list of files in the zip
                file_list = zip_ref.namelist()
                
                # Process each file
                for filename in file_list:
                    # Determine the target directory based on filename
                    if filename.endswith('.joblib') or filename.endswith('.csv'):
                        target_dir = MODELS_DIR
                    elif filename.endswith('.npy') or filename.endswith('.txt'):
                        target_dir = PROCESSED_DIR
                    else:
                        # Skip unknown file types
                        continue
                    
                    # Extract the file
                    zip_ref.extract(filename, target_dir)
                    
                    # Move file to correct location if needed
                    source_path = os.path.join(target_dir, filename)
                    target_path = os.path.join(target_dir, os.path.basename(filename))
                    
                    if source_path != target_path and os.path.exists(source_path):
                        os.rename(source_path, target_path)
        
        return True
    except Exception as e:
        st.error(f"Error downloading and extracting model files: {e}")
        return False

def download_individual_models(base_url):
    """
    Download individual model files from a base URL.
    
    Args:
        base_url: Base URL for model files
        
    Returns:
        bool: True if all downloads were successful, False otherwise
    """
    try:
        # Create directories if they don't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # List of files to download
        model_files = [
            ('logistic_regression.joblib', MODELS_DIR),
            ('random_forest.joblib', MODELS_DIR),
            ('fair_demographic_parity.joblib', MODELS_DIR),
            ('fair_equalized_odds.joblib', MODELS_DIR),
            ('preprocessor.joblib', PROCESSED_DIR),
            ('model_performance.csv', MODELS_DIR),
            ('fairness_metrics.csv', MODELS_DIR),
            ('feature_names.txt', PROCESSED_DIR),
            ('X_train.npy', PROCESSED_DIR),
            ('X_test.npy', PROCESSED_DIR),
            ('y_train.npy', PROCESSED_DIR),
            ('y_test.npy', PROCESSED_DIR)
        ]
        
        # Download each file
        success_count = 0
        for filename, target_dir in model_files:
            try:
                url = f"{base_url}{filename}"
                with st.spinner(f"Downloading {filename}..."):
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Save the file
                    with open(os.path.join(target_dir, filename), 'wb') as f:
                        f.write(response.content)
                    
                    success_count += 1
            except Exception as e:
                st.warning(f"Failed to download {filename}: {e}")
        
        return success_count == len(model_files)
    except Exception as e:
        st.error(f"Error downloading model files: {e}")
        return False

# Removed check_demo_mode function as we now check for model existence directly in the main app

# Removed create_dummy_data function as we now use only real data

# Removed create_dummy_metrics function as we now use only real metrics

def load_adult_data_sample():
    """
    Load a sample of the Adult Income Dataset.
    
    Returns:
        pandas.DataFrame: Sample of the dataset
    """
    try:
        # Try to load the actual data
        data_path = os.path.join(BASE_DIR, 'data', 'adult.data')
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        data = pd.read_csv(data_path, header=None, names=column_names, 
                          skipinitialspace=True, na_values='?')
        return data.sample(n=min(1000, len(data)))
    except Exception:
        # Create synthetic data if actual data is not available
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        # Get column values
        column_values = get_column_values()
        
        # Create synthetic data
        n_samples = 1000
        synthetic_data = {
            'age': np.random.randint(18, 90, n_samples),
            'workclass': np.random.choice(column_values['workclass'], n_samples),
            'fnlwgt': np.random.randint(10000, 1000000, n_samples),
            'education': np.random.choice(column_values['education'], n_samples),
            'education-num': np.random.randint(1, 16, n_samples),
            'marital-status': np.random.choice(column_values['marital-status'], n_samples),
            'occupation': np.random.choice(column_values['occupation'], n_samples),
            'relationship': np.random.choice(column_values['relationship'], n_samples),
            'race': np.random.choice(column_values['race'], n_samples),
            'sex': np.random.choice(column_values['sex'], n_samples),
            'capital-gain': np.random.choice([0, *np.random.randint(1000, 100000, 100)], n_samples, p=[0.9, *np.ones(100)/1000]),
            'capital-loss': np.random.choice([0, *np.random.randint(1000, 5000, 100)], n_samples, p=[0.95, *np.ones(100)/2000]),
            'hours-per-week': np.random.randint(20, 80, n_samples),
            'native-country': np.random.choice(column_values['native-country'], n_samples, p=[0.8, *np.ones(len(column_values['native-country'])-1)/5]),
            'income': np.random.choice(['>50K', '<=50K'], n_samples, p=[0.25, 0.75])
        }
        
        return pd.DataFrame(synthetic_data)
