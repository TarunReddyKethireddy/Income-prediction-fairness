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

def check_demo_mode():
    """
    Check if the app is running in demo mode.
    
    Returns:
        bool: True if in demo mode, False otherwise
    """
    try:
        # Check if key model files exist
        joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
        joblib.load(os.path.join(PROCESSED_DIR, 'preprocessor.joblib'))
        return False  # Files exist, not in demo mode
    except Exception:
        return True  # Files don't exist, in demo mode

def create_dummy_models():
    """
    Create dummy models for demo mode.
    
    Returns:
        dict: Dictionary of dummy models
    """
    from sklearn.dummy import DummyClassifier
    
    # Create a more robust dummy classifier with proper dimensions
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.randint(0, 2, 100)  # Binary target
    
    # Create and train different dummy models
    dummy_lr = DummyClassifier(strategy='prior')
    dummy_lr.fit(X_dummy, y_dummy)
    
    dummy_rf = DummyClassifier(strategy='stratified')
    dummy_rf.fit(X_dummy, y_dummy)
    
    # Add predict_proba method to dummy models if they don't have it
    if not hasattr(dummy_lr, 'predict_proba'):
        dummy_lr.predict_proba = lambda X: np.column_stack([np.zeros(len(X)), np.ones(len(X))])
        dummy_rf.predict_proba = lambda X: np.column_stack([np.zeros(len(X)), np.ones(len(X))])
    
    models = {
        'logistic_regression': dummy_lr,
        'random_forest': dummy_rf,
        'fair_demographic_parity': dummy_lr,
        'fair_equalized_odds': dummy_rf
    }
    
    return models

def create_dummy_preprocessor():
    """
    Create a dummy preprocessor for demo mode.
    
    Returns:
        object: Dummy preprocessor
    """
    from sklearn.preprocessing import FunctionTransformer
    
    # Create a custom transformer that returns a fixed number of features
    def transform_input(X):
        # Convert input to a fixed size feature vector of 10 dimensions
        if isinstance(X, pd.DataFrame):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        return np.ones((n_samples, 10))  # Return dummy features
    
    dummy_preprocessor = FunctionTransformer(transform_input)
    
    # Add necessary attributes to mimic a real preprocessor
    dummy_preprocessor.feature_names_in_ = ['age', 'workclass', 'fnlwgt', 'education', 
                                        'education-num', 'marital-status', 'occupation', 
                                        'relationship', 'race', 'sex', 'capital-gain', 
                                        'capital-loss', 'hours-per-week', 'native-country']
    
    # Add transform method if it doesn't exist
    if not hasattr(dummy_preprocessor, 'transform'):
        dummy_preprocessor.transform = transform_input
        
    return dummy_preprocessor

def create_dummy_data():
    """
    Create dummy data for demo mode.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Create dummy feature names
    feature_names = [
        'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
        'workclass_Private', 'education_HS-grad', 'marital-status_Married-civ-spouse',
        'sex_Female'
    ]
    
    # Create dummy data
    X_train = np.random.rand(1000, 10)
    X_test = np.random.rand(200, 10)
    y_train = np.random.randint(0, 2, 1000)
    y_test = np.random.randint(0, 2, 200)
    
    return X_train, X_test, y_train, y_test, feature_names

def create_dummy_metrics():
    """
    Create dummy metrics for demo mode.
    
    Returns:
        tuple: (performance_df, fairness_df)
    """
    # Create dummy performance metrics
    performance_data = {
        'logistic_regression': {
            'accuracy': 0.82, 'precision': 0.65, 'recall': 0.55, 'f1': 0.60, 'roc_auc': 0.85
        },
        'random_forest': {
            'accuracy': 0.84, 'precision': 0.70, 'recall': 0.60, 'f1': 0.65, 'roc_auc': 0.88
        },
        'fair_demographic_parity': {
            'accuracy': 0.80, 'precision': 0.62, 'recall': 0.58, 'f1': 0.60, 'roc_auc': 0.83
        },
        'fair_equalized_odds': {
            'accuracy': 0.81, 'precision': 0.63, 'recall': 0.59, 'f1': 0.61, 'roc_auc': 0.84
        }
    }
    performance_df = pd.DataFrame(performance_data).T
    
    # Create dummy fairness metrics
    fairness_data = {
        'logistic_regression': {
            'demographic_parity_difference': 0.15, 'equalized_odds_difference': 0.18
        },
        'random_forest': {
            'demographic_parity_difference': 0.18, 'equalized_odds_difference': 0.20
        },
        'fair_demographic_parity': {
            'demographic_parity_difference': 0.05, 'equalized_odds_difference': 0.12
        },
        'fair_equalized_odds': {
            'demographic_parity_difference': 0.10, 'equalized_odds_difference': 0.06
        }
    }
    fairness_df = pd.DataFrame(fairness_data).T
    
    return performance_df, fairness_df

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
