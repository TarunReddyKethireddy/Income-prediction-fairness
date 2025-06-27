"""
Data preprocessing module for the Income Prediction project.
Handles loading, cleaning, and transforming the Adult Income Dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Define column names for the Adult Income Dataset
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RAW_TRAIN_PATH = os.path.join(DATA_DIR, 'adult.data')
RAW_TEST_PATH = os.path.join(DATA_DIR, 'adult.test')

def load_data():
    """
    Load the Adult Income Dataset from the data directory.
    
    Returns:
        tuple: (train_data, test_data) as pandas DataFrames
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load training data
    train_data = pd.read_csv(RAW_TRAIN_PATH, header=None, names=COLUMN_NAMES, 
                           skipinitialspace=True, na_values='?')
    
    # Load test data (note: the test file has an extra row at the beginning)
    test_data = pd.read_csv(RAW_TEST_PATH, header=None, names=COLUMN_NAMES, 
                          skipinitialspace=True, na_values='?', skiprows=1)
    
    # Clean the income column (remove the dot in test data)
    test_data['income'] = test_data['income'].str.replace('.', '')
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data

def create_preprocessor(data):
    """
    Create a preprocessing pipeline for the data.
    
    Args:
        data: DataFrame to base the preprocessor on
        
    Returns:
        tuple: (preprocessor, feature_names)
    """
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('income')  # Remove the target variable
    
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines for numerical and categorical data
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor, numerical_cols, categorical_cols

def preprocess_data(train_data, test_data):
    """
    Preprocess the data: handle missing values, encode categorical features,
    and scale numerical features.
    
    Args:
        train_data: Training data DataFrame
        test_data: Test data DataFrame
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    # Create preprocessor
    preprocessor, numerical_cols, categorical_cols = create_preprocessor(train_data)
    
    # Prepare the target variable
    y_train = (train_data['income'] == '>50K').astype(int)
    y_test = (test_data['income'] == '>50K').astype(int)
    
    # Drop the target variable from features
    X_train = train_data.drop('income', axis=1)
    X_test = test_data.drop('income', axis=1)
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    categorical_features = preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + categorical_features.tolist()
    
    print(f"Processed data shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names):
    """
    Save the processed data and preprocessing pipeline for later use.
    
    Args:
        X_train: Processed training features
        X_test: Processed test features
        y_train: Training labels
        y_test: Test labels
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
    """
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save the processed data
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    # Save the preprocessor for later use
    joblib.dump(preprocessor, os.path.join(PROCESSED_DIR, 'preprocessor.joblib'))
    
    # Save feature names
    with open(os.path.join(PROCESSED_DIR, 'feature_names.txt'), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Saved processed data and preprocessor to {PROCESSED_DIR}")

def load_processed_data():
    """
    Load the preprocessed data.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
    
    # Load feature names
    with open(os.path.join(PROCESSED_DIR, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded processed data: X_train shape={X_train.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

def get_sensitive_feature_idx(feature_names, sensitive_feature='sex_Female'):
    """
    Get the index of the sensitive feature in the feature names list.
    
    Args:
        feature_names: List of feature names
        sensitive_feature: Name of the sensitive feature
        
    Returns:
        int: Index of the sensitive feature
    """
    try:
        return feature_names.index(sensitive_feature)
    except ValueError:
        print(f"Warning: Sensitive feature '{sensitive_feature}' not found in feature names.")
        # Return a default index (sex is usually encoded as a binary feature)
        return 9  # Default index for sex in the Adult dataset

if __name__ == "__main__":
    print("Loading data...")
    train_data, test_data = load_data()
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(train_data, test_data)
    
    print("\nSaving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names)
    
    print("\nData preprocessing complete!")
