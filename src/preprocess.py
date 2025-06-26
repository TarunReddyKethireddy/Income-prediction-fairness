import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_data():
    """
    Load the Adult Income Dataset from the data directory.
    """
    # Define column names for the dataset
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Load training data
    train_path = os.path.join('data', 'adult.data')
    train_data = pd.read_csv(train_path, header=None, names=column_names, 
                            skipinitialspace=True, na_values='?')
    
    # Load test data (note: the test file has an extra row at the beginning)
    test_path = os.path.join('data', 'adult.test')
    test_data = pd.read_csv(test_path, header=None, names=column_names, 
                           skipinitialspace=True, na_values='?', skiprows=1)
    
    # Clean the income column (remove the dot in test data)
    test_data['income'] = test_data['income'].str.replace('.', '')
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    Preprocess the data: handle missing values, encode categorical features,
    and scale numerical features.
    """
    # Identify categorical and numerical columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('income')  # Remove the target variable
    
    numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines for numerical and categorical data
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    # Prepare the target variable
    y_train = (train_data['income'] == '>50K').astype(int)
    y_test = (test_data['income'] == '>50K').astype(int)
    
    # Drop the target variable from features
    X_train = train_data.drop('income', axis=1)
    X_test = test_data.drop('income', axis=1)
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert sparse matrices to dense arrays if needed
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    
    # Get feature names after one-hot encoding
    categorical_features = preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + categorical_features.tolist()
    
    print(f"Processed data shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names):
    """
    Save the processed data and preprocessing pipeline for later use.
    """
    import joblib
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the processed data with allow_pickle=True
    np.save('data/processed/X_train.npy', X_train, allow_pickle=True)
    np.save('data/processed/X_test.npy', X_test, allow_pickle=True)
    np.save('data/processed/y_train.npy', y_train, allow_pickle=True)
    np.save('data/processed/y_test.npy', y_test, allow_pickle=True)
    
    # Save the preprocessor for later use
    joblib.dump(preprocessor, 'data/processed/preprocessor.joblib')
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print("Saved processed data and preprocessor to data/processed/")

if __name__ == "__main__":
    print("Loading data...")
    train_data, test_data = load_data()
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(train_data, test_data)
    
    print("\nSaving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names)
    
    print("\nData preprocessing complete!")
