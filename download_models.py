import os
import requests
import streamlit as st

# Create necessary directories
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Function to download a file from a URL and save it to a local path
def download_file(url, local_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"Error downloading {local_path}: {e}")
        return False

# Function to download all model files
def download_all_models(base_url):
    """
    Download all model files from a base URL.
    
    Parameters:
    -----------
    base_url : str
        The base URL where model files are stored. Should end with a slash.
    
    Returns:
    --------
    bool
        True if all downloads were successful, False otherwise.
    """
    # List of model files to download
    model_files = [
        'data/models/logistic_regression.joblib',
        'data/models/random_forest.joblib',
        'data/models/fair_demographic_parity.joblib',
        'data/models/fair_equalized_odds.joblib',
        'data/models/preprocessor.joblib',
        'data/models/model_performance.csv',
        'data/models/fairness_metrics.csv',
        'data/processed/feature_names.txt',
        'data/processed/X_train.npy',
        'data/processed/X_test.npy',
        'data/processed/y_train.npy',
        'data/processed/y_test.npy'
    ]
    
    success = True
    for file_path in model_files:
        # Extract the filename from the path
        filename = os.path.basename(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download the file
        file_url = f"{base_url}{filename}"
        success = success and download_file(file_url, file_path)
    
    return success

if __name__ == "__main__":
    # Example usage
    base_url = "https://example.com/models/"
    download_all_models(base_url)
