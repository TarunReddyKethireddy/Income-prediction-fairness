#!/usr/bin/env python
"""
Script to extract model files from a zip archive.
This script is used by the Streamlit app to extract model files downloaded from cloud storage.
"""

import os
import zipfile
import tempfile
import requests
import streamlit as st

def download_zip_from_url(url):
    """
    Download a zip file from a URL and save it to a temporary file.
    
    Parameters:
    -----------
    url : str
        URL of the zip file
    
    Returns:
    --------
    str
        Path to the downloaded zip file
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_file.close()
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error downloading zip file: {e}")
        return None

def extract_models_from_zip(zip_path):
    """
    Extract model files from a zip archive.
    
    Parameters:
    -----------
    zip_path : str
        Path to the zip file
    
    Returns:
    --------
    bool
        True if extraction was successful, False otherwise
    """
    try:
        # Create directories if they don't exist
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Extract files from zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the zip
            file_list = zip_ref.namelist()
            
            # Process each file
            for filename in file_list:
                # Determine the target directory based on filename
                if filename.endswith('.joblib') or filename.endswith('.csv'):
                    target_dir = 'data/models'
                elif filename.endswith('.npy') or filename.endswith('.txt'):
                    target_dir = 'data/processed'
                else:
                    # Skip unknown file types
                    continue
                
                # Extract the file
                zip_ref.extract(filename, target_dir)
                
                # Move file to correct location if needed
                source_path = os.path.join(target_dir, filename)
                target_path = os.path.join(target_dir, os.path.basename(filename))
                
                if source_path != target_path:
                    os.rename(source_path, target_path)
        
        return True
    except Exception as e:
        st.error(f"Error extracting model files: {e}")
        return False

def download_and_extract_models(url):
    """
    Download a zip file from a URL and extract model files.
    
    Parameters:
    -----------
    url : str
        URL of the zip file
    
    Returns:
    --------
    bool
        True if download and extraction were successful, False otherwise
    """
    try:
        # Download the zip file
        zip_path = download_zip_from_url(url)
        if not zip_path:
            return False
        
        # Extract model files
        success = extract_models_from_zip(zip_path)
        
        # Clean up
        os.unlink(zip_path)
        
        return success
    except Exception as e:
        st.error(f"Error downloading and extracting model files: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    url = "https://example.com/models.zip"
    download_and_extract_models(url)
