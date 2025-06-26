#!/usr/bin/env python
"""
Script to package model files for upload to cloud storage.
This script will create a zip file containing all model files needed for the app.
"""

import os
import zipfile
import shutil
from datetime import datetime

def package_models(output_dir="./"):
    """
    Package all model files into a zip file for easy upload to cloud storage.
    
    Parameters:
    -----------
    output_dir : str
        Directory where the zip file will be created
    
    Returns:
    --------
    str
        Path to the created zip file
    """
    # Create a timestamp for the zip file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(output_dir, f"income_prediction_models_{timestamp}.zip")
    
    # List of model files to package
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
    
    # Check if all files exist
    missing_files = [f for f in model_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: The following files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nOnly existing files will be packaged.")
    
    # Create a temporary directory for flat file structure
    temp_dir = os.path.join(output_dir, "temp_models")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copy files to temp directory with flat structure
        copied_files = []
        for file_path in model_files:
            if os.path.exists(file_path):
                # Get just the filename without directories
                filename = os.path.basename(file_path)
                # Copy to temp dir
                shutil.copy2(file_path, os.path.join(temp_dir, filename))
                copied_files.append(filename)
        
        # Create zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in copied_files:
                zipf.write(os.path.join(temp_dir, filename), filename)
        
        print(f"\nSuccessfully created {zip_filename}")
        print(f"Contains {len(copied_files)} files:")
        for filename in copied_files:
            print(f"  - {filename}")
        
        return zip_filename
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_upload_instructions(zip_filename):
    """
    Create instructions for uploading the models to cloud storage
    """
    instructions = f"""
# Instructions for Uploading Model Files

You've successfully packaged your model files into {os.path.basename(zip_filename)}.
Follow these steps to make them available to your Streamlit app:

## Option 1: Using GitHub Releases (Recommended)

1. Go to your GitHub repository: https://github.com/TarunReddyKethireddy/Income-prediction-fairness
2. Click on "Releases" on the right side
3. Click "Create a new release"
4. Tag version: v1.0.0 (or increment as needed)
5. Release title: "Model Files for Income Prediction"
6. Drag and drop the {os.path.basename(zip_filename)} file into the assets area
7. Click "Publish release"
8. After publishing, click on the uploaded zip file to get its download URL
9. In your Streamlit app, use this URL pattern:
   https://github.com/TarunReddyKethireddy/Income-prediction-fairness/releases/download/v1.0.0/

## Option 2: Using Google Drive

1. Upload {os.path.basename(zip_filename)} to Google Drive
2. Right-click on the file and select "Get link"
3. Make sure to set sharing to "Anyone with the link can view"
4. Copy the link ID (the long string after /d/ in the URL)
5. Use this URL pattern in your Streamlit app:
   https://drive.google.com/uc?export=download&id=YOUR_FILE_ID

## Option 3: Using Dropbox

1. Upload {os.path.basename(zip_filename)} to Dropbox
2. Create a shared link
3. In the shared link, change "www.dropbox.com" to "dl.dropboxusercontent.com"
4. Remove "?dl=0" at the end and replace with "?dl=1"
5. Use this modified URL in your Streamlit app

After uploading, extract the files to create individual model file URLs.
Then use the cloud URL in your Streamlit app's "Download Models from Cloud" feature.
"""
    
    # Write instructions to a file
    instructions_file = os.path.join(os.path.dirname(zip_filename), "upload_instructions.md")
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"\nInstructions written to {instructions_file}")
    return instructions_file

if __name__ == "__main__":
    print("Packaging model files for cloud storage...")
    zip_file = package_models()
    create_upload_instructions(zip_file)
    print("\nDone! Follow the instructions in upload_instructions.md to upload your models.")
