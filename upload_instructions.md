
# Instructions for Uploading Model Files

You've successfully packaged your model files into income_prediction_models_20250626_125259.zip.
Follow these steps to make them available to your Streamlit app:

## Option 1: Using GitHub Releases (Recommended)

1. Go to your GitHub repository: https://github.com/TarunReddyKethireddy/Income-prediction-fairness
2. Click on "Releases" on the right side
3. Click "Create a new release"
4. Tag version: v1.0.0 (or increment as needed)
5. Release title: "Model Files for Income Prediction"
6. Drag and drop the income_prediction_models_20250626_125259.zip file into the assets area
7. Click "Publish release"
8. After publishing, click on the uploaded zip file to get its download URL
9. In your Streamlit app, use this URL pattern:
   https://github.com/TarunReddyKethireddy/Income-prediction-fairness/releases/download/v1.0.0/

## Option 2: Using Google Drive

1. Upload income_prediction_models_20250626_125259.zip to Google Drive
2. Right-click on the file and select "Get link"
3. Make sure to set sharing to "Anyone with the link can view"
4. Copy the link ID (the long string after /d/ in the URL)
5. Use this URL pattern in your Streamlit app:
   https://drive.google.com/uc?export=download&id=YOUR_FILE_ID

## Option 3: Using Dropbox

1. Upload income_prediction_models_20250626_125259.zip to Dropbox
2. Create a shared link
3. In the shared link, change "www.dropbox.com" to "dl.dropboxusercontent.com"
4. Remove "?dl=0" at the end and replace with "?dl=1"
5. Use this modified URL in your Streamlit app

After uploading, extract the files to create individual model file URLs.
Then use the cloud URL in your Streamlit app's "Download Models from Cloud" feature.
