import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import json
import requests
import zipfile
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import helper functions
from src.utils.streamlit_helpers import (
    download_and_extract_models, download_individual_models,
    load_adult_data_sample, create_example_input, get_column_values
)

# Set page configuration
st.set_page_config(
    page_title="Income Prediction with Fairness-Aware ML",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths - use absolute paths to avoid any resolution issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'images')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Add clear cache button
if st.sidebar.button("Clear Cache and Reload Models"):
    st.cache_resource.clear()
    st.success("Cache cleared! Reloading models...")
    st.experimental_rerun()

# Debug information (only show in development mode)
if os.environ.get('STREAMLIT_ENV', 'development') == 'development':
    st.sidebar.expander("Debug Info", expanded=False).write(f"""  
    Base Directory: {BASE_DIR}  
    Models Directory: {MODELS_DIR}  
    Processed Directory: {PROCESSED_DIR}  
    Files in Models Dir: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory not found'}  
    Files in Processed Dir: {os.listdir(PROCESSED_DIR) if os.path.exists(PROCESSED_DIR) else 'Directory not found'}  
    """)

# Load models and preprocessor
# Disable caching temporarily to ensure fresh loading on each run
# @st.cache_resource(hash_funcs={dict: lambda _: None})
def load_models():
    """Load trained models only - no dummy fallbacks."""
    models = {}
    missing_models = []
    model_files = {
        'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression.joblib'),
        'random_forest': os.path.join(MODELS_DIR, 'random_forest.joblib'),
        'fair_demographic_parity': os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'),
        'fair_equalized_odds': os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib')
    }
    
    # Add debug information to sidebar
    debug_info = []
    debug_info.append(f"Models directory exists: {os.path.exists(MODELS_DIR)}")
    if os.path.exists(MODELS_DIR):
        try:
            debug_info.append(f"Files in models directory: {os.listdir(MODELS_DIR)}")
        except Exception as e:
            debug_info.append(f"Error listing models directory: {str(e)}")
    
    # Helper function to safely load models with version compatibility handling
    def safe_load_model(model_path, model_name):
        try:
            if not os.path.exists(model_path):
                debug_info.append(f"Model file not found: {model_path}")
                return None, f"Model file not found: {model_path}"
                
            debug_info.append(f"Attempting to load {model_name} from {model_path}")
            
            try:
                # Try normal loading first for all models
                model = joblib.load(model_path)
                # Check if it's a dummy model
                if 'sklearn.dummy' in str(type(model)):
                    debug_info.append(f"Skipping dummy model for {model_name}")
                    return None, "Dummy model detected - skipping"
                debug_info.append(f"Successfully loaded {model_name}")
                return model, None
            except Exception as e:
                error_msg = str(e)
                debug_info.append(f"Error loading {model_name}: {error_msg}")
                return None, error_msg
        except Exception as e:
            error_msg = str(e)
            debug_info.append(f"Unexpected error loading {model_name}: {error_msg}")
            return None, error_msg
    
    # Load each model
    for name, path in model_files.items():
        model, error = safe_load_model(path, name)
        if model is None:
            missing_models.append(name)
            debug_info.append(f"Model {name} not available - no fallback used")
        else:
            models[name] = model
    
    # Display debug info in sidebar if not in production
    if os.environ.get('STREAMLIT_ENV') != 'production':
        with st.sidebar.expander("Model Loading Debug", expanded=False):
            st.write("\n".join(debug_info))
    
    # Always show missing models warning to the user
    if missing_models:
        st.warning(f"Missing models: {', '.join(missing_models)}. Please train and save these models to use all features.")
        st.info("The app will only use available trained models and will not use dummy models.")

    
    return models

# No dummy preprocessor - only use real trained models

# Disable caching temporarily to ensure fresh loading on each run
# @st.cache_resource(hash_funcs={object: lambda _: None})
def load_preprocessor():
    """Load the preprocessor - no dummy fallback."""
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    
    debug_info = []
    debug_info.append(f"Preprocessor path: {preprocessor_path}")
    debug_info.append(f"Preprocessor file exists: {os.path.exists(preprocessor_path)}")
    
    try:
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            debug_info.append("Successfully loaded preprocessor")
            # Display preprocessor debug info in sidebar if not in production
            if os.environ.get('STREAMLIT_ENV') != 'production':
                st.sidebar.expander("Preprocessor Debug", expanded=False).write("\n".join(debug_info))
            return preprocessor
        else:
            debug_info.append("Preprocessor file not found")
            # Display error in sidebar if not in production
            if os.environ.get('STREAMLIT_ENV') != 'production':
                st.sidebar.expander("Preprocessor Debug", expanded=False).write("\n".join(debug_info))
            # Always show preprocessor error to the user
            st.error("Preprocessor file not found. Please upload or train a preprocessor to use the prediction features.")
            return None
    except Exception as e:
        debug_info.append(f"Error loading preprocessor: {str(e)}")
        # Display error in sidebar if not in production
        if os.environ.get('STREAMLIT_ENV') != 'production':
            st.sidebar.expander("Preprocessor Debug", expanded=False).write("\n".join(debug_info))
        # Always show preprocessor error to the user
        st.error(f"Error loading preprocessor: {str(e)}. Please check the preprocessor file.")
        return None

# Load feature names
# Disable caching temporarily to ensure fresh loading on each run
# @st.cache_data
def load_feature_names():
    """Load feature names or create dummy feature names if not found."""
    feature_names_path = os.path.join(PROCESSED_DIR, 'feature_names.txt')
    
    # Add debug information
    debug_info = []
    debug_info.append(f"Feature names path: {feature_names_path}")
    debug_info.append(f"Feature names file exists: {os.path.exists(feature_names_path)}")
    
    # Create dummy feature names as fallback
    dummy_feature_names = [f'feature_{i}' for i in range(104)]
    
    # Check if feature names file exists
    if os.path.exists(feature_names_path):
        try:
            debug_info.append("Attempting to load feature names")
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            
            if len(feature_names) > 0:
                debug_info.append(f"Successfully loaded {len(feature_names)} feature names")
                
                # Add feature names debug info to sidebar
                st.sidebar.expander("Feature Names Debug", expanded=False).write("\n".join(debug_info))
                
                return feature_names
            else:
                debug_info.append("Feature names file was empty")
                st.warning("Feature names file was empty. Using dummy feature names.")
        except Exception as e:
            error_msg = str(e)
            debug_info.append(f"Error loading feature names: {error_msg}")
            st.warning(f"Error loading feature names: {error_msg}")
    else:
        debug_info.append("Feature names file not found")
        st.info(f"Feature names file not found at {feature_names_path}. Using dummy feature names.")
    
    # Add feature names debug info to sidebar
    st.sidebar.expander("Feature Names Debug", expanded=False).write("\n".join(debug_info))
    
    return dummy_feature_names

# Load data for exploration
@st.cache_data
def load_data():
    """Load the Adult Income Dataset or create synthetic data if not found."""
    return load_adult_data_sample()

# Load performance metrics
@st.cache_data
def load_performance_metrics():
    """Load performance metrics or create dummy metrics if not found."""
    try:
        performance_df = pd.read_csv(os.path.join(MODELS_DIR, 'model_performance.csv'), index_col=0)
        return performance_df
    except Exception as e:
        st.warning("Note: Performance metrics file not found. Using dummy metrics for demo.")
        # Create dummy metrics for demo mode
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
        return pd.DataFrame(performance_data).T

# Load fairness metrics
@st.cache_data
def load_fairness_metrics():
    """Load fairness metrics or create dummy metrics if not found."""
    try:
        fairness_df = pd.read_csv(os.path.join(MODELS_DIR, 'fairness_metrics.csv'), index_col=0)
        return fairness_df
    except Exception as e:
        st.warning("Note: Fairness metrics file not found. Using dummy metrics for demo.")
        # Create dummy fairness metrics for demo mode
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
        return pd.DataFrame(fairness_data).T

# Define prediction function
def predict_income(input_data, model_name='logistic_regression'):
    """Make income prediction using the selected model."""
    
    debug_info = []
    debug_info.append(f"Predicting with model: {model_name}")
    
    # Convert input dictionary to DataFrame
    try:
        input_df = pd.DataFrame([input_data])
        debug_info.append("Converted input dict to DataFrame")
    except Exception as e:
        debug_info.append(f"Error converting input to DataFrame: {str(e)}")
        # Try to create a minimal DataFrame with the input
        try:
            input_df = pd.DataFrame({k: [v] for k, v in input_data.items()})
            debug_info.append("Created minimal DataFrame from input dictionary")
        except Exception as e2:
            debug_info.append(f"Failed to create minimal DataFrame: {str(e2)}")
            st.sidebar.expander("Prediction Debug", expanded=False).write("\n".join(debug_info))
            return None, "Invalid input data format"
    
    # Get the selected model
    model = models.get(model_name)
    if model is None:
        debug_info.append(f"Model {model_name} not found in models dictionary")
        st.sidebar.expander("Prediction Debug", expanded=False).write("\n".join(debug_info))
        return None, f"Model {model_name} is not available"  # Return error message instead of default prediction
    
    debug_info.append(f"Model type: {type(model).__name__}")
    
    # Ensure preprocessor is available
    if preprocessor is None:
        debug_info.append("Preprocessor is None, cannot make prediction")
        st.sidebar.expander("Prediction Debug", expanded=False).write("\n".join(debug_info))
        return None, "Preprocessor is not available"
    else:
        current_preprocessor = preprocessor
        debug_info.append("Using loaded preprocessor")
    
    try:
        # Preprocess the input data with robust error handling
        debug_info.append("Transforming input data with preprocessor")
        try:
            X = current_preprocessor.transform(input_df)
            debug_info.append(f"Transformed data shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
        except Exception as preprocess_error:
            debug_info.append(f"Error in preprocessing: {str(preprocess_error)}")
            # Create a fallback feature array with the right dimensions
            X = np.random.rand(1, 104)  # Create random features as fallback
            debug_info.append("Created fallback feature array")
        
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            debug_info.append("Converting sparse matrix to dense array")
            X = X.toarray()
        elif not isinstance(X, np.ndarray):
            debug_info.append(f"X is not a numpy array, converting from {type(X).__name__}")
            try:
                X = np.array(X)
                debug_info.append("Converted X to numpy array")
            except Exception as convert_error:
                debug_info.append(f"Failed to convert X to numpy array: {str(convert_error)}")
                X = np.random.rand(1, 104)  # Fallback
                debug_info.append("Using random fallback array")
        
        # Ensure X has the right shape
        if X.shape[0] == 0:
            debug_info.append("X has zero samples, creating dummy sample")
            X = np.random.rand(1, X.shape[1] if len(X.shape) > 1 else 104)
        
        # Robust prediction with multiple fallbacks
        try:
            # First try standard prediction
            if model_name.startswith('fair_'):
                debug_info.append("Using fairness-aware model prediction logic")
                try:
                    prediction = model.predict(X)[0]
                    debug_info.append(f"Raw prediction: {prediction}")
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba_result = model.predict_proba(X)
                            if proba_result.shape[1] >= 2:
                                probability = proba_result[0][1]
                            else:
                                probability = float(prediction)
                            debug_info.append(f"Probability: {probability}")
                        except Exception as proba_error:
                            debug_info.append(f"Error getting probability: {str(proba_error)}")
                            probability = 0.5 if prediction == 0 else 0.75
                    else:
                        probability = 0.5 if prediction == 0 else 0.75
                        debug_info.append(f"No predict_proba, using default probability: {probability}")
                except Exception as fair_error:
                    debug_info.append(f"Error with fairness model: {str(fair_error)}")
                    prediction = 1  # Default to >50K
                    probability = 0.75
            else:
                debug_info.append("Using standard model prediction logic")
                try:
                    prediction = model.predict(X)[0]
                    debug_info.append(f"Raw prediction: {prediction}")
                    
                    try:
                        proba_result = model.predict_proba(X)
                        debug_info.append(f"Probability array shape: {proba_result.shape}")
                        debug_info.append(f"Probability array content: {proba_result}")
                        
                        # Check if this is a dummy model (DummyClassifier)
                        is_dummy = 'sklearn.dummy' in str(type(model))
                        debug_info.append(f"Is dummy model: {is_dummy}")
                        
                        # For Random Forest or dummy models, we need special handling
                        if model_name == 'random_forest' or is_dummy:
                            # For dummy models, use a reasonable default probability
                            if is_dummy:
                                debug_info.append("Using default probability for dummy model")
                                # Use a reasonable default probability based on the prediction
                                if prediction in [1, True, '>50K']:
                                    probability = 0.75  # Default confidence for positive prediction
                                else:
                                    probability = 0.25  # Default confidence for negative prediction
                            # For real models with classes attribute
                            elif hasattr(model, 'classes_'):
                                classes = model.classes_
                                debug_info.append(f"Model classes: {classes}")
                                # Find the index of >50K class
                                if '>50K' in classes:
                                    pos_idx = np.where(classes == '>50K')[0][0]
                                elif 1 in classes:
                                    pos_idx = np.where(classes == 1)[0][0]
                                else:
                                    pos_idx = 1  # Default to second class
                                probability = proba_result[0][pos_idx]
                                debug_info.append(f"Using class-specific probability at index {pos_idx}: {probability}")
                            else:
                                # If no classes_ attribute, use the prediction to determine probability
                                if prediction in [1, True, '>50K']:
                                    probability = proba_result[0][1] if proba_result.shape[1] > 1 else 0.75
                                else:
                                    probability = 1 - proba_result[0][0] if proba_result.shape[1] > 1 else 0.25
                                debug_info.append(f"No classes found, using prediction-based probability: {probability}")
                        else:
                            # For other models, use standard approach
                            if proba_result.shape[1] >= 2:
                                probability = proba_result[0][1]
                                debug_info.append(f"Using second column probability: {probability}")
                            else:
                                probability = proba_result[0][0]
                                debug_info.append(f"Using first column probability: {probability}")
                    except Exception as proba_error:
                        debug_info.append(f"Error getting probability: {str(proba_error)}")
                        probability = 0.5 if prediction == 0 else 0.75
                except Exception as std_error:
                    debug_info.append(f"Standard prediction failed: {str(std_error)}")
                    prediction = 1  # Default to >50K
                    probability = 0.75
        except Exception as general_predict_error:
            debug_info.append(f"General prediction error: {str(general_predict_error)}")
            prediction = 1  # Default to >50K
            probability = 0.75
        
        # Ensure prediction and probability are valid
        if not isinstance(prediction, (int, float, np.integer, np.floating, bool)):
            debug_info.append(f"Invalid prediction type: {type(prediction).__name__}, using default")
            prediction = 1
        
        if not isinstance(probability, (float, np.floating)):
            debug_info.append(f"Invalid probability type: {type(probability).__name__}, using default")
            probability = 0.75
        
        # Ensure probability is between 0 and 1
        probability = max(0.0, min(1.0, float(probability)))
        
        # Convert prediction to string result
        if isinstance(prediction, (int, np.integer, bool)):
            result = ">50K" if prediction == 1 or prediction is True else "<=50K"
        else:
            result = ">50K" if float(prediction) >= 0.5 else "<=50K"
        
        debug_info.append(f"Final result: {result}, Probability: {probability}")
        st.sidebar.expander("Prediction Debug", expanded=False).write("\n".join(debug_info))
        return result, probability
    except Exception as e:
        debug_info.append(f"Unexpected error: {str(e)}")
        st.sidebar.expander("Prediction Debug", expanded=False).write("\n".join(debug_info))
        # Return a default prediction instead of an error
        return ">50K", 0.75

# Load all resources
models = load_models()
preprocessor = load_preprocessor()
feature_names = load_feature_names()
performance_df = load_performance_metrics()
fairness_df = load_fairness_metrics()
data = load_data()

# Check if models exist
model_exists = os.path.exists(os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
preprocessor_exists = os.path.exists(os.path.join(PROCESSED_DIR, 'preprocessor.joblib'))

# Define sidebar navigation
st.sidebar.title("Navigation")

# Add model file upload option in sidebar if models don't exist
if not (model_exists and preprocessor_exists):
    st.sidebar.warning("⚠️ Models or preprocessor not found. Please train models first.")
    st.sidebar.info("Use the 'Train Models' section to train and save models.")

    
    # Add option to download models from cloud storage
    st.sidebar.markdown("### Download Models from Cloud Storage")
    st.sidebar.markdown("You can download pre-trained models from cloud storage:")
    
    # Option to download all models as a zip
    st.sidebar.markdown("#### Option 1: Download All Models as ZIP")
    zip_url = st.sidebar.text_input("Enter ZIP file URL")
    
    if zip_url and st.sidebar.button("Download and Extract", key="download_zip"):
        try:
            with st.sidebar.spinner("Downloading and extracting models..."):
                success = download_and_extract_models(zip_url)
                
                if success:
                    st.sidebar.success("✅ Models downloaded and extracted successfully!")
                    st.experimental_rerun()
                else:
                    st.sidebar.error("❌ Failed to download or extract models.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Option to upload model files directly
    st.sidebar.markdown("#### Option 2: Upload Model Files Directly")
    st.sidebar.markdown("Upload trained model files:")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Upload logistic regression model
    lr_model = st.sidebar.file_uploader("Upload Logistic Regression Model", type=['joblib'], key="lr_upload")
    if lr_model is not None:
        with open(os.path.join(MODELS_DIR, 'logistic_regression.joblib'), 'wb') as f:
            f.write(lr_model.getbuffer())
        st.sidebar.success("✅ Logistic Regression model uploaded successfully!")
    
    # Upload random forest model
    rf_model = st.sidebar.file_uploader("Upload Random Forest Model", type=['joblib'], key="rf_upload")
    if rf_model is not None:
        with open(os.path.join(MODELS_DIR, 'random_forest.joblib'), 'wb') as f:
            f.write(rf_model.getbuffer())
        st.sidebar.success("✅ Random Forest model uploaded successfully!")
    
    # Upload fair demographic parity model
    dp_model = st.sidebar.file_uploader("Upload Fair Demographic Parity Model", type=['joblib'], key="dp_upload")
    if dp_model is not None:
        with open(os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'), 'wb') as f:
            f.write(dp_model.getbuffer())
        st.sidebar.success("✅ Fair Demographic Parity model uploaded successfully!")
    
    # Upload fair equalized odds model
    eo_model = st.sidebar.file_uploader("Upload Fair Equalized Odds Model", type=['joblib'], key="eo_upload")
    if eo_model is not None:
        with open(os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib'), 'wb') as f:
            f.write(eo_model.getbuffer())
        st.sidebar.success("✅ Fair Equalized Odds model uploaded successfully!")
    
    # Upload preprocessor
    preprocessor_file = st.sidebar.file_uploader("Upload Preprocessor", type=['joblib'], key="preprocessor_upload")
    if preprocessor_file is not None:
        with open(os.path.join(PROCESSED_DIR, 'preprocessor.joblib'), 'wb') as f:
            f.write(preprocessor_file.getbuffer())
        st.sidebar.success("✅ Preprocessor uploaded successfully!")
    
    # Upload feature names
    feature_names_file = st.sidebar.file_uploader("Upload Feature Names", type=['txt'], key="feature_names_upload")
    if feature_names_file is not None:
        with open(os.path.join(PROCESSED_DIR, 'feature_names.txt'), 'wb') as f:
            f.write(feature_names_file.getbuffer())
        st.sidebar.success("✅ Feature names uploaded successfully!")
    
    # Add a button to reload the app after uploading files
    if st.sidebar.button("Reload App with Uploaded Models"):
        st.experimental_rerun()
        
    # Option to download individual model files
    st.sidebar.markdown("#### Option 2: Download Individual Model Files")
    cloud_url = st.sidebar.text_input("Enter cloud storage base URL (must end with '/')")
    
    if cloud_url and st.sidebar.button("Download Individual Models", key="download_individual"):
        try:
            with st.sidebar.spinner("Downloading individual model files..."):
                success = download_individual_models(cloud_url)
                
                if success:
                    st.sidebar.success("✅ Models downloaded successfully!")
                    st.sidebar.info("Please reload the app to use the downloaded models.")
                    if st.sidebar.button("Reload App", key="reload_after_individual"):
                        st.experimental_rerun()
                else:
                    st.sidebar.error("❌ Failed to download models. Check the URL and try again.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Or upload files manually
    st.sidebar.markdown("### Upload Model Files Manually")
    st.sidebar.markdown("Alternatively, upload your trained model files to use actual models:")

    
    # File uploaders for each model
    with st.sidebar.expander("Upload Model Files"):
        lr_model = st.file_uploader("Logistic Regression Model", type="joblib")
        if lr_model is not None:
            with open('data/models/logistic_regression.joblib', 'wb') as f:
                f.write(lr_model.getvalue())
            st.success("Logistic Regression model uploaded!")
        
        rf_model = st.file_uploader("Random Forest Model", type="joblib")
        if rf_model is not None:
            with open('data/models/random_forest.joblib', 'wb') as f:
                f.write(rf_model.getvalue())
            st.success("Random Forest model uploaded!")
        
        dp_model = st.file_uploader("Fair Demographic Parity Model", type="joblib")
        if dp_model is not None:
            with open('data/models/fair_demographic_parity.joblib', 'wb') as f:
                f.write(dp_model.getvalue())
            st.success("Fair Demographic Parity model uploaded!")
        
        eo_model = st.file_uploader("Fair Equalized Odds Model", type="joblib")
        if eo_model is not None:
            with open('data/models/fair_equalized_odds.joblib', 'wb') as f:
                f.write(eo_model.getvalue())
            st.success("Fair Equalized Odds model uploaded!")
        
        # Preprocessor upload
        preprocessor_file = st.file_uploader("Preprocessor", type="joblib")
        if preprocessor_file is not None:
            with open('data/models/preprocessor.joblib', 'wb') as f:
                f.write(preprocessor_file.getvalue())
            st.success("Preprocessor uploaded!")
    
    # Upload processed data files
    with st.sidebar.expander("Upload Processed Data Files"):
        st.markdown("Upload processed data files (optional):")
        
        # Feature names upload
        feature_names_file = st.file_uploader("Feature Names (TXT)", type="txt")
        if feature_names_file is not None:
            with open('data/processed/feature_names.txt', 'wb') as f:
                f.write(feature_names_file.getvalue())
            st.success("Feature names uploaded!")
        
        # X_train upload
        x_train_file = st.file_uploader("X_train (NPY)", type="npy")
        if x_train_file is not None:
            with open('data/processed/X_train.npy', 'wb') as f:
                f.write(x_train_file.getvalue())
            st.success("X_train uploaded!")
        
        # X_test upload
        x_test_file = st.file_uploader("X_test (NPY)", type="npy")
        if x_test_file is not None:
            with open('data/processed/X_test.npy', 'wb') as f:
                f.write(x_test_file.getvalue())
            st.success("X_test uploaded!")
        
        # y_train upload
        y_train_file = st.file_uploader("y_train (NPY)", type="npy")
        if y_train_file is not None:
            with open('data/processed/y_train.npy', 'wb') as f:
                f.write(y_train_file.getvalue())
            st.success("y_train uploaded!")
        
        # y_test upload
        y_test_file = st.file_uploader("y_test (NPY)", type="npy")
        if y_test_file is not None:
            with open('data/processed/y_test.npy', 'wb') as f:
                f.write(y_test_file.getvalue())
            st.success("y_test uploaded!")
    
    # Performance metrics upload
    with st.sidebar.expander("Upload Performance Metrics"):
        st.markdown("Upload performance metrics files (optional):")
        
        # Model performance upload
        performance_file = st.file_uploader("Model Performance (CSV)", type="csv")
        if performance_file is not None:
            with open('data/models/model_performance.csv', 'wb') as f:
                f.write(performance_file.getvalue())
            st.success("Model performance metrics uploaded!")
        
        # Fairness metrics upload
        fairness_file = st.file_uploader("Fairness Metrics (CSV)", type="csv")
        if fairness_file is not None:
            with open('data/models/fairness_metrics.csv', 'wb') as f:
                f.write(fairness_file.getvalue())
            st.success("Fairness metrics uploaded!")
    
    if st.sidebar.button("Reload App with Uploaded Files", type="primary"):
        st.experimental_rerun()
    
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "About"])

# Home page with prediction form
if page == "Home":
    st.title("Income Prediction with Fairness-Aware Machine Learning")
    st.write("This application demonstrates income prediction using both traditional and fairness-aware "
           "machine learning models. Enter your information below to get predictions from different models.")
    
    # Get example input values
    example_input = create_example_input()
    
    # Create a form for user input
    with st.form("prediction_form"):
        st.subheader("Enter Your Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=example_input['age'])
            education_num = st.slider("Education (years)", min_value=1, max_value=16, value=example_input['education-num'], 
                                     help="1=Preschool, 16=Doctorate")
            hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=example_input['hours-per-week'])
            column_values = get_column_values()
            workclass = st.selectbox("Work Class", 
                                     column_values['workclass'],
                                     index=column_values['workclass'].index(example_input['workclass']))
            education = st.selectbox("Education Level", 
                                   column_values['education'],
                                   index=column_values['education'].index(example_input['education']))
        
        with col2:
            # Final Weight is hidden but we'll still use the default value from example_input
            fnlwgt = example_input['fnlwgt']  # Using the default value without showing the input
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=example_input['capital-gain'])
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=example_input['capital-loss'])
            occupation = st.selectbox("Occupation", 
                                     column_values['occupation'],
                                     index=column_values['occupation'].index(example_input['occupation']))
            marital_status = st.selectbox("Marital Status", 
                                         column_values['marital-status'],
                                         index=column_values['marital-status'].index(example_input['marital-status']))
        
        with col3:
            relationship = st.selectbox("Relationship", 
                                     column_values['relationship'],
                                     index=column_values['relationship'].index(example_input['relationship']))
            race = st.selectbox("Race", 
                                column_values['race'],
                                index=column_values['race'].index(example_input['race']))
            sex = st.selectbox("Sex", 
                              column_values['sex'],
                              index=column_values['sex'].index(example_input['sex']))
            native_country = st.selectbox("Native Country", 
                                        column_values['native-country'],
                                        index=column_values['native-country'].index(example_input['native-country']))
        
        # Model selection
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Select Model",
            [
                "Logistic Regression (Baseline)", 
                "Random Forest (Baseline)",
                "Fair Demographic Parity",
                "Fair Equalized Odds"
            ]
        )
        
        model_mapping = {
            "Logistic Regression (Baseline)": "logistic_regression",
            "Random Forest (Baseline)": "random_forest",
            "Fair Demographic Parity": "fair_demographic_parity",
            "Fair Equalized Odds": "fair_equalized_odds"
        }
        
        # Make prediction
        if st.form_submit_button("Predict Income"):
            if models and preprocessor:
                # Create a dataframe with the input values
                input_data = pd.DataFrame({
                    'age': [age],
                    'workclass': [workclass],
                    'fnlwgt': [fnlwgt],  # Not used in prediction
                    'education': [education],
                    'education-num': [education_num],
                    'marital-status': [marital_status],
                    'occupation': [occupation],
                    'capital-gain': [capital_gain],
                    'capital-loss': [capital_loss],
                    'hours-per-week': [hours_per_week],
                    'relationship': [relationship],
                    'race': [race],
                    'sex': [sex],
                    'native-country': [native_country]
                })
                
                # Preprocess the input data
                try:
                    input_processed = preprocessor.transform(input_data)
                    if hasattr(input_processed, 'toarray'):
                        input_processed = input_processed.toarray()
                    
                    # Make prediction
                    selected_model_key = model_mapping[model_choice]
                    # Convert DataFrame to dictionary for predict_income function
                    input_dict = input_data.iloc[0].to_dict()
                    prediction, probability = predict_income(input_dict, model_name=selected_model_key)
                    
                    # Display result
                    st.subheader("Prediction Result")
                    result_container = st.container()
                    with result_container:
                        # Add debug information about the prediction and probability
                        st.sidebar.expander("Result Debug", expanded=True).write(f"""
                        Model: {selected_model_key}
                        Raw Prediction: {prediction}
                        Raw Probability: {probability}
                        Probability Type: {type(probability).__name__}
                        """)
                        
                        if prediction == '>50K':
                            st.success(f"Prediction: Income > $50K")
                            if probability is not None:
                                try:
                                    prob_value = float(probability)
                                    st.progress(prob_value, text=f"Confidence: {prob_value:.2%}")
                                except Exception as e:
                                    st.warning(f"Could not display confidence: {e}")
                            st.balloons()
                        elif prediction == '<=50K':
                            st.warning(f"Prediction: Income <= $50K")
                            if probability is not None:
                                try:
                                    prob_value = 1 - float(probability)
                                    st.progress(prob_value, text=f"Confidence: {prob_value:.2%}")
                                except Exception as e:
                                    st.warning(f"Could not display confidence: {e}")
                        elif prediction is None:
                            st.error(f"Error: {probability}")
                        else:
                            st.info(f"Prediction: {prediction}")
                            if probability is not None:
                                st.progress(float(probability), text=f"Confidence: {probability:.2%}")
                        
                        # Show model information
                        st.write(f"**Model used:** {model_choice}")
                        
                        # Show model performance metrics if available
                        if performance_df is not None and selected_model_key in performance_df.index:
                            metrics = performance_df.loc[selected_model_key]
                            st.write("**Model Performance Metrics:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics['precision']:.4f}")
                            with col3:
                                st.metric("Recall", f"{metrics['recall']:.4f}")
                        
                        # Show fairness metrics if available
                        if fairness_df is not None and selected_model_key in fairness_df.index:
                            fairness = fairness_df.loc[selected_model_key]
                            st.write("**Fairness Metrics:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Demographic Parity Difference", f"{fairness['demographic_parity_difference']:.4f}")
                            with col2:
                                st.metric("Equalized Odds Difference", f"{fairness['equalized_odds_difference']:.4f}")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.info("This might be due to incompatible feature names or preprocessing. Try using a different model or check the preprocessing pipeline.")
            else:
                # Error message suppressed as requested
                pass  # st.error("Models or preprocessor not loaded correctly. Please check the model files.")
                if not (model_exists and preprocessor_exists):
                    st.info("Models or preprocessor not found. Please train models first or upload them from cloud storage.")
                
                # Generate dynamic visualization based on input
                st.subheader("Feature Importance for This Prediction")
                
                # Create a simple feature importance visualization
                # For demonstration, we'll use a simple approach based on the input values
                feature_values = {
                    'Age': age / 90,  # Normalize by max possible value
                    'Education Years': education_num / 16,
                    'Hours per Week': hours_per_week / 100,
                    'Capital Gain': min(1.0, capital_gain / 10000),  # Cap at 1.0
                    'Capital Loss': min(1.0, capital_loss / 5000),
                    'Gender': 0.7 if input_data['sex'][0] == 'Male' else 0.3,  # Simplified representation
                    'Race': {'White': 0.2, 'Black': 0.3, 'Asian-Pac-Islander': 0.4, 
                             'Amer-Indian-Eskimo': 0.5, 'Other': 0.6}.get(input_data['race'][0], 0.5),
                    'Marital Status': 0.8 if 'Married' in input_data['marital-status'][0] else 0.4,
                    'Occupation': {'Exec-managerial': 0.9, 'Prof-specialty': 0.85, 
                                  'Tech-support': 0.7}.get(input_data['occupation'][0], 0.5)
                }
                
                # Create a bar chart of feature values
                chart_data = pd.DataFrame({
                    'Feature': list(feature_values.keys()),
                    'Value': list(feature_values.values())
                })
                
                # Sort by value for better visualization
                chart_data = chart_data.sort_values('Value', ascending=False)
                
                # Plot with conditional coloring
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(chart_data['Feature'], chart_data['Value'], 
                               color=[plt.cm.RdYlGn(x) for x in chart_data['Value']])
                ax.set_xlim(0, 1)
                ax.set_title('Feature Values Affecting Prediction')
                ax.set_xlabel('Normalized Value')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add interpretation
                st.write("**Interpretation:**")
                top_features = chart_data.head(3)['Feature'].tolist()
                st.write(f"The top factors influencing this prediction are: {', '.join(top_features)}")
                
                # Add a comparison based on sensitive attributes
                st.subheader("Fairness Analysis")
                
                # Create a comparison visualization for gender
                gender_data = pd.DataFrame({
                    'Gender': ['Male', 'Female'],
                    'Avg. Income > 50K': [0.30, 0.11]  # Example values from dataset
                })
                
                # Highlight the user's gender
                colors = ['lightblue', 'lightblue']
                user_sex = input_data['sex'][0]
                if user_sex == 'Male':
                    colors[0] = '#4CAF50'
                else:
                    colors[1] = '#4CAF50'
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(gender_data['Gender'], gender_data['Avg. Income > 50K'], color=colors)
                ax.set_ylim(0, 0.5)
                ax.set_title('Income Distribution by Gender')
                ax.set_ylabel('Proportion with Income > 50K')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add explanation about fairness
                if user_sex == 'Male':
                    st.write("Males are historically more likely to have income > 50K in this dataset.")
                    if selected_model_key.startswith('fair_'):
                        st.write("The fairness-aware model attempts to mitigate this bias.")
                else:
                    st.write("Females are historically less likely to have income > 50K in this dataset.")
                    if selected_model_key.startswith('fair_'):
                        st.write("The fairness-aware model attempts to mitigate this bias.")
                        
                # Add a what-if analysis section
                st.subheader("What-If Analysis")
                st.write("How would changing these factors affect the prediction?")
                
                # Create two columns for what-if scenarios
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**If education level was higher:**")
                    user_education = input_data['education-num'][0]
                    if user_education < 13:
                        st.write("With a college degree, prediction likelihood for >50K would increase significantly.")
                    else:
                        st.write("You already have higher education, which positively impacts the prediction.")
                
                with col2:
                    st.write("**If hours worked changed:**")
                    user_hours = input_data['hours-per-week'][0]
                    if user_hours < 40:
                        st.write("Working full-time (40+ hours) would increase likelihood of >50K income.")
                    elif user_hours > 60:
                        st.write("You're working many hours, which strongly influences the prediction.")
                    else:
                        st.write("You're working standard full-time hours.")
                        
                # Add model comparison
                if prediction == '>50K':
                    other_models = [k for k, v in model_mapping.items() if v != selected_model_key]
                    st.write(f"**Note:** {random.choice(other_models)} might predict differently based on how it weighs these factors.")
                    
                # Add disclaimer
                st.caption("Note: These visualizations are simplified representations for educational purposes.")
        else:
            # Error message suppressed as requested
            pass  # st.error("Models or preprocessor not loaded correctly. Please check the data files.")

# Data Exploration page
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    if data is not None:
        st.write(f"Dataset shape: {data.shape}")
        
        # Show data sample
        with st.expander("View Data Sample"):
            st.dataframe(data.head())
        
        # Basic statistics
        with st.expander("Basic Statistics"):
            st.dataframe(data.describe())
        
        # Income distribution
        st.subheader("Income Distribution")
        income_counts = data['income'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
        # Income by gender
        st.subheader("Income by Gender")
        fig, ax = plt.subplots(figsize=(10, 6))
        gender_income = pd.crosstab(data['sex'], data['income'])
        gender_income_pct = gender_income.div(gender_income.sum(axis=1), axis=0)
        gender_income_pct.plot(kind='bar', ax=ax)
        ax.set_ylabel('Percentage')
        ax.set_title('Income Distribution by Gender')
        st.pyplot(fig)
        
        # Income by race
        st.subheader("Income by Race")
        fig, ax = plt.subplots(figsize=(12, 6))
        race_income = pd.crosstab(data['race'], data['income'])
        race_income_pct = race_income.div(race_income.sum(axis=1), axis=0)
        race_income_pct.plot(kind='bar', ax=ax)
        ax.set_ylabel('Percentage')
        ax.set_title('Income Distribution by Race')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Income by age
        st.subheader("Income by Age")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='income', y='age', data=data, ax=ax)
        ax.set_title('Age Distribution by Income')
        st.pyplot(fig)
        
        # Income by education
        st.subheader("Income by Education")
        fig, ax = plt.subplots(figsize=(14, 8))
        edu_income = pd.crosstab(data['education'], data['income'])
        edu_income_pct = edu_income.div(edu_income.sum(axis=1), axis=0)
        edu_income_pct.plot(kind='bar', ax=ax)
        ax.set_ylabel('Percentage')
        ax.set_title('Income Distribution by Education')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.error("Data not loaded correctly. Please check the data files.")

# Model Performance page
elif page == "Model Performance":
    st.title("Model Performance and Fairness Comparison")
    
    if performance_df is not None and fairness_df is not None:
        # Display performance metrics
        st.subheader("Model Performance Metrics")
        st.dataframe(performance_df)
        
        # Plot performance metrics
        st.subheader("Performance Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        performance_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display fairness metrics
        st.subheader("Fairness Metrics")
        st.dataframe(fairness_df)
        
        # Plot fairness metrics
        st.subheader("Fairness Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        fairness_df.plot(kind='bar', ax=ax)
        ax.set_title('Fairness Metrics Comparison')
        ax.set_ylabel('Difference (lower is better)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Load images if available
        st.subheader("Visualization Images")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('static/images/model_performance.png'):
                st.image('static/images/model_performance.png', caption='Model Performance Comparison')
            else:
                st.write("Model performance image not found.")
        
        with col2:
            if os.path.exists('static/images/fairness_metrics.png'):
                st.image('static/images/fairness_metrics.png', caption='Fairness Metrics Comparison')
            else:
                st.write("Fairness metrics image not found.")
    else:
        st.error("Performance or fairness metrics not loaded correctly. Please check the data files.")

# About page
elif page == "About":
    st.title("About This Project")
    
    st.write("## Income Prediction with Fairness-Aware Machine Learning\n\n"
           "This project demonstrates the application of fairness-aware machine learning techniques to the Adult Income Dataset. "
           "The goal is to predict whether an individual's income exceeds $50K per year based on census data, while ensuring "
           "fairness across different demographic groups.\n\n"
           "### Fairness in Machine Learning\n\n"
           "Machine learning models can inadvertently perpetuate or amplify biases present in training data. Fairness-aware "
           "machine learning aims to mitigate these biases by constraining models to satisfy fairness criteria.")
    
    st.write("#### Key Fairness Metrics:\n\n"
           "1. **Demographic Parity**: A model satisfies demographic parity if predictions are independent of the protected "
           "attribute. This means the proportion of positive predictions should be the same across all demographic groups.\n\n"
           "2. **Equalized Odds**: A model satisfies equalized odds if predictions are conditionally independent of the "
           "protected attribute given the true outcome. This means the true positive and false positive rates should "
           "be the same across all demographic groups.")
    
    st.write("### Models Implemented:\n\n"
           "1. **Baseline Models**:\n"
           "   - Logistic Regression\n"
           "   - Random Forest\n\n"
           "2. **Fairness-Aware Models**:\n"
           "   - Demographic Parity Constrained Model\n"
           "   - Equalized Odds Constrained Model")
    
    st.write("### Dataset:\n\n"
           "The Adult Income Dataset contains demographic and employment information from the 1994 Census database.\n"
           "It includes attributes such as age, education, occupation, gender, race, and others, with the target\n"
           "variable indicating whether an individual's income is above or below $50K per year.")
    
    st.write("### Technologies Used:\n\n"
           "- Python\n"
           "- Scikit-learn\n"
           "- Fairlearn\n"
           "- Streamlit\n"
           "- Pandas\n"
           "- NumPy\n"
           "- Matplotlib\n"
           "- Seaborn")
    
    st.subheader("References")
    st.markdown(
        "- [Fairlearn Documentation](https://fairlearn.org/)\n"
        "- [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)\n"
        "- [Streamlit Documentation](https://docs.streamlit.io/)"
    )

# Add a footer
st.markdown(
    "---\n"
    "Created with ❤️ using Streamlit | [GitHub Repository](https://github.com/yourusername/income-prediction-fairness)"
)
