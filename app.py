from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import json
from src.preprocess import load_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load models and preprocessor
@app.before_first_request
def load_models():
    global models, preprocessor, feature_names
    
    # Load preprocessor
    preprocessor = joblib.load('data/processed/preprocessor.joblib')
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Load all available models
    models = {}
    model_dir = 'data/models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.joblib'):
                model_name = os.path.splitext(filename)[0]
                models[model_name] = joblib.load(os.path.join(model_dir, filename))
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")

# Home page
@app.route('/')
def home():
    # Get model performance data
    performance_data = {}
    fairness_data = {}
    
    if os.path.exists('data/models/model_performance.csv'):
        performance_df = pd.read_csv('data/models/model_performance.csv', index_col=0)
        performance_data = performance_df.to_dict('index')
    
    if os.path.exists('data/models/fairness_metrics.csv'):
        fairness_df = pd.read_csv('data/models/fairness_metrics.csv', index_col=0)
        fairness_data = fairness_df.to_dict('index')
    
    return render_template('index.html', 
                          performance_data=performance_data,
                          fairness_data=fairness_data,
                          model_names=list(performance_data.keys()) if performance_data else [])

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    model_name = form_data.pop('model', 'logistic_regression')
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([form_data])
    
    # Preprocess the input data
    input_processed = preprocessor.transform(input_df)
    
    # Make prediction
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]
        
        result = {
            'prediction': '> $50K' if prediction == 1 else '<= $50K',
            'probability': float(probability),
            'model_used': model_name
        }
    else:
        result = {
            'error': f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        }
    
    return jsonify(result)

# Data exploration page
@app.route('/explore')
def explore():
    # Load raw data for exploration
    train_data, _ = load_data()
    
    # Generate some basic statistics
    stats = {
        'num_samples': len(train_data),
        'num_features': len(train_data.columns) - 1,  # Excluding target
        'income_distribution': train_data['income'].value_counts().to_dict(),
        'categorical_features': train_data.select_dtypes(include=['object']).columns.tolist(),
        'numerical_features': train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    }
    
    # Generate some visualizations
    os.makedirs('static/images/explore', exist_ok=True)
    
    # Income distribution by gender
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_data, x='sex', hue='income')
    plt.title('Income Distribution by Gender')
    plt.tight_layout()
    plt.savefig('static/images/explore/income_by_gender.png')
    plt.close()
    
    # Income distribution by race
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_data, x='race', hue='income')
    plt.title('Income Distribution by Race')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/explore/income_by_race.png')
    plt.close()
    
    # Age distribution by income
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data, x='age', hue='income', bins=30, multiple='stack')
    plt.title('Age Distribution by Income')
    plt.tight_layout()
    plt.savefig('static/images/explore/age_by_income.png')
    plt.close()
    
    return render_template('explore.html', stats=stats)

# About page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Ensure the static/images directory exists
    os.makedirs('static/images', exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
