import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.base import BaseEstimator, TransformerMixin

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create Flask app
app = Flask(__name__, 
            static_folder=STATIC_DIR,
            template_folder=TEMPLATES_DIR)

# Load models and preprocessor
def load_models():
    """Load trained models."""
    models = {}
    model_files = {
        'logistic_regression': os.path.join(MODELS_DIR, 'logistic_regression.joblib'),
        'random_forest': os.path.join(MODELS_DIR, 'random_forest.joblib'),
        'fair_demographic_parity': os.path.join(MODELS_DIR, 'fair_demographic_parity.joblib'),
        'fair_equalized_odds': os.path.join(MODELS_DIR, 'fair_equalized_odds.joblib')
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"Warning: Model {name} not found at {path}")
    
    return models

def load_preprocessor():
    """Load trained preprocessor."""
    preprocessor_path = os.path.join(PROCESSED_DIR, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        return joblib.load(preprocessor_path)
    else:
        print(f"Warning: Preprocessor not found at {preprocessor_path}")
        return None

def load_feature_names():
    """Load feature names."""
    feature_names_path = os.path.join(PROCESSED_DIR, 'feature_names.txt')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            return f.read().splitlines()
    else:
        print(f"Warning: Feature names not found at {feature_names_path}")
        return None

# Get categorical column values
def get_column_values():
    return {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        'sex': ['Female', 'Male'],
        'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    }

# Create example input
def create_example_input():
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

# Load models and preprocessor
models = load_models()
preprocessor = load_preprocessor()
feature_names = load_feature_names()

# Routes
@app.route('/')
def home():
    return render_template('index.html', column_values=get_column_values(), example_input=create_example_input())

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    
    # Convert numeric fields
    for field in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        if field in data:
            data[field] = float(data[field])
    
    # Create dataframe from input
    input_df = pd.DataFrame([data])
    
    # Preprocess input
    if preprocessor:
        X = preprocessor.transform(input_df)
    else:
        return jsonify({'error': 'Preprocessor not available'})
    
    # Make predictions with all available models
    results = {}
    for name, model in models.items():
        try:
            # Handle fairness-aware models differently
            if name.startswith('fair_'):
                # Fairness models might use a different interface
                # First try to get predictions directly
                prediction = model.predict(X)[0]
                
                # For probability, check if the model has a predict_proba method
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0][1]
                else:
                    # If not, use the prediction as a binary outcome
                    proba = float(prediction)
            else:
                # Standard scikit-learn models
                proba = model.predict_proba(X)[0][1]
                prediction = 1 if proba >= 0.5 else 0
                
            results[name] = {
                'prediction': '>50K' if prediction == 1 else '<=50K',
                'probability': float(proba)
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return jsonify(results)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
    
    # Create HTML template
    with open(os.path.join(TEMPLATES_DIR, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Income Prediction with Fairness-Aware Machine Learning</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .form-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .form-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 15px;
        }
        .form-group {
            flex: 1;
            min-width: 200px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .results {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            display: none;
        }
        .model-result {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid #4CAF50;
            background-color: #f0f0f0;
        }
        .prediction {
            font-weight: bold;
            font-size: 18px;
        }
        .probability {
            margin-top: 5px;
            color: #666;
        }
        .high-income {
            color: #4CAF50;
        }
        .low-income {
            color: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Income Prediction with Fairness-Aware Machine Learning</h1>
        <p>This application demonstrates income prediction using both traditional and fairness-aware machine learning models. Enter your information below to get predictions.</p>
        
        <div class="form-container">
            <h2>Enter Your Information</h2>
            <form id="prediction-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="age">Age</label>
                        <input type="number" id="age" name="age" min="17" max="90" value="{{ example_input['age'] }}" required>
                    </div>
                    <div class="form-group">
                        <label for="education-num">Education (years)</label>
                        <input type="number" id="education-num" name="education-num" min="1" max="16" value="{{ example_input['education-num'] }}" required>
                    </div>
                    <div class="form-group">
                        <label for="hours-per-week">Hours per Week</label>
                        <input type="number" id="hours-per-week" name="hours-per-week" min="1" max="100" value="{{ example_input['hours-per-week'] }}" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="workclass">Work Class</label>
                        <select id="workclass" name="workclass" required>
                            {% for option in column_values['workclass'] %}
                            <option value="{{ option }}" {% if option == example_input['workclass'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="education">Education Level</label>
                        <select id="education" name="education" required>
                            {% for option in column_values['education'] %}
                            <option value="{{ option }}" {% if option == example_input['education'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="capital-gain">Capital Gain</label>
                        <input type="number" id="capital-gain" name="capital-gain" min="0" max="100000" value="{{ example_input['capital-gain'] }}" required>
                    </div>
                    <div class="form-group">
                        <label for="capital-loss">Capital Loss</label>
                        <input type="number" id="capital-loss" name="capital-loss" min="0" max="10000" value="{{ example_input['capital-loss'] }}" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="occupation">Occupation</label>
                        <select id="occupation" name="occupation" required>
                            {% for option in column_values['occupation'] %}
                            <option value="{{ option }}" {% if option == example_input['occupation'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="marital-status">Marital Status</label>
                        <select id="marital-status" name="marital-status" required>
                            {% for option in column_values['marital-status'] %}
                            <option value="{{ option }}" {% if option == example_input['marital-status'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="relationship">Relationship</label>
                        <select id="relationship" name="relationship" required>
                            {% for option in column_values['relationship'] %}
                            <option value="{{ option }}" {% if option == example_input['relationship'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="race">Race</label>
                        <select id="race" name="race" required>
                            {% for option in column_values['race'] %}
                            <option value="{{ option }}" {% if option == example_input['race'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="sex">Sex</label>
                        <select id="sex" name="sex" required>
                            {% for option in column_values['sex'] %}
                            <option value="{{ option }}" {% if option == example_input['sex'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="native-country">Native Country</label>
                        <select id="native-country" name="native-country" required>
                            {% for option in column_values['native-country'] %}
                            <option value="{{ option }}" {% if option == example_input['native-country'] %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                
                <input type="hidden" name="fnlwgt" value="{{ example_input['fnlwgt'] }}">
                
                <button type="submit">Predict Income</button>
            </form>
        </div>
        
        <div class="results" id="results-container">
            <h2>Prediction Results</h2>
            <div id="results-content"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const form = e.target;
            const formData = new FormData(form);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const resultsContainer = document.getElementById('results-container');
                const resultsContent = document.getElementById('results-content');
                resultsContent.innerHTML = '';
                
                // Display results for each model
                for (const [modelName, result] of Object.entries(data)) {
                    const modelDiv = document.createElement('div');
                    modelDiv.className = 'model-result';
                    
                    let modelTitle = '';
                    switch (modelName) {
                        case 'logistic_regression':
                            modelTitle = 'Logistic Regression (Baseline)';
                            break;
                        case 'random_forest':
                            modelTitle = 'Random Forest (Baseline)';
                            break;
                        case 'fair_demographic_parity':
                            modelTitle = 'Fair Model (Demographic Parity)';
                            break;
                        case 'fair_equalized_odds':
                            modelTitle = 'Fair Model (Equalized Odds)';
                            break;
                        default:
                            modelTitle = modelName;
                    }
                    
                    const titleElement = document.createElement('h3');
                    titleElement.textContent = modelTitle;
                    modelDiv.appendChild(titleElement);
                    
                    if (result.error) {
                        const errorElement = document.createElement('p');
                        errorElement.textContent = `Error: ${result.error}`;
                        errorElement.style.color = 'red';
                        modelDiv.appendChild(errorElement);
                    } else {
                        const predictionElement = document.createElement('div');
                        predictionElement.className = 'prediction';
                        
                        const incomeClass = result.prediction === '>50K' ? 'high-income' : 'low-income';
                        predictionElement.innerHTML = `Prediction: <span class="${incomeClass}">${result.prediction}</span>`;
                        modelDiv.appendChild(predictionElement);
                        
                        const probabilityElement = document.createElement('div');
                        probabilityElement.className = 'probability';
                        probabilityElement.textContent = `Probability of >50K income: ${(result.probability * 100).toFixed(2)}%`;
                        modelDiv.appendChild(probabilityElement);
                    }
                    
                    resultsContent.appendChild(modelDiv);
                }
                
                resultsContainer.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while making the prediction. Please try again.');
            });
        });
    </script>
</body>
</html>
        """)
    
    # Run the app
    app.run(host='0.0.0.0', port=8080, debug=True)
