import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set page configuration
st.set_page_config(
    page_title="Income Prediction with Fairness-Aware ML",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and preprocessor
@st.cache_resource
def load_models():
    models = {}
    try:
        models['logistic_regression'] = joblib.load('data/models/logistic_regression.joblib')
        models['random_forest'] = joblib.load('data/models/random_forest.joblib')
        models['fair_demographic_parity'] = joblib.load('data/models/fair_demographic_parity.joblib')
        models['fair_equalized_odds'] = joblib.load('data/models/fair_equalized_odds.joblib')
        return models
    except Exception as e:
        st.warning(f"Note: Model files not found. Running in demo mode with simulated predictions.")
        # Create dummy models for demo purposes
        from sklearn.dummy import DummyClassifier
        dummy_lr = DummyClassifier(strategy='stratified')
        dummy_lr.fit(np.array([[0, 0]]), np.array([0, 1]))
        
        models['logistic_regression'] = dummy_lr
        models['random_forest'] = dummy_lr
        models['fair_demographic_parity'] = dummy_lr
        models['fair_equalized_odds'] = dummy_lr
        return models

@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load('data/processed/preprocessor.joblib')
    except Exception as e:
        st.warning(f"Note: Preprocessor not found. Running in demo mode with simulated preprocessing.")
        # Create a dummy preprocessor for demo purposes
        from sklearn.preprocessing import FunctionTransformer
        dummy_preprocessor = FunctionTransformer()
        dummy_preprocessor.feature_names_in_ = ['age', 'workclass', 'fnlwgt', 'education', 
                                            'education-num', 'marital-status', 'occupation', 
                                            'relationship', 'race', 'sex', 'capital-gain', 
                                            'capital-loss', 'hours-per-week', 'native-country']
        return dummy_preprocessor

# Load feature names
@st.cache_data
def load_feature_names():
    try:
        with open('data/processed/feature_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        # Return default feature names if file not found
        return ['age', 'workclass', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week', 'native-country']

# Load data for exploration
@st.cache_data
def load_data():
    try:
        # Load raw data for exploration
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        train_data = pd.read_csv('data/adult.data', header=None, names=column_names, skipinitialspace=True)
        test_data = pd.read_csv('data/adult.test', header=None, names=column_names, skipinitialspace=True, skiprows=1)
        
        # Clean income column
        test_data['income'] = test_data['income'].str.replace('.', '')
        
        # Combine data for exploration
        data = pd.concat([train_data, test_data], axis=0)
        return data
    except Exception as e:
        st.warning("Using sample data for demo mode.")
        # Create sample data for demo mode
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        data = pd.DataFrame({
            'age': np.random.randint(18, 90, n_samples),
            'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov'], n_samples),
            'education': np.random.choice(['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'], n_samples),
            'education-num': np.random.randint(8, 16, n_samples),
            'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated'], n_samples),
            'occupation': np.random.choice(['Prof-specialty', 'Exec-managerial', 'Tech-support', 'Sales', 'Craft-repair'], n_samples),
            'relationship': np.random.choice(['Husband', 'Wife', 'Own-child', 'Not-in-family'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'capital-gain': np.random.choice([0, 0, 0, 0, *np.random.randint(1000, 20000, 50)], n_samples),
            'capital-loss': np.random.choice([0, 0, 0, 0, *np.random.randint(500, 4000, 50)], n_samples),
            'hours-per-week': np.random.randint(20, 80, n_samples),
            'native-country': np.random.choice(['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada'], n_samples),
            'income': np.random.choice(['>50K', '<=50K'], n_samples, p=[0.24, 0.76])  # Approximate class distribution
        })
        
        # Create correlations similar to the real dataset
        # Higher education and certain occupations correlate with higher income
        for i in range(n_samples):
            if data.loc[i, 'education-num'] >= 13 and data.loc[i, 'occupation'] in ['Prof-specialty', 'Exec-managerial']:
                data.loc[i, 'income'] = np.random.choice(['>50K', '<=50K'], 1, p=[0.6, 0.4])[0]
            
            # Gender bias in the original dataset
            if data.loc[i, 'sex'] == 'Male':
                data.loc[i, 'income'] = np.random.choice(['>50K', '<=50K'], 1, p=[0.3, 0.7])[0]
            else:
                data.loc[i, 'income'] = np.random.choice(['>50K', '<=50K'], 1, p=[0.1, 0.9])[0]
        
        return data

# Load performance metrics
@st.cache_data
def load_performance_metrics():
    try:
        performance_df = pd.read_csv('data/models/model_performance.csv', index_col=0)
        return performance_df
    except Exception as e:
        # Create dummy performance metrics for demo mode
        models_list = ['logistic_regression', 'random_forest', 'fair_demographic_parity', 'fair_equalized_odds']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        data = {
            'accuracy': [0.85, 0.87, 0.83, 0.84],
            'precision': [0.75, 0.78, 0.72, 0.73],
            'recall': [0.65, 0.68, 0.70, 0.71],
            'f1': [0.70, 0.73, 0.71, 0.72],
            'roc_auc': [0.82, 0.85, 0.80, 0.81]
        }
        return pd.DataFrame(data, index=models_list)

# Load fairness metrics
@st.cache_data
def load_fairness_metrics():
    try:
        fairness_df = pd.read_csv('data/models/fairness_metrics.csv', index_col=0)
        return fairness_df
    except Exception as e:
        # Create dummy fairness metrics for demo mode
        models_list = ['logistic_regression', 'random_forest', 'fair_demographic_parity', 'fair_equalized_odds']
        data = {
            'demographic_parity_difference': [0.18, 0.20, 0.05, 0.08],
            'equalized_odds_difference': [0.15, 0.17, 0.12, 0.04]
        }
        return pd.DataFrame(data, index=models_list)

# Load all resources
models = load_models()
preprocessor = load_preprocessor()
feature_names = load_feature_names()
performance_df = load_performance_metrics()
fairness_df = load_fairness_metrics()
data = load_data()

# Check if running in demo mode
demo_mode = False
try:
    # Try to access one of the model files to determine if we're in demo mode
    with open('data/models/logistic_regression.joblib', 'rb') as f:
        pass  # File exists, not in demo mode
except FileNotFoundError:
    demo_mode = True

# Define sidebar navigation
st.sidebar.title("Navigation")
if demo_mode:
    st.sidebar.warning("⚠️ Running in demo mode with simulated data and predictions")
    
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "About"])

# Home page with prediction form
if page == "Home":
    st.title("Income Prediction with Fairness-Aware Machine Learning")
    st.write("""
    This application demonstrates income prediction using both traditional and fairness-aware 
    machine learning models. Enter your information below to get a prediction.
    """)
    
    # Create columns for form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=17, max_value=90, value=30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        education = st.selectbox("Education", [
            "Bachelors", "HS-grad", "11th", "Masters", "9th", 
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", 
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", 
            "Preschool", "12th"
        ])
        education_num = st.number_input("Education (years)", min_value=1, max_value=16, value=13)
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated", 
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
    
    with col2:
        st.subheader("Employment Information")
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales", 
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
            "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
        ])
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=0)
        relationship = st.selectbox("Relationship", [
            "Wife", "Own-child", "Husband", "Not-in-family", 
            "Other-relative", "Unmarried"
        ])
        native_country = st.selectbox("Native Country", [
            "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", 
            "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", 
            "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", 
            "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", 
            "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", 
            "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", 
            "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
        ])
    
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
    if st.button("Predict Income"):
        if models and preprocessor:
            # Create a dataframe with the input values
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [1],  # Not used in prediction
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
            
            # Preprocess the input data
            input_processed = preprocessor.transform(input_data)
            if hasattr(input_processed, 'toarray'):
                input_processed = input_processed.toarray()
            
            # Get the selected model
            selected_model_key = model_mapping[model_choice]
            selected_model = models[selected_model_key]
            
            # Make prediction
            prediction = selected_model.predict(input_processed)[0]
            
            # Try to get probability if available
            probability = None
            if hasattr(selected_model, 'predict_proba'):
                try:
                    probability = selected_model.predict_proba(input_processed)[0, 1]
                except (AttributeError, IndexError):
                    pass
            
            # Display result
            st.subheader("Prediction Result")
            result_container = st.container(border=True)
            with result_container:
                if prediction == 1:
                    st.success(f"Prediction: Income > $50K")
                    if probability is not None:
                        st.progress(float(probability), text=f"Confidence: {probability:.2%}")
                    st.balloons()
                else:
                    st.info(f"Prediction: Income <= $50K")
                    if probability is not None:
                        st.progress(float(1-probability), text=f"Confidence: {1-probability:.2%}")
                
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
                        st.metric("Demographic Parity Diff", f"{fairness['demographic_parity_difference']:.4f}", 
                                 delta=-fairness['demographic_parity_difference'], delta_color="inverse")
                    with col2:
                        st.metric("Equalized Odds Diff", f"{fairness['equalized_odds_difference']:.4f}", 
                                 delta=-fairness['equalized_odds_difference'], delta_color="inverse")
                    
                    st.info("Lower values for fairness metrics indicate less bias in the model.")
                
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
                    'Gender': 0.7 if sex == 'Male' else 0.3,  # Simplified representation
                    'Race': {'White': 0.2, 'Black': 0.3, 'Asian-Pac-Islander': 0.4, 
                             'Amer-Indian-Eskimo': 0.5, 'Other': 0.6}.get(race, 0.5),
                    'Marital Status': 0.8 if 'Married' in marital_status else 0.4,
                    'Occupation': {'Exec-managerial': 0.9, 'Prof-specialty': 0.85, 
                                  'Tech-support': 0.7}.get(occupation, 0.5)
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
                if sex == 'Male':
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
                if sex == 'Male':
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
                    if education_num < 13:
                        st.write("With a college degree, prediction likelihood for >50K would increase significantly.")
                    else:
                        st.write("You already have higher education, which positively impacts the prediction.")
                
                with col2:
                    st.write("**If hours worked changed:**")
                    if hours_per_week < 40:
                        st.write("Working full-time (40+ hours) would increase likelihood of >50K income.")
                    elif hours_per_week > 60:
                        st.write("You're working many hours, which strongly influences the prediction.")
                    else:
                        st.write("You're working standard full-time hours.")
                        
                # Add model comparison
                if prediction == 1:
                    other_models = [k for k, v in model_mapping.items() if v != selected_model_key]
                    st.write(f"**Note:** {random.choice(other_models)} might predict differently based on how it weighs these factors.")
                    
                # Add disclaimer
                st.caption("Note: These visualizations are simplified representations for educational purposes.")
        else:
            st.error("Models or preprocessor not loaded correctly. Please check the data files.")

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
    
    st.write("""
    ## Income Prediction with Fairness-Aware Machine Learning
    
    This project demonstrates the application of fairness-aware machine learning techniques to the Adult Income Dataset.
    The goal is to predict whether an individual's income exceeds $50K per year based on census data, while ensuring
    fairness across different demographic groups.
    
    ### Fairness in Machine Learning
    
    Machine learning models can inadvertently perpetuate or amplify biases present in training data. Fairness-aware
    machine learning aims to mitigate these biases by constraining models to satisfy fairness criteria.
    
    #### Key Fairness Metrics:
    
    1. **Demographic Parity**: A model satisfies demographic parity if predictions are independent of the protected
       attribute (e.g., gender, race). In other words, the proportion of positive predictions should be the same
       across all demographic groups.
       
    2. **Equalized Odds**: A model satisfies equalized odds if predictions are conditionally independent of the
       protected attribute given the true outcome. This means the true positive and false positive rates should
       be the same across all demographic groups.
    
    ### Models Implemented:
    
    1. **Baseline Models**:
       - Logistic Regression
       - Random Forest
       
    2. **Fairness-Aware Models**:
       - Demographic Parity Constrained Model
       - Equalized Odds Constrained Model
    
    ### Dataset:
    
    The Adult Income Dataset contains demographic and employment information from the 1994 Census database.
    It includes attributes such as age, education, occupation, gender, race, and others, with the target
    variable indicating whether an individual's income is above or below $50K per year.
    
    ### Technologies Used:
    
    - Python
    - Scikit-learn
    - Fairlearn
    - Streamlit
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    """)
    
    st.subheader("References")
    st.markdown("""
    - [Fairlearn Documentation](https://fairlearn.org/)
    - [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)

# Add a footer
st.markdown("""
---
Created with ❤️ using Streamlit | [GitHub Repository](https://github.com/yourusername/income-prediction-fairness)
""")
