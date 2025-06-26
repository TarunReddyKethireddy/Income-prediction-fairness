import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
def load_models_and_preprocessor():
    models = {}
    try:
        models['logistic_regression'] = joblib.load('data/models/logistic_regression.joblib')
        models['random_forest'] = joblib.load('data/models/random_forest.joblib')
        models['fair_demographic_parity'] = joblib.load('data/models/fair_demographic_parity.joblib')
        models['fair_equalized_odds'] = joblib.load('data/models/fair_equalized_odds.joblib')
        preprocessor = joblib.load('data/processed/preprocessor.joblib')
        
        # Load feature names
        with open('data/processed/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load performance metrics
        performance_df = pd.read_csv('data/models/model_performance.csv', index_col=0)
        fairness_df = pd.read_csv('data/models/fairness_metrics.csv', index_col=0)
        
        return models, preprocessor, feature_names, performance_df, fairness_df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

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
        st.error(f"Error loading data: {e}")
        return None

# Load models and data
models, preprocessor, feature_names, performance_df, fairness_df = load_models_and_preprocessor()
data = load_data()

# Define sidebar navigation
st.sidebar.title("Navigation")
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
            
            # Display result
            st.subheader("Prediction Result")
            result_container = st.container(border=True)
            with result_container:
                if prediction == 1:
                    st.success("Prediction: Income > $50K")
                    st.balloons()
                else:
                    st.info("Prediction: Income <= $50K")
                
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
