"""
Model training module for the Income Prediction project.
Handles training, evaluation, and fairness metrics for machine learning models.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def train_baseline_models(X_train, y_train, hyperparameter_tuning=True):
    """
    Train baseline machine learning models with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        dict: Trained models
    """
    print("Training baseline models...")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    trained_models = {}
    
    # Logistic Regression
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning for Logistic Regression...")
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        lr_model = grid_search.best_estimator_
        print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    else:
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
    
    trained_models['logistic_regression'] = lr_model
    
    # Random Forest
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning for Random Forest...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        rf_model = grid_search.best_estimator_
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")
    else:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
    
    trained_models['random_forest'] = rf_model
    
    return trained_models

def train_fairness_aware_model(X_train, y_train, sensitive_feature_idx, constraint_name='demographic_parity'):
    """
    Train a fairness-aware model using Fairlearn's exponentiated gradient reduction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_feature_idx: Index of the sensitive feature in X_train
        constraint_name: Type of fairness constraint ('demographic_parity' or 'equalized_odds')
        
    Returns:
        object: Trained fairness-aware model
    """
    print(f"Training fairness-aware model with {constraint_name} constraint...")
    
    # Extract sensitive feature (assuming it's binary)
    sensitive_features = X_train[:, sensitive_feature_idx].reshape(-1, 1)
    
    # Base classifier
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    # Choose constraint
    if constraint_name == 'demographic_parity':
        constraint = DemographicParity()
    else:  # equalized_odds
        constraint = EqualizedOdds()
    
    # Train with fairness constraint
    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint,
        eps=0.01,
        max_iter=50,
        nu=1e-6
    )
    
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    
    return mitigator

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics for each model
    """
    print("Evaluating models...")
    
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        # Calculate ROC AUC if predict_proba is available
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            except (AttributeError, IndexError):
                metrics['roc_auc'] = None
        else:
            # For fairness-aware models that might not have predict_proba
            metrics['roc_auc'] = None
        
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            print(f"  ROC AUC: Not available")
    
    return results

def evaluate_fairness(models, X_test, y_test, sensitive_feature_idx):
    """
    Evaluate fairness metrics for the models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        sensitive_feature_idx: Index of the sensitive feature in X_test
        
    Returns:
        dict: Fairness metrics for each model
    """
    print("Evaluating fairness metrics...")
    
    # Extract sensitive feature
    sensitive_features = X_test[:, sensitive_feature_idx].reshape(-1, 1)
    
    fairness_results = {}
    
    for name, model in models.items():
        print(f"Evaluating fairness for {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate fairness metrics
        dp_diff = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        eo_diff = equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        fairness_results[name] = {
            'demographic_parity_difference': dp_diff,
            'equalized_odds_difference': eo_diff
        }
        
        print(f"  Demographic Parity Difference: {dp_diff:.4f}")
        print(f"  Equalized Odds Difference: {eo_diff:.4f}")
    
    return fairness_results

def save_models(models, results, fairness_results):
    """
    Save trained models and evaluation results.
    
    Args:
        models: Dictionary of trained models
        results: Dictionary of evaluation metrics
        fairness_results: Dictionary of fairness metrics
    """
    print("Saving models and results...")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    
    # Save results as CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_performance.csv'))
    
    # Save fairness results
    fairness_df = pd.DataFrame({
        name: {
            'demographic_parity_difference': metrics['demographic_parity_difference'],
            'equalized_odds_difference': metrics['equalized_odds_difference']
        }
        for name, metrics in fairness_results.items()
    }).T
    fairness_df.to_csv(os.path.join(MODELS_DIR, 'fairness_metrics.csv'))
    
    print(f"Saved models and evaluation results to {MODELS_DIR}")

def main(hyperparameter_tuning=False):
    """
    Main function to train and evaluate models.
    
    Args:
        hyperparameter_tuning: Whether to perform hyperparameter tuning
    """
    # Import here to avoid circular imports
    from src.data.preprocessing import load_processed_data, get_sensitive_feature_idx
    
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    print("\nTraining baseline models...")
    baseline_models = train_baseline_models(X_train, y_train, hyperparameter_tuning)
    
    print("\nEvaluating baseline models...")
    baseline_results = evaluate_models(baseline_models, X_test, y_test)
    
    # Get sensitive feature index
    sensitive_feature_idx = get_sensitive_feature_idx(feature_names, 'sex_Female')
    
    print("\nEvaluating fairness of baseline models...")
    baseline_fairness = evaluate_fairness(baseline_models, X_test, y_test, sensitive_feature_idx)
    
    print("\nTraining fairness-aware models...")
    fairness_models = {
        'fair_demographic_parity': train_fairness_aware_model(
            X_train, y_train, sensitive_feature_idx, 'demographic_parity'
        ),
        'fair_equalized_odds': train_fairness_aware_model(
            X_train, y_train, sensitive_feature_idx, 'equalized_odds'
        )
    }
    
    print("\nEvaluating fairness-aware models...")
    fairness_results = evaluate_models(fairness_models, X_test, y_test)
    
    print("\nEvaluating fairness of fairness-aware models...")
    fairness_fairness = evaluate_fairness(fairness_models, X_test, y_test, sensitive_feature_idx)
    
    # Combine all models and results
    all_models = {**baseline_models, **fairness_models}
    all_results = {**baseline_results, **fairness_results}
    all_fairness = {**baseline_fairness, **fairness_fairness}
    
    print("\nSaving models and results...")
    save_models(all_models, all_results, all_fairness)
    
    print("\nModel training and evaluation complete!")
    
    return all_models, all_results, all_fairness

if __name__ == "__main__":
    main(hyperparameter_tuning=False)
