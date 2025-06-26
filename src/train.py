import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

def load_processed_data():
    """
    Load the preprocessed data.
    """
    X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
    X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
    y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
    y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded processed data: X_train shape={X_train.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

def train_baseline_models(X_train, y_train):
    """
    Train baseline machine learning models.
    """
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data.
    """
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

def train_fairness_aware_model(X_train, y_train, sensitive_feature_idx, constraint_name='demographic_parity'):
    """
    Train a fairness-aware model using Fairlearn's exponentiated gradient reduction.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - sensitive_feature_idx: Index of the sensitive feature in X_train
    - constraint_name: Type of fairness constraint ('demographic_parity' or 'equalized_odds')
    
    Returns:
    - Trained fairness-aware model
    """
    # Extract sensitive feature (assuming it's binary)
    sensitive_features = X_train[:, sensitive_feature_idx].reshape(-1, 1)
    
    # Base classifier
    estimator = LogisticRegression(max_iter=1000)
    
    # Choose constraint
    if constraint_name == 'demographic_parity':
        constraint = DemographicParity()
    else:  # equalized_odds
        constraint = EqualizedOdds()
    
    # Train with fairness constraint
    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint,
        eps=0.01
    )
    
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    
    return mitigator

def evaluate_fairness(models, X_test, y_test, sensitive_feature_idx):
    """
    Evaluate fairness metrics for the models.
    """
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
    """
    # Create output directory if it doesn't exist
    os.makedirs('data/models', exist_ok=True)
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, f'data/models/{name}.joblib')
    
    # Save results as CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('data/models/model_performance.csv')
    
    # Save fairness results
    fairness_df = pd.DataFrame({
        name: {
            'demographic_parity_difference': metrics['demographic_parity_difference'],
            'equalized_odds_difference': metrics['equalized_odds_difference']
        }
        for name, metrics in fairness_results.items()
    }).T
    fairness_df.to_csv('data/models/fairness_metrics.csv')
    
    print("Saved models and evaluation results to data/models/")

def plot_results(results, fairness_results):
    """
    Create visualizations of model performance and fairness metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Plot performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results_df = pd.DataFrame(results).T
    
    # Add ROC AUC if available for all models
    if all('roc_auc' in model_results and model_results['roc_auc'] is not None for model_results in results.values()):
        metrics.append('roc_auc')
    
    plt.figure(figsize=(12, 6))
    results_df[metrics].plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/images/model_performance.png')
    
    # Plot fairness metrics
    fairness_df = pd.DataFrame({
        name: {
            'Demographic Parity Difference': metrics['demographic_parity_difference'],
            'Equalized Odds Difference': metrics['equalized_odds_difference']
        }
        for name, metrics in fairness_results.items()
    }).T
    
    plt.figure(figsize=(10, 6))
    fairness_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Fairness Metrics Comparison')
    plt.ylabel('Difference (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/images/fairness_metrics.png')
    
    print("Saved visualization plots to static/images/")

if __name__ == "__main__":
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    print("\nTraining baseline models...")
    baseline_models = train_baseline_models(X_train, y_train)
    
    print("\nEvaluating baseline models...")
    baseline_results = evaluate_models(baseline_models, X_test, y_test)
    
    # Assuming 'sex' feature is at index 9 (based on the Adult dataset)
    # This would need to be adjusted based on the actual preprocessed data
    sensitive_feature_idx = 9
    
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
    
    print("\nCreating visualizations...")
    plot_results(all_results, all_fairness)
    
    print("\nModel training and evaluation complete!")
