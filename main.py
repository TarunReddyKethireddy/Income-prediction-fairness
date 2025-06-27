"""
Main script to run the entire income prediction pipeline.
This script orchestrates data preprocessing, model training, evaluation, and visualization.
"""

import os
import argparse
from src.data.preprocessing import load_data, preprocess_data, save_processed_data
from src.models.training import train_baseline_models, train_fairness_aware_model, evaluate_models
from src.models.training import evaluate_fairness, save_models, get_sensitive_feature_idx
from src.visualization.plots import create_performance_plot, create_fairness_plot, save_static_plots

def main(args):
    """
    Main function to run the entire pipeline.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print("Income Prediction with Fairness-Aware Machine Learning")
    print("="*80 + "\n")
    
    # Step 1: Data preprocessing
    print("\n[Step 1] Loading and preprocessing data...")
    train_data, test_data = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(train_data, test_data)
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, feature_names)
    
    # Step 2: Train baseline models
    print("\n[Step 2] Training baseline models...")
    baseline_models = train_baseline_models(X_train, y_train, args.hyperparameter_tuning)
    
    # Step 3: Evaluate baseline models
    print("\n[Step 3] Evaluating baseline models...")
    baseline_results = evaluate_models(baseline_models, X_test, y_test)
    
    # Get sensitive feature index
    sensitive_feature_idx = get_sensitive_feature_idx(feature_names, 'sex_Female')
    print(f"\nUsing 'sex_Female' as sensitive feature at index {sensitive_feature_idx}")
    
    # Step 4: Evaluate fairness of baseline models
    print("\n[Step 4] Evaluating fairness of baseline models...")
    baseline_fairness = evaluate_fairness(baseline_models, X_test, y_test, sensitive_feature_idx)
    
    # Step 5: Train fairness-aware models
    print("\n[Step 5] Training fairness-aware models...")
    fairness_models = {
        'fair_demographic_parity': train_fairness_aware_model(
            X_train, y_train, sensitive_feature_idx, 'demographic_parity'
        ),
        'fair_equalized_odds': train_fairness_aware_model(
            X_train, y_train, sensitive_feature_idx, 'equalized_odds'
        )
    }
    
    # Step 6: Evaluate fairness-aware models
    print("\n[Step 6] Evaluating fairness-aware models...")
    fairness_results = evaluate_models(fairness_models, X_test, y_test)
    
    # Step 7: Evaluate fairness of fairness-aware models
    print("\n[Step 7] Evaluating fairness of fairness-aware models...")
    fairness_fairness = evaluate_fairness(fairness_models, X_test, y_test, sensitive_feature_idx)
    
    # Combine all models and results
    all_models = {**baseline_models, **fairness_models}
    all_results = {**baseline_results, **fairness_results}
    all_fairness = {**baseline_fairness, **fairness_fairness}
    
    # Step 8: Save models and results
    print("\n[Step 8] Saving models and results...")
    save_models(all_models, all_results, all_fairness)
    
    # Step 9: Create and save visualizations
    print("\n[Step 9] Creating and saving visualizations...")
    import pandas as pd
    results_df = pd.DataFrame(all_results).T
    fairness_df = pd.DataFrame({
        name: {
            'demographic_parity_difference': metrics['demographic_parity_difference'],
            'equalized_odds_difference': metrics['equalized_odds_difference']
        }
        for name, metrics in all_fairness.items()
    }).T
    
    # Save static plots
    save_static_plots(results_df, fairness_df)
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Package models for deployment: python package_models.py")
    print("3. Deploy to Streamlit Cloud")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Income Prediction Pipeline")
    parser.add_argument("--hyperparameter-tuning", action="store_true", 
                        help="Perform hyperparameter tuning for models")
    args = parser.parse_args()
    
    main(args)
