"""
Visualization module for the Income Prediction project.
Creates interactive plots for model performance and fairness metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'images')

def create_performance_plot(results_df):
    """
    Create an interactive performance comparison plot.
    
    Args:
        results_df: DataFrame with model performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    # Select metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'roc_auc' in results_df.columns and not results_df['roc_auc'].isna().any():
        metrics.append('roc_auc')
    
    # Reshape data for plotting
    plot_data = results_df[metrics].reset_index()
    plot_data = pd.melt(plot_data, id_vars=['index'], value_vars=metrics, 
                        var_name='Metric', value_name='Score')
    plot_data.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create plot
    fig = px.bar(
        plot_data, 
        x='Model', 
        y='Score', 
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Score': 'Score (higher is better)'},
        height=500
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        legend_title='Metric',
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid lines
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    
    return fig

def create_fairness_plot(fairness_df):
    """
    Create an interactive fairness metrics comparison plot.
    
    Args:
        fairness_df: DataFrame with fairness metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    # Reshape data for plotting
    plot_data = fairness_df.reset_index()
    plot_data = pd.melt(plot_data, id_vars=['index'], 
                        value_vars=['demographic_parity_difference', 'equalized_odds_difference'],
                        var_name='Metric', value_name='Difference')
    plot_data.rename(columns={'index': 'Model'}, inplace=True)
    
    # Create plot
    fig = px.bar(
        plot_data, 
        x='Model', 
        y='Difference', 
        color='Metric',
        barmode='group',
        title='Fairness Metrics Comparison',
        labels={'Difference': 'Difference (lower is better)'},
        height=500
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Difference',
        legend_title='Metric',
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid lines
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """
    Create a feature importance plot for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    top_n = 15  # Show top 15 features
    
    # Create plot data
    plot_data = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices[:top_n]],
        'Importance': importances[indices[:top_n]]
    })
    
    # Create plot
    fig = px.bar(
        plot_data, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance',
        height=500
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),  # Highest importance at the top
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid lines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    
    return fig

def create_prediction_distribution_plot(models, X, feature_names, sensitive_feature_idx):
    """
    Create a plot showing prediction distributions across sensitive groups.
    
    Args:
        models: Dictionary of trained models
        X: Feature matrix
        feature_names: List of feature names
        sensitive_feature_idx: Index of the sensitive feature
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    # Extract sensitive feature values
    sensitive_values = X[:, sensitive_feature_idx]
    
    # Get unique values of sensitive feature
    unique_values = np.unique(sensitive_values)
    
    # Create subplots
    fig = make_subplots(
        rows=len(models), 
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(models.keys()),
        vertical_spacing=0.1
    )
    
    # Colors for different groups
    colors = ['#636EFA', '#EF553B']
    
    row = 1
    for name, model in models.items():
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.predict(X).astype(float)
        
        # Plot distributions for each group
        for i, value in enumerate(unique_values):
            mask = (sensitive_values == value)
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=y_prob[mask],
                    name=f'Group {value}' if row == 1 else None,  # Only add to legend once
                    opacity=0.7,
                    marker_color=colors[i % len(colors)],
                    showlegend=(row == 1)  # Only show in legend for first row
                ),
                row=row, 
                col=1
            )
        
        row += 1
    
    # Update layout
    fig.update_layout(
        title='Prediction Distribution by Sensitive Group',
        barmode='overlay',
        height=200 * len(models),
        legend_title='Sensitive Group',
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Update x and y axes
    fig.update_xaxes(
        title_text='Prediction Score',
        range=[0, 1],
        row=len(models), 
        col=1
    )
    
    fig.update_yaxes(
        title_text='Count',
        row=len(models) // 2 + 1, 
        col=1
    )
    
    return fig

def save_static_plots(results_df, fairness_df):
    """
    Save static plots for the web application.
    
    Args:
        results_df: DataFrame with model performance metrics
        fairness_df: DataFrame with fairness metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # Plot performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'roc_auc' in results_df.columns and not results_df['roc_auc'].isna().any():
        metrics.append('roc_auc')
    
    plt.figure(figsize=(12, 6))
    results_df[metrics].plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'model_performance.png'))
    
    # Plot fairness metrics
    plt.figure(figsize=(10, 6))
    fairness_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Fairness Metrics Comparison')
    plt.ylabel('Difference (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'fairness_metrics.png'))
    
    print(f"Saved visualization plots to {STATIC_DIR}")

def main():
    """
    Main function to create and save visualizations.
    """
    # Load results
    try:
        results_df = pd.read_csv(os.path.join(MODELS_DIR, 'model_performance.csv'), index_col=0)
        fairness_df = pd.read_csv(os.path.join(MODELS_DIR, 'fairness_metrics.csv'), index_col=0)
        
        # Save static plots
        save_static_plots(results_df, fairness_df)
        
        print("Visualization creation complete!")
    except FileNotFoundError:
        print("Error: Model performance or fairness metrics files not found.")
        print("Please run model training first.")

if __name__ == "__main__":
    main()
