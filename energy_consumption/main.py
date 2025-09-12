"""
Main script for Energy Load Forecasting with Facebook Prophet.
Includes baseline model, hyperparameter tuning, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import custom modules
from data.data_loader import DataLoader
from data.preprocessor import ProphetPreprocessor
from models.prophet_model import EnergyProphetModel
from evaluation.metrics import ForecastMetrics, CrossValidationMetrics
from visualization.plots import EnergyForecastVisualizer

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_results_directories():
    """Create necessary directories for saving results."""
    directories = [
        'results/figures',
        'results/models',
        'results/reports',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_data_loading_and_preprocessing(config: dict):
    """
    Execute data loading and preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print("="*60)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    # Load and merge datasets
    loader = DataLoader(config)
    merged_data = loader.load_and_merge_data()
    
    # Save raw merged data
    merged_data.to_csv('data/processed/merged_raw_data.csv')
    print(f"Saved merged raw data: {merged_data.shape}")
    
    # Preprocess for Prophet
    preprocessor = ProphetPreprocessor(config)
    train_df, test_df = preprocessor.prepare_data_for_prophet(merged_data)
    
    # Save processed data
    train_df.to_csv('data/processed/prophet_train_data.csv', index=False)
    test_df.to_csv('data/processed/prophet_test_data.csv', index=False)
    
    print(f"Training data saved: {train_df.shape}")
    print(f"Test data saved: {test_df.shape}")
    
    return train_df, test_df, merged_data

def run_exploratory_data_analysis(merged_data: pd.DataFrame, config: dict):
    """
    Generate exploratory data analysis visualizations.
    
    Args:
        merged_data: Merged raw dataset
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    visualizer = EnergyForecastVisualizer()
    
    # Time series overview
    print("Generating time series overview...")
    fig1 = visualizer.plot_time_series_overview(
        merged_data, 
        target_col=config['target']['column'],
        save_path='results/figures/01_time_series_overview.png'
    )
    
    # Seasonal decomposition
    print("Generating seasonal decomposition...")
    fig2 = visualizer.plot_seasonal_decomposition(
        merged_data,
        target_col=config['target']['column'],
        save_path='results/figures/02_seasonal_decomposition.png'
    )
    
    # Weather correlations
    print("Generating weather correlation analysis...")
    fig3 = visualizer.plot_weather_correlations(
        merged_data,
        target_col=config['target']['column'],
        weather_cols=config['weather_features'],
        save_path='results/figures/03_weather_correlations.png'
    )
    
    print("EDA visualizations saved to results/figures/")

def run_baseline_prophet_model(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                              config: dict):
    """
    Train and evaluate baseline Prophet model.
    
    Args:
        train_df: Training data
        test_df: Test data
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, forecast, metrics)
    """
    print("\n" + "="*60)
    print("STEP 3: BASELINE PROPHET MODEL")
    print("="*60)
    
    # Initialize model
    prophet_model = EnergyProphetModel(config)
    
    # Fit baseline model
    model = prophet_model.fit_baseline_model(train_df)
    
    # Generate predictions
    forecast = prophet_model.predict(test_df)
    
    # Calculate metrics
    metrics_calc = ForecastMetrics()
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    baseline_metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
    metrics_calc.print_metrics_summary(baseline_metrics, "Baseline Prophet Model")
    
    # Generate visualizations
    visualizer = EnergyForecastVisualizer()
    
    # Prophet forecast plot
    print("Generating forecast visualization...")
    fig1 = visualizer.plot_prophet_forecast(
        model, forecast, test_df,
        save_path='results/figures/04_baseline_prophet_forecast.png'
    )
    
    # Components plot
    print("Generating components visualization...")
    fig2 = visualizer.plot_prophet_components(
        model, forecast,
        save_path='results/figures/05_baseline_prophet_components.png'
    )
    
    # Forecast vs actual
    print("Generating forecast vs actual visualization...")
    fig3 = visualizer.plot_forecast_vs_actual(
        y_true, y_pred, 
        dates=test_df['ds'],
        confidence_intervals=(forecast['yhat_lower'].values, forecast['yhat_upper'].values),
        title="Baseline Prophet: Forecast vs Actual",
        save_path='results/figures/06_baseline_forecast_vs_actual.png'
    )
    
    # Residuals analysis
    print("Generating residuals analysis...")
    fig4 = visualizer.plot_residuals_analysis(
        y_true, y_pred,
        dates=test_df['ds'],
        save_path='results/figures/07_baseline_residuals_analysis.png'
    )
    
    # Feature importance
    importance_df = prophet_model.get_feature_importance()
    if not importance_df.empty:
        print("Generating feature importance plot...")
        fig5 = visualizer.plot_feature_importance(
            importance_df,
            save_path='results/figures/08_baseline_feature_importance.png'
        )
    
    # Save model
    prophet_model.save_model('results/models/baseline_prophet_model.pkl')
    
    return prophet_model, forecast, baseline_metrics

def run_hyperparameter_tuning(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             config: dict):
    """
    Perform hyperparameter tuning and train optimized model.
    
    Args:
        train_df: Training data
        test_df: Test data
        config: Configuration dictionary
        
    Returns:
        Tuple of (tuned_model, tuned_forecast, tuned_metrics)
    """
    print("\n" + "="*60)
    print("STEP 4: HYPERPARAMETER TUNING")
    print("="*60)
    
    # Initialize model for tuning
    prophet_model = EnergyProphetModel(config)
    
    # Perform hyperparameter tuning
    tuning_results = prophet_model.hyperparameter_tuning(train_df)
    
    # Fit model with best parameters
    tuned_model = prophet_model.fit_tuned_model(train_df)
    
    # Generate predictions
    tuned_forecast = prophet_model.predict(test_df)
    
    # Calculate metrics
    metrics_calc = ForecastMetrics()
    y_true = test_df['y'].values
    y_pred = tuned_forecast['yhat'].values
    
    tuned_metrics = metrics_calc.calculate_all_metrics(y_true, y