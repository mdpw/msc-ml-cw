"""
Facebook Prophet model implementation for energy load forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from sklearn.model_selection import ParameterGrid
import pickle
import os

# Configure logging
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')


class EnergyProphetModel:
    """
    Facebook Prophet model wrapper for energy load forecasting.
    Includes baseline model, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.forecast = None
        self.cv_results = None
        self.best_params = None
        self.is_fitted = False
        
    def create_baseline_model(self) -> Prophet:
        """
        Create baseline Prophet model with default parameters.
        
        Returns:
            Prophet: Configured baseline model
        """
        print("Creating baseline Prophet model...")
        
        baseline_config = self.config['prophet']['baseline']
        
        model = Prophet(
            growth=baseline_config['growth'],
            seasonality_mode=baseline_config['seasonality_mode'],
            yearly_seasonality=baseline_config['yearly_seasonality'],
            weekly_seasonality=baseline_config['weekly_seasonality'],
            daily_seasonality=baseline_config['daily_seasonality'],
            interval_width=0.95,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0
        )
        
        return model
    
    def add_regressors(self, model: Prophet, train_df: pd.DataFrame) -> Prophet:
        """
        Add external regressors to the Prophet model.
        
        Args:
            model: Prophet model instance
            train_df: Training dataframe
            
        Returns:
            Prophet: Model with added regressors
        """
        print("Adding external regressors...")
        
        # Weather regressors
        weather_features = self.config['weather_features']
        for feature in weather_features:
            if feature in train_df.columns:
                model.add_regressor(feature, standardize=True)
                print(f"  Added regressor: {feature}")
        
        # Temporal regressors
        temporal_features = ['is_weekend', 'is_business_hour', 'heating_demand', 'cooling_demand']
        for feature in temporal_features:
            if feature in train_df.columns:
                model.add_regressor(feature, standardize=True)
                print(f"  Added regressor: {feature}")
        
        return model
    
    def add_custom_seasonalities(self, model: Prophet) -> Prophet:
        """
        Add custom seasonality patterns to the model.
        
        Args:
            model: Prophet model instance
            
        Returns:
            Prophet: Model with custom seasonalities
        """
        print("Adding custom seasonalities...")
        
        custom_seasonalities = self.config['prophet'].get('custom_seasonalities', [])
        
        for seasonality in custom_seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order']
            )
            print(f"  Added seasonality: {seasonality['name']} (period={seasonality['period']})")
        
        return model
    
    def fit_baseline_model(self, train_df: pd.DataFrame) -> Prophet:
        """
        Fit the baseline Prophet model.
        
        Args:
            train_df: Training data
            
        Returns:
            Prophet: Fitted model
        """
        print("Fitting baseline Prophet model...")
        
        # Create model
        model = self.create_baseline_model()
        
        # Add regressors and seasonalities
        model = self.add_regressors(model, train_df)
        model = self.add_custom_seasonalities(model)
        
        # Fit model
        model.fit(train_df)
        
        self.model = model
        self.is_fitted = True
        
        print("Baseline model fitted successfully!")
        return model
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted model.
        
        Args:
            test_df: Test data
            
        Returns:
            pd.DataFrame: Predictions with confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("Generating predictions...")
        
        # Create future dataframe
        future_df = test_df[['ds'] + [col for col in test_df.columns if col != 'y']].copy()
        
        # Generate forecast
        forecast = self.model.predict(future_df)
        
        self.forecast = forecast
        return forecast
    
    def hyperparameter_tuning(self, train_df: pd.DataFrame, 
                            cv_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            train_df: Training data
            cv_params: Cross-validation parameters
            
        Returns:
            Dict: Best parameters and performance
        """
        print("Starting hyperparameter tuning...")
        
        if cv_params is None:
            cv_params = self.config['cross_validation']
        
        # Parameter grid
        param_grid = self.config['prophet']['tuning']
        
        # Convert to sklearn ParameterGrid format
        grid_params = []
        for changepoint in param_grid['changepoint_prior_scale']:
            for seasonality in param_grid['seasonality_prior_scale']:
                for holidays in param_grid['holidays_prior_scale']:
                    for mode in param_grid['seasonality_mode']:
                        grid_params.append({
                            'changepoint_prior_scale': changepoint,
                            'seasonality_prior_scale': seasonality,
                            'holidays_prior_scale': holidays,
                            'seasonality_mode': mode
                        })
        
        print(f"Testing {len(grid_params)} parameter combinations...")
        
        best_score = float('inf')
        best_params = None
        best_cv_results = None
        
        for i, params in enumerate(grid_params):
            print(f"  Testing combination {i+1}/{len(grid_params)}: {params}")
            
            try:
                # Create model with current parameters
                model = Prophet(
                    growth='linear',
                    seasonality_mode=params['seasonality_mode'],
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    interval_width=0.95
                )
                
                # Add regressors and seasonalities
                model = self.add_regressors(model, train_df)
                model = self.add_custom_seasonalities(model)
                
                # Fit model
                model.fit(train_df)
                
                # Cross-validation
                cv_results = cross_validation(
                    model, 
                    initial=cv_params['initial'],
                    period=cv_params['period'],
                    horizon=cv_params['horizon'],
                    disable_tqdm=True
                )
                
                # Calculate performance metrics
                cv_metrics = performance_metrics(cv_results)
                avg_rmse = cv_metrics['rmse'].mean()
                
                print(f"    Average RMSE: {avg_rmse:.2f}")
                
                # Update best parameters
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = params.copy()
                    best_cv_results = cv_results.copy()
                    print(f"    New best RMSE: {best_score:.2f}")
                
            except Exception as e:
                print(f"    Error with parameters {params}: {str(e)}")
                continue
        
        self.best_params = best_params
        self.cv_results = best_cv_results
        
        print(f"\nHyperparameter tuning completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best RMSE: {best_score:.2f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': best_cv_results
        }
    
    def fit_tuned_model(self, train_df: pd.DataFrame) -> Prophet:
        """
        Fit Prophet model with tuned hyperparameters.
        
        Args:
            train_df: Training data
            
        Returns:
            Prophet: Fitted tuned model
        """
        if self.best_params is None:
            raise ValueError("Must run hyperparameter tuning first")
        
        print("Fitting tuned Prophet model...")
        
        # Create model with best parameters
        model = Prophet(
            growth='linear',
            seasonality_mode=self.best_params['seasonality_mode'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=self.best_params['changepoint_prior_scale'],
            seasonality_prior_scale=self.best_params['seasonality_prior_scale'],
            holidays_prior_scale=self.best_params['holidays_prior_scale'],
            interval_width=0.95
        )
        
        # Add regressors and seasonalities
        model = self.add_regressors(model, train_df)
        model = self.add_custom_seasonalities(model)
        
        # Fit model
        model.fit(train_df)
        
        self.model = model
        self.is_fitted = True
        
        print("Tuned model fitted successfully!")
        return model
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'best_params': self.best_params,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.best_params = model_data.get('best_params')
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from the fitted model.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get regressor coefficients
        regressors = []
        if hasattr(self.model, 'extra_regressors'):
            for regressor_name, regressor_data in self.model.extra_regressors.items():
                # Get the coefficient from the model's parameters
                if hasattr(self.model, 'params') and regressor_name in self.model.params:
                    coef = self.model.params[regressor_name].mean()
                    regressors.append({
                        'feature': regressor_name,
                        'coefficient': coef,
                        'abs_coefficient': abs(coef)
                    })
        
        if regressors:
            importance_df = pd.DataFrame(regressors)
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            return importance_df
        else:
            print("No regressor coefficients available")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the Prophet model
    import yaml
    import sys
    sys.path.append('../data')
    from data_loader import DataLoader
    from preprocessor import ProphetPreprocessor
    
    # Load configuration
    with open('../../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    loader = DataLoader(config)
    data = loader.load_and_merge_data()
    
    preprocessor = ProphetPreprocessor(config)
    train_df, test_df = preprocessor.prepare_data_for_prophet(data)
    
    # Test baseline model
    prophet_model = EnergyProphetModel(config)
    
    print("Testing baseline model...")
    model = prophet_model.fit_baseline_model(train_df)
    forecast = prophet_model.predict(test_df)
    
    print("\nForecast sample:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    
    # Test feature importance
    importance = prophet_model.get_feature_importance()
    print("\nFeature importance:")
    print(importance)