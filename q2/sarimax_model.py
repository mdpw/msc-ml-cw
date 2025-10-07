import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any, Tuple
import warnings
import pickle
import os
warnings.filterwarnings('ignore')


class EnergySARIMAXModel:    
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        
        # SARIMAX parameters from config
        sarimax_config = config.get('sarimax', {})
        self.order = tuple(sarimax_config['order'])
        self.seasonal_order = tuple(sarimax_config['seasonal_order'])
        
        # External regressors (weather features)
        self.exog_features = config['weather_features']
        
    def fit_baseline_model(self, train_df: pd.DataFrame) -> 'EnergySARIMAXModel':
        """
        Fit baseline SARIMAX model.
        
        Args:
            train_df: Training dataframe with 'ds', 'y', and weather features
            
        Returns:
            self
        """
        print("Fitting SARIMAX baseline model...")
        print(f"Order (p,d,q): {self.order}")
        print(f"Seasonal Order (P,D,Q,s): {self.seasonal_order}")
        
        # Extract target variable
        if isinstance(train_df['y'], pd.Series):
            y_train = train_df['y'].values
        else:
            y_train = np.array(train_df['y'])
        
        # Extract exogenous variables (weather features)
        exog_cols = [col for col in self.exog_features if col in train_df.columns]
        
        if exog_cols:
            X_train = train_df[exog_cols].values
            print(f"Using {len(exog_cols)} external regressors: {exog_cols}")
        else:
            X_train = None
            print("No external regressors found, using SARIMA only")
        
        # Fit SARIMAX model
        try:
            self.model = SARIMAX(
                y_train,
                exog=X_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            print("Training SARIMAX model (this may take a few minutes)...")
            self.model_fit = self.model.fit(disp=False, maxiter=200)
            
            print(f"Model converged: {self.model_fit.mle_retvals['converged']}")
            print(f"AIC: {self.model_fit.aic:.2f}")
            print(f"BIC: {self.model_fit.bic:.2f}")
            
        except Exception as e:
            print(f"Error fitting SARIMAX model: {str(e)}")
            print("Trying simpler ARIMA model without seasonality...")
            
            # Fallback to simpler ARIMA
            self.seasonal_order = (0, 0, 0, 0)
            self.model = SARIMAX(
                y_train,
                exog=X_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = self.model.fit(disp=False, maxiter=200)
        
        self.exog_cols = exog_cols
        self.is_fitted = True
        
        print("SARIMAX model fitted successfully!")
        return self
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted SARIMAX model.
        
        Args:
            test_df: Test dataframe with 'ds' and weather features
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("Generating SARIMAX predictions...")
        
        # Extract exogenous variables for test set
        if self.exog_cols:
            X_test = test_df[self.exog_cols].values
        else:
            X_test = None
        
        # Generate forecast
        forecast_result = self.model_fit.forecast(
            steps=len(test_df),
            exog=X_test
        )
        
        # Get prediction intervals
        forecast_df_full = self.model_fit.get_forecast(
            steps=len(test_df),
            exog=X_test
        )
        
        forecast_ci = forecast_df_full.conf_int()
        
        # Handle ds column - convert to array safely
        if isinstance(test_df['ds'], pd.Series):
            ds_values = test_df['ds'].values
        else:
            ds_values = np.array(test_df['ds'])
        
        # Handle forecast_result - convert to array safely
        if isinstance(forecast_result, pd.Series):
            yhat_values = forecast_result.values
        else:
            yhat_values = np.array(forecast_result)
        
        # Handle confidence intervals
        if isinstance(forecast_ci, pd.DataFrame):
            lower_values = forecast_ci.iloc[:, 0].values
            upper_values = forecast_ci.iloc[:, 1].values
        else:
            lower_values = np.array(forecast_ci[:, 0])
            upper_values = np.array(forecast_ci[:, 1])
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': ds_values,
            'yhat': yhat_values,
            'yhat_lower': lower_values,
            'yhat_upper': upper_values
        })
        
        print(f"Generated {len(forecast_df)} predictions")
        return forecast_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from exogenous variables.
        
        Returns:
            DataFrame with feature coefficients
        """
        if not self.is_fitted:
            print("Model must be fitted first")
            return pd.DataFrame()
        
        if not self.exog_cols:
            print("No external regressors in model")
            return pd.DataFrame()
        
        # Get parameters
        params = self.model_fit.params
        
        # Extract exogenous variable coefficients
        importance_data = []
        for i, feature in enumerate(self.exog_cols):
            param_name = f'x{i+1}' if len(self.exog_cols) > 1 else 'x1'
            if param_name in params.index:
                coef = params[param_name]
                importance_data.append({
                    'feature': feature,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef)
                })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
            return importance_df
        else:
            return pd.DataFrame()
    
    def get_model_summary(self) -> str:
        """Get model summary statistics."""
        if not self.is_fitted:
            return "Model not fitted yet"
        
        return str(self.model_fit.summary())
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'config': self.config,
            'model_fit': self.model_fit,
            'exog_cols': self.exog_cols,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.model_fit = model_data['model_fit']
        self.exog_cols = model_data['exog_cols']
        self.order = model_data['order']
        self.seasonal_order = model_data['seasonal_order']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")