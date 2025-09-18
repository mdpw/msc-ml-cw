"""
Chronos model implementation for energy forecasting.
"""

from chronos import ChronosPipeline
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnergyChronosModel:
    """Chronos model wrapper for energy load forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.train_data = None
        
        # Initialize Chronos pipeline
        # Available models: "amazon/chronos-t5-tiny", "amazon/chronos-t5-mini", 
        # "amazon/chronos-t5-small", "amazon/chronos-t5-base", "amazon/chronos-t5-large"
        model_name = config.get('chronos', {}).get('model_name', 'amazon/chronos-t5-small')
        
        print(f"Initializing Chronos model: {model_name}")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
    def fit_baseline_model(self, train_df: pd.DataFrame) -> 'EnergyChronosModel':
        """
        Fit baseline Chronos model.
        Note: Chronos is pre-trained, so 'fitting' mainly involves storing training data.
        """
        print("Preparing Chronos baseline model...")
        
        # Store training data for context
        self.train_data = train_df.copy()
        
        # Chronos doesn't require traditional fitting, but we prepare the data
        self.y_train = train_df['y'].values
        
        # Set context length (Chronos uses the last N points for forecasting)
        self.context_length = min(len(self.y_train), 512)  # Chronos typically uses up to 512 context points
        
        self.is_fitted = True
        print("Chronos model prepared successfully!")
        return self
    
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecast using Chronos."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("Generating Chronos predictions...")
        
        # Prepare context (last part of training data)
        context = torch.tensor(self.y_train[-self.context_length:], dtype=torch.float32)
        
        # Forecast horizon
        prediction_length = len(future_df)
        
        # Generate forecast
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=100,  # Number of sample paths for uncertainty quantification
        )
        
        # Extract predictions and quantiles
        forecast_median = np.median(forecast[0].numpy(), axis=0)
        forecast_lower = np.percentile(forecast[0].numpy(), 5, axis=0)  # 5th percentile
        forecast_upper = np.percentile(forecast[0].numpy(), 95, axis=0)  # 95th percentile
        
        # Create forecast dataframe in Prophet-like format
        forecast_df = pd.DataFrame({
            'ds': future_df['ds'],
            'yhat': forecast_median,
            'yhat_lower': forecast_lower,
            'yhat_upper': forecast_upper
        })
        
        print(f"Forecast generated for {len(forecast_df)} periods")
        return forecast_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Chronos doesn't provide feature importance like Prophet.
        Return empty dataframe for compatibility.
        """
        print("Note: Chronos doesn't provide feature importance analysis")
        return pd.DataFrame()
    
    def save_model(self, path: str):
        """Save model state (mainly training data and config)."""
        import pickle
        
        model_state = {
            'config': self.config,
            'train_data': self.train_data,
            'y_train': self.y_train if hasattr(self, 'y_train') else None,
            'context_length': self.context_length if hasattr(self, 'context_length') else None,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model state saved to {path}")
    
    def load_model(self, path: str):
        """Load model state."""
        import pickle
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.train_data = model_state['train_data']
        self.y_train = model_state['y_train']
        self.context_length = model_state['context_length']
        self.is_fitted = model_state['is_fitted']
        
        print(f"Model state loaded from {path}")


if __name__ == "__main__":
    # Test Chronos model
    import yaml
    
    # Sample configcd
    config = {
        'chronos': {
            'model_name': 'amazon/chronos-t5-tiny'  # Use tiny for faster testing
        }
    }
    
    # Create sample data
    dates = pd.date_range(start='2018-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'ds': dates,
        'y': np.random.normal(25000, 5000, 100)
    })
    
    print("Testing Chronos model...")
    
    # Test model
    model = EnergyChronosModel(config)
    model.fit_baseline_model(data)
    
    # Generate forecast
    future_dates = pd.date_range(start='2018-04-11', periods=30, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    
    forecast = model.predict(future_df)
    print("\nForecast sample:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())