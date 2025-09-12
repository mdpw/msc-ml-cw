"""
Data preprocessing utilities for Prophet model preparation.
"""

import pandas as pd
import numpy as np
import data.preprocessor as base_preprocessor
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ProphetPreprocessor:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()        
   
    def create_prophet_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:         
        df = df.copy()  # safe copy

        # Ensure numeric
        df['total_load_actual'] = pd.to_numeric(df['total_load_actual'], errors='coerce')

        # Create dataframe
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['total_load_actual'].values  # <- use .values to avoid index alignment issues
        })

        print(prophet_df.head())
        print(f"Prophet dataframe created with shape: {prophet_df.shape}")
        return prophet_df   
    
    def scale_regressors(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        # Identify numerical regressors (exclude ds, y, and categorical features)
        exclude_cols = ['ds', 'y', 'is_holiday', 'is_weekend', 'season']
        numerical_cols = [col for col in df.columns 
                         if df[col].dtype in ['int64', 'float64'] and col not in exclude_cols]
        
        if len(numerical_cols) > 0:
            if fit:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
                print(f"Fitted scaler on {len(numerical_cols)} features")
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                print(f"Applied existing scaler to {len(numerical_cols)} features")
        
        return df
    
   
    def prepare_data_for_prophet(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:        
        print("Starting preprocessing pipeline...")
        
        # Train-test split
        preprocessor = base_preprocessor.Preprocessor(self.config)
        train_df, test_df = preprocessor.train_test_split(df)

        # Convert to Prophet format
        train_df = self.create_prophet_dataframe(train_df)
        test_df = self.create_prophet_dataframe(test_df)   
        
        # Scale regressors
        train_df = self.scale_regressors(train_df, fit=True)
        test_df = self.scale_regressors(test_df, fit=False)
        
        print("Preprocessing completed successfully!")
        return train_df, test_df



if __name__ == "__main__":
    # Test preprocessing
    import yaml
    from data_loader import DataLoader
    
    with open('../../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    loader = DataLoader(config)
    data = loader.load_and_merge_data()
    
    # Preprocess
    preprocessor = ProphetPreprocessor(config)
    train_df, test_df = preprocessor.prepare_data_for_prophet(data)
    
    print("\nTrain data sample:")
    print(train_df.head())
    print("\nTest data sample:")  
    print(test_df.head())