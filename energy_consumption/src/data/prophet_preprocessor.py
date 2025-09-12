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
        df = df.copy()

        # Reset index to make sure 'ds' is a column
        if not 'ds' in df.columns:
            df = df.reset_index().rename(columns={'time': 'ds'})

        # Ensure datetime type
        df['ds'] = pd.to_datetime(df['ds'])

        # Ensure numeric target
        df['y'] = pd.to_numeric(df['total_load_actual'], errors='coerce')

        # Drop duplicate 'total_load_actual' (since it's now 'y')
        if 'total_load_actual' in df.columns:
            df = df.drop(columns=['total_load_actual'])

        print(df.head())
        print(f"Prophet dataframe created with shape: {df.shape}")
        return df 
    
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

        # 🔹 Encode categorical before Prophet
        for split in [train_df, test_df]:
            split['season'] = split['season'].astype(str)  # ensure string
        train_df = pd.get_dummies(train_df, columns=['season'], prefix='season', drop_first=True)
        test_df  = pd.get_dummies(test_df,  columns=['season'], prefix='season', drop_first=True)

        # Convert to Prophet format
        train_df = self.create_prophet_dataframe(train_df)
        test_df  = self.create_prophet_dataframe(test_df)   

        # Scale regressors
        train_df = self.scale_regressors(train_df, fit=True)
        test_df  = self.scale_regressors(test_df, fit=False)

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