"""
Data preprocessing utilities for Chronos model preparation.
Note: Chronos primarily uses univariate time series, so external regressors are not used.
"""

import pandas as pd
import numpy as np
import data.preprocessor as base_preprocessor
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ChronosPreprocessor:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
   
    def create_chronos_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Chronos dataframe (Chronos is univariate).
        """
        df = df.copy()

        # Reset index to make sure 'ds' is a column
        if not 'ds' in df.columns:
            df = df.reset_index().rename(columns={'time': 'ds'})

        # Ensure datetime type
        df['ds'] = pd.to_datetime(df['ds'])

        # Ensure numeric target
        df['y'] = pd.to_numeric(df['total_load_actual'], errors='coerce')

        # For Chronos, we primarily need just ds and y
        essential_cols = ['ds', 'y']
        df = df[essential_cols]

        print(f"Chronos dataframe created with shape: {df.shape}")
        print("Note: Chronos uses univariate time series (only 'y' values)")
        return df 
   
    def prepare_data_for_chronos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Chronos model (simplified version of Prophet preprocessing).
        """
        print("Starting Chronos preprocessing pipeline...")

        # Train-test split using base preprocessor
        preprocessor = base_preprocessor.Preprocessor(self.config)
        train_df, test_df = preprocessor.train_test_split(df)

        # Convert to Chronos format (much simpler than Prophet)
        train_df = self.create_chronos_dataframe(train_df)
        test_df = self.create_chronos_dataframe(test_df)   

        # Remove any NaN values
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        # Sort by date to ensure chronological order
        train_df = train_df.sort_values('ds').reset_index(drop=True)
        test_df = test_df.sort_values('ds').reset_index(drop=True)

        print("Chronos preprocessing completed successfully!")
        print(f"Train data: {len(train_df)} samples")
        print(f"Test data: {len(test_df)} samples")
        
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
    preprocessor = ChronosPreprocessor(config)
    train_df, test_df = preprocessor.prepare_data_for_chronos(data)
    
    print("\nTrain data sample:")
    print(train_df.head())
    print("\nTest data sample:")  
    print(test_df.head())