"""
Data preprocessing utilities for Prophet model preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ProphetPreprocessor:
    """Preprocessing pipeline for Facebook Prophet model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        
    def create_prophet_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data to Prophet format (ds, y columns).
        
        Args:
            df: Input dataframe with datetime index and target column
            
        Returns:
            pd.DataFrame: Prophet-formatted dataframe
        """
        prophet_df = pd.DataFrame()
        
        # Prophet requires 'ds' (datestamp) and 'y' (target) columns
        prophet_df['ds'] = df.index
        prophet_df['y'] = df[self.config['target']['column']]
        
        # Add weather regressors
        weather_features = self.config['weather_features']
        for feature in weather_features:
            if feature in df.columns:
                prophet_df[feature] = df[feature].values
        
        # Reset index
        prophet_df = prophet_df.reset_index(drop=True)
        
        print(f"Prophet dataframe created with shape: {prophet_df.shape}")
        return prophet_df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features that might help with forecasting.
        
        Args:
            df: Prophet-formatted dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional temporal features
        """
        df = df.copy()
        
        # Convert ds to datetime if it's not already
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Extract temporal components
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['quarter'] = df['ds'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Business hour indicator (assuming 8-18 as business hours)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        
        # Season indicators
        df['season'] = df['month'].apply(self._get_season)
        
        # Lag features for target variable
        df = self._add_lag_features(df)
        
        # Temperature-based features
        if 'temp' in df.columns:
            # Temperature categories
            df['temp_category'] = pd.cut(df['temp'], 
                                       bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                       labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
            
            # Heating/cooling degree days approximation
            df['heating_demand'] = np.maximum(18 - df['temp'], 0)
            df['cooling_demand'] = np.maximum(df['temp'] - 25, 0)
        
        print(f"Added temporal features. New shape: {df.shape}")
        return df
    
    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for target variable."""
        # Sort by datetime to ensure proper lag calculation
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Add lag features (previous day same hour, previous week same hour)
        df['y_lag_24h'] = df['y'].shift(24)  # Previous day, same hour
        df['y_lag_168h'] = df['y'].shift(168)  # Previous week, same hour
        
        # Rolling statistics
        df['y_rolling_mean_24h'] = df['y'].rolling(window=24, min_periods=1).mean().shift(1)
        df['y_rolling_std_24h'] = df['y'].rolling(window=24, min_periods=1).std().shift(1)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in the target variable.
        
        Args:
            df: Input dataframe
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df['y'].quantile(0.25)
            Q3 = df['y'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df['y'] < lower_bound) | (df['y'] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df['y'] - df['y'].mean()) / df['y'].std())
            outliers = z_scores > 3
        
        print(f"Identified {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")
        
        # Cap outliers instead of removing them
        if outliers.sum() > 0:
            df.loc[outliers & (df['y'] > df['y'].median()), 'y'] = df['y'].quantile(0.99)
            df.loc[outliers & (df['y'] < df['y'].median()), 'y'] = df['y'].quantile(0.01)
            print("Outliers capped to 1st and 99th percentiles")
        
        return df
    
    def scale_regressors(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical regressors for better Prophet performance.
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler or use existing fit
            
        Returns:
            pd.DataFrame: Dataframe with scaled regressors
        """
        df = df.copy()
        
        # Identify numerical regressors (exclude ds, y, and categorical features)
        exclude_cols = ['ds', 'y', 'temp_category', 'season']
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
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets using time-based split.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.sort_values('ds').reset_index(drop=True)
        
        test_size = self.config['evaluation']['test_size']
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_df)} samples ({train_df['ds'].min()} to {train_df['ds'].max()})")
        print(f"Test set: {len(test_df)} samples ({test_df['ds'].min()} to {test_df['ds'].max()})")
        
        return train_df, test_df
    
    def prepare_data_for_prophet(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline for Prophet.
        
        Args:
            df: Raw merged dataset
            
        Returns:
            Tuple of (train_df, test_df) ready for Prophet
        """
        print("Starting preprocessing pipeline...")
        
        # Convert to Prophet format
        prophet_df = self.create_prophet_dataframe(df)
        
        # Handle outliers
        prophet_df = self.handle_outliers(prophet_df)
        
        # Add temporal features
        prophet_df = self.add_temporal_features(prophet_df)
        
        # Drop rows with NaN values (from lag features)
        initial_len = len(prophet_df)
        prophet_df.head()
        prophet_df = safe_handle_missing_values(prophet_df)
        print(f"Dropped {initial_len - len(prophet_df)} rows with NaN values")
        
        # Train-test split
        train_df, test_df = self.train_test_split(prophet_df)
        
        # Scale regressors
        train_df = self.scale_regressors(train_df, fit=True)
        test_df = self.scale_regressors(test_df, fit=False)
        
        print("Preprocessing completed successfully!")
        return train_df, test_df

def safe_handle_missing_values(prophet_df):
    """
    Safely handle missing values without removing all data.
    """
    initial_len = len(prophet_df)
    
    # 1. Only drop rows where target variable is missing
    prophet_df = prophet_df.dropna(subset=['y'])
    
    # 2. Fill missing values in feature columns instead of dropping rows
    feature_columns = [col for col in prophet_df.columns if col not in ['ds', 'y']]
    
    for col in feature_columns:
        if prophet_df[col].isna().any():
            if prophet_df[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                prophet_df[col] = prophet_df[col].fillna(prophet_df[col].median())
            else:
                # Fill categorical columns with mode
                mode_val = prophet_df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 0
                prophet_df[col] = prophet_df[col].fillna(fill_val)
    
    print(f"Safely handled missing values. Removed {initial_len - len(prophet_df)} rows with missing target only")
    
    return prophet_df

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