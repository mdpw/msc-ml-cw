"""
Data loading utilities for energy and weather datasets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and initial processing of energy and weather data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_energy_data(self) -> pd.DataFrame:
        """
        Load and clean energy dataset.
        
        Returns:
            pd.DataFrame: Cleaned energy data with datetime index
        """
        print("Loading energy dataset...")
        
        # Load data
        df_energy = pd.read_csv(self.config['data']['raw_energy_path'])
        
        # Convert time column to datetime
        df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
        df_energy = df_energy.set_index('time')
        
        # Clean column names (replace spaces with underscores)
        df_energy.columns = [col.replace(' ', '_').lower() for col in df_energy.columns]
        
        # Handle missing values
        print(f"Missing values in target variable: {df_energy['total_load_actual'].isnull().sum()}")
        
        # Forward fill missing values (reasonable for hourly energy data)
        df_energy['total_load_actual'] = df_energy['total_load_actual'].fillna(method='ffill')
        
        # Remove any remaining missing values
        df_energy = df_energy.dropna(subset=['total_load_actual'])
        
        print(f"Energy data shape after cleaning: {df_energy.shape}")
        print(f"Date range: {df_energy.index.min()} to {df_energy.index.max()}")
        
        return df_energy
    
    def load_weather_data(self) -> pd.DataFrame:
        """
        Load and clean weather dataset.
        
        Returns:
            pd.DataFrame: Cleaned weather data with datetime index
        """
        print("Loading weather dataset...")
        
        # Load data
        df_weather = pd.read_csv(self.config['data']['raw_weather_path'])
        
        # Convert datetime
        df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
        
        # Clean city names (remove leading/trailing spaces)
        df_weather['city_name'] = df_weather['city_name'].str.strip()
        
        # Convert temperature from Kelvin to Celsius
        temp_cols = ['temp', 'temp_min', 'temp_max']
        for col in temp_cols:
            if col in df_weather.columns:
                df_weather[col] = df_weather[col] - 273.15
        
        # Handle missing values in weather data
        numeric_cols = df_weather.select_dtypes(include=[np.number]).columns
        df_weather[numeric_cols] = df_weather[numeric_cols].fillna(method='ffill')
        
        print(f"Weather data shape: {df_weather.shape}")
        print(f"Available cities: {df_weather['city_name'].unique()}")
        
        return df_weather
    
    def aggregate_weather_by_cities(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate weather data across cities using weighted average.
        
        Args:
            df_weather: Raw weather data
            
        Returns:
            pd.DataFrame: Aggregated weather data with datetime index
        """
        print("Aggregating weather data across cities...")
        
        # Get city weights from config
        city_weights = self.config['cities']
        
        # Filter for relevant cities
        relevant_cities = list(city_weights.keys())
        df_weather_filtered = df_weather[df_weather['city_name'].isin(relevant_cities)]
        
        # Aggregate by datetime using weighted average
        weather_features = self.config['weather_features']
        
        aggregated_data = []
        
        for timestamp in df_weather_filtered['dt_iso'].unique():
            timestamp_data = df_weather_filtered[df_weather_filtered['dt_iso'] == timestamp]
            
            agg_row = {'dt_iso': timestamp}
            
            for feature in weather_features:
                if feature in timestamp_data.columns:
                    # Calculate weighted average
                    weighted_sum = 0
                    total_weight = 0
                    
                    for _, row in timestamp_data.iterrows():
                        city = row['city_name']
                        if city in city_weights and not pd.isna(row[feature]):
                            weighted_sum += row[feature] * city_weights[city]
                            total_weight += city_weights[city]
                    
                    if total_weight > 0:
                        agg_row[feature] = weighted_sum / total_weight
                    else:
                        agg_row[feature] = np.nan
            
            aggregated_data.append(agg_row)
        
        # Create aggregated dataframe
        df_weather_agg = pd.DataFrame(aggregated_data)
        df_weather_agg['dt_iso'] = pd.to_datetime(df_weather_agg['dt_iso'])
        df_weather_agg = df_weather_agg.set_index('dt_iso')
        
        # Sort by index
        df_weather_agg = df_weather_agg.sort_index()
        
        print(f"Aggregated weather data shape: {df_weather_agg.shape}")
        
        return df_weather_agg
    
    def merge_datasets(self, df_energy: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
        """
        Merge energy and weather datasets on timestamp.
        
        Args:
            df_energy: Energy consumption data
            df_weather: Aggregated weather data
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        print("Merging energy and weather datasets...")
        
        # Merge on index (datetime)
        df_merged = df_energy.join(df_weather, how='left')
        
        # Forward fill weather data for any missing timestamps
        weather_cols = self.config['weather_features']
        for col in weather_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(method='ffill')
                df_merged[col] = df_merged[col].fillna(method='bfill')
        
        print(f"Merged dataset shape: {df_merged.shape}")
        print(f"Missing values after merge:")
        for col in ['total_load_actual'] + weather_cols:
            if col in df_merged.columns:
                missing = df_merged[col].isnull().sum()
                print(f"  {col}: {missing}")
        
        return df_merged
    
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Complete data loading pipeline.
        
        Returns:
            pd.DataFrame: Final merged and cleaned dataset
        """
        # Load individual datasets
        df_energy = self.load_energy_data()
        df_weather = self.load_weather_data()
        
        # Aggregate weather data
        df_weather_agg = self.aggregate_weather_by_cities(df_weather)
        
        # Merge datasets
        df_merged = self.merge_datasets(df_energy, df_weather_agg)
        
        # Final cleaning - remove any rows with missing target
        df_merged = df_merged.dropna(subset=['total_load_actual'])
        
        print(f"\nFinal dataset ready: {df_merged.shape}")
        return df_merged

# Use this safer approach:
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
    # Test data loading
    import yaml
    
    with open('../../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config)
    data = loader.load_and_merge_data()
    print(data.head())
    print(data.info())