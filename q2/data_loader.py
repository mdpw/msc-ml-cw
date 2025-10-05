import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import holidays
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_energy_data(self) -> pd.DataFrame:       
        print("Loading energy dataset...")
        
        # Load data
        df_energy = pd.read_csv(self.config['data']['raw_energy_path'])
        
        # Extract only the date part from the string
        df_energy['time'] = df_energy['time'].astype(str).str.split(' ').str[0]

        # Convert to datetime
        df_energy['time'] = pd.to_datetime(df_energy['time'])

        # Keep only the needed columns
        df_energy = df_energy[['time', 'total load actual']]

        # Clean column names
        df_energy.columns = [col.replace(' ', '_').lower() for col in df_energy.columns]

        # Set time as index
        df_energy = df_energy.set_index('time')

        # Sort by index
        df_energy = df_energy.sort_index()   

        # Resample to daily frequency and compute the mean for each day and roundup the values.
        df_energy = df_energy.resample('D').mean().round()        

        # Handle missing values
        print(f"Missing values in target variable: {df_energy['total_load_actual'].isnull().sum()}")
        
        # Remove rows with any missing values in target variable
        df_energy = df_energy.dropna(subset=['total_load_actual'])
        
        print(f"Energy data shape after cleaning: {df_energy.shape}")
        print(f"Date range: {df_energy.index.min()} to {df_energy.index.max()}")      
        
        return df_energy
    
    def load_weather_data(self) -> pd.DataFrame:        
        print("Loading weather dataset...")
        
        # Load data
        df_weather = pd.read_csv(self.config['data']['raw_weather_path'])

        # Extract only the date part from the string
        df_weather['dt_iso'] = df_weather['dt_iso'].astype(str).str.split(' ').str[0]

        # Keep only the needed columns
        required_cols = ['dt_iso', 'temp', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all']
        df_weather = df_weather[required_cols]
        
        # Parse datetime - handle both full datetime strings and date-only strings
        df_weather['dt_iso'] = pd.to_datetime(df_weather['dt_iso'], errors='coerce')
        
        # Drop rows with invalid dates
        df_weather = df_weather.dropna(subset=['dt_iso'])
        
        # Convert other columns to numeric
        numeric_cols = [col for col in df_weather.columns if col not in ['dt_iso']]
        for col in numeric_cols:
            df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
        
        # Convert temperature from Kelvin to Celsius BEFORE resampling
        temp_cols = ['temp', 'temp_min', 'temp_max']
        for col in temp_cols:
            if col in df_weather.columns:
                df_weather[col] = df_weather[col] - 273.15                                        

        # Group by date and aggregate
        daily_weather = df_weather.groupby('dt_iso').agg({
            'temp': 'mean',           # Average temperature
            'temp_min': 'min',        # Minimum temperature of the day
            'temp_max': 'max',        # Maximum temperature of the day  
            'humidity': 'mean',       # Average humidity
            'wind_speed': 'mean',     # Average wind speed
            'rain_1h': 'sum',         # Total rainfall (sum of hourly values)
            'rain_3h': 'sum',         # Total rainfall (sum of 3-hourly values)
            'snow_3h': 'sum',         # Total snowfall
            'clouds_all': 'mean'      # Average cloud coverage
        }).round(2)  # Round to 2 decimal places for better precision
        
        # Convert the date index back to datetime for consistency
        daily_weather.index = pd.to_datetime(daily_weather.index)
        daily_weather.index.name = 'dt_iso'
        
        # Handle missing values - use forward fill then backward fill
        daily_weather = daily_weather.fillna(method='ffill').fillna(method='bfill')
        
        # Sort by index (date)
        daily_weather = daily_weather.sort_index()
        
        print(f"Daily weather data shape: {daily_weather.shape}")
        print(f"Date range: {daily_weather.index.min()} to {daily_weather.index.max()}")        
        
        return daily_weather  
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame: 
        print("Creating temporal features...")
        
        # Holidays in Spain
        es_holidays = holidays.Spain(years=range(df.index.min().year,
                                                 df.index.max().year + 1))
        df['is_holiday'] = df.index.isin(es_holidays).astype(int)

        # Weekend flag (Saturday=5, Sunday=6)
        df['is_weekend'] = df.index.weekday >= 5
        df['is_weekend'] = df['is_weekend'].astype(int)        
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        # Month
        df['month'] = df.index.month
        
        # Day of year (for cyclical patterns)
        df['day_of_year'] = df.index.dayofyear
        
        # Seasons (Winter, Spring, Summer, Autumn)
        def get_season(date):
            Y = date.year
            seasons = {
                "spring": (pd.Timestamp(f"{Y}-03-21"), pd.Timestamp(f"{Y}-06-20")),
                "summer": (pd.Timestamp(f"{Y}-06-21"), pd.Timestamp(f"{Y}-09-22")),
                "autumn": (pd.Timestamp(f"{Y}-09-23"), pd.Timestamp(f"{Y}-12-20")),
            }
            if seasons["spring"][0] <= date <= seasons["spring"][1]:
                return "spring"
            elif seasons["summer"][0] <= date <= seasons["summer"][1]:
                return "summer"
            elif seasons["autumn"][0] <= date <= seasons["autumn"][1]:
                return "autumn"
            else:
                return "winter"

        df['season'] = df.index.map(get_season)        
        print("Temporal features created.")
        
        return df
    
    def merge_datasets(self, df_energy: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:        
        print("Merging datasets...")
        
        # Merge on date index
        merged_df = df_energy.join(df_weather, how='inner')
        
        print(f"Merged dataset shape: {merged_df.shape}")
        print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        
        return merged_df