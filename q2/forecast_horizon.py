import pandas as pd
import numpy as np
import yaml
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from prophet_model import EnergyProphetModel
from lstm_model import EnergyLSTMModel
from chronos_model import EnergyChronosModel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_simple_forecast(train_df, forecast_df, target_col, save_path='results/30day_forecast/forecast_visualization.png'):
    """Simple visualization of actual vs forecast."""
    
    plt.figure(figsize=(18, 7))
    
    # Plot last 60 days of historical data
    historical = train_df.tail(60)
    plt.plot(historical.index, historical[target_col], 
            label='Actual Historical Load', 
            color='blue', 
            linewidth=2.5,
            marker='o',
            markersize=4)
    
    # Plot 30-day forecast
    forecast_dates = pd.to_datetime(forecast_df['Date'])
    plt.plot(forecast_dates, forecast_df['Predicted_Load_MW'], 
            label='30-Day Forecast', 
            color='red', 
            linewidth=2.5,
            marker='s',
            markersize=4,
            linestyle='--')
    
    # Add vertical line at forecast start
    plt.axvline(x=forecast_dates.iloc[0], color='green', 
                linestyle=':', linewidth=2, label='Forecast Start')
    
    plt.xlabel('Date', fontsize=13, fontweight='bold')
    plt.ylabel('Energy Load (MW)', fontsize=13, fontweight='bold')
    plt.title('Energy Load: Actual Historical vs 30-Day Forecast', 
            fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

def main():
    print("\n" + "="*80)
    print("  30-DAY AHEAD ENERGY LOAD FORECAST")
    print("="*80 + "\n")
    
    config = load_config()
    horizon_days = config.get('forecasting', {}).get('horizon', 30)
    
    # Load all available data for training
    train_df = pd.read_csv('data/processed/train_data.csv', index_col=0, parse_dates=True)
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Last training date: {train_df.index.max()}")
    print(f"Forecast horizon: {horizon_days} days")
    print(f"Forecast end date: {train_df.index.max() + timedelta(days=horizon_days)}")
    
    # Generate future dates
    last_date = train_df.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    
    print(f"\nForecasting from {future_dates[0]} to {future_dates[-1]}")
    
    # Prepare future dataframe with weather features
    print("\nNote: Using historical weather averages for future predictions")
    weather_features = ['temp', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 
                       'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all']
    
    future_weather = {}
    for feature in weather_features:
        if feature in train_df.columns:
            future_weather[feature] = [train_df[feature].tail(30).mean()] * horizon_days
    
    # Create future dataframe
    future_df = pd.DataFrame({
        'ds': future_dates,
        **future_weather,
        'is_weekend': [1 if d.weekday() >= 5 else 0 for d in future_dates],
        'day_of_week': [d.weekday() for d in future_dates],
        'month': [d.month for d in future_dates],
        'day_of_year': [d.dayofyear for d in future_dates]
    })
        
    # Add season
    def get_season(date):
        month = date.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'
    
    future_df['season'] = [get_season(d) for d in future_dates]
    
    # Prepare Prophet format
    train_prophet = train_df.reset_index().rename(columns={
        train_df.index.name: 'ds',
        config['target']['column']: 'y'
    })
    
    # Train Prophet model
    print("\n" + "="*80)
    print("TRAINING PROPHET MODEL FOR 30-DAY FORECAST")
    print("="*80)
    
    prophet_model = EnergyProphetModel(config)
    prophet_model.fit_baseline_model(train_prophet)
    
        # Generate 30-day forecast
    forecast = prophet_model.predict(future_df)

    # Select only essential columns
    forecast_simple = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    # Rename columns for clarity
    forecast_simple.columns = ['Date', 'Predicted_Load_MW', 'Lower_Bound_MW', 'Upper_Bound_MW']

    # Save results
    os.makedirs('results/30day_forecast', exist_ok=True)
    forecast_simple.to_csv('results/30day_forecast/prophet_30day_forecast.csv', index=False)

    # Display results
    print("\n" + "="*80)
    print("  30-DAY FORECAST RESULTS")
    print("="*80)
    print(forecast_simple.to_string(index=False))

    plot_simple_forecast(
        train_df=train_df,
        forecast_df=forecast_simple,
        target_col=config['target']['column'],
        save_path='results/30day_forecast/forecast_visualization.png'
    )

if __name__ == "__main__":
    main()