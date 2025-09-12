"""
Visualization utilities for energy forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnergyForecastVisualizer:
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_time_series_overview(self, df: pd.DataFrame, 
                                 target_col: str = 'total_load_actual',
                                 save_path: Optional[str] = None) -> plt.Figure:
        
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # Ensure target column is numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Drop any NaNs in the target column
        df = df.dropna(subset=[target_col])

        fig, axes = plt.subplots(2, 1, figsize=(15, 12), dpi=100)  # self.dpi removed for simplicity

        # Main time series
        axes[0].plot(df.index, df[target_col], linewidth=0.8, alpha=0.8)
        axes[0].set_title('Energy Load Time Series', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Load (MW)', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Monthly averages
        monthly_avg = df[target_col].resample('M').mean()
        axes[1].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
        axes[1].set_title('Monthly Average Energy Load', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Load (MW)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.show()  # Ensures plot appears in Jupyter notebooks
        return fig

    def plot_weather_correlations(self, df: pd.DataFrame, 
                                 target_col: str = 'total_load_actual',
                                 weather_cols: List[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        
        if weather_cols is None:
            weather_cols = ['temp', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'rain_1h', 'rain_3h' , 'snow_3h', 'clouds_all']
        
        # Filter available columns
        available_cols = [col for col in weather_cols if col in df.columns]
        
        if not available_cols:
            print("No weather columns found in dataframe")
            return None
        
        n_cols = len(available_cols)
        fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 8), dpi=self.dpi)
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, weather_col in enumerate(available_cols):
            # Scatter plot with trend line
            x = df[weather_col].dropna()
            y = df.loc[x.index, target_col]
            
            axes[i].scatter(x, y, alpha=0.3, s=1)
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = x.corr(y)
            axes[i].set_title(f'{weather_col.title()} vs Energy Load\nCorrelation: {corr:.3f}')
            axes[i].set_xlabel(weather_col.title())
            axes[i].set_ylabel('Load (MW)')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
if __name__ == "__main__":
    # Test visualization functions
    import sys
    sys.path.append('../data')
    from data_loader import DataLoader
    
    # Create sample data for testing
    dates = pd.date_range(start='2018-01-01', end='2018-01-07', freq='H')
    n_points = len(dates)
    
    # Generate sample energy load data
    np.random.seed(42)
    base_load = 25000
    daily_pattern = 5000 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    noise = np.random.normal(0, 1000, n_points)
    energy_load = base_load + daily_pattern + noise
    
    
    