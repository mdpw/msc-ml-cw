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

    def plot_prophet_forecast(self, model, forecast: pd.DataFrame, 
                             actual_df: pd.DataFrame = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Prophet forecast results.
        
        Args:
            model: Fitted Prophet model
            forecast: Prophet forecast dataframe
            actual_df: Actual test data for comparison
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig = model.plot(forecast, figsize=self.figsize)
        
        # Add actual test data if provided
        if actual_df is not None:
            plt.plot(actual_df['ds'], actual_df['y'], 'ro', markersize=3, 
                    alpha=0.6, label='Actual Test Data')
            plt.legend()
        
        plt.title('Prophet Forecast vs Actual', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy Load (MW)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_prophet_components(self, model, forecast: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Prophet forecast components.
        
        Args:
            model: Fitted Prophet model
            forecast: Prophet forecast dataframe
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig = model.plot_components(forecast, figsize=(15, 10))
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    
    def plot_forecast_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                               dates: pd.DatetimeIndex = None,
                               confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                               title: str = "Forecast vs Actual",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot forecast vs actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Datetime index for x-axis
            confidence_intervals: Tuple of (lower, upper) confidence bounds
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), dpi=self.dpi)
        
        if dates is None:
            dates = pd.date_range(start='2018-01-01', periods=len(y_true), freq='H')
        
        # Time series plot
        axes[0].plot(dates, y_true, label='Actual', linewidth=1.5, alpha=0.8)
        axes[0].plot(dates, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
        
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            axes[0].fill_between(dates, lower, upper, alpha=0.2, label='Confidence Interval')
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Energy Load (MW)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect Prediction')
        
        axes[1].set_xlabel('Actual Load (MW)', fontsize=12)
        axes[1].set_ylabel('Predicted Load (MW)', fontsize=12)
        axes[1].set_title('Predicted vs Actual Scatter Plot', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add R² to scatter plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               dates: pd.DatetimeIndex = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot residuals analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Datetime index
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        residuals = y_true - y_pred
        
        if dates is None:
            dates = pd.date_range(start='2018-01-01', periods=len(y_true), freq='H')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Residuals over time
        axes[0, 0].plot(dates, residuals, linewidth=0.8, alpha=0.8)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Residuals (MW)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residuals (MW)', fontsize=10)
        axes[0, 1].set_ylabel('Density', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Predicted Values (MW)', fontsize=10)
        axes[1, 1].set_ylabel('Residuals (MW)', fontsize=10)
        axes[1, 1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of metrics across different models.
        
        Args:
            metrics_dict: Dictionary of model_name -> metrics
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        # Convert to dataframe for easier plotting
        metrics_df = pd.DataFrame(metrics_dict).T
        
        # Select key metrics for comparison
        key_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        axes = axes.flatten()
        
        colors = sns.color_palette("husl", len(metrics_df))
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                bars = axes[i].bar(metrics_df.index, metrics_df[metric], color=colors)
                axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with features and their importance scores
            top_n: Number of top features to display
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        if importance_df.empty:
            print("No importance data to plot")
            return None
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        bars = ax.barh(range(len(top_features)), top_features['abs_coefficient'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title(f'Top {len(top_features)} Feature Importance (Prophet Model)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_chronos_forecast(self, forecast: pd.DataFrame, 
                         actual_df: pd.DataFrame = None,
                         train_df: pd.DataFrame = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Chronos forecast results.
        
        Args:
            forecast: Chronos forecast dataframe with columns [ds, yhat, yhat_lower, yhat_upper]
            actual_df: Actual test data for comparison
            train_df: Training data to show context
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot training data if provided (for context)
        if train_df is not None:
            # Show only last 100 points of training data for clarity
            train_subset = train_df.tail(100) if len(train_df) > 100 else train_df
            ax.plot(train_subset['ds'], train_subset['y'], 
                label='Training Data', color='blue', alpha=0.7, linewidth=1)
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
            label='Chronos Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        ax.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    alpha=0.3, color='red', label='Confidence Interval')
        
        # Plot actual test data if provided
        if actual_df is not None:
            ax.plot(actual_df['ds'], actual_df['y'], 
                'go', markersize=4, alpha=0.8, label='Actual Test Data')
        
        ax.set_title('Chronos Forecast vs Actual', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Energy Load (MW)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig

    def plot_chronos_vs_prophet(self, chronos_forecast: pd.DataFrame,
                            prophet_forecast: pd.DataFrame,
                            actual_df: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare Chronos and Prophet forecasts.
        
        Args:
            chronos_forecast: Chronos predictions
            prophet_forecast: Prophet predictions  
            actual_df: Actual test data
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot actual data
        ax.plot(actual_df['ds'], actual_df['y'], 
            'ko-', markersize=3, linewidth=1.5, label='Actual', alpha=0.8)
        
        # Plot Chronos forecast
        ax.plot(chronos_forecast['ds'], chronos_forecast['yhat'],
            'r-', linewidth=2, label='Chronos Forecast')
        ax.fill_between(chronos_forecast['ds'],
                    chronos_forecast['yhat_lower'],
                    chronos_forecast['yhat_upper'],
                    alpha=0.2, color='red')
        
        # Plot Prophet forecast
        ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'],
            'b-', linewidth=2, label='Prophet Forecast')
        ax.fill_between(prophet_forecast['ds'],
                    prophet_forecast['yhat_lower'],
                    prophet_forecast['yhat_upper'],
                    alpha=0.2, color='blue')
        
        ax.set_title('Chronos vs Prophet Forecast Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Energy Load (MW)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
    
    
    