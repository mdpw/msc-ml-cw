import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import yaml
import warnings
warnings.filterwarnings('ignore')

# Import your classes
from data_loader import DataLoader
from preprocessor import Preprocessor

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_config():
    """Load configuration file"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_time_series(df, target_col, save_path='plots/timeseries.png'):
    """Plot the time series data"""
    print("Plotting time series...")
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[target_col], linewidth=1, alpha=0.8)
    plt.title('Energy Load Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Load (MW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to {save_path}")

def plot_decomposition(df, target_col, save_path='plots/decomposition.png'):
    """Decompose and plot trend, seasonality, and residuals"""
    print("Decomposing time series...")
    
    # Use yearly seasonality for energy data
    decomposition = seasonal_decompose(df[target_col], model='additive', period=365, extrapolate_trend='freq')
    
    # Create figure with white background
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # Original
    ax1 = plt.subplot(411)
    ax1.plot(decomposition.observed.index, decomposition.observed.values, 
             color='#0066CC', linewidth=0.5)
    ax1.set_ylabel('Observed', fontsize=12, fontweight='bold')
    ax1.set_title('Original', fontsize=13, loc='center')
    ax1.grid(False)
    ax1.set_facecolor('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Trend - smooth and thick
    ax2 = plt.subplot(412)
    ax2.plot(decomposition.trend.index, decomposition.trend.values, 
             color='#CC0000', linewidth=2)
    ax2.set_ylabel('Trend', fontsize=12, fontweight='bold')
    ax2.set_title('Trend', fontsize=13, loc='center')
    ax2.grid(False)
    ax2.set_facecolor('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Seasonality
    ax3 = plt.subplot(413)
    ax3.plot(decomposition.seasonal.index, decomposition.seasonal.values, 
             color='#009900', linewidth=0.5)
    ax3.set_ylabel('Seasonal', fontsize=12, fontweight='bold')
    ax3.set_title('Seasonality', fontsize=13, loc='center')
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.grid(False)
    ax3.set_facecolor('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Residuals
    ax4 = plt.subplot(414)
    ax4.plot(decomposition.resid.index, decomposition.resid.values, 
             color='#9933CC', linewidth=0.5)
    ax4.set_ylabel('Residual', fontsize=12, fontweight='bold')
    ax4.set_xlabel('time', fontsize=12, fontweight='bold')
    ax4.set_title('Residuals', fontsize=13, loc='center')
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.grid(False)
    ax4.set_facecolor('white')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Decomposition plot saved to {save_path}")

def plot_statistics(df, target_col, save_path='plots/statistics.png'):
    """Plot distribution and box plot"""
    print("Plotting statistics...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribution plot
    axes[0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Energy Load', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Total Load (MW)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', label='Mean')
    axes[0].axvline(df[target_col].median(), color='green', linestyle='--', label='Median')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df[target_col], vert=True)
    axes[1].set_title('Box Plot of Energy Load', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Total Load (MW)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Statistics plot saved to {save_path}")

def plot_seasonal_patterns(df, target_col, save_path='plots/seasonal_patterns.png'):
    """Plot seasonal patterns by month and day of week"""
    print("Plotting seasonal patterns...")
    
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    df_copy['day_of_week'] = df_copy.index.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Monthly pattern
    monthly_avg = df_copy.groupby('month')[target_col].mean()
    axes[0].bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='black')
    axes[0].set_title('Average Load by Month', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Load (MW)')
    axes[0].set_xticks(range(1, 13))
    
    # Day of week pattern
    dow_avg = df_copy.groupby('day_of_week')[target_col].mean()
    axes[1].bar(dow_avg.index, dow_avg.values, color='coral', edgecolor='black')
    axes[1].set_title('Average Load by Day of Week', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Day of Week (0=Monday)')
    axes[1].set_ylabel('Average Load (MW)')
    axes[1].set_xticks(range(7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Seasonal patterns plot saved to {save_path}")

def plot_correlation_matrix(df, save_path='plots/correlation_matrix.png'):
    """Plot correlation matrix of features"""
    print("Plotting correlation matrix...")
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to {save_path}")

def print_summary_statistics(df, target_col):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTarget Variable: {target_col}")
    print(f"Mean: {df[target_col].mean():.2f} MW")
    print(f"Median: {df[target_col].median():.2f} MW")
    print(f"Std Dev: {df[target_col].std():.2f} MW")
    print(f"Min: {df[target_col].min():.2f} MW")
    print(f"Max: {df[target_col].max():.2f} MW")
    print(f"25th Percentile: {df[target_col].quantile(0.25):.2f} MW")
    print(f"75th Percentile: {df[target_col].quantile(0.75):.2f} MW")
    print(f"\nTotal Observations: {len(df)}")
    print(f"Missing Values: {df[target_col].isnull().sum()}")
    print("="*60 + "\n")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load configuration
    config = load_config()
    target_col = config['target']['column']
    
    # Initialize data loader and preprocessor
    loader = DataLoader(config)
    preprocessor = Preprocessor(config)
    
    # Load data
    print("Step 1: Loading datasets...")
    df_energy = loader.load_energy_data()
    df_weather = loader.load_weather_data()
    
    # Merge datasets
    print("\nStep 2: Merging datasets...")
    df = loader.merge_datasets(df_energy, df_weather)
    
    # Create temporal features
    print("\nStep 3: Creating temporal features...")
    df = loader.create_temporal_features(df)
    
    # Handle outliers
    print("\nStep 4: Handling outliers...")
    df = preprocessor.handle_outliers(df, method='iqr')
    
    # Create lag features
    print("\nStep 5: Creating lag features...")
    df = preprocessor.create_lag_features(df, lags=[1, 7, 30])
    
    # Create rolling features
    print("\nStep 6: Creating rolling features...")
    df = preprocessor.create_rolling_features(df, windows=[7, 30])
    
    # Remove rows with NaN values created by lag/rolling features
    print("\nStep 7: Removing NaN values from feature engineering...")
    initial_rows = len(df)
    df = df.dropna()
    print(f"Removed {initial_rows - len(df)} rows with NaN values")
    print(f"Final dataset shape: {df.shape}")
    
    # Print summary statistics
    print_summary_statistics(df, target_col)
    
    # Generate all plots
    print("\nStep 8: Generating visualizations...")
    plot_time_series(df, target_col)
    plot_decomposition(df, target_col)
    plot_statistics(df, target_col)
    plot_seasonal_patterns(df, target_col)
    plot_correlation_matrix(df)
    
    # Split data
    print("\nStep 9: Splitting data into train and test sets...")
    train_df, test_df = preprocessor.train_test_split(df)
    
    # Save processed data
    print("\nStep 10: Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train_data.csv')
    test_df.to_csv('data/processed/test_data.csv')
    print("Data saved to data/processed/")
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS COMPLETED!")
    print("="*60)
    print("\nCheck the 'plots/' folder for visualizations")
    print("Check the 'data/processed/' folder for train and test datasets\n")

if __name__ == "__main__":
    main()