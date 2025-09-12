"""
Evaluation metrics for time series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ForecastMetrics:
    """Comprehensive evaluation metrics for forecasting models."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Handles zero values by adding small epsilon to avoid division by zero.
        """
        epsilon = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient of determination."""
        return r2_score(y_true, y_pred)
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        Better alternative to MAPE for values close to zero.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8)) * 100
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy - how often the model predicts the correct direction of change.
        """
        if len(y_true) < 2:
            return np.nan
        
        # Calculate differences (direction of change)
        true_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)
        
        # Check if directions match (both positive, both negative, or both zero)
        correct_direction = np.sign(true_direction) == np.sign(pred_direction)
        
        return np.mean(correct_direction) * 100
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dict containing all calculated metrics
        """
        metrics = {
            'MAE': self.calculate_mae(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'SMAPE': self.calculate_smape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred),
            'Directional_Accuracy': self.calculate_directional_accuracy(y_true, y_pred)
        }
        
        return metrics
    
    def calculate_metrics_by_period(self, df_results: pd.DataFrame, 
                                   period: str = 'D') -> pd.DataFrame:
        """
        Calculate metrics aggregated by time period.
        
        Args:
            df_results: DataFrame with columns ['ds', 'y_true', 'y_pred']
            period: Aggregation period ('H', 'D', 'W', 'M')
            
        Returns:
            DataFrame with metrics by period
        """
        df = df_results.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.set_index('ds')
        
        # Group by period
        period_groups = df.groupby(pd.Grouper(freq=period))
        
        period_metrics = []
        for period_start, group in period_groups:
            if len(group) > 0:
                metrics = self.calculate_all_metrics(group['y_true'].values, 
                                                   group['y_pred'].values)
                metrics['period'] = period_start
                metrics['n_samples'] = len(group)
                period_metrics.append(metrics)
        
        return pd.DataFrame(period_metrics)
    
    def print_metrics_summary(self, metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print formatted metrics summary.
        
        Args:
            metrics: Dictionary of calculated metrics
            model_name: Name of the model for display
        """
        print(f"\n{'='*50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*50}")
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                if 'Accuracy' in metric or metric in ['R2']:
                    print(f"{metric:20}: {value:8.2f}%")
                elif metric in ['MAE', 'RMSE']:
                    print(f"{metric:20}: {value:8.2f} MW")
                else:
                    print(f"{metric:20}: {value:8.2f}%")
        
        print(f"{'='*50}")


class CrossValidationMetrics:
    """Metrics calculation for time series cross-validation."""
    
    def __init__(self):
        self.cv_results = []
    
    def add_fold_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        fold_id: int, cutoff_date: str):
        """Add results from a single CV fold."""
        metrics_calc = ForecastMetrics()
        fold_metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
        fold_metrics.update({
            'fold_id': fold_id,
            'cutoff_date': cutoff_date,
            'n_samples': len(y_true)
        })
        self.cv_results.append(fold_metrics)
    
    def get_cv_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all CV folds.
        
        Returns:
            Dict with mean, std, min, max for each metric
        """
        if not self.cv_results:
            return {}
        
        df_cv = pd.DataFrame(self.cv_results)
        
        # Calculate summary statistics for each metric
        metric_cols = ['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2', 'Directional_Accuracy']
        summary = {}
        
        for metric in metric_cols:
            if metric in df_cv.columns:
                summary[metric] = {
                    'mean': df_cv[metric].mean(),
                    'std': df_cv[metric].std(),
                    'min': df_cv[metric].min(),
                    'max': df_cv[metric].max()
                }
        
        return summary
    
    def print_cv_summary(self):
        """Print formatted cross-validation summary."""
        summary = self.get_cv_summary()
        
        print(f"\n{'='*60}")
        print("Cross-Validation Results Summary")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'-'*60}")
        
        for metric, stats in summary.items():
            if metric in ['MAE', 'RMSE']:
                unit = " MW"
            else:
                unit = "%"
            
            print(f"{metric:<20} "
                  f"{stats['mean']:<10.2f} "
                  f"{stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} "
                  f"{stats['max']:<10.2f}")
        
        print(f"{'='*60}")


if __name__ == "__main__":
    # Test metrics calculation
    np.random.seed(42)
    y_true = np.random.normal(25000, 5000, 100)  # Energy load values
    y_pred = y_true + np.random.normal(0, 1000, 100)  # Add some prediction error
    
    # Calculate metrics
    metrics_calc = ForecastMetrics()
    metrics = metrics_calc.calculate_all_metrics(y_true, y_pred)
    metrics_calc.print_metrics_summary(metrics, "Test Model")
    
    # Test CV metrics
    cv_metrics = CrossValidationMetrics()
    for i in range(5):
        fold_true = y_true[i*20:(i+1)*20]
        fold_pred = y_pred[i*20:(i+1)*20]
        cv_metrics.add_fold_results(fold_true, fold_pred, i, f"2018-{i+1:02d}-01")
    
    cv_metrics.print_cv_summary()