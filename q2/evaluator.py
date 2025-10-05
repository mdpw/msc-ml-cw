"""
Common evaluation module for all forecasting models.
Calculates metrics: MAE, RMSE, MAPE, R²
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluator class for time series forecasting models.
    Provides metrics calculation, comparison, and visualization.
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str) -> Dict[str, float]:
        """
        Calculate forecasting performance metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Handle any NaN or infinite values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | 
                 np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE - handle zero values
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # R²
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"  {model_name} - Performance Metrics")
        print(f"{'='*60}")
        print(f"  MAE:   {metrics['MAE']:,.2f}")
        print(f"  RMSE:  {metrics['RMSE']:,.2f}")
        print(f"  MAPE:  {metrics['MAPE']:.2f}%")
        print(f"  R²:    {metrics['R2']:.4f}")
        print(f"{'='*60}\n")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with all models and their metrics
        """
        if not self.results:
            print("No models evaluated yet!")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(2)
        
        # Add ranking for each metric (lower is better for MAE, RMSE, MAPE; higher is better for R²)
        comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
        comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
        comparison_df['MAPE_Rank'] = comparison_df['MAPE'].rank()
        comparison_df['R2_Rank'] = comparison_df['R2'].rank(ascending=False)
        
        # Calculate average rank
        comparison_df['Avg_Rank'] = comparison_df[['MAE_Rank', 'RMSE_Rank', 
                                                     'MAPE_Rank', 'R2_Rank']].mean(axis=1)
        
        # Sort by average rank
        comparison_df = comparison_df.sort_values('Avg_Rank')
        
        return comparison_df
    
    def plot_predictions(self, y_true: pd.Series, predictions: Dict[str, pd.Series], 
                        save_path: str = 'plots/predictions_comparison.png'):
        """
        Plot actual vs predicted values for all models.
        
        Args:
            y_true: Actual values with datetime index
            predictions: Dictionary of {model_name: predicted_series}
            save_path: Path to save the plot
        """
        plt.figure(figsize=(18, 8))
        
        # Plot actual values
        plt.plot(y_true.index, y_true.values, label='Actual', 
                color='black', linewidth=2, alpha=0.7)
        
        # Plot predictions for each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            plt.plot(y_pred.index, y_pred.values, 
                    label=model_name, color=colors[i % len(colors)], 
                    linewidth=1.5, alpha=0.8, linestyle='--')
        
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Energy Load (MW)', fontsize=12, fontweight='bold')
        plt.title('Actual vs Predicted Energy Load - All Models', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Predictions plot saved to {save_path}")
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str, save_path: str = None):
        """
        Plot residuals analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, color='blue', alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # Predicted vs Actual
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[1, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual analysis plot saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, save_path: str = 'plots/error_distribution.png'):
        """
        Plot error distribution comparison for all models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No models evaluated yet!")
            return
        
        metrics = ['MAE', 'RMSE', 'MAPE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, metric in enumerate(metrics):
            model_names = list(self.results.keys())
            values = [self.results[model][metric] for model in model_names]
            
            axes[i].bar(model_names, values, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error distribution plot saved to {save_path}")
    
    def generate_report(self, save_path: str = 'results/evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("  ENERGY LOAD FORECASTING - MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Individual model results
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}\n")
                f.write("-"*60 + "\n")
                f.write(f"  MAE:   {metrics['MAE']:,.2f}\n")
                f.write(f"  RMSE:  {metrics['RMSE']:,.2f}\n")
                f.write(f"  MAPE:  {metrics['MAPE']:.2f}%\n")
                f.write(f"  R²:    {metrics['R2']:.4f}\n")
            
            # Comparison table
            f.write("\n\n" + "="*80 + "\n")
            f.write("  MODEL COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            comparison_df = self.compare_models()
            f.write(comparison_df.to_string())
            
            # Best model
            f.write("\n\n" + "="*80 + "\n")
            best_model = comparison_df.index[0]
            f.write(f"  BEST MODEL: {best_model}\n")
            f.write("="*80 + "\n")
        
        print(f"Evaluation report saved to {save_path}")


if __name__ == "__main__":
    # Test evaluator
    print("Testing ModelEvaluator...")
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.normal(25000, 5000, 100)
    
    # Simulate predictions from different models
    y_pred_prophet = y_true + np.random.normal(0, 1000, 100)
    y_pred_lstm = y_true + np.random.normal(0, 1200, 100)
    y_pred_chronos = y_true + np.random.normal(0, 900, 100)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    metrics_prophet = evaluator.calculate_metrics(y_true, y_pred_prophet, "Prophet")
    evaluator.print_metrics("Prophet", metrics_prophet)
    
    metrics_lstm = evaluator.calculate_metrics(y_true, y_pred_lstm, "LSTM")
    evaluator.print_metrics("LSTM", metrics_lstm)
    
    metrics_chronos = evaluator.calculate_metrics(y_true, y_pred_chronos, "Chronos")
    evaluator.print_metrics("Chronos", metrics_chronos)
    
    # Compare models
    print("\nModel Comparison:")
    print(evaluator.compare_models())
    
    print("\nEvaluator test completed!")