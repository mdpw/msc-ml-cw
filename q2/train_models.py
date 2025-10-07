"""
Main script to train all models and evaluate performance.
Trains: Prophet, Chronos, LSTM, SARIMAX
"""

import pandas as pd
import numpy as np
import yaml
import warnings
import os
warnings.filterwarnings('ignore')

# Import model classes
from prophet_model import EnergyProphetModel
from chronos_model import EnergyChronosModel
from lstm_model import EnergyLSTMModel
from sarimax_model import EnergySARIMAXModel
from evaluator import ModelEvaluator


def load_config():
    """Load configuration file"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_processed_data():
    """Load preprocessed train and test data"""
    print("Loading processed data...")
    train_df = pd.read_csv('data/processed/train_data.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('data/processed/test_data.csv', index_col=0, parse_dates=True)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def prepare_prophet_data(train_df, test_df, target_col):
    """Prepare data for Prophet and SARIMAX (needs 'ds' and 'y' columns)"""
    print("\nPreparing data for Prophet/SARIMAX...")
    
    # Get the actual index name (should be 'time' based on your data)
    index_name = train_df.index.name if train_df.index.name else 'time'
    
    # Prophet requires specific column names
    train_prophet = train_df.reset_index().rename(columns={
        index_name: 'ds',
        target_col: 'y'
    })
    
    test_prophet = test_df.reset_index().rename(columns={
        index_name: 'ds',
        target_col: 'y'
    })
    
    print(f"Prophet data columns: {train_prophet.columns.tolist()}")
    
    return train_prophet, test_prophet


def run_prophet(config, train_df, test_df, y_true, evaluator, all_predictions):
    """Train and evaluate Prophet model"""
    print("\n" + "="*80)
    print("TRAINING PROPHET MODEL")
    print("="*80)
    
    try:
        target_col = config['target']['column']
        
        # Prepare data
        train_prophet, test_prophet = prepare_prophet_data(train_df, test_df, target_col)
        
        # Initialize and train model
        prophet_model = EnergyProphetModel(config)
        prophet_model.fit_baseline_model(train_prophet)
        
        # Generate predictions
        forecast = prophet_model.predict(test_prophet)
        y_pred_prophet = forecast['yhat'].values
        
        # Save model
        os.makedirs('models', exist_ok=True)
        prophet_model.save_model('models/prophet_model.pkl')
        
        # Align predictions with test data
        if len(y_pred_prophet) != len(y_true):
            min_len = min(len(y_pred_prophet), len(y_true))
            y_pred_prophet = y_pred_prophet[:min_len]
            y_true_prophet = y_true[:min_len]
        else:
            y_true_prophet = y_true
        
        # Evaluate
        metrics_prophet = evaluator.calculate_metrics(y_true_prophet, y_pred_prophet, "Prophet")
        evaluator.print_metrics("Prophet", metrics_prophet)
        
        # Plot residuals
        evaluator.plot_residuals(y_true_prophet, y_pred_prophet, "Prophet", 
                                save_path='plots/prophet_residuals.png')
        
        # Store predictions
        all_predictions['Prophet'] = pd.Series(y_pred_prophet, 
                                               index=test_df.index[:len(y_pred_prophet)])
        
        print("Prophet model completed successfully!")
        return True
        
    except Exception as e:
        print(f"Prophet model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_chronos(config, train_df, test_df, y_true, evaluator, all_predictions):
    """Train and evaluate Chronos model"""
    print("\n" + "="*80)
    print("TRAINING CHRONOS MODEL")
    print("="*80)
    
    try:
        target_col = config['target']['column']
        
        # Prepare data (Chronos also uses Prophet format)
        train_chronos, test_chronos = prepare_prophet_data(train_df, test_df, target_col)
        
        # Initialize and train model
        chronos_model = EnergyChronosModel(config)
        chronos_model.fit_baseline_model(train_chronos)
        
        # Generate predictions
        forecast = chronos_model.predict(test_chronos)
        y_pred_chronos = forecast['yhat'].values
        
        # Save model
        os.makedirs('models', exist_ok=True)
        chronos_model.save_model('models/chronos_model.pkl')
        
        # Align predictions
        if len(y_pred_chronos) != len(y_true):
            min_len = min(len(y_pred_chronos), len(y_true))
            y_pred_chronos = y_pred_chronos[:min_len]
            y_true_chronos = y_true[:min_len]
        else:
            y_true_chronos = y_true
        
        # Evaluate
        metrics_chronos = evaluator.calculate_metrics(y_true_chronos, y_pred_chronos, "Chronos")
        evaluator.print_metrics("Chronos", metrics_chronos)
        
        # Plot residuals
        evaluator.plot_residuals(y_true_chronos, y_pred_chronos, "Chronos",
                                save_path='plots/chronos_residuals.png')
        
        # Store predictions
        all_predictions['Chronos'] = pd.Series(y_pred_chronos,
                                              index=test_df.index[:len(y_pred_chronos)])
        
        print("Chronos model completed successfully!")
        return True
        
    except Exception as e:
        print(f"Chronos model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_lstm(config, train_df, test_df, y_true, evaluator, all_predictions):
    """Train and evaluate LSTM model"""
    print("\n" + "="*80)
    print("TRAINING LSTM MODEL")
    print("="*80)
    
    try:
        # Initialize and train model
        lstm_model = EnergyLSTMModel(config)
        lstm_model.fit_baseline_model(train_df)
        
        # Generate predictions
        forecast = lstm_model.predict(test_df)
        y_pred_lstm = forecast['yhat'].values
        
        # Save model
        os.makedirs('models', exist_ok=True)
        lstm_model.save_model('models/lstm_model.pkl')
        
        # LSTM predictions start after sequence_length
        sequence_length = config.get('lstm', {}).get('sequence_length', 30)
        y_true_lstm = y_true[sequence_length:]
        
        # Evaluate
        metrics_lstm = evaluator.calculate_metrics(y_true_lstm, y_pred_lstm, "LSTM")
        evaluator.print_metrics("LSTM", metrics_lstm)
        
        # Plot residuals
        evaluator.plot_residuals(y_true_lstm, y_pred_lstm, "LSTM",
                                save_path='plots/lstm_residuals.png')
        
        # Store predictions
        all_predictions['LSTM'] = pd.Series(y_pred_lstm,
                                           index=test_df.index[sequence_length:])
        
        print("LSTM model completed successfully!")
        return True
        
    except Exception as e:
        print(f"LSTM model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_sarimax(config, train_df, test_df, y_true, evaluator, all_predictions):
    """Train and evaluate SARIMAX model"""
    print("\n" + "="*80)
    print("TRAINING SARIMAX MODEL")
    print("="*80)
    
    try:
        target_col = config['target']['column']
        
        # Prepare data (SARIMAX uses Prophet format)
        train_sarimax, test_sarimax = prepare_prophet_data(train_df, test_df, target_col)
        
        # Initialize and train model
        sarimax_model = EnergySARIMAXModel(config)
        sarimax_model.fit_baseline_model(train_sarimax)
        
        # Generate predictions
        forecast = sarimax_model.predict(test_sarimax)
        
        # Extract predictions safely
        if isinstance(forecast['yhat'], pd.Series):
            y_pred_sarimax = forecast['yhat'].values
        else:
            y_pred_sarimax = np.array(forecast['yhat'])
        
        # Save model
        os.makedirs('models', exist_ok=True)
        sarimax_model.save_model('models/sarimax_model.pkl')
        
        # Align predictions
        if len(y_pred_sarimax) != len(y_true):
            min_len = min(len(y_pred_sarimax), len(y_true))
            y_pred_sarimax = y_pred_sarimax[:min_len]
            y_true_sarimax = y_true[:min_len]
        else:
            y_true_sarimax = y_true
        
        # Evaluate
        metrics_sarimax = evaluator.calculate_metrics(y_true_sarimax, y_pred_sarimax, "SARIMAX")
        evaluator.print_metrics("SARIMAX", metrics_sarimax)
        
        # Plot residuals
        evaluator.plot_residuals(y_true_sarimax, y_pred_sarimax, "SARIMAX",
                                save_path='plots/sarimax_residuals.png')
        
        # Store predictions
        all_predictions['SARIMAX'] = pd.Series(y_pred_sarimax,
                                              index=test_df.index[:len(y_pred_sarimax)])
        
        print("SARIMAX model completed successfully!")
        return True
        
    except Exception as e:
        print(f"SARIMAX model failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_comparison_and_visualizations(evaluator, all_predictions, test_df, y_true):
    """Generate comparison tables and visualizations"""
    print("\n" + "="*80)
    print("  MODEL COMPARISON & RESULTS")
    print("="*80 + "\n")
    
    # Compare all models
    if len(evaluator.results) > 0:
        comparison_df = evaluator.compare_models()
        print("\nComparison Table:")
        print(comparison_df.to_string())
        
        # Save comparison with better formatting
        comparison_df.to_csv('results/model_comparison.csv', float_format='%.2f')
        print("\nComparison saved to results/model_comparison.csv")
        print(f"   Total models evaluated: {len(evaluator.results)}")
    else:
        print("\nWarning: No models were successfully evaluated!")
        print("   Check the error messages above for details.")
        return
    
    # Plot predictions comparison
    if all_predictions:
        # Get common index for all predictions
        common_index = test_df.index
        for model_name, pred_series in all_predictions.items():
            if len(pred_series) < len(common_index):
                common_index = pred_series.index
                break
        
        y_true_series = pd.Series(y_true, index=test_df.index)
        y_true_series = y_true_series.loc[common_index]
        
        evaluator.plot_predictions(y_true_series, all_predictions,
                                  save_path='plots/predictions_comparison.png')
    
    # Plot error distribution
    evaluator.plot_error_distribution(save_path='plots/error_distribution.png')
    
    # Generate comprehensive report
    evaluator.generate_report(save_path='results/evaluation_report.txt')


def print_final_summary(evaluator):
    """Print final summary with best model"""
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    
    if len(evaluator.results) > 0:
        comparison_df = evaluator.compare_models()
        best_model = comparison_df.index[0]
        best_metrics = evaluator.results[best_model]
        
        print(f"\nBEST MODEL: {best_model}")
        print(f"   MAE:  {best_metrics['MAE']:,.2f}")
        print(f"   RMSE: {best_metrics['RMSE']:,.2f}")
        print(f"   MAPE: {best_metrics['MAPE']:.2f}%")
        print(f"   R²:   {best_metrics['R2']:.4f}")
        
        # Summary of all models
        print("\nMODEL RANKINGS:")
        for idx, model in enumerate(comparison_df.index, 1):
            metrics = evaluator.results[model]
            print(f"   {idx}. {model:15s} - MAE: {metrics['MAE']:>8,.2f}, "
                  f"RMSE: {metrics['RMSE']:>8,.2f}, "
                  f"MAPE: {metrics['MAPE']:>6.2f}%, "
                  f"R²: {metrics['R2']:>7.4f}")
    
    print("\n" + "="*80)
    print("  ALL TASKS COMPLETED!")
    print("="*80)
    print("\nOutput Files:")
    print("   - Models saved in: models/")
    print("   - Plots saved in: plots/")
    print("   - Results saved in: results/")
    print("\nTraining and evaluation completed successfully!\n")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function - Initialize and setup"""
    print("\n" + "="*80)
    print("  ENERGY LOAD FORECASTING - MODEL TRAINING & EVALUATION")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load configuration
    config = load_config()
    target_col = config['target']['column']
    
    # Load processed data
    train_df, test_df = load_processed_data()
    
    # Get actual test values
    y_true = test_df[target_col].values
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Dictionary to store predictions for comparison
    all_predictions = {}
    
    return config, train_df, test_df, y_true, evaluator, all_predictions


# ============================================================
# PIPELINE EXECUTION
# ============================================================

if __name__ == "__main__":
    # Initialize
    config, train_df, test_df, y_true, evaluator, all_predictions = main()    
    
    run_prophet(config, train_df, test_df, y_true, evaluator, all_predictions)
    run_chronos(config, train_df, test_df, y_true, evaluator, all_predictions)
    run_lstm(config, train_df, test_df, y_true, evaluator, all_predictions)    
    run_sarimax(config, train_df, test_df, y_true, evaluator, all_predictions)  
    
    generate_comparison_and_visualizations(evaluator, all_predictions, test_df, y_true)
    
    print_final_summary(evaluator)