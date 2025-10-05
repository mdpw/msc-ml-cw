
"""
Main script to train all models and evaluate performance.
Trains: Prophet, Chronos, LSTM
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
    """Prepare data for Prophet (needs 'ds' and 'y' columns)"""
    print("\nPreparing data for Prophet...")
    
    # Prophet requires specific column names
    train_prophet = train_df.reset_index().rename(columns={
        train_df.index.name or 'index': 'ds',
        target_col: 'y'
    })
    
    test_prophet = test_df.reset_index().rename(columns={
        test_df.index.name or 'index': 'ds',
        target_col: 'y'
    })
    
    return train_prophet, test_prophet


def train_prophet(config, train_df, test_df):
    """Train and evaluate Prophet model"""
    print("\n" + "="*80)
    print("TRAINING PROPHET MODEL")
    print("="*80)
    
    target_col = config['target']['column']
    
    # Prepare data
    train_prophet, test_prophet = prepare_prophet_data(train_df, test_df, target_col)
    
    # Initialize and train model
    prophet_model = EnergyProphetModel(config)
    prophet_model.fit_baseline_model(train_prophet)
    
    # Generate predictions
    forecast = prophet_model.predict(test_prophet)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    prophet_model.save_model('models/prophet_model.pkl')
    
    return forecast['yhat'].values, prophet_model


def train_chronos(config, train_df, test_df):
    """Train and evaluate Chronos model"""
    print("\n" + "="*80)
    print("TRAINING CHRONOS MODEL")
    print("="*80)
    
    target_col = config['target']['column']
    
    # Prepare data (Chronos also uses Prophet format)
    train_chronos, test_chronos = prepare_prophet_data(train_df, test_df, target_col)
    
    # Initialize and train model
    chronos_model = EnergyChronosModel(config)
    chronos_model.fit_baseline_model(train_chronos)
    
    # Generate predictions
    forecast = chronos_model.predict(test_chronos)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    chronos_model.save_model('models/chronos_model.pkl')
    
    return forecast['yhat'].values, chronos_model


def train_lstm(config, train_df, test_df):
    """Train and evaluate LSTM model"""
    print("\n" + "="*80)
    print("TRAINING LSTM MODEL")
    print("="*80)
    
    # Initialize and train model
    lstm_model = EnergyLSTMModel(config)
    lstm_model.fit_baseline_model(train_df)
    
    # Generate predictions
    forecast = lstm_model.predict(test_df)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    lstm_model.save_model('models/lstm_model.pkl')
    
    return forecast['yhat'].values, lstm_model


def main():
    """Main execution function"""
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
    
    # ============================================================
    # 1. TRAIN PROPHET
    # ============================================================
    try:
        y_pred_prophet, prophet_model = train_prophet(config, train_df, test_df)
        
        # Align predictions with test data (Prophet might return different length)
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
        
        print("✓ Prophet model completed successfully!")
        
    except Exception as e:
        print(f"✗ Prophet model failed: {str(e)}")
    
    # ============================================================
    # 2. TRAIN CHRONOS
    # ============================================================
    try:
        y_pred_chronos, chronos_model = train_chronos(config, train_df, test_df)
        
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
        
        print("✓ Chronos model completed successfully!")
        
    except Exception as e:
        print(f"✗ Chronos model failed: {str(e)}")
    
    # ============================================================
    # 3. TRAIN LSTM
    # ============================================================
    try:
        y_pred_lstm, lstm_model = train_lstm(config, train_df, test_df)
        
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
        
        print("✓ LSTM model completed successfully!")
        
    except Exception as e:
        print(f"✗ LSTM model failed: {str(e)}")
    
    # ============================================================
    # 4. COMPARISON & VISUALIZATION
    # ============================================================
    print("\n" + "="*80)
    print("  MODEL COMPARISON & RESULTS")
    print("="*80 + "\n")
    
    # Compare all models
    comparison_df = evaluator.compare_models()
    print("\nComparison Table:")
    print(comparison_df)
    
    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv')
    print("\n✓ Comparison saved to results/model_comparison.csv")
    
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
    
    # ============================================================
    # 5. BEST MODEL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    
    best_model = comparison_df.index[0]
    best_metrics = evaluator.results[best_model]
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   MAE:  {best_metrics['MAE']:,.2f}")
    print(f"   RMSE: {best_metrics['RMSE']:,.2f}")
    print(f"   MAPE: {best_metrics['MAPE']:.2f}%")
    print(f"   R²:   {best_metrics['R2']:.4f}")
    
    print("\n" + "="*80)
    print("  ALL TASKS COMPLETED!")
    print("="*80)
    print("\n📁 Output Files:")
    print("   - Models saved in: models/")
    print("   - Plots saved in: plots/")
    print("   - Results saved in: results/")
    print("\n✓ Training and evaluation completed successfully!\n")


if __name__ == "__main__":
    main()