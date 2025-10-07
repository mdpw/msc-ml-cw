import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple
import warnings
import pickle
import os
warnings.filterwarnings('ignore')


class EnergyLSTMModel:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_fitted = False
        self.history = None
        
        # LSTM parameters
        self.sequence_length = config.get('lstm', {}).get('sequence_length', 30)
        self.epochs = config.get('lstm', {}).get('epochs', 100)
        self.batch_size = config.get('lstm', {}).get('batch_size', 32)
        self.units = config.get('lstm', {}).get('units', 50)
        self.dropout = config.get('lstm', {}).get('dropout', 0.2)
        self.learning_rate = config.get('lstm', {}).get('learning_rate', 0.001)
        
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Feature data
            target: Target values
            sequence_length: Length of input sequences
            
        Returns:
            X, y: Sequences and targets
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        print("Building LSTM model...")
        
        model = Sequential([
            LSTM(self.units, activation='relu', return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.dropout),
            
            LSTM(self.units // 2, activation='relu', return_sequences=False),
            Dropout(self.dropout),
            
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"Model built with {model.count_params()} parameters")
        return model
    
    def fit_baseline_model(self, train_df: pd.DataFrame) -> 'EnergyLSTMModel':
        """
        Fit baseline LSTM model.
        
        Args:
            train_df: Training dataframe with features
            
        Returns:
            self
        """
        print("Fitting LSTM baseline model...")
        
        # Prepare data
        target_col = self.config['target']['column']
        
        # Select features (exclude target and non-numeric columns)
        feature_cols = [col for col in train_df.columns 
                       if col != target_col and train_df[col].dtype in ['float64', 'int64']]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values.reshape(-1, 1)
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create sequences
        X_seq, y_seq = self.prepare_sequences(
            X_train_scaled, 
            y_train_scaled.flatten(), 
            self.sequence_length
        )
        
        print(f"Training sequences shape: X={X_seq.shape}, y={y_seq.shape}")
        
        # Build model
        self.model = self.build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        
        # Callbacks
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        # Train model
        print("Training LSTM model...")
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.is_fitted = True
        self.feature_cols = feature_cols
        
        print("LSTM model trained successfully!")
        return self
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the fitted LSTM model.
        
        Args:
            test_df: Test dataframe with features
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("Generating LSTM predictions...")
        
        target_col = self.config['target']['column']
        
        # Prepare test data
        X_test = test_df[self.feature_cols].values
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Create sequences
        predictions = []
        
        # For each test point, we need sequence_length previous points
        for i in range(len(X_test_scaled)):
            if i < self.sequence_length:
                # Use last points from training data if needed
                # For simplicity, we'll start predictions after sequence_length
                continue
            
            # Get sequence
            X_seq = X_test_scaled[i-self.sequence_length:i].reshape(1, self.sequence_length, -1)
            
            # Predict
            y_pred_scaled = self.model.predict(X_seq, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            
            predictions.append(y_pred[0, 0])
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': test_df.index[self.sequence_length:],
            'yhat': predictions,
            'yhat_lower': np.array(predictions) * 0.95,  # Simple confidence interval
            'yhat_upper': np.array(predictions) * 1.05
        })
        
        print(f"Generated {len(predictions)} predictions")
        return forecast_df
    
    def get_feature_importance(self) -> pd.DataFrame:        
        print("Note: LSTM doesn't provide traditional feature importance")
        return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # Save other components
        model_data = {
            'config': self.config,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'is_fitted': self.is_fitted,
            'history': self.history.history if self.history else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath} and {model_path}")
    
    def load_model(self, filepath: str):
        """Load a fitted model from disk."""
        # Load Keras model
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load other components
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_cols = model_data['feature_cols']
        self.sequence_length = model_data['sequence_length']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")