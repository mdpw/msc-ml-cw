import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class Preprocessor:    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()        
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:    
        print("Handling outliers started...")    
        df = df.copy()
        target_col = self.config['target']['column']

        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
            outliers = z_scores > 3
        
        print(f"Identified {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")
        
        # Cap outliers instead of removing them
        if outliers.sum() > 0:
            df.loc[outliers & (df[target_col] > df[target_col].median()), target_col] = df[target_col].quantile(0.99)
            df.loc[outliers & (df[target_col] < df[target_col].median()), target_col] = df[target_col].quantile(0.01)
            print("Outliers capped to 1st and 99th percentiles")
        
        print("Handling outliers completed successfully!")
        return df   
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Train and Test splitting started...")        
        test_size = self.config['evaluation']['test_size']
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
        print(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
        print("Train and Test splitting completed successfully!")
        return train_df, test_df

if __name__ == "__main__":
    # Test preprocessing
    import yaml
    from data_loader import DataLoader
    
    with open('../../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    loader = DataLoader(config)
    data = loader.load_and_merge_data()
    
    # Preprocess
    preprocessor = Preprocessor(config)
    train_df, test_df = preprocessor.prepare_data_for_prophet(data)
    
    print("\nTrain data sample:")
    print(train_df.head())
    print("\nTest data sample:")  
    print(test_df.head())