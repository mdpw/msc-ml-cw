import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from sklearn.ensemble import IsolationForest

# ============================================================================
# 1. MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    strategy : str
        'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill', 'knn'
    columns : list, optional
        Specific columns to handle. If None, handles all columns
    
    Returns:
    --------
    pandas.DataFrame, dict
        Processed dataframe and fitted imputers
    """
    df_processed = df.copy()
    fitted_imputers = {}
    
    if columns is None:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        numeric_cols = [col for col in columns if df_processed[col].dtype in [np.number]]
        categorical_cols = [col for col in columns if df_processed[col].dtype in ['object', 'category']]
    
    print(f"Handling missing values using {strategy} strategy...")
    print(f"Missing values before: {df_processed.isnull().sum().sum()}")
    
    if strategy == 'drop':
        df_processed = df_processed.dropna()
    
    elif strategy in ['forward_fill', 'backward_fill']:
        method = 'ffill' if strategy == 'forward_fill' else 'bfill'
        df_processed = df_processed.fillna(method=method)
    
    elif strategy == 'knn':
        if numeric_cols:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_processed[numeric_cols] = knn_imputer.fit_transform(df_processed[numeric_cols])
            fitted_imputers['knn_numeric'] = knn_imputer
        
        if categorical_cols:
            # Use mode for categorical in KNN
            mode_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = mode_imputer.fit_transform(df_processed[categorical_cols])
            fitted_imputers['mode_categorical'] = mode_imputer
    
    else:
        # Handle numeric columns
        if numeric_cols and strategy in ['mean', 'median']:
            num_imputer = SimpleImputer(strategy=strategy)
            df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])
            fitted_imputers['numeric'] = num_imputer
        
        # Handle categorical columns
        if categorical_cols:
            cat_strategy = 'most_frequent' if strategy == 'mode' else 'most_frequent'
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
            fitted_imputers['categorical'] = cat_imputer
    
    print(f"Missing values after: {df_processed.isnull().sum().sum()}")
    return df_processed, fitted_imputers

# ============================================================================
# 2. OUTLIER DETECTION AND HANDLING
# ============================================================================

def handle_outliers(df, method='iqr', threshold=1.5, action='remove', columns=None):
    """
    Detect and handle outliers in dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    method : str
        'iqr', 'zscore', 'isolation_forest'
    threshold : float
        Threshold for outlier detection
    action : str
        'remove', 'cap', 'transform'
    columns : list, optional
        Specific columns to check for outliers
    
    Returns:
    --------
    pandas.DataFrame, dict
        Processed dataframe and outlier information
    """
    df_processed = df.copy()
    outlier_info = {}
    
    if columns is None:
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_columns = [col for col in columns if df_processed[col].dtype in [np.number]]
    
    if not numeric_columns:
        print("No numeric columns found for outlier detection")
        return df_processed, outlier_info
    
    print(f"Detecting outliers using {method} method...")
    
    outlier_indices = set()
    outlier_details = {}
    
    if method == 'isolation_forest':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(df_processed[numeric_columns])
        outlier_indices = set(df_processed[outlier_pred == -1].index)
        outlier_details['isolation_forest'] = outlier_indices
    
    else:
        for col in numeric_columns:
            col_outliers = []
            
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = df_processed[(df_processed[col] < lower_bound) | 
                                          (df_processed[col] > upper_bound)].index.tolist()
                outlier_details[col] = {'lower': lower_bound, 'upper': upper_bound}
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_processed[col]))
                col_outliers = df_processed[z_scores > threshold].index.tolist()
                outlier_details[col] = {'threshold': threshold}
            
            outlier_indices.update(col_outliers)
    
    print(f"Found {len(outlier_indices)} outlier rows")
    
    # Handle outliers based on action
    if action == 'remove':
        df_processed = df_processed.drop(index=list(outlier_indices))
    
    elif action == 'cap':
        for col in numeric_columns:
            if method == 'iqr' and col in outlier_details:
                lower = outlier_details[col]['lower']
                upper = outlier_details[col]['upper']
                df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
    
    elif action == 'transform':
        # Log transformation for positive values
        for col in numeric_columns:
            if (df_processed[col] > 0).all():
                df_processed[col] = np.log1p(df_processed[col])
    
    outlier_info = {
        'method': method,
        'outlier_indices': list(outlier_indices),
        'details': outlier_details,
        'action': action
    }
    
    return df_processed, outlier_info

# ============================================================================
# 3. CATEGORICAL ENCODING
# ============================================================================

def encode_categorical(df, method='onehot', columns=None, drop_first=True):
    """
    Encode categorical variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    method : str
        'onehot', 'label', 'ordinal', 'target' (target requires target column)
    columns : list, optional
        Specific columns to encode
    drop_first : bool
        Whether to drop first category in one-hot encoding
    
    Returns:
    --------
    pandas.DataFrame, dict
        Processed dataframe and fitted encoders
    """
    df_processed = df.copy()
    fitted_encoders = {}
    
    if columns is None:
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_columns = [col for col in columns if col in df_processed.columns]
    
    if not categorical_columns:
        print("No categorical columns found")
        return df_processed, fitted_encoders
    
    print(f"Encoding categorical variables using {method} method...")
    print(f"Categorical columns: {categorical_columns}")
    
    if method == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            fitted_encoders[col] = le
    
    elif method == 'onehot':
        df_processed = pd.get_dummies(df_processed, 
                                    columns=categorical_columns, 
                                    drop_first=drop_first,
                                    prefix=categorical_columns)
        fitted_encoders['onehot_columns'] = categorical_columns
    
    elif method == 'ordinal':
        from sklearn.preprocessing import OrdinalEncoder
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_processed[categorical_columns] = ordinal_encoder.fit_transform(df_processed[categorical_columns].astype(str))
        fitted_encoders['ordinal'] = ordinal_encoder
    
    return df_processed, fitted_encoders

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================

def scale_features(df, method='standard', columns=None):
    """
    Scale numerical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    method : str
        'standard', 'minmax', 'robust', 'normalize'
    columns : list, optional
        Specific columns to scale
    
    Returns:
    --------
    pandas.DataFrame, object
        Processed dataframe and fitted scaler
    """
    df_processed = df.copy()
    
    if columns is None:
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_columns = [col for col in columns if df_processed[col].dtype in [np.number]]
    
    if not numeric_columns:
        print("No numeric columns found for scaling")
        return df_processed, None
    
    print(f"Scaling features using {method} method...")
    print(f"Scaling columns: {numeric_columns}")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'normalize':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer()
    else:
        raise ValueError("Invalid scaling method")
    
    df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
    
    return df_processed, scaler

# ============================================================================
# 5. DATA TYPE CONVERSION
# ============================================================================

def convert_data_types(df, conversions=None, auto_convert=True):
    """
    Convert data types of columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    conversions : dict
        Dictionary mapping column names to desired types
        e.g., {'col1': 'int64', 'col2': 'category'}
    auto_convert : bool
        Whether to automatically convert obvious cases
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with converted types
    """
    df_processed = df.copy()
    
    print("Converting data types...")
    
    # Manual conversions
    if conversions:
        for col, dtype in conversions.items():
            if col in df_processed.columns:
                try:
                    df_processed[col] = df_processed[col].astype(dtype)
                    print(f"Converted {col} to {dtype}")
                except Exception as e:
                    print(f"Could not convert {col} to {dtype}: {e}")
    
    # Automatic conversions
    if auto_convert:
        for col in df_processed.columns:
            # Convert numeric strings to numeric
            if df_processed[col].dtype == 'object':
                # Try to convert to numeric
                numeric_converted = pd.to_numeric(df_processed[col], errors='coerce')
                if not numeric_converted.isnull().all():
                    df_processed[col] = numeric_converted
                    print(f"Auto-converted {col} to numeric")
                
                # Convert low cardinality strings to category
                elif df_processed[col].nunique() / len(df_processed) < 0.5:
                    df_processed[col] = df_processed[col].astype('category')
                    print(f"Auto-converted {col} to category")
    
    return df_processed

# ============================================================================
# 6. FEATURE SELECTION
# ============================================================================

def select_features(df, target_column, method='correlation', threshold=0.1, k=10):
    """
    Select relevant features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_column : str
        Name of target column
    method : str
        'correlation', 'mutual_info', 'chi2', 'variance', 'univariate'
    threshold : float
        Threshold for feature selection
    k : int
        Number of features to select (for k-best methods)
    
    Returns:
    --------
    pandas.DataFrame, list
        Dataframe with selected features and list of selected feature names
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found")
    
    df_processed = df.copy()
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Selecting features using {method} method...")
    
    if method == 'correlation' and numeric_columns:
        # Calculate correlation with target
        correlations = X[numeric_columns].corrwith(y).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
    
    elif method == 'variance':
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X[numeric_columns])
        selected_features = np.array(numeric_columns)[selector.get_support()].tolist()
    
    elif method == 'univariate':
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        # Choose score function based on target type
        if y.dtype == 'object' or y.nunique() < 10:
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_columns)))
        X_selected = selector.fit_transform(X[numeric_columns], y)
        selected_features = np.array(numeric_columns)[selector.get_support()].tolist()
    
    else:
        selected_features = numeric_columns
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Return dataframe with selected features + target
    selected_columns = selected_features + [target_column]
    return df_processed[selected_columns], selected_features

# ============================================================================
# 7. EXAMPLE USAGE FUNCTION
# ============================================================================

def preprocess_pipeline_example(df, target_column=None):
    """
    Example of how to use all preprocessing methods together
    """
    print("="*50)
    print("PREPROCESSING PIPELINE EXAMPLE")
    print("="*50)
    
    # Step 1: Handle missing values
    df, imputers = handle_missing_values(df, strategy='mean')
    print()
    
    # Step 2: Handle outliers
    df, outlier_info = handle_outliers(df, method='iqr', action='cap')
    print()
    
    # Step 3: Convert data types
    df = convert_data_types(df, auto_convert=True)
    print()
    
    # Step 4: Encode categorical variables
    df, encoders = encode_categorical(df, method='onehot')
    print()
    
    # Step 5: Scale features
    df, scaler = scale_features(df, method='standard')
    print()
    
    # Step 6: Feature selection (if target is provided)
    if target_column:
        df, selected_features = select_features(df, target_column, method='correlation')
        print()
    
    print("Preprocessing completed!")
    print(f"Final dataset shape: {df.shape}")
    
    return df