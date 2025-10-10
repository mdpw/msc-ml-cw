import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shap_summary(trained_pipeline, X_sample, max_features=5):
    """
    Universal SHAP function that works with any model type.
    
    Args:
        trained_pipeline: Trained sklearn pipeline with 'pre' and 'clf' steps
        X_sample: Sample data for SHAP analysis  
        max_features: Number of top features to display
    
    Returns:
        tuple: (top_features, importance_scores, model_type)
    """
    
    model = trained_pipeline.named_steps['clf']
    pre = trained_pipeline.named_steps['pre']
    
    model_name = type(model).__name__.lower()
    print(f"Analyzing {model_name} model with SHAP...")
    print(f"Original sample shape: {X_sample.shape}")
    
    # Use reasonable sample size to avoid memory issues
    if len(X_sample) > 200:
        X_sample = X_sample.sample(200, random_state=42)
        print(f"Reduced sample to: {X_sample.shape}")
    
    # Transform the data
    X_enc = pre.transform(X_sample)
    print(f"Encoded shape: {X_enc.shape}")
    
    # Convert sparse to dense properly
    if hasattr(X_enc, 'toarray'):
        X_enc_dense = X_enc.toarray()
        print("Converted sparse to dense")
    else:
        X_enc_dense = np.array(X_enc)
    
    # Ensure it's a proper numpy array
    X_enc_dense = np.asarray(X_enc_dense, dtype=np.float32)
    print(f"Final dense shape: {X_enc_dense.shape}")
    
    # Get feature names safely
    feature_names = []
    try:
        for name, transformer, columns in pre.transformers_:
            if name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    # Fallback for older sklearn versions
                    for i, col in enumerate(columns):
                        for cat in transformer.categories_[i]:
                            feature_names.append(f"{col}_{cat}")
            elif name == 'num':
                feature_names.extend(columns)
    except Exception as e:
        print(f"Feature name extraction failed: {e}")
        # Create generic names as fallback
        feature_names = [f"feature_{i}" for i in range(X_enc_dense.shape[1])]
    
    print(f"Number of features: {len(feature_names)}")
    
    try:
        print(f"Selecting appropriate SHAP explainer...")
        
        # Tree-based models: Use TreeExplainer (fastest and most accurate)
        if any(tree_type in model_name for tree_type in ['forest', 'tree', 'xgb', 'lightgbm', 'lgbm', 'catboost']):
            print("Using TreeExplainer for tree-based model")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_enc_dense)
            
        # Linear models: Use LinearExplainer
        elif any(linear_type in model_name for linear_type in ['logistic', 'linear', 'ridge', 'lasso']):
            print("Using LinearExplainer for linear model")
            explainer = shap.LinearExplainer(model, X_enc_dense)
            shap_values = explainer.shap_values(X_enc_dense)
            
        # Neural networks and other models: Use KernelExplainer
        else:
            print("Using KernelExplainer for complex model (may take longer...)")
            # Use smaller background sample for KernelExplainer (it's slow)
            background_size = min(50, X_enc_dense.shape[0])
            background = X_enc_dense[:background_size]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Analyze smaller sample
            analyze_size = min(100, X_enc_dense.shape[0])
            shap_values = explainer.shap_values(X_enc_dense[:analyze_size])
            X_enc_dense = X_enc_dense[:analyze_size]  # Match the size
        
        print("SHAP computation successful!")
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # Binary classification - use positive class
                shap_vals = shap_values[1]
                print("Using positive class SHAP values")
            elif len(shap_values) == 1:
                shap_vals = shap_values[0]
            else:
                # Multi-class: use first non-zero class
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
            
        print(f"SHAP values shape: {shap_vals.shape}")
        
        # Calculate feature importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Get top features
        n_show = min(max_features, len(feature_names))
        top_indices = np.argsort(mean_abs_shap)[-n_show:][::-1]
        
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        print(f"\nTop {n_show} Features by SHAP ({model_name.upper()}):")
        for i, (feature, value) in enumerate(zip(top_features, top_values)):
            print(f"{i+1}. {feature}: {value:.4f}")
        
        # Create and save feature importance plot
        _create_shap_plots(top_features, top_values, shap_vals, top_indices, 
                          X_enc_dense, feature_names, model_name, n_show)
        
        return top_features, top_values, model_name
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Falling back to model-specific feature importance...")
        
        # Fallback to model-specific feature importance
        top_features, top_values = _get_fallback_importance(model, feature_names, max_features)
        
        if top_features is not None:
            print(f"\nTop {len(top_features)} Features ({model_name.upper()} built-in importance):")
            for i, (feature, value) in enumerate(zip(top_features, top_values)):
                print(f"{i+1}. {feature}: {value:.4f}")
            
            # Create plot for fallback method
            _create_fallback_plot(top_features, top_values, model_name)
        else:
            print("No feature importance method available for this model")
        
        return top_features, top_values, model_name

def _create_shap_plots(top_features, top_values, shap_vals, top_indices, 
                      X_enc_dense, feature_names, model_name, n_show):
    """Create and save SHAP plots"""
    
    # Color mapping for different models
    color_map = {
        'randomforest': 'forestgreen',
        'forest': 'forestgreen',
        'tree': 'darkgreen', 
        'xgb': 'orange',
        'logistic': 'blue',
        'mlp': 'purple'
    }
    
    color = color_map.get(model_name, 'skyblue')
    
    # 1. Feature importance bar plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))
    
    bars = plt.barh(y_pos, top_values, color=color, alpha=0.8)
    plt.yticks(y_pos, top_features)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title(f'Top {n_show} Most Important Features - {model_name.upper()} Model (SHAP)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_values)):
        plt.text(bar.get_width() + max(top_values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    importance_filename = f'shap_importance_{model_name}.png'
    plt.savefig(importance_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"SHAP importance plot saved as '{importance_filename}'")
    
    # 2. Try to create official SHAP summary plot
    try:
        sample_size = min(50, X_enc_dense.shape[0])
        plt.figure(figsize=(10, 6))
        
        # Create DataFrame for better plotting
        top_X_df = pd.DataFrame(
            X_enc_dense[:sample_size, top_indices], 
            columns=top_features
        )
        
        shap.summary_plot(
            shap_vals[:sample_size, top_indices], 
            top_X_df,
            plot_type="bar",
            max_display=n_show,
            show=False
        )
        plt.title(f'SHAP Summary - Top {n_show} Features ({model_name.upper()})')
        plt.tight_layout()
        
        summary_filename = f'shap_summary_{model_name}.png'
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"SHAP summary plot saved as '{summary_filename}'")
        
    except Exception as e:
        print(f"Official SHAP summary plot failed: {e}")
    
    # 3. Try SHAP waterfall plot for first instance
    try:
        plt.figure(figsize=(10, 8))
        
        # Create explanation object for waterfall plot
        if hasattr(shap, 'Explanation'):
            # For newer SHAP versions
            explanation = shap.Explanation(
                values=shap_vals[0, top_indices],
                base_values=0,  # or explainer.expected_value if available
                data=X_enc_dense[0, top_indices],
                feature_names=top_features
            )
            shap.waterfall_plot(explanation, show=False)
        else:
            # Fallback for older versions
            print("Waterfall plot requires newer SHAP version")
            
        plt.title(f'SHAP Waterfall - First Instance ({model_name.upper()})')
        plt.tight_layout()
        
        waterfall_filename = f'shap_waterfall_{model_name}.png'
        plt.savefig(waterfall_filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"SHAP waterfall plot saved as '{waterfall_filename}'")
        
    except Exception as e:
        print(f"SHAP waterfall plot failed: {e}")

def _create_fallback_plot(top_features, top_values, model_name):
    """Create plot for fallback feature importance"""
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos, top_values, color='lightcoral', alpha=0.8)
    plt.yticks(y_pos, top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {len(top_features)} Features - {model_name.upper()} Model (Built-in Importance)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (pos, val) in enumerate(zip(y_pos, top_values)):
        plt.text(val + max(top_values)*0.01, pos, f'{val:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    filename = f'feature_importance_{model_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved as '{filename}'")

def _get_fallback_importance(model, feature_names, max_features=5):
    """Get feature importance using model-specific methods"""
    
    try:
        # Tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-max_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = importances[top_indices]
            return top_features, top_values
        
        # Linear models have coef_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                coef = model.coef_
            else:
                # For binary classification, take first class coefficients
                coef = model.coef_[0]
            
            # Use absolute values for importance
            importances = np.abs(coef)
            top_indices = np.argsort(importances)[-max_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = importances[top_indices]
            return top_features, top_values
        
        else:
            return None, None
            
    except Exception as e:
        print(f"Failed to extract fallback importance: {e}")
        return None, None

# Convenience function for easy usage
def analyze_model(model_path, X_sample, max_features=5):
    """Load and analyze any saved model"""
    import joblib
    
    trained_pipeline = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    return shap_summary(trained_pipeline, X_sample, max_features)