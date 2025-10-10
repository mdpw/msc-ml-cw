import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import os
import glob
import mlflow

def shap_summary(trained_pipeline, X_sample, max_features=5, save_dir=None):
    """
    Universal SHAP function that works with any model type.
    
    Args:
        trained_pipeline: Trained sklearn pipeline with 'pre' and 'clf' steps
        X_sample: Sample data for SHAP analysis  
        max_features: Number of top features to display
        save_dir: Directory to save plots (if None, saves to current directory)
    
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
        
        # Create and save plots
        plot_files = _create_shap_plots(top_features, top_values, shap_vals, top_indices, 
                          X_enc_dense, feature_names, model_name, n_show, save_dir)
        
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
            _create_fallback_plot(top_features, top_values, model_name, save_dir)
        else:
            print("No feature importance method available for this model")
        
        return top_features, top_values, model_name

def run_shap_analysis_with_mlflow(best_model, best_model_name, best_model_info, best_auc, X_val, out_dir):
    """
    Run SHAP analysis on the best model and log results to MLflow.
    
    Args:
        best_model: Trained pipeline (best performing model)
        best_model_name: String name of the best model
        best_model_info: Dictionary with model metadata
        best_auc: Best AUC score achieved
        X_val: Validation dataset for SHAP analysis
        out_dir: Output directory for saving artifacts
    
    Returns:
        dict: SHAP analysis results summary
    """
    print("\nGenerating SHAP explanations for best model...")
    
    # Start a new MLflow run specifically for SHAP analysis
    with mlflow.start_run(run_name=f"{best_model_name}_shap_analysis"):
        try:
            # Generate SHAP analysis with proper save directory
            top_features, scores, model_type = shap_summary(
                best_model, 
                X_val.sample(min(500, len(X_val)), random_state=42), 
                max_features=5,
                save_dir=str(out_dir)
            )
            
            # Log SHAP results as metrics and parameters
            log_shap_results_to_mlflow(top_features, scores, model_type, X_val)
            
            # Log SHAP artifacts to MLflow
            log_shap_artifacts_to_mlflow(model_type, out_dir)
            
            # Create and log summary
            create_and_log_shap_summary(
                model_type, best_model_name, best_auc, best_model_info, 
                top_features, scores, X_val, out_dir
            )
            
            print(f"SHAP analysis completed for {model_type}")
            print("Top 5 features:")
            for i, (feature, score) in enumerate(zip(top_features, scores)):
                print(f"  {i+1}. {feature}: {score:.4f}")
            
            return {
                'status': 'success',
                'model_type': model_type,
                'top_features': top_features,
                'importance_scores': scores
            }
                
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            
            # Handle SHAP failure with fallback
            fallback_result = handle_shap_failure(
                e, best_model, best_model_info, out_dir
            )
            
            return fallback_result

def log_shap_results_to_mlflow(top_features, scores, model_type, X_val):
    """Log SHAP results as MLflow metrics and parameters"""
    
    # Log feature importance scores as metrics (for searching/filtering)
    for i, (feature, score) in enumerate(zip(top_features, scores)):
        mlflow.log_metric(f"shap_feature_{i+1}_importance", score)
        mlflow.log_param(f"shap_feature_{i+1}_name", feature)
    
    # Log overall SHAP metadata
    mlflow.log_param("shap_model_type", model_type)
    mlflow.log_param("shap_sample_size", min(500, len(X_val)))
    mlflow.log_param("shap_num_features", len(top_features))
    mlflow.log_param("shap_status", "success")

def log_shap_artifacts_to_mlflow(model_type, out_dir):
    """Find and log SHAP plot artifacts to MLflow"""
    
    # Find and log SHAP plots created in the artifacts directory
    shap_plots = glob.glob(os.path.join(str(out_dir), f"shap_*_{model_type}.png"))
    for plot_path in shap_plots:
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path="shap_plots")
            print(f"Logged {plot_path} to MLflow")
    
    # Also log feature importance plot if it exists
    feature_plots = glob.glob(os.path.join(str(out_dir), f"feature_importance_{model_type}.png"))
    for plot_path in feature_plots:
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path="shap_plots")
            print(f"Logged {plot_path} to MLflow")

def create_and_log_shap_summary(model_type, best_model_name, best_auc, best_model_info, 
                                top_features, scores, X_val, out_dir):
    """Create and log SHAP summary text file"""
    
    shap_summary_path = out_dir / f"shap_summary_{model_type}.txt"
    with open(shap_summary_path, 'w') as f:
        f.write(f"SHAP Analysis Summary for {model_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Test AUC: {best_auc:.4f}\n")
        f.write(f"Test F1: {best_model_info['test_f1']:.4f}\n\n")
        f.write("Top 5 Most Important Features:\n")
        f.write("-" * 30 + "\n")
        for i, (feature, score) in enumerate(zip(top_features, scores)):
            f.write(f"{i+1:2d}. {feature:30s} {score:.4f}\n")
        f.write(f"\nSample size used: {min(500, len(X_val))}\n")
        f.write(f"Analysis method: SHAP with {model_type} explainer\n")
    
    mlflow.log_artifact(str(shap_summary_path), artifact_path="summaries")
    print(f"SHAP summary saved to MLflow: {shap_summary_path}")

def handle_shap_failure(error, best_model, best_model_info, out_dir):
    """Handle SHAP failure and try fallback methods"""
    
    # Log the failure
    mlflow.log_param("shap_status", "failed")
    mlflow.log_param("shap_error", str(error))
    
    # Try fallback to model feature importance
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        return create_fallback_analysis(best_model, best_model_info, out_dir)
    else:
        mlflow.log_param("fallback_method", "none_available")
        print("No feature importance available")
        return {
            'status': 'failed',
            'error': str(error),
            'fallback_available': False
        }

def create_fallback_analysis(best_model, best_model_info, out_dir):
    """Create fallback feature importance analysis"""
    
    model = best_model.named_steps['clf']
    pre = best_model.named_steps['pre']
    
    # Extract feature names
    feature_names = []
    for name, transformer, columns in pre.transformers_:
        if name == 'cat':
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                for i, col in enumerate(columns):
                    for cat in transformer.categories_[i]:
                        feature_names.append(f"{col}_{cat}")
        elif name == 'num':
            feature_names.extend(columns)
    
    # Get feature importance
    importances = model.feature_importances_
    top_5_indices = np.argsort(importances)[-5:][::-1]
    top_5_features = [feature_names[i] for i in top_5_indices]
    top_5_values = importances[top_5_indices]
    
    # Log fallback results to MLflow
    for i, (feature, score) in enumerate(zip(top_5_features, top_5_values)):
        mlflow.log_metric(f"fallback_feature_{i+1}_importance", score)
        mlflow.log_param(f"fallback_feature_{i+1}_name", feature)
    
    mlflow.log_param("fallback_method", "model_feature_importances")
    
    print("Top 5 Features (from model feature_importances_):")
    for i, (feature, value) in enumerate(zip(top_5_features, top_5_values)):
        print(f"{i+1}. {feature}: {value:.4f}")
    
    # Create fallback plot
    create_fallback_plot(top_5_features, top_5_values, best_model_info, out_dir)
    
    # Create fallback summary
    create_fallback_summary(top_5_features, top_5_values, best_model_info, out_dir)
    
    return {
        'status': 'fallback_success',
        'model_type': best_model_info['model_type'],
        'top_features': top_5_features,
        'importance_scores': top_5_values,
        'method': 'model_feature_importances'
    }

def create_fallback_plot(top_5_features, top_5_values, best_model_info, out_dir):
    """Create and save fallback feature importance plot"""
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_5_features))
    
    plt.barh(y_pos, top_5_values, color='lightcoral', alpha=0.8)
    plt.yticks(y_pos, top_5_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top 5 Features - {best_model_info["model_type"].upper()} Model (Built-in Importance)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (pos, val) in enumerate(zip(y_pos, top_5_values)):
        plt.text(val + max(top_5_values)*0.01, pos, f'{val:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save to artifacts directory
    fallback_plot_path = out_dir / f"feature_importance_{best_model_info['model_type']}.png"
    plt.savefig(fallback_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Log to MLflow
    mlflow.log_artifact(str(fallback_plot_path), artifact_path="shap_plots")
    print(f"Fallback plot saved to: {fallback_plot_path}")

def create_fallback_summary(top_5_features, top_5_values, best_model_info, out_dir):
    """Create and save fallback summary text file"""
    
    fallback_summary_path = out_dir / f"fallback_importance_{best_model_info['model_type']}.txt"
    with open(fallback_summary_path, 'w') as f:
        f.write(f"Feature Importance (Fallback) for {best_model_info['model_type'].upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("SHAP analysis failed, using model built-in feature importance\n\n")
        f.write("Top 5 Most Important Features:\n")
        f.write("-" * 30 + "\n")
        for i, (feature, score) in enumerate(zip(top_5_features, top_5_values)):
            f.write(f"{i+1:2d}. {feature:30s} {score:.4f}\n")
    
    mlflow.log_artifact(str(fallback_summary_path), artifact_path="summaries")
    print(f"Fallback summary saved to: {fallback_summary_path}")

# ===== SHAP PLOTTING FUNCTIONS =====

def _create_shap_plots(top_features, top_values, shap_vals, top_indices, 
                      X_enc_dense, feature_names, model_name, n_show, save_dir=None):
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
    plot_files = []
    
    # Helper function to get save path
    def get_save_path(filename):
        if save_dir:
            return os.path.join(save_dir, filename)
        return filename
    
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
    importance_filename = get_save_path(f'shap_importance_{model_name}.png')
    plt.savefig(importance_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    plot_files.append(importance_filename)
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
        
        summary_filename = get_save_path(f'shap_summary_{model_name}.png')
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(summary_filename)
        print(f"SHAP summary plot saved as '{summary_filename}'")
        
    except Exception as e:
        print(f"Official SHAP summary plot failed: {e}")
    
    return plot_files

def _create_fallback_plot(top_features, top_values, model_name, save_dir=None):
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
    
    if save_dir:
        filename = os.path.join(save_dir, f'feature_importance_{model_name}.png')
    else:
        filename = f'feature_importance_{model_name}.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
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

# ===== CONVENIENCE FUNCTIONS =====

def analyze_model(model_path, X_sample, max_features=5, save_dir=None):
    """Load and analyze any saved model"""
    import joblib
    
    trained_pipeline = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    return shap_summary(trained_pipeline, X_sample, max_features, save_dir)

def analyze_model_with_mlflow(model_path, X_sample, model_name="loaded_model", max_features=5, save_dir=None):
    """Load model and run full SHAP analysis with MLflow logging"""
    import joblib
    
    trained_pipeline = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Create mock model info for the function
    model_info = {
        'model_type': type(trained_pipeline.named_steps['clf']).__name__.lower(),
        'test_f1': 0.0  # Unknown for loaded model
    }
    
    # Use current directory if no save_dir provided
    if save_dir is None:
        save_dir = "."
    
    return run_shap_analysis_with_mlflow(
        trained_pipeline, model_name, model_info, 0.0, X_sample, save_dir
    )