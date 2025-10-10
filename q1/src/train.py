import os
from pathlib import Path
import argparse
import mlflow
# Explicitly tell MLflow to use local file storage
mlruns_path = Path("mlruns").absolute()
mlflow.set_tracking_uri(f"file:///{str(mlruns_path).replace(os.sep, '/')}")
# Also set a default artifacts location (important!)
mlflow.set_experiment("bank_marketing")
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from .data import load_splits
from .modeling import build_search, MODELS
from .evaluate import eval_at_threshold, best_f1_threshold, plot_and_save_curves, plot_and_save_confusion
from sklearn.metrics import roc_auc_score
import joblib
from .explain import shap_summary

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smote', action='store_true', help='Apply SMOTE in training pipeline')
    ap.add_argument('--models', nargs='+', default=['logreg','rf','xgb','mlp'], help='Subset of models to run')
    ap.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis for faster training')
    return ap.parse_args()

def run_shap_analysis(best_model, best_model_name, best_model_info, best_auc, X_val, out_dir):
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
            _log_shap_results_to_mlflow(top_features, scores, model_type, X_val)
            
            # Log SHAP artifacts to MLflow
            _log_shap_artifacts_to_mlflow(model_type, out_dir)
            
            # Create and log summary
            _create_and_log_shap_summary(
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
            fallback_result = _handle_shap_failure(
                e, best_model, best_model_info, out_dir
            )
            
            return fallback_result

def _log_shap_results_to_mlflow(top_features, scores, model_type, X_val):
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

def _log_shap_artifacts_to_mlflow(model_type, out_dir):
    """Find and log SHAP plot artifacts to MLflow"""
    
    import glob
    import os
    
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

def _create_and_log_shap_summary(model_type, best_model_name, best_auc, best_model_info, 
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

def _handle_shap_failure(error, best_model, best_model_info, out_dir):
    """Handle SHAP failure and try fallback methods"""
    
    # Log the failure
    mlflow.log_param("shap_status", "failed")
    mlflow.log_param("shap_error", str(error))
    
    # Try fallback to model feature importance
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        return _create_fallback_analysis(best_model, best_model_info, out_dir)
    else:
        mlflow.log_param("fallback_method", "none_available")
        print("No feature importance available")
        return {
            'status': 'failed',
            'error': str(error),
            'fallback_available': False
        }

def _create_fallback_analysis(best_model, best_model_info, out_dir):
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
    _create_fallback_plot(top_5_features, top_5_values, best_model_info, out_dir)
    
    # Create fallback summary
    _create_fallback_summary(top_5_features, top_5_values, best_model_info, out_dir)
    
    return {
        'status': 'fallback_success',
        'model_type': best_model_info['model_type'],
        'top_features': top_5_features,
        'importance_scores': top_5_values,
        'method': 'model_feature_importances'
    }

def _create_fallback_plot(top_5_features, top_5_values, best_model_info, out_dir):
    """Create and save fallback feature importance plot"""
    
    import matplotlib.pyplot as plt
    
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

def _create_fallback_summary(top_5_features, top_5_values, best_model_info, out_dir):
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

def main():
    args = parse_args()
    X_tr, y_tr, X_val, y_val, X_test, y_test, pipe_base = load_splits(smote=args.smote)

    results = []
    
    # Create separate output directories based on SMOTE usage
    suffix = '_smote' if args.smote else '_no_smote'
    out_dir = Path(f'artifacts{suffix}')
    out_dir.mkdir(exist_ok=True)
    
    models_dir = Path(f'models{suffix}')
    models_dir.mkdir(exist_ok=True)
    
    # Track best model across all experiments
    best_model = None
    best_auc = 0
    best_model_name = ""
    best_model_info = {}

    for key in args.models:
        if key not in MODELS:
            print(f"Warning: Unknown model '{key}', skipping...")
            continue
            
        print(f"Training {key}{'_smote' if args.smote else ''} ({args.models.index(key)+1}/{len(args.models)})...")
        
        with mlflow.start_run(run_name=f"{key}{'_smote' if args.smote else ''}"):
            # Create the complete pipeline using the base pipeline type
            if args.smote:
                # Use imblearn Pipeline
                from imblearn.pipeline import Pipeline as ImbPipeline
                full_pipe = ImbPipeline(pipe_base.steps + [('clf', MODELS[key])])
            else:
                # Use sklearn Pipeline  
                from sklearn.pipeline import Pipeline
                full_pipe = Pipeline(pipe_base.steps + [('clf', MODELS[key])])
            
            # Build grid search with the correct pipeline
            grid = build_search(key, full_pipe)
            grid.fit(X_tr, y_tr)

            y_val_prob = grid.predict_proba(X_val)[:,1]
            th, best_f1, pr_auc = best_f1_threshold(y_val, y_val_prob)
            mlflow.log_metric('best_val_f1', best_f1)
            mlflow.log_metric('val_pr_auc', pr_auc)
            mlflow.log_metric('val_auc', roc_auc_score(y_val, y_val_prob))
            mlflow.log_param('best_threshold', th)
            mlflow.log_param('smote', args.smote)
            mlflow.log_param('best_params', grid.best_params_)

            # Final refit is already done in .best_estimator_
            pipe_best = grid.best_estimator_
            y_test_prob = pipe_best.predict_proba(X_test)[:,1]

            # Metrics at tuned threshold
            metrics = eval_at_threshold(y_test, y_test_prob, thresh=th)
            for k,v in metrics.items():
                mlflow.log_metric(f'test_{k.lower()}', v)

            # Plots and confusion matrix - save with suffix in filename
            roc_path, pr_path = plot_and_save_curves(y_test, y_test_prob, f'{key}{suffix}', str(out_dir))
            y_test_pred = (y_test_prob >= th).astype(int)
            cm_path = plot_and_save_confusion(y_test, y_test_pred, f'{key}{suffix}', str(out_dir))
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.log_artifact(cm_path)

            # Check if this is the best model so far
            test_auc = metrics['AUC']
            if test_auc > best_auc:
                best_auc = test_auc
                best_model = pipe_best
                best_model_name = f"{key}{'_smote' if args.smote else ''}"
                best_model_info = {
                    'model_type': key,
                    'smote': args.smote,
                    'test_auc': test_auc,
                    'test_f1': metrics['F1'],
                    'threshold': th,
                    'best_params': grid.best_params_
                }

            results.append({
                'model': key,
                'smote': args.smote,
                'val_auc': roc_auc_score(y_val, y_val_prob),
                'val_pr_auc': pr_auc,
                'best_threshold': th,
                **{f'test_{m.lower()}': metrics[m] for m in ['AUC','F1','Precision','Recall','Accuracy']}
            })

            # Log model to MLflow
            # Save as pickle artifact (works for both sklearn and imblearn)
            import pickle
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model_file = os.path.join(tmpdir, 'model.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(pipe_best, f)
                mlflow.log_artifact(model_file, artifact_path='model')
            
            print(f"  → Test AUC: {test_auc:.3f}, F1: {metrics['F1']:.3f}")

    # Save the best model for serving
    if best_model is not None:
        model_path = models_dir / 'best_model.joblib'
        joblib.dump(best_model, model_path)
        
        # Save model info for reference
        info_path = models_dir / 'best_model_info.json'
        import json
        with open(info_path, 'w') as f:
            json.dump(best_model_info, f, indent=2, default=str)
        
        print(f"\nBest model saved:")
        print(f"  Model: {best_model_name}")
        print(f"  Path: {model_path}")
        print(f"  Test AUC: {best_auc:.3f}")
        print(f"  Test F1: {best_model_info['test_f1']:.3f}")

        # Run SHAP analysis (unless skipped)
        if not args.skip_shap:
            shap_results = run_shap_analysis(
                best_model, best_model_name, best_model_info, 
                best_auc, X_val, out_dir
            )
            print(f"SHAP analysis result: {shap_results['status']}")
        else:
            print("SHAP analysis skipped (--skip-shap flag used)")

    else:
        print("\nNo models were trained successfully")

    # Save results summary with suffix
    if results:
        df = pd.DataFrame(results).sort_values('test_auc', ascending=False)
        csv_path = out_dir / f'results_summary{suffix}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\nResults summary saved to: {csv_path}")
        print("\nTop 3 models by AUC:")
        print(df[['model', 'smote', 'test_auc', 'test_f1']].head(3).to_string(index=False))

if __name__ == '__main__':
    main()