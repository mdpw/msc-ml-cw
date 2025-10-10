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
from .explain import run_shap_analysis_with_mlflow

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smote', action='store_true', help='Apply SMOTE in training pipeline')
    ap.add_argument('--models', nargs='+', default=['logreg','rf','xgb','mlp'], help='Subset of models to run')
    ap.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis for faster training')
    return ap.parse_args()

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
            shap_results = run_shap_analysis_with_mlflow(
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