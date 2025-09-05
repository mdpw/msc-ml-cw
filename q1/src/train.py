import os
import argparse
import mlflow
# Explicitly tell MLflow to use local file storage
tracking_dir = "G:/My Drive/github/msc-ml-cw/q1/mlruns"
mlflow.set_tracking_uri(f"file:///{tracking_dir.replace(os.sep, '/')}")
# Also set a default artifacts location (important!)
mlflow.set_experiment("bank_marketing")

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from .data import load_splits
from .modeling import build_search, MODELS
from .evaluate import eval_at_threshold, best_f1_threshold, plot_and_save_curves, plot_and_save_confusion
from sklearn.metrics import roc_auc_score
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smote', action='store_true', help='Apply SMOTE in training pipeline')
    ap.add_argument('--models', nargs='+', default=['logreg','rf','xgb','lgbm','mlp'], help='Subset of models to run')
    return ap.parse_args()

def main():
    args = parse_args()
    X_tr, y_tr, X_val, y_val, X_test, y_test, pre = load_splits(smote=args.smote)

    results = []
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)

    for key in args.models:
        if key not in MODELS:
            continue
        with mlflow.start_run(run_name=f"{key}{'_smote' if args.smote else ''}"):
            pipe = Pipeline([('pre', pre), ('clf', MODELS[key])])
            grid = build_search(key, Pipeline([('pre', pre), ('clf', MODELS[key])]))
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

            # Plots and confusion matrix
            roc_path, pr_path = plot_and_save_curves(y_test, y_test_prob, f'{key}', str(out_dir))
            y_test_pred = (y_test_prob >= th).astype(int)
            cm_path = plot_and_save_confusion(y_test, y_test_pred, f'{key}', str(out_dir))
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.log_artifact(cm_path)

            results.append({
                'model': key,
                'val_auc': roc_auc_score(y_val, y_val_prob),
                'val_pr_auc': pr_auc,
                'best_threshold': th,
                **{f'test_{m.lower()}': metrics[m] for m in ['AUC','F1','Precision','Recall','Accuracy']}
            })

            # Log model
            mlflow.sklearn.log_model(pipe_best, artifact_path='model')

    if results:
        df = pd.DataFrame(results).sort_values('test_auc', ascending=False)
        csv_path = out_dir / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

if __name__ == '__main__':
    main()
