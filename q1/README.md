# Bank Marketing Starter

This repository contains a full scaffold to complete end-to-end on the UCI Bank Marketing dataset.

## Quick start
## 1. Add `bank-additional-full.csv` (semicolon-separated) from the UCI repository and put it into `data/`.
## 2. Create a virtual environment and install requirements:
   python -m venv banking-env </br>
   banking-env\Scripts\activate </br>
   pip install -r requirements.txt </br>

## 3. (Optional) Launch MLflow UI in a separate terminal:
   mlflow ui

## 4. Train models (logs to MLflow and saves results/plots as artifacts):
   python -m src.train --models logreg rf xgb lgbm mlp
   
## Try SMOTE:
   python -m src.train --smote --models xgb lgbm rf

## 5. Serve the best model (after choosing one and exporting it to `models/best_model.joblib` or using MLflow Model URI):
   uvicorn src.serve:app --reload

