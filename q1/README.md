# Bank Marketing ML Assignment Starter

This repository contains a full scaffold to complete your assignment end-to-end on the UCI Bank Marketing dataset.

## Quick start
1. Download `bank-additional-full.csv` (semicolon-separated) from the UCI repository and put it into `data/`.
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. (Optional) Launch MLflow UI in a separate terminal:
   ```bash
   mlflow ui
   ```
4. Train models (logs to MLflow and saves results/plots as artifacts):
   ```bash
   python -m src.train --models logreg rf xgb lgbm mlp
   # Try SMOTE:
   python -m src.train --smote --models xgb lgbm rf
   ```
5. Serve the best model (after choosing one and exporting it to `models/best_model.joblib` or using MLflow Model URI):
   ```bash
   uvicorn src.serve:app --reload
   ```

## Notes
- We drop `duration` to avoid label leakage (the call length is known only after the call).
- We use stratified splits, threshold tuning, and PR-AUC/F1 emphasis due to class imbalance.
