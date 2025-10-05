# Energy Load Forecasting Project

Time series forecasting for energy load prediction using Prophet, LSTM, and Exponential Smoothing models.

## Quick Start

# 1. Setup
python -m venv energy-env
energy-env\Scripts\activate
pip install -r requirements.txt

### 2. Run Complete Pipeline
# Step 1: EDA and Preprocessing
python main_eda.py

# Step 2: Train and Evaluate Models
python train_models.py
