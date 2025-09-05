import os
import joblib
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_URI = os.getenv('MODEL_URI', None)
app = FastAPI(title='Bank Marketing Classifier')

class Record(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

@app.on_event('startup')
async def load_model():
    global model
    if MODEL_URI:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    else:
        model = joblib.load('models/best_model.joblib')

@app.post('/predict')
async def predict(rec: Record):
    import pandas as pd
    df = pd.DataFrame([rec.dict()])
    # Engineer same features as training
    df['pdays_bucket'] = pd.cut(df['pdays'], bins=[-1,0,3,10,999, 1e9], labels=['0','1-3','4-10','11-999','999+'])
    df['contact_last'] = (df['previous'] > 0).astype(int)
    df['campaign_intensity'] = df['campaign'] / (1 + df['previous'])
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
    # If using mlflow.pyfunc model, predict() returns probabilities directly for classifiers
    try:
        proba = model.predict(df)
        if hasattr(proba, 'shape') and proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:,1]
    except Exception:
        proba = model.predict_proba(df)[:,1]
    return {'probability_yes': float(proba[0])}
