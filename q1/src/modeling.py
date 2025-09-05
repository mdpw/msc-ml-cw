import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from .config import SEED

MODELS = {
    'logreg': LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=SEED),
    'rf': RandomForestClassifier(n_estimators=600, random_state=SEED, n_jobs=-1),
    'xgb': XGBClassifier(random_state=SEED, n_estimators=800, tree_method='hist', eval_metric='logloss', use_label_encoder=False),
    'lgbm': LGBMClassifier(random_state=SEED, n_estimators=800),
    'mlp': MLPClassifier(random_state=SEED, max_iter=300)
}

PARAMS = {
    'logreg': { 'C':[0.1,1,10] },
    'rf': { 'max_depth':[6,12,None], 'min_samples_leaf':[1,5] },
    'xgb': { 'max_depth':[3,6], 'learning_rate':[0.01,0.05,0.1], 'subsample':[0.8,1.0], 'colsample_bytree':[0.8,1.0] },
    'lgbm': { 'max_depth':[-1,6,12], 'learning_rate':[0.01,0.05,0.1], 'num_leaves':[31,63,127] },
    'mlp': { 'hidden_layer_sizes':[(64,),(128,64)], 'alpha':[1e-4,1e-3] }
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def build_search(model_key, pipeline):
    model = MODELS[model_key]
    params = PARAMS[model_key]
    grid = GridSearchCV(estimator=pipeline.set_params(**{'clf': model}),
                        param_grid={f'clf__{k}': v for k,v in params.items()},
                        scoring='roc_auc', cv=CV, n_jobs=-1, refit=True, verbose=0)
    return grid
