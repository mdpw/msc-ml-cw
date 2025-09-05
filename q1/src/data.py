import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from .config import RAW_CSV, TARGET, SEED, TEST_SIZE, VAL_SIZE

CAT_COLS = [
    'job','marital','education','default','housing','loan','contact',
    'month','day_of_week','poutcome'
]
NUM_COLS = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx',
            'cons.conf.idx','euribor3m','nr.employed']

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Feature engineering
    df['pdays_bucket'] = pd.cut(df['pdays'], bins=[-1,0,3,10,999, 1e9],
                                labels=['0','1-3','4-10','11-999','999+'])
    df['contact_last'] = (df['previous'] > 0).astype(int)
    df['campaign_intensity'] = df['campaign'] / (1 + df['previous'])
    return df

def load_splits(smote: bool=False):
    df = pd.read_csv(RAW_CSV, sep=';')
    df = engineer(df)
    y = (df[TARGET] == 'yes').astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=SEED, stratify=y_train
    )

    # Drop 'duration' to avoid leakage (only known post-call)
    for split in (X_tr, X_val, X_test):
        if 'duration' in split.columns:
            split.drop(columns=['duration'], inplace=True)

    cat_cols = [c for c in CAT_COLS + ['pdays_bucket'] if c in X_tr.columns]
    num_cols = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx',
                'cons.conf.idx','euribor3m','nr.employed','campaign_intensity','contact_last']

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(with_mean=False), num_cols)
    ])

    if smote:
        pipe = ImbPipeline([
            ('pre', preproc),
            ('sm', SMOTE(random_state=SEED))
        ])
    else:
        pipe = Pipeline([
            ('pre', preproc)
        ])

    return (X_tr, y_tr, X_val, y_val, X_test, y_test, pipe)
