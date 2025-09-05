from pathlib import Path
SEED = 42
DATA_DIR = Path('data')
RAW_CSV = DATA_DIR / 'bank-additional-full.csv'  # semicolon-separated
TARGET = 'y'
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # of the train split
