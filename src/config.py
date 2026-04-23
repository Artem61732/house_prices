from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / 'data'

TRAIN_DATA_PATH = DATA_DIR / 'train.csv'
TEST_DATA_PATH = DATA_DIR / 'test.csv'