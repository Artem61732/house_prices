import pandas as pd
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH

def load_data():

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Размер тренировочной выборки: {train_df.shape}")
    print(f"Размер тестовой выборки: {test_df.shape}")

    return train_df, test_df



if __name__ == '__main__':
    train, test = load_data()
    print(train.head())
    print(test.head())