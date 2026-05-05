import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if 'GrLivArea' in train_data.columns:
        train_data = train_data[train_data['GrLivArea'] < 4000]

    X = train_data.drop('SalePrice', axis=1)
    y = train_data['SalePrice']
    
    return X, y, test_data

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)