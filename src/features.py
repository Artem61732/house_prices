import pandas as pd

def feature_engineering(data):
    """
    Добавляет новые признаки и удаляет ненужные
    """
    # Делаем копию, чтобы случайно не изменить оригинальные данные
    df = data.copy()
    
    # Удаляем Id, так как он не нужен для предсказаний
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    # Создаем новый признак TotalSF (общая площадь)
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    df['TotalBath'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']  
    return df