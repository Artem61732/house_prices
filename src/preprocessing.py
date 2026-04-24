import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_loader import load_data

def create_preprocessor(numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
      ])

    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

if __name__ == '__main__':
    # 1. Загружаем данные
    train_df, test_df = load_data()
    
    # 2. Отделяем целевую переменную (ответы) от признаков
    # ТВОЯ ЗАДАЧА: удали колонку 'SalePrice' из train_df с помощью метода .drop(columns=[...])
    X_train = train_df.drop(columns=['SalePrice'])
    
    # 3. Для теста возьмем только 4 колонки
    num_cols = ['LotArea', 'GrLivArea']
    cat_cols = ['Neighborhood', 'BldgType']
    
    # Оставляем в X_train только эти 4 колонки
    X_train_subset = X_train[num_cols + cat_cols]
    
    # 4. Создаем наш препроцессор
    preprocessor = create_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    
    # 5. Обучаем препроцессор и сразу трансформируем данные
    # Метод fit_transform вычисляет медианы, находит уникальные категории и применяет изменения
    processed_data = preprocessor.fit_transform(X_train_subset)
    
    # Распечатаем результат
    print(f"Размер исходных данных: {X_train_subset.shape}")
    print(f"Размер обработанных данных: {processed_data.shape}")
    
    # Выведем первую строчку обработанных данных
    print("Первая строка после обработки:")
    print(processed_data[0])