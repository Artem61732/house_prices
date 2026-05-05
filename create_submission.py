import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

from src.data import load_data
from src.features import feature_engineering
from src.preprocess import get_preprocessor

# 1. Загружаем данные
X_train_full, y_train_full, test_data = load_data()

# Сохраняем Id для сабмита
test_ids = test_data['Id']

# 2. Создаем новые признаки (Feature Engineering)
X_train_full = feature_engineering(X_train_full)
X_test = feature_engineering(test_data)

# Логарифмируем таргет
y_train_full = np.log1p(y_train_full)

# 3. Получаем списки числовых и категориальных колонок
numeric_features = X_train_full.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train_full.select_dtypes(include=['object']).columns

# 4. Собираем препроцессор
preprocessor = get_preprocessor(numeric_features, categorical_features)

# 5. Инициализируем лучшую модель (CatBoost показывает себя отлично)
model = CatBoostRegressor(random_state=42, verbose=False)

# 6. Собираем пайплайн
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# 7. Обучаем на ВСЕХ данных (без разбиения на train/test)
print("Обучение модели на всех данных...")
pipeline.fit(X_train_full, y_train_full)

# 8. Делаем предсказания на тестовой выборке
print("Создание предсказаний...")
y_pred_log = pipeline.predict(X_test)
y_pred_dollars = np.expm1(y_pred_log)

# 9. Собираем сабмит
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_dollars
})

# 10. Сохраняем в csv
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Готово! Файл '{submission_path}' сохранен. Теперь его можно загрузить на Kaggle.")