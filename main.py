import numpy as np

from sklearn.pipeline import Pipeline
from src.data import load_data, split_data
from src.features import feature_engineering
from src.preprocess import get_preprocessor
from src.models import get_models
from src.evaluate import evaluate_model

# 1. Загружаем данные
X, y, test_data = load_data()

# 2. Создаем новые признаки (Feature Engineering)
X = feature_engineering(X)
y = np.log1p(y)
# 3. Разбиваем на обучающую и валидационную выборки
X_train, X_test, y_train, y_test = split_data(X, y)

# 4. Получаем списки числовых и категориальных колонок
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 5. Собираем препроцессор
preprocessor = get_preprocessor(numeric_features, categorical_features)

# 6. Получаем словарь с моделями
models = get_models()

# 7. Проходимся по каждой модели, обучаем и оцениваем
for model_name, model in models.items():
    
    # Собираем пайплайн для текущей модели
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучаем и предсказываем
    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)
    
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)
    # Оцениваем (функция сама все посчитает и выведет на экран)
    evaluate_model(model_name, y_test_dollars, y_pred_dollars)