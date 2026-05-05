# src/evaluate.py
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    root_mean_squared_error, 
    r2_score, 
    root_mean_squared_log_error
)

def evaluate_model(model_name, y_true, y_pred):
    """
    Считает и выводит метрики качества для переданной модели.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Для RMSLE предсказания должны быть неотрицательными 
    # (иногда линейные модели могут выдавать значения < 0)
    y_pred_positive = np.maximum(y_pred, 0)
    rmsle = root_mean_squared_log_error(y_true, y_pred_positive)
    
    # Красивый вывод в консоль
    print(f"--- {model_name} ---")
    print(f'MAE:   {mae:.2f}')
    print(f'RMSE:  {rmse:.2f}')
    print(f'R2:    {r2:.4f}')
    print(f'RMSLE: {rmsle:.4f}\n')
    
    # Возвращаем метрики в виде словаря (пригодится, если захотим сохранить их в таблицу/файл)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'RMSLE': rmsle
    }