"""Загрузка данных House Prices (общий модуль для ML и DL)."""

from __future__ import annotations

import pandas as pd

from config import cfg


def load_data(
    train_path=cfg.paths.train,
    test_path=cfg.paths.test,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Загружает train/test и удаляет классические выбросы GrLivArea."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if {'GrLivArea', 'SalePrice'} <= set(train_data.columns):
        before = len(train_data)
        outlier_mask = (
            (train_data['GrLivArea'] > 4000)
            & (train_data['SalePrice'] < 300000)
        )
        train_data = train_data.drop(train_data[outlier_mask].index)
        if verbose:
            print(f"Удалено выбросов GrLivArea: {before - len(train_data)} строк")

    X = train_data.drop('SalePrice', axis=1)
    y = train_data['SalePrice']
    return X, y, test_data
