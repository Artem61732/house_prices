"""Загрузка и подготовка данных House Prices (общий модуль для ML и DL)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import cfg
from features import preprocess, prepare_for_dl


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


def load_preprocessed_train_target(
    skew_threshold: float | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Загружает train, применяет preprocess и возвращает (X, y_log).

    y_log = log1p(SalePrice) — таргет для обучения и CV.
    На log-таргете RMSLE совпадает с RMSE.
    """
    X, y, _ = load_data(verbose=verbose)
    threshold = skew_threshold if skew_threshold is not None else float(cfg.preprocess.skew_threshold)
    X, _ = preprocess(X, skew_threshold=threshold)
    y_log = np.log1p(y).to_numpy()
    return X, y_log


def load_preprocessed_dl_train_target(
    skew_threshold: float | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """load_preprocessed_train_target + списки числовых и категориальных колонок для DNN."""
    X, y_log = load_preprocessed_train_target(
        skew_threshold=skew_threshold, verbose=verbose,
    )
    numeric_cols, categorical_cols = prepare_for_dl(X)
    return X, y_log, numeric_cols, categorical_cols
