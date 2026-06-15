"""Фабрики моделей и sklearn-препроцессор для ML-пайплайна."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import cfg
from features import prepare_for_catboost, prepare_for_lightgbm

RANDOM_STATE = int(cfg.random_state)


def get_preprocessor(numeric_features, categorical_features):
    """ColumnTransformer для числовых и категориальных признаков (Ridge и sklearn-модели)."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=float('nan'), strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])


def get_sklearn_models(ridge_params: dict | None = None, *, quick: bool = False):
    """Модели через общий препроцессор (impute + scale + OHE)."""
    ridge_kwargs = {'random_state': RANDOM_STATE, **(ridge_params or {})}
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(**ridge_kwargs),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE),
    }
    if quick:
        return {k: models[k] for k in ('Linear Regression', 'Ridge')}
    return models


def get_catboost_model(cat_features, params: dict | None = None):
    """CatBoost в нативном режиме (без OHE, с явным cat_features)."""
    kwargs = {'random_state': RANDOM_STATE, 'verbose': False, **(params or {})}
    return CatBoostRegressor(cat_features=cat_features, **kwargs)


def get_lightgbm_model(params: dict | None = None):
    """LightGBM в нативном режиме (pandas category колонки auto-detect)."""
    kwargs = {'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1, **(params or {})}
    return LGBMRegressor(**kwargs)


def _feature_columns(X: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    numeric = X.select_dtypes(include=['int64', 'float64']).columns
    categorical = X.select_dtypes(include='object').columns
    return numeric, categorical


def train_catboost_predict(
    X_train: pd.DataFrame,
    y_log,
    X_test: pd.DataFrame,
    params: dict | None = None,
) -> np.ndarray:
    """Обучает CatBoost на train и предсказывает log-таргет на test."""
    X_tr, cat_features = prepare_for_catboost(X_train)
    X_te, _ = prepare_for_catboost(X_test)
    model = get_catboost_model(cat_features, params)
    model.fit(X_tr, y_log)
    return model.predict(X_te)


def train_ridge_predict(
    X_train: pd.DataFrame,
    y_log,
    X_test: pd.DataFrame,
    params: dict | None = None,
) -> np.ndarray:
    """Обучает Ridge (с препроцессором) на train и предсказывает log-таргет на test."""
    numeric, categorical = _feature_columns(X_train)
    pipe = Pipeline([
        ('preprocessor', get_preprocessor(numeric, categorical)),
        ('model', Ridge(random_state=RANDOM_STATE, **(params or {}))),
    ])
    pipe.fit(X_train, y_log)
    return pipe.predict(X_test)


def train_lightgbm_predict(
    X_train: pd.DataFrame,
    y_log,
    X_test: pd.DataFrame,
    params: dict | None = None,
) -> np.ndarray:
    """Обучает LightGBM на train и предсказывает log-таргет на test."""
    X_tr, _ = prepare_for_lightgbm(X_train)
    X_te, _ = prepare_for_lightgbm(X_test)
    model = get_lightgbm_model(params)
    model.fit(X_tr, y_log)
    return model.predict(X_te)
