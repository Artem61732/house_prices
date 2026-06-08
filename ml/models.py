"""Фабрики моделей и sklearn-препроцессор для ML-пайплайна."""

from __future__ import annotations

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


def get_sklearn_models(ridge_params: dict | None = None):
    """Модели через общий препроцессор (impute + scale + OHE)."""
    ridge_kwargs = {'random_state': RANDOM_STATE, **(ridge_params or {})}
    return {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(**ridge_kwargs),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE),
    }


def get_catboost_model(cat_features, params: dict | None = None):
    """CatBoost в нативном режиме (без OHE, с явным cat_features)."""
    kwargs = {'random_state': RANDOM_STATE, 'verbose': False, **(params or {})}
    return CatBoostRegressor(cat_features=cat_features, **kwargs)


def get_lightgbm_model(params: dict | None = None):
    """LightGBM в нативном режиме (pandas category колонки auto-detect)."""
    kwargs = {'random_state': RANDOM_STATE, 'verbose': -1, **(params or {})}
    return LGBMRegressor(**kwargs)
