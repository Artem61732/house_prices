"""
Создание сабмита: blend CatBoost + Ridge + LightGBM на тюненных параметрах.

Веса блeнда и random_state читаются из config.yaml.
Параметры моделей — из best_params.json (если файла нет — используются дефолты).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from config import cfg
from features import preprocess, prepare_for_catboost, prepare_for_lightgbm
from main import get_preprocessor, load_data


warnings.filterwarnings('ignore', message=".*select_dtypes.*str.*")

PARAMS_PATH = Path(__file__).parent / "best_params.json"
RANDOM_STATE = int(cfg.random_state)
BLEND_WEIGHTS = {k: float(v) for k, v in cfg.blend.weights.items()}


def load_best_params() -> tuple[dict, dict, dict]:
    if not PARAMS_PATH.exists():
        print("best_params.json не найден — использую дефолтные параметры")
        return {}, {}, {}
    payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    cb = payload.get('catboost', {})
    rg = payload.get('ridge', {})
    lg = payload.get('lightgbm', {})
    if 'catboost_cv_rmsle' in payload:
        print(f"CatBoost: тюненные (CV RMSLE = {payload['catboost_cv_rmsle']:.4f})")
    if 'ridge_cv_rmsle' in payload:
        print(f"Ridge:    тюненный  (CV RMSLE = {payload['ridge_cv_rmsle']:.4f})")
    if 'lightgbm_cv_rmsle' in payload:
        print(f"LightGBM: тюненные (CV RMSLE = {payload['lightgbm_cv_rmsle']:.4f})")
    return cb, rg, lg


def _train_catboost(X_train, y_log, X_test, params):
    X_tr, cat_features = prepare_for_catboost(X_train)
    X_te, _ = prepare_for_catboost(X_test)
    kwargs = {'random_state': RANDOM_STATE, 'verbose': False, **params}
    model = CatBoostRegressor(cat_features=cat_features, **kwargs)
    model.fit(X_tr, y_log)
    return model.predict(X_te)


def _train_ridge(X_train, y_log, X_test, params):
    numeric = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical = X_train.select_dtypes(include='object').columns
    pipe = Pipeline([
        ('preprocessor', get_preprocessor(numeric, categorical)),
        ('model', Ridge(random_state=RANDOM_STATE, **params)),
    ])
    pipe.fit(X_train, y_log)
    return pipe.predict(X_test)


def _train_lightgbm(X_train, y_log, X_test, params):
    X_tr, _ = prepare_for_lightgbm(X_train)
    X_te, _ = prepare_for_lightgbm(X_test)
    kwargs = {'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1, **params}
    model = LGBMRegressor(**kwargs)
    model.fit(X_tr, y_log)
    return model.predict(X_te)


def create_submission():
    print("=== СОЗДАНИЕ САБМИТА (blend: CatBoost + Ridge + LightGBM) ===")
    print(
        f"Веса блeнда: CatBoost={BLEND_WEIGHTS['catboost']:.3f}, "
        f"Ridge={BLEND_WEIGHTS['ridge']:.3f}, "
        f"LightGBM={BLEND_WEIGHTS['lightgbm']:.3f}"
    )

    X_train_raw, y_train, test_data = load_data()
    test_ids = test_data['Id']

    skew_threshold = float(cfg.preprocess.skew_threshold)
    X_train, skewed_cols = preprocess(X_train_raw, skew_threshold=skew_threshold)
    X_test, _ = preprocess(test_data, skewed_cols=skewed_cols)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")

    y_log = np.log1p(y_train)
    cb_params, ridge_params, lgb_params = load_best_params()

    print("\n[1/3] Обучение CatBoost (native)...")
    cb_pred_log = _train_catboost(X_train, y_log, X_test, cb_params)

    print("[2/3] Обучение Ridge (через препроцессор)...")
    ridge_pred_log = _train_ridge(X_train, y_log, X_test, ridge_params)

    print("[3/3] Обучение LightGBM (native)...")
    lgb_pred_log = _train_lightgbm(X_train, y_log, X_test, lgb_params)

    blend_pred_log = (
        BLEND_WEIGHTS['catboost'] * cb_pred_log
        + BLEND_WEIGHTS['ridge'] * ridge_pred_log
        + BLEND_WEIGHTS['lightgbm'] * lgb_pred_log
    )
    y_pred = np.expm1(blend_pred_log)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
    submission_path = getattr(cfg.paths, 'submission', 'submission.csv')
    submission.to_csv(submission_path, index=False)

    print(f"\nГотово! Файл '{submission_path}' сохранён ({len(submission)} строк).")
    print(f"Sale price: min={y_pred.min():.0f}, "
          f"median={np.median(y_pred):.0f}, max={y_pred.max():.0f}")


if __name__ == "__main__":
    create_submission()
