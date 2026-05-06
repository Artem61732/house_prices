import json
from pathlib import Path

import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from config import cfg
from features import (
    feature_engineering,
    fill_na_domain,
    get_skewed_columns,
    log_skewed_features,
    prepare_for_catboost,
    prepare_for_lightgbm,
)
from main import get_preprocessor, load_data


PARAMS_PATH = Path(__file__).parent / "best_params.json"

BLEND_WEIGHTS = {
    'catboost': 0.30,
    'ridge':    0.45,
    'lightgbm': 0.25,
}


def load_best_params() -> tuple[dict, dict, dict]:
    if not PARAMS_PATH.exists():
        print("best_params.json не найден — использую дефолтные параметры")
        return {}, {}, {}
    payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    cb_params = payload.get('catboost', {})
    ridge_params = payload.get('ridge', {})
    lgb_params = payload.get('lightgbm', {})
    if 'catboost_cv_rmsle' in payload:
        print(f"CatBoost: тюненные (CV RMSLE = {payload['catboost_cv_rmsle']:.4f})")
    if 'ridge_cv_rmsle' in payload:
        print(f"Ridge:    тюненный  (CV RMSLE = {payload['ridge_cv_rmsle']:.4f})")
    if 'lightgbm_cv_rmsle' in payload:
        print(f"LightGBM: тюненные (CV RMSLE = {payload['lightgbm_cv_rmsle']:.4f})")
    return cb_params, ridge_params, lgb_params


def create_submission():
    print("=== СОЗДАНИЕ САБМИТА (blend: CatBoost + Ridge + LightGBM) ===")
    print(
        f"Веса блeнда: CatBoost={BLEND_WEIGHTS['catboost']:.3f}, "
        f"Ridge={BLEND_WEIGHTS['ridge']:.3f}, "
        f"LightGBM={BLEND_WEIGHTS['lightgbm']:.3f}"
    )

    X_train_full, y_train_full, test_data = load_data()
    test_ids = test_data['Id']

    X_train_full = fill_na_domain(X_train_full)
    X_train_full = feature_engineering(X_train_full)

    X_test = fill_na_domain(test_data)
    X_test = feature_engineering(X_test)

    skewed_cols = get_skewed_columns(X_train_full, threshold=0.75)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")
    X_train_full = log_skewed_features(X_train_full, skewed_cols)
    X_test = log_skewed_features(X_test, skewed_cols)

    y_log = np.log1p(y_train_full)

    cb_params, ridge_params, lgb_params = load_best_params()

    print("\n[1/3] Обучение CatBoost (native)...")
    X_train_cat, cat_features = prepare_for_catboost(X_train_full)
    X_test_cat, _ = prepare_for_catboost(X_test)
    cb_kwargs = {'random_state': 42, 'verbose': False, **cb_params}
    cb_model = CatBoostRegressor(cat_features=cat_features, **cb_kwargs)
    cb_model.fit(X_train_cat, y_log)
    cb_pred_log = cb_model.predict(X_test_cat)

    print("[2/3] Обучение Ridge (через препроцессор)...")
    numeric_features = X_train_full.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train_full.select_dtypes(include=['object']).columns
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    ridge_kwargs = {'random_state': 42, **ridge_params}
    ridge_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(**ridge_kwargs)),
    ])
    ridge_pipe.fit(X_train_full, y_log)
    ridge_pred_log = ridge_pipe.predict(X_test)

    print("[3/3] Обучение LightGBM (native)...")
    X_train_lgb, _ = prepare_for_lightgbm(X_train_full)
    X_test_lgb, _ = prepare_for_lightgbm(X_test)
    lgb_kwargs = {'random_state': 42, 'verbose': -1, 'n_jobs': -1, **lgb_params}
    lgb_model = LGBMRegressor(**lgb_kwargs)
    lgb_model.fit(X_train_lgb, y_log)
    lgb_pred_log = lgb_model.predict(X_test_lgb)

    blend_pred_log = (
        BLEND_WEIGHTS['catboost'] * cb_pred_log
        + BLEND_WEIGHTS['ridge'] * ridge_pred_log
        + BLEND_WEIGHTS['lightgbm'] * lgb_pred_log
    )
    y_pred_dollars = np.expm1(blend_pred_log)

    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': y_pred_dollars,
    })

    submission_path = getattr(cfg.paths, 'submission', 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nГотово! Файл '{submission_path}' сохранён ({len(submission)} строк).")
    print(f"Sale price: min={y_pred_dollars.min():.0f}, "
          f"median={np.median(y_pred_dollars):.0f}, max={y_pred_dollars.max():.0f}")


if __name__ == "__main__":
    create_submission()
