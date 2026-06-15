"""
Создание сабмита: blend CatBoost + Ridge + LightGBM на тюненных параметрах.

Веса blend — ml/config.yaml (через cfg.blend).
Параметры моделей — outputs/ml/best_params.json (если нет — дефолты).
"""

from __future__ import annotations

import warnings

import bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from config import cfg
from data import load_data
from features import preprocess
from ml.models import (
    train_catboost_predict,
    train_lightgbm_predict,
    train_ridge_predict,
)
from ml.train_config import load_best_params

warnings.filterwarnings('ignore', message=".*select_dtypes.*str.*")

BLEND_WEIGHTS = {k: float(v) for k, v in cfg.blend.weights.items()}


def create_submission():
    """Blend CatBoost + Ridge + LightGBM → submission.csv."""
    print("=== СОЗДАНИЕ САБМИТА (blend: CatBoost + Ridge + LightGBM) ===")
    print(
        f"Веса blend: CatBoost={BLEND_WEIGHTS['catboost']:.3f}, "
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
    cb_pred_log = train_catboost_predict(X_train, y_log, X_test, cb_params)

    print("[2/3] Обучение Ridge (через препроцессор)...")
    ridge_pred_log = train_ridge_predict(X_train, y_log, X_test, ridge_params)

    print("[3/3] Обучение LightGBM (native)...")
    lgb_pred_log = train_lightgbm_predict(X_train, y_log, X_test, lgb_params)

    blend_pred_log = (
        BLEND_WEIGHTS['catboost'] * cb_pred_log
        + BLEND_WEIGHTS['ridge'] * ridge_pred_log
        + BLEND_WEIGHTS['lightgbm'] * lgb_pred_log
    )
    y_pred = np.expm1(blend_pred_log)

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
    submission.to_csv(cfg.paths.submission, index=False)

    print(f"\nГотово! Файл '{cfg.paths.submission}' сохранён ({len(submission)} строк).")
    print(
        f"Sale price: min={y_pred.min():.0f}, "
        f"median={np.median(y_pred):.0f}, max={y_pred.max():.0f}"
    )


if __name__ == "__main__":
    create_submission()
