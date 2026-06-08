"""
Публичный API параметров ML-моделей.

- load_tuned_params — чтение outputs/ml/best_params.json
- load_best_params  — то же для create_submission (с логированием CV RMSLE)
"""

from __future__ import annotations

import json

from paths import ML_BEST_PARAMS_PATH, ensure_output_dirs

ensure_output_dirs()
PARAMS_PATH = ML_BEST_PARAMS_PATH


def load_tuned_params() -> tuple[dict, dict, dict]:
    """CatBoost / Ridge / LightGBM params из best_params.json."""
    if not PARAMS_PATH.exists():
        return {}, {}, {}
    try:
        payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, {}, {}
    return (
        payload.get('catboost', {}),
        payload.get('ridge', {}),
        payload.get('lightgbm', {}),
    )


def load_best_params() -> tuple[dict, dict, dict]:
    """Как load_tuned_params, но с выводом CV RMSLE в консоль."""
    if not PARAMS_PATH.exists():
        print(f"best_params.json не найден ({PARAMS_PATH}) — использую дефолтные параметры")
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
