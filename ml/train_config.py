"""Параметры ML-моделей из outputs/ml/best_params.json."""

from __future__ import annotations

import json

from paths import ML_BEST_PARAMS_PATH, ensure_output_dirs

ensure_output_dirs()
PARAMS_PATH = ML_BEST_PARAMS_PATH


def _parse_payload(payload: dict) -> tuple[dict, dict, dict]:
    return (
        payload.get('catboost', {}),
        payload.get('ridge', {}),
        payload.get('lightgbm', {}),
    )


def load_tuned_params(verbose: bool = False) -> tuple[dict, dict, dict]:
    """CatBoost / Ridge / LightGBM params из best_params.json."""
    if not PARAMS_PATH.exists():
        if verbose:
            print(f"best_params.json не найден ({PARAMS_PATH}) — использую дефолтные параметры")
        return {}, {}, {}
    try:
        payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, {}, {}

    if verbose:
        if 'catboost_cv_rmsle' in payload:
            print(f"CatBoost: тюненные (CV RMSLE = {payload['catboost_cv_rmsle']:.4f})")
        if 'ridge_cv_rmsle' in payload:
            print(f"Ridge:    тюненный  (CV RMSLE = {payload['ridge_cv_rmsle']:.4f})")
        if 'lightgbm_cv_rmsle' in payload:
            print(f"LightGBM: тюненные (CV RMSLE = {payload['lightgbm_cv_rmsle']:.4f})")

    return _parse_payload(payload)


def load_best_params() -> tuple[dict, dict, dict]:
    """Алиас load_tuned_params(verbose=True) для create_submission."""
    return load_tuned_params(verbose=True)
