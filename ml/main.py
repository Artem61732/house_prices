"""
Оценка моделей через K-Fold кросс-валидацию + поиск весов blend.
"""

from __future__ import annotations

import warnings

import bootstrap  # noqa: F401
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from config import cfg
from data import load_data
from features import preprocess, prepare_for_catboost, prepare_for_lightgbm
from ml.blend import find_blend_weights
from ml.cv import cv_evaluate_manual, cv_evaluate_sklearn, print_metrics
from ml.models import (
    get_catboost_model,
    get_lightgbm_model,
    get_preprocessor,
    get_sklearn_models,
)
from ml.train_config import load_tuned_params

warnings.filterwarnings('ignore', message=".*select_dtypes.*str.*")

RANDOM_STATE = int(cfg.random_state)


def run_evaluation(n_splits: int = 5, random_state: int = RANDOM_STATE):
    print(f"=== ОЦЕНКА МОДЕЛЕЙ (KFold, {n_splits} фолдов) ===")
    X, y, _ = load_data()

    skew_threshold = float(cfg.preprocess.skew_threshold)
    X, skewed_cols = preprocess(X, skew_threshold=skew_threshold)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")

    y_log = np.log1p(y)
    y_arr = y_log.to_numpy()

    cb_params, ridge_params, lgb_params = load_tuned_params()
    if cb_params:
        print("CatBoost: тюненные параметры из best_params.json")
    if ridge_params:
        print(f"Ridge: тюненный alpha = {ridge_params.get('alpha', 1.0):.4f}")
    if lgb_params:
        print("LightGBM: тюненные параметры из best_params.json")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include='object').columns
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    for name, model in get_sklearn_models(ridge_params=ridge_params).items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        m = cv_evaluate_sklearn(pipeline, X, y_log, kf)
        print_metrics(name, m)
        results[name] = m

    X_cat, cat_features = prepare_for_catboost(X)
    m = cv_evaluate_manual(
        lambda: get_catboost_model(cat_features, params=cb_params),
        X_cat, y_log, kf,
    )
    print_metrics('CatBoost (native)', m)
    results['CatBoost (native)'] = m

    X_lgb, _ = prepare_for_lightgbm(X)
    m = cv_evaluate_manual(
        lambda: get_lightgbm_model(params=lgb_params),
        X_lgb, y_log, kf,
    )
    print_metrics('LightGBM (native)', m)
    results['LightGBM (native)'] = m

    blend_keys = ('Ridge', 'CatBoost (native)', 'LightGBM (native)')
    if all(k in results for k in blend_keys):
        oof_preds = {k: results[k]['y_pred_log'] for k in blend_keys}
        info = find_blend_weights(oof_preds, y_arr, step=0.1)

        print("--- Blend (Ridge + CatBoost + LightGBM на OOF) ---")
        print(
            f"  1/{len(info['names'])} each   "
            f"-> RMSLE = {info['equal_score']:.4f}"
        )
        bw = info['best_weights']
        bn = info['names']
        print(
            "  best " + " + ".join(f"{w:.2f} {n}" for w, n in zip(bw, bn))
            + f"  -> RMSLE = {info['best_score']:.4f}"
        )
        print("  топ-5 комбинаций:")
        for weights, score in info['top5']:
            ws = " ".join(f"{n}={w:.2f}" for n, w in zip(bn, weights))
            print(f"    {ws}  -> RMSLE = {score:.4f}")
        print()

        results['Blend equal'] = {
            'rmsle_mean': info['equal_score'], 'rmsle_std': 0.0,
            'rmsle_folds': np.array([info['equal_score']]),
            'mae': float('nan'), 'r2': float('nan'),
            'y_pred_log': None,
        }
        results[f'Blend best ({", ".join(f"{w:.2f}" for w in bw)})'] = {
            'rmsle_mean': info['best_score'], 'rmsle_std': 0.0,
            'rmsle_folds': np.array([info['best_score']]),
            'mae': float('nan'), 'r2': float('nan'),
            'y_pred_log': None,
        }

    print("=== ИТОГИ (отсортировано по RMSLE) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]['rmsle_mean']):
        print(f"{name:50s}  RMSLE = {m['rmsle_mean']:.4f} ± {m['rmsle_std']:.4f}")

    return results


if __name__ == "__main__":
    run_evaluation(n_splits=int(cfg.cv.n_splits), random_state=RANDOM_STATE)
