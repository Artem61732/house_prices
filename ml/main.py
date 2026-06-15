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
from ml.results import build_ml_validation_report, save_results, save_validation_report
from ml.train_config import load_tuned_params

warnings.filterwarnings('ignore', message=".*select_dtypes.*str.*")

RANDOM_STATE = int(cfg.random_state)


def run_evaluation(
    n_splits: int | None = None,
    random_state: int = RANDOM_STATE,
    *,
    quick: bool = False,
    save: bool = True,
):
    """KFold CV для всех ML-моделей и blend; возвращает словарь метрик."""
    if n_splits is None:
        n_splits = 3 if quick else int(cfg.cv.n_splits)

    print(f"=== ОЦЕНКА МОДЕЛЕЙ (KFold, {n_splits} фолдов"
          f"{', quick' if quick else ''}) ===")
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
    blend_info = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include='object').columns
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    for name, model in get_sklearn_models(ridge_params=ridge_params, quick=quick).items():
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
        blend_info = find_blend_weights(oof_preds, y_arr, step=0.1)

        print("--- Blend (Ridge + CatBoost + LightGBM на OOF) ---")
        print(
            f"  1/{len(blend_info['names'])} each   "
            f"-> RMSLE = {blend_info['equal_score']:.4f}"
        )
        bw = blend_info['best_weights']
        bn = blend_info['names']
        print(
            "  best " + " + ".join(f"{w:.2f} {n}" for w, n in zip(bw, bn))
            + f"  -> RMSLE = {blend_info['best_score']:.4f}"
        )
        print("  топ-5 комбинаций:")
        for weights, score in blend_info['top5']:
            ws = " ".join(f"{n}={w:.2f}" for n, w in zip(bn, weights))
            print(f"    {ws}  -> RMSLE = {score:.4f}")
        print()

        results['Blend equal'] = {'rmsle_mean': blend_info['equal_score']}
        results[f'Blend best ({", ".join(f"{w:.2f}" for w in bw)})'] = {
            'rmsle_mean': blend_info['best_score'],
        }

    print("=== ИТОГИ (отсортировано по RMSLE) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]['rmsle_mean']):
        if 'rmsle_std' in m:
            print(f"{name:50s}  RMSLE = {m['rmsle_mean']:.4f} ± {m['rmsle_std']:.4f}")
        else:
            print(f"{name:50s}  RMSLE = {m['rmsle_mean']:.4f}")

    if save:
        save_results(results, pipeline='ml')
        save_validation_report(build_ml_validation_report(
            results,
            n_splits=n_splits,
            random_state=random_state,
            skewed_cols=skewed_cols,
            blend_info=blend_info,
            quick=quick,
        ))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ML CV (House Prices)')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    run_evaluation(quick=args.quick, save=not args.no_save)
