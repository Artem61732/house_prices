"""
Оценка моделей через K-Fold кросс-валидацию + поиск весов блeнда.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from data import load_data
from features import (
    preprocess,
    prepare_for_catboost,
    prepare_for_lightgbm,
)


warnings.filterwarnings('ignore', message=".*select_dtypes.*str.*")

BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"
RANDOM_STATE = int(cfg.random_state)


# =================================================================
# PREPROCESSOR (для линейных моделей: impute + scale + OHE)
# =================================================================

def get_preprocessor(numeric_features, categorical_features):
    """ColumnTransformer для числовых и категориальных признаков."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
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


# =================================================================
# MODELS
# =================================================================

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


def load_tuned_params() -> tuple[dict, dict, dict]:
    """Подхватывает тюненные параметры CatBoost / Ridge / LightGBM из best_params.json."""
    if not BEST_PARAMS_PATH.exists():
        return {}, {}, {}
    try:
        payload = json.loads(BEST_PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, {}, {}
    return (
        payload.get('catboost', {}),
        payload.get('ridge', {}),
        payload.get('lightgbm', {}),
    )


# =================================================================
# CROSS-VALIDATION HELPERS
# =================================================================

def _summarize_cv(rmsle_folds, y_true_log, y_pred_log):
    rmsle_folds = np.asarray(rmsle_folds)
    y_pred_dollars = np.expm1(y_pred_log)
    y_true_dollars = np.expm1(y_true_log)
    return {
        'rmsle_mean': float(rmsle_folds.mean()),
        'rmsle_std': float(rmsle_folds.std()),
        'rmsle_folds': rmsle_folds,
        'mae': float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        'r2': float(r2_score(y_true_dollars, y_pred_dollars)),
        'y_pred_log': np.asarray(y_pred_log),
    }


def _cv_evaluate_sklearn(estimator, X, y, kf):
    """KFold через sklearn cross_val_score / cross_val_predict."""
    neg_rmse = cross_val_score(
        estimator, X, y, cv=kf,
        scoring='neg_root_mean_squared_error', n_jobs=-1,
    )
    rmsle_folds = -neg_rmse
    y_pred_log = cross_val_predict(estimator, X, y, cv=kf, n_jobs=-1)
    return _summarize_cv(rmsle_folds, y, y_pred_log)


def _cv_evaluate_manual(make_model, X, y, kf):
    """
    Ручной KFold: на каждом фолде свежая модель через make_model().
    Используется для CatBoost (sklearn.clone ломает cat_features) и LightGBM
    (для единообразия и сохранения category-колонок).
    """
    rmsle_folds = []
    y_pred_log = np.empty_like(np.asarray(y, dtype=float))
    y_arr = np.asarray(y)
    for train_idx, valid_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y_arr[train_idx], y_arr[valid_idx]
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        y_pred_log[valid_idx] = pred
        rmsle_folds.append(root_mean_squared_error(y_va, pred))
    return _summarize_cv(rmsle_folds, y_arr, y_pred_log)


def _print_metrics(name, m):
    print(f"--- {name} ---")
    print(f"RMSLE (CV):  {m['rmsle_mean']:.4f} ± {m['rmsle_std']:.4f}")
    print(f"  per-fold:  {np.array2string(m['rmsle_folds'], precision=4)}")
    print(f"MAE  ($):    {m['mae']:.2f}")
    print(f"R2:          {m['r2']:.4f}\n")


# =================================================================
# BLEND
# =================================================================

def find_blend_weights(oof_preds: dict, y_log, step: float = 0.1) -> dict:
    """
    Полный перебор весов на сетке с шагом `step` (сумма весов = 1).
    Возвращает {names: tuple, weights: tuple, score: float, top5: list, equal: dict}.
    """
    names = list(oof_preds)
    arrs = [np.asarray(oof_preds[n]) for n in names]
    y_arr = np.asarray(y_log)
    n = len(names)

    grid = np.arange(0.0, 1.0 + 1e-9, step)
    candidates = []

    def _enumerate(prefix, remaining):
        if len(prefix) == n - 1:
            last = remaining
            if last < -1e-9:
                return
            weights = (*prefix, max(last, 0.0))
            pred = sum(w * a for w, a in zip(weights, arrs))
            score = float(root_mean_squared_error(y_arr, pred))
            candidates.append((weights, score))
            return
        for w in grid:
            if w > remaining + 1e-9:
                break
            _enumerate(prefix + (round(float(w), 2),), round(remaining - float(w), 4))

    _enumerate((), 1.0)

    candidates.sort(key=lambda t: t[1])
    best_weights, best_score = candidates[0]
    equal_weights = (1.0 / n,) * n
    equal_pred = sum(w * a for w, a in zip(equal_weights, arrs))
    equal_score = float(root_mean_squared_error(y_arr, equal_pred))

    return {
        'names': names,
        'best_weights': best_weights,
        'best_score': best_score,
        'equal_weights': equal_weights,
        'equal_score': equal_score,
        'top5': candidates[:5],
    }


# =================================================================
# RUN: full evaluation
# =================================================================

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
        m = _cv_evaluate_sklearn(pipeline, X, y_log, kf)
        _print_metrics(name, m)
        results[name] = m

    X_cat, cat_features = prepare_for_catboost(X)
    m = _cv_evaluate_manual(
        lambda: get_catboost_model(cat_features, params=cb_params),
        X_cat, y_log, kf,
    )
    _print_metrics('CatBoost (native)', m)
    results['CatBoost (native)'] = m

    X_lgb, _ = prepare_for_lightgbm(X)
    m = _cv_evaluate_manual(
        lambda: get_lightgbm_model(params=lgb_params),
        X_lgb, y_log, kf,
    )
    _print_metrics('LightGBM (native)', m)
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

        results['Blend equal'] = {'rmsle_mean': info['equal_score'], 'rmsle_std': 0.0,
                                  'rmsle_folds': np.array([info['equal_score']]),
                                  'mae': float('nan'), 'r2': float('nan'),
                                  'y_pred_log': None}
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
    n_splits = int(cfg.cv.n_splits)
    run_evaluation(n_splits=n_splits, random_state=RANDOM_STATE)
