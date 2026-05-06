import json
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from config import cfg

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    root_mean_squared_log_error
)

from features import (
    feature_engineering,
    fill_na_domain,
    get_skewed_columns,
    log_skewed_features,
    prepare_for_catboost,
    prepare_for_lightgbm,
)
from lightgbm import LGBMRegressor


BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"


def load_tuned_params() -> tuple[dict, dict, dict]:
    """Подхватываем тюненные параметры CatBoost / Ridge / LightGBM, если они есть."""
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

# --- DATA LOADING ---
def load_data(train_path=cfg.paths.train, test_path=cfg.paths.test, verbose=True):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if {'GrLivArea', 'SalePrice'} <= set(train_data.columns):
        before = len(train_data)
        outlier_mask = (
            (train_data['GrLivArea'] > 4000)
            & (train_data['SalePrice'] < 300000)
        )
        train_data = train_data.drop(train_data[outlier_mask].index)
        if verbose:
            print(f"Удалено выбросов GrLivArea: {before - len(train_data)} строк")

    X = train_data.drop('SalePrice', axis=1)
    y = train_data['SalePrice']

    return X, y, test_data

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- PREPROCESSING ---
def get_preprocessor(numeric_features, categorical_features):
    """
    Создает ColumnTransformer для обработки числовых и категориальных признаков.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

# --- MODELS ---
def get_sklearn_models(ridge_params: dict | None = None):
    """
    Модели, которые ходят через общий препроцессор (impute + scale + OHE).
    Ridge принимает тюненные параметры (например, alpha) если переданы.
    """
    ridge_kwargs = {'random_state': 42, **(ridge_params or {})}
    return {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(**ridge_kwargs),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
    }


def get_catboost_model(cat_features, params: dict | None = None):
    """
    CatBoost в нативном режиме: без OHE, с явным списком cat_features.
    Тюненные параметры из best_params.json накладываются поверх дефолтов.
    """
    kwargs = {'random_state': 42, 'verbose': False, **(params or {})}
    return CatBoostRegressor(cat_features=cat_features, **kwargs)


def get_lightgbm_model(params: dict | None = None):
    """
    LightGBM в нативном режиме: pandas category колонки auto-detect.
    """
    kwargs = {'random_state': 42, 'verbose': -1, **(params or {})}
    return LGBMRegressor(**kwargs)

# --- EVALUATION ---
def evaluate_model(model_name, y_true, y_pred):
    """
    Считает и выводит метрики качества для переданной модели.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_pred_positive = np.maximum(y_pred, 0)
    rmsle = root_mean_squared_log_error(y_true, y_pred_positive)
    
    print(f"--- {model_name} ---")
    print(f'MAE:   {mae:.2f}')
    print(f'RMSE:  {rmse:.2f}')
    print(f'R2:    {r2:.4f}')
    print(f'RMSLE: {rmsle:.4f}\n')
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'RMSLE': rmsle
    }

def _summarize_cv(rmsle_folds, y_true_log, y_pred_log):
    rmsle_folds = np.asarray(rmsle_folds)
    y_pred_dollars = np.expm1(y_pred_log)
    y_true_dollars = np.expm1(y_true_log)
    return {
        'rmsle_mean': rmsle_folds.mean(),
        'rmsle_std': rmsle_folds.std(),
        'rmsle_folds': rmsle_folds,
        'mae': mean_absolute_error(y_true_dollars, y_pred_dollars),
        'r2': r2_score(y_true_dollars, y_pred_dollars),
        'y_pred_log': np.asarray(y_pred_log),
    }


def _cv_evaluate_sklearn(estimator, X, y, kf):
    """
    KFold-оценка через sklearn cross_val_score / cross_val_predict.
    Подходит для всего, что корректно клонируется sklearn'ом.
    """
    neg_rmse = cross_val_score(
        estimator, X, y,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )
    rmsle_folds = -neg_rmse
    y_pred_log = cross_val_predict(estimator, X, y, cv=kf, n_jobs=-1)
    return _summarize_cv(rmsle_folds, y, y_pred_log)


def _cv_evaluate_manual(make_model, X, y, kf):
    """
    Ручной KFold-цикл: на каждом фолде создаём свежую модель через make_model().
    Используется для CatBoost (sklearn.clone ломает cat_features) и LightGBM
    (для единообразия и чтобы category-колонки не «терялись» при clone).
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


_cv_evaluate_catboost = _cv_evaluate_manual


def _print_metrics(name, m):
    print(f"--- {name} ---")
    print(f"RMSLE (CV):  {m['rmsle_mean']:.4f} ± {m['rmsle_std']:.4f}")
    print(f"  per-fold:  {np.array2string(m['rmsle_folds'], precision=4)}")
    print(f"MAE  ($):    {m['mae']:.2f}")
    print(f"R2:          {m['r2']:.4f}\n")


def run_evaluation(n_splits: int = 5, random_state: int = 42):
    """
    Оценка моделей через K-Fold кросс-валидацию.

    y приведён к log1p, поэтому RMSE на нём == RMSLE на исходной шкале —
    это и есть метрика Kaggle для House Prices.
    Дополнительно считаем MAE/R2 в долларах через cross_val_predict.

    CatBoost запускается в нативном режиме (без OHE, с cat_features).
    """
    print(f"=== ОЦЕНКА МОДЕЛЕЙ (KFold, {n_splits} фолдов) ===")
    X, y, _ = load_data()

    X = fill_na_domain(X)
    X = feature_engineering(X)
    skewed_cols = get_skewed_columns(X, threshold=0.75)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")
    X = log_skewed_features(X, skewed_cols)

    y = np.log1p(y)
    y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)

    cb_params, ridge_params, lgb_params = load_tuned_params()
    if cb_params:
        print(f"CatBoost: тюненные параметры из best_params.json")
    if ridge_params:
        print(f"Ridge: тюненный alpha = {ridge_params.get('alpha', 1.0):.4f}")
    if lgb_params:
        print(f"LightGBM: тюненные параметры из best_params.json")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    for model_name, model in get_sklearn_models(ridge_params=ridge_params).items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        m = _cv_evaluate_sklearn(pipeline, X, y, kf)
        _print_metrics(model_name, m)
        results[model_name] = m

    X_cat, cat_features = prepare_for_catboost(X)
    m = _cv_evaluate_manual(
        lambda: get_catboost_model(cat_features, params=cb_params),
        X_cat, y, kf,
    )
    _print_metrics('CatBoost (native)', m)
    results['CatBoost (native)'] = m

    X_lgb, _ = prepare_for_lightgbm(X)
    m = _cv_evaluate_manual(
        lambda: get_lightgbm_model(params=lgb_params),
        X_lgb, y, kf,
    )
    _print_metrics('LightGBM (native)', m)
    results['LightGBM (native)'] = m

    if all(k in results for k in ('Ridge', 'CatBoost (native)', 'LightGBM (native)')):
        ridge_oof = results['Ridge']['y_pred_log']
        cb_oof = results['CatBoost (native)']['y_pred_log']
        lgb_oof = results['LightGBM (native)']['y_pred_log']

        best_w, best_score = (None, np.inf)
        all_w = []
        for w_cb in np.linspace(0.0, 1.0, 11):
            for w_ridge in np.linspace(0.0, 1.0 - w_cb, max(int(round((1 - w_cb) * 10)) + 1, 1)):
                w_lgb = 1.0 - w_cb - w_ridge
                if w_lgb < -1e-9:
                    continue
                w_lgb = max(w_lgb, 0.0)
                pred = w_cb * cb_oof + w_ridge * ridge_oof + w_lgb * lgb_oof
                score = root_mean_squared_error(y_arr, pred)
                all_w.append((w_cb, w_ridge, w_lgb, score))
                if score < best_score:
                    best_score = float(score)
                    best_w = (round(w_cb, 2), round(w_ridge, 2), round(w_lgb, 2))

        equal_pred = (cb_oof + ridge_oof + lgb_oof) / 3
        equal_score = root_mean_squared_error(y_arr, equal_pred)

        def _make_entry(rmsle, pred):
            pred_d = np.expm1(pred)
            return {
                'rmsle_mean': rmsle,
                'rmsle_std': 0.0,
                'rmsle_folds': np.array([rmsle]),
                'mae': mean_absolute_error(np.expm1(y_arr), pred_d),
                'r2': r2_score(np.expm1(y_arr), pred_d),
                'y_pred_log': pred,
            }

        results['Blend 1/3 each'] = _make_entry(equal_score, equal_pred)
        bw_cb, bw_r, bw_lgb = best_w
        best_pred = bw_cb * cb_oof + bw_r * ridge_oof + bw_lgb * lgb_oof
        results[f'Blend best ({bw_cb} CB + {bw_r} R + {bw_lgb} LGB)'] = _make_entry(
            best_score, best_pred
        )

        print("--- Blend (CB + Ridge + LightGBM на OOF) ---")
        print(f"  1/3 each            -> RMSLE = {equal_score:.4f}")
        print(
            f"  best {bw_cb} CB + {bw_r} R + {bw_lgb} LGB"
            f"   -> RMSLE = {best_score:.4f}"
        )
        top5 = sorted(all_w, key=lambda t: t[3])[:5]
        print("  топ-5 комбинаций:")
        for w_cb, w_r, w_lgb, s in top5:
            print(f"    CB={w_cb:.2f} R={w_r:.2f} LGB={w_lgb:.2f} -> RMSLE = {s:.4f}")
        print()

    print("=== ИТОГИ (отсортировано по RMSLE) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]['rmsle_mean']):
        print(f"{name:50s}  RMSLE = {m['rmsle_mean']:.4f} ± {m['rmsle_std']:.4f}")

    return results

if __name__ == "__main__":
    cv_cfg = getattr(cfg, 'cv', None)
    n_splits = int(getattr(cv_cfg, 'n_splits', 5)) if cv_cfg is not None else 5
    random_state = int(getattr(cv_cfg, 'random_state', 42)) if cv_cfg is not None else 42
    run_evaluation(n_splits=n_splits, random_state=random_state)