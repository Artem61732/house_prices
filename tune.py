"""
Тюнинг моделей через Optuna.

- KFold-CV (n_splits из config.yaml) на лог-таргете => RMSLE = RMSE.
- CatBoost: ранняя остановка по eval_set + Optuna-pruning (плохие trials
  отсекаются после первых фолдов).
- Ridge: лёгкий поиск alpha в логарифмической шкале.
- Лучшие параметры сохраняются в best_params.json и автоматически
  подхватываются create_submission.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
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

CATBOOST_FIXED = dict(
    iterations=1500,
    random_state=42,
    verbose=False,
    thread_count=-1,
    od_type='Iter',
    od_wait=30,
    allow_writing_files=False,
)

LIGHTGBM_FIXED = dict(
    n_estimators=2000,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)


def prepare_data():
    X, y, _ = load_data()
    X = fill_na_domain(X)
    X = feature_engineering(X)
    skewed_cols = get_skewed_columns(X, threshold=0.75)
    X = log_skewed_features(X, skewed_cols)
    y = np.log1p(y).to_numpy()
    return X, y


# ============================ CatBoost ============================

def cv_score_catboost(params: dict, X, y, cat_features, kf) -> float:
    rmsle_folds = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = CatBoostRegressor(cat_features=cat_features, **CATBOOST_FIXED, **params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        pred = model.predict(X_va)
        rmsle_folds.append(root_mean_squared_error(y_va, pred))
    return float(np.mean(rmsle_folds))


def make_objective_catboost(X, y, cat_features, kf):
    import time

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
            depth=trial.suggest_int('depth', 4, 6),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 1, 20),
            random_strength=trial.suggest_float('random_strength', 0.1, 5.0),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
            border_count=trial.suggest_categorical('border_count', [64, 128, 254]),
        )
        print(f"[trial {trial.number:02d}] start: {params}", flush=True)
        t0 = time.time()
        rmsle_folds = []
        for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model = CatBoostRegressor(cat_features=cat_features, **CATBOOST_FIXED, **params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            pred = model.predict(X_va)
            score = root_mean_squared_error(y_va, pred)
            rmsle_folds.append(score)
            print(
                f"[trial {trial.number:02d}] fold {fold_i + 1}/{kf.get_n_splits()} "
                f"RMSLE={score:.4f} | best_iter={model.get_best_iteration()} "
                f"| running={time.time() - t0:.1f}s",
                flush=True,
            )

            trial.report(float(np.mean(rmsle_folds)), fold_i)
            if trial.should_prune():
                print(f"[trial {trial.number:02d}] PRUNED after fold {fold_i + 1}", flush=True)
                raise optuna.TrialPruned()

        return float(np.mean(rmsle_folds))
    return objective


def tune_catboost(n_trials: int, n_splits: int, random_state: int):
    X, y = prepare_data()
    X_cat, cat_features = prepare_for_catboost(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"=== ТЮНИНГ CATBOOST (Optuna, {n_trials} trials, KFold {n_splits}) ===")
    print(
        f"Признаков: {X_cat.shape[1]} | категориальных: {len(cat_features)} "
        f"| объектов: {X_cat.shape[0]}"
    )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def callback(study_, trial):
        state = trial.state.name
        val = f"{trial.value:.4f}" if trial.value is not None else "—"
        try:
            best = f"{study_.best_value:.4f}"
        except ValueError:
            best = "—"
        print(
            f"=== trial {trial.number:02d} done [{state:<8}] "
            f"RMSLE={val} | best={best} ===",
            flush=True,
        )

    study.optimize(
        make_objective_catboost(X_cat, y, cat_features, kf),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    print("\n=== ЛУЧШИЕ ПАРАМЕТРЫ CATBOOST ===")
    print(f"CV RMSLE: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    return study.best_params, study.best_value


# ============================ Ridge ============================

def tune_ridge(n_trials: int, n_splits: int, random_state: int):
    X, y = prepare_data()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"\n=== ТЮНИНГ RIDGE (Optuna, {n_trials} trials, KFold {n_splits}) ===")

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=alpha, random_state=random_state)),
        ])
        neg_rmse = cross_val_score(
            pipe, X, y, cv=kf,
            scoring='neg_root_mean_squared_error', n_jobs=-1,
        )
        return float(-neg_rmse.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"CV RMSLE: {study.best_value:.4f}")
    print(f"  alpha = {study.best_params['alpha']:.4f}")
    return study.best_params, study.best_value


# ============================ LightGBM ============================

def make_objective_lightgbm(X, y, kf):
    import time

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 100),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            subsample_freq=trial.suggest_int('subsample_freq', 0, 5),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        )
        print(f"[trial {trial.number:02d}] start: {params}", flush=True)
        t0 = time.time()
        rmsle_folds = []
        for fold_i, (tr_idx, va_idx) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model = LGBMRegressor(**LIGHTGBM_FIXED, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
            )
            pred = model.predict(X_va)
            score = root_mean_squared_error(y_va, pred)
            rmsle_folds.append(score)
            print(
                f"[trial {trial.number:02d}] fold {fold_i + 1}/{kf.get_n_splits()} "
                f"RMSLE={score:.4f} | best_iter={model.best_iteration_} "
                f"| running={time.time() - t0:.1f}s",
                flush=True,
            )
            trial.report(float(np.mean(rmsle_folds)), fold_i)
            if trial.should_prune():
                print(f"[trial {trial.number:02d}] PRUNED after fold {fold_i + 1}", flush=True)
                raise optuna.TrialPruned()
        return float(np.mean(rmsle_folds))
    return objective


def tune_lightgbm(n_trials: int, n_splits: int, random_state: int):
    X, y = prepare_data()
    X_lgb, _ = prepare_for_lightgbm(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"\n=== ТЮНИНГ LIGHTGBM (Optuna, {n_trials} trials, KFold {n_splits}) ===")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def callback(study_, trial):
        state = trial.state.name
        val = f"{trial.value:.4f}" if trial.value is not None else "—"
        try:
            best = f"{study_.best_value:.4f}"
        except ValueError:
            best = "—"
        print(
            f"=== trial {trial.number:02d} done [{state:<8}] "
            f"RMSLE={val} | best={best} ===",
            flush=True,
        )

    study.optimize(
        make_objective_lightgbm(X_lgb, y, kf),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    print(f"\n=== ЛУЧШИЕ ПАРАМЕТРЫ LIGHTGBM ===")
    print(f"CV RMSLE: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    return study.best_params, study.best_value


# ============================ Сохранение ============================

def save_best_params(
    catboost_params=None, catboost_score=None,
    ridge_params=None, ridge_score=None,
    lightgbm_params=None, lightgbm_score=None,
):
    payload = {}
    if PARAMS_PATH.exists():
        try:
            payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            payload = {}

    if catboost_params is not None:
        payload['catboost'] = {**CATBOOST_FIXED, **catboost_params}
        payload['catboost_cv_rmsle'] = catboost_score
    if ridge_params is not None:
        payload['ridge'] = ridge_params
        payload['ridge_cv_rmsle'] = ridge_score
    if lightgbm_params is not None:
        payload['lightgbm'] = {**LIGHTGBM_FIXED, **lightgbm_params}
        payload['lightgbm_cv_rmsle'] = lightgbm_score

    PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"\nСохранено в {PARAMS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models', nargs='+',
        default=['lightgbm'],
        choices=['catboost', 'ridge', 'lightgbm'],
        help="Какие модели тюнить (по умолчанию: lightgbm)",
    )
    parser.add_argument('--n-trials', type=int, default=None)
    args = parser.parse_args()

    cv_cfg = getattr(cfg, 'cv', None)
    n_splits = int(getattr(cv_cfg, 'n_splits', 5)) if cv_cfg is not None else 5
    random_state = int(getattr(cv_cfg, 'random_state', 42)) if cv_cfg is not None else 42

    tune_cfg = getattr(cfg, 'tune', None)
    n_trials_default = int(getattr(tune_cfg, 'n_trials', 30)) if tune_cfg is not None else 30
    n_trials = args.n_trials if args.n_trials is not None else n_trials_default

    cb_params = cb_score = ridge_params = ridge_score = lgb_params = lgb_score = None

    if 'catboost' in args.models:
        cb_params, cb_score = tune_catboost(n_trials, n_splits, random_state)
    if 'ridge' in args.models:
        ridge_params, ridge_score = tune_ridge(min(25, n_trials), n_splits, random_state)
    if 'lightgbm' in args.models:
        lgb_params, lgb_score = tune_lightgbm(n_trials, n_splits, random_state)

    save_best_params(
        cb_params, cb_score,
        ridge_params, ridge_score,
        lgb_params, lgb_score,
    )
