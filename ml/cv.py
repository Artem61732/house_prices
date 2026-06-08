"""KFold CV helpers для ML-пайплайна."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_predict, cross_val_score


def summarize_cv(rmsle_folds, y_true_log, y_pred_log):
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


def cv_evaluate_sklearn(estimator, X, y, kf):
    """KFold через sklearn cross_val_score / cross_val_predict."""
    neg_rmse = cross_val_score(
        estimator, X, y, cv=kf,
        scoring='neg_root_mean_squared_error', n_jobs=-1,
    )
    rmsle_folds = -neg_rmse
    y_pred_log = cross_val_predict(estimator, X, y, cv=kf, n_jobs=-1)
    return summarize_cv(rmsle_folds, y, y_pred_log)


def cv_evaluate_manual(make_model, X, y, kf):
    """
    Ручной KFold: на каждом фолде свежая модель через make_model().
    Используется для CatBoost и LightGBM (native mode).
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
    return summarize_cv(rmsle_folds, y_arr, y_pred_log)


def print_metrics(name, metrics):
    print(f"--- {name} ---")
    print(f"RMSLE (CV):  {metrics['rmsle_mean']:.4f} ± {metrics['rmsle_std']:.4f}")
    print(f"  per-fold:  {np.array2string(metrics['rmsle_folds'], precision=4)}")
    print(f"MAE  ($):    {metrics['mae']:.2f}")
    print(f"R2:          {metrics['r2']:.4f}\n")
