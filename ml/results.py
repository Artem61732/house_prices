"""Сохранение CV-результатов ML в outputs/ml/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from paths import ML_OUTPUTS_DIR, ML_VALIDATION_REPORT


def _row_from_metrics(name: str, metrics: dict[str, Any], pipeline: str = 'ml') -> dict[str, Any]:
    row: dict[str, Any] = {
        'pipeline': pipeline,
        'name': name,
        'rmsle_mean': metrics.get('rmsle_mean'),
        'rmsle_std': metrics.get('rmsle_std'),
        'mae': metrics.get('mae'),
        'r2': metrics.get('r2'),
    }
    folds = metrics.get('rmsle_folds')
    if folds is not None:
        row['rmsle_folds'] = [float(x) for x in folds]
    return row


def metrics_to_rows(results: dict[str, dict], pipeline: str = 'ml') -> list[dict[str, Any]]:
    """Преобразует словарь метрик в строки для CSV/JSON."""
    return [_row_from_metrics(name, m, pipeline=pipeline) for name, m in results.items()]


def save_results(
    results: dict[str, dict],
    *,
    pipeline: str = 'ml',
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Сохраняет results.csv и results.json; возвращает пути к файлам."""
    out_dir = out_dir or ML_OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = metrics_to_rows(results, pipeline=pipeline)
    df = pd.DataFrame(rows).sort_values('rmsle_mean', na_position='last')

    csv_path = out_dir / 'results.csv'
    json_path = out_dir / 'results.json'
    df.to_csv(csv_path, index=False)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {csv_path}")
    return csv_path, json_path


def save_validation_report(
    report: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Сохраняет validation_report.json."""
    path = path or ML_VALIDATION_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Validation report -> {path}")
    return path


def build_ml_validation_report(
    results: dict[str, dict],
    *,
    n_splits: int,
    random_state: int,
    skewed_cols: list[str],
    blend_info: dict | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Сводка лучшей модели и blend для validation_report.json."""
    ranked = sorted(
        ((name, m) for name, m in results.items() if m.get('rmsle_mean') is not None),
        key=lambda kv: kv[1]['rmsle_mean'],
    )
    best_name, best_metrics = ranked[0] if ranked else ('', {})

    report: dict[str, Any] = {
        'task': 'house_prices_regression',
        'metric': 'rmsle',
        'n_splits': n_splits,
        'random_state': random_state,
        'quick': quick,
        'n_skewed_features': len(skewed_cols),
        'best_model': best_name,
        'best_rmsle': best_metrics.get('rmsle_mean'),
        'best_rmsle_std': best_metrics.get('rmsle_std'),
        'leaderboard': [
            {
                'name': name,
                'rmsle_mean': m.get('rmsle_mean'),
                'rmsle_std': m.get('rmsle_std'),
            }
            for name, m in ranked
        ],
    }
    if blend_info:
        report['blend'] = {
            'equal_weights': list(blend_info['equal_weights']),
            'equal_rmsle': blend_info['equal_score'],
            'best_weights': list(blend_info['best_weights']),
            'best_rmsle': blend_info['best_score'],
            'model_names': blend_info['names'],
        }
    return report
