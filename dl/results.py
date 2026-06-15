"""Сохранение CV-результатов DL в outputs/dl/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from paths import DL_OUTPUTS_DIR


def dl_results_to_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append({
            'pipeline': 'dl',
            'name': r['name'],
            'rmsle_mean': r['rmsle_mean'],
            'rmsle_std': r['rmsle_std'],
            'rmsle_folds': [float(x) for x in r['rmsle_folds']],
            'hidden_layers': r['config'].hidden_layers,
            'cat_encoding': r['config'].cat_encoding,
            'dropout': r['config'].dropout,
            'batch_norm': r['config'].batch_norm,
        })
    return rows


def save_dl_results(
    results: list[dict[str, Any]],
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or DL_OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = dl_results_to_rows(results)
    df = pd.DataFrame(rows).sort_values('rmsle_mean')

    csv_path = out_dir / 'results.csv'
    json_path = out_dir / 'results.json'
    df.to_csv(csv_path, index=False)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {csv_path}")
    return csv_path, json_path
