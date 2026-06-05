"""
Оценка DNN-экспериментов через K-Fold CV.

Эксперименты описаны в config.yaml (секция dl.experiments) и покрывают:
  1. Простая 2-слойная MLP
  2. Больше слоёв
  3. BatchNorm
  4. Dropout (разные значения)
  5. Разные размеры слоёв и активации
  6. Разные оптимизаторы
  7. Scheduler (косинусовый и др.)
  8. Разные LR, batch_size, n_epochs, loss_fn
  9. Embedding для категориальных фичей (задание со звёздочкой)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from data import load_data
from features import preprocess, prepare_for_dl
from dl.train import TrainConfig, cross_validate


def _build_train_config(exp_cfg, dl_defaults) -> TrainConfig:
    """Собирает TrainConfig из эксперимента + дефолтов dl-секции."""
    merged = OmegaConf.merge(dl_defaults, exp_cfg)
    d = OmegaConf.to_container(merged, resolve=True)
    d.pop('experiments', None)
    d.pop('device', None)
    return TrainConfig(**d)


def run_experiments(
    experiment_names: list[str] | None = None,
    n_splits: int | None = None,
    random_state: int | None = None,
):
    n_splits = n_splits or int(cfg.cv.n_splits)
    random_state = random_state or int(cfg.random_state)
    dl_cfg = cfg.dl
    device_cfg = str(dl_cfg.get('device', 'auto'))

    print(f"=== ОЦЕНКА DNN (KFold, {n_splits} фолдов) ===")

    X, y, _ = load_data()
    skew_threshold = float(cfg.preprocess.skew_threshold)
    X, skewed_cols = preprocess(X, skew_threshold=skew_threshold)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")

    y_log = np.log1p(y).to_numpy()
    numeric_cols, categorical_cols = prepare_for_dl(X)
    print(
        f"Признаков: {len(numeric_cols)} числовых, "
        f"{len(categorical_cols)} категориальных"
    )

    dl_defaults = OmegaConf.create({
        'cat_encoding': dl_cfg.get('cat_encoding', 'freq'),
        'cv_strategy': dl_cfg.get('cv_strategy', 'stratified'),
        'stratify_bins': dl_cfg.get('stratify_bins', 10),
        'batch_size': dl_cfg.batch_size,
        'n_epochs': dl_cfg.n_epochs,
        'learning_rate': dl_cfg.learning_rate,
        'patience': dl_cfg.patience,
        'loss_fn': dl_cfg.loss_fn,
        'optimizer': dl_cfg.optimizer,
        'scheduler': dl_cfg.scheduler,
    })

    experiments = list(dl_cfg.experiments)
    if experiment_names:
        experiments = [
            e for e in experiments
            if e.name in experiment_names
        ]

    results = []
    for exp in experiments:
        train_cfg = _build_train_config(exp, dl_defaults)
        print(f"\n--- {train_cfg.name} ---")
        print(
            f"  layers={train_cfg.hidden_layers}, act={train_cfg.activation}, "
            f"bn={train_cfg.batch_norm}, dropout={train_cfg.dropout}, "
            f"cat_encoding={train_cfg.cat_encoding}"
        )
        print(
            f"  opt={train_cfg.optimizer}, sched={train_cfg.scheduler}, "
            f"lr={train_cfg.learning_rate}, bs={train_cfg.batch_size}, "
            f"epochs={train_cfg.n_epochs}, loss={train_cfg.loss_fn}"
        )

        result = cross_validate(
            X, y_log, numeric_cols, categorical_cols,
            train_cfg, n_splits, random_state, device_cfg,
        )
        print(
            f"RMSLE (CV): {result['rmsle_mean']:.4f} "
            f"± {result['rmsle_std']:.4f}"
        )
        print(
            f"  per-fold: {np.array2string(result['rmsle_folds'], precision=4)}"
        )
        results.append(result)

    print("\n=== ИТОГИ DNN (отсортировано по RMSLE) ===")
    for r in sorted(results, key=lambda x: x['rmsle_mean']):
        print(
            f"{r['name']:30s}  RMSLE = {r['rmsle_mean']:.4f} "
            f"± {r['rmsle_std']:.4f}"
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DNN experiments (House Prices)")
    parser.add_argument(
        '--experiments', nargs='+', default=None,
        help="Имена экспериментов из config.yaml (по умолчанию — все)",
    )
    args = parser.parse_args()
    run_experiments(experiment_names=args.experiments)
