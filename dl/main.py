"""
Оценка DNN-экспериментов через K-Fold CV.

Эксперименты описаны в dl/config.yaml (секция dl.experiments).
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import numpy as np

from config import cfg
from data import load_preprocessed_dl_train_target
from dl.train_config import build_train_config_from_yaml
from dl.train import cross_validate


def evaluate_dnn_experiments(
    experiment_names: list[str] | None = None,
    n_splits: int | None = None,
    random_state: int | None = None,
):
    """
    Запускает KFold CV для списка DNN-экспериментов из dl/config.yaml.

    experiment_names — подмножество имён; None — все эксперименты.
    """
    n_splits = n_splits or int(cfg.cv.n_splits)
    random_state = random_state or int(cfg.random_state)
    dl_cfg = cfg.dl
    device_cfg = str(dl_cfg.get('device', 'auto'))

    print(f"=== ОЦЕНКА DNN (KFold, {n_splits} фолдов) ===")

    X, y_log, numeric_cols, categorical_cols = load_preprocessed_dl_train_target()
    print(
        f"Признаков: {len(numeric_cols)} числовых, "
        f"{len(categorical_cols)} категориальных"
    )

    experiments = list(dl_cfg.experiments)
    if experiment_names:
        experiments = [
            e for e in experiments
            if e.name in experiment_names
        ]

    results = []
    for exp in experiments:
        train_cfg = build_train_config_from_yaml(exp, dl_cfg)
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
        help="Имена экспериментов из dl/config.yaml (по умолчанию — все)",
    )
    args = parser.parse_args()
    evaluate_dnn_experiments(experiment_names=args.experiments)
