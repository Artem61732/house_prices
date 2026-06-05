"""
Создание сабмита на DNN-модели.

По умолчанию используются параметры из dl/best_params.json (после tune).
Можно указать эксперимент из config.yaml через --experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from data import load_data
from features import preprocess, prepare_for_dl
from dl.main import _build_train_config
from dl.train import fit_full_model, predict
from dl.tune import dict_to_train_config, load_tuned_params


def _resolve_train_config(use_tuned: bool, experiment_name: str | None):
    dl_cfg = cfg.dl

    if use_tuned:
        tuned_params, cv_rmsle = load_tuned_params()
        if tuned_params:
            if cv_rmsle is not None:
                print(f"DNN: тюненные параметры (CV RMSLE = {cv_rmsle:.4f})")
            return dict_to_train_config(tuned_params, dl_cfg)

    if experiment_name is None:
        experiment_name = 'embeddings'

    experiments = {e.name: e for e in dl_cfg.experiments}
    if experiment_name not in experiments:
        available = ', '.join(experiments)
        raise ValueError(
            f"Эксперимент '{experiment_name}' не найден. Доступны: {available}"
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
    return _build_train_config(experiments[experiment_name], dl_defaults)


def create_submission(use_tuned: bool = True, experiment_name: str | None = None):
    dl_cfg = cfg.dl
    device_cfg = str(dl_cfg.get('device', 'auto'))
    random_state = int(cfg.random_state)

    train_cfg = _resolve_train_config(use_tuned, experiment_name)

    print(f"=== СОЗДАНИЕ DL-САБМИТА (модель: {train_cfg.name}) ===")

    X_train_raw, y_train, test_data = load_data()
    test_ids = test_data['Id']

    skew_threshold = float(cfg.preprocess.skew_threshold)
    X_train, skewed_cols = preprocess(X_train_raw, skew_threshold=skew_threshold)
    X_test, _ = preprocess(test_data, skewed_cols=skewed_cols)
    print(f"log1p применён к {len(skewed_cols)} скошенным колонкам")

    y_log = np.log1p(y_train).to_numpy()
    numeric_cols, categorical_cols = prepare_for_dl(X_train)

    print("Обучение DNN на полном train...")
    model, encoder = fit_full_model(
        X_train, y_log, numeric_cols, categorical_cols,
        train_cfg, random_state, device_cfg,
    )

    pred_log = predict(model, encoder, X_test, train_cfg, device_cfg)
    y_pred = np.expm1(pred_log)

    submission_path = getattr(cfg.paths, 'dl_submission', 'submission_dl.csv')
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
    submission.to_csv(submission_path, index=False)

    print(f"\nГотово! Файл '{submission_path}' сохранён ({len(submission)} строк).")
    print(
        f"Sale price: min={y_pred.min():.0f}, "
        f"median={np.median(y_pred):.0f}, max={y_pred.max():.0f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', default=None,
        help="Имя эксперимента из config.yaml (игнорирует --tuned)",
    )
    parser.add_argument(
        '--tuned', action='store_true', default=True,
        help="Использовать dl/best_params.json (по умолчанию: да)",
    )
    parser.add_argument(
        '--no-tuned', dest='tuned', action='store_false',
        help="Не использовать best_params.json, взять --experiment",
    )
    args = parser.parse_args()
    create_submission(use_tuned=args.tuned, experiment_name=args.experiment)
