"""
Создание сабмита на DNN-модели.

По умолчанию используются параметры из outputs/dl/best_params.json (после tune).
Можно указать эксперимент из dl/config.yaml через --experiment.
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from config import cfg
from data import load_data
from features import preprocess, prepare_for_dl
from dl.train_config import (
    build_train_config_from_json,
    build_train_config_from_yaml,
    load_tuned_params,
)
from dl.train import fit_full_model, predict


def resolve_train_config(use_tuned: bool, experiment_name: str | None):
    """Выбирает TrainConfig из best_params.json или эксперимента dl/config.yaml."""
    dl_cfg = cfg.dl

    if use_tuned:
        tuned_params, cv_rmsle = load_tuned_params()
        if tuned_params:
            if cv_rmsle is not None:
                print(f"DNN: тюненные параметры (CV RMSLE = {cv_rmsle:.4f})")
            return build_train_config_from_json(tuned_params, dl_cfg)

    if experiment_name is None:
        experiment_name = 'embeddings'

    experiments = {e.name: e for e in dl_cfg.experiments}
    if experiment_name not in experiments:
        available = ', '.join(experiments)
        raise ValueError(
            f"Эксперимент '{experiment_name}' не найден. Доступны: {available}"
        )

    return build_train_config_from_yaml(experiments[experiment_name], dl_cfg)


def create_submission(use_tuned: bool = True, experiment_name: str | None = None):
    """Обучает DNN на полном train и сохраняет submission_dl.csv."""
    dl_cfg = cfg.dl
    device_cfg = str(dl_cfg.get('device', 'auto'))
    random_state = int(cfg.random_state)

    train_cfg = resolve_train_config(use_tuned, experiment_name)

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

    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred})
    submission.to_csv(cfg.paths.dl_submission, index=False)

    print(f"\nГотово! Файл '{cfg.paths.dl_submission}' сохранён ({len(submission)} строк).")
    print(
        f"Sale price: min={y_pred.min():.0f}, "
        f"median={np.median(y_pred):.0f}, max={y_pred.max():.0f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', default=None,
        help="Имя эксперимента из dl/config.yaml (игнорирует --tuned)",
    )
    parser.add_argument(
        '--tuned', action='store_true', default=True,
        help="Использовать outputs/dl/best_params.json (по умолчанию: да)",
    )
    parser.add_argument(
        '--no-tuned', dest='tuned', action='store_false',
        help="Не использовать best_params.json, взять --experiment",
    )
    args = parser.parse_args()
    create_submission(use_tuned=args.tuned, experiment_name=args.experiment)
