"""Загрузка и слияние конфигурации проекта."""

import os

from omegaconf import OmegaConf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATHS = {
    'root': os.path.join(BASE_DIR, 'config.yaml'),
    'ml': os.path.join(BASE_DIR, 'ml', 'config.yaml'),
    'dl': os.path.join(BASE_DIR, 'dl', 'config.yaml'),
}


def load_config():
    """
    Собирает единый cfg из трёх файлов (порядок: root → ml → dl).

    Источники правды:
      - config.yaml      — paths, random_state, cv, preprocess
      - ml/config.yaml   — blend.weights (ручные), tune (Optuna ML)
      - dl/config.yaml   — dl.* (DNN, experiments, Optuna DL)
    """
    cfg = OmegaConf.create({})
    for path in CONFIG_PATHS.values():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return cfg


cfg = load_config()
