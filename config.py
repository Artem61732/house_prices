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
    """Собирает cfg из config.yaml, ml/config.yaml, dl/config.yaml (в этом порядке)."""
    cfg = OmegaConf.create({})
    for path in CONFIG_PATHS.values():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return cfg


cfg = load_config()
