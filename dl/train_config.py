"""
TrainConfig и сборка конфигурации DNN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, field

from omegaconf import OmegaConf

from config import cfg
from paths import DL_BEST_PARAMS_PATH, ensure_output_dirs

ensure_output_dirs()
PARAMS_PATH = DL_BEST_PARAMS_PATH


@dataclass
class TrainConfig:
    """Гиперпараметры одного DNN-эксперимента или тюненной модели."""

    name: str = 'default'
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = 'relu'
    batch_norm: bool = False
    dropout: float = 0.0
    cat_encoding: str = 'freq'
    cv_strategy: str = 'stratified'
    stratify_bins: int = 10
    batch_size: int = 64
    n_epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 15
    loss_fn: str = 'mse'
    optimizer: str = 'adam'
    scheduler: str | None = 'cosine'
    weight_decay: float = 0.0


def _normalize_scheduler(scheduler) -> str | None:
    if scheduler is None or scheduler == 'none':
        return None
    return scheduler


def _dl_defaults_dict(dl_cfg=None) -> dict:
    dl_cfg = dl_cfg or cfg.dl
    return {
        'name': 'default',
        'cat_encoding': str(dl_cfg.get('cat_encoding', 'freq')),
        'cv_strategy': str(dl_cfg.get('cv_strategy', 'stratified')),
        'stratify_bins': int(dl_cfg.get('stratify_bins', 10)),
        'batch_size': int(dl_cfg.batch_size),
        'n_epochs': int(dl_cfg.n_epochs),
        'learning_rate': float(dl_cfg.learning_rate),
        'patience': int(dl_cfg.patience),
        'loss_fn': str(dl_cfg.loss_fn),
        'optimizer': str(dl_cfg.optimizer),
        'scheduler': _normalize_scheduler(dl_cfg.get('scheduler')),
        'weight_decay': 0.0,
        'hidden_layers': [128, 64],
        'activation': 'relu',
        'batch_norm': False,
        'dropout': 0.0,
    }


def _to_train_config(params: dict, name: str | None = None) -> TrainConfig:
    clean = dict(params)
    for key in ('experiments', 'device', 'tune'):
        clean.pop(key, None)
    if name is not None:
        clean['name'] = name
    clean['scheduler'] = _normalize_scheduler(clean.get('scheduler'))
    allowed = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in clean.items() if k in allowed})


def build_train_config_from_yaml(exp_cfg, dl_cfg=None) -> TrainConfig:
    """Собирает TrainConfig из записи dl.experiments + дефолтов dl/config.yaml."""
    base = _dl_defaults_dict(dl_cfg)
    exp = OmegaConf.to_container(exp_cfg, resolve=True)
    base.update(exp)
    return _to_train_config(base)


def build_train_config_from_json(params: dict, dl_cfg=None) -> TrainConfig:
    """Собирает TrainConfig из outputs/dl/best_params.json (ключ dnn)."""
    base = _dl_defaults_dict(dl_cfg)
    base.update(params)
    return _to_train_config(base, name='tuned')


def train_config_to_dict(cfg_obj: TrainConfig) -> dict:
    """Сериализует TrainConfig в dict для сохранения в best_params.json."""
    return {
        'hidden_layers': cfg_obj.hidden_layers,
        'activation': cfg_obj.activation,
        'batch_norm': cfg_obj.batch_norm,
        'dropout': cfg_obj.dropout,
        'cat_encoding': cfg_obj.cat_encoding,
        'batch_size': cfg_obj.batch_size,
        'n_epochs': cfg_obj.n_epochs,
        'learning_rate': cfg_obj.learning_rate,
        'patience': cfg_obj.patience,
        'loss_fn': cfg_obj.loss_fn,
        'optimizer': cfg_obj.optimizer,
        'scheduler': cfg_obj.scheduler,
        'weight_decay': cfg_obj.weight_decay,
        'cv_strategy': cfg_obj.cv_strategy,
        'stratify_bins': cfg_obj.stratify_bins,
    }


def load_tuned_params() -> tuple[dict, float | None]:
    """Возвращает (dnn_params, dnn_cv_rmsle) из outputs/dl/best_params.json."""
    if not PARAMS_PATH.exists():
        return {}, None
    try:
        payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, None
    return payload.get('dnn', {}), payload.get('dnn_cv_rmsle')
