"""
TrainConfig и сборка конфигурации DNN.

- TrainConfig                  — dataclass гиперпараметров
- build_train_config_from_yaml — эксперимент из dl/config.yaml
- build_train_config_from_json — параметры из outputs/dl/best_params.json
- load_tuned_params            — чтение best_params.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from config import cfg
from paths import DL_BEST_PARAMS_PATH, ensure_output_dirs

ensure_output_dirs()
PARAMS_PATH = DL_BEST_PARAMS_PATH


@dataclass
class TrainConfig:
    """
    Гиперпараметры обучения DNN.

    cat_encoding — способ кодирования категориальных признаков:
      - embedding: отдельные Embedding-слои в MLP
      - onehot:    OneHotEncoder, конкатенация с числовыми
      - freq:      частота категории в train
      - target:    сглаженное среднее log-таргета по категории (fit на train-фолде)
    """

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


def get_dl_defaults(dl_cfg=None) -> OmegaConf:
    """Базовые гиперпараметры DL из dl/config.yaml."""
    dl_cfg = dl_cfg or cfg.dl
    return OmegaConf.create({
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


def _normalize_scheduler(scheduler) -> str | None:
    if scheduler is None or scheduler == 'none':
        return None
    return scheduler


def build_train_config_from_yaml(exp_cfg, dl_cfg=None) -> TrainConfig:
    """Собирает TrainConfig из записи dl.experiments + дефолтов dl-секции."""
    merged = OmegaConf.merge(get_dl_defaults(dl_cfg), exp_cfg)
    params = OmegaConf.to_container(merged, resolve=True)
    for key in ('experiments', 'device', 'tune'):
        params.pop(key, None)
    params['scheduler'] = _normalize_scheduler(params.get('scheduler'))
    return TrainConfig(**params)


def build_train_config_from_json(params: dict, dl_cfg=None) -> TrainConfig:
    """Собирает TrainConfig из сохранённого JSON (после Optuna-тюнинга)."""
    dl_cfg = dl_cfg or cfg.dl
    merged = {
        'name': 'tuned',
        'cat_encoding': dl_cfg.get('cat_encoding', 'freq'),
        'cv_strategy': dl_cfg.get('cv_strategy', 'stratified'),
        'stratify_bins': int(dl_cfg.get('stratify_bins', 10)),
        'batch_size': int(dl_cfg.batch_size),
        'n_epochs': int(dl_cfg.n_epochs),
        'learning_rate': float(dl_cfg.learning_rate),
        'patience': int(dl_cfg.patience),
        'loss_fn': str(dl_cfg.loss_fn),
        'optimizer': str(dl_cfg.optimizer),
        'scheduler': dl_cfg.get('scheduler'),
        'weight_decay': 0.0,
        'hidden_layers': [128, 64],
        'activation': 'relu',
        'batch_norm': True,
        'dropout': 0.0,
    }
    merged.update(params)
    merged['scheduler'] = _normalize_scheduler(merged.get('scheduler'))
    return TrainConfig(**merged)


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
    """Читает секцию dnn из outputs/dl/best_params.json."""
    if not PARAMS_PATH.exists():
        return {}, None
    try:
        payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, None
    return payload.get('dnn', {}), payload.get('dnn_cv_rmsle')
