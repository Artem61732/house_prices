"""
Константы и search space для DL-пайплайна.

Все допустимые значения гиперпараметров и пространства поиска Optuna
собраны здесь, чтобы не размазывать их по tune.py.
"""

from __future__ import annotations

# --- кодирование категориальных признаков для DNN ---
CAT_ENCODINGS = ('embedding', 'onehot', 'freq', 'target')

# --- общие списки гиперпараметров ---
OPTIMIZERS = ('adam', 'adamw', 'sgd', 'rmsprop')
SCHEDULERS = ('cosine', 'step', 'plateau', 'none')
LOSS_FNS = ('mse', 'mae', 'huber')
ACTIVATIONS = ('relu', 'elu', 'gelu', 'leaky_relu')
BATCH_SIZES = (32, 64, 128)
HIDDEN_WIDTHS = (64, 128, 256, 512)

# --- refined search space (после первого раунда тюнинга) ---
REFINED_CAT_ENCODINGS = ('onehot', 'target')
REFINED_OPTIMIZERS = ('adam',)
REFINED_SCHEDULERS = ('cosine', 'none')
REFINED_LOSS_FNS = ('mae', 'huber')
REFINED_ACTIVATIONS = ('elu', 'gelu')
REFINED_BATCH_SIZES = (32, 64)
REFINED_ARCHITECTURES: dict[str, list[int]] = {
    '256_512_512': [256, 512, 512],
    '256_512_256': [256, 512, 256],
    '256_256_512': [256, 256, 512],
    '512_512_256': [512, 512, 256],
    '512_256_512': [512, 256, 512],
    '128_512_512': [128, 512, 512],
    '256_512_128': [256, 512, 128],
    '512_512_512': [512, 512, 512],
}

# --- wide search space (exploratory) ---
WIDE_CAT_ENCODINGS = CAT_ENCODINGS
WIDE_OPTIMIZERS = OPTIMIZERS
WIDE_SCHEDULERS = SCHEDULERS
WIDE_LOSS_FNS = LOSS_FNS
WIDE_ACTIVATIONS = ACTIVATIONS
WIDE_BATCH_SIZES = BATCH_SIZES
WIDE_HIDDEN_WIDTHS = HIDDEN_WIDTHS

SEARCH_SPACES = frozenset({'refined', 'wide'})
