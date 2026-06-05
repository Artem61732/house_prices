"""
Тюнинг DNN через Optuna.

- Stratified CV (квантильные бины log-таргета) => RMSLE = RMSE на log1p(SalePrice).
- Pruning после каждого фолда (MedianPruner).
- Лучшие гиперпараметры сохраняются в dl/best_params.json и подхватываются
  dl/create_submission.py (--tuned по умолчанию).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from data import load_data
from features import preprocess, prepare_for_dl
from dl.train import (
    TrainConfig,
    _make_stratify_bins,
    build_cv_splitter,
    train_one_fold,
)


PARAMS_PATH = Path(__file__).parent / "best_params.json"
BACKUP_PATH = Path(__file__).parent / "best_params.backup.json"

# --- refined search space (по результатам первого раунда тюнинга) ---
REFINED_CAT_ENCODINGS = ['onehot', 'target']
REFINED_OPTIMIZERS = ['adam']
REFINED_SCHEDULERS = ['cosine', 'none']
REFINED_LOSS_FNS = ['mae', 'huber']
REFINED_ACTIVATIONS = ['elu', 'gelu']
REFINED_BATCH_SIZES = [32, 64]
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

# --- wide search space (первый раунд / exploratory) ---
WIDE_CAT_ENCODINGS = ['embedding', 'onehot', 'freq', 'target']
WIDE_OPTIMIZERS = ['adam', 'adamw', 'sgd', 'rmsprop']
WIDE_SCHEDULERS = ['cosine', 'step', 'plateau', 'none']
WIDE_LOSS_FNS = ['mse', 'mae', 'huber']
WIDE_ACTIVATIONS = ['relu', 'elu', 'gelu', 'leaky_relu']
WIDE_BATCH_SIZES = [32, 64, 128]
WIDE_HIDDEN_WIDTHS = [64, 128, 256, 512]

SEARCH_SPACES = {'refined', 'wide'}


def prepare_data():
    X, y, _ = load_data()
    skew_threshold = float(cfg.preprocess.skew_threshold)
    X, _ = preprocess(X, skew_threshold=skew_threshold)
    y_log = np.log1p(y).to_numpy()
    numeric_cols, categorical_cols = prepare_for_dl(X)
    return X, y_log, numeric_cols, categorical_cols


def _sample_hidden_layers_wide(trial: optuna.Trial) -> list[int]:
    n_layers = trial.suggest_int('n_layers', 2, 4)
    return [
        trial.suggest_categorical(f'hidden_{i}', WIDE_HIDDEN_WIDTHS)
        for i in range(n_layers)
    ]


def _sample_hidden_layers_refined(trial: optuna.Trial) -> list[int]:
    arch_key = trial.suggest_categorical('architecture', list(REFINED_ARCHITECTURES))
    return REFINED_ARCHITECTURES[arch_key]


def trial_to_train_config(
    trial: optuna.Trial,
    base: TrainConfig,
    search_space: str = 'refined',
) -> TrainConfig:
    search_space = search_space.lower()
    if search_space == 'refined':
        scheduler = trial.suggest_categorical('scheduler', REFINED_SCHEDULERS)
        return TrainConfig(
            name='optuna_trial',
            hidden_layers=_sample_hidden_layers_refined(trial),
            activation=trial.suggest_categorical('activation', REFINED_ACTIVATIONS),
            batch_norm=False,
            dropout=trial.suggest_float('dropout', 0.15, 0.4),
            cat_encoding=trial.suggest_categorical('cat_encoding', REFINED_CAT_ENCODINGS),
            cv_strategy=base.cv_strategy,
            stratify_bins=base.stratify_bins,
            batch_size=trial.suggest_categorical('batch_size', REFINED_BATCH_SIZES),
            n_epochs=base.n_epochs,
            learning_rate=trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True),
            patience=base.patience,
            loss_fn=trial.suggest_categorical('loss_fn', REFINED_LOSS_FNS),
            optimizer='adam',
            scheduler=None if scheduler == 'none' else scheduler,
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        )

    scheduler = trial.suggest_categorical('scheduler', WIDE_SCHEDULERS)
    return TrainConfig(
        name='optuna_trial',
        hidden_layers=_sample_hidden_layers_wide(trial),
        activation=trial.suggest_categorical('activation', WIDE_ACTIVATIONS),
        batch_norm=trial.suggest_categorical('batch_norm', [True, False]),
        dropout=trial.suggest_float('dropout', 0.0, 0.5),
        cat_encoding=trial.suggest_categorical('cat_encoding', WIDE_CAT_ENCODINGS),
        cv_strategy=base.cv_strategy,
        stratify_bins=base.stratify_bins,
        batch_size=trial.suggest_categorical('batch_size', WIDE_BATCH_SIZES),
        n_epochs=base.n_epochs,
        learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        patience=base.patience,
        loss_fn=trial.suggest_categorical('loss_fn', WIDE_LOSS_FNS),
        optimizer=trial.suggest_categorical('optimizer', WIDE_OPTIMIZERS),
        scheduler=None if scheduler == 'none' else scheduler,
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
    )


def train_config_to_dict(cfg_obj: TrainConfig) -> dict:
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


def dict_to_train_config(params: dict, dl_cfg) -> TrainConfig:
    """Собирает TrainConfig из сохранённого JSON + дефолтов dl-секции."""
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
    if merged.get('scheduler') == 'none':
        merged['scheduler'] = None
    return TrainConfig(**merged)


def load_tuned_params() -> tuple[dict, float | None]:
    if not PARAMS_PATH.exists():
        return {}, None
    try:
        payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}, None
    return payload.get('dnn', {}), payload.get('dnn_cv_rmsle')


def make_objective(
    X,
    y,
    numeric_cols,
    categorical_cols,
    base_cfg: TrainConfig,
    n_splits: int,
    random_state: int,
    device_cfg: str,
    search_space: str = 'refined',
):
    from dl.train import get_device

    device = get_device(device_cfg)
    splitter = build_cv_splitter(
        y,
        n_splits=n_splits,
        random_state=random_state,
        strategy=base_cfg.cv_strategy,
        stratify_bins=base_cfg.stratify_bins,
    )
    y_bins = None
    if base_cfg.cv_strategy.lower() == 'stratified':
        y_bins = _make_stratify_bins(y, base_cfg.stratify_bins)

    if isinstance(splitter, StratifiedKFold):
        splits = list(splitter.split(X, y_bins))
    else:
        splits = list(splitter.split(X))

    def objective(trial: optuna.Trial) -> float:
        train_cfg = trial_to_train_config(trial, base_cfg, search_space=search_space)
        print(
            f"[trial {trial.number:02d}] "
            f"layers={train_cfg.hidden_layers}, act={train_cfg.activation}, "
            f"dropout={train_cfg.dropout:.2f}, cat={train_cfg.cat_encoding}, "
            f"loss={train_cfg.loss_fn}, sched={train_cfg.scheduler}, "
            f"bs={train_cfg.batch_size}, lr={train_cfg.learning_rate:.2e}",
            flush=True,
        )
        t0 = time.time()
        rmsle_folds = []

        for fold_i, (train_idx, valid_idx) in enumerate(splits):
            rmsle, _, _ = train_one_fold(
                X, y, train_idx, valid_idx,
                numeric_cols, categorical_cols,
                train_cfg, device,
                random_state + fold_i,
            )
            rmsle_folds.append(rmsle)
            running = float(np.mean(rmsle_folds))
            print(
                f"[trial {trial.number:02d}] fold {fold_i + 1}/{n_splits} "
                f"RMSLE={rmsle:.4f} | running={running:.4f} "
                f"| elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
            trial.report(running, fold_i)
            if trial.should_prune():
                print(f"[trial {trial.number:02d}] PRUNED after fold {fold_i + 1}", flush=True)
                raise optuna.TrialPruned()

        return float(np.mean(rmsle_folds))

    return objective


def tune_dnn(
    n_trials: int,
    n_splits: int,
    random_state: int,
    n_epochs: int | None = None,
    patience: int | None = None,
    search_space: str = 'refined',
) -> tuple[dict, float]:
    X, y, numeric_cols, categorical_cols = prepare_data()
    dl_cfg = cfg.dl
    device_cfg = str(dl_cfg.get('device', 'auto'))
    tune_cfg = dl_cfg.get('tune', {})

    base_cfg = TrainConfig(
        cv_strategy=str(dl_cfg.get('cv_strategy', 'stratified')),
        stratify_bins=int(dl_cfg.get('stratify_bins', 10)),
        n_epochs=n_epochs or int(tune_cfg.get('n_epochs', dl_cfg.n_epochs)),
        patience=patience or int(tune_cfg.get('patience', dl_cfg.patience)),
    )

    search_space = search_space.lower()
    if search_space not in SEARCH_SPACES:
        raise ValueError(f"Unknown search_space='{search_space}'. Use: {sorted(SEARCH_SPACES)}")

    print(f"=== ТЮНИНГ DNN (Optuna, {n_trials} trials, CV {n_splits}, space={search_space}) ===")
    print(
        f"Признаков: {len(numeric_cols)} числ. + {len(categorical_cols)} кат. "
        f"| объектов: {len(X)} | epochs/trial: {base_cfg.n_epochs}"
    )

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def callback(study_, trial):
        state = trial.state.name
        val = f"{trial.value:.4f}" if trial.value is not None else "—"
        try:
            best = f"{study_.best_value:.4f}"
        except ValueError:
            best = "—"
        print(
            f"=== trial {trial.number:02d} done [{state:<8}] "
            f"RMSLE={val} | best={best} ===",
            flush=True,
        )

    study.optimize(
        make_objective(
            X, y, numeric_cols, categorical_cols,
            base_cfg, n_splits, random_state, device_cfg,
            search_space=search_space,
        ),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_cfg = trial_to_train_config(best_trial, base_cfg, search_space=search_space)
    best_params = train_config_to_dict(best_cfg)

    print("\n=== ЛУЧШИЕ ПАРАМЕТРЫ DNN ===")
    print(f"CV RMSLE: {study.best_value:.4f}")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    return best_params, study.best_value


def _backup_params():
    if not PARAMS_PATH.exists():
        return
    shutil.copy2(PARAMS_PATH, BACKUP_PATH)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped = PARAMS_PATH.with_name(f"best_params.{ts}.json")
    shutil.copy2(PARAMS_PATH, timestamped)
    print(f"Бэкап: {BACKUP_PATH.name}, {timestamped.name}")


def save_best_params(dnn_params: dict, dnn_score: float):
    _backup_params()
    payload = {}
    if PARAMS_PATH.exists():
        try:
            payload = json.loads(PARAMS_PATH.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            payload = {}
    payload['dnn'] = dnn_params
    payload['dnn_cv_rmsle'] = dnn_score
    PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Сохранено в {PARAMS_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna tuning for DNN")
    parser.add_argument('--n-trials', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=None)
    parser.add_argument('--n-epochs', type=int, default=None, help="Эпох на trial (быстрее тюнинг)")
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument(
        '--search-space', default=None, choices=sorted(SEARCH_SPACES),
        help="refined (по умолчанию) — узкий search space; wide — exploratory",
    )
    args = parser.parse_args()

    n_splits = args.n_splits or int(cfg.cv.n_splits)
    random_state = int(cfg.random_state)
    dl_tune = cfg.dl.get('tune', {})
    n_trials = args.n_trials or int(dl_tune.get('n_trials', cfg.tune.n_trials))
    search_space = args.search_space or str(dl_tune.get('search_space', 'refined'))

    best_params, best_score = tune_dnn(
        n_trials=n_trials,
        n_splits=n_splits,
        random_state=random_state,
        n_epochs=args.n_epochs,
        patience=args.patience,
        search_space=search_space,
    )
    save_best_params(best_params, best_score)
