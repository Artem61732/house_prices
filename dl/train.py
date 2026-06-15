"""
Обучение DNN: оптимизаторы, scheduler, early stopping, KFold CV.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from dl.dataset import FeatureEncoder, make_dataloader
from dl.model import HousePriceMLP
from dl.train_config import TrainConfig


LOSS_FNS = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'huber': nn.HuberLoss,
}

CV_STRATEGIES = {'kfold', 'stratified'}


def get_device(device_cfg: str = 'auto') -> torch.device:
    """Возвращает torch.device; 'auto' выбирает cuda при наличии."""
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Создаёт оптимизатор по cfg.optimizer (adam, adamw, sgd, rmsprop)."""
    params = model.parameters()
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    name = cfg.optimizer.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Создаёт LR-scheduler по cfg.scheduler или None."""
    if not cfg.scheduler:
        return None
    name = cfg.scheduler.lower()
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5,
        )
    return None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Фиксирует seed для numpy, torch и cudnn."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_stratify_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if n_bins < 2:
        return np.zeros_like(y, dtype=np.int64)

    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 2:
        return np.zeros_like(y, dtype=np.int64)
    return np.digitize(y, edges[1:-1], right=True).astype(np.int64)


def build_cv_splitter(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    strategy: str = 'stratified',
    stratify_bins: int = 10,
):
    """KFold или StratifiedKFold (по квантильным бинам y_log)."""
    strategy = (strategy or 'kfold').lower()
    if strategy not in CV_STRATEGIES:
        raise ValueError(f"Unknown cv_strategy='{strategy}'. Use: {sorted(CV_STRATEGIES)}")

    if strategy == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_bins = _make_stratify_bins(y, stratify_bins)
    if len(np.unique(y_bins)) < 2 or np.min(np.bincount(y_bins)) < n_splits:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _fit_encoder(
    X_df: pd.DataFrame,
    y: np.ndarray | None,
    train_idx: np.ndarray | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
) -> FeatureEncoder:
    encoder = FeatureEncoder(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_cardinalities=[],
        cat_encoding=cfg.cat_encoding,
    )
    fit_df = X_df if train_idx is None else X_df.iloc[train_idx]
    fit_y = y if train_idx is None else (y[train_idx] if y is not None else None)
    if cfg.cat_encoding.lower() == 'target':
        encoder.fit(fit_df, y=fit_y)
    else:
        encoder.fit(fit_df)
    return encoder


def _make_model(
    encoder: FeatureEncoder,
    n_numeric: int,
    cfg: TrainConfig,
    use_embeddings: bool,
    device: torch.device,
) -> HousePriceMLP:
    return HousePriceMLP(
        n_numeric=n_numeric,
        cat_cardinalities=encoder.cat_cardinalities,
        hidden_layers=cfg.hidden_layers,
        activation=cfg.activation,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        use_embeddings=use_embeddings,
    ).to(device)


def _forward_batch(
    model: HousePriceMLP,
    batch: tuple,
    device: torch.device,
    use_embeddings: bool,
):
    if use_embeddings:
        numeric, categorical = batch[0], batch[1]
        targets = batch[2] if len(batch) > 2 else None
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        preds = model(numeric, categorical)
    else:
        numeric = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        numeric = numeric.to(device)
        preds = model(numeric)

    if targets is None:
        return preds
    return preds, targets.to(device)


def _predict_loader(
    model: HousePriceMLP,
    loader: DataLoader,
    device: torch.device,
    use_embeddings: bool,
) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = _forward_batch(model, batch, device, use_embeddings)
            preds.append(out[0].cpu().numpy() if isinstance(out, tuple) else out.cpu().numpy())
    return np.concatenate(preds)


def _train_model(
    model: HousePriceMLP,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    use_embeddings: bool,
    valid_loader: DataLoader | None = None,
) -> HousePriceMLP:
    criterion = LOSS_FNS[cfg.loss_fn]()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_state = None
    best_rmsle = float('inf')
    epochs_no_improve = 0

    for _ in range(cfg.n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds, targets = _forward_batch(model, batch, device, use_embeddings)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        if valid_loader is None:
            if scheduler is not None:
                scheduler.step()
            continue

        val_preds = _predict_loader(model, valid_loader, device, use_embeddings)
        val_targets = np.concatenate([
            batch[-1].numpy() for batch in valid_loader
        ])
        rmsle = root_mean_squared_error(val_targets, val_preds)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(rmsle)
            else:
                scheduler.step()

        if rmsle < best_rmsle - 1e-5:
            best_rmsle = rmsle
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_one_fold(
    X_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    device: torch.device,
    random_state: int,
) -> tuple[float, np.ndarray, HousePriceMLP]:
    """Обучает DNN на одном CV-фолде; возвращает RMSLE, OOF-предсказания и модель."""
    seed_everything(random_state, deterministic=True)
    use_embeddings = cfg.cat_encoding.lower() == 'embedding'

    encoder = _fit_encoder(
        X_df, y, train_idx, numeric_cols, categorical_cols, cfg,
    )
    X_tr_num, X_tr_cat = encoder.transform(X_df.iloc[train_idx], use_embeddings=use_embeddings)
    X_va_num, X_va_cat = encoder.transform(X_df.iloc[valid_idx], use_embeddings=use_embeddings)

    train_loader = make_dataloader(X_tr_num, X_tr_cat, y[train_idx], cfg.batch_size, True)
    valid_loader = make_dataloader(X_va_num, X_va_cat, y[valid_idx], cfg.batch_size, False)

    model = _make_model(encoder, X_tr_num.shape[1], cfg, use_embeddings, device)
    model = _train_model(model, train_loader, cfg, device, use_embeddings, valid_loader)

    oof_preds = _predict_loader(model, valid_loader, device, use_embeddings)
    rmsle = root_mean_squared_error(y[valid_idx], oof_preds)
    return rmsle, oof_preds, model


def cross_validate(
    X_df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    n_splits: int,
    random_state: int,
    device_cfg: str = 'auto',
) -> dict:
    """KFold CV для одной конфигурации DNN; возвращает метрики и OOF-предсказания."""
    device = get_device(device_cfg)
    y_arr = np.asarray(y, dtype=np.float32)

    splitter = build_cv_splitter(
        y_arr, n_splits, random_state, cfg.cv_strategy, cfg.stratify_bins,
    )
    y_bins = _make_stratify_bins(y_arr, cfg.stratify_bins) if cfg.cv_strategy.lower() == 'stratified' else None
    split_iter = (
        splitter.split(X_df, y_bins)
        if isinstance(splitter, StratifiedKFold) else splitter.split(X_df)
    )

    rmsle_folds = []
    oof = np.empty_like(y_arr)

    for fold_i, (train_idx, valid_idx) in enumerate(split_iter):
        rmsle, fold_oof, _ = train_one_fold(
            X_df, y_arr, train_idx, valid_idx,
            numeric_cols, categorical_cols, cfg, device,
            random_state + fold_i,
        )
        oof[valid_idx] = fold_oof
        rmsle_folds.append(rmsle)

    rmsle_folds = np.asarray(rmsle_folds)
    return {
        'name': cfg.name,
        'rmsle_mean': float(rmsle_folds.mean()),
        'rmsle_std': float(rmsle_folds.std()),
        'rmsle_folds': rmsle_folds,
        'y_pred_log': oof,
        'config': cfg,
    }


def fit_full_model(
    X_df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    random_state: int,
    device_cfg: str = 'auto',
) -> tuple[HousePriceMLP, FeatureEncoder]:
    """Обучает DNN на полном train без валидации (для сабмита)."""
    device = get_device(device_cfg)
    seed_everything(random_state, deterministic=True)
    use_embeddings = cfg.cat_encoding.lower() == 'embedding'

    encoder = _fit_encoder(X_df, y, None, numeric_cols, categorical_cols, cfg)
    X_num, X_cat = encoder.transform(X_df, use_embeddings=use_embeddings)
    loader = make_dataloader(X_num, X_cat, np.asarray(y, dtype=np.float32), cfg.batch_size, True)

    model = _make_model(encoder, X_num.shape[1], cfg, use_embeddings, device)
    _train_model(model, loader, cfg, device, use_embeddings)
    return model, encoder


def predict(
    model: HousePriceMLP,
    encoder: FeatureEncoder,
    X_df: pd.DataFrame,
    cfg: TrainConfig,
    device_cfg: str = 'auto',
) -> np.ndarray:
    """Предсказание log-таргета на новых данных обученной моделью."""
    device = get_device(device_cfg)
    use_embeddings = cfg.cat_encoding.lower() == 'embedding'
    X_num, X_cat = encoder.transform(X_df, use_embeddings=use_embeddings)
    loader = make_dataloader(X_num, X_cat, None, batch_size=256, shuffle=False)
    return _predict_loader(model, loader, device, use_embeddings)
