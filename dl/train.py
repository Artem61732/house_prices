"""
Обучение DNN: оптимизаторы, scheduler, early stopping, KFold CV.

Метрика CV — RMSLE на log1p(SalePrice). Таргет обучается в log-шкале,
поэтому RMSLE = RMSE между y_log и pred_log (root_mean_squared_error).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

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
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
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
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not cfg.scheduler:
        return None
    name = cfg.scheduler.lower()
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs,
        )
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    if name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5,
        )
    return None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Делает обучение воспроизводимым (насколько это возможно).

    deterministic=True может немного замедлить обучение на GPU.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_stratify_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Строит квантильные бины для регрессии, чтобы использовать StratifiedKFold.
    Работает устойчиво даже при одинаковых значениях (дедуплицирует границы).
    """
    y = np.asarray(y, dtype=np.float32)
    if n_bins < 2:
        return np.zeros_like(y, dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(y, dtype=np.int64)

    # digitize: (edges[0], ..., edges[-1]) -> bins 0..k-2
    bins = np.digitize(y, edges[1:-1], right=True).astype(np.int64)
    return bins


def build_cv_splitter(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    strategy: str = 'stratified',
    stratify_bins: int = 10,
):
    strategy = (strategy or 'kfold').lower()
    if strategy not in CV_STRATEGIES:
        raise ValueError(f"Unknown cv_strategy='{strategy}'. Use: {sorted(CV_STRATEGIES)}")

    if strategy == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # stratified regression via quantile bins
    y_bins = _make_stratify_bins(y, stratify_bins)
    # если классов получилось мало — откатываемся в KFold, чтобы не падать
    unique_bins = np.unique(y_bins)
    if len(unique_bins) < 2 or np.min(np.bincount(y_bins)) < n_splits:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _forward_batch(model, batch, device, use_embeddings: bool):
    if use_embeddings:
        numeric, categorical, targets = batch
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        targets = targets.to(device)
        preds = model(numeric, categorical)
    else:
        numeric, targets = batch
        numeric = numeric.to(device)
        targets = targets.to(device)
        preds = model(numeric)
    return preds, targets


def train_one_fold(
    X_df,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    device: torch.device,
    random_state: int,
) -> tuple[float, np.ndarray, HousePriceMLP]:
    seed_everything(random_state, deterministic=True)

    use_embeddings = cfg.cat_encoding.lower() == 'embedding'
    encoder = FeatureEncoder(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_cardinalities=[],
        cat_encoding=cfg.cat_encoding,
    )
    if cfg.cat_encoding.lower() == 'target':
        encoder.fit(X_df.iloc[train_idx], y=y[train_idx])
    else:
        encoder.fit(X_df.iloc[train_idx])

    X_tr_num, X_tr_cat = encoder.transform(
        X_df.iloc[train_idx], use_embeddings=use_embeddings,
    )
    X_va_num, X_va_cat = encoder.transform(
        X_df.iloc[valid_idx], use_embeddings=use_embeddings,
    )
    y_tr, y_va = y[train_idx], y[valid_idx]

    train_loader = make_dataloader(
        X_tr_num, X_tr_cat, y_tr, cfg.batch_size, shuffle=True,
    )
    valid_loader = make_dataloader(
        X_va_num, X_va_cat, y_va, cfg.batch_size, shuffle=False,
    )

    model = HousePriceMLP(
        n_numeric=X_tr_num.shape[1],
        cat_cardinalities=encoder.cat_cardinalities,
        hidden_layers=cfg.hidden_layers,
        activation=cfg.activation,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        use_embeddings=use_embeddings,
    ).to(device)

    criterion = LOSS_FNS.get(cfg.loss_fn, nn.MSELoss)()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    best_state = None
    best_rmsle = float('inf')
    epochs_no_improve = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds, targets = _forward_batch(model, batch, device, use_embeddings)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in valid_loader:
                preds, targets = _forward_batch(
                    model, batch, device, use_embeddings,
                )
                val_preds.append(preds.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
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
    oof_preds = np.empty(len(valid_idx), dtype=np.float32)
    with torch.no_grad():
        offset = 0
        for batch in valid_loader:
            if use_embeddings:
                numeric, categorical, _ = batch
                numeric = numeric.to(device)
                categorical = categorical.to(device)
                preds = model(numeric, categorical).cpu().numpy()
            else:
                numeric, _ = batch
                numeric = numeric.to(device)
                preds = model(numeric).cpu().numpy()
            bs = len(preds)
            oof_preds[offset:offset + bs] = preds
            offset += bs

    return best_rmsle, oof_preds, model


def cross_validate(
    X_df,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    n_splits: int,
    random_state: int,
    device_cfg: str = 'auto',
) -> dict:
    device = get_device(device_cfg)
    y_arr = np.asarray(y, dtype=np.float32)

    splitter = build_cv_splitter(
        y_arr,
        n_splits=n_splits,
        random_state=random_state,
        strategy=cfg.cv_strategy,
        stratify_bins=cfg.stratify_bins,
    )
    y_bins = None
    if cfg.cv_strategy.lower() == 'stratified':
        y_bins = _make_stratify_bins(y_arr, cfg.stratify_bins)

    rmsle_folds = []
    oof = np.empty_like(y_arr)

    if isinstance(splitter, StratifiedKFold):
        split_iter = splitter.split(X_df, y_bins)
    else:
        split_iter = splitter.split(X_df)

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
    X_df,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
    random_state: int,
    device_cfg: str = 'auto',
) -> tuple[HousePriceMLP, FeatureEncoder]:
    """Обучает модель на всём train для инференса на test."""
    device = get_device(device_cfg)
    seed_everything(random_state, deterministic=True)

    encoder = FeatureEncoder(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_cardinalities=[],
        cat_encoding=cfg.cat_encoding,
    )
    if cfg.cat_encoding.lower() == 'target':
        encoder.fit(X_df, y=y)
    else:
        encoder.fit(X_df)

    use_embeddings = cfg.cat_encoding.lower() == 'embedding'
    X_num, X_cat = encoder.transform(X_df, use_embeddings=use_embeddings)
    y_arr = np.asarray(y, dtype=np.float32)

    loader = make_dataloader(X_num, X_cat, y_arr, cfg.batch_size, shuffle=True)

    model = HousePriceMLP(
        n_numeric=X_num.shape[1],
        cat_cardinalities=encoder.cat_cardinalities,
        hidden_layers=cfg.hidden_layers,
        activation=cfg.activation,
        batch_norm=cfg.batch_norm,
        dropout=cfg.dropout,
        use_embeddings=use_embeddings,
    ).to(device)

    criterion = LOSS_FNS.get(cfg.loss_fn, nn.MSELoss)()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loader))

    for _ in range(cfg.n_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            preds, targets = _forward_batch(model, batch, device, use_embeddings)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            scheduler.step()

    model.eval()
    return model, encoder


def predict(
    model: HousePriceMLP,
    encoder: FeatureEncoder,
    X_df,
    cfg: TrainConfig,
    device_cfg: str = 'auto',
) -> np.ndarray:
    device = get_device(device_cfg)
    use_embeddings = cfg.cat_encoding.lower() == 'embedding'
    X_num, X_cat = encoder.transform(X_df, use_embeddings=use_embeddings)
    loader = make_dataloader(X_num, X_cat, None, batch_size=256, shuffle=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if use_embeddings:
                numeric, categorical = batch
                numeric = numeric.to(device)
                categorical = categorical.to(device)
                out = model(numeric, categorical).cpu().numpy()
            else:
                numeric = batch.to(device)
                out = model(numeric).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)
