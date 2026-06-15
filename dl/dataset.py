"""Подготовка данных и PyTorch DataLoader для DNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


def _cat_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna('None').astype(str)


@dataclass
class FeatureEncoder:
    """
    Кодирует числовые и категориальные признаки для DNN.

    Поддерживает cat_encoding: embedding, onehot, freq, target.
    fit() вызывается только на train-фолде (или полном train для сабмита).
    """

    numeric_cols: list[str]
    categorical_cols: list[str]
    cat_cardinalities: list[int]
    cat_encoding: str = 'freq'
    target_smoothing: float = 20.0
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    cat_maps: dict[str, dict[str, int]] | None = None
    onehot: OneHotEncoder | None = None
    freq_maps: dict[str, dict[str, float]] | None = None
    target_maps: dict[str, dict[str, float]] | None = None
    target_global_mean: float | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> FeatureEncoder:
        """Обучает imputer/scaler и маппинги категорий на train-выборке."""
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.cat_maps = {}
        self.onehot = None
        self.freq_maps = None
        self.target_maps = None
        self.target_global_mean = None

        if self.numeric_cols:
            num = self.imputer.fit_transform(df[self.numeric_cols].values)
            self.scaler.fit(num)

        enc = self.cat_encoding.lower()
        if enc == 'embedding':
            for col in self.categorical_cols:
                values = _cat_series(df, col)
                mapping = {v: i + 1 for i, v in enumerate(sorted(values.unique()))}
                self.cat_maps[col] = mapping
            self.cat_cardinalities = [len(self.cat_maps[c]) for c in self.categorical_cols]
        elif enc == 'onehot' and self.categorical_cols:
            self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            values = df[self.categorical_cols].copy()
            for col in self.categorical_cols:
                values[col] = _cat_series(values, col)
            self.onehot.fit(values.values)
            self.cat_cardinalities = []
        elif enc == 'freq':
            self.freq_maps = {}
            total = len(df)
            for col in self.categorical_cols:
                counts = _cat_series(df, col).value_counts(dropna=False)
                self.freq_maps[col] = (counts / max(total, 1)).to_dict()
            self.cat_cardinalities = []
        elif enc == 'target':
            if y is None:
                raise ValueError("cat_encoding='target' requires y in fit()")
            y_arr = np.asarray(y, dtype=np.float32)
            self.target_global_mean = float(y_arr.mean()) if len(y_arr) else 0.0
            self.target_maps = {}
            for col in self.categorical_cols:
                values = _cat_series(df, col)
                temp = pd.DataFrame({'cat': values.values, 'y': y_arr})
                stats = temp.groupby('cat')['y'].agg(['mean', 'count'])
                smoothed = (
                    (stats['mean'] * stats['count'] + self.target_global_mean * self.target_smoothing)
                    / (stats['count'] + self.target_smoothing)
                )
                self.target_maps[col] = smoothed.to_dict()
            self.cat_cardinalities = []
        else:
            raise ValueError(
                f"Unknown cat_encoding='{self.cat_encoding}'. Use embedding/onehot/target/freq."
            )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        use_embeddings: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Преобразует DataFrame в (numeric, categorical) массивы.

        При embedding + use_embeddings=True возвращает int-индексы для Embedding-слоёв.
        Иначе категории конкатенируются с числовыми в один numeric-вектор.
        """
        numeric = np.zeros((len(df), 0), dtype=np.float32)
        if self.numeric_cols:
            raw = self.imputer.transform(df[self.numeric_cols].values)
            numeric = self.scaler.transform(raw).astype(np.float32)

        if not self.categorical_cols:
            return numeric, None

        enc = self.cat_encoding.lower()
        if enc == 'embedding' and use_embeddings:
            cat = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
            for j, col in enumerate(self.categorical_cols):
                mapping = self.cat_maps.get(col, {})
                cat[:, j] = _cat_series(df, col).map(lambda v: mapping.get(v, 0)).values
            return numeric, cat

        if enc == 'onehot':
            values = df[self.categorical_cols].copy()
            for col in self.categorical_cols:
                values[col] = _cat_series(values, col)
            cat_features = self.onehot.transform(values.values).astype(np.float32)
            return np.concatenate([numeric, cat_features], axis=1), None

        if enc == 'freq':
            cols_out = [
                _cat_series(df, col)
                .map(lambda v, c=col: self.freq_maps[c].get(v, 0.0))
                .values.astype(np.float32)
                .reshape(-1, 1)
                for col in self.categorical_cols
            ]
            return np.concatenate([numeric, *cols_out], axis=1), None

        if enc == 'target':
            cols_out = [
                _cat_series(df, col)
                .map(lambda v, c=col: self.target_maps[c].get(v, self.target_global_mean))
                .values.astype(np.float32)
                .reshape(-1, 1)
                for col in self.categorical_cols
            ]
            return np.concatenate([numeric, *cols_out], axis=1), None

        return numeric, None


class HousePriceDataset(Dataset):
    """PyTorch Dataset: (numeric[, categorical][, target]) тензоры."""

    def __init__(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray | None,
        targets: np.ndarray | None = None,
    ):
        self.numeric = torch.from_numpy(numeric)
        self.categorical = (
            torch.from_numpy(categorical) if categorical is not None else None
        )
        self.targets = (
            torch.from_numpy(targets.astype(np.float32))
            if targets is not None else None
        )

    def __len__(self) -> int:
        return len(self.numeric)

    def __getitem__(self, idx: int):
        items = [self.numeric[idx]]
        if self.categorical is not None:
            items.append(self.categorical[idx])
        if self.targets is not None:
            items.append(self.targets[idx])
        return tuple(items)


def make_dataloader(
    numeric: np.ndarray,
    categorical: np.ndarray | None,
    targets: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """DataLoader поверх HousePriceDataset с заданным batch_size и shuffle."""
    return DataLoader(
        HousePriceDataset(numeric, categorical, targets),
        batch_size=batch_size,
        shuffle=shuffle,
    )
