"""Подготовка данных и PyTorch DataLoader для DNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class FeatureEncoder:
    """Кодирует числовые (impute + scale) и категориальные (label encoding) признаки."""

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

        # Категориальные кодировки
        enc = self.cat_encoding.lower()
        if enc == 'embedding':
            for col in self.categorical_cols:
                values = df[col].fillna('None').astype(str)
                uniq = sorted(values.unique())
                mapping = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = unknown
                self.cat_maps[col] = mapping

            self.cat_cardinalities = [
                len(self.cat_maps[c]) for c in self.categorical_cols
            ]
        elif enc == 'onehot':
            if self.categorical_cols:
                self.onehot = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                )
                values = df[self.categorical_cols].copy()
                for c in self.categorical_cols:
                    values[c] = values[c].fillna('None').astype(str)
                self.onehot.fit(values.values)
            self.cat_cardinalities = []
        elif enc == 'freq':
            self.freq_maps = {}
            total = len(df)
            for col in self.categorical_cols:
                values = df[col].fillna('None').astype(str)
                counts = values.value_counts(dropna=False)
                self.freq_maps[col] = (counts / max(total, 1)).to_dict()
            self.cat_cardinalities = []
        elif enc == 'target':
            if y is None:
                raise ValueError("cat_encoding='target' requires y in fit()")
            y_arr = np.asarray(y, dtype=np.float32)
            self.target_global_mean = float(y_arr.mean()) if len(y_arr) else 0.0
            self.target_maps = {}
            for col in self.categorical_cols:
                values = df[col].fillna('None').astype(str)
                temp = pd.DataFrame({'cat': values.values, 'y': y_arr})
                stats = temp.groupby('cat')['y'].agg(['mean', 'count'])
                # Smoothing по схеме: (mean*count + global*alpha)/(count + alpha)
                smoothed = (
                    (stats['mean'] * stats['count'] + self.target_global_mean * self.target_smoothing)
                    / (stats['count'] + self.target_smoothing)
                )
                self.target_maps[col] = smoothed.to_dict()
            self.cat_cardinalities = []
        else:
            raise ValueError(f"Unknown cat_encoding='{self.cat_encoding}'. Use embedding/onehot/target/freq.")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        use_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
        numeric = np.zeros((len(df), 0), dtype=np.float32)
        if self.numeric_cols:
            raw = self.imputer.transform(df[self.numeric_cols].values)
            numeric = self.scaler.transform(raw).astype(np.float32)

        enc = self.cat_encoding.lower()
        if not self.categorical_cols:
            return numeric, None

        # embedding: возвращаем отдельный тензор индексов
        if enc == 'embedding' and use_embeddings:
            cat = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
            for j, col in enumerate(self.categorical_cols):
                values = df[col].fillna('None').astype(str)
                mapping = self.cat_maps[col] if self.cat_maps is not None else {}
                cat[:, j] = values.map(lambda v: mapping.get(v, 0)).values
            return numeric, cat

        # остальные варианты: кодированные категории добавляем в "numeric"
        if enc == 'onehot':
            if self.onehot is None:
                raise RuntimeError("OneHotEncoder is not fitted")
            values = df[self.categorical_cols].copy()
            for c in self.categorical_cols:
                values[c] = values[c].fillna('None').astype(str)
            cat_features = self.onehot.transform(values.values).astype(np.float32)
            numeric = np.concatenate([numeric, cat_features], axis=1)
            return numeric, None

        if enc == 'freq':
            if self.freq_maps is None:
                raise RuntimeError("freq_maps is not fitted")
            cols_out = []
            for col in self.categorical_cols:
                values = df[col].fillna('None').astype(str)
                col_freq = values.map(lambda v: self.freq_maps[col].get(v, 0.0)).values.astype(np.float32)
                cols_out.append(col_freq.reshape(-1, 1))
            cat_features = np.concatenate(cols_out, axis=1) if cols_out else np.zeros((len(df), 0), dtype=np.float32)
            numeric = np.concatenate([numeric, cat_features], axis=1)
            return numeric, None

        if enc == 'target':
            if self.target_maps is None or self.target_global_mean is None:
                raise RuntimeError("target_maps is not fitted")
            cols_out = []
            for col in self.categorical_cols:
                values = df[col].fillna('None').astype(str)
                mapped = values.map(lambda v: self.target_maps[col].get(v, self.target_global_mean)).values.astype(np.float32)
                cols_out.append(mapped.reshape(-1, 1))
            cat_features = np.concatenate(cols_out, axis=1) if cols_out else np.zeros((len(df), 0), dtype=np.float32)
            numeric = np.concatenate([numeric, cat_features], axis=1)
            return numeric, None

        # Для совместимости: если запросили embedding но cat_encoding не embedding — просто вернём numeric
        return numeric, None


class HousePriceDataset(Dataset):
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
        if self.targets is None:
            if self.categorical is None:
                return self.numeric[idx]
            return self.numeric[idx], self.categorical[idx]
        if self.categorical is None:
            return self.numeric[idx], self.targets[idx]
        return self.numeric[idx], self.categorical[idx], self.targets[idx]


def make_dataloader(
    numeric: np.ndarray,
    categorical: np.ndarray | None,
    targets: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = HousePriceDataset(numeric, categorical, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
