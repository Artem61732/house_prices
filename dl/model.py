"""Конфигурируемая MLP-модель с опциональными Embedding, BatchNorm и Dropout."""

from __future__ import annotations

import torch
import torch.nn as nn


ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}


def _embedding_dim(cardinality: int) -> int:
    """Эвристика размера embedding: min(50, ceil(sqrt(cardinality)) + 1)."""
    return min(50, int(cardinality ** 0.5) + 1)


class HousePriceMLP(nn.Module):
    """
    MLP для регрессии House Prices.

    Поддерживает:
    - произвольное число скрытых слоёв;
    - BatchNorm и Dropout;
    - Embedding для категориальных признаков (задание со звёздочкой).
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int] | None = None,
        embedding_dims: list[int] | None = None,
        hidden_layers: list[int] | None = None,
        activation: str = 'relu',
        batch_norm: bool = False,
        dropout: float = 0.0,
        use_embeddings: bool = True,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 64]
        cat_cardinalities = cat_cardinalities or []
        use_embeddings = use_embeddings and len(cat_cardinalities) > 0

        self.use_embeddings = use_embeddings
        self.embeddings = nn.ModuleList()
        embed_out = 0

        if use_embeddings:
            if embedding_dims is None:
                embedding_dims = [_embedding_dim(c) for c in cat_cardinalities]
            for card, dim in zip(cat_cardinalities, embedding_dims):
                self.embeddings.append(nn.Embedding(card + 1, dim, padding_idx=0))
                embed_out += dim

        input_dim = n_numeric + embed_out
        act_cls = ACTIVATIONS.get(activation, nn.ReLU)

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden

        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [numeric]
        if self.use_embeddings and categorical is not None:
            embeds = [
                emb(categorical[:, i])
                for i, emb in enumerate(self.embeddings)
            ]
            parts.append(torch.cat(embeds, dim=1))
        x = torch.cat(parts, dim=1)
        return self.mlp(x).squeeze(-1)
