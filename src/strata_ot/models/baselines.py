from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn

from strata_ot.models.components import ProbabilisticHead


class MLPBaseline(nn.Module):
    """Parameter-conscious neural baseline; trees remain diagnostic comparators only."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 4, **_: object):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        width = input_dim
        for _index in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(0.1)))
            width = hidden_dim
        self.network = nn.Sequential(*layers)
        self.head = ProbabilisticHead(hidden_dim)

    def forward(self, features: Tensor, baseline: Tensor | None = None) -> dict[str, Tensor]:
        if features.ndim == 3:
            features = features[:, -1]
        output: dict[str, Tensor] = self.head(self.network(features), baseline)
        return output


@dataclass(frozen=True)
class ScalarBaseline:
    name: str
    value: float

    def predict(self, count: int) -> np.ndarray:
        return np.full(count, self.value, dtype=np.float64)


def fit_climatology(target: np.ndarray) -> ScalarBaseline:
    return ScalarBaseline("climatology", float(np.nanmedian(target)))


def persistence_predictions(target: np.ndarray, initial: float) -> np.ndarray:
    if len(target) == 0:
        return target.copy()
    return np.concatenate(([initial], target[:-1]))
