from __future__ import annotations

import torch
from torch import Tensor, nn

from strata_ot.models.components import (
    FourierCoordinateEmbedding,
    ProbabilisticHead,
    RegimeMixture,
)


class StrataOTColumn(nn.Module):
    """Coordinate-aware encoder/decoder for probabilistic vertical Cn² profiles."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        num_experts: int = 4,
        pressure_fourier_bands: int = 16,
        dropout: float = 0.1,
        use_baseline_residual: bool = True,
        **_: object,
    ):
        super().__init__()
        self.use_baseline_residual = use_baseline_residual
        self.coordinate = FourierCoordinateEmbedding(pressure_fourier_bands)
        self.input_projection = nn.Linear(
            input_dim + self.coordinate.output_dim, hidden_dim
        )
        self.local_mixer = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                padding=2,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.query_projection = nn.Linear(self.coordinate.output_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.regimes = RegimeMixture(hidden_dim, num_experts, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = ProbabilisticHead(hidden_dim)

    def forward(
        self,
        level_features: Tensor,
        input_pressure_hpa: Tensor,
        output_pressure_hpa: Tensor,
        baseline: Tensor | None = None,
    ) -> dict[str, Tensor]:
        coordinates = self.coordinate(input_pressure_hpa)
        hidden = self.input_projection(torch.cat((level_features, coordinates), dim=-1))
        hidden = hidden + self.local_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        memory = self.encoder(hidden)
        queries = self.query_projection(self.coordinate(output_pressure_hpa))
        decoded, _ = self.cross_attention(queries, memory, memory, need_weights=False)
        decoded, regime_weights = self.regimes(decoded)
        outputs: dict[str, Tensor] = self.head(
            self.output_norm(decoded),
            baseline if self.use_baseline_residual else None,
        )
        outputs["regime_weights"] = regime_weights
        return outputs
