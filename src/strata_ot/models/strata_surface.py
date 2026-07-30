from __future__ import annotations

import torch
from torch import Tensor, nn

from strata_ot.models.components import ProbabilisticHead, RegimeMixture


class StrataOTSurface(nn.Module):
    """Compact temporal, regime-routed probabilistic Cn² regressor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        num_experts: int = 4,
        dropout: float = 0.1,
        max_context: int = 256,
        min_log_scale: float = -5.0,
        max_log_scale: float = 2.0,
        scale_parameterization: str = "clamp",
        min_scale: float = 1e-3,
        initial_scale: float = 0.3,
        use_baseline_residual: bool = True,
        **_: object,
    ):
        super().__init__()
        self.use_baseline_residual = use_baseline_residual
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, max_context, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
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
        self.regimes = RegimeMixture(hidden_dim, num_experts, dropout)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = ProbabilisticHead(
            hidden_dim,
            min_log_scale,
            max_log_scale,
            scale_parameterization,
            min_scale,
            initial_scale,
        )

    def forward(self, features: Tensor, baseline: Tensor | None = None) -> dict[str, Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        context = features.shape[1]
        if context > self.position.shape[1]:
            raise ValueError(f"context length {context} exceeds configured maximum")
        hidden = self.input_projection(features) + self.position[:, :context]
        hidden = self.encoder(hidden)
        hidden, regime_weights = self.regimes(hidden)
        attention = self.pool_score(hidden).squeeze(-1).softmax(dim=-1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        outputs: dict[str, Tensor] = self.head(
            self.output_norm(pooled),
            baseline if self.use_baseline_residual else None,
        )
        outputs["regime_weights"] = (
            regime_weights * attention.unsqueeze(-1)
        ).sum(dim=1)
        return outputs
