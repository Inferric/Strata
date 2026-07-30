from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from strata_ot.models.components import ProbabilisticHead, RegimeMixture


class CausalGatedTemporalBlock(nn.Module):
    """Residual gated convolution with explicit left-only causal padding."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 2 or dilation < 1:
            raise ValueError("Causal blocks require kernel_size >= 2 and dilation >= 1")
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.filter_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
        )
        self.gate_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            dilation=dilation,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        normalized = self.norm(hidden).transpose(1, 2)
        padded = F.pad(normalized, (self.left_padding, 0))
        filtered = torch.tanh(self.filter_conv(padded))
        gate = torch.sigmoid(self.gate_conv(padded))
        update = self.output((filtered * gate).transpose(1, 2))
        return cast(Tensor, residual + update)


class HorizonFourierEmbedding(nn.Module):
    """Fixed Fourier forecast-minute features followed by a learned projection."""

    frequencies: Tensor

    def __init__(self, hidden_dim: int, bands: int) -> None:
        super().__init__()
        if bands < 1:
            raise ValueError("At least one horizon Fourier band is required")
        self.register_buffer(
            "frequencies",
            torch.logspace(0, math.log10(16.0), bands),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(1 + bands * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, horizon_minutes: Tensor) -> Tensor:
        normalized = horizon_minutes.float().unsqueeze(-1) / 60.0
        phase = 2.0 * math.pi * normalized * self.frequencies
        return cast(
            Tensor,
            self.projection(
                torch.cat((normalized, phase.sin(), phase.cos()), dim=-1)
            ),
        )


class StrataOTHorizon(nn.Module):
    """Persistence-anchored endogenous/exogenous multi-horizon adaptation."""

    requires_horizon_minutes = True

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 192,
        weather_dim: int = 7,
        num_heads: int = 6,
        num_experts: int = 4,
        dropout: float = 0.1,
        horizon_fourier_bands: int = 8,
        causal_kernel_size: int = 3,
        causal_dilations: list[int] | tuple[int, ...] = (1, 2),
        weather_enabled: bool = True,
        min_log_scale: float = -5.0,
        max_log_scale: float = 2.0,
        scale_parameterization: str = "softplus",
        min_scale: float = 1e-3,
        initial_scale: float = 0.3,
        **_: object,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if input_dim != weather_dim + 3:
            raise ValueError(
                "Horizon input layout must be weather, history, cadence, horizon"
            )
        self.weather_dim = weather_dim
        self.weather_enabled = weather_enabled
        self.history_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
        )
        self.weather_projection = nn.Sequential(
            nn.LayerNorm(weather_dim * 2),
            nn.Linear(weather_dim * 2, hidden_dim),
        )
        self.history_blocks = nn.ModuleList(
            [
                CausalGatedTemporalBlock(
                    hidden_dim,
                    kernel_size=causal_kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in causal_dilations
            ]
        )
        self.weather_blocks = nn.ModuleList(
            [
                CausalGatedTemporalBlock(
                    hidden_dim,
                    kernel_size=causal_kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in causal_dilations
            ]
        )
        self.horizon_embedding = HorizonFourierEmbedding(
            hidden_dim, horizon_fourier_bands
        )
        self.history_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.weather_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.regimes = RegimeMixture(hidden_dim, num_experts, dropout)
        self.history_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.weather_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.weather_gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head = ProbabilisticHead(
            hidden_dim,
            min_log_scale=min_log_scale,
            max_log_scale=max_log_scale,
            scale_parameterization=scale_parameterization,
            min_scale=min_scale,
            initial_scale=initial_scale,
        )

    @staticmethod
    def _first_difference(values: Tensor) -> Tensor:
        first = torch.zeros_like(values[:, :1])
        return torch.cat((first, values[:, 1:] - values[:, :-1]), dim=1)

    def forward(
        self,
        features: Tensor,
        baseline: Tensor | None = None,
        *,
        horizon_minutes: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        if baseline is None:
            raise ValueError("Strata-OT Horizon requires the persistence baseline")
        if horizon_minutes is None:
            raise ValueError("Strata-OT Horizon requires forecast minutes")
        weather = features[..., : self.weather_dim]
        history_level = features[..., self.weather_dim : self.weather_dim + 1]
        cadence = features[..., self.weather_dim + 1 : self.weather_dim + 2]
        history_input = torch.cat(
            (history_level, self._first_difference(history_level), cadence), dim=-1
        )
        history_hidden = self.history_projection(history_input)
        for block in self.history_blocks:
            history_hidden = block(history_hidden)

        horizon = self.horizon_embedding(horizon_minutes)
        query = horizon.unsqueeze(1)
        history_summary, history_attention = self.history_attention(
            query,
            history_hidden,
            history_hidden,
            need_weights=True,
            average_attn_weights=False,
        )
        history_summary = history_summary.squeeze(1)

        if self.weather_enabled:
            weather_input = torch.cat(
                (weather, self._first_difference(weather)), dim=-1
            )
            weather_hidden = self.weather_projection(weather_input)
            for block in self.weather_blocks:
                weather_hidden = block(weather_hidden)
            weather_summary, weather_attention = self.weather_attention(
                query,
                weather_hidden,
                weather_hidden,
                need_weights=True,
                average_attn_weights=False,
            )
            weather_summary = weather_summary.squeeze(1)
        else:
            weather_summary = torch.zeros_like(history_summary)
            weather_attention = features.new_zeros(
                (
                    features.shape[0],
                    self.history_attention.num_heads,
                    1,
                    features.shape[1],
                )
            )

        fused = self.fusion(
            torch.cat((history_summary, weather_summary, horizon), dim=-1)
        )
        routed, regime_weights = self.regimes(fused)
        history_delta = self.history_delta_head(
            routed + history_summary
        ).squeeze(-1)
        if self.weather_enabled:
            weather_delta = self.weather_delta_head(
                routed + weather_summary
            ).squeeze(-1)
            weather_gate = torch.sigmoid(
                self.weather_gate_head(routed + weather_summary).squeeze(-1)
            )
            weather_contribution = weather_gate * weather_delta
        else:
            weather_delta = torch.zeros_like(history_delta)
            weather_gate = torch.zeros_like(history_delta)
            weather_contribution = torch.zeros_like(history_delta)
        location = baseline + history_delta + weather_contribution
        output = cast(
            dict[str, Tensor],
            self.head(
                self.output_norm(routed),
                location_override=location,
            ),
        )
        output.update(
            {
                "history_delta": history_delta,
                "weather_delta": weather_delta,
                "weather_gate": weather_gate,
                "weather_contribution": weather_contribution,
                "regime_weights": regime_weights,
                "history_attention": history_attention.squeeze(2),
                "weather_attention": weather_attention.squeeze(2),
            }
        )
        return output
