from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FourierCoordinateEmbedding(nn.Module):
    """Log-coordinate Fourier features for pressure or altitude coordinates."""

    frequencies: Tensor

    def __init__(self, bands: int, min_frequency: float = 1.0, max_frequency: float = 64.0):
        super().__init__()
        frequencies = torch.logspace(
            math.log10(min_frequency), math.log10(max_frequency), bands
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return int(self.frequencies.numel()) * 2

    def forward(self, coordinate: Tensor) -> Tensor:
        normalized = torch.log(coordinate.clamp_min(1e-6))
        normalized = (normalized - normalized.mean(dim=-1, keepdim=True)) / (
            normalized.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        )
        phase = normalized.unsqueeze(-1) * self.frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class RegimeMixture(nn.Module):
    """Small differentiable mixture-of-experts for turbulence-regime specialization."""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float):
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.router(hidden).softmax(dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        mixed = (expert_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden + mixed, weights


class ProbabilisticHead(nn.Module):
    """Gaussian location/scale plus ordered 10/50/90% quantiles."""

    def __init__(
        self,
        hidden_dim: int,
        min_log_scale: float = -5.0,
        max_log_scale: float = 2.0,
        scale_parameterization: str = "clamp",
        min_scale: float = 1e-3,
        initial_scale: float = 0.3,
    ):
        super().__init__()
        if scale_parameterization not in {"clamp", "softplus"}:
            raise ValueError("scale_parameterization must be 'clamp' or 'softplus'")
        if min_scale <= 0 or initial_scale <= min_scale:
            raise ValueError("Require 0 < min_scale < initial_scale")
        self.location = nn.Linear(hidden_dim, 1)
        self.log_scale = nn.Linear(hidden_dim, 1)
        self.quantile_offsets = nn.Linear(hidden_dim, 3)
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale
        self.scale_parameterization = scale_parameterization
        self.min_scale = min_scale
        if scale_parameterization == "softplus":
            nn.init.zeros_(self.log_scale.weight)
            inverse_softplus = math.log(math.expm1(initial_scale - min_scale))
            nn.init.constant_(self.log_scale.bias, inverse_softplus)

    def forward(self, hidden: Tensor, baseline: Tensor | None = None) -> dict[str, Tensor]:
        location = self.location(hidden).squeeze(-1)
        if baseline is not None:
            location = location + baseline
        raw_scale = self.log_scale(hidden).squeeze(-1)
        if self.scale_parameterization == "softplus":
            log_scale = (self.min_scale + F.softplus(raw_scale)).log()
        else:
            log_scale = raw_scale.clamp(self.min_log_scale, self.max_log_scale)
        raw = self.quantile_offsets(hidden)
        median = location + raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        return {
            "location": location,
            "log_scale": log_scale,
            "quantiles": torch.stack((lower, median, upper), dim=-1),
        }


def pinball_loss(predictions: Tensor, target: Tensor) -> Tensor:
    levels = predictions.new_tensor([0.10, 0.50, 0.90])
    error = target.unsqueeze(-1) - predictions
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def gaussian_nll(location: Tensor, log_scale: Tensor, target: Tensor) -> Tensor:
    inverse_variance = torch.exp(-2.0 * log_scale)
    return (log_scale + 0.5 * (target - location).square() * inverse_variance).mean()
