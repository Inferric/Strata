from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from strata_ot.models.strata_horizon import StrataOTHorizon


def flatten_fusion_batch(batch: dict[str, Tensor]) -> Tensor:
    return torch.cat(
        (
            batch["short_history"].flatten(1),
            batch["short_weather"].flatten(1),
            batch["medium_history"].flatten(1),
            batch["medium_weather"].flatten(1),
            batch["slow_history"].flatten(1),
            batch["slow_weather"].flatten(1),
            batch["horizon_minutes"].unsqueeze(-1) / 60.0,
        ),
        dim=-1,
    )


def flatten_fusion_feature_names(weather_names: list[str]) -> list[str]:
    """Return the exact semantic ordering used by ``flatten_fusion_batch``."""
    names: list[str] = []
    for scale, rows, spacing in (
        ("short", 6, 5),
        ("medium", 12, 15),
        ("slow", 24, 60),
    ):
        for row in range(rows):
            lag = (rows - 1 - row) * spacing
            for field in ("cn2_level", "cn2_first_difference", "cadence"):
                names.append(f"{scale}/lag_{lag:04d}m/{field}")
        for row in range(rows):
            lag = (rows - 1 - row) * spacing
            names.extend(
                f"{scale}/lag_{lag:04d}m/{field}" for field in weather_names
            )
    names.append("forecast/horizon_minutes_scaled")
    return names


class FusionControlHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.residual = nn.Linear(hidden_dim, 1)
        self.raw_scale = nn.Linear(hidden_dim, 1)
        self.raw_df = nn.Linear(hidden_dim, 1)
        self.quantile_offsets = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        hidden: Tensor,
        persistence: Tensor,
    ) -> dict[str, Tensor]:
        residual = self.residual(hidden).squeeze(-1)
        location = persistence + residual
        scale = 1e-3 + F.softplus(self.raw_scale(hidden).squeeze(-1))
        degrees_of_freedom = 2.0 + F.softplus(self.raw_df(hidden).squeeze(-1))
        offsets = self.quantile_offsets(hidden)
        median = location + offsets[:, 1]
        quantiles = torch.stack(
            (
                median - F.softplus(offsets[:, 0]),
                median,
                median + F.softplus(offsets[:, 2]),
            ),
            dim=-1,
        )
        return {
            "location": location,
            "log_scale": scale.log(),
            "student_t_scale": scale,
            "student_t_df": degrees_of_freedom,
            "quantiles": quantiles,
            "embedding": hidden,
            "residual": residual,
        }


class FusionMLPControl(nn.Module):
    requires_fusion_batch = True

    def __init__(
        self,
        *,
        weather_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = (
            6 * (3 + weather_dim)
            + 12 * (3 + weather_dim)
            + 24 * (3 + weather_dim)
            + 1
        )
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current = input_dim
        for _ in range(depth):
            layers.extend(
                (
                    nn.Linear(current, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            current = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = FusionControlHead(hidden_dim)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden = self.encoder(flatten_fusion_batch(batch))
        return cast(
            dict[str, Tensor], self.head(hidden, batch["persistence"])
        )


class FusionTCNControl(nn.Module):
    requires_fusion_batch = True

    def __init__(
        self,
        *,
        weather_dim: int,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = weather_dim + 3
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.horizon = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = FusionControlHead(hidden_dim)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        values = torch.cat(
            (batch["slow_history"], batch["slow_weather"]), dim=-1
        )
        hidden = self.projection(values).transpose(1, 2)
        encoded = self.temporal(hidden)[..., : values.shape[1]]
        pooled = encoded[..., -1]
        horizon = self.horizon(batch["horizon_minutes"].unsqueeze(-1) / 60.0)
        representation = self.norm(pooled + horizon)
        return cast(
            dict[str, Tensor],
            self.head(representation, batch["persistence"]),
        )


class HorizonV1FusionAdapter(nn.Module):
    """Cycle 0 architecture on the aligned short USNA context."""

    requires_fusion_batch = True

    def __init__(
        self,
        *,
        weather_dim: int,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = StrataOTHorizon(
            input_dim=weather_dim + 3,
            weather_dim=weather_dim,
            hidden_dim=hidden_dim,
            num_heads=6,
            num_experts=4,
            dropout=dropout,
            weather_enabled=True,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        history = batch["short_history"]
        horizon_channel = (
            batch["horizon_minutes"][:, None, None]
            .expand(-1, history.shape[1], 1)
            / 60.0
        )
        features = torch.cat(
            (
                batch["short_weather"],
                history[..., :1],
                history[..., 2:3],
                horizon_channel,
            ),
            dim=-1,
        )
        output = cast(
            dict[str, Tensor],
            self.model(
                features,
                baseline=batch["persistence"],
                horizon_minutes=batch["horizon_minutes"],
            ),
        )
        output["student_t_scale"] = output["log_scale"].exp()
        output["student_t_df"] = output["location"].new_full(
            output["location"].shape, 30.0
        )
        output["embedding"] = output["regime_weights"]
        output["residual"] = output["location"] - batch["persistence"]
        return output
