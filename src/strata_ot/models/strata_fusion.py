from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from strata_ot.models.components import RegimeMixture
from strata_ot.models.strata_horizon import HorizonFourierEmbedding


class WeatherConditionedCausalBlock(nn.Module):
    """Causal history block with optional pointwise FiLM weather modulation."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dilation: int,
        dropout: float,
        use_film: bool,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        self.left_padding = 2 * dilation
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.weather_norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.filter_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=dilation)
        self.gate_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=dilation)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, history: Tensor, weather: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.history_norm(history)
        if self.use_film:
            gamma, beta = self.film(self.weather_norm(weather)).chunk(2, dim=-1)
            normalized = normalized * (1.0 + 0.1 * torch.tanh(gamma)) + beta
            modulation_norm = torch.sqrt(
                gamma.square().mean(dim=(-1, -2)) + beta.square().mean(dim=(-1, -2)) + 1e-12
            )
        else:
            modulation_norm = history.new_zeros(history.shape[0])
        channel_first = F.pad(normalized.transpose(1, 2), (self.left_padding, 0))
        update = torch.tanh(self.filter_conv(channel_first))
        update = update * torch.sigmoid(self.gate_conv(channel_first))
        return history + self.output(update.transpose(1, 2)), modulation_norm


class ScaleEncoder(nn.Module):
    def __init__(
        self,
        weather_dim: int,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float,
        fusion: str,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        history_input_dim = 3 + (weather_dim if fusion == "concat" else 0)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(history_input_dim),
            nn.Linear(history_input_dim, hidden_dim),
        )
        self.weather_projection = nn.Sequential(
            nn.LayerNorm(weather_dim),
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                WeatherConditionedCausalBlock(
                    hidden_dim,
                    dilation=dilation,
                    dropout=dropout,
                    use_film=fusion in {"film", "film_cross_attention"},
                )
                for dilation in (1, 2)
            ]
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        history: Tensor,
        weather: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weather_hidden = self.weather_projection(weather)
        history_input = (
            torch.cat((history, weather), dim=-1) if self.fusion == "concat" else history
        )
        hidden = self.history_projection(history_input)
        modulation: list[Tensor] = []
        for block in self.blocks:
            hidden, norm = block(hidden, weather_hidden)
            modulation.append(norm)
        if self.fusion == "film_cross_attention":
            attended, weights = self.cross_attention(
                hidden,
                weather_hidden,
                weather_hidden,
                need_weights=True,
                average_attn_weights=False,
            )
            hidden = self.cross_norm(hidden + attended)
        else:
            weights = hidden.new_zeros(
                hidden.shape[0],
                self.cross_attention.num_heads,
                hidden.shape[1],
                hidden.shape[1],
            )
        return (
            hidden,
            weather_hidden,
            weights,
            torch.stack(modulation, dim=-1).mean(dim=-1),
        )


class HorizonBilinearHistoryShortcut(nn.Module):
    """Additive history shortcut with a low-rank horizon interaction."""

    def __init__(
        self,
        *,
        horizon_dim: int,
        history_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Horizon-bilinear shortcut rank must be positive")
        self.additive = nn.Linear(horizon_dim + history_dim, 1)
        self.history_projection = nn.Linear(history_dim, rank, bias=False)
        self.horizon_projection = nn.Linear(horizon_dim, rank, bias=False)
        self.interaction = nn.Linear(rank, 1, bias=False)
        nn.init.zeros_(self.additive.weight)
        nn.init.zeros_(self.additive.bias)
        nn.init.zeros_(self.interaction.weight)

    def forward(
        self,
        horizon: Tensor,
        history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        additive = self.additive(torch.cat((horizon, history), dim=-1))
        interaction_features = torch.tanh(self.history_projection(history)) * torch.tanh(
            self.horizon_projection(horizon)
        )
        interaction = self.interaction(interaction_features)
        return additive, interaction


class StrataOTFusionV2(nn.Module):
    """Multiscale persistence-residual forecaster with deep weather fusion."""

    requires_fusion_batch = True
    scales = ("short", "medium", "slow")
    scale_rows = {"short": 6, "medium": 12, "slow": 24}

    def __init__(
        self,
        *,
        weather_dim: int,
        hidden_dim: int = 192,
        num_heads: int = 6,
        num_experts: int = 4,
        dropout: float = 0.1,
        fusion: str = "film_cross_attention",
        active_scales: tuple[str, ...] = ("short", "medium", "slow"),
        horizon_fourier_bands: int = 8,
        physics_start: int | None = None,
        residual_horizon_exponent: float = 0.0,
        residual_shortcut: str = "none",
        residual_cap: float = 0.0,
        shortcut_rank: int = 0,
    ) -> None:
        super().__init__()
        if fusion not in {"concat", "film", "film_cross_attention"}:
            raise ValueError("Unsupported Fusion v2 weather mechanism")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not active_scales or not set(active_scales).issubset(self.scales):
            raise ValueError("Fusion v2 requires known active scales")
        if "short" not in active_scales:
            raise ValueError("Fusion v2 keeps the short branch as its anchor")
        if not 0.0 <= residual_horizon_exponent <= 1.5:
            raise ValueError("residual_horizon_exponent must be between 0 and 1.5")
        if not 0.0 <= residual_cap <= 1.5:
            raise ValueError("residual_cap must be between 0 and 1.5")
        if residual_shortcut not in {
            "none",
            "history_linear",
            "history_bilinear",
            "all_linear",
            "all_mlp",
        }:
            raise ValueError("Unsupported Fusion v2 residual shortcut")
        if shortcut_rank < 0:
            raise ValueError("shortcut_rank cannot be negative")
        if residual_shortcut == "history_bilinear" and shortcut_rank <= 0:
            raise ValueError("history_bilinear requires a positive shortcut_rank")
        if residual_shortcut != "history_bilinear" and shortcut_rank != 0:
            raise ValueError("shortcut_rank is only valid for history_bilinear")
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.fusion = fusion
        self.active_scales = active_scales
        self.physics_start = physics_start
        self.residual_horizon_exponent = float(residual_horizon_exponent)
        self.residual_cap = float(residual_cap)
        self.residual_shortcut_mode = residual_shortcut
        self.shortcut_rank = int(shortcut_rank)
        self.scale_encoders = nn.ModuleDict(
            {
                scale: ScaleEncoder(
                    weather_dim,
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    fusion=fusion,
                )
                for scale in self.scales
            }
        )
        self.horizon_embedding = HorizonFourierEmbedding(hidden_dim, horizon_fourier_bands)
        self.horizon_attention = nn.ModuleDict(
            {
                scale: nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for scale in self.scales
            }
        )
        self.scale_router = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.regimes = RegimeMixture(hidden_dim, num_experts, dropout)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        history_shortcut_dim = sum(self.scale_rows[scale] * 3 for scale in active_scales)
        weather_shortcut_dim = sum(self.scale_rows[scale] * weather_dim for scale in active_scales)
        shortcut_dim = hidden_dim + history_shortcut_dim
        if residual_shortcut in {"all_linear", "all_mlp"}:
            shortcut_dim += weather_shortcut_dim
        self.residual_shortcut: nn.Module | None
        if residual_shortcut == "none":
            self.residual_shortcut = None
        elif residual_shortcut == "history_bilinear":
            self.residual_shortcut = HorizonBilinearHistoryShortcut(
                horizon_dim=hidden_dim,
                history_dim=history_shortcut_dim,
                rank=shortcut_rank,
            )
        elif residual_shortcut in {"history_linear", "all_linear"}:
            self.residual_shortcut = nn.Linear(shortcut_dim, 1)
            nn.init.zeros_(self.residual_shortcut.weight)
            nn.init.zeros_(self.residual_shortcut.bias)
        else:
            self.residual_shortcut = nn.Sequential(
                nn.LayerNorm(shortcut_dim),
                nn.Linear(shortcut_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            final = self.residual_shortcut[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("Residual shortcut must end in a linear layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.raw_scale = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.raw_df = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.quantile_offsets = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.future_weather_head = nn.Linear(hidden_dim, weather_dim)
        self.weather_reconstruction = nn.Linear(hidden_dim, weather_dim)
        self.cadence_cn2_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        masked_short_weather: Tensor | None = None,
    ) -> dict[str, Tensor]:
        horizon = self.horizon_embedding(batch["horizon_minutes"])
        query = horizon.unsqueeze(1)
        summaries: list[Tensor] = []
        scale_logits: list[Tensor] = []
        horizon_attentions: list[Tensor] = []
        weather_attentions: list[Tensor] = []
        modulation_norms: list[Tensor] = []
        short_weather_hidden: Tensor | None = None
        for scale_name in self.active_scales:
            weather = (
                masked_short_weather
                if scale_name == "short" and masked_short_weather is not None
                else batch[f"{scale_name}_weather"]
            )
            encoded, weather_hidden, weather_attention, modulation = self.scale_encoders[
                scale_name
            ](
                batch[f"{scale_name}_history"],
                weather,
            )
            if scale_name == "short":
                short_weather_hidden = weather_hidden
            summary, horizon_attention = self.horizon_attention[scale_name](
                query,
                encoded,
                encoded,
                need_weights=True,
                average_attn_weights=False,
            )
            summary = summary.squeeze(1)
            summaries.append(summary)
            scale_logits.append(
                self.scale_router(torch.cat((summary, horizon), dim=-1)).squeeze(-1)
            )
            horizon_attentions.append(horizon_attention.squeeze(2))
            weather_attentions.append(weather_attention.mean(dim=-2))
            modulation_norms.append(modulation)
        weights = torch.stack(scale_logits, dim=-1).softmax(dim=-1)
        stacked = torch.stack(summaries, dim=1)
        fused = self.fusion_norm((stacked * weights.unsqueeze(-1)).sum(dim=1) + horizon)
        routed, expert_weights = self.regimes(fused)
        base_residual = self.residual_head(routed).squeeze(-1)
        if self.residual_shortcut is None:
            shortcut_residual = base_residual.new_zeros(len(base_residual))
            shortcut_additive_residual = shortcut_residual
            shortcut_interaction_residual = shortcut_residual
        else:
            history_parts = [
                batch[f"{scale}_history"].flatten(start_dim=1) for scale in self.active_scales
            ]
            history_input = torch.cat(history_parts, dim=-1)
            shortcut_parts = [horizon, history_input]
            if self.residual_shortcut_mode in {"all_linear", "all_mlp"}:
                shortcut_parts.extend(
                    batch[f"{scale}_weather"].flatten(start_dim=1) for scale in self.active_scales
                )
            if isinstance(
                self.residual_shortcut,
                HorizonBilinearHistoryShortcut,
            ):
                additive, interaction = self.residual_shortcut(
                    horizon,
                    history_input,
                )
                shortcut_additive_residual = additive.squeeze(-1)
                shortcut_interaction_residual = interaction.squeeze(-1)
                shortcut_residual = shortcut_additive_residual + shortcut_interaction_residual
            else:
                shortcut_input = torch.cat(shortcut_parts, dim=-1)
                shortcut_residual = self.residual_shortcut(shortcut_input).squeeze(-1)
                shortcut_additive_residual = shortcut_residual
                shortcut_interaction_residual = shortcut_residual.new_zeros(len(shortcut_residual))
        raw_residual = base_residual + shortcut_residual
        bounded_residual = (
            raw_residual.clamp(-self.residual_cap, self.residual_cap)
            if self.residual_cap > 0.0
            else raw_residual
        )
        residual_scale = (batch["horizon_minutes"].clamp_min(5.0) / 5.0).pow(
            self.residual_horizon_exponent
        )
        residual = bounded_residual * residual_scale
        location = batch["persistence"] + residual
        predictive_scale = 1e-3 + F.softplus(self.raw_scale(routed).squeeze(-1))
        degrees_of_freedom = 2.0 + F.softplus(self.raw_df(routed).squeeze(-1))
        offsets = self.quantile_offsets(routed)
        median = location + offsets[:, 1]
        quantiles = torch.stack(
            (
                median - F.softplus(offsets[:, 0]),
                median,
                median + F.softplus(offsets[:, 2]),
            ),
            dim=-1,
        )
        if short_weather_hidden is None:
            raise RuntimeError("Short scale must remain active for Fusion v2")
        physics_norm = (
            batch["short_weather"][..., self.physics_start :].norm(dim=-1).mean(dim=-1)
            if self.physics_start is not None
            else location.new_zeros(len(location))
        )
        return {
            "location": location,
            "log_scale": predictive_scale.log(),
            "student_t_scale": predictive_scale,
            "student_t_df": degrees_of_freedom,
            "quantiles": quantiles,
            "embedding": routed,
            "base_residual": base_residual,
            "shortcut_residual": shortcut_residual,
            "shortcut_additive_residual": shortcut_additive_residual,
            "shortcut_interaction_residual": shortcut_interaction_residual,
            "raw_residual": raw_residual,
            "residual_scale": residual_scale,
            "residual": residual,
            "scale_weights": weights,
            "horizon_attention": torch.cat(horizon_attentions, dim=-1),
            "weather_attention": torch.cat(weather_attentions, dim=-1),
            "modulation_norm": torch.stack(modulation_norms, dim=-1),
            "physics_token_norm": physics_norm,
            "expert_weights": expert_weights,
            "weather_reconstruction": self.weather_reconstruction(short_weather_hidden),
            "future_weather_prediction": self.future_weather_head(routed),
            "cadence_cn2_prediction": self.cadence_cn2_head(routed).squeeze(-1),
        }


def student_t_nll(
    location: Tensor,
    scale: Tensor,
    degrees_of_freedom: Tensor,
    target: Tensor,
) -> Tensor:
    distribution = torch.distributions.StudentT(
        df=degrees_of_freedom,
        loc=location,
        scale=scale,
    )
    return cast(Tensor, -distribution.log_prob(target).mean())  # type: ignore[no-untyped-call]
