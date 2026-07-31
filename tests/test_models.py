from __future__ import annotations

import torch

from strata_ot.models import (
    MLPBaseline,
    StrataOTColumn,
    StrataOTFusionV2,
    StrataOTHorizon,
    StrataOTSurface,
)
from strata_ot.models.strata_horizon import CausalGatedTemporalBlock


def test_surface_shapes_and_ordered_quantiles() -> None:
    model = StrataOTSurface(
        input_dim=10,
        hidden_dim=48,
        depth=2,
        num_heads=4,
        num_experts=3,
    )
    output = model(torch.randn(4, 6, 10))
    assert output["location"].shape == (4,)
    assert output["log_scale"].shape == (4,)
    assert output["quantiles"].shape == (4, 3)
    assert output["regime_weights"].shape == (4, 3)
    assert torch.all(output["quantiles"][:, 0] <= output["quantiles"][:, 1])
    assert torch.all(output["quantiles"][:, 1] <= output["quantiles"][:, 2])


def test_column_supports_arbitrary_output_grid() -> None:
    model = StrataOTColumn(
        input_dim=8,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_experts=3,
        pressure_fourier_bands=4,
    )
    output = model(
        torch.randn(2, 12, 8),
        torch.linspace(1000, 100, 12).repeat(2, 1),
        torch.linspace(950, 150, 7).repeat(2, 1),
    )
    assert output["location"].shape == (2, 7)
    assert output["quantiles"].shape == (2, 7, 3)
    assert output["regime_weights"].shape == (2, 7, 3)


def test_mlp_is_a_neural_baseline() -> None:
    model = MLPBaseline(input_dim=9, hidden_dim=32, depth=2)
    output = model(torch.randn(5, 1, 9))
    assert output["location"].shape == (5,)


def test_horizon_v1_parameter_budget_and_diagnostics() -> None:
    model = StrataOTHorizon(input_dim=10, weather_enabled=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 5_000_000
    output = model(
        torch.randn(4, 6, 10),
        torch.randn(4),
        horizon_minutes=torch.tensor([5.0, 15.0, 30.0, 60.0]),
    )
    assert output["location"].shape == (4,)
    assert output["regime_weights"].shape == (4, 4)
    assert output["history_attention"].shape == (4, 6, 6)
    assert output["weather_attention"].shape == (4, 6, 6)
    assert torch.all(
        (output["weather_gate"] >= 0) & (output["weather_gate"] <= 1)
    )
    assert torch.isfinite(output["location"]).all()


def test_horizon_history_arm_hard_zeros_weather_term() -> None:
    model = StrataOTHorizon(input_dim=10, weather_enabled=False)
    output = model(
        torch.randn(3, 6, 10),
        torch.randn(3),
        horizon_minutes=torch.tensor([15.0, 30.0, 60.0]),
    )
    assert torch.count_nonzero(output["weather_delta"]) == 0
    assert torch.count_nonzero(output["weather_gate"]) == 0
    assert torch.count_nonzero(output["weather_contribution"]) == 0
    assert torch.count_nonzero(output["weather_attention"]) == 0


def test_fusion_v2_multiscale_outputs_and_parameter_budget() -> None:
    weather_dim = 26
    model = StrataOTFusionV2(
        weather_dim=weather_dim,
        hidden_dim=192,
        num_heads=6,
        num_experts=4,
        fusion="film_cross_attention",
        physics_start=8,
        residual_horizon_exponent=0.5,
        residual_shortcut="all_linear",
    )
    batch_size = 3
    batch = {
        "short_history": torch.randn(batch_size, 6, 3),
        "short_weather": torch.randn(batch_size, 6, weather_dim),
        "medium_history": torch.randn(batch_size, 12, 3),
        "medium_weather": torch.randn(batch_size, 12, weather_dim),
        "slow_history": torch.randn(batch_size, 24, 3),
        "slow_weather": torch.randn(batch_size, 24, weather_dim),
        "persistence": torch.full((batch_size,), -14.5),
        "horizon_minutes": torch.tensor([5.0, 30.0, 60.0]),
    }
    output = model(batch)
    assert output["location"].shape == (batch_size,)
    assert output["quantiles"].shape == (batch_size, 3)
    assert output["scale_weights"].shape == (batch_size, 3)
    assert torch.allclose(
        output["scale_weights"].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5,
    )
    assert torch.all(output["student_t_df"] > 2)
    assert torch.all(output["student_t_scale"] > 0)
    assert torch.allclose(
        output["residual_scale"],
        torch.tensor([1.0, 6.0**0.5, 12.0**0.5]),
    )
    assert torch.allclose(
        output["residual"],
        output["raw_residual"] * output["residual_scale"],
    )
    assert torch.allclose(
        output["raw_residual"],
        output["base_residual"] + output["shortcut_residual"],
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 8_000_000


def test_causal_block_does_not_read_future_tokens() -> None:
    torch.manual_seed(7)
    block = CausalGatedTemporalBlock(
        12,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
    ).eval()
    original = torch.randn(2, 8, 12)
    changed = original.clone()
    changed[:, 5:] += 100
    first = block(original)
    second = block(changed)
    assert torch.allclose(first[:, :5], second[:, :5], atol=1e-6)
