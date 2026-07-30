from __future__ import annotations

import torch

from strata_ot.models import MLPBaseline, StrataOTColumn, StrataOTSurface


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
