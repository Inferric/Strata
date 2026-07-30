from __future__ import annotations

import numpy as np
import torch

from strata_ot.models.components import ProbabilisticHead
from strata_ot.training.forecast import (
    block_bootstrap_improvement,
    fit_scale_factor,
)


def test_softplus_scale_is_positive_and_has_gradient() -> None:
    head = ProbabilisticHead(
        hidden_dim=4,
        scale_parameterization="softplus",
        min_scale=1e-3,
        initial_scale=0.3,
    )
    hidden = torch.randn(8, 4, requires_grad=True)
    output = head(hidden)
    scale = output["log_scale"].exp()
    assert torch.all(scale > 1e-3)
    scale.mean().backward()
    assert head.log_scale.weight.grad is not None
    assert torch.isfinite(head.log_scale.weight.grad).all()


def test_scale_factor_recovers_known_miscalibration() -> None:
    target = np.array([-2.0, -1.0, 1.0, 2.0])
    location = np.zeros(4)
    raw_scale = np.full(4, np.sqrt(2.5) / 2)
    assert np.isclose(fit_scale_factor(target, location, raw_scale), 2.0)


def test_block_bootstrap_detects_better_forecast() -> None:
    target = np.linspace(-1, 1, 240)
    persistence = target + 0.4
    neural = target + 0.1
    result = block_bootstrap_improvement(
        target,
        neural,
        persistence,
        block_rows=24,
        resamples=200,
        seed=17,
    )
    assert result["ci95_low"] > 0
    assert result["mean_relative_improvement"] > 0.5
