from __future__ import annotations

import torch

from strata_ot.evaluation.fusion_horizon_bilinear import (
    BILINEAR_PARAMETERS,
    _select_bilinear_candidate,
)
from strata_ot.models.strata_fusion import HorizonBilinearHistoryShortcut


def _row(
    candidate_id: str,
    *,
    rank: int,
    rmse: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rank": rank,
        "primary": {"rmse_log10_cn2": rmse},
    }


def test_horizon_bilinear_shortcut_is_zero_initialized() -> None:
    torch.manual_seed(11)
    module = HorizonBilinearHistoryShortcut(
        horizon_dim=192,
        history_dim=126,
        rank=8,
    )
    additive, interaction = module(
        torch.randn(5, 192),
        torch.randn(5, 126),
    )
    assert torch.count_nonzero(additive) == 0
    assert torch.count_nonzero(interaction) == 0


def test_horizon_bilinear_parameter_counts_are_frozen() -> None:
    assert list(BILINEAR_PARAMETERS.values()) == [
        3_611_849,
        3_613_125,
        3_614_401,
        3_616_953,
    ]


def test_horizon_bilinear_tie_prefers_lower_rank() -> None:
    lower = _row("horizon-bilinear-r4", rank=4, rmse=0.2955)
    higher = _row("horizon-bilinear-r8", rank=8, rmse=0.2950)
    decisions = {
        "horizon-bilinear-r4": {"advances": True},
        "horizon-bilinear-r8": {"advances": True},
    }
    selected = _select_bilinear_candidate([lower, higher], decisions)
    assert selected is not None
    assert selected["candidate_id"] == "horizon-bilinear-r4"
