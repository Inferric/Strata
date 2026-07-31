from __future__ import annotations

import numpy as np
import pytest

from strata_ot.data.fusion import FusionSequenceDataset
from strata_ot.evaluation.fusion_tail_objective import (
    _select_tail_candidate,
    _tail_decision,
)


def _row(candidate_id: str, rmse: float, tail: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "weight": 1.0,
        "primary": {
            "rmse_log10_cn2": rmse,
            "tail_mae_top_decile": tail,
            "crps_gaussian": 0.14,
            "interval_80_coverage": 0.80,
            "bias_log10_cn2": 0.01,
        },
    }


def test_tail_decision_requires_point_and_tail_improvement() -> None:
    control = _row("tail-huber-0p00", 0.30, 0.20)
    passing = _tail_decision(
        control,
        _row("tail-huber-1p00", 0.297, 0.198),
    )
    assert passing["advances"] is True
    missing_tail = _tail_decision(
        control,
        _row("tail-huber-0p50", 0.297, 0.20),
    )
    assert missing_tail["advances"] is False
    assert (
        missing_tail["checks"]["tail_improvement_at_least_0_5pct"]
        == "FAIL"
    )


def test_tail_selection_uses_lowest_rmse_then_lower_weight() -> None:
    first = _row("tail-huber-0p50", 0.295, 0.19)
    second = _row("tail-huber-1p00", 0.294, 0.19)
    decisions = {
        "tail-huber-0p50": {"advances": True},
        "tail-huber-1p00": {"advances": True},
    }
    selected = _select_tail_candidate([first, second], decisions)
    assert selected is not None
    assert selected["candidate_id"] == "tail-huber-1p00"


def test_sequence_tail_threshold_uses_materialized_targets() -> None:
    dataset = FusionSequenceDataset.__new__(FusionSequenceDataset)
    dataset.log_target = np.asarray([-2.0, -1.0, 0.0, 1.0], dtype=np.float64)
    empty_indices = tuple()
    dataset.examples = [
        (index, index, empty_indices) for index in range(len(dataset.log_target))
    ]
    threshold = dataset.target_quantile(0.75)
    assert threshold == pytest.approx(0.25)
    assert dataset.target_fraction_at_or_above(threshold) == pytest.approx(0.25)
