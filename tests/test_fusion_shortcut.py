from __future__ import annotations

from strata_ot.evaluation.fusion_shortcut import (
    SHORTCUT_MODES,
    SHORTCUT_PARAMETER_COUNTS,
    _shortcut_decision,
)


def _row(
    *,
    rmse: float,
    tail: float = 0.16,
    crps: float = 0.15,
    coverage: float = 0.80,
    bias: float = 0.02,
) -> dict[str, object]:
    return {
        "primary": {
            "rmse_log10_cn2": rmse,
            "tail_mae_top_decile": tail,
            "crps_gaussian": crps,
            "interval_80_coverage": coverage,
            "bias_log10_cn2": bias,
        }
    }


def test_shortcut_decision_requires_all_frozen_conditions() -> None:
    control = _row(rmse=0.30)
    passing = _shortcut_decision(control, _row(rmse=0.294))
    failing_tail = _shortcut_decision(
        control,
        _row(rmse=0.293, tail=0.18),
    )
    assert passing["advances"] is True
    assert failing_tail["checks"]["tail_no_two_percent_regression"] == "FAIL"
    assert failing_tail["advances"] is False


def test_shortcut_family_has_three_fixed_interventions() -> None:
    assert list(SHORTCUT_MODES.values()) == [
        "none",
        "history_linear",
        "all_linear",
        "all_mlp",
    ]
    assert list(SHORTCUT_PARAMETER_COUNTS.values()) == [
        3611530,
        3611849,
        3612941,
        3885455,
    ]
