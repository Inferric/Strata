from __future__ import annotations

from strata_ot.evaluation.fusion_residual_scale import (
    RESIDUAL_SCALE_EXPONENTS,
    _residual_scale_decision,
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


def test_residual_scale_decision_requires_all_frozen_conditions() -> None:
    control = _row(rmse=0.30)
    passing = _residual_scale_decision(control, _row(rmse=0.294))
    failing_coverage = _residual_scale_decision(
        control,
        _row(rmse=0.293, coverage=0.65),
    )
    assert passing["advances"] is True
    assert (
        failing_coverage["checks"]["coverage_70_to_90_percent"] == "FAIL"
    )
    assert failing_coverage["advances"] is False


def test_residual_scale_family_has_three_fixed_interventions() -> None:
    assert list(RESIDUAL_SCALE_EXPONENTS.values()) == [0.0, 0.25, 0.5, 1.0]
