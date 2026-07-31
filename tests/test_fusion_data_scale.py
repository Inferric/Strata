from __future__ import annotations

from strata_ot.evaluation.fusion_data_scale import (
    DATA_SCALE_CAPS,
    _data_scale_decision,
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


def test_data_scale_decision_requires_complete_two_percent_rule() -> None:
    control = _row(rmse=0.30)
    passing = _data_scale_decision(control, _row(rmse=0.294))
    failing_tail = _data_scale_decision(
        control,
        _row(rmse=0.293, tail=0.17),
    )
    assert passing["advances"] is True
    assert failing_tail["checks"]["tail_no_two_percent_regression"] == "FAIL"
    assert failing_tail["advances"] is False


def test_data_scale_full_cap_is_exact_fold_one_sequence_count() -> None:
    assert DATA_SCALE_CAPS["data-scale-full"] == 62263
