from __future__ import annotations

from strata_ot.evaluation.fusion_residual_cap import _select_cap_candidate
from strata_ot.evaluation.fusion_tail_objective import _tail_decision


def _row(
    candidate_id: str,
    *,
    cap: float,
    rmse: float,
    tail: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "cap": cap,
        "primary": {
            "rmse_log10_cn2": rmse,
            "tail_mae_top_decile": tail,
            "crps_gaussian": 0.14,
            "interval_80_coverage": 0.80,
            "bias_log10_cn2": 0.01,
        },
    }


def test_residual_cap_uses_joint_point_and_tail_gate() -> None:
    control = _row(
        "residual-cap-unbounded",
        cap=0.0,
        rmse=0.30,
        tail=0.20,
    )
    passing = _tail_decision(
        control,
        _row("residual-cap-0p35", cap=0.35, rmse=0.297, tail=0.198),
    )
    assert passing["advances"] is True
    tail_failure = _tail_decision(
        control,
        _row("residual-cap-0p20", cap=0.20, rmse=0.297, tail=0.21),
    )
    assert tail_failure["advances"] is False


def test_residual_cap_tie_prefers_least_restrictive_bound() -> None:
    smaller = _row(
        "residual-cap-0p20",
        cap=0.20,
        rmse=0.2950,
        tail=0.19,
    )
    larger = _row(
        "residual-cap-0p35",
        cap=0.35,
        rmse=0.2955,
        tail=0.19,
    )
    decisions = {
        "residual-cap-0p20": {"advances": True},
        "residual-cap-0p35": {"advances": True},
    }
    selected = _select_cap_candidate([smaller, larger], decisions)
    assert selected is not None
    assert selected["candidate_id"] == "residual-cap-0p35"
