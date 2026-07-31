from __future__ import annotations

from strata_ot.evaluation.fusion_convergence import (
    CONVERGENCE_BASELINE_RUN_ID,
    CONVERGENCE_EPOCHS,
    CONVERGENCE_PARAMETERS,
    _convergence_decision,
    _select_convergence_candidate,
)


def _row(
    *,
    candidate_id: str = "candidate",
    epochs: int = 6,
    rmse: float,
    tail: float = 0.16,
    crps: float = 0.15,
    coverage: float = 0.80,
    bias: float = 0.02,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "max_epochs_requested": epochs,
        "primary": {
            "rmse_log10_cn2": rmse,
            "tail_mae_top_decile": tail,
            "crps_gaussian": crps,
            "interval_80_coverage": coverage,
            "bias_log10_cn2": bias,
        },
    }


def test_convergence_decision_requires_all_frozen_conditions() -> None:
    baseline = _row(rmse=0.30)
    passing = _convergence_decision(baseline, _row(rmse=0.294))
    failing_coverage = _convergence_decision(
        baseline,
        _row(rmse=0.293, coverage=0.69),
    )
    assert passing["advances"] is True
    assert (
        failing_coverage["checks"]["coverage_70_to_90_percent"] == "FAIL"
    )
    assert failing_coverage["advances"] is False


def test_convergence_selection_prefers_fewer_epochs_within_tolerance() -> None:
    rows = [
        _row(candidate_id="convergence-6", epochs=6, rmse=0.2945),
        _row(candidate_id="convergence-9", epochs=9, rmse=0.2940),
        _row(candidate_id="convergence-12", epochs=12, rmse=0.2939),
    ]
    decisions = {
        str(row["candidate_id"]): {"advances": True} for row in rows
    }
    selected = _select_convergence_candidate(rows, decisions)
    assert selected is not None
    assert selected["candidate_id"] == "convergence-6"


def test_convergence_cycle_identities_are_frozen() -> None:
    assert CONVERGENCE_BASELINE_RUN_ID == "46af532e62ed4aaf9f50b9faab1e131d"
    assert list(CONVERGENCE_EPOCHS.values()) == [3, 6, 9, 12]
    assert CONVERGENCE_PARAMETERS == 3611849
