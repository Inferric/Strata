from __future__ import annotations

from strata_ot.evaluation.fusion_v25_robustness import (
    EXPECTED_MAX_EPOCHS,
    EXPECTED_PARAMETERS,
    PREDECESSOR_CANDIDATE,
    REUSED_CONTROL_IDS,
    SELECTED_CANDIDATE,
    _control_keys,
    _expected_selected_keys,
    _seed_value,
)


def test_v25_robustness_matrix_is_frozen() -> None:
    assert SELECTED_CANDIDATE == "selected-fusion-v25"
    assert PREDECESSOR_CANDIDATE == "selected-fusion-v2"
    assert EXPECTED_MAX_EPOCHS == 9
    assert EXPECTED_PARAMETERS == 3611849
    assert len(_expected_selected_keys()) == 9
    assert len(_control_keys()) == 45


def test_v25_robustness_reuses_all_required_controls() -> None:
    assert REUSED_CONTROL_IDS == (
        "control-persistence",
        "control-climatology",
        "diagnostic-lightgbm",
        "selected-fusion-v2",
        "control-mlp",
        "control-tcn",
        "control-horizon-v1",
    )


def test_v25_robustness_ignores_seedless_unrelated_manifests() -> None:
    assert _seed_value({"seed": None}) == -1
    assert _seed_value({"seed": 41}) == 41
