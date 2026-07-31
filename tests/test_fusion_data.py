from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strata_ot.config import load_yaml
from strata_ot.data.fusion import (
    FusionDataModule,
    _point_features,
    _RoleArrays,
    identity_sha256,
)
from strata_ot.models.fusion_controls import flatten_fusion_feature_names


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def test_fusion_identity_hash_has_frozen_minute_serialization() -> None:
    values = np.asarray(
        ["2020-01-01T00:00", "2020-01-01T00:01"],
        dtype="datetime64[m]",
    )
    assert (
        identity_sha256(values)
        == "e3587761e2528af90ae85cb76f4401f65320c9e87859537f20119f6d87cea12a"
    )


def test_physics_features_and_missing_masks_are_finite_or_explicit() -> None:
    raw = {
        "T_3m": np.asarray([20.0, np.nan]),
        "P_3m": np.asarray([1013.0, 1012.0]),
        "RH_3m": np.asarray([50.0, 60.0]),
        "Spd_3m": np.asarray([3.0, 4.0]),
        "Rad_1m": np.asarray([200.0, 0.0]),
        "T_0m": np.asarray([18.0, 19.0]),
        "Dir_3m": np.asarray(["SW", "nan"], dtype=object),
    }
    timestamps = np.asarray(
        ["2020-01-01T12:00", "2020-01-01T12:01"],
        dtype="datetime64[m]",
    )
    values, names = _point_features(raw, timestamps, physics_tokens="full")
    assert values.shape[1] == len(names)
    assert "total_refractivity" in names
    assert "Dir_3m_sin" in names
    assert values[1, names.index("T_3m_missing")] == 1
    assert values[1, names.index("Dir_3m_sin_missing")] == 1


def test_confirmation_requires_release_flag() -> None:
    root = _repository()
    data = load_yaml("configs/data/otbench_usna_lg_fusion_v2.yaml", root=root)
    split = load_yaml(str(data["split_config"]), root=root)
    module = FusionDataModule(
        data,
        split,
        fold_id="final",
        physics_tokens="none",
        batch_size=4,
        num_workers=0,
        max_train_examples=8,
        max_evaluation_examples=8,
        release_confirmation=False,
    )
    with pytest.raises(RuntimeError, match="explicit --release-confirmation"):
        module.confirmation_dataloader()


def test_final_setup_does_not_consume_or_load_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository()
    data = load_yaml("configs/data/otbench_usna_lg_fusion_v2.yaml", root=root)
    split = load_yaml(str(data["split_config"]), root=root)
    module = FusionDataModule(
        data,
        split,
        fold_id="final",
        physics_tokens="none",
        batch_size=4,
        num_workers=0,
        max_train_examples=8,
        max_evaluation_examples=8,
        release_confirmation=True,
    )
    timestamps = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-03T00:00"),
        np.timedelta64(1, "m"),
    )
    arrays = _RoleArrays(
        timestamps=timestamps,
        target=np.full(len(timestamps), 1e-14),
        features=np.ones((len(timestamps), 1)),
        feature_present=np.ones((len(timestamps), 1)),
        feature_names=["T_3m"],
        raw_start_offset=0,
    )
    loaded_roles: list[str] = []

    def fake_load_role(role: str, *, labels: bool) -> _RoleArrays:
        assert labels
        loaded_roles.append(role)
        return arrays

    monkeypatch.setattr(module, "_load_role", fake_load_role)
    module.materialized_split = {"sha256": "test"}
    module.setup("fit")
    assert loaded_roles == ["train", "calibration"]
    assert module.confirmation_set is None
    assert module.confirmation_consumption is None


def test_flattened_feature_names_match_control_matrix_width() -> None:
    weather = ["temperature", "humidity"]
    names = flatten_fusion_feature_names(weather)
    expected = 6 * 5 + 12 * 5 + 24 * 5 + 1
    assert len(names) == expected
    assert names[15] == "short/lag_0000m/cn2_level"
    assert names[-1] == "forecast/horizon_minutes_scaled"
