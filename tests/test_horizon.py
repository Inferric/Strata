from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from strata_ot.config import load_yaml
from strata_ot.data.horizon import HorizonDataModule
from strata_ot.training.horizon import _paired_bootstrap, _tracked_artifact_bytes
from strata_ot.training.safety import resource_peaks


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frozen_horizon_split_and_assessment_isolation() -> None:
    root = _repository()
    data = load_yaml("configs/data/otbench_mlo_weather_horizon_v1.yaml", root=root)
    destination = (
        root
        / str(data["cache_dir"])
        / str(data["source_dataset_id"])
        / "validation_features.csv"
    )
    if not destination.is_file():
        pytest.skip("Verified public MLO snapshot is not present")
    split = load_yaml(str(data["split_config"]), root=root)
    module = HorizonDataModule(
        data,
        split,
        feature_set="operational_weather",
        batch_size=32,
        num_workers=0,
        allow_assessment=False,
    )
    module.prepare_data()
    module.setup("fit")
    assert set(module.role_row_ids) == {"selection", "calibration"}
    assert module.assessment_set is None
    assert module.materialized_split["sha256"] == split["materialized_payload_sha256"]
    assert module.normalization["fit_partition"] == "official_train_after_outer_purge"
    assert module.metadata is not None
    assert module.metadata.horizon_minutes == [5.0, 15.0, 30.0, 60.0]
    with pytest.raises(RuntimeError, match="Assessment labels have not been released"):
        module.assessment_dataloader()
    with pytest.raises(RuntimeError, match="official MLO test partition is sealed"):
        module.test_dataloader()


def test_consumed_mlo_assessment_cannot_be_reopened() -> None:
    root = _repository()
    data = load_yaml("configs/data/otbench_mlo_weather_horizon_v1.yaml", root=root)
    split = load_yaml(str(data["split_config"]), root=root)
    module = HorizonDataModule(
        data,
        split,
        feature_set="history_only",
        batch_size=8,
        num_workers=0,
        allow_assessment=True,
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        module.prepare_data()


def test_paired_horizon_bootstrap_detects_improvement(tmp_path: Path) -> None:
    target_ids = np.asarray([f"row-{index}" for index in range(480)])
    horizons = np.tile(np.asarray([15, 30, 60]), 160)
    target = np.linspace(-16.0, -14.0, len(target_ids))

    def run(seed: int, name: str, offset: float) -> dict[str, object]:
        run_id = f"{name}-{seed}"
        destination = tmp_path / "artifacts" / "runs" / run_id
        destination.mkdir(parents=True)
        relative = Path("artifacts") / "runs" / run_id / "predictions.npz"
        np.savez_compressed(
            tmp_path / relative,
            target_ids=target_ids,
            horizon_minutes=horizons,
            target=target,
            prediction=target + offset,
        )
        return {
            "run_id": run_id,
            "seed": seed,
            "artifacts": {"predictions": relative.as_posix()},
        }

    candidate = [run(seed, "candidate", 0.05) for seed in (17, 41)]
    comparator = [run(seed, "comparator", 0.20) for seed in (17, 41)]
    result = _paired_bootstrap(
        tmp_path,
        candidate,
        comparator,
        horizons=[15, 30, 60],
        block_rows=48,
        resamples=200,
        seed=20260730,
    )
    assert result["ci95_low"] > 0
    assert result["mean_relative_improvement"] == pytest.approx(0.75)


def test_split_configuration_keeps_48_row_internal_gaps() -> None:
    split = load_yaml(
        "configs/splits/frozen/otbench_mlo_weather_horizon_v1.yaml",
        root=_repository(),
    )
    roles = split["roles"]
    assert roles["calibration"]["raw_start_offset"] - roles["selection"][
        "raw_end_offset_inclusive"
    ] - 1 == 48
    assert roles["assessment"]["raw_start_offset"] - roles["calibration"][
        "raw_end_offset_inclusive"
    ] - 1 == 48
    materialized = _repository() / str(split["materialized_path"])
    if materialized.is_file():
        payload = json.loads(materialized.read_text(encoding="utf-8"))
        assert payload["sha256"] == split["materialized_payload_sha256"]


def test_resource_peaks_separate_board_allocated_and_reserved() -> None:
    samples = [
        {
            "gpu": {
                "used_memory_gib": 2.5,
                "process_peak_allocated_gib": 0.7,
                "process_peak_reserved_gib": 0.9,
            }
        },
        {
            "gpu": {
                "used_memory_gib": 3.25,
                "process_peak_allocated_gib": 1.2,
                "process_peak_reserved_gib": 1.5,
            }
        },
    ]
    assert resource_peaks(samples) == {
        "peak_total_board_vram_gib": 3.25,
        "peak_process_allocated_vram_gib": 1.2,
        "peak_process_reserved_vram_gib": 1.5,
    }


def test_tracked_artifact_bytes_counts_unique_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    checkpoint = tmp_path / "checkpoint.ckpt"
    artifact.write_bytes(b"12345")
    checkpoint.write_bytes(b"1234567")
    size = _tracked_artifact_bytes(
        tmp_path,
        {"a": "artifact.bin", "duplicate": "artifact.bin"},
        checkpoint=str(checkpoint),
    )
    assert size == 12
