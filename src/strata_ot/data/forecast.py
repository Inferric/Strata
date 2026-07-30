from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from strata_ot.config import find_repo_root
from strata_ot.data.acquire import acquire_otbench
from strata_ot.data.registry import verify_manifest


class ForecastSequenceDataset(Dataset[dict[str, Tensor]]):
    """Leakage-safe history windows for one-step Cn² forecasting."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        target: np.ndarray,
        persistence: np.ndarray,
        history_target: np.ndarray,
        input_end_ids: np.ndarray,
        target_ids: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.persistence = torch.as_tensor(persistence, dtype=torch.float32)
        self.history_target = torch.as_tensor(history_target, dtype=torch.float32)
        self.input_end_ids = np.asarray(input_end_ids, dtype=str)
        self.target_ids = np.asarray(target_ids, dtype=str)
        self.target_timestamps = np.asarray(target_timestamps, dtype="datetime64[ns]")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "target": self.target[index],
            "persistence": self.persistence[index],
            "history_target": self.history_target[index],
        }


@dataclass(frozen=True)
class ForecastMetadata:
    input_dim: int
    context: int
    feature_names: list[str]
    target_transform: str
    split_id: str
    dataset_id: str
    source_dataset_id: str
    site: str
    forecast_horizon_minutes: float
    history_minutes: float


@dataclass(frozen=True)
class _Partition:
    features: np.ndarray
    target: np.ndarray
    timestamps: pd.DatetimeIndex
    row_ids: np.ndarray

    def slice(self, selection: slice) -> _Partition:
        return _Partition(
            self.features[selection],
            self.target[selection],
            self.timestamps[selection],
            self.row_ids[selection],
        )


class ForecastDataModule(L.LightningDataModule):
    """Public OTBench forecasting with train-only transforms and sealed test release."""

    def __init__(
        self,
        data_config: dict[str, Any],
        split_config: dict[str, Any],
        *,
        batch_size: int,
        num_workers: int,
        allow_test: bool = False,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.split_config = split_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_test = allow_test
        forecast = data_config["forecast"]
        self.context = int(forecast["window_rows"])
        self.horizon = int(forecast["forecast_horizon_rows"])
        self.nominal_cadence_minutes = float(forecast["nominal_cadence_minutes"])
        self.maximum_gap_minutes = float(forecast["maximum_gap_minutes"])
        self.calibration_fraction = float(forecast["validation_calibration_fraction"])
        self.metadata: ForecastMetadata | None = None
        self.train_set: ForecastSequenceDataset | None = None
        self.validation_set: ForecastSequenceDataset | None = None
        self.calibration_set: ForecastSequenceDataset | None = None
        self.test_set: ForecastSequenceDataset | None = None
        self.normalization: dict[str, Any] = {}
        self.sequence_qc: dict[str, Any] = {}

    @property
    def destination(self) -> Path:
        root = find_repo_root()
        return root / str(self.data_config["cache_dir"]) / str(
            self.data_config["source_dataset_id"]
        )

    def prepare_data(self) -> None:
        root = find_repo_root()
        manifest = root / str(self.data_config["manifest_path"])
        if not (self.destination / "train_features.csv").exists():
            acquire_otbench(self.data_config, root)
        failures = verify_manifest(manifest, root)
        if failures:
            raise RuntimeError("Raw snapshot identity failure: " + ", ".join(failures))

    def _load_partition(
        self,
        name: str,
        usable: list[str] | None = None,
    ) -> tuple[_Partition, list[str]]:
        raw_features = pd.read_csv(self.destination / f"{name}_features.csv")
        raw_target = pd.read_csv(self.destination / f"{name}_target.csv")
        if not raw_features["_upstream_index"].astype(str).equals(
            raw_target["_upstream_index"].astype(str)
        ):
            raise RuntimeError(f"{name} feature and target identities are misaligned")
        candidates = [
            column for column in raw_features.columns if not column.startswith("_")
        ]
        if usable is None:
            numeric = raw_features[candidates].apply(pd.to_numeric, errors="coerce")
            usable = [
                column
                for column in numeric
                if float(numeric[column].notna().mean()) >= 0.80
            ]
        matrix = (
            raw_features[usable]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        target = pd.to_numeric(raw_target["target"], errors="coerce").to_numpy(float)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(raw_target["_upstream_index"], errors="raise")
        )
        row_ids = raw_target["_upstream_index"].astype(str).to_numpy()
        return _Partition(matrix, target, timestamps, row_ids), usable

    def _apply_boundary_purge(self, partitions: dict[str, _Partition]) -> None:
        purge = int(self.split_config.get("purge_gap_rows", 0))
        selections = {
            "train": slice(None, -purge if purge else None),
            "validation": slice(purge, -purge if purge else None),
            "test": slice(purge, None),
        }
        for name, selection in selections.items():
            if name in partitions:
                partitions[name] = partitions[name].slice(selection)

    def _transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        median = np.asarray(self.normalization["feature_median"], dtype=np.float64)
        mean = np.asarray(self.normalization["feature_mean"], dtype=np.float64)
        std = np.asarray(self.normalization["feature_std"], dtype=np.float64)
        return (np.where(np.isfinite(matrix), matrix, median) - mean) / std

    def _build_sequences(
        self,
        partition: _Partition,
        *,
        name: str,
    ) -> ForecastSequenceDataset:
        matrix = self._transform_matrix(partition.features)
        target_mean = float(self.normalization["history_target_mean"])
        target_std = float(self.normalization["history_target_std"])
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        persistence: list[float] = []
        histories: list[np.ndarray] = []
        input_end_ids: list[str] = []
        target_ids: list[str] = []
        target_timestamps: list[np.datetime64] = []
        rejected_gap = 0
        rejected_target = 0
        rejected_non_monotonic = 0
        for end in range(self.context - 1, len(partition.target) - self.horizon):
            start = end - self.context + 1
            target_index = end + self.horizon
            timestamps = partition.timestamps[start : target_index + 1]
            deltas = np.diff(timestamps.asi8) / 60e9
            if np.any(deltas <= 0):
                rejected_non_monotonic += 1
                continue
            if np.any(deltas > self.maximum_gap_minutes):
                rejected_gap += 1
                continue
            history = partition.target[start : end + 1]
            future = partition.target[target_index]
            if not np.all(np.isfinite(history)) or not np.isfinite(future):
                rejected_target += 1
                continue
            elapsed = np.empty(self.context, dtype=np.float64)
            elapsed[0] = 1.0
            if self.context > 1:
                elapsed[1:] = (
                    np.diff(partition.timestamps[start : end + 1].asi8)
                    / 60e9
                    / self.nominal_cadence_minutes
                )
            history_scaled = (history - target_mean) / target_std
            sequence = np.column_stack(
                (matrix[start : end + 1], history_scaled, elapsed)
            )
            sequences.append(sequence)
            targets.append(float(future))
            persistence.append(float(history[-1]))
            histories.append(history.copy())
            input_end_ids.append(str(partition.row_ids[end]))
            target_ids.append(str(partition.row_ids[target_index]))
            target_timestamps.append(partition.timestamps[target_index].to_datetime64())
        if not sequences:
            raise ValueError(f"{name} produced no valid forecast sequences")
        self.sequence_qc[name] = {
            "candidate_endpoints": (
                len(partition.target) - self.context - self.horizon + 1
            ),
            "retained": len(sequences),
            "rejected_gap": rejected_gap,
            "rejected_target": rejected_target,
            "rejected_non_monotonic": rejected_non_monotonic,
        }
        return ForecastSequenceDataset(
            features=np.stack(sequences),
            target=np.asarray(targets),
            persistence=np.asarray(persistence),
            history_target=np.stack(histories),
            input_end_ids=np.asarray(input_end_ids),
            target_ids=np.asarray(target_ids),
            target_timestamps=np.asarray(target_timestamps),
        )

    def setup(self, stage: str | None = None) -> None:
        del stage
        partitions: dict[str, _Partition] = {}
        train, usable = self._load_partition("train")
        validation, _ = self._load_partition("validation", usable)
        partitions["train"] = train
        partitions["validation"] = validation
        if self.allow_test:
            test, _ = self._load_partition("test", usable)
            partitions["test"] = test
        self._apply_boundary_purge(partitions)
        train_matrix = partitions["train"].features
        feature_median = np.nanmedian(train_matrix, axis=0)
        feature_median = np.where(np.isfinite(feature_median), feature_median, 0.0)
        imputed_train = np.where(np.isfinite(train_matrix), train_matrix, feature_median)
        feature_mean = imputed_train.mean(axis=0)
        feature_std = imputed_train.std(axis=0)
        feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
        finite_train_target = partitions["train"].target[
            np.isfinite(partitions["train"].target)
        ]
        history_target_mean = float(finite_train_target.mean())
        history_target_std = float(finite_train_target.std())
        if history_target_std <= 1e-8:
            history_target_std = 1.0
        self.normalization = {
            "feature_median": feature_median.tolist(),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "history_target_mean": history_target_mean,
            "history_target_std": history_target_std,
            "fit_partition": "train",
        }
        validation_partition = partitions["validation"]
        split_at = int(len(validation_partition.target) * (1.0 - self.calibration_fraction))
        inner_purge = int(self.split_config.get("purge_gap_rows", 0))
        if split_at - inner_purge <= self.context:
            raise ValueError("Validation selection block is too small")
        if len(validation_partition.target) - (split_at + inner_purge) <= self.context:
            raise ValueError("Validation calibration block is too small")
        selection = validation_partition.slice(slice(None, split_at - inner_purge))
        calibration = validation_partition.slice(slice(split_at + inner_purge, None))
        self.train_set = self._build_sequences(partitions["train"], name="train")
        self.validation_set = self._build_sequences(selection, name="selection")
        self.calibration_set = self._build_sequences(calibration, name="calibration")
        if "test" in partitions:
            self.test_set = self._build_sequences(partitions["test"], name="test")
        manifest_payload = json.loads(
            (find_repo_root() / str(self.data_config["manifest_path"])).read_text(
                encoding="utf-8"
            )
        )
        target_transform = (
            "canonical_log10(raw_float32_cn2)"
            if manifest_payload.get("qc", {}).get("canonical_serialization")
            else "upstream_log10_cn2"
        )
        site = str(self.data_config.get("site", "unknown"))
        self.metadata = ForecastMetadata(
            input_dim=len(usable) + 2,
            context=self.context,
            feature_names=[*usable, "history_log10_cn2", "elapsed_cadence_ratio"],
            target_transform=target_transform,
            split_id=str(self.split_config["id"]),
            dataset_id=str(self.data_config["id"]),
            source_dataset_id=str(self.data_config["source_dataset_id"]),
            site=site,
            forecast_horizon_minutes=self.nominal_cadence_minutes * self.horizon,
            history_minutes=self.nominal_cadence_minutes * self.context,
        )
        if self.allow_test and bool(self.split_config.get("materialize_on_first_use", False)):
            self._seal_materialized_split(partitions)

    def _seal_materialized_split(self, partitions: dict[str, _Partition]) -> None:
        root = find_repo_root()
        payload = {
            "split_id": self.split_config["id"],
            "dataset_id": self.data_config["id"],
            "strategy": self.split_config["strategy"],
            "indices": {
                name: partition.row_ids.tolist()
                for name, partition in partitions.items()
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()
        destination = (
            root
            / "data"
            / "sealed"
            / "splits"
            / f"{self.split_config['id']}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"Refusing to mutate sealed split {destination}")
        if not destination.exists():
            destination.write_text(rendered, encoding="utf-8")

    def _loader(
        self,
        dataset: ForecastSequenceDataset,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and shuffle,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        assert self.train_set is not None
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        assert self.validation_set is not None
        return self._loader(self.validation_set, shuffle=False)

    def calibration_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        assert self.calibration_set is not None
        return self._loader(self.calibration_set, shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if not self.allow_test or self.test_set is None:
            raise RuntimeError("The sealed test partition has not been released")
        return self._loader(self.test_set, shuffle=False)
