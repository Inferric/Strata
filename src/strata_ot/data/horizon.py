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
from strata_ot.data.registry import verify_manifest
from strata_ot.evaluation.consumption import require_unconsumed


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=isinstance(value, dict),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class HorizonSequenceDataset(Dataset[dict[str, Tensor]]):
    """Stacked, cadence-checked multi-horizon Cn2 forecast examples."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        target: np.ndarray,
        persistence: np.ndarray,
        history_target: np.ndarray,
        horizon_rows: np.ndarray,
        horizon_minutes: np.ndarray,
        input_end_ids: np.ndarray,
        target_ids: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.persistence = torch.as_tensor(persistence, dtype=torch.float32)
        self.history_target = torch.as_tensor(history_target, dtype=torch.float32)
        self.horizon_rows = torch.as_tensor(horizon_rows, dtype=torch.long)
        self.horizon_minutes = torch.as_tensor(horizon_minutes, dtype=torch.float32)
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
            "horizon_rows": self.horizon_rows[index],
            "horizon_minutes": self.horizon_minutes[index],
        }


@dataclass(frozen=True)
class HorizonMetadata:
    input_dim: int
    context: int
    feature_names: list[str]
    raw_feature_allowlist: list[str]
    resolved_weather_channels: list[str]
    target_transform: str
    split_id: str
    dataset_id: str
    source_dataset_id: str
    site: str
    feature_set: str
    weather_enabled: bool
    horizon_rows: list[int]
    horizon_minutes: list[float]
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


class HorizonDataModule(L.LightningDataModule):
    """MLO-only development roles with assessment-label release isolation."""

    def __init__(
        self,
        data_config: dict[str, Any],
        split_config: dict[str, Any],
        *,
        feature_set: str,
        batch_size: int,
        num_workers: int,
        allow_assessment: bool = False,
    ) -> None:
        super().__init__()
        feature_sets = data_config.get("feature_sets", {})
        if feature_set not in feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.data_config = data_config
        self.split_config = split_config
        self.feature_set = feature_set
        self.feature_config = dict(feature_sets[feature_set])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_assessment = allow_assessment
        forecast = data_config["forecast"]
        self.context = int(forecast["window_rows"])
        self.horizon_rows = [int(value) for value in forecast["horizon_rows"]]
        self.horizon_minutes = [
            float(value) for value in forecast["horizon_minutes"]
        ]
        if len(self.horizon_rows) != len(self.horizon_minutes):
            raise ValueError("Horizon row and minute identities must align")
        self.nominal_cadence_minutes = float(forecast["nominal_cadence_minutes"])
        self.maximum_context_gap_minutes = float(
            forecast["maximum_context_gap_minutes"]
        )
        self.horizon_tolerance_minutes = float(
            forecast["horizon_tolerance_minutes"]
        )
        self.resolved_weather_channels = [
            str(value) for value in data_config["resolved_weather_channels"]
        ]
        self.raw_feature_allowlist = [
            str(value) for value in self.feature_config["raw_allowlist"]
        ]
        self.weather_enabled = bool(self.feature_config["weather_enabled"])
        self.metadata: HorizonMetadata | None = None
        self.train_set: HorizonSequenceDataset | None = None
        self.validation_set: HorizonSequenceDataset | None = None
        self.calibration_set: HorizonSequenceDataset | None = None
        self.assessment_set: HorizonSequenceDataset | None = None
        self.normalization: dict[str, Any] = {}
        self.sequence_qc: dict[str, Any] = {}
        self.role_row_ids: dict[str, np.ndarray] = {}
        self.materialized_split: dict[str, Any] = {}

    @property
    def destination(self) -> Path:
        root = find_repo_root()
        return root / str(self.data_config["cache_dir"]) / str(
            self.data_config["source_dataset_id"]
        )

    def prepare_data(self) -> None:
        root = find_repo_root()
        if self.allow_assessment:
            consumption_id = self.data_config.get("assessment_consumption_id")
            if not isinstance(consumption_id, str) or not consumption_id:
                raise RuntimeError(
                    "Assessment release requires a registered consumption identity"
                )
            require_unconsumed(consumption_id, root=root)
        required = (
            "train_features.csv",
            "train_target.csv",
            "validation_features.csv",
            "validation_target.csv",
        )
        missing = [name for name in required if not (self.destination / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "The frozen MLO snapshot is incomplete; automatic acquisition is "
                f"disabled for this experiment: {', '.join(missing)}"
            )
        manifest = root / str(self.data_config["manifest_path"])
        failures = verify_manifest(manifest, root)
        if failures:
            raise RuntimeError("Raw snapshot identity failure: " + ", ".join(failures))
        self.materialized_split = self._materialize_role_identities()

    def _materialize_role_identities(self) -> dict[str, Any]:
        identity_frame = pd.read_csv(
            self.destination / "validation_features.csv",
            usecols=["_upstream_index"],
            dtype={"_upstream_index": str},
        )
        all_ids = identity_frame["_upstream_index"].astype(str).tolist()
        role_values: dict[str, list[str]] = {}
        for role_name, raw_config in self.split_config["roles"].items():
            role = dict(raw_config)
            start = int(role["raw_start_offset"])
            stop = int(role["raw_end_offset_inclusive"]) + 1
            values = all_ids[start:stop]
            if len(values) != int(role["rows"]):
                raise RuntimeError(f"{role_name} row count drift")
            if not values:
                raise RuntimeError(f"{role_name} role is empty")
            if values[0] != str(role["first_identity"]):
                raise RuntimeError(f"{role_name} first identity drift")
            if values[-1] != str(role["last_identity"]):
                raise RuntimeError(f"{role_name} last identity drift")
            if _canonical_sha256(values) != str(role["identities_sha256"]):
                raise RuntimeError(f"{role_name} identity hash drift")
            role_values[str(role_name)] = values
        role_sets = {name: set(values) for name, values in role_values.items()}
        role_names = list(role_sets)
        if any(
            not role_sets[role_names[left]].isdisjoint(role_sets[role_names[right]])
            for left in range(len(role_names))
            for right in range(left + 1, len(role_names))
        ):
            raise RuntimeError("Development role identities overlap")
        canonical_payload = {
            "split_id": str(self.split_config["id"]),
            "source_dataset_id": str(self.data_config["source_dataset_id"]),
            "roles": role_values,
        }
        payload_sha = _canonical_sha256(canonical_payload)
        if payload_sha != str(self.split_config["materialized_payload_sha256"]):
            raise RuntimeError("Materialized development split hash drift")
        payload = {**canonical_payload, "sha256": payload_sha}
        root = find_repo_root()
        destination = root / str(self.split_config["materialized_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            if destination.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"Refusing to mutate sealed split {destination}")
        else:
            destination.write_text(rendered, encoding="utf-8")
        return payload

    def _feature_matrix(self, name: str, *, nrows: int | None) -> tuple[np.ndarray, list[str]]:
        usecols = ["_upstream_index", *self.raw_feature_allowlist]
        frame = pd.read_csv(
            self.destination / f"{name}_features.csv",
            usecols=usecols,
            nrows=nrows,
        )
        identities = frame["_upstream_index"].astype(str).tolist()
        if not self.weather_enabled:
            return np.zeros(
                (len(frame), len(self.resolved_weather_channels)), dtype=np.float64
            ), identities
        values: list[np.ndarray] = []
        direction = pd.to_numeric(frame["Dir_10m"], errors="coerce").to_numpy(float)
        radians = np.deg2rad(direction)
        values.extend((np.sin(radians), np.cos(radians)))
        for name in ("Spd_10m", "P_2m", "T_2m", "RH_2m", "Tdew_2m"):
            values.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(float))
        matrix = np.column_stack(values)
        if matrix.shape[1] != len(self.resolved_weather_channels):
            raise RuntimeError("Resolved weather channel count drift")
        return matrix, identities

    def _load_partition(
        self,
        name: str,
        *,
        nrows: int | None = None,
    ) -> _Partition:
        matrix, feature_ids = self._feature_matrix(name, nrows=nrows)
        raw_target = pd.read_csv(
            self.destination / f"{name}_target.csv",
            nrows=nrows,
        )
        target_ids = raw_target["_upstream_index"].astype(str).tolist()
        if feature_ids != target_ids:
            raise RuntimeError(f"{name} feature and target identities are misaligned")
        target = pd.to_numeric(raw_target["target"], errors="coerce").to_numpy(float)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(raw_target["_upstream_index"], errors="raise")
        )
        return _Partition(
            features=matrix,
            target=target,
            timestamps=timestamps,
            row_ids=np.asarray(target_ids),
        )

    def _fit_normalization(self, train: _Partition) -> None:
        feature_median = np.nanmedian(train.features, axis=0)
        feature_median = np.where(np.isfinite(feature_median), feature_median, 0.0)
        imputed = np.where(np.isfinite(train.features), train.features, feature_median)
        feature_mean = imputed.mean(axis=0)
        feature_std = imputed.std(axis=0)
        feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
        finite_target = train.target[np.isfinite(train.target)]
        if not len(finite_target):
            raise RuntimeError("Training target contains no finite values")
        history_target_std = float(finite_target.std())
        self.normalization = {
            "weather_median": feature_median.tolist(),
            "weather_mean": feature_mean.tolist(),
            "weather_std": feature_std.tolist(),
            "history_target_mean": float(finite_target.mean()),
            "history_target_std": (
                history_target_std if history_target_std > 1e-8 else 1.0
            ),
            "fit_partition": "official_train_after_outer_purge",
        }

    def _transform_weather(self, matrix: np.ndarray) -> np.ndarray:
        median = np.asarray(self.normalization["weather_median"], dtype=np.float64)
        mean = np.asarray(self.normalization["weather_mean"], dtype=np.float64)
        std = np.asarray(self.normalization["weather_std"], dtype=np.float64)
        return (np.where(np.isfinite(matrix), matrix, median) - mean) / std

    def _build_sequences(self, partition: _Partition, *, role: str) -> HorizonSequenceDataset:
        weather = self._transform_weather(partition.features)
        target_mean = float(self.normalization["history_target_mean"])
        target_std = float(self.normalization["history_target_std"])
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        persistence: list[float] = []
        histories: list[np.ndarray] = []
        horizon_rows_out: list[int] = []
        horizon_minutes_out: list[float] = []
        input_end_ids: list[str] = []
        target_ids: list[str] = []
        target_timestamps: list[np.datetime64] = []
        qc_by_horizon: dict[str, dict[str, int]] = {}
        for horizon_rows, horizon_minutes in zip(
            self.horizon_rows, self.horizon_minutes, strict=True
        ):
            retained = 0
            rejected_context_gap = 0
            rejected_elapsed = 0
            rejected_target = 0
            rejected_non_monotonic = 0
            candidate_count = max(
                0, len(partition.target) - self.context - horizon_rows + 1
            )
            for end in range(
                self.context - 1,
                len(partition.target) - horizon_rows,
            ):
                start = end - self.context + 1
                target_index = end + horizon_rows
                time_values = partition.timestamps[start : target_index + 1]
                all_deltas = np.diff(time_values.asi8) / 60e9
                if np.any(all_deltas <= 0):
                    rejected_non_monotonic += 1
                    continue
                context_deltas = (
                    np.diff(partition.timestamps[start : end + 1].asi8) / 60e9
                )
                if np.any(context_deltas > self.maximum_context_gap_minutes):
                    rejected_context_gap += 1
                    continue
                target_elapsed = (
                    partition.timestamps[target_index] - partition.timestamps[end]
                ).total_seconds() / 60
                if abs(target_elapsed - horizon_minutes) > self.horizon_tolerance_minutes:
                    rejected_elapsed += 1
                    continue
                history = partition.target[start : end + 1]
                future = partition.target[target_index]
                if not np.all(np.isfinite(history)) or not np.isfinite(future):
                    rejected_target += 1
                    continue
                elapsed = np.ones(self.context, dtype=np.float64)
                if self.context > 1:
                    elapsed[1:] = (
                        np.diff(partition.timestamps[start : end + 1].asi8)
                        / 60e9
                        / self.nominal_cadence_minutes
                    )
                history_scaled = (history - target_mean) / target_std
                horizon_ratio = np.full(
                    self.context, horizon_minutes / 60.0, dtype=np.float64
                )
                sequence = np.column_stack(
                    (
                        weather[start : end + 1],
                        history_scaled,
                        elapsed,
                        horizon_ratio,
                    )
                )
                sequences.append(sequence)
                targets.append(float(future))
                persistence.append(float(history[-1]))
                histories.append(history.copy())
                horizon_rows_out.append(horizon_rows)
                horizon_minutes_out.append(horizon_minutes)
                input_end_ids.append(str(partition.row_ids[end]))
                target_ids.append(str(partition.row_ids[target_index]))
                target_timestamps.append(
                    partition.timestamps[target_index].to_datetime64()
                )
                retained += 1
            qc_by_horizon[str(int(horizon_minutes))] = {
                "candidate_endpoints": candidate_count,
                "retained": retained,
                "rejected_context_gap": rejected_context_gap,
                "rejected_horizon_elapsed": rejected_elapsed,
                "rejected_target": rejected_target,
                "rejected_non_monotonic": rejected_non_monotonic,
            }
        if not sequences:
            raise ValueError(f"{role} produced no valid horizon sequences")
        self.sequence_qc[role] = qc_by_horizon
        return HorizonSequenceDataset(
            features=np.stack(sequences),
            target=np.asarray(targets),
            persistence=np.asarray(persistence),
            history_target=np.stack(histories),
            horizon_rows=np.asarray(horizon_rows_out),
            horizon_minutes=np.asarray(horizon_minutes_out),
            input_end_ids=np.asarray(input_end_ids),
            target_ids=np.asarray(target_ids),
            target_timestamps=np.asarray(target_timestamps),
        )

    def setup(self, stage: str | None = None) -> None:
        del stage
        if not self.materialized_split:
            self.prepare_data()
        outer_purge = int(self.split_config["outer_purge_rows"])
        train = self._load_partition("train")
        train = train.slice(slice(None, -outer_purge))
        roles = self.split_config["roles"]
        last_permitted_role = "assessment" if self.allow_assessment else "calibration"
        target_limit = (
            int(roles[last_permitted_role]["raw_end_offset_inclusive"]) + 1
        )
        validation = self._load_partition("validation", nrows=target_limit)
        role_partitions: dict[str, _Partition] = {}
        for role_name in ("selection", "calibration"):
            role = roles[role_name]
            role_partitions[role_name] = validation.slice(
                slice(
                    int(role["raw_start_offset"]),
                    int(role["raw_end_offset_inclusive"]) + 1,
                )
            )
        if self.allow_assessment:
            role = roles["assessment"]
            role_partitions["assessment"] = validation.slice(
                slice(
                    int(role["raw_start_offset"]),
                    int(role["raw_end_offset_inclusive"]) + 1,
                )
            )
        self.role_row_ids = {
            name: partition.row_ids.copy()
            for name, partition in role_partitions.items()
        }
        self._fit_normalization(train)
        self.train_set = self._build_sequences(train, role="train")
        self.validation_set = self._build_sequences(
            role_partitions["selection"], role="selection"
        )
        self.calibration_set = self._build_sequences(
            role_partitions["calibration"], role="calibration"
        )
        if "assessment" in role_partitions:
            self.assessment_set = self._build_sequences(
                role_partitions["assessment"], role="assessment"
            )
        manifest_payload = json.loads(
            (
                find_repo_root() / str(self.data_config["manifest_path"])
            ).read_text(encoding="utf-8")
        )
        target_transform = (
            "canonical_log10(raw_float32_cn2)"
            if manifest_payload.get("qc", {}).get("canonical_serialization")
            else "upstream_log10_cn2"
        )
        feature_names = [
            *self.resolved_weather_channels,
            "history_log10_cn2_scaled",
            "elapsed_cadence_ratio",
            "horizon_minutes_over_60",
        ]
        self.metadata = HorizonMetadata(
            input_dim=len(feature_names),
            context=self.context,
            feature_names=feature_names,
            raw_feature_allowlist=self.raw_feature_allowlist,
            resolved_weather_channels=self.resolved_weather_channels,
            target_transform=target_transform,
            split_id=str(self.split_config["id"]),
            dataset_id=str(self.data_config["id"]),
            source_dataset_id=str(self.data_config["source_dataset_id"]),
            site=str(self.data_config["site"]),
            feature_set=self.feature_set,
            weather_enabled=self.weather_enabled,
            horizon_rows=self.horizon_rows,
            horizon_minutes=self.horizon_minutes,
            history_minutes=self.nominal_cadence_minutes * self.context,
        )

    def feature_audit(self) -> dict[str, Any]:
        """Training-only target association audit for all source fields."""
        features = pd.read_csv(self.destination / "train_features.csv")
        target_frame = pd.read_csv(self.destination / "train_target.csv")
        if not features["_upstream_index"].astype(str).equals(
            target_frame["_upstream_index"].astype(str)
        ):
            raise RuntimeError("Training audit identities are misaligned")
        purge = int(self.split_config["outer_purge_rows"])
        features = features.iloc[:-purge]
        target = pd.to_numeric(
            target_frame.iloc[:-purge]["target"], errors="coerce"
        ).to_numpy(float)

        def category(name: str) -> str:
            lowered = name.lower()
            if name in self.raw_feature_allowlist:
                return "operational_weather"
            if "flag" in lowered:
                return "qc_flag"
            if lowered.startswith("diag"):
                return "instrument_diagnostic"
            if "count" in lowered:
                return "sample_count"
            if "kh2o" in lowered:
                return "instrument_water_vapor"
            covariance_tokens = (
                "u_u_",
                "u_v_",
                "u_w_",
                "u_tc_",
                "v_v_",
                "v_w_",
                "v_tc_",
                "w_w_",
                "w_tc_",
                "tc_tc_",
                "u_kh2o_",
                "v_kh2o_",
                "w_kh2o_",
            )
            if any(token in lowered for token in covariance_tokens):
                return "covariance_or_flux"
            return "excluded_other"

        fields: list[dict[str, Any]] = []
        for name in features.columns:
            if name.startswith("_"):
                continue
            values = pd.to_numeric(features[name], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(target)
            correlation: float | None = None
            if int(valid.sum()) >= 3 and float(np.std(values[valid])) > 1e-12:
                correlation = float(np.corrcoef(values[valid], target[valid])[0, 1])
            field_category = category(name)
            fields.append(
                {
                    "name": name,
                    "category": field_category,
                    "primary_allowed": name in self.raw_feature_allowlist,
                    "missing_fraction": float(1.0 - np.mean(np.isfinite(values))),
                    "pearson_target_correlation": correlation,
                    "possible_target_proxy": (
                        field_category
                        in {
                            "qc_flag",
                            "instrument_diagnostic",
                            "sample_count",
                            "instrument_water_vapor",
                            "covariance_or_flux",
                        }
                        or (
                            correlation is not None
                            and abs(correlation) >= 0.5
                            and name not in self.raw_feature_allowlist
                        )
                    ),
                }
            )
        category_counts: dict[str, int] = {}
        for field in fields:
            field_category = str(field["category"])
            category_counts[field_category] = category_counts.get(field_category, 0) + 1
        return {
            "fit_partition": "official_train_after_outer_purge",
            "source_feature_count": len(fields),
            "raw_allowlist": self.raw_feature_allowlist,
            "category_counts": category_counts,
            "fields": fields,
        }

    def _loader(
        self, dataset: HorizonSequenceDataset, *, shuffle: bool
    ) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and shuffle,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.train_set is None:
            raise RuntimeError("Training data is not prepared")
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.validation_set is None:
            raise RuntimeError("Selection data is not prepared")
        return self._loader(self.validation_set, shuffle=False)

    def calibration_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if self.calibration_set is None:
            raise RuntimeError("Calibration data is not prepared")
        return self._loader(self.calibration_set, shuffle=False)

    def assessment_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if not self.allow_assessment or self.assessment_set is None:
            raise RuntimeError("Assessment labels have not been released")
        return self._loader(self.assessment_set, shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        raise RuntimeError("The official MLO test partition is sealed for this experiment")
