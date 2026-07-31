from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from strata_ot.config import find_repo_root
from strata_ot.data.registry import verify_manifest, write_manifest
from strata_ot.data.schema import DatasetFile, DatasetManifest, DatasetTerms
from strata_ot.evaluation.consumption import consume_partition

RAW_WEATHER = ("T_3m", "P_3m", "RH_3m", "Spd_3m", "Rad_1m", "T_0m")
COMPASS_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}
SCALE_SPECS = {
    "short": (6, 5),
    "medium": (12, 15),
    "slow": (24, 60),
}


def identity_sha256(timestamps: np.ndarray) -> str:
    values = np.datetime_as_string(timestamps.astype("datetime64[m]"), unit="m")
    payload = ("\n".join(values.tolist()) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_fusion_split(
    data_config: dict[str, Any],
    split_config: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    repository = root or find_repo_root()
    source = repository / str(data_config["source_path"])
    if _source_sha256(source) != str(data_config["source_sha256"]):
        raise RuntimeError("USNA-large source hash drift")
    records: list[dict[str, Any]] = []
    with xr.open_dataset(source, decode_cf=True) as dataset:
        times = dataset["time"].values.astype("datetime64[ns]")
        role_groups = [
            (str(fold["id"]), role_name, fold[role_name])
            for fold in split_config["folds"]
            for role_name in ("train", "calibration", "selection")
        ]
        role_groups.extend(
            (
                "final",
                role_name,
                split_config["final_fit"][role_name],
            )
            for role_name in ("train", "calibration")
        )
        for fold_id, role_name, role in role_groups:
            start = int(role["raw_start_offset"])
            end = int(role["raw_end_offset_inclusive"]) + 1
            selected = times[start:end]
            observed_hash = identity_sha256(selected)
            if observed_hash != str(role["identities_sha256"]):
                raise RuntimeError(
                    f"Identity drift for {fold_id} {role_name}: {observed_hash}"
                )
            records.append(
                {
                    "fold": fold_id,
                    "role": role_name,
                    "raw_start_offset": start,
                    "raw_end_offset_inclusive": end - 1,
                    "identities_sha256": observed_hash,
                    "identities": np.datetime_as_string(
                        selected, unit="m"
                    ).tolist(),
                }
            )
        confirmation = split_config["confirmation"]
        start = int(confirmation["raw_start_offset"])
        end = int(confirmation["raw_end_offset_inclusive"]) + 1
        selected = times[start:end]
        observed_hash = identity_sha256(selected)
        if observed_hash != str(confirmation["identities_sha256"]):
            raise RuntimeError("Confirmation identity drift")
        records.append(
            {
                "fold": "confirmation",
                "role": "confirmation_identity_only",
                "raw_start_offset": start,
                "raw_end_offset_inclusive": end - 1,
                "identities_sha256": observed_hash,
                "identities": np.datetime_as_string(selected, unit="m").tolist(),
                "labels_loaded": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "split_id": split_config["id"],
        "source_sha256": data_config["source_sha256"],
        "records": records,
        "official_mlo_test_loaded": False,
        "confirmation_labels_loaded": False,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    destination = repository / str(split_config["materialized_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_fusion_manifest(
    data_config: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    repository = root or find_repo_root()
    source = repository / str(data_config["source_path"])
    observed_hash = _source_sha256(source)
    if observed_hash != str(data_config["source_sha256"]):
        raise RuntimeError("USNA-large source hash drift")
    with xr.open_dataset(source, decode_cf=True) as dataset:
        times = dataset["time"].values
        variable_metadata = {
            name: {
                "units": dataset[name].attrs.get("units"),
                "source": dataset[name].attrs.get("source"),
                "description": dataset[name].attrs.get("description"),
            }
            for name in ("Cn2_3m", *RAW_WEATHER, "Dir_3m")
        }
    manifest = DatasetManifest(
        dataset_id=str(data_config["source_dataset_id"]),
        source_url=str(data_config["upstream_repository"]),
        citation=str(data_config["citation"]),
        accessed_at=datetime.now(UTC),
        terms=DatasetTerms(status="open", redistribution="metadata_only"),
        files=[
            DatasetFile(
                relative_path=source.relative_to(repository).as_posix(),
                sha256=observed_hash,
                bytes=source.stat().st_size,
            )
        ],
        label_provenance="direct_observation",
        geometry={
            "kind": "surface_layer_path_observation",
            "site": data_config["site"],
            **dict(data_config["geometry"]),
        },
        instrument=dict(data_config["instrument"]),
        wavelength_nm=None,
        units={
            "raw_target": "m^(-2/3)",
            "target": "log10(m^(-2/3))",
            "variables": variable_metadata,
        },
        coverage={
            "rows": len(times),
            "temporal_start": str(
                np.datetime_as_string(times[0], unit="m")
            ),
            "temporal_end": str(np.datetime_as_string(times[-1], unit="m")),
            "official_partitions": {
                "train": [0, 524339],
                "validation": [524340, 788208],
                "test": [788209, 1291224],
            },
        },
        qc={
            "upstream_commit": data_config["upstream_commit"],
            "source_sha256_verified": True,
            "raw_data_edited": False,
            "official_mlo_test_loaded": False,
            "usna_validation_confirmation_eligible": False,
            "confirmation_release_required": True,
            "acknowledgement": (
                "Preserve OTBench USNA, ONR, and DE-JTO acknowledgements."
            ),
        },
    )
    destination = repository / str(data_config["manifest_path"])
    write_manifest(manifest, destination)
    failures = verify_manifest(destination, repository)
    if failures:
        raise RuntimeError("Fusion source manifest failure: " + ", ".join(failures))
    return destination


def _wind_channels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degrees = np.asarray(
        [COMPASS_DEGREES.get(str(value).strip().upper(), np.nan) for value in values],
        dtype=np.float64,
    )
    present = np.isfinite(degrees)
    radians = np.deg2rad(np.nan_to_num(degrees))
    return np.sin(radians), np.cos(radians), present.astype(np.float64)


def _point_features(
    raw: dict[str, np.ndarray],
    timestamps: np.ndarray,
    *,
    physics_tokens: str,
) -> tuple[np.ndarray, list[str]]:
    base = np.column_stack([raw[name] for name in RAW_WEATHER]).astype(np.float64)
    wind_sin, wind_cos, wind_present = _wind_channels(raw["Dir_3m"])
    base = np.column_stack((base, wind_sin, wind_cos))
    names = [*RAW_WEATHER, "Dir_3m_sin", "Dir_3m_cos"]
    original_present = np.isfinite(base)
    original_present[:, -2:] = wind_present[:, None].astype(bool)
    missing = (~original_present).astype(np.float64)

    physics_parts: list[np.ndarray] = []
    physics_names: list[str] = []
    if physics_tokens != "none":
        temperature = raw["T_3m"].astype(np.float64)
        pressure = raw["P_3m"].astype(np.float64)
        humidity = np.clip(raw["RH_3m"].astype(np.float64), 1e-3, 100.0)
        gamma = np.log(humidity / 100.0) + (
            17.625 * temperature / (243.04 + temperature)
        )
        dew_point = 243.04 * gamma / (17.625 - gamma)
        vapor_pressure = 6.1094 * np.exp(
            17.625 * dew_point / (243.04 + dew_point)
        )
        kelvin = temperature + 273.15
        dry_refractivity = 77.6 * pressure / kelvin
        wet_refractivity = 3.73e5 * vapor_pressure / kelvin**2
        total_refractivity = dry_refractivity + wet_refractivity
        surface_delta = temperature - raw["T_0m"].astype(np.float64)
        physics_parts.extend(
            [dew_point, vapor_pressure, total_refractivity, surface_delta]
        )
        physics_names.extend(
            [
                "dew_point",
                "vapor_pressure",
                "total_refractivity",
                "air_surface_temperature_difference",
            ]
        )
        if physics_tokens == "full":
            minute = timestamps.astype("datetime64[m]").astype(np.int64)
            day_phase = 2.0 * np.pi * ((minute % 1440) / 1440.0)
            year_phase = 2.0 * np.pi * ((minute % 525960) / 525960.0)
            physics_parts.extend(
                [
                    dry_refractivity,
                    wet_refractivity,
                    np.sin(day_phase),
                    np.cos(day_phase),
                    np.sin(year_phase),
                    np.cos(year_phase),
                ]
            )
            physics_names.extend(
                [
                    "dry_refractivity",
                    "wet_refractivity",
                    "diurnal_sin",
                    "diurnal_cos",
                    "annual_sin",
                    "annual_cos",
                ]
            )
    physics = (
        np.column_stack(physics_parts)
        if physics_parts
        else np.empty((len(base), 0), dtype=np.float64)
    )
    values = np.column_stack((base, physics, missing))
    feature_names = [*names, *physics_names, *[f"{name}_missing" for name in names]]
    return values, feature_names


@dataclass(frozen=True)
class FusionNormalization:
    target_mean: float
    target_std: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    fit_fold: str
    fit_role: str = "train"


@dataclass(frozen=True)
class _RoleArrays:
    timestamps: np.ndarray
    target: np.ndarray
    features: np.ndarray
    feature_present: np.ndarray
    feature_names: list[str]
    raw_start_offset: int


class FusionSequenceDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        arrays: _RoleArrays,
        normalization: FusionNormalization,
        *,
        horizons: list[int],
        max_examples: int | None,
    ) -> None:
        self.timestamps = arrays.timestamps.astype("datetime64[m]")
        self.target = arrays.target
        self.feature_names = arrays.feature_names
        mean = normalization.feature_mean
        std = normalization.feature_std
        finite = np.isfinite(arrays.features)
        self.features = np.where(
            finite,
            (arrays.features - mean) / std,
            0.0,
        ).astype(np.float32)
        self.feature_present = arrays.feature_present.astype(np.float32)
        log_target = np.full_like(arrays.target, np.nan, dtype=np.float64)
        valid_target = np.isfinite(arrays.target) & (arrays.target > 0)
        log_target[valid_target] = np.log10(arrays.target[valid_target])
        self.log_target = log_target
        self.target_scaled = (
            (log_target - normalization.target_mean) / normalization.target_std
        ).astype(np.float32)
        minute_values = self.timestamps.astype(np.int64)
        lookup = {int(value): index for index, value in enumerate(minute_values)}
        examples: list[tuple[int, int, tuple[np.ndarray, ...]]] = []
        for origin, minute in enumerate(minute_values):
            if minute % 5 or not np.isfinite(log_target[origin]):
                continue
            scale_indices: list[np.ndarray] = []
            valid_context = True
            for rows, spacing in SCALE_SPECS.values():
                required = np.arange(
                    int(minute) - (rows - 1) * spacing,
                    int(minute) + 1,
                    spacing,
                    dtype=np.int64,
                )
                indices = np.asarray([lookup.get(int(value), -1) for value in required])
                if np.any(indices < 0) or np.any(~np.isfinite(log_target[indices])):
                    valid_context = False
                    break
                scale_indices.append(indices)
            if not valid_context:
                continue
            for horizon in horizons:
                target_index = lookup.get(int(minute) + horizon)
                if target_index is None or not np.isfinite(log_target[target_index]):
                    continue
                examples.append((origin, target_index, tuple(scale_indices)))
        if not examples:
            raise ValueError("Role produced no valid Fusion v2 sequences")
        if max_examples is not None and len(examples) > max_examples:
            selected = np.linspace(
                0, len(examples) - 1, max_examples, dtype=np.int64
            )
            examples = [examples[int(index)] for index in selected]
        self.examples = examples
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.examples)

    def target_quantile(self, quantile: float) -> float:
        """Return a target quantile from this role's materialized sequences."""
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Target quantile must be between zero and one")
        targets = np.asarray(
            [self.log_target[target_index] for _, target_index, _ in self.examples],
            dtype=np.float64,
        )
        if not len(targets) or not np.all(np.isfinite(targets)):
            raise ValueError("Sequence targets are unavailable for quantile fitting")
        return float(np.quantile(targets, quantile))

    def target_fraction_at_or_above(self, threshold: float) -> float:
        targets = np.asarray(
            [self.log_target[target_index] for _, target_index, _ in self.examples],
            dtype=np.float64,
        )
        if not len(targets) or not np.all(np.isfinite(targets)):
            raise ValueError("Sequence targets are unavailable for tail accounting")
        return float(np.mean(targets >= threshold))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        origin, target_index, scale_indices = self.examples[index]
        item: dict[str, Tensor] = {}
        for scale, indices in zip(SCALE_SPECS, scale_indices, strict=True):
            levels = self.target_scaled[indices]
            differences = np.concatenate(([0.0], np.diff(levels))).astype(np.float32)
            cadence = np.ones_like(levels, dtype=np.float32)
            item[f"{scale}_history"] = torch.from_numpy(
                np.column_stack((levels, differences, cadence)).astype(np.float32)
            )
            item[f"{scale}_weather"] = torch.from_numpy(self.features[indices])
            item[f"{scale}_present"] = torch.from_numpy(
                self.feature_present[indices]
            )
        origin_minute = int(self.timestamps[origin].astype(np.int64))
        target_minute = int(self.timestamps[target_index].astype(np.int64))
        item.update(
            {
                "target": torch.tensor(
                    self.log_target[target_index], dtype=torch.float32
                ),
                "persistence": torch.tensor(
                    self.log_target[origin], dtype=torch.float32
                ),
                "horizon_minutes": torch.tensor(
                    target_minute - origin_minute, dtype=torch.float32
                ),
                "future_weather": torch.from_numpy(self.features[target_index]),
                "future_weather_present": torch.from_numpy(
                    self.feature_present[target_index]
                ),
                "target_timestamp_minute": torch.tensor(
                    target_minute, dtype=torch.int64
                ),
                "origin_timestamp_minute": torch.tensor(
                    origin_minute, dtype=torch.int64
                ),
            }
        )
        return item


class FusionDataModule(L.LightningDataModule):
    """Purged rolling-origin USNA-large data with explicit confirmation release."""

    def __init__(
        self,
        data_config: dict[str, Any],
        split_config: dict[str, Any],
        *,
        fold_id: str,
        physics_tokens: str,
        batch_size: int,
        num_workers: int,
        max_train_examples: int | None = None,
        max_evaluation_examples: int | None = None,
        release_confirmation: bool = False,
    ) -> None:
        super().__init__()
        if physics_tokens not in {"none", "base", "full"}:
            raise ValueError("physics_tokens must be none, base, or full")
        self.data_config = data_config
        self.split_config = split_config
        self.fold_id = fold_id
        self.physics_tokens = physics_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_examples = max_train_examples
        self.max_evaluation_examples = max_evaluation_examples
        self.release_confirmation = release_confirmation
        self.materialized_split: dict[str, Any] = {}
        self.normalization: FusionNormalization | None = None
        self.train_set: FusionSequenceDataset | None = None
        self.selection_set: FusionSequenceDataset | None = None
        self.calibration_set: FusionSequenceDataset | None = None
        self.confirmation_set: FusionSequenceDataset | None = None
        self.feature_names: list[str] = []
        self.role_rows: dict[str, int] = {}
        self.confirmation_consumption: dict[str, Any] | None = None

    @property
    def source(self) -> Path:
        return find_repo_root() / str(self.data_config["source_path"])

    def prepare_data(self) -> None:
        root = find_repo_root()
        manifest = root / str(self.data_config["manifest_path"])
        if manifest.is_file():
            failures = verify_manifest(manifest, root)
            if failures:
                raise RuntimeError(
                    "Fusion dataset manifest verification failed: "
                    + ", ".join(failures)
                )
        else:
            write_fusion_manifest(self.data_config)
        materialized = root / str(self.split_config["materialized_path"])
        if materialized.is_file():
            payload = json.loads(materialized.read_text(encoding="utf-8"))
            if (
                payload.get("split_id") != self.split_config["id"]
                or payload.get("source_sha256") != self.data_config["source_sha256"]
                or payload.get("confirmation_labels_loaded") is not False
            ):
                raise RuntimeError("Materialized Fusion split identity failure")
            self.materialized_split = payload
        else:
            self.materialized_split = materialize_fusion_split(
                self.data_config,
                self.split_config,
            )

    def _role_config(self, role: str) -> dict[str, Any]:
        if self.fold_id == "final":
            if role == "confirmation":
                return dict(self.split_config["confirmation"])
            return dict(self.split_config["final_fit"][role])
        fold = next(
            (
                item
                for item in self.split_config["folds"]
                if item["id"] == self.fold_id
            ),
            None,
        )
        if fold is None:
            raise ValueError(f"Unknown rolling fold: {self.fold_id}")
        return dict(fold[role])

    def _load_role(self, role: str, *, labels: bool) -> _RoleArrays:
        role_config = self._role_config(role)
        start = int(role_config["raw_start_offset"])
        end = int(role_config["raw_end_offset_inclusive"]) + 1
        variables = [*RAW_WEATHER, "Dir_3m"]
        if labels:
            variables.append("Cn2_3m")
        with xr.open_dataset(self.source, decode_cf=True) as dataset:
            selected = dataset[variables].isel(time=slice(start, end)).load()
            timestamps = dataset["time"].isel(time=slice(start, end)).values
            raw = {name: selected[name].values for name in variables}
        observed_hash = identity_sha256(timestamps)
        if observed_hash != str(role_config["identities_sha256"]):
            raise RuntimeError(f"Identity drift while loading {self.fold_id} {role}")
        target = (
            np.asarray(raw.pop("Cn2_3m"), dtype=np.float64)
            if labels
            else np.full(len(timestamps), np.nan, dtype=np.float64)
        )
        features, names = _point_features(
            raw,
            timestamps,
            physics_tokens=self.physics_tokens,
        )
        present = np.isfinite(features).astype(np.float64)
        self.role_rows[role] = len(timestamps)
        return _RoleArrays(
            timestamps=timestamps,
            target=target,
            features=features,
            feature_present=present,
            feature_names=names,
            raw_start_offset=start,
        )

    @staticmethod
    def _fit_normalization(
        train: _RoleArrays,
        fold_id: str,
    ) -> FusionNormalization:
        valid = np.isfinite(train.target) & (train.target > 0)
        target = np.log10(train.target[valid])
        if len(target) < 100:
            raise ValueError("Too few valid direct Cn2 training labels")
        feature_mean = np.nanmean(train.features, axis=0)
        feature_std = np.nanstd(train.features, axis=0)
        feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
        feature_std = np.where(
            np.isfinite(feature_std) & (feature_std > 1e-6),
            feature_std,
            1.0,
        )
        return FusionNormalization(
            target_mean=float(target.mean()),
            target_std=float(max(target.std(), 1e-6)),
            feature_mean=feature_mean,
            feature_std=feature_std,
            fit_fold=fold_id,
        )

    def setup(self, stage: str | None = None) -> None:
        del stage
        if not self.materialized_split:
            self.prepare_data()
        train = self._load_role("train", labels=True)
        calibration = self._load_role("calibration", labels=True)
        self.normalization = self._fit_normalization(train, self.fold_id)
        self.feature_names = train.feature_names
        horizons = [
            int(value) for value in self.data_config["sampling"]["horizons_minutes"]
        ]
        self.train_set = FusionSequenceDataset(
            train,
            self.normalization,
            horizons=horizons,
            max_examples=self.max_train_examples,
        )
        self.calibration_set = FusionSequenceDataset(
            calibration,
            self.normalization,
            horizons=horizons,
            max_examples=self.max_evaluation_examples,
        )
        if self.fold_id != "final":
            selection = self._load_role("selection", labels=True)
            self.selection_set = FusionSequenceDataset(
                selection,
                self.normalization,
                horizons=horizons,
                max_examples=self.max_evaluation_examples,
            )

    def release_confirmation_data(self) -> FusionSequenceDataset:
        """Atomically consume and then load confirmation labels exactly once.

        Final-fit training and calibration call ``setup`` without ever constructing
        this dataset. The explicit release happens only after the frozen model has
        finished fitting, which keeps confirmation labels out of trainer state.
        """
        if self.fold_id != "final" or not self.release_confirmation:
            raise RuntimeError(
                "Confirmation labels require final fold and explicit release"
            )
        if self.normalization is None:
            raise RuntimeError("Final-fit normalization is unavailable")
        if self.confirmation_set is not None:
            return self.confirmation_set
        confirmation_config = self.split_config["confirmation"]
        self.confirmation_consumption = consume_partition(
            {
                "consumption_id": confirmation_config["consumption_id"],
                "dataset_id": self.data_config["id"],
                "split_id": self.split_config["id"],
                "role": "confirmation",
                "identities_sha256": confirmation_config["identities_sha256"],
                "candidate_commit_required": True,
                "official_mlo_test_loaded": False,
            }
        )
        confirmation = self._load_role("confirmation", labels=True)
        horizons = [
            int(value) for value in self.data_config["sampling"]["horizons_minutes"]
        ]
        self.confirmation_set = FusionSequenceDataset(
            confirmation,
            self.normalization,
            horizons=horizons,
            max_examples=self.max_evaluation_examples,
        )
        return self.confirmation_set

    def _loader(
        self,
        dataset: FusionSequenceDataset | None,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Tensor]]:
        if dataset is None:
            raise RuntimeError("Requested Fusion v2 role is unavailable")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.selection_set, shuffle=False)

    def calibration_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        return self._loader(self.calibration_set, shuffle=False)

    def confirmation_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        if not self.release_confirmation:
            raise RuntimeError(
                "Confirmation labels require the explicit --release-confirmation flag"
            )
        return self._loader(self.confirmation_set, shuffle=False)
