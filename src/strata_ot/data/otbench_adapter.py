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


class SequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        context: int,
        row_ids: np.ndarray | None = None,
    ):
        if context < 1:
            raise ValueError("context must be positive")
        if len(features) < context:
            raise ValueError("split is shorter than the requested context")
        if row_ids is not None and len(row_ids) != len(features):
            raise ValueError("row identities must align with features")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.row_ids = (
            np.asarray(row_ids, dtype=str)
            if row_ids is not None
            else np.arange(len(features)).astype(str)
        )
        self.context = context

    def __len__(self) -> int:
        return len(self.features) - self.context + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        end = index + self.context
        return self.features[index:end], self.target[end - 1]

    @property
    def endpoint_row_ids(self) -> np.ndarray:
        return self.row_ids[self.context - 1 :]


@dataclass(frozen=True)
class PreparedMetadata:
    input_dim: int
    feature_names: list[str]
    target_transform: str
    split_id: str
    dataset_id: str


class OtbenchDataModule(L.LightningDataModule):
    """Blocked-time preparation with train-only imputation and scaling."""

    def __init__(
        self,
        data_config: dict[str, Any],
        split_config: dict[str, Any],
        *,
        batch_size: int,
        num_workers: int,
        context: int = 1,
    ):
        super().__init__()
        self.data_config = data_config
        self.split_config = split_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context = context
        self.metadata: PreparedMetadata | None = None
        self.train_set: SequenceDataset | None = None
        self.validation_set: SequenceDataset | None = None
        self.test_set: SequenceDataset | None = None
        self.normalization: dict[str, list[float]] = {}

    def prepare_data(self) -> None:
        root = find_repo_root()
        manifest = root / str(self.data_config["manifest_path"])
        destination = (
            root
            / str(self.data_config["cache_dir"])
            / str(self.data_config["id"])
        )
        if not (destination / "train_features.csv").exists():
            acquire_otbench(self.data_config, root)
        failures = verify_manifest(manifest, root)
        if failures:
            raise RuntimeError("Raw snapshot identity failure: " + ", ".join(failures))
        if not destination.exists():
            raise FileNotFoundError(destination)

    def setup(self, stage: str | None = None) -> None:
        del stage
        root = find_repo_root()
        destination = (
            root
            / str(self.data_config["cache_dir"])
            / str(self.data_config["id"])
        )
        raw_features = {
            partition: pd.read_csv(destination / f"{partition}_features.csv")
            for partition in ("train", "validation", "test")
        }
        raw_targets = {
            partition: pd.read_csv(destination / f"{partition}_target.csv")
            for partition in ("train", "validation", "test")
        }
        candidate_columns = [
            column
            for column in raw_features["train"]
            if not column.startswith("_")
        ]
        train_numeric = raw_features["train"][candidate_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        usable = [
            column
            for column in train_numeric
            if float(train_numeric[column].notna().mean()) >= 0.80
        ]
        if not usable:
            raise ValueError("No sufficiently populated numeric otbench features")
        matrices: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        row_ids: dict[str, np.ndarray] = {}
        for partition in ("train", "validation", "test"):
            matrices[partition] = (
                raw_features[partition][usable]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            targets[partition] = pd.to_numeric(
                raw_targets[partition]["target"], errors="coerce"
            ).to_numpy(float)
            row_ids[partition] = raw_targets[partition]["_upstream_index"].astype(str).to_numpy()

        finite_positive = targets["train"][
            np.isfinite(targets["train"]) & (targets["train"] > 0)
        ]
        if finite_positive.size and float(np.nanmedian(finite_positive)) < 1e-4:
            for partition in targets:
                targets[partition] = np.where(
                    targets[partition] > 0,
                    np.log10(targets[partition]),
                    np.nan,
                )
            target_transform = "log10(raw_cn2)"
        else:
            target_transform = "upstream_log10_cn2"
        manifest_payload = json.loads(
            (root / str(self.data_config["manifest_path"])).read_text(encoding="utf-8")
        )
        if manifest_payload.get("qc", {}).get("canonical_serialization"):
            target_transform = "canonical_log10(raw_float32_cn2)"

        for partition in targets:
            valid = np.isfinite(targets[partition])
            matrices[partition] = matrices[partition][valid]
            targets[partition] = targets[partition][valid]
            row_ids[partition] = row_ids[partition][valid]
        self._apply_boundary_purge(matrices, targets, row_ids)

        train_median = np.nanmedian(matrices["train"], axis=0)
        train_median = np.where(np.isfinite(train_median), train_median, 0.0)
        for partition in matrices:
            matrices[partition] = np.where(
                np.isfinite(matrices[partition]),
                matrices[partition],
                train_median,
            )
        train_mean = matrices["train"].mean(axis=0)
        train_std = matrices["train"].std(axis=0)
        train_std = np.where(train_std > 1e-8, train_std, 1.0)
        for partition in matrices:
            matrices[partition] = (matrices[partition] - train_mean) / train_std
        self.normalization = {
            "median": train_median.tolist(),
            "mean": train_mean.tolist(),
            "std": train_std.tolist(),
        }

        self.train_set = SequenceDataset(
            matrices["train"], targets["train"], self.context, row_ids["train"]
        )
        self.validation_set = SequenceDataset(
            matrices["validation"],
            targets["validation"],
            self.context,
            row_ids["validation"],
        )
        self.test_set = SequenceDataset(
            matrices["test"], targets["test"], self.context, row_ids["test"]
        )
        self.metadata = PreparedMetadata(
            input_dim=len(usable),
            feature_names=usable,
            target_transform=target_transform,
            split_id=str(self.split_config["id"]),
            dataset_id=str(self.data_config["id"]),
        )
        self._seal_materialized_split(
            root,
            row_ids["train"],
            row_ids["validation"],
            row_ids["test"],
        )

    def _apply_boundary_purge(
        self,
        matrices: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        row_ids: dict[str, np.ndarray],
    ) -> None:
        purge = int(self.split_config.get("purge_gap_rows", 0))
        slices = {
            "train": slice(None, -purge if purge else None),
            "validation": slice(purge, -purge if purge else None),
            "test": slice(purge, None),
        }
        for partition, selection in slices.items():
            matrices[partition] = matrices[partition][selection]
            targets[partition] = targets[partition][selection]
            row_ids[partition] = row_ids[partition][selection]
            if len(targets[partition]) <= self.context:
                raise ValueError(f"{partition} split is too small after purge gaps")

    def _seal_materialized_split(
        self, root: Path, train: np.ndarray, validation: np.ndarray, test: np.ndarray
    ) -> None:
        payload = {
            "split_id": self.split_config["id"],
            "dataset_id": self.data_config["id"],
            "strategy": self.split_config["strategy"],
            "indices": {
                "train": train.tolist(),
                "validation": validation.tolist(),
                "test": test.tolist(),
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

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        assert self.validation_set is not None
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
