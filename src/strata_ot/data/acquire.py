from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strata_ot.config import find_repo_root, load_yaml
from strata_ot.data.registry import file_records, write_manifest
from strata_ot.data.schema import DatasetManifest, DatasetTerms

CANONICAL_TARGET_DECIMAL_PLACES = 12
MAX_TASK_CANONICALIZATION_DIFFERENCE = 2e-6


def _as_frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, pd.Series):
        frame = value.to_frame(name=value.name or name)
    elif hasattr(value, "to_dataframe"):
        frame = value.to_dataframe().reset_index()
    else:
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[:, None]
        frame = pd.DataFrame(array)
    frame.columns = [str(column) for column in frame.columns]
    return frame.reset_index(names="_upstream_index")


def _canonical_log10(values: pd.Series) -> list[str]:
    """Render the upstream float32 target deterministically across CPU math libraries."""
    rendered: list[str] = []
    quantum = Decimal(1).scaleb(-CANONICAL_TARGET_DECIMAL_PLACES)
    with localcontext() as context:
        context.prec = 50
        log_ten = Decimal(10).ln()
        for value in values:
            numeric = float(value)
            if not np.isfinite(numeric) or numeric <= 0:
                raise ValueError("The canonical log10 target requires finite positive values")
            logged = Decimal.from_float(numeric).ln() / log_ten
            rendered.append(
                format(logged.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")
            )
    return rendered


def _canonical_target_frame(task: Any, target: Any) -> pd.DataFrame:
    frame = _as_frame(target, "target")
    if "target" not in frame:
        non_index = [column for column in frame if column != "_upstream_index"]
        if len(non_index) != 1:
            raise ValueError("Could not identify the target column")
        frame = frame.rename(columns={non_index[0]: "target"})
    transforms = task.get_transforms()
    if bool(transforms.get("log_transform", {}).get("value")):
        if not isinstance(target, (pd.DataFrame, pd.Series)):
            raise TypeError("Canonical task targets require a pandas index")
        raw_target = task.get_df().loc[target.index, task.get_target_name()]
        if not isinstance(raw_target, pd.Series) or len(raw_target) != len(frame):
            raise ValueError("Could not align canonical targets to TaskApi row identities")
        canonical = _canonical_log10(raw_target)
        upstream = pd.to_numeric(frame["target"], errors="coerce").to_numpy(float)
        difference = np.abs(upstream - np.asarray(canonical, dtype=np.float64))
        if not np.all(np.isfinite(difference)):
            raise ValueError("TaskApi returned non-finite transformed targets")
        if float(difference.max(initial=0.0)) > MAX_TASK_CANONICALIZATION_DIFFERENCE:
            raise RuntimeError(
                "Canonical target differs materially from the pinned TaskApi output"
            )
        frame["target"] = canonical
    return frame[["_upstream_index", "target"]]


def _checkout_upstream(config: dict[str, Any], destination: Path) -> str:
    repository = str(config["upstream_repository"])
    expected_commit = str(config["upstream_commit"])
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                repository,
                str(destination),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Could not clone the pinned otbench source: {completed.stderr}")
        completed = subprocess.run(
            ["git", "-C", str(destination), "config", "core.autocrlf", "false"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Could not configure the pinned checkout: {completed.stderr}")
        completed = subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", expected_commit],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Could not check out pinned otbench commit: {completed.stderr}")
    completed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    actual_commit = completed.stdout.strip()
    if completed.returncode != 0 or actual_commit != expected_commit:
        raise RuntimeError(
            f"otbench source identity mismatch: expected {expected_commit}, got {actual_commit}"
        )
    return actual_commit


def _record_or_verify_manifest(manifest: DatasetManifest, path: Path) -> None:
    if not path.exists():
        write_manifest(manifest, path)
        return
    expected = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    expected_files = {
        record.relative_path: (record.sha256, record.bytes) for record in expected.files
    }
    observed_files = {
        record.relative_path: (record.sha256, record.bytes) for record in manifest.files
    }
    expected_commit = expected.qc.get("upstream_commit")
    observed_commit = manifest.qc.get("upstream_commit")
    if (
        expected.dataset_id != manifest.dataset_id
        or expected_files != observed_files
        or expected_commit != observed_commit
    ):
        raise RuntimeError(
            "Acquired otbench bytes do not match the checked-in expected manifest"
        )


def acquire_otbench(config: dict[str, Any], root: Path) -> Path:
    """Snapshot the exact public task output and write a content-addressed manifest."""
    destination = root / str(config["cache_dir"]) / str(config["id"])
    upstream = destination / "upstream"
    commit = _checkout_upstream(config, upstream)
    sys.path.insert(0, str(upstream))
    try:
        from otbench import TaskApi
    except ImportError as error:
        raise RuntimeError("Install the data extra with `uv sync --extra data`") from error

    package_root = upstream / "otbench"
    task = TaskApi(root_dir=str(package_root)).get_task(
        str(config["task"]),
        benchmark_fp=str(package_root / "benchmark" / "experiments.json"),
    )
    destination.mkdir(parents=True, exist_ok=True)
    snapshots: list[Path] = []
    partition_sizes: dict[str, int] = {}
    getters = {
        "train": task.get_train_data,
        "validation": task.get_validation_data,
        "test": task.get_test_data,
    }
    expected_paths = [
        destination / f"{partition}_{kind}.csv"
        for partition in getters
        for kind in ("features", "target")
    ]
    if any(path.exists() for path in expected_paths):
        raise FileExistsError(
            f"Raw snapshot already exists at {destination}; verify it instead of overwriting"
        )
    for partition, getter in getters.items():
        features, target = getter()
        feature_frame = _as_frame(features, "feature")
        target_frame = _canonical_target_frame(task, target)
        if len(feature_frame) != len(target_frame):
            raise ValueError(f"otbench returned misaligned {partition} features and target")
        features_path = destination / f"{partition}_features.csv"
        target_path = destination / f"{partition}_target.csv"
        feature_frame.to_csv(features_path, index=False, lineterminator="\n")
        target_frame.to_csv(target_path, index=False, lineterminator="\n")
        snapshots.extend((features_path, target_path))
        partition_sizes[partition] = len(feature_frame)

    source_files = (
        package_root / "config" / "tasks.json",
        package_root / "config" / "datasets.json",
        package_root / "data" / "mlo_cn2" / "mlo_cn2.nc",
    )
    snapshots.extend(source_files)

    manifest = DatasetManifest(
        dataset_id=str(config["id"]),
        source_url="https://github.com/CDJellen/otbench",
        citation=(
            "Jellen, Nelson, Brownell, and Burkhardt (2024), "
            "Effective Benchmarks for Optical Turbulence Modeling, "
            "doi:10.1175/AIES-D-24-0003.1."
        ),
        accessed_at=datetime.now(UTC),
        terms=DatasetTerms(status="review_required", redistribution="metadata_only"),
        files=file_records(snapshots, root),
        label_provenance="direct_observation",
        geometry={"kind": "surface_point", "site": config.get("site", "Mauna Loa")},
        instrument={"name": "see upstream task metadata"},
        wavelength_nm=None,
        units={"target": "log10(m^(-2/3)), transformed by the upstream task"},
        coverage={
            "partitions": partition_sizes,
            "rows": sum(partition_sizes.values()),
            "features": len(feature_frame.columns) - 1,
        },
        qc={
            "task": config["task"],
            "upstream_commit": commit,
            "upstream_task_info": task.get_info(),
            "upstream_transforms": task.get_transforms(),
            "dropna_task": ".dropna." in str(config["task"]),
            "canonical_serialization": {
                "line_terminator": "LF",
                "target_log10_source": "exact upstream raw float32 observation",
                "target_decimal_places": CANONICAL_TARGET_DECIMAL_PLACES,
                "maximum_allowed_task_difference": (
                    MAX_TASK_CANONICALIZATION_DIFFERENCE
                ),
            },
            "upstream_metadata_review_required_before_redistribution": True,
        },
    )
    _record_or_verify_manifest(manifest, root / str(config["manifest_path"]))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire a public Strata-OT dataset")
    parser.add_argument("--config", required=True, help="Dataset YAML path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(args.config, root=root)
    manifest_path = root / str(config["manifest_path"])
    if args.verify_only:
        from strata_ot.data.registry import verify_manifest

        failures = verify_manifest(manifest_path, root)
        if failures:
            raise SystemExit("Manifest verification failed: " + ", ".join(failures))
        print(f"Verified {manifest_path}")
        return
    if config.get("adapter") != "otbench":
        raise SystemExit(f"Unsupported adapter: {config.get('adapter')}")
    destination = acquire_otbench(config, root)
    print(f"Acquired immutable snapshot at {destination}")


if __name__ == "__main__":
    main()
