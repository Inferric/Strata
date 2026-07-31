from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
            if not np.isfinite(numeric):
                rendered.append("")
                continue
            if numeric <= 0:
                raise ValueError("The canonical log10 target requires positive values")
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
        canonical_numeric = pd.to_numeric(
            pd.Series(canonical), errors="coerce"
        ).to_numpy(float)
        if not np.array_equal(np.isfinite(upstream), np.isfinite(canonical_numeric)):
            raise ValueError("TaskApi missing targets do not align with the source")
        valid = np.isfinite(upstream)
        difference = np.abs(upstream[valid] - canonical_numeric[valid])
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
            "Acquired bytes do not match the checked-in expected manifest"
        )


def _profile_column(headers: list[str], prefix: str) -> str:
    matches = [header for header in headers if header.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one profile column beginning with {prefix!r}, got {matches}"
        )
    return matches[0]


def _parse_archive_number(value: str, *, column: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.upper() == "N/A":
        return None
    try:
        numeric = float(stripped)
    except ValueError as error:
        raise ValueError(f"Non-numeric archive value in {column}: {value!r}") from error
    if not np.isfinite(numeric):
        raise ValueError(f"Non-finite archive value in {column}: {value!r}")
    return numeric


def _validate_eso_mass_csv(
    payload: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    text = payload.decode("utf-8-sig")
    csv_lines = text.splitlines()
    while csv_lines and not csv_lines[0].strip():
        csv_lines.pop(0)
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    headers = list(reader.fieldnames or [])
    output_fields = list(config["query"]["output_fields"])
    expected_field_count = 1 + len(output_fields)
    if not headers or headers[0] != "Date time":
        raise ValueError("ESO MASS response is missing the Date time header")
    if len(headers) != expected_field_count or len(set(headers)) != len(headers):
        raise ValueError(
            "ESO MASS response field count or uniqueness changed: "
            f"expected {expected_field_count}, got {len(headers)}"
        )

    layer_columns: dict[int, dict[str, str]] = {}
    for layer in range(1, 7):
        layer_columns[layer] = {
            "strength": _profile_column(headers, f"Layer {layer} Cn2 ["),
            "uncertainty": _profile_column(headers, f"Layer {layer} Cn2 RMS"),
            "height": _profile_column(headers, f"Layer {layer} height [m]"),
        }
    layer_columns[0] = {
        "strength": _profile_column(headers, "Layer 0 Cn2 ["),
        "uncertainty": _profile_column(headers, "Layer 0 Cn2 RMS"),
        "height": _profile_column(headers, "Layer 0 height [m]"),
    }
    grid_size_column = _profile_column(headers, "Number of layers")
    expected_heights = {
        0: 0.0,
        1: 500.0,
        2: 1000.0,
        3: 2000.0,
        4: 4000.0,
        5: 8000.0,
        6: 16000.0,
    }
    rows = list(reader)
    if not rows:
        raise ValueError("ESO MASS query returned no rows")
    max_rows = int(config["query"]["max_rows_returned"])
    if len(rows) >= max_rows:
        raise ValueError(
            "ESO MASS query reached its row limit; the frozen interval may be truncated"
        )

    timestamps: list[datetime] = []
    missing_counts = {header: 0 for header in headers[1:]}
    finite_counts = {header: 0 for header in headers[1:]}
    for row_number, row in enumerate(rows, start=2):
        raw_timestamp = str(row.get("Date time", "")).strip()
        try:
            timestamp = datetime.fromisoformat(raw_timestamp)
        except ValueError as error:
            raise ValueError(
                f"Invalid ESO MASS timestamp on CSV row {row_number}: {raw_timestamp!r}"
            ) from error
        timestamps.append(timestamp)

        for header in headers[1:]:
            numeric = _parse_archive_number(str(row.get(header, "")), column=header)
            if numeric is None:
                missing_counts[header] += 1
            else:
                finite_counts[header] += 1

        grid_size = _parse_archive_number(
            str(row.get(grid_size_column, "")),
            column=grid_size_column,
        )
        if grid_size is not None and grid_size != 7.0:
            raise ValueError(
                f"Unexpected MASS-DIMM grid size {grid_size} on CSV row {row_number}"
            )
        for layer, columns in layer_columns.items():
            height = _parse_archive_number(
                str(row.get(columns["height"], "")),
                column=columns["height"],
            )
            if height is not None and height != expected_heights[layer]:
                raise ValueError(
                    f"Unexpected layer-{layer} height {height} on CSV row {row_number}"
                )
            for kind in ("strength", "uncertainty"):
                value = _parse_archive_number(
                    str(row.get(columns[kind], "")),
                    column=columns[kind],
                )
                if value is not None and value < 0:
                    raise ValueError(
                        f"Negative layer-{layer} {kind} on CSV row {row_number}"
                    )

    if timestamps != sorted(timestamps):
        raise ValueError("ESO MASS timestamps are not nondecreasing")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("ESO MASS timestamps contain duplicates")
    query_start, query_end = str(config["query"]["start_date"]).split("..", maxsplit=1)
    start = datetime.fromisoformat(query_start)
    end = datetime.fromisoformat(query_end)
    if timestamps[0] < start or timestamps[-1] > end:
        raise ValueError("ESO MASS timestamps fall outside the frozen query interval")
    cadences = np.diff(
        np.asarray([timestamp.timestamp() for timestamp in timestamps], dtype=float)
    )
    return {
        "status": "PASS",
        "rows": len(rows),
        "columns": len(headers),
        "headers": headers,
        "temporal_start": timestamps[0].isoformat(),
        "temporal_end": timestamps[-1].isoformat(),
        "unique_timestamps": len(set(timestamps)),
        "cadence_seconds": {
            "median": float(np.median(cadences)) if len(cadences) else None,
            "p95": float(np.quantile(cadences, 0.95)) if len(cadences) else None,
            "maximum": float(np.max(cadences)) if len(cadences) else None,
        },
        "missing_counts": missing_counts,
        "finite_counts": finite_counts,
        "layer_heights_m": [expected_heights[layer] for layer in range(7)],
        "row_limit_reached": False,
        "timestamps_nondecreasing": True,
        "timestamps_unique": True,
        "profile_values_nonnegative": True,
    }


def acquire_eso_paranal_mass(config: dict[str, Any], root: Path) -> Path:
    """Acquire one immutable, bounded ESO Paranal MASS profile snapshot."""
    destination = root / str(config["cache_dir"])
    snapshot_path = destination / str(config["snapshot_file"])
    request_path = destination / str(config["request_record_file"])
    if snapshot_path.exists() or request_path.exists():
        raise FileExistsError(
            f"Raw snapshot already exists at {destination}; verify it instead of overwriting"
        )
    query = dict(config["query"])
    form_fields: dict[str, str] = {
        "wdbo": "csv/download",
        "max_rows_returned": str(query["max_rows_returned"]),
        "start_date": str(query["start_date"]),
    }
    for field in query["output_fields"]:
        form_fields[f"tab_{field}"] = "on"
    query_url = str(config["query_url"])
    request = Request(
        query_url,
        data=urlencode(form_fields).encode("ascii"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Strata-OT public-research acquisition/1",
        },
        method="POST",
    )
    maximum_bytes = int(config["maximum_download_bytes"])
    with urlopen(request, timeout=60) as response:  # noqa: S310 - frozen HTTPS URL
        final_url = str(response.geturl())
        content_type = str(response.headers.get("Content-Type", ""))
        response_headers = {
            str(key): str(value)
            for key, value in response.headers.items()
            if str(key).lower()
            in {"content-type", "content-length", "content-disposition", "date", "server"}
        }
        payload = response.read(maximum_bytes + 1)
    if final_url.rstrip("/") != query_url.rstrip("/"):
        raise RuntimeError(f"ESO MASS request redirected unexpectedly to {final_url}")
    if not content_type.lower().startswith("text/plain"):
        raise RuntimeError(f"ESO MASS response is not CSV text: {content_type!r}")
    if not payload or len(payload) > maximum_bytes:
        raise RuntimeError(
            f"ESO MASS response size {len(payload)} is outside the frozen byte bound"
        )
    qc = _validate_eso_mass_csv(payload, config)

    accessed_at = datetime.now(UTC)
    request_record = {
        "schema_version": 1,
        "dataset_id": config["id"],
        "accessed_at": accessed_at.isoformat(),
        "method": "POST",
        "query_url": query_url,
        "form_fields": form_fields,
        "final_url": final_url,
        "response_headers": response_headers,
        "response_bytes": len(payload),
        "anonymous_access": True,
        "credentials_used": False,
        "clickthrough_accepted": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(payload)
    request_path.write_text(
        json.dumps(request_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        dataset_id=str(config["id"]),
        source_url=str(config["source_url"]),
        citation=str(config["citation"]),
        accessed_at=accessed_at,
        terms=DatasetTerms(status="open", redistribution="allowed"),
        files=file_records((snapshot_path, request_path), root),
        label_provenance="direct_observation",
        geometry=dict(config["geometry"]),
        instrument=dict(config["instrument"]),
        wavelength_nm=float(config["instrument"]["wavelength_nm"]),
        units={
            "layer_strength": "10^-15 m^(1/3), archive integrated J_i units",
            "layer_height": "m above telescope pupil",
            "time": "UTC",
            "warning": "integrated layer strength, not pointwise Cn2",
        },
        coverage={
            "rows": qc["rows"],
            "columns": qc["columns"],
            "temporal_start": qc["temporal_start"],
            "temporal_end": qc["temporal_end"],
            "query_interval": query["start_date"],
        },
        qc={
            **qc,
            "adapter": "eso_paranal_mass",
            "anonymous_https": True,
            "license_url": config["license_url"],
            "help_url": config["help_url"],
            "raw_data_edited": False,
            "observation_operator": (
                "stellar-scintillation inversion to integrated MASS layer strengths; "
                "ground layer is DIMM minus MASS"
            ),
            "column_training_authorized": False,
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
    )
    _record_or_verify_manifest(manifest, root / str(config["manifest_path"]))
    return destination


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

    task_info = task.get_info()
    dataset_name = str(task_info["ds_name"])
    datasets_config = json.loads(
        (package_root / "config" / "datasets.json").read_text(encoding="utf-8")
    )
    local_data_path = str(datasets_config[dataset_name]["local_data_path"])
    dataset_directory = package_root / "data" / dataset_name
    source_files = (
        package_root / "config" / "tasks.json",
        package_root / "config" / "datasets.json",
        dataset_directory / local_data_path,
        dataset_directory / "citation.md",
        dataset_directory / "README.md",
    )
    snapshots.extend(source_files)

    manifest = DatasetManifest(
        dataset_id=str(config["id"]),
        source_url="https://github.com/CDJellen/otbench",
        citation=str(
            config.get(
                "citation",
                "Jellen, Nelson, Brownell, and Burkhardt (2024), "
                "Effective Benchmarks for Optical Turbulence Modeling, "
                "doi:10.1175/AIES-D-24-0003.1.",
            )
        ),
        accessed_at=datetime.now(UTC),
        terms=DatasetTerms(status="review_required", redistribution="metadata_only"),
        files=file_records(snapshots, root),
        label_provenance="direct_observation",
        geometry={
            "kind": str(config.get("geometry_kind", "surface_point")),
            "site": config.get("site", dataset_name),
            "latitude": task_info.get("obs_lat"),
            "longitude": task_info.get("obs_lon"),
            "time_zone": task_info.get("obs_tz"),
            **dict(config.get("geometry", {})),
        },
        instrument=dict(
            config.get("instrument", {"name": "see upstream task metadata"})
        ),
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
            "upstream_task_info": task_info,
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
    adapter = config.get("adapter")
    if adapter == "otbench":
        destination = acquire_otbench(config, root)
    elif adapter == "eso_paranal_mass":
        destination = acquire_eso_paranal_mass(config, root)
    else:
        raise SystemExit(f"Unsupported adapter: {config.get('adapter')}")
    print(f"Acquired immutable snapshot at {destination}")


if __name__ == "__main__":
    main()
