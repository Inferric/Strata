from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import mlflow

from strata_ot.config import find_repo_root, load_yaml
from strata_ot.data.registry import sha256_file, verify_manifest
from strata_ot.data.schema import DatasetManifest
from strata_ot.training.train import _configure_utf8_output

CONFIG_PATH = Path("configs/data/eso_paranal_mass_profile_v1.yaml")
PROTOCOL_PATH = Path("docs/ESO_PARANAL_MASS_DATASET.md")
MANIFEST_PATH = Path("data/manifests/eso-paranal-mass-profile-2020-01-v1.json")
SUMMARY_PATH = Path("artifacts/data-qc/eso-paranal-mass-profile-2020-01-v1.json")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], payload)


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not resolve repository revision: {completed.stderr}")
    return completed.stdout.strip()


def build_profile_dataset_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    config = load_yaml(str(CONFIG_PATH), root=root)
    manifest_file = root / MANIFEST_PATH
    manifest = DatasetManifest.model_validate_json(
        manifest_file.read_text(encoding="utf-8")
    )
    failures = verify_manifest(manifest_file, root)
    qc = manifest.qc
    missing_counts = cast(dict[str, int], qc["missing_counts"])
    nonzero_missing = [
        {
            "field": field,
            "missing": int(count),
            "fraction": int(count) / int(manifest.coverage["rows"]),
        }
        for field, count in sorted(
            missing_counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
        if int(count) > 0
    ]
    checks = {
        "anonymous_https_access": (
            "PASS" if qc.get("anonymous_https") is True else "FAIL"
        ),
        "manifest_checksum_verification": "PASS" if not failures else "FAIL",
        "bounded_response_size": (
            "PASS"
            if sum(record.bytes for record in manifest.files)
            <= int(config["maximum_download_bytes"])
            else "FAIL"
        ),
        "exact_requested_schema": (
            "PASS"
            if int(qc["columns"]) == 1 + len(config["query"]["output_fields"])
            else "FAIL"
        ),
        "timestamps_nondecreasing": (
            "PASS" if qc.get("timestamps_nondecreasing") is True else "FAIL"
        ),
        "timestamps_unique": (
            "PASS" if qc.get("timestamps_unique") is True else "FAIL"
        ),
        "seven_layer_geometry": (
            "PASS"
            if qc.get("layer_heights_m")
            == [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
            else "FAIL"
        ),
        "profile_values_nonnegative": (
            "PASS" if qc.get("profile_values_nonnegative") is True else "FAIL"
        ),
        "official_mlo_test_sealed": (
            "PASS" if qc.get("official_mlo_test_loaded") is False else "FAIL"
        ),
        "usna_confirmation_labels_sealed": (
            "PASS"
            if qc.get("usna_confirmation_labels_loaded") is False
            else "FAIL"
        ),
        "column_training_authorized": "NOT_EVALUATED",
        "champion_promotion": "NOT_EVALUATED",
    }
    status = "PASS" if all(
        state == "PASS"
        for name, state in checks.items()
        if name not in {"column_training_authorized", "champion_promotion"}
    ) else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "eso-paranal-mass-profile-2020-01-v1-qc",
        "task_kind": "dataset_qc",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "status": status,
        "plain_language_question": (
            "Can a bounded public profile snapshot be acquired reproducibly "
            "without treating inverted MASS layers as pointwise Cn2?"
        ),
        "plain_language_conclusion": (
            "Yes for adapter and geometry validation: 7,857 unique January "
            "2020 observations passed the frozen checks. The snapshot is too "
            "narrow and too gappy for Column training or a promotion claim."
            if status == "PASS"
            else (
                "No. At least one frozen acquisition or profile-QC condition "
                "failed, so the snapshot cannot support model work."
            )
        ),
        "protocol": PROTOCOL_PATH.as_posix(),
        "config": CONFIG_PATH.as_posix(),
        "repository_revision": _git_revision(root),
        "dataset_id": manifest.dataset_id,
        "data_identity": {
            "manifest_path": MANIFEST_PATH.as_posix(),
            "manifest_sha256": sha256_file(manifest_file),
            "files": [
                {
                    "relative_path": record.relative_path.replace("\\", "/"),
                    "bytes": record.bytes,
                    "sha256": record.sha256,
                }
                for record in manifest.files
            ],
        },
        "provenance": {
            "source_url": manifest.source_url,
            "help_url": qc["help_url"],
            "license_url": qc["license_url"],
            "citation": manifest.citation,
            "accessed_at": manifest.accessed_at.isoformat(),
            "terms_status": manifest.terms.status,
            "redistribution": manifest.terms.redistribution,
            "label_provenance": manifest.label_provenance,
            "observation_operator": qc["observation_operator"],
            "raw_data_edited": qc["raw_data_edited"],
        },
        "request": {
            "interval": config["query"]["start_date"],
            "row_limit": config["query"]["max_rows_returned"],
            "maximum_download_bytes": config["maximum_download_bytes"],
            "requested_field_count": len(config["query"]["output_fields"]),
        },
        "coverage": manifest.coverage,
        "geometry": manifest.geometry,
        "instrument": manifest.instrument,
        "wavelength_nm": manifest.wavelength_nm,
        "units": manifest.units,
        "qc": {
            "rows": qc["rows"],
            "columns": qc["columns"],
            "cadence_seconds": qc["cadence_seconds"],
            "nonzero_missing_fields": nonzero_missing,
            "row_limit_reached": qc["row_limit_reached"],
        },
        "checks": checks,
        "limitations": [
            "The snapshot covers one month at one nighttime observatory.",
            (
                "The maximum cadence gap is several days; model-ready sequences "
                "require an explicit gap policy."
            ),
            (
                "MASS layer strengths are response-weighted scintillation "
                "inversions and the ground layer is a DIMM-minus-MASS difference."
            ),
            "No weather join, chronological model split, or baseline is frozen.",
            "Passing dataset QC does not authorize Column training or promotion.",
        ],
        "next_phase": (
            "Freeze a separate multi-season acquisition and purged profile "
            "development protocol, then reproduce a response-aware baseline "
            "before implementing Strata-OT Column."
        ),
        "resources": {
            "downloaded_bytes": sum(record.bytes for record in manifest.files),
            "cloud_cost_usd": 0.0,
            "gpu_hours": 0.0,
        },
        "report_pdf": None,
    }


def write_profile_dataset_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = root / SUMMARY_PATH
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_profile_dataset_summary(root)
    for key in ("evidence_run_id", "report_pdf", "drive_url"):
        if isinstance(existing.get(key), str):
            summary[key] = existing[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_profile_dataset_evidence(summary_path: Path, report_path: Path) -> str:
    root = find_repo_root(summary_path.parent)
    summary = _load_object(summary_path)
    existing = summary.get("evidence_run_id")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    if isinstance(existing, str) and existing:
        run_context = mlflow.start_run(run_id=existing)
    else:
        mlflow.set_experiment("strata-ot-data-qc")
        run_context = mlflow.start_run(
            run_name="eso-paranal-mass-profile-2020-01-v1-qc",
            tags={
                "dataset_id": summary["dataset_id"],
                "run_kind": "dataset_evidence",
                "evidence_role": "profile-acquisition-qc",
                "gate_status": summary["status"],
                "column_training_authorized": "false",
                "official_mlo_test_loaded": "false",
                "usna_confirmation_labels_loaded": "false",
            },
        )
    with run_context as run:
        summary["evidence_run_id"] = run.info.run_id
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        mlflow.log_metrics(
            {
                "data/rows": float(summary["coverage"]["rows"]),
                "data/columns": float(summary["coverage"]["columns"]),
                "data/downloaded_bytes": float(summary["resources"]["downloaded_bytes"]),
                "data/cadence_median_seconds": float(
                    summary["qc"]["cadence_seconds"]["median"]
                ),
                "data/cadence_p95_seconds": float(
                    summary["qc"]["cadence_seconds"]["p95"]
                ),
                "data/cadence_maximum_seconds": float(
                    summary["qc"]["cadence_seconds"]["maximum"]
                ),
            }
        )
        for path, artifact_path in (
            (summary_path, "evidence"),
            (root / MANIFEST_PATH, "evidence"),
            (root / CONFIG_PATH, "protocol"),
            (root / PROTOCOL_PATH, "protocol"),
            (report_path, "reports"),
            (report_path.with_suffix(".tex"), "reports"),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(description="Build ESO MASS profile dataset evidence")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/eso-paranal-mass-qc/"
            "eso-paranal-mass-qc-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    summary_path = write_profile_dataset_summary(root)
    if args.log_mlflow:
        print(log_profile_dataset_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
