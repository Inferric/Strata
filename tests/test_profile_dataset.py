from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import strata_ot.evaluation.profile_dataset as profile_dataset
from strata_ot.data.registry import file_records, verify_manifest, write_manifest
from strata_ot.data.schema import DatasetManifest, DatasetTerms
from strata_ot.reporting.render import _environment


def test_profile_dataset_summary_and_latex_preserve_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("config.json")
    protocol_path = Path("protocol.md")
    manifest_path = Path("manifest.json")
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    snapshot = raw_dir / "snapshot.csv"
    request = raw_dir / "request.json"
    snapshot.write_text("Date time,Layer 1\n2020-01-01T00:00:00,1\n", encoding="utf-8")
    request.write_text("{}\n", encoding="utf-8")
    config = {
        "query": {
            "start_date": "2020-01-01..2020-02-01",
            "max_rows_returned": 100,
            "output_fields": ["field"],
        },
        "maximum_download_bytes": 1000,
    }
    (tmp_path / config_path).write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / protocol_path).write_text("# protocol\n", encoding="utf-8")
    manifest = DatasetManifest(
        dataset_id="eso-test",
        source_url="https://archive.eso.org/form",
        citation="ESO citation",
        accessed_at=datetime(2026, 7, 31, tzinfo=UTC),
        terms=DatasetTerms(status="open", redistribution="allowed"),
        files=file_records((snapshot, request), tmp_path),
        label_provenance="direct_observation",
        geometry={
            "kind": "profile",
            "site": "Paranal",
            "latitude_deg": -24.6,
            "longitude_deg": -70.4,
            "observatory_altitude_m": 2635,
            "layer_heights_m": [0, 500, 1000, 2000, 4000, 8000, 16000],
        },
        instrument={"name": "MASS", "wavelength_nm": 500},
        wavelength_nm=500,
        units={"layer_strength": "integrated J"},
        coverage={
            "rows": 2,
            "columns": 2,
            "temporal_start": "2020-01-01T00:00:00",
            "temporal_end": "2020-01-01T00:01:20",
        },
        qc={
            "status": "PASS",
            "missing_counts": {"Layer 1": 0},
            "anonymous_https": True,
            "columns": 2,
            "timestamps_nondecreasing": True,
            "timestamps_unique": True,
            "layer_heights_m": [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
            "profile_values_nonnegative": True,
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
            "help_url": "https://archive.eso.org/help",
            "license_url": "https://www.eso.org/license",
            "observation_operator": "scintillation inversion",
            "raw_data_edited": False,
            "rows": 2,
            "cadence_seconds": {"median": 80.0, "p95": 80.0, "maximum": 80.0},
            "row_limit_reached": False,
        },
    )
    write_manifest(manifest, tmp_path / manifest_path)
    manifest_payload = json.loads(
        (tmp_path / manifest_path).read_text(encoding="utf-8")
    )
    assert all(
        "\\" not in record["relative_path"]
        for record in manifest_payload["files"]
    )
    manifest_payload["files"][0]["relative_path"] = manifest_payload["files"][0][
        "relative_path"
    ].replace("/", "\\")
    (tmp_path / manifest_path).write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    assert verify_manifest(tmp_path / manifest_path, tmp_path) == []
    monkeypatch.setattr(profile_dataset, "CONFIG_PATH", config_path)
    monkeypatch.setattr(profile_dataset, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(profile_dataset, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(profile_dataset, "_git_revision", lambda root: "a" * 40)

    summary = profile_dataset.build_profile_dataset_summary(tmp_path)
    assert summary["status"] == "PASS"
    assert summary["checks"]["column_training_authorized"] == "NOT_EVALUATED"
    assert summary["checks"]["official_mlo_test_sealed"] == "PASS"
    rendered = (
        _environment(
            Path(__file__).resolve().parents[1] / "reports" / "templates"
        )
        .get_template("dataset_qc_report.tex.j2")
        .render(summary=summary)
    )
    assert "pointwise" in rendered
    assert "Strata-OT Column" in rendered

    manifest_payload["files"][0]["relative_path"] = "../outside.csv"
    (tmp_path / manifest_path).write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    assert verify_manifest(tmp_path / manifest_path, tmp_path) == [
        "unsafe-path:../outside.csv"
    ]
