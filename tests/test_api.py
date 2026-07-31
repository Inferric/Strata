from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from strata_ot.api import main as api_main


def test_fusion_endpoint_prefers_program_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = tmp_path / "artifacts" / "experiments" / "strata-fusion-v2-program"
    evidence.mkdir(parents=True)
    (evidence / "screen-summary.json").write_text(
        json.dumps(
            {
                "task_kind": "fusion_v2_screen",
                "experiment_id": "screen",
            }
        ),
        encoding="utf-8",
    )
    (evidence / "program-summary.json").write_text(
        json.dumps(
            {
                "task_kind": "fusion_v2_program",
                "experiment_id": "program",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "_root_or_app", lambda: tmp_path)
    response = TestClient(api_main.app).get("/api/fusion")
    assert response.status_code == 200
    assert response.json()["experiment_id"] == "program"


def test_latest_report_prefers_program_synthesis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = (
        tmp_path / "reports" / "generated" / "fusion-v2-program" / "fusion-v2-program-report.pdf"
    )
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4\nprogram\n%%EOF\n")
    evidence = tmp_path / "artifacts" / "experiments" / "strata-fusion-v2-program"
    evidence.mkdir(parents=True)
    (evidence / "program-summary.json").write_text(
        json.dumps(
            {
                "task_kind": "fusion_v2_program",
                "report_pdf": ("reports/generated/fusion-v2-program/fusion-v2-program-report.pdf"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "_root_or_app", lambda: tmp_path)
    response = TestClient(api_main.app).get("/api/reports/latest")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-1.4")


def test_cycle_report_serves_only_recorded_program_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = (
        tmp_path
        / "reports"
        / "generated"
        / "fusion-v28-horizon-bilinear"
        / "fusion-v28-horizon-bilinear-report.pdf"
    )
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4\ncycle\n%%EOF\n")
    evidence = tmp_path / "artifacts" / "experiments" / "strata-fusion-v2-program"
    evidence.mkdir(parents=True)
    (evidence / "program-summary.json").write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "cycle_id": "v28-horizon-bilinear",
                        "report_pdf": report.relative_to(tmp_path).as_posix(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "_root_or_app", lambda: tmp_path)
    client = TestClient(api_main.app)
    response = client.get("/api/reports/cycle/v28-horizon-bilinear")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-1.4")
    assert client.get("/api/reports/cycle/not-recorded").status_code == 404


def test_datasets_normalizes_profile_qc_gate_states(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "profile.json").write_text(
        json.dumps({"dataset_id": "profile"}),
        encoding="utf-8",
    )
    qc_root = tmp_path / "artifacts" / "data-qc"
    qc_root.mkdir(parents=True)
    (qc_root / "profile.json").write_text(
        json.dumps(
            {
                "checks": {
                    "manifest_checksum_verification": "PASS",
                    "geometry": "PASS",
                    "column_training_authorized": "NOT_EVALUATED",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "_root_or_app", lambda: tmp_path)

    response = TestClient(api_main.app).get("/api/datasets")

    assert response.status_code == 200
    verification = response.json()["items"][0]["verification"]
    assert verification == {
        "manifest_verified": True,
        "checks_passed": 2,
        "checks_total": 2,
        "split_id": None,
        "split_sha256": None,
    }
