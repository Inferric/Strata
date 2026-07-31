from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from strata_ot.api import main as api_main


def test_fusion_endpoint_prefers_program_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = (
        tmp_path
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
    )
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
        tmp_path
        / "reports"
        / "generated"
        / "fusion-v2-program"
        / "fusion-v2-program-report.pdf"
    )
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4\nprogram\n%%EOF\n")
    evidence = (
        tmp_path
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
    )
    evidence.mkdir(parents=True)
    (evidence / "program-summary.json").write_text(
        json.dumps(
            {
                "task_kind": "fusion_v2_program",
                "report_pdf": (
                    "reports/generated/fusion-v2-program/"
                    "fusion-v2-program-report.pdf"
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "_root_or_app", lambda: tmp_path)
    response = TestClient(api_main.app).get("/api/reports/latest")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-1.4")
