from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from strata_ot.data.acquire import (
    _validate_eso_mass_csv,
    acquire_eso_paranal_mass,
)
from strata_ot.data.registry import verify_manifest
from strata_ot.data.schema import DatasetManifest


def _headers() -> list[str]:
    headers = ["Date time"]
    for layer in range(1, 7):
        headers.extend(
            (
                f"Layer {layer} Cn2 [10**(-15)m**(1/3)]",
                f"Layer {layer} Cn2 RMS",
                f"Layer {layer} height [m]",
            )
        )
    headers.extend(
        (
            "Layer 0 Cn2 [10**(-15)m**(1/3)]",
            "Layer 0 Cn2 RMS",
            "Layer 0 height [m]",
            "Number of layers",
        )
    )
    return headers


def _payload(*, duplicate_time: bool = False, leading_blank: bool = False) -> bytes:
    headers = _headers()
    rows: list[list[str]] = []
    for row_index, timestamp in enumerate(
        (
            "2020-01-01T00:00:00",
            "2020-01-01T00:01:20",
        )
    ):
        if duplicate_time and row_index == 1:
            timestamp = "2020-01-01T00:00:00"
        row = [timestamp]
        for layer, height in enumerate((500, 1000, 2000, 4000, 8000, 16000), start=1):
            row.extend((str(layer + row_index), "0.1", str(height)))
        row.extend(("7.0", "0.2", "0", "7"))
        rows.append(row)
    lines = [",".join(headers), *(",".join(row) for row in rows)]
    prefix = "\n" if leading_blank else ""
    return (prefix + "\n".join(lines) + "\n").encode()


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "id": "eso-test",
        "source_url": "https://archive.eso.org/form",
        "query_url": "https://archive.eso.org/query",
        "help_url": "https://archive.eso.org/help",
        "license_url": "https://www.eso.org/license",
        "citation": "ESO MASS test citation",
        "query": {
            "start_date": "2020-01-01..2020-01-02",
            "max_rows_returned": 100,
            "output_fields": [f"field-{index}" for index in range(len(_headers()) - 1)],
        },
        "cache_dir": "data/raw/eso-test",
        "snapshot_file": "profile.csv",
        "request_record_file": "request.json",
        "manifest_path": "data/manifests/eso-test.json",
        "maximum_download_bytes": 1_000_000,
        "geometry": {
            "kind": "profile",
            "layer_heights_m": [0, 500, 1000, 2000, 4000, 8000, 16000],
        },
        "instrument": {
            "name": "MASS",
            "wavelength_nm": 500,
        },
    }


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(payload)),
        }

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_eso_mass_acquisition_records_provenance_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    payload = _payload(leading_blank=True)
    monkeypatch.setattr(
        "strata_ot.data.acquire.urlopen",
        lambda request, timeout: _Response(payload, config["query_url"]),
    )

    destination = acquire_eso_paranal_mass(config, tmp_path)
    manifest_path = tmp_path / config["manifest_path"]
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    request = json.loads(
        (destination / config["request_record_file"]).read_text(encoding="utf-8")
    )

    assert manifest.terms.status == "open"
    assert manifest.terms.redistribution == "allowed"
    assert manifest.label_provenance == "direct_observation"
    assert manifest.coverage["rows"] == 2
    assert manifest.qc["status"] == "PASS"
    assert manifest.qc["column_training_authorized"] is False
    assert manifest.qc["official_mlo_test_loaded"] is False
    assert request["anonymous_access"] is True
    assert request["credentials_used"] is False
    assert verify_manifest(manifest_path, tmp_path) == []
    with pytest.raises(FileExistsError, match="verify it instead"):
        acquire_eso_paranal_mass(config, tmp_path)


def test_eso_mass_qc_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _validate_eso_mass_csv(_payload(duplicate_time=True), _config(tmp_path))
