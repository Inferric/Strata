from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from strata_ot.data.schema import DatasetFile, DatasetManifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(paths: Iterable[Path], root: Path) -> list[DatasetFile]:
    return [
        DatasetFile(
            relative_path=str(path.relative_to(root)),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in sorted(paths)
    ]


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_manifest(manifest_path: Path, repository_root: Path) -> list[str]:
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest.files:
        path = repository_root / record.relative_path
        if not path.exists():
            failures.append(f"missing:{record.relative_path}")
        elif path.stat().st_size != record.bytes:
            failures.append(f"size:{record.relative_path}")
        elif sha256_file(path) != record.sha256:
            failures.append(f"checksum:{record.relative_path}")
    return failures
