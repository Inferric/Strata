from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from strata_ot.config import find_repo_root

DEFAULT_REGISTRY = Path("configs/evaluation/consumed_partitions.yaml")
RUNTIME_REGISTRY = Path("data/sealed/evaluation/consumed_partitions.json")


def _records(root: Path, registry_path: Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    configured = root / registry_path
    payload = yaml.safe_load(configured.read_text(encoding="utf-8"))
    records = list(payload.get("consumed", []))
    runtime = root / RUNTIME_REGISTRY
    if runtime.is_file():
        runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
        records.extend(runtime_payload.get("consumed", []))
    return [dict(record) for record in records]


def consumed_partition(
    consumption_id: str,
    *,
    root: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any] | None:
    repository = root or find_repo_root()
    for record in _records(repository, registry_path):
        if record.get("consumption_id") == consumption_id:
            return record
    return None


def require_unconsumed(
    consumption_id: str,
    *,
    root: Path | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    record = consumed_partition(
        consumption_id,
        root=root,
        registry_path=registry_path,
    )
    if record is None:
        return
    role = record.get("role", "evaluation partition")
    consumed_at = record.get("consumed_at", "unknown time")
    raise RuntimeError(
        f"{role} is already consumed ({consumption_id}, {consumed_at}); "
        "released labels cannot be loaded again for tuning or reruns"
    )
