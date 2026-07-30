from __future__ import annotations

import json
import os
from datetime import UTC, datetime
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


def consume_partition(
    record: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Atomically consume a preregistered evaluation role before label access."""
    repository = root or find_repo_root()
    consumption_id = record.get("consumption_id")
    if not isinstance(consumption_id, str) or not consumption_id:
        raise ValueError("Consumption record requires a non-empty consumption_id")
    require_unconsumed(consumption_id, root=repository)
    destination = repository / RUNTIME_REGISTRY
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "consumed": []}
    if destination.is_file():
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    completed = {
        **record,
        "consumed_at": datetime.now(UTC).isoformat(),
        "process_id": os.getpid(),
        "disposition": "consumed_no_retuning",
    }
    records = list(payload.get("consumed", []))
    records.append(completed)
    payload["consumed"] = records
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return completed
