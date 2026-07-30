#!/usr/bin/env python3
"""Validate the minimum safety and reproducibility contract for an experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED = {
    "experiment_id",
    "hypothesis",
    "dataset_id",
    "split_id",
    "model_id",
    "seeds",
    "max_gpu_hours",
    "max_cost_usd",
    "allowed_paths",
    "stop_conditions",
}

FORBIDDEN_PREFIXES = (
    "data/raw",
    "data/sealed",
    "configs/splits/frozen",
    "configs/evaluation/gates",
    ".env",
    ".git",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_run_contract.py EXPERIMENT.json", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED - payload.keys())
    errors: list[str] = []

    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    if not isinstance(payload.get("seeds"), list) or not payload.get("seeds"):
        errors.append("seeds must be a non-empty list")
    if payload.get("max_gpu_hours", 0) <= 0:
        errors.append("max_gpu_hours must be positive")
    if payload.get("max_cost_usd", -1) < 0:
        errors.append("max_cost_usd must be non-negative")

    for candidate in payload.get("allowed_paths", []):
        normalized = str(Path(candidate)).replace("\\", "/").lstrip("./")
        if normalized.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden mutable path: {candidate}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"valid experiment contract: {payload['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
