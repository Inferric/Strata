from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root without assuming the caller's working directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate the Strata-OT repository root")


def load_yaml(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (root or find_repo_root()) / candidate
    with candidate.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {candidate}")
    return value


def resolve_experiment(path: str | Path) -> dict[str, Any]:
    root = find_repo_root()
    experiment = load_yaml(path, root=root)
    for key in ("data", "model", "trainer", "evaluation"):
        reference = experiment.get(key)
        if not isinstance(reference, str):
            raise ValueError(f"Experiment field {key!r} must be a config path")
        experiment[f"{key}_config"] = load_yaml(reference, root=root)
    experiment["_repo_root"] = str(root)
    return experiment
