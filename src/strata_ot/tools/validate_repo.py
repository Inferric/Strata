from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from strata_ot.config import find_repo_root

REQUIRED_PATHS = (
    "AGENTS.md",
    "GOAL.md",
    "configs/experiments/first_real_mlo.yaml",
    "configs/evaluation/gates/public_v1.yaml",
    "schemas/experiment_proposal.schema.json",
    "reports/templates/experiment_report.tex.j2",
    "research/references.bib",
    "scripts/wait_for_services.py",
    "web/package.json",
)


def validate_repository(root: Path | None = None) -> list[str]:
    repository = root or find_repo_root()
    failures = [
        f"missing:{relative}"
        for relative in REQUIRED_PATHS
        if not (repository / relative).exists()
    ]
    for schema_path in (repository / "schemas").glob("*.schema.json"):
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            failures.append(f"schema:{schema_path.name}:{error}")
    for yaml_path in (repository / "configs").rglob("*.yaml"):
        try:
            yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append(f"yaml:{yaml_path.relative_to(repository)}:{error}")
    local_skills = repository / ".agents" / "skills"
    if not local_skills.exists() or not list(local_skills.glob("*/SKILL.md")):
        failures.append("skills:no project-local skills installed")
    return failures


def main() -> None:
    failures = validate_repository()
    if failures:
        raise SystemExit("\n".join(failures))
    print("Repository contract is valid")


if __name__ == "__main__":
    main()
