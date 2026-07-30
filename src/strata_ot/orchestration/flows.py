from __future__ import annotations

import json
from typing import Any

from prefect import flow, task

from strata_ot.config import find_repo_root
from strata_ot.evaluation.evaluate import evaluate_gates
from strata_ot.reporting.render import render_report
from strata_ot.training.train import run_experiment


@task(retries=1, retry_delay_seconds=30, log_prints=True)
def train_first_real(config_path: str, fast_dev_run: bool) -> dict[str, Any]:
    return run_experiment(config_path, fast_dev_run=fast_dev_run)


@task(log_prints=True)
def report_and_gate(
    summary: dict[str, Any],
    output_directory: str,
) -> dict[str, Any]:
    root = find_repo_root()
    summary_path = root / "artifacts" / "latest" / "summary.json"
    output_dir = (root / output_directory).resolve()
    reports_root = (root / "reports" / "generated").resolve()
    if not output_dir.is_relative_to(reports_root):
        raise ValueError("Report output must remain under reports/generated")
    pdf = render_report(summary_path, output_dir)
    refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
    import yaml

    gates = yaml.safe_load(
        (root / "configs" / "evaluation" / "gates" / "public_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    result = evaluate_gates(refreshed, gates)
    refreshed["gate_result"] = result
    summary_path.write_text(
        json.dumps(refreshed, indent=2) + "\n",
        encoding="utf-8",
    )
    pdf = render_report(summary_path, output_dir)
    refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
    result = evaluate_gates(refreshed, gates)
    refreshed["gate_result"] = result
    summary_path.write_text(
        json.dumps(refreshed, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    runs = summary.get("runs", [])
    run_count = len(runs) if isinstance(runs, list) else 0
    return {"pdf": str(pdf), "gate": result, "run_count": run_count}


@flow(name="strata-ot-first-real-run", log_prints=True)
def first_real_flow(
    config_path: str = "configs/experiments/first_real_mlo.yaml",
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    summary = train_first_real(config_path, fast_dev_run)
    return report_and_gate(summary, "reports/generated/first-real-run")


@flow(name="strata-ot-bounded-candidate", log_prints=True)
def candidate_flow(config_path: str, output_directory: str) -> dict[str, Any]:
    summary = train_first_real(config_path, False)
    return report_and_gate(summary, output_directory)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the first Prefect research flow")
    parser.add_argument("--config", default="configs/experiments/first_real_mlo.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    print(first_real_flow(args.config, args.fast_dev_run))


if __name__ == "__main__":
    main()
