from __future__ import annotations

import json
from typing import Any

from prefect import flow, task

from strata_ot.config import find_repo_root
from strata_ot.evaluation.evaluate import evaluate_gates, gate_status
from strata_ot.reporting.render import render_report
from strata_ot.training.forecast import run_forecast_experiment
from strata_ot.training.fusion import run_fusion_candidate
from strata_ot.training.horizon import run_horizon_experiment
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
    experiment_summary = (
        root
        / "artifacts"
        / "experiments"
        / str(refreshed["experiment_id"])
        / "summary.json"
    )
    experiment_summary.parent.mkdir(parents=True, exist_ok=True)
    experiment_summary.write_text(
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


@task(retries=1, retry_delay_seconds=30, log_prints=True)
def train_forecast(
    config_path: str,
    fast_dev_run: bool,
    release_test: bool,
) -> dict[str, Any]:
    return run_forecast_experiment(
        config_path,
        fast_dev_run=fast_dev_run,
        release_test=release_test,
    )


@flow(name="strata-ot-short-forecast", log_prints=True)
def forecast_flow(
    config_path: str = "configs/experiments/mlo_short_forecast_dev.yaml",
    fast_dev_run: bool = False,
    release_test: bool = False,
) -> dict[str, Any]:
    summary = train_forecast(config_path, fast_dev_run, release_test)
    return report_and_gate(summary, "reports/generated/short-forecast-validation")


@task(log_prints=True)
def train_horizon(
    config_path: str,
    fast_dev_run: bool,
    release_assessment: bool,
) -> dict[str, Any]:
    return run_horizon_experiment(
        config_path,
        fast_dev_run=fast_dev_run,
        release_assessment=release_assessment,
    )


@task(log_prints=True)
def report_horizon(summary: dict[str, Any], output_directory: str) -> dict[str, Any]:
    root = find_repo_root()
    summary_path = root / "artifacts" / "latest" / "summary.json"
    output_dir = (root / output_directory).resolve()
    reports_root = (root / "reports" / "generated").resolve()
    if not output_dir.is_relative_to(reports_root):
        raise ValueError("Report output must remain under reports/generated")
    pdf = render_report(summary_path, output_dir)
    refreshed = json.loads(summary_path.read_text(encoding="utf-8"))
    claim = dict(refreshed.get("assessment_claim", {}))
    checks = dict(refreshed.get("checks", {}))
    required_checks = (
        "dataset_manifest_valid",
        "materialized_split_verified",
        "no_random_row_primary_split",
        "no_time_overlap_across_partitions",
        "preprocessing_fit_on_train_only",
        "official_mlo_test_sealed",
        "uncertainty_reported",
        "calibration_reported",
        "resource_metrics_reported",
        "within_gpu_budget",
        "within_storage_budget",
        "latex_pdf_compiled",
    )
    assessment_released = refreshed.get("assessment_released") is True
    condition_statuses = {
        name: gate_status(checks.get(name), evaluated=name in checks)
        for name in required_checks
    }
    failures = [
        name for name, status in condition_statuses.items() if status == "FAIL"
    ]
    condition_statuses["assessment_released"] = gate_status(
        assessment_released,
        evaluated=assessment_released,
    )
    for name, passed in dict(claim.get("conditions", {})).items():
        condition_name = f"primary_{name}"
        condition_statuses[condition_name] = gate_status(
            bool(passed),
            evaluated=assessment_released,
        )
        if assessment_released and not passed:
            failures.append(condition_name)
    passed = bool(claim.get("passed")) and not failures
    gate_result = {
        "gate_id": "mlo-weather-horizon-v1-preregistered",
        "status": gate_status(passed, evaluated=assessment_released),
        "passed": passed,
        "failures": sorted(set(failures)),
        "condition_statuses": condition_statuses,
        "evaluated_run_count": len(refreshed.get("runs", [])),
        "assessment_status": claim.get("status", "not_released"),
        "champion_promoted": False,
    }
    refreshed["gate_result"] = gate_result
    summary_path.write_text(
        json.dumps(refreshed, indent=2) + "\n", encoding="utf-8"
    )
    experiment_summary = (
        root
        / "artifacts"
        / "experiments"
        / str(refreshed["experiment_id"])
        / "summary.json"
    )
    experiment_summary.write_text(
        json.dumps(refreshed, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "gate-result.json").write_text(
        json.dumps(gate_result, indent=2) + "\n", encoding="utf-8"
    )
    pdf = render_report(summary_path, output_dir)
    return {
        "pdf": str(pdf),
        "gate": gate_result,
        "run_count": len(summary.get("runs", [])),
    }


@flow(name="strata-ot-mlo-weather-horizon-v1", log_prints=True)
def horizon_flow(
    config_path: str = "configs/experiments/mlo_weather_horizon_v1.yaml",
    fast_dev_run: bool = False,
    release_assessment: bool = False,
) -> dict[str, Any]:
    summary = train_horizon(config_path, fast_dev_run, release_assessment)
    return report_horizon(summary, "reports/generated/mlo-weather-horizon-v1")


@task(log_prints=True)
def train_fusion(
    config_path: str,
    candidate_id: str,
    fold_id: str,
    seed: int,
    screen: bool,
    release_confirmation: bool,
) -> dict[str, Any]:
    return run_fusion_candidate(
        config_path,
        candidate_id=candidate_id,
        fold_id=fold_id,
        seed=seed,
        screen=screen,
        release_confirmation=release_confirmation,
    )


@flow(name="strata-ot-fusion-v2-candidate", log_prints=True)
def fusion_flow(
    config_path: str = "configs/experiments/fusion_v2_program.yaml",
    candidate_id: str = "context-all",
    fold_id: str = "fold-1",
    seed: int = 17,
    screen: bool = True,
    release_confirmation: bool = False,
) -> dict[str, Any]:
    return train_fusion(
        config_path,
        candidate_id,
        fold_id,
        seed,
        screen,
        release_confirmation,
    )


@flow(name="strata-ot-fusion-v2-screen-program", log_prints=True)
def fusion_screen_program_flow(
    config_path: str = "configs/experiments/fusion_v2_program.yaml",
    fold_id: str = "fold-1",
    seed: int = 17,
) -> list[dict[str, Any]]:
    root = find_repo_root()
    import yaml

    experiment = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    candidate_ids = [
        str(candidate["id"]) for candidate in experiment["candidates"]
    ]
    results: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        results.append(
            train_fusion(
                config_path,
                candidate_id,
                fold_id,
                seed,
                True,
                False,
            )
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the first Prefect research flow")
    parser.add_argument("--config", default="configs/experiments/first_real_mlo.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    print(first_real_flow(args.config, args.fast_dev_run))


def forecast_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the short forecast Prefect flow")
    parser.add_argument(
        "--config", default="configs/experiments/mlo_short_forecast_dev.yaml"
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--release-test", action="store_true")
    args = parser.parse_args()
    print(forecast_flow(args.config, args.fast_dev_run, args.release_test))


def horizon_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the frozen MLO weather-horizon Prefect flow"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/mlo_weather_horizon_v1.yaml",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--release-assessment", action="store_true")
    args = parser.parse_args()
    print(
        horizon_flow(
            args.config,
            args.fast_dev_run,
            args.release_assessment,
        )
    )


def fusion_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one Fusion v2 candidate through Prefect"
    )
    parser.add_argument(
        "--config", default="configs/experiments/fusion_v2_program.yaml"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", default="fold-1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--release-confirmation", action="store_true")
    args = parser.parse_args()
    print(
        fusion_flow(
            args.config,
            args.candidate,
            args.fold,
            args.seed,
            not args.full,
            args.release_confirmation,
        )
    )


def fusion_program_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete preregistered Fusion v2 screen matrix"
    )
    parser.add_argument(
        "--config", default="configs/experiments/fusion_v2_program.yaml"
    )
    parser.add_argument("--fold", default="fold-1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    results = fusion_screen_program_flow(args.config, args.fold, args.seed)
    print(
        json.dumps(
            [
                {
                    "run_id": result["run_id"],
                    "candidate_id": result["candidate_id"],
                    "family": result["family"],
                }
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
