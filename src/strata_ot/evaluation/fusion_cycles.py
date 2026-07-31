from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import mlflow

from strata_ot.config import find_repo_root
from strata_ot.evaluation.fusion_program import (
    PRIMARY_HORIZONS,
    _load_object,
    _primary_mean,
)
from strata_ot.training.train import _configure_utf8_output

POINT_CONTROL = "point-huber-0p00"
POINT_INTERVENTIONS = (
    "point-huber-0p25",
    "point-huber-0p50",
    "point-huber-1p00",
)
POINT_CANDIDATES = (POINT_CONTROL, *POINT_INTERVENTIONS)
POINT_WEIGHTS = {
    "point-huber-0p00": 0.0,
    "point-huber-0p25": 0.25,
    "point-huber-0p50": 0.5,
    "point-huber-1p00": 1.0,
}


def _collect_point_loss_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in POINT_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        if run.get("resolved_configuration", {}).get("screen") is not True:
            continue
        repository = run.get("repository", {})
        if repository.get("git_dirty") is not False:
            raise RuntimeError(f"Dirty point-loss run: {candidate_id}")
        configured = float(
            run["resolved_configuration"]["candidate"].get(
                "point_loss_weight",
                0.0,
            )
        )
        if configured != POINT_WEIGHTS[candidate_id]:
            raise RuntimeError(
                f"Point-loss weight mismatch for {candidate_id}: {configured}"
            )
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate point-loss run for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(
                f"Point-loss run did not finish normally: {candidate_id}"
            )
        primary_values = (
            float(run["by_horizon"][str(horizon)][metric])
            for horizon in (5, *PRIMARY_HORIZONS)
            for metric in (
                "rmse_log10_cn2",
                "mae_log10_cn2",
                "bias_log10_cn2",
                "tail_mae_top_decile",
                "crps_gaussian",
                "interval_80_coverage",
            )
        )
        if not all(math.isfinite(value) for value in primary_values):
            raise RuntimeError(
                f"Non-finite point-loss evidence: {candidate_id}"
            )
        missing_artifacts = [
            relative
            for relative in run.get("artifacts", {}).values()
            if not (root / str(relative)).is_file()
        ]
        checkpoint = run.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            missing_artifacts.append(str(checkpoint))
        if missing_artifacts:
            raise RuntimeError(
                f"Incomplete point-loss artifacts for {candidate_id}: "
                + ", ".join(missing_artifacts)
            )
        matches[candidate_id] = run
    missing = sorted(set(POINT_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError(
            "Point-loss cycle is incomplete: " + ", ".join(missing)
        )
    return matches


def _row(run: dict[str, Any]) -> dict[str, Any]:
    primary = {
        metric: _primary_mean(run, metric)
        for metric in (
            "rmse_log10_cn2",
            "mae_log10_cn2",
            "bias_log10_cn2",
            "tail_mae_top_decile",
            "crps_gaussian",
            "interval_80_coverage",
            "interval_80_width",
            "student_t_nll",
        )
    }
    return {
        "candidate_id": run["candidate_id"],
        "run_id": run["run_id"],
        "weight": float(
            run["resolved_configuration"]["candidate"].get(
                "point_loss_weight",
                0.0,
            )
        ),
        "primary": primary,
        "operational": {
            metric: float(run["metrics"][metric])
            for metric in (
                "ood_score_mean",
                "selective_80_rmse_log10_cn2",
                "ood_top_quintile_rmse_log10_cn2",
                "inference_latency_ms_per_sample",
                "throughput_samples_per_second",
            )
            if isinstance(run["metrics"].get(metric), (int, float))
        },
        "by_horizon": run["by_horizon"],
        "parameters": run["parameters"],
        "wall_clock_seconds": float(run["metrics"]["wall_clock_seconds"]),
        "peak_total_board_vram_gib": float(
            run["metrics"]["peak_total_board_vram_gib"]
        ),
        "peak_process_allocated_vram_gib": float(
            run["metrics"].get("peak_process_allocated_vram_gib", 0.0)
        ),
        "process_rss_gib": float(run["metrics"].get("process_rss_gib", 0.0)),
        "artifact_storage_bytes": int(run["metrics"]["artifact_storage_bytes"]),
        "checkpoint": run.get("checkpoint"),
        "artifacts": run["artifacts"],
        "repository": run["repository"],
    }


def _point_decision(
    control: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    improvement = (
        float(control["primary"]["rmse_log10_cn2"])
        - float(intervention["primary"]["rmse_log10_cn2"])
    ) / float(control["primary"]["rmse_log10_cn2"])
    checks = {
        "two_percent_rmse_improvement": (
            "PASS" if improvement >= 0.02 else "FAIL"
        ),
        "tail_no_two_percent_regression": (
            "PASS"
            if float(intervention["primary"]["tail_mae_top_decile"])
            <= float(control["primary"]["tail_mae_top_decile"]) * 1.02
            else "FAIL"
        ),
        "crps_no_two_percent_regression": (
            "PASS"
            if float(intervention["primary"]["crps_gaussian"])
            <= float(control["primary"]["crps_gaussian"]) * 1.02
            else "FAIL"
        ),
        "coverage_70_to_90_percent": (
            "PASS"
            if 0.70
            <= float(intervention["primary"]["interval_80_coverage"])
            <= 0.90
            else "FAIL"
        ),
        "absolute_bias_at_most_0_25": (
            "PASS"
            if abs(float(intervention["primary"]["bias_log10_cn2"])) <= 0.25
            else "FAIL"
        ),
    }
    return {
        "relative_improvement_over_control": improvement,
        "checks": checks,
        "advances": all(value == "PASS" for value in checks.values()),
    }


def build_point_loss_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_point_loss_runs(root)
    revisions = {
        str(run["repository"]["git_revision"]) for run in runs.values()
    }
    if len(revisions) != 1:
        raise RuntimeError("Point-loss runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v21-point-loss",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V21_POINT_LOSS.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in POINT_CANDIDATES
        },
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_point_loss_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "point-loss-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_point_loss_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_point_loss_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_point_loss_runs(root)
    rows = [_row(manifests[candidate_id]) for candidate_id in POINT_CANDIDATES]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _point_decision(control, row)
        for row in rows[1:]
    }
    advancing = [
        row
        for row in rows[1:]
        if decisions[str(row["candidate_id"])]["advances"]
    ]
    selected = (
        min(
            advancing,
            key=lambda row: float(row["primary"]["rmse_log10_cn2"]),
        )
        if advancing
        else None
    )
    revisions = {
        str(row["repository"]["git_revision"]) for row in rows
    }
    if len(revisions) != 1:
        raise RuntimeError("Point-loss cycle has mixed repository revisions")
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Point-loss cycle has mixed data/split identities")
    status = "PASS" if selected is not None else "FAIL"
    parameter_counts = {int(row["parameters"]) for row in rows}
    if len(parameter_counts) != 1:
        raise RuntimeError(
            "Point-loss cycle changed model size across objective weights"
        )
    within_resource_envelope = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resource_envelope:
        raise RuntimeError("Point-loss cycle exceeded its frozen run envelope")
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v21-point-loss",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v21_point_loss",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "An explicit robust point objective will make the persistence "
            "residual learn location error that the heavy-tail NLL can absorb."
        ),
        "plain_language_question": (
            "Does directly penalizing forecast error help the custom model "
            "move beyond simply copying the latest observation?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed the frozen screen and is "
                "eligible for rolling robustness characterization."
            )
            if selected is not None
            else (
                "No explicit point-loss weight passed the frozen screen. The "
                "three non-improvements close this loss-weight family."
            )
        ),
        "protocol": "docs/FUSION_V21_POINT_LOSS.md",
        "freeze": (
            "research/experiments/fusion-v2-program/point-loss-freeze.json"
        ),
        "repository_revision": next(iter(revisions)),
        "data_identity": {
            "source_sha256": next(iter(source_hashes)),
            "split_sha256": next(iter(split_hashes)),
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "primary_horizon_minutes": list(PRIMARY_HORIZONS),
        "anchor_horizon_minutes": 5,
        "runs": rows,
        "decisions": decisions,
        "selected_candidate": (
            str(selected["candidate_id"]) if selected is not None else None
        ),
        "checks": {
            "four_arms_completed": "PASS",
            "three_interventions_evaluated": "PASS",
            "repository_identity": "PASS",
            "data_and_split_identity": "PASS",
            "architecture_parameter_identity": "PASS",
            "finite_metrics": "PASS",
            "artifacts_complete": "PASS",
            "per_run_resource_envelope": "PASS",
            "official_mlo_test_sealed": "PASS",
            "usna_confirmation_labels_sealed": "PASS",
            "family_advancement": status,
            "confirmation_evaluation": "NOT_EVALUATED",
        },
        "resources": {
            "neural_wall_clock_hours": sum(
                float(row["wall_clock_seconds"]) for row in rows
            )
            / 3600.0,
            "peak_total_board_vram_gib": max(
                float(row["peak_total_board_vram_gib"]) for row in rows
            ),
            "peak_process_allocated_vram_gib": max(
                float(row["peak_process_allocated_vram_gib"]) for row in rows
            ),
            "peak_process_rss_gib": max(
                float(row["process_rss_gib"]) for row in rows
            ),
            "artifact_storage_bytes": sum(
                int(row["artifact_storage_bytes"]) for row in rows
            ),
            "cloud_cost_usd": 0.0,
        },
        "next_hypothesis_family": (
            "rolling_robustness"
            if selected is not None
            else "training_data_scale"
        ),
        "report_pdf": None,
    }


def write_point_loss_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "point-loss-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_point_loss_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_point_loss_evidence(summary_path: Path, report_path: Path) -> str:
    try:
        root = find_repo_root()
    except FileNotFoundError:
        root = find_repo_root(summary_path.parent)
    summary = _load_object(summary_path)
    existing = summary.get("evidence_run_id")
    if isinstance(existing, str) and existing:
        return existing
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("strata-ot-fusion-v2-program")
    with mlflow.start_run(
        run_name="fusion-v21-point-loss-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-point-loss",
            "task_kind": summary["task_kind"],
            "gate_status": summary["status"],
            "confirmation_labels_loaded": "false",
            "official_mlo_test_loaded": "false",
        },
    ) as run:
        summary["evidence_run_id"] = run.info.run_id
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        control = summary["runs"][0]
        mlflow.log_metrics(
            {
                "point_loss/control_primary_rmse": float(
                    control["primary"]["rmse_log10_cn2"]
                ),
                "resources/neural_wall_clock_hours": float(
                    summary["resources"]["neural_wall_clock_hours"]
                ),
                "resources/peak_total_board_vram_gib": float(
                    summary["resources"]["peak_total_board_vram_gib"]
                ),
            }
        )
        for candidate_id, decision in summary["decisions"].items():
            mlflow.log_metric(
                f"point_loss/{candidate_id}_improvement",
                float(decision["relative_improvement_over_control"]),
            )
        for path, artifact_path in (
            (summary_path, "evidence"),
            (report_path, "reports"),
            (report_path.with_suffix(".tex"), "reports"),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "point-loss-freeze.json",
                "evidence",
            ),
            (
                root
                / "docs"
                / "FUSION_V21_POINT_LOSS.md",
                "protocol",
            ),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2 follow-up cycle evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v21-point-loss/"
            "fusion-v21-point-loss-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_point_loss_freeze(root)
    summary_path = write_point_loss_summary(root)
    if args.log_mlflow:
        print(log_point_loss_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
