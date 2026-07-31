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
from strata_ot.evaluation.fusion_cycles import _point_decision, _row
from strata_ot.evaluation.fusion_program import PRIMARY_HORIZONS, _load_object
from strata_ot.training.train import _configure_utf8_output

DATA_SCALE_CONTROL = "data-scale-12k"
DATA_SCALE_INTERVENTIONS = (
    "data-scale-30k",
    "data-scale-50k",
    "data-scale-full",
)
DATA_SCALE_CANDIDATES = (DATA_SCALE_CONTROL, *DATA_SCALE_INTERVENTIONS)
DATA_SCALE_CAPS = {
    "data-scale-12k": 12000,
    "data-scale-30k": 30000,
    "data-scale-50k": 50000,
    "data-scale-full": 62263,
}


def _collect_data_scale_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in DATA_SCALE_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        repository = run.get("repository", {})
        if repository.get("git_dirty") is not False:
            raise RuntimeError(f"Dirty data-scale run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        configured_cap = int(candidate.get("max_train_examples", -1))
        expected_cap = DATA_SCALE_CAPS[candidate_id]
        if configured_cap != expected_cap:
            raise RuntimeError(
                f"Training cap mismatch for {candidate_id}: {configured_cap}"
            )
        actual_examples = int(resolved.get("training_examples_actual", -1))
        if actual_examples != expected_cap:
            raise RuntimeError(
                f"Actual training count mismatch for {candidate_id}: "
                f"{actual_examples}"
            )
        if float(candidate.get("point_loss_weight", 0.0)) != 0.0:
            raise RuntimeError(
                f"Data-scale run changed point loss: {candidate_id}"
            )
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate data-scale run for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(
                f"Data-scale run did not finish normally: {candidate_id}"
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
                f"Non-finite data-scale evidence: {candidate_id}"
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
                f"Incomplete data-scale artifacts for {candidate_id}: "
                + ", ".join(missing_artifacts)
            )
        matches[candidate_id] = run
    missing = sorted(set(DATA_SCALE_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError(
            "Data-scale cycle is incomplete: " + ", ".join(missing)
        )
    return matches


def _data_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    row.pop("weight", None)
    resolved = run["resolved_configuration"]
    row["training_example_cap"] = int(resolved["training_example_cap"])
    row["training_examples_actual"] = int(
        resolved["training_examples_actual"]
    )
    row["calibration_examples_actual"] = int(
        resolved["calibration_examples_actual"]
    )
    row["evaluation_examples_actual"] = int(
        resolved["evaluation_examples_actual"]
    )
    return row


def _data_scale_decision(
    control: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    return _point_decision(control, intervention)


def build_data_scale_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_data_scale_runs(root)
    revisions = {
        str(run["repository"]["git_revision"]) for run in runs.values()
    }
    if len(revisions) != 1:
        raise RuntimeError("Data-scale runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v22-data-scale",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V22_DATA_SCALE.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in DATA_SCALE_CANDIDATES
        },
        "training_example_caps": DATA_SCALE_CAPS,
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_data_scale_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "data-scale-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_data_scale_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_data_scale_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_data_scale_runs(root)
    rows = [
        _data_row(manifests[candidate_id])
        for candidate_id in DATA_SCALE_CANDIDATES
    ]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _data_scale_decision(control, row)
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
        raise RuntimeError("Data-scale cycle has mixed repository revisions")
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Data-scale cycle has mixed data/split identities")
    parameter_counts = {int(row["parameters"]) for row in rows}
    if len(parameter_counts) != 1:
        raise RuntimeError(
            "Data-scale cycle changed model size across sample caps"
        )
    within_resource_envelope = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resource_envelope:
        raise RuntimeError("Data-scale cycle exceeded its frozen run envelope")
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v22-data-scale",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v22_data_scale",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "Denser use of the same chronological training fold will improve "
            "the unchanged Fusion v2 model beyond the sparse 12,000-sequence "
            "screen without weakening tails or uncertainty."
        ),
        "plain_language_question": (
            "Does showing the unchanged custom model more of the available "
            "training sequences make it meaningfully better?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed the frozen density screen "
                "and is eligible for rolling robustness characterization."
            )
            if selected is not None
            else (
                "No larger training-sequence cap passed the frozen screen. "
                "The three non-improvements close this within-fold density "
                "family."
            )
        ),
        "protocol": "docs/FUSION_V22_DATA_SCALE.md",
        "freeze": (
            "research/experiments/fusion-v2-program/data-scale-freeze.json"
        ),
        "repository_revision": next(iter(revisions)),
        "data_identity": {
            "source_sha256": next(iter(source_hashes)),
            "split_sha256": next(iter(split_hashes)),
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "sample_selection": {
            "method": "deterministic evenly spaced indices",
            "same_fold_dates_for_all_arms": True,
            "full_valid_fold_1_sequences": 62263,
            "normalization_fit_role": "complete fold-1 training role",
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
            "actual_training_counts_match_caps": "PASS",
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
            else "representation_or_training_target"
        ),
        "report_pdf": None,
    }


def write_data_scale_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "data-scale-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_data_scale_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_data_scale_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v22-data-scale-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-data-scale",
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
                "data_scale/control_primary_rmse": float(
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
                f"data_scale/{candidate_id}_improvement",
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
                / "data-scale-freeze.json",
                "evidence",
            ),
            (
                root
                / "docs"
                / "FUSION_V22_DATA_SCALE.md",
                "protocol",
            ),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2 training-density evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v22-data-scale/"
            "fusion-v22-data-scale-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_data_scale_freeze(root)
    summary_path = write_data_scale_summary(root)
    if args.log_mlflow:
        print(log_data_scale_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
