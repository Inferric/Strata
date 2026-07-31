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

CONVERGENCE_BASELINE_RUN_ID = "46af532e62ed4aaf9f50b9faab1e131d"
CONVERGENCE_CONTROL = "convergence-3"
CONVERGENCE_INTERVENTIONS = (
    "convergence-6",
    "convergence-9",
    "convergence-12",
)
CONVERGENCE_CANDIDATES = (CONVERGENCE_CONTROL, *CONVERGENCE_INTERVENTIONS)
CONVERGENCE_EPOCHS = {
    "convergence-3": 3,
    "convergence-6": 6,
    "convergence-9": 9,
    "convergence-12": 12,
}
CONVERGENCE_PARAMETERS = 3611849


def _artifact_check(root: Path, run: dict[str, Any], candidate_id: str) -> None:
    missing = [
        relative
        for relative in run.get("artifacts", {}).values()
        if not (root / str(relative)).is_file()
    ]
    checkpoint = run.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        missing.append(str(checkpoint))
    if missing:
        raise RuntimeError(
            f"Incomplete convergence artifacts for {candidate_id}: "
            + ", ".join(missing)
        )


def _metric_check(run: dict[str, Any], candidate_id: str) -> None:
    values = (
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
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite convergence evidence: {candidate_id}")


def _component_check(run: dict[str, Any], candidate_id: str) -> None:
    for horizon in (5, *PRIMARY_HORIZONS):
        components = run.get("component_summary", {}).get(str(horizon), {})
        for key in (
            "base_residual",
            "shortcut_residual",
            "raw_residual",
            "residual",
        ):
            if key not in components:
                raise RuntimeError(
                    f"Missing {key} for {candidate_id} at {horizon} minutes"
                )


def _collect_convergence_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in CONVERGENCE_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Dirty convergence run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        requested = int(resolved.get("max_epochs_requested", -1))
        expected = CONVERGENCE_EPOCHS[candidate_id]
        if requested != expected or int(candidate.get("screen_max_epochs", -1)) != expected:
            raise RuntimeError(
                f"Epoch-cap mismatch for {candidate_id}: {requested}"
            )
        if str(candidate.get("residual_shortcut")) != "history_linear":
            raise RuntimeError(f"Architecture drift for {candidate_id}")
        if int(run.get("parameters", -1)) != CONVERGENCE_PARAMETERS:
            raise RuntimeError(f"Parameter drift for {candidate_id}")
        if int(resolved.get("training_examples_actual", -1)) != 12000:
            raise RuntimeError(f"Training-count drift for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(f"Convergence run stopped for budget: {candidate_id}")
        if int(resolved.get("epochs_completed", -1)) < 1:
            raise RuntimeError(f"Missing completed epochs for {candidate_id}")
        if int(resolved.get("optimizer_steps", -1)) < 1:
            raise RuntimeError(f"Missing optimizer steps for {candidate_id}")
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate convergence run: {candidate_id}")
        _metric_check(run, candidate_id)
        _component_check(run, candidate_id)
        _artifact_check(root, run, candidate_id)
        matches[candidate_id] = run
    missing = sorted(set(CONVERGENCE_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError("Convergence cycle is incomplete: " + ", ".join(missing))
    return matches


def _load_baseline(root: Path) -> dict[str, Any]:
    path = (
        root
        / "artifacts"
        / "runs"
        / CONVERGENCE_BASELINE_RUN_ID
        / "run-manifest.json"
    )
    baseline = _load_object(path)
    if baseline.get("candidate_id") != "shortcut-none":
        raise RuntimeError("Frozen convergence baseline identity drifted")
    if baseline.get("repository", {}).get("git_dirty") is not False:
        raise RuntimeError("Frozen convergence baseline is dirty")
    _metric_check(baseline, "shortcut-none")
    _artifact_check(root, baseline, "shortcut-none")
    return baseline


def _convergence_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    row.pop("weight", None)
    resolved = run["resolved_configuration"]
    row["max_epochs_requested"] = int(resolved["max_epochs_requested"])
    row["epochs_completed"] = int(resolved["epochs_completed"])
    row["optimizer_steps"] = int(resolved["optimizer_steps"])
    diagnostics: dict[str, dict[str, float]] = {}
    for horizon in (5, *PRIMARY_HORIZONS):
        values = run["component_summary"][str(horizon)]
        diagnostics[str(horizon)] = {
            "base_mean": float(values["base_residual"]["mean"]),
            "shortcut_mean": float(values["shortcut_residual"]["mean"]),
            "shortcut_std": float(values["shortcut_residual"]["std"]),
            "final_mean": float(values["residual"]["mean"]),
        }
    row["convergence_diagnostics"] = diagnostics
    return row


def _convergence_decision(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    return _point_decision(baseline, intervention)


def _select_convergence_candidate(
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    advancing = [
        row
        for row in rows
        if decisions[str(row["candidate_id"])]["advances"]
    ]
    if not advancing:
        return None
    best_rmse = min(
        float(row["primary"]["rmse_log10_cn2"]) for row in advancing
    )
    near_best = [
        row
        for row in advancing
        if (
            float(row["primary"]["rmse_log10_cn2"]) - best_rmse
        )
        / best_rmse
        <= 0.0025
    ]
    return min(
        near_best,
        key=lambda row: (
            int(row["max_epochs_requested"]),
            float(row["primary"]["rmse_log10_cn2"]),
        ),
    )


def build_convergence_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_convergence_runs(root)
    revisions = {str(run["repository"]["git_revision"]) for run in runs.values()}
    if len(revisions) != 1:
        raise RuntimeError("Convergence runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v25-convergence",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V25_CONVERGENCE.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "baseline_run_id": CONVERGENCE_BASELINE_RUN_ID,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in CONVERGENCE_CANDIDATES
        },
        "epoch_caps": CONVERGENCE_EPOCHS,
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_convergence_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "convergence-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_convergence_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_convergence_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_convergence_runs(root)
    baseline = _row(_load_baseline(root))
    baseline.pop("weight", None)
    rows = [
        _convergence_row(manifests[candidate_id])
        for candidate_id in CONVERGENCE_CANDIDATES
    ]
    decisions = {
        str(row["candidate_id"]): _convergence_decision(baseline, row)
        for row in rows
    }
    selected = _select_convergence_candidate(rows[1:], decisions)
    revisions = {str(row["repository"]["git_revision"]) for row in rows}
    if len(revisions) != 1:
        raise RuntimeError("Convergence cycle has mixed repository revisions")
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Convergence cycle has mixed data/split identities")
    within_resources = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resources:
        raise RuntimeError("Convergence cycle exceeded its resource envelope")
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v25-convergence",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v25_convergence",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "Additional optimization of the fixed history-linear shortcut "
            "will retain its point gain while restoring tail and coverage."
        ),
        "plain_language_question": (
            "Does the partially promising fixed model simply need more "
            "training to become accurate without sacrificing its hard cases?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed the convergence screen "
                "and is eligible for rolling robustness."
            )
            if selected is not None
            else (
                "No longer training duration passed every frozen condition. "
                "The convergence-duration family is closed."
            )
        ),
        "protocol": "docs/FUSION_V25_CONVERGENCE.md",
        "freeze": (
            "research/experiments/fusion-v2-program/convergence-freeze.json"
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
        "baseline": baseline,
        "runs": rows,
        "decisions": decisions,
        "selected_candidate": (
            str(selected["candidate_id"]) if selected is not None else None
        ),
        "checks": {
            "four_arms_completed": "PASS",
            "three_longer_durations_evaluated": "PASS",
            "repository_identity": "PASS",
            "data_and_split_identity": "PASS",
            "architecture_parameter_identity": "PASS",
            "convergence_records_logged": "PASS",
            "residual_components_logged": "PASS",
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
            "rolling_robustness" if selected is not None else "distinct_family"
        ),
        "report_pdf": None,
    }


def write_convergence_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "convergence-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_convergence_summary(root)
    for key in ("evidence_run_id", "report_pdf"):
        if isinstance(existing.get(key), str):
            summary[key] = existing[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_convergence_evidence(summary_path: Path, report_path: Path) -> str:
    root = find_repo_root()
    summary = _load_object(summary_path)
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    existing = summary.get("evidence_run_id")
    if isinstance(existing, str) and existing:
        with mlflow.start_run(run_id=existing):
            mlflow.log_artifact(str(summary_path), artifact_path="evidence")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(
                str(report_path.with_suffix(".tex")),
                artifact_path="reports",
            )
        return existing
    mlflow.set_experiment("strata-ot-fusion-v2-program")
    with mlflow.start_run(
        run_name="fusion-v25-convergence-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-convergence",
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
        mlflow.log_metric(
            "convergence/baseline_primary_rmse",
            float(summary["baseline"]["primary"]["rmse_log10_cn2"]),
        )
        for candidate_id, decision in summary["decisions"].items():
            mlflow.log_metric(
                f"convergence/{candidate_id}_improvement",
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
                / "convergence-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "convergence-report-inspection.json",
                "evidence",
            ),
            (root / "docs" / "FUSION_V25_CONVERGENCE.md", "protocol"),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2 convergence evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v25-convergence/"
            "fusion-v25-convergence-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_convergence_freeze(root)
    summary_path = write_convergence_summary(root)
    if args.log_mlflow:
        print(log_convergence_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
