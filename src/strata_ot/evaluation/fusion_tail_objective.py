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
from strata_ot.evaluation.fusion_cycles import _row
from strata_ot.evaluation.fusion_program import PRIMARY_HORIZONS, _load_object
from strata_ot.training.train import _configure_utf8_output

TAIL_CONTROL = "tail-huber-0p00"
TAIL_INTERVENTIONS = (
    "tail-huber-0p50",
    "tail-huber-1p00",
    "tail-huber-2p00",
)
TAIL_CANDIDATES = (TAIL_CONTROL, *TAIL_INTERVENTIONS)
TAIL_WEIGHTS = {
    "tail-huber-0p00": 0.0,
    "tail-huber-0p50": 0.5,
    "tail-huber-1p00": 1.0,
    "tail-huber-2p00": 2.0,
}
TAIL_PARAMETERS = 3_611_849


def _artifact_check(root: Path, run: dict[str, Any], candidate_id: str) -> None:
    missing = [
        str(relative)
        for relative in run.get("artifacts", {}).values()
        if not (root / str(relative)).is_file()
    ]
    checkpoint = run.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        missing.append(str(checkpoint))
    if missing:
        raise RuntimeError(
            f"Incomplete tail-objective artifacts for {candidate_id}: "
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
        raise RuntimeError(f"Non-finite tail-objective evidence: {candidate_id}")


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


def _collect_tail_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in TAIL_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Dirty tail-objective run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        weight = float(candidate.get("tail_huber_weight", -1.0))
        if weight != TAIL_WEIGHTS[candidate_id]:
            raise RuntimeError(f"Tail-weight drift for {candidate_id}: {weight}")
        if float(candidate.get("point_loss_weight", -1.0)) != 0.0:
            raise RuntimeError(f"Generic point-loss drift for {candidate_id}")
        if str(candidate.get("residual_shortcut")) != "history_linear":
            raise RuntimeError(f"Architecture drift for {candidate_id}")
        if int(run.get("parameters", -1)) != TAIL_PARAMETERS:
            raise RuntimeError(f"Parameter drift for {candidate_id}")
        if int(resolved.get("max_epochs_requested", -1)) != 9:
            raise RuntimeError(f"Epoch-cap drift for {candidate_id}")
        if int(resolved.get("training_examples_actual", -1)) != 12_000:
            raise RuntimeError(f"Training-count drift for {candidate_id}")
        if int(resolved.get("evaluation_examples_actual", -1)) != 4_000:
            raise RuntimeError(f"Selection-count drift for {candidate_id}")
        tail = resolved.get("training_tail_objective", {})
        if (
            tail.get("threshold_fit_role") != "train"
            or tail.get("threshold_fit_fold") != "fold-1"
            or float(tail.get("threshold_quantile", -1.0)) != 0.90
            or float(tail.get("weight", -1.0)) != weight
            or not math.isfinite(float(tail.get("threshold_log10_cn2", math.nan)))
            or not 0.095
            <= float(tail.get("training_fraction_at_or_above", -1.0))
            <= 0.105
        ):
            raise RuntimeError(f"Training-tail provenance drift for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(f"Tail-objective run stopped for budget: {candidate_id}")
        if int(resolved.get("epochs_completed", -1)) < 1:
            raise RuntimeError(f"Missing completed epochs for {candidate_id}")
        if int(resolved.get("optimizer_steps", -1)) < 1:
            raise RuntimeError(f"Missing optimizer steps for {candidate_id}")
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate tail-objective run: {candidate_id}")
        _metric_check(run, candidate_id)
        _component_check(run, candidate_id)
        _artifact_check(root, run, candidate_id)
        matches[candidate_id] = run
    missing = sorted(set(TAIL_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError(
            "Tail-objective cycle is incomplete: " + ", ".join(missing)
        )
    return matches


def _tail_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    resolved = run["resolved_configuration"]
    objective = resolved["training_tail_objective"]
    row["weight"] = float(objective["weight"])
    row["training_tail_threshold"] = float(objective["threshold_log10_cn2"])
    row["training_tail_fraction"] = float(
        objective["training_fraction_at_or_above"]
    )
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
    row["residual_diagnostics"] = diagnostics
    return row


def _tail_decision(
    control: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    control_rmse = float(control["primary"]["rmse_log10_cn2"])
    control_tail = float(control["primary"]["tail_mae_top_decile"])
    improvement = (
        control_rmse - float(intervention["primary"]["rmse_log10_cn2"])
    ) / control_rmse
    tail_improvement = (
        control_tail - float(intervention["primary"]["tail_mae_top_decile"])
    ) / control_tail
    checks = {
        "rmse_improvement_at_least_0_75pct": (
            "PASS" if improvement >= 0.0075 else "FAIL"
        ),
        "tail_improvement_at_least_0_5pct": (
            "PASS" if tail_improvement >= 0.005 else "FAIL"
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
        "relative_tail_improvement_over_control": tail_improvement,
        "checks": checks,
        "advances": all(value == "PASS" for value in checks.values()),
    }


def _select_tail_candidate(
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
    return min(
        advancing,
        key=lambda row: (
            float(row["primary"]["rmse_log10_cn2"]),
            float(row["weight"]),
        ),
    )


def build_tail_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_tail_runs(root)
    revisions = {str(run["repository"]["git_revision"]) for run in runs.values()}
    thresholds = {
        float(
            run["resolved_configuration"]["training_tail_objective"][
                "threshold_log10_cn2"
            ]
        )
        for run in runs.values()
    }
    if len(revisions) != 1 or len(thresholds) != 1:
        raise RuntimeError("Tail-objective runs do not share code/threshold identity")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v26-tail-objective",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V26_TAIL_OBJECTIVE.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in TAIL_CANDIDATES
        },
        "tail_huber_weights": TAIL_WEIGHTS,
        "training_tail_threshold_log10_cn2": next(iter(thresholds)),
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_tail_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "v26-tail-objective-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_tail_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_tail_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_tail_runs(root)
    rows = [_tail_row(manifests[candidate_id]) for candidate_id in TAIL_CANDIDATES]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _tail_decision(control, row)
        for row in rows[1:]
    }
    selected = _select_tail_candidate(rows[1:], decisions)
    revisions = {str(row["repository"]["git_revision"]) for row in rows}
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    thresholds = {float(row["training_tail_threshold"]) for row in rows}
    if len(revisions) != 1:
        raise RuntimeError("Tail-objective cycle has mixed repository revisions")
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Tail-objective cycle has mixed data/split identities")
    if len(thresholds) != 1:
        raise RuntimeError("Tail-objective cycle has mixed training thresholds")
    within_resources = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resources:
        raise RuntimeError("Tail-objective cycle exceeded its resource envelope")
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v26-tail-objective",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v26_tail_objective",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "A training-only upper-decile location penalty will reduce tail "
            "error while preserving the fixed v2.5 point-forecast advantage."
        ),
        "plain_language_question": (
            "Can the model pay extra attention to rare high-turbulence training "
            "targets without becoming less accurate overall?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed every frozen screen "
                "condition and is eligible for rolling robustness."
            )
            if selected is not None
            else (
                "No masked tail-loss weight passed both the point-error and "
                "tail-error requirements. The tail-objective family is closed."
            )
        ),
        "protocol": "docs/FUSION_V26_TAIL_OBJECTIVE.md",
        "freeze": (
            "research/experiments/fusion-v2-program/"
            "v26-tail-objective-freeze.json"
        ),
        "repository_revision": next(iter(revisions)),
        "data_identity": {
            "source_sha256": next(iter(source_hashes)),
            "split_sha256": next(iter(split_hashes)),
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "training_tail_threshold_log10_cn2": next(iter(thresholds)),
        "primary_horizon_minutes": list(PRIMARY_HORIZONS),
        "anchor_horizon_minutes": 5,
        "control": control,
        "runs": rows,
        "decisions": decisions,
        "selected_candidate": (
            str(selected["candidate_id"]) if selected is not None else None
        ),
        "checks": {
            "four_arms_completed": "PASS",
            "three_masked_weights_evaluated": "PASS",
            "repository_identity": "PASS",
            "data_and_split_identity": "PASS",
            "architecture_parameter_identity": "PASS",
            "training_only_threshold_identity": "PASS",
            "objective_records_logged": "PASS",
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
            "tail_objective_rolling_robustness"
            if selected is not None
            else "distinct_family"
        ),
        "report_pdf": None,
    }


def write_tail_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "v26-tail-objective-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_tail_summary(root)
    for key in ("evidence_run_id", "report_pdf"):
        if isinstance(existing.get(key), str):
            summary[key] = existing[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_tail_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v26-tail-objective-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-tail-objective",
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
            "tail_objective/control_primary_rmse",
            float(summary["control"]["primary"]["rmse_log10_cn2"]),
        )
        for candidate_id, decision in summary["decisions"].items():
            mlflow.log_metric(
                f"tail_objective/{candidate_id}_rmse_improvement",
                float(decision["relative_improvement_over_control"]),
            )
            mlflow.log_metric(
                f"tail_objective/{candidate_id}_tail_improvement",
                float(decision["relative_tail_improvement_over_control"]),
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
                / "v26-tail-objective-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "v26-tail-objective-report-inspection.json",
                "evidence",
            ),
            (root / "docs" / "FUSION_V26_TAIL_OBJECTIVE.md", "protocol"),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2.6 tail-objective evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v26-tail-objective/"
            "fusion-v26-tail-objective-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_tail_freeze(root)
    summary_path = write_tail_summary(root)
    if args.log_mlflow:
        print(log_tail_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
