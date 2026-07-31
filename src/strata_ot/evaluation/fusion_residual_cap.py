from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import mlflow

from strata_ot.config import find_repo_root
from strata_ot.evaluation.fusion_cycles import _row
from strata_ot.evaluation.fusion_program import PRIMARY_HORIZONS, _load_object
from strata_ot.evaluation.fusion_tail_objective import (
    _artifact_check,
    _component_check,
    _metric_check,
    _tail_decision,
)
from strata_ot.training.train import _configure_utf8_output

CAP_CONTROL = "residual-cap-unbounded"
CAP_INTERVENTIONS = (
    "residual-cap-0p20",
    "residual-cap-0p35",
    "residual-cap-0p60",
)
CAP_CANDIDATES = (CAP_CONTROL, *CAP_INTERVENTIONS)
CAP_VALUES = {
    "residual-cap-unbounded": 0.0,
    "residual-cap-0p20": 0.20,
    "residual-cap-0p35": 0.35,
    "residual-cap-0p60": 0.60,
}
CAP_PARAMETERS = 3_611_849


def _collect_cap_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in CAP_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Dirty residual-cap run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        cap = float(candidate.get("residual_cap", -1.0))
        if cap != CAP_VALUES[candidate_id]:
            raise RuntimeError(f"Residual-cap drift for {candidate_id}: {cap}")
        if (
            float(candidate.get("point_loss_weight", -1.0)) != 0.0
            or float(candidate.get("tail_huber_weight", -1.0)) != 0.0
        ):
            raise RuntimeError(f"Objective drift for {candidate_id}")
        if str(candidate.get("residual_shortcut")) != "history_linear":
            raise RuntimeError(f"Architecture drift for {candidate_id}")
        if float(candidate.get("residual_horizon_exponent", -1.0)) != 0.0:
            raise RuntimeError(f"Residual exponent drift for {candidate_id}")
        if int(run.get("parameters", -1)) != CAP_PARAMETERS:
            raise RuntimeError(f"Parameter drift for {candidate_id}")
        if int(resolved.get("max_epochs_requested", -1)) != 9:
            raise RuntimeError(f"Epoch-cap drift for {candidate_id}")
        if int(resolved.get("training_examples_actual", -1)) != 12_000:
            raise RuntimeError(f"Training-count drift for {candidate_id}")
        if int(resolved.get("evaluation_examples_actual", -1)) != 4_000:
            raise RuntimeError(f"Selection-count drift for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(f"Residual-cap run stopped for budget: {candidate_id}")
        if int(resolved.get("epochs_completed", -1)) < 1:
            raise RuntimeError(f"Missing completed epochs for {candidate_id}")
        if int(resolved.get("optimizer_steps", -1)) < 1:
            raise RuntimeError(f"Missing optimizer steps for {candidate_id}")
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate residual-cap run: {candidate_id}")
        _metric_check(run, candidate_id)
        _component_check(run, candidate_id)
        _artifact_check(root, run, candidate_id)
        matches[candidate_id] = run
    missing = sorted(set(CAP_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError(
            "Residual-cap cycle is incomplete: " + ", ".join(missing)
        )
    return matches


def _cap_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    resolved = run["resolved_configuration"]
    row["cap"] = float(resolved["candidate"]["residual_cap"])
    row["max_epochs_requested"] = int(resolved["max_epochs_requested"])
    row["epochs_completed"] = int(resolved["epochs_completed"])
    row["optimizer_steps"] = int(resolved["optimizer_steps"])
    diagnostics: dict[str, dict[str, float]] = {}
    for horizon in (5, *PRIMARY_HORIZONS):
        values = run["component_summary"][str(horizon)]
        diagnostics[str(horizon)] = {
            "raw_minimum": float(values["raw_residual"]["minimum"]),
            "raw_maximum": float(values["raw_residual"]["maximum"]),
            "final_minimum": float(values["residual"]["minimum"]),
            "final_maximum": float(values["residual"]["maximum"]),
            "final_mean": float(values["residual"]["mean"]),
            "final_std": float(values["residual"]["std"]),
        }
    row["cap_diagnostics"] = diagnostics
    return row


def _select_cap_candidate(
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
    return max(
        near_best,
        key=lambda row: (
            float(row["cap"]),
            -float(row["primary"]["rmse_log10_cn2"]),
        ),
    )


def build_cap_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_cap_runs(root)
    revisions = {str(run["repository"]["git_revision"]) for run in runs.values()}
    if len(revisions) != 1:
        raise RuntimeError("Residual-cap runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v27-residual-cap",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V27_RESIDUAL_CAP.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in CAP_CANDIDATES
        },
        "residual_caps": CAP_VALUES,
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_cap_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "v27-residual-cap-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_cap_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_cap_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_cap_runs(root)
    rows = [_cap_row(manifests[candidate_id]) for candidate_id in CAP_CANDIDATES]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _tail_decision(control, row)
        for row in rows[1:]
    }
    selected = _select_cap_candidate(rows[1:], decisions)
    revisions = {str(row["repository"]["git_revision"]) for row in rows}
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(revisions) != 1:
        raise RuntimeError("Residual-cap cycle has mixed repository revisions")
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Residual-cap cycle has mixed data/split identities")
    within_resources = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resources:
        raise RuntimeError("Residual-cap cycle exceeded its resource envelope")
    for row in rows[1:]:
        cap = float(row["cap"])
        for values in row["cap_diagnostics"].values():
            if (
                float(values["final_minimum"]) < -cap - 1e-5
                or float(values["final_maximum"]) > cap + 1e-5
            ):
                raise RuntimeError(
                    f"Residual bound not enforced for {row['candidate_id']}"
                )
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v27-residual-cap",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v27_residual_cap",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "A symmetric bound on the learned persistence correction will "
            "prevent rare residual outliers while retaining useful corrections."
        ),
        "plain_language_question": (
            "Can the model stay close enough to persistence to avoid rare bad "
            "corrections while keeping the improvements it usually learns?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed every frozen screen "
                "condition and is eligible for rolling robustness."
            )
            if selected is not None
            else (
                "No residual bound passed both point-error and tail-error "
                "requirements. The bounded-residual family is closed."
            )
        ),
        "protocol": "docs/FUSION_V27_RESIDUAL_CAP.md",
        "freeze": (
            "research/experiments/fusion-v2-program/"
            "v27-residual-cap-freeze.json"
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
        "control": control,
        "runs": rows,
        "decisions": decisions,
        "selected_candidate": (
            str(selected["candidate_id"]) if selected is not None else None
        ),
        "checks": {
            "four_arms_completed": "PASS",
            "three_positive_caps_evaluated": "PASS",
            "repository_identity": "PASS",
            "data_and_split_identity": "PASS",
            "architecture_parameter_identity": "PASS",
            "residual_caps_enforced": "PASS",
            "raw_and_final_residuals_logged": "PASS",
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
            "residual_cap_rolling_robustness"
            if selected is not None
            else "distinct_family"
        ),
        "report_pdf": None,
    }


def write_cap_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "v27-residual-cap-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_cap_summary(root)
    for key in ("evidence_run_id", "report_pdf"):
        if isinstance(existing.get(key), str):
            summary[key] = existing[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_cap_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v27-residual-cap-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-residual-cap",
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
        for candidate_id, decision in summary["decisions"].items():
            mlflow.log_metric(
                f"residual_cap/{candidate_id}_rmse_improvement",
                float(decision["relative_improvement_over_control"]),
            )
            mlflow.log_metric(
                f"residual_cap/{candidate_id}_tail_improvement",
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
                / "v27-residual-cap-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "v27-residual-cap-report-inspection.json",
                "evidence",
            ),
            (root / "docs" / "FUSION_V27_RESIDUAL_CAP.md", "protocol"),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2.7 residual-cap evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v27-residual-cap/"
            "fusion-v27-residual-cap-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_cap_freeze(root)
    summary_path = write_cap_summary(root)
    if args.log_mlflow:
        print(log_cap_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
