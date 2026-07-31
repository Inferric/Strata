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

SHORTCUT_CONTROL = "shortcut-none"
SHORTCUT_INTERVENTIONS = (
    "shortcut-history-linear",
    "shortcut-all-linear",
    "shortcut-all-mlp",
)
SHORTCUT_CANDIDATES = (SHORTCUT_CONTROL, *SHORTCUT_INTERVENTIONS)
SHORTCUT_MODES = {
    "shortcut-none": "none",
    "shortcut-history-linear": "history_linear",
    "shortcut-all-linear": "all_linear",
    "shortcut-all-mlp": "all_mlp",
}
SHORTCUT_PARAMETER_COUNTS = {
    "shortcut-none": 3611530,
    "shortcut-history-linear": 3611849,
    "shortcut-all-linear": 3612941,
    "shortcut-all-mlp": 3885455,
}


def _collect_shortcut_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in SHORTCUT_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Dirty shortcut run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        mode = str(candidate.get("residual_shortcut", "none"))
        if mode != SHORTCUT_MODES[candidate_id]:
            raise RuntimeError(f"Shortcut mode mismatch for {candidate_id}: {mode}")
        if float(candidate.get("residual_horizon_exponent", -1.0)) != 0.0:
            raise RuntimeError(f"Shortcut arm changed residual scale: {candidate_id}")
        if float(candidate.get("point_loss_weight", -1.0)) != 0.0:
            raise RuntimeError(f"Shortcut arm changed point loss: {candidate_id}")
        if int(resolved.get("training_examples_actual", -1)) != 12000:
            raise RuntimeError(f"Shortcut arm changed training count: {candidate_id}")
        if int(run.get("parameters", -1)) != SHORTCUT_PARAMETER_COUNTS[candidate_id]:
            raise RuntimeError(
                f"Parameter mismatch for {candidate_id}: {run.get('parameters')}"
            )
        if candidate_id in matches:
            raise RuntimeError(f"Duplicate shortcut run for {candidate_id}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(f"Shortcut run did not finish normally: {candidate_id}")
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
            raise RuntimeError(f"Non-finite shortcut evidence: {candidate_id}")
        for horizon in (5, *PRIMARY_HORIZONS):
            components = run.get("component_summary", {}).get(str(horizon), {})
            for key in (
                "base_residual",
                "shortcut_residual",
                "raw_residual",
                "residual_scale",
                "residual",
            ):
                if key not in components:
                    raise RuntimeError(
                        f"Missing {key} for {candidate_id} at {horizon} minutes"
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
                f"Incomplete shortcut artifacts for {candidate_id}: "
                + ", ".join(missing_artifacts)
            )
        matches[candidate_id] = run
    missing = sorted(set(SHORTCUT_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError("Shortcut cycle is incomplete: " + ", ".join(missing))
    return matches


def _shortcut_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    row.pop("weight", None)
    candidate = run["resolved_configuration"]["candidate"]
    row["residual_shortcut"] = str(candidate["residual_shortcut"])
    row["training_examples_actual"] = int(
        run["resolved_configuration"]["training_examples_actual"]
    )
    diagnostics: dict[str, dict[str, float]] = {}
    for horizon in (5, *PRIMARY_HORIZONS):
        values = run["component_summary"][str(horizon)]
        diagnostics[str(horizon)] = {
            "base_mean": float(values["base_residual"]["mean"]),
            "base_std": float(values["base_residual"]["std"]),
            "shortcut_mean": float(values["shortcut_residual"]["mean"]),
            "shortcut_std": float(values["shortcut_residual"]["std"]),
            "raw_mean": float(values["raw_residual"]["mean"]),
            "final_mean": float(values["residual"]["mean"]),
        }
    row["shortcut_diagnostics"] = diagnostics
    return row


def _shortcut_decision(
    control: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    return _point_decision(control, intervention)


def build_shortcut_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_shortcut_runs(root)
    revisions = {str(run["repository"]["git_revision"]) for run in runs.values()}
    if len(revisions) != 1:
        raise RuntimeError("Shortcut runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v24-direct-shortcut",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V24_DIRECT_SHORTCUT.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in SHORTCUT_CANDIDATES
        },
        "shortcut_modes": SHORTCUT_MODES,
        "parameter_counts": SHORTCUT_PARAMETER_COUNTS,
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_shortcut_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "shortcut-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_shortcut_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_shortcut_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_shortcut_runs(root)
    rows = [
        _shortcut_row(manifests[candidate_id])
        for candidate_id in SHORTCUT_CANDIDATES
    ]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _shortcut_decision(control, row)
        for row in rows[1:]
    }
    advancing = [
        row
        for row in rows[1:]
        if decisions[str(row["candidate_id"])]["advances"]
    ]
    selected = (
        min(advancing, key=lambda row: float(row["primary"]["rmse_log10_cn2"]))
        if advancing
        else None
    )
    revisions = {str(row["repository"]["git_revision"]) for row in rows}
    if len(revisions) != 1:
        raise RuntimeError("Shortcut cycle has mixed repository revisions")
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Shortcut cycle has mixed data/split identities")
    if {int(row["parameters"]) for row in rows} != set(
        SHORTCUT_PARAMETER_COUNTS.values()
    ):
        raise RuntimeError("Shortcut cycle parameter identities drifted")
    within_resource_envelope = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resource_envelope:
        raise RuntimeError("Shortcut cycle exceeded its frozen run envelope")
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v24-direct-shortcut",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v24_direct_shortcut",
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "A zero-initialized direct residual shortcut from safe past "
            "multiscale context will recover information compressed by the "
            "deep attention and routing path."
        ),
        "plain_language_question": (
            "Can a small direct path preserve useful recent history and "
            "weather information that the deeper custom model loses?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed the frozen shortcut "
                "screen and is eligible for rolling robustness."
            )
            if selected is not None
            else (
                "No direct shortcut passed the frozen screen. The three "
                "non-improvements close this shortcut family."
            )
        ),
        "protocol": "docs/FUSION_V24_DIRECT_SHORTCUT.md",
        "freeze": (
            "research/experiments/fusion-v2-program/shortcut-freeze.json"
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
            "shortcut_components_logged": "PASS",
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


def write_shortcut_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "shortcut-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_shortcut_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_shortcut_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v24-direct-shortcut-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-direct-shortcut",
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
        mlflow.log_metrics(
            {
                "shortcut/control_primary_rmse": float(
                    summary["runs"][0]["primary"]["rmse_log10_cn2"]
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
                f"shortcut/{candidate_id}_improvement",
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
                / "shortcut-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "shortcut-report-inspection.json",
                "evidence",
            ),
            (
                root / "docs" / "FUSION_V24_DIRECT_SHORTCUT.md",
                "protocol",
            ),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2 direct-shortcut evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v24-direct-shortcut/"
            "fusion-v24-direct-shortcut-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_shortcut_freeze(root)
    summary_path = write_shortcut_summary(root)
    if args.log_mlflow:
        print(log_shortcut_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
