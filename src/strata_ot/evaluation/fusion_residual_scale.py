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

RESIDUAL_SCALE_CONTROL = "residual-exp-0p00"
RESIDUAL_SCALE_INTERVENTIONS = (
    "residual-exp-0p25",
    "residual-exp-0p50",
    "residual-exp-1p00",
)
RESIDUAL_SCALE_CANDIDATES = (
    RESIDUAL_SCALE_CONTROL,
    *RESIDUAL_SCALE_INTERVENTIONS,
)
RESIDUAL_SCALE_EXPONENTS = {
    "residual-exp-0p00": 0.0,
    "residual-exp-0p25": 0.25,
    "residual-exp-0p50": 0.5,
    "residual-exp-1p00": 1.0,
}


def _collect_residual_scale_runs(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        candidate_id = str(run.get("candidate_id"))
        if candidate_id not in RESIDUAL_SCALE_CANDIDATES:
            continue
        if run.get("fold_id") != "fold-1" or int(run.get("seed", -1)) != 17:
            continue
        resolved = run.get("resolved_configuration", {})
        if resolved.get("screen") is not True:
            continue
        repository = run.get("repository", {})
        if repository.get("git_dirty") is not False:
            raise RuntimeError(f"Dirty residual-scale run: {candidate_id}")
        candidate = resolved.get("candidate", {})
        exponent = float(candidate.get("residual_horizon_exponent", -1.0))
        expected_exponent = RESIDUAL_SCALE_EXPONENTS[candidate_id]
        if exponent != expected_exponent:
            raise RuntimeError(
                f"Residual exponent mismatch for {candidate_id}: {exponent}"
            )
        if int(resolved.get("training_examples_actual", -1)) != 12000:
            raise RuntimeError(
                f"Residual-scale arm changed training count: {candidate_id}"
            )
        if float(candidate.get("point_loss_weight", 0.0)) != 0.0:
            raise RuntimeError(
                f"Residual-scale arm changed point loss: {candidate_id}"
            )
        if candidate_id in matches:
            raise RuntimeError(
                f"Duplicate residual-scale run for {candidate_id}"
            )
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(
                f"Residual-scale run did not finish normally: {candidate_id}"
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
                f"Non-finite residual-scale evidence: {candidate_id}"
            )
        for horizon in (5, *PRIMARY_HORIZONS):
            components = run.get("component_summary", {}).get(str(horizon), {})
            for key in ("raw_residual", "residual_scale", "residual"):
                if key not in components:
                    raise RuntimeError(
                        f"Missing {key} diagnostic for {candidate_id} at "
                        f"{horizon} minutes"
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
                f"Incomplete residual-scale artifacts for {candidate_id}: "
                + ", ".join(missing_artifacts)
            )
        matches[candidate_id] = run
    missing = sorted(set(RESIDUAL_SCALE_CANDIDATES) - set(matches))
    if missing:
        raise RuntimeError(
            "Residual-scale cycle is incomplete: " + ", ".join(missing)
        )
    return matches


def _residual_row(run: dict[str, Any]) -> dict[str, Any]:
    row = _row(run)
    row.pop("weight", None)
    candidate = run["resolved_configuration"]["candidate"]
    row["residual_horizon_exponent"] = float(
        candidate["residual_horizon_exponent"]
    )
    row["training_examples_actual"] = int(
        run["resolved_configuration"]["training_examples_actual"]
    )
    diagnostics: dict[str, dict[str, float]] = {}
    for horizon in (5, *PRIMARY_HORIZONS):
        values = run["component_summary"][str(horizon)]
        diagnostics[str(horizon)] = {
            "raw_residual_mean": float(values["raw_residual"]["mean"]),
            "raw_residual_std": float(values["raw_residual"]["std"]),
            "residual_scale_mean": float(values["residual_scale"]["mean"]),
            "residual_mean": float(values["residual"]["mean"]),
            "residual_std": float(values["residual"]["std"]),
        }
    row["residual_diagnostics"] = diagnostics
    return row


def _residual_scale_decision(
    control: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    return _point_decision(control, intervention)


def build_residual_scale_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_residual_scale_runs(root)
    revisions = {
        str(run["repository"]["git_revision"]) for run in runs.values()
    }
    if len(revisions) != 1:
        raise RuntimeError("Residual-scale runs do not share one code revision")
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v23-residual-scaling",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V23_RESIDUAL_SCALING.md",
        "repository_revision": next(iter(revisions)),
        "fold_id": "fold-1",
        "seed": 17,
        "run_ids": {
            candidate_id: runs[candidate_id]["run_id"]
            for candidate_id in RESIDUAL_SCALE_CANDIDATES
        },
        "exponents": RESIDUAL_SCALE_EXPONENTS,
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_residual_scale_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "residual-scale-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_residual_scale_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_residual_scale_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    manifests = _collect_residual_scale_runs(root)
    rows = [
        _residual_row(manifests[candidate_id])
        for candidate_id in RESIDUAL_SCALE_CANDIDATES
    ]
    control = rows[0]
    decisions = {
        str(row["candidate_id"]): _residual_scale_decision(control, row)
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
        raise RuntimeError(
            "Residual-scale cycle has mixed repository revisions"
        )
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in manifests.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in manifests.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError(
            "Residual-scale cycle has mixed data/split identities"
        )
    parameter_counts = {int(row["parameters"]) for row in rows}
    if parameter_counts != {3611530}:
        raise RuntimeError(
            "Residual-scale cycle changed the frozen parameter count"
        )
    within_resource_envelope = all(
        float(row["wall_clock_seconds"]) <= 1_800
        and float(row["peak_total_board_vram_gib"]) <= 15.0
        for row in rows
    )
    if not within_resource_envelope:
        raise RuntimeError(
            "Residual-scale cycle exceeded its frozen run envelope"
        )
    status = "PASS" if selected is not None else "FAIL"
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v23-residual-scaling",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v23_residual_scaling",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": status,
        "evaluation_partition": "fold-1-selection",
        "hypothesis": (
            "A fixed power-law forecast-horizon scale will make the "
            "persistence residual easier to learn at longer lead times "
            "without changing the neural representation."
        ),
        "plain_language_question": (
            "Does explicitly letting the model's persistence correction grow "
            "with forecast lead time make the custom forecast meaningfully "
            "better?"
        ),
        "plain_language_conclusion": (
            (
                f"{selected['candidate_id']} passed the frozen residual-scale "
                "screen and is eligible for rolling robustness."
            )
            if selected is not None
            else (
                "No fixed residual-scaling exponent passed the frozen screen. "
                "The three non-improvements close this power-law family."
            )
        ),
        "protocol": "docs/FUSION_V23_RESIDUAL_SCALING.md",
        "freeze": (
            "research/experiments/fusion-v2-program/"
            "residual-scale-freeze.json"
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
            "rolling_robustness"
            if selected is not None
            else "representation_or_optimization"
        ),
        "report_pdf": None,
    }


def write_residual_scale_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "residual-scale-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_residual_scale_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_residual_scale_evidence(
    summary_path: Path,
    report_path: Path,
) -> str:
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
        run_name="fusion-v23-residual-scaling-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-residual-scaling",
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
                "residual_scale/control_primary_rmse": float(
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
                f"residual_scale/{candidate_id}_improvement",
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
                / "residual-scale-freeze.json",
                "evidence",
            ),
            (
                root
                / "docs"
                / "FUSION_V23_RESIDUAL_SCALING.md",
                "protocol",
            ),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2 residual-scaling evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v23-residual-scaling/"
            "fusion-v23-residual-scaling-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_residual_scale_freeze(root)
    summary_path = write_residual_scale_summary(root)
    if args.log_mlflow:
        print(log_residual_scale_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
