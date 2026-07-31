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

from strata_ot.config import find_repo_root, load_yaml
from strata_ot.evaluation.fusion_program import (
    PRIMARY_HORIZONS,
    ROBUSTNESS_FIXED_CONTROLS,
    ROBUSTNESS_FOLDS,
    ROBUSTNESS_SEEDS,
    _aggregate_candidate,
    _load_object,
    _paired_bootstrap,
    _sha256_file,
)
from strata_ot.training.train import _configure_utf8_output

SELECTED_CANDIDATE = "selected-fusion-v25"
PREDECESSOR_CANDIDATE = "selected-fusion-v2"
REUSED_NEURAL_CONTROLS = (
    PREDECESSOR_CANDIDATE,
    "control-mlp",
    "control-tcn",
    "control-horizon-v1",
)
REUSED_CONTROL_IDS = (*ROBUSTNESS_FIXED_CONTROLS, *REUSED_NEURAL_CONTROLS)
EXPECTED_PARAMETERS = 3611849
EXPECTED_MAX_EPOCHS = 9
CONTROL_FREEZE = (
    "research/experiments/fusion-v2-program/robustness-freeze.json"
)


def _expected_selected_keys() -> set[tuple[str, str, int]]:
    return {
        (SELECTED_CANDIDATE, fold_id, seed)
        for fold_id in ROBUSTNESS_FOLDS
        for seed in ROBUSTNESS_SEEDS
    }


def _seed_value(run: dict[str, Any]) -> int:
    value = run.get("seed")
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _control_keys() -> list[tuple[str, str, int]]:
    keys = [
        (candidate_id, fold_id, 17)
        for fold_id in ROBUSTNESS_FOLDS
        for candidate_id in ROBUSTNESS_FIXED_CONTROLS
    ]
    keys.extend(
        (candidate_id, fold_id, seed)
        for candidate_id in REUSED_NEURAL_CONTROLS
        for fold_id in ROBUSTNESS_FOLDS
        for seed in ROBUSTNESS_SEEDS
    )
    return keys


def _artifact_check(root: Path, run: dict[str, Any], identity: str) -> None:
    missing = [
        str(relative)
        for relative in run.get("artifacts", {}).values()
        if not (root / str(relative)).is_file()
    ]
    if run.get("kind") == "neural":
        checkpoint = run.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            missing.append(str(checkpoint))
    if missing:
        raise RuntimeError(
            f"Incomplete v2.5 robustness artifacts for {identity}: "
            + ", ".join(missing)
        )


def _metric_check(run: dict[str, Any], identity: str) -> None:
    values = (
        float(run["by_horizon"][str(horizon)][metric])
        for horizon in (5, *PRIMARY_HORIZONS)
        for metric in (
            "rmse_log10_cn2",
            "mae_log10_cn2",
            "bias_log10_cn2",
            "tail_mae_top_decile",
            "crps_gaussian",
            "student_t_nll",
            "interval_80_coverage",
            "interval_80_width",
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite v2.5 robustness evidence: {identity}")


def _load_reused_controls(
    root: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    freeze_path = root / CONTROL_FREEZE
    freeze = _load_object(freeze_path)
    controls: dict[tuple[str, str, int], dict[str, Any]] = {}
    for candidate_id, fold_id, seed in _control_keys():
        identity = f"{candidate_id}/{fold_id}/seed-{seed}"
        run_id = str(freeze["run_ids"][identity])
        run = _load_object(
            root / "artifacts" / "runs" / run_id / "run-manifest.json"
        )
        if (
            run.get("run_id") != run_id
            or run.get("candidate_id") != candidate_id
            or run.get("fold_id") != fold_id
            or int(run.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"Reused control identity drift: {identity}")
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Reused control provenance is dirty: {identity}")
        _metric_check(run, identity)
        _artifact_check(root, run, identity)
        controls[(candidate_id, fold_id, seed)] = run
    if len(controls) != 45:
        raise RuntimeError("Expected exactly 45 frozen reused control runs")
    return controls


def _component_check(run: dict[str, Any], identity: str) -> None:
    for horizon in (5, *PRIMARY_HORIZONS):
        components = run.get("component_summary", {}).get(str(horizon), {})
        for key in (
            "residual",
            "base_residual",
            "shortcut_residual",
            "raw_residual",
            "scale_weights",
            "expert_weights",
            "physics_token_norm",
        ):
            if key not in components:
                raise RuntimeError(
                    f"Missing {key} for {identity} at {horizon} minutes"
                )


def _collect_selected_runs(
    root: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    expected = _expected_selected_keys()
    matches: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        key = (
            str(run.get("candidate_id")),
            str(run.get("fold_id")),
            _seed_value(run),
        )
        if key not in expected:
            continue
        resolved = run.get("resolved_configuration", {})
        candidate = resolved.get("candidate", {})
        identity = f"{key[0]}/{key[1]}/seed-{key[2]}"
        if run.get("repository", {}).get("git_dirty") is not False:
            raise RuntimeError(f"Dirty selected v2.5 run: {identity}")
        if resolved.get("screen") is not False:
            raise RuntimeError(f"Selected v2.5 run is not full-data: {identity}")
        if str(candidate.get("residual_shortcut")) != "history_linear":
            raise RuntimeError(f"History shortcut identity drift: {identity}")
        if int(candidate.get("full_max_epochs", -1)) != EXPECTED_MAX_EPOCHS:
            raise RuntimeError(f"Candidate epoch identity drift: {identity}")
        if int(resolved.get("max_epochs_requested", -1)) != EXPECTED_MAX_EPOCHS:
            raise RuntimeError(f"Resolved epoch identity drift: {identity}")
        if int(resolved.get("epochs_completed", -1)) < 1:
            raise RuntimeError(f"Missing completed epochs: {identity}")
        if int(resolved.get("optimizer_steps", -1)) < 1:
            raise RuntimeError(f"Missing optimizer steps: {identity}")
        if int(resolved.get("training_examples_actual", -1)) != 50000:
            raise RuntimeError(f"Training-count drift: {identity}")
        if int(resolved.get("evaluation_examples_actual", -1)) != 12000:
            raise RuntimeError(f"Selection-count drift: {identity}")
        if int(run.get("parameters", -1)) != EXPECTED_PARAMETERS:
            raise RuntimeError(f"Parameter identity drift: {identity}")
        if run.get("stopped_for_budget") is not False:
            raise RuntimeError(f"Run stopped for budget: {identity}")
        _metric_check(run, identity)
        _component_check(run, identity)
        _artifact_check(root, run, identity)
        if key in matches:
            raise RuntimeError(f"Duplicate selected v2.5 run: {identity}")
        matches[key] = run
    missing = sorted(expected - set(matches))
    if missing:
        raise RuntimeError(
            "V2.5 rolling matrix is incomplete: "
            + ", ".join(
                f"{candidate}/{fold}/seed-{seed}"
                for candidate, fold, seed in missing
            )
        )
    revisions = {
        str(run["repository"]["git_revision"]) for run in matches.values()
    }
    if len(revisions) != 1:
        raise RuntimeError("V2.5 rolling runs have mixed repository revisions")
    return matches


def _excluded_dirty_selected_runs(root: Path) -> list[dict[str, Any]]:
    expected = _expected_selected_keys()
    excluded: list[dict[str, Any]] = []
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        key = (
            str(run.get("candidate_id")),
            str(run.get("fold_id")),
            _seed_value(run),
        )
        if key not in expected:
            continue
        repository = run.get("repository", {})
        if repository.get("git_dirty") is False:
            continue
        excluded.append(
            {
                "run_id": run.get("run_id"),
                "candidate_id": key[0],
                "fold_id": key[1],
                "seed": key[2],
                "reason": "repository_worktree_dirty",
                "working_tree_status": repository.get(
                    "working_tree_status",
                    "",
                ),
                "wall_clock_seconds": float(
                    run.get("metrics", {}).get("wall_clock_seconds", 0.0)
                ),
                "artifact_storage_bytes": int(
                    run.get("metrics", {}).get(
                        "artifact_storage_bytes",
                        0,
                    )
                ),
            }
        )
    return excluded


def _combined_runs(
    root: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    controls = _load_reused_controls(root)
    selected = _collect_selected_runs(root)
    combined = {**controls, **selected}
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in combined.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in combined.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Reused controls and v2.5 runs do not share data identities")
    return combined


def build_v25_robustness_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    selected = _collect_selected_runs(root)
    controls = _load_reused_controls(root)
    revision = next(
        iter(
            {
                str(run["repository"]["git_revision"])
                for run in selected.values()
            }
        )
    )
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v25-rolling-robustness",
        "frozen_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "protocol": "docs/FUSION_V25_ROBUSTNESS.md",
        "selected_repository_revision": revision,
        "selected_run_ids": {
            f"{candidate}/{fold}/seed-{seed}": run["run_id"]
            for (candidate, fold, seed), run in selected.items()
        },
        "reused_control_freeze": CONTROL_FREEZE,
        "reused_control_freeze_sha256": _sha256_file(root / CONTROL_FREEZE),
        "reused_control_run_ids": {
            f"{candidate}/{fold}/seed-{seed}": run["run_id"]
            for (candidate, fold, seed), run in controls.items()
        },
        "bootstrap": {
            "block_minutes": 1440,
            "resamples": 2000,
            "seed": 20260730,
        },
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_v25_robustness_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "v25-robustness-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_v25_robustness_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_v25_robustness_summary(
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _combined_runs(root)
    candidate_ids = (
        *ROBUSTNESS_FIXED_CONTROLS,
        "control-mlp",
        "control-tcn",
        "control-horizon-v1",
        PREDECESSOR_CANDIDATE,
        SELECTED_CANDIDATE,
    )
    aggregates = {
        candidate_id: _aggregate_candidate(candidate_id, runs)
        for candidate_id in candidate_ids
    }
    selected = aggregates[SELECTED_CANDIDATE]
    predecessor = aggregates[PREDECESSOR_CANDIDATE]
    persistence = aggregates["control-persistence"]
    lightgbm = aggregates["diagnostic-lightgbm"]
    stronger_neural_id = min(
        ("control-mlp", "control-tcn"),
        key=lambda candidate_id: aggregates[candidate_id]["primary"][
            "rmse_log10_cn2"
        ],
    )
    stronger_neural = aggregates[stronger_neural_id]
    selected_rmse = float(selected["primary"]["rmse_log10_cn2"])
    persistence_rmse = float(persistence["primary"]["rmse_log10_cn2"])
    neural_rmse = float(stronger_neural["primary"]["rmse_log10_cn2"])
    lightgbm_rmse = float(lightgbm["primary"]["rmse_log10_cn2"])
    predecessor_rmse = float(predecessor["primary"]["rmse_log10_cn2"])
    improvement_persistence = (
        persistence_rmse - selected_rmse
    ) / persistence_rmse
    improvement_neural = (neural_rmse - selected_rmse) / neural_rmse
    improvement_lightgbm = (
        lightgbm_rmse - selected_rmse
    ) / lightgbm_rmse
    improvement_predecessor = (
        predecessor_rmse - selected_rmse
    ) / predecessor_rmse
    bootstrap_persistence = _paired_bootstrap(
        root,
        runs,
        candidate_id=SELECTED_CANDIDATE,
        benchmark_id="control-persistence",
    )
    bootstrap_neural = _paired_bootstrap(
        root,
        runs,
        candidate_id=SELECTED_CANDIDATE,
        benchmark_id=stronger_neural_id,
    )
    seed_directions = {
        str(seed): (
            persistence_rmse
            - float(selected["seed_primary_rmse"][str(seed)])
        )
        / persistence_rmse
        for seed in ROBUSTNESS_SEEDS
    }
    coverage = float(selected["primary"]["interval_80_coverage"])
    tail_limit = min(
        float(persistence["primary"]["tail_mae_top_decile"]),
        float(stronger_neural["primary"]["tail_mae_top_decile"]),
    ) * 1.02
    crps_limit = min(
        float(persistence["primary"]["crps_gaussian"]),
        float(stronger_neural["primary"]["crps_gaussian"]),
    ) * 1.02
    checks = {
        "nine_fresh_candidate_runs_complete": "PASS",
        "forty_five_control_runs_reused_by_exact_id": "PASS",
        "paired_endpoint_identity": "PASS",
        "candidate_repository_identity": "PASS",
        "data_and_split_identity": "PASS",
        "architecture_parameter_identity": "PASS",
        "epoch_and_optimizer_records": "PASS",
        "component_diagnostics": "PASS",
        "finite_metrics": "PASS",
        "artifacts_complete": "PASS",
        "official_mlo_test_sealed": "PASS",
        "usna_confirmation_labels_sealed": "PASS",
        "two_percent_over_persistence": (
            "PASS" if improvement_persistence >= 0.02 else "FAIL"
        ),
        "two_percent_over_stronger_neural": (
            "PASS" if improvement_neural >= 0.02 else "FAIL"
        ),
        "bootstrap_vs_persistence_positive": (
            "PASS"
            if bootstrap_persistence["relative_improvement_ci95"][0] > 0
            else "FAIL"
        ),
        "bootstrap_vs_stronger_neural_positive": (
            "PASS"
            if bootstrap_neural["relative_improvement_ci95"][0] > 0
            else "FAIL"
        ),
        "same_direction_all_three_seeds": (
            "PASS"
            if all(value > 0 for value in seed_directions.values())
            else "FAIL"
        ),
        "coverage_70_to_90_percent": (
            "PASS" if 0.70 <= coverage <= 0.90 else "FAIL"
        ),
        "absolute_bias_at_most_0_25": (
            "PASS"
            if float(selected["primary"]["absolute_bias"]) <= 0.25
            else "FAIL"
        ),
        "tail_mae_no_material_regression": (
            "PASS"
            if float(selected["primary"]["tail_mae_top_decile"]) <= tail_limit
            else "FAIL"
        ),
        "crps_no_material_regression": (
            "PASS"
            if float(selected["primary"]["crps_gaussian"]) <= crps_limit
            else "FAIL"
        ),
        "relative_seed_fold_std_at_most_0_15": (
            "PASS"
            if float(selected["relative_run_rmse_std"]) <= 0.15
            else "FAIL"
        ),
    }
    final_gate_names = (
        "two_percent_over_persistence",
        "two_percent_over_stronger_neural",
        "bootstrap_vs_persistence_positive",
        "bootstrap_vs_stronger_neural_positive",
        "same_direction_all_three_seeds",
        "coverage_70_to_90_percent",
        "absolute_bias_at_most_0_25",
        "tail_mae_no_material_regression",
        "crps_no_material_regression",
        "relative_seed_fold_std_at_most_0_15",
        "candidate_repository_identity",
        "data_and_split_identity",
        "paired_endpoint_identity",
    )
    passed = all(checks[name] == "PASS" for name in final_gate_names)
    checks["confirmation_eligibility"] = "PASS" if passed else "FAIL"
    checks["confirmation_evaluation"] = "NOT_EVALUATED"
    failed_conditions = [
        name for name in final_gate_names if checks[name] == "FAIL"
    ]
    selected_runs = [
        run
        for (candidate, _fold, _seed), run in runs.items()
        if candidate == SELECTED_CANDIDATE
    ]
    excluded = _excluded_dirty_selected_runs(root)
    selected_revisions = {
        str(run["repository"]["git_revision"]) for run in selected_runs
    }
    source_hashes = {
        str(run["resolved_configuration"]["source_sha256"])
        for run in runs.values()
    }
    split_hashes = {
        str(run["resolved_configuration"]["split_sha256"])
        for run in runs.values()
    }
    selected_manifest = runs[(SELECTED_CANDIDATE, "fold-1", 17)]
    resolved = selected_manifest["resolved_configuration"]
    trainer = resolved["trainer_config"]
    split_config = load_yaml(
        "configs/splits/frozen/otbench_usna_lg_fusion_v2.yaml",
        root=root,
    )
    conclusion = (
        (
            "Fusion v2.5 passed every rolling-development condition and is "
            "eligible for one frozen confirmation evaluation. This is still "
            "not a champion or promotion decision."
        )
        if passed
        else (
            "Fusion v2.5 did not qualify for confirmation. Its mean primary "
            f"RMSE change versus persistence was "
            f"{100 * improvement_persistence:.2f}%, and "
            f"{len(failed_conditions)} frozen conditions failed. "
            "Confirmation labels remain sealed."
        )
    )
    report_path = (
        root
        / "reports"
        / "generated"
        / "fusion-v25-robustness"
        / "fusion-v25-robustness-report.pdf"
    )
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v25-rolling-robustness",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v25_robustness",
        "report_short_title": "Fusion v2.5 robustness",
        "report_model_label": "Strata-OT Fusion v2.5",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "evaluation_partition": "rolling_selection",
        "hypothesis": (
            "The fixed nine-epoch history-linear shortcut candidate will "
            "retain its selection improvement across three chronological "
            "weather periods and three seeds without tail, calibration, "
            "bias, or stability regression."
        ),
        "plain_language_question": (
            "Does the promising history shortcut still help when the custom "
            "model is trained on more data and tested across seasons and "
            "random seeds?"
        ),
        "plain_language_conclusion": conclusion,
        "protocol": "docs/FUSION_V25_ROBUSTNESS.md",
        "freeze": (
            "research/experiments/fusion-v2-program/"
            "v25-robustness-freeze.json"
        ),
        "repository_revision": next(iter(selected_revisions)),
        "data_identity": {
            "source_sha256": next(iter(source_hashes)),
            "split_sha256": next(iter(split_hashes)),
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "rolling_split": {
            "id": split_config["id"],
            "minimum_boundary_purge_minutes": split_config[
                "minimum_boundary_purge_minutes"
            ],
            "folds": split_config["folds"],
        },
        "frozen_configuration": {
            "candidate": resolved["candidate"],
            "parameters": selected_manifest["parameters"],
            "feature_names": resolved["feature_names"],
            "trainer": {
                "precision": trainer["precision"],
                "full_max_epochs": resolved["max_epochs_requested"],
                "batch_size": trainer["batch_size"],
                "gradient_accumulation": trainer[
                    "gradient_accumulation"
                ],
                "learning_rate": trainer["learning_rate"],
                "weight_decay": trainer["weight_decay"],
                "early_stopping_patience": trainer[
                    "early_stopping_patience"
                ],
                "full_max_train_examples": trainer[
                    "full_max_train_examples"
                ],
                "full_max_evaluation_examples": trainer[
                    "full_max_evaluation_examples"
                ],
                "full_max_seconds": trainer["full_max_seconds"],
            },
        },
        "primary_horizon_minutes": list(PRIMARY_HORIZONS),
        "anchor_horizon_minutes": 5,
        "stronger_neural_control": stronger_neural_id,
        "aggregates": aggregates,
        "selected_aggregate": selected,
        "result": {
            "selected_primary_rmse": selected_rmse,
            "persistence_primary_rmse": persistence_rmse,
            "stronger_neural_primary_rmse": neural_rmse,
            "diagnostic_lightgbm_primary_rmse": lightgbm_rmse,
            "predecessor_primary_rmse": predecessor_rmse,
            "relative_improvement_over_persistence": (
                improvement_persistence
            ),
            "relative_improvement_over_stronger_neural": improvement_neural,
            "relative_improvement_over_diagnostic_lightgbm": (
                improvement_lightgbm
            ),
            "relative_improvement_over_predecessor": (
                improvement_predecessor
            ),
            "seed_directions": seed_directions,
            "bootstrap_vs_persistence": bootstrap_persistence,
            "bootstrap_vs_stronger_neural": bootstrap_neural,
            "confirmation_eligible": passed,
            "failed_conditions": failed_conditions,
        },
        "checks": checks,
        "excluded_dirty_runs": excluded,
        "resource_narrative": (
            "This cycle trained exactly nine fresh v2.5 neural runs. The 45 "
            "unchanged comparison runs were reused by exact frozen ID and are "
            "not charged again. Dirty runs, if any, are retained as failure "
            "evidence and excluded. No control was retrained or substituted."
        ),
        "next_cycle_narrative": (
            (
                "If the result and frozen identity are committed unchanged, "
                "the explicit one-shot confirmation interface may be used."
            )
            if passed
            else (
                "The history-shortcut robustness family closes without "
                "confirmation release; the next cycle must be a distinct "
                "evidence-justified hypothesis."
            )
        ),
        "resources": {
            "matrix_run_count": len(runs),
            "neural_run_count": len(selected_runs),
            "reused_control_run_count": len(runs) - len(selected_runs),
            "neural_wall_clock_hours": sum(
                float(run["metrics"]["wall_clock_seconds"])
                for run in selected_runs
            )
            / 3600.0,
            "excluded_dirty_neural_wall_clock_hours": sum(
                float(run["wall_clock_seconds"]) for run in excluded
            )
            / 3600.0,
            "consumed_neural_wall_clock_hours": (
                sum(
                    float(run["metrics"]["wall_clock_seconds"])
                    for run in selected_runs
                )
                + sum(
                    float(run["wall_clock_seconds"]) for run in excluded
                )
            )
            / 3600.0,
            "peak_total_board_vram_gib": max(
                float(run["metrics"]["peak_total_board_vram_gib"])
                for run in selected_runs
            ),
            "peak_process_allocated_vram_gib": max(
                float(
                    run["metrics"].get(
                        "peak_process_allocated_vram_gib",
                        0.0,
                    )
                )
                for run in selected_runs
            ),
            "peak_process_rss_gib": max(
                float(run["metrics"].get("process_rss_gib", 0.0))
                for run in selected_runs
            ),
            "artifact_storage_bytes": sum(
                int(run["metrics"]["artifact_storage_bytes"])
                for run in selected_runs
            ),
            "excluded_dirty_artifact_storage_bytes": sum(
                int(run["artifact_storage_bytes"]) for run in excluded
            ),
            "consumed_artifact_storage_bytes": (
                sum(
                    int(run["metrics"]["artifact_storage_bytes"])
                    for run in selected_runs
                )
                + sum(
                    int(run["artifact_storage_bytes"]) for run in excluded
                )
            ),
            "cloud_cost_usd": 0.0,
        },
        "report_pdf": (
            report_path.relative_to(root).as_posix()
            if report_path.is_file()
            else None
        ),
    }


def write_v25_robustness_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "v25-robustness-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_v25_robustness_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_v25_robustness_evidence(
    summary_path: Path,
    report_path: Path,
) -> str:
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
        run_name="fusion-v25-rolling-robustness-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "rolling-selection-robustness",
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
                "robustness/selected_primary_rmse": float(
                    summary["result"]["selected_primary_rmse"]
                ),
                "robustness/persistence_primary_rmse": float(
                    summary["result"]["persistence_primary_rmse"]
                ),
                "robustness/stronger_neural_primary_rmse": float(
                    summary["result"]["stronger_neural_primary_rmse"]
                ),
                "robustness/improvement_over_persistence": float(
                    summary["result"][
                        "relative_improvement_over_persistence"
                    ]
                ),
                "robustness/improvement_over_predecessor": float(
                    summary["result"]["relative_improvement_over_predecessor"]
                ),
                "resources/neural_wall_clock_hours": float(
                    summary["resources"]["neural_wall_clock_hours"]
                ),
                "resources/peak_total_board_vram_gib": float(
                    summary["resources"]["peak_total_board_vram_gib"]
                ),
            }
        )
        paths = (
            (summary_path, "evidence"),
            (report_path, "reports"),
            (report_path.with_suffix(".tex"), "reports"),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "v25-robustness-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "v25-robustness-report-inspection.json",
                "evidence",
            ),
            (root / "docs" / "FUSION_V25_ROBUSTNESS.md", "protocol"),
        )
        for path, artifact_path in paths:
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build frozen Fusion v2.5 rolling robustness evidence"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v25-robustness/"
            "fusion-v25-robustness-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    if args.freeze:
        write_v25_robustness_freeze(root)
    summary_path = write_v25_robustness_summary(root)
    if args.log_mlflow:
        print(
            log_v25_robustness_evidence(
                summary_path,
                root / args.report,
            )
        )
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
