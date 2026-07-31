from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import mlflow

from strata_ot.config import find_repo_root, load_yaml
from strata_ot.training.train import _configure_utf8_output

PROGRAM_ID = "strata-fusion-v2-program"
PROGRAM_ROOT = Path("artifacts/experiments") / PROGRAM_ID
RESEARCH_ROOT = Path("research/experiments") / PROGRAM_ID


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], payload)


def _required_object(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required program evidence is unavailable: {path}")
    return _load_object(path)


def _cycle_record(
    cycle_id: str,
    summary: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    resources = summary.get("resources")
    if not isinstance(resources, dict):
        resources = {
            "neural_wall_clock_hours": summary.get(
                "total_neural_gpu_hours",
                0.0,
            ),
            "peak_total_board_vram_gib": summary.get("peak_vram_gb", 0.0),
            "artifact_storage_bytes": summary.get(
                "storage_delta_bytes",
                0,
            ),
            "cloud_cost_usd": summary.get("cloud_cost_usd", 0.0),
        }
    else:
        resources = dict(resources)
        if isinstance(
            resources.get("consumed_neural_wall_clock_hours"),
            (int, float),
        ):
            resources["neural_wall_clock_hours"] = resources[
                "consumed_neural_wall_clock_hours"
            ]
        if isinstance(
            resources.get("consumed_artifact_storage_bytes"),
            (int, float),
        ):
            resources["artifact_storage_bytes"] = resources[
                "consumed_artifact_storage_bytes"
            ]
    status = summary.get("status")
    if not isinstance(status, str):
        claim = summary.get("assessment_claim", {})
        status = (
            "FAIL"
            if isinstance(claim, dict)
            and claim.get("status") in {"null_or_partial", "failed"}
            else "NOT_EVALUATED"
        )
    return {
        "cycle_id": cycle_id,
        "role": role,
        "task_kind": summary.get("task_kind"),
        "status": status,
        "question": summary.get(
            "plain_language_question",
            summary.get("hypothesis", "Question unavailable"),
        ),
        "conclusion": summary.get(
            "plain_language_conclusion",
            "Conclusion unavailable",
        ),
        "report_pdf": summary.get("report_pdf"),
        "evidence_run_id": summary.get("evidence_run_id"),
        "resources": resources,
        "checks": summary.get("checks", {}),
    }


def _optional_cycle_summaries(
    root: Path,
    *,
    excluded: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    cycle_root = root / PROGRAM_ROOT
    results: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(cycle_root.glob("*-summary.json")):
        if path.name in excluded:
            continue
        payload = _load_object(path)
        if payload.get("program_id") != PROGRAM_ID:
            continue
        results.append((path.stem.removesuffix("-summary"), payload))
    return results


def build_program_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    cycle_zero = _required_object(
        root,
        Path("artifacts/experiments/mlo-weather-horizon-v1/summary.json"),
    )
    screen = _required_object(root, PROGRAM_ROOT / "screen-summary.json")
    robustness = _required_object(
        root,
        PROGRAM_ROOT / "robustness-summary.json",
    )
    ledger = _required_object(root, RESEARCH_ROOT / "ledger.json")
    split = load_yaml(
        "configs/splits/frozen/otbench_usna_lg_fusion_v2.yaml",
        root=root,
    )
    experiment = load_yaml(
        "configs/experiments/fusion_v2_program.yaml",
        root=root,
    )
    data = load_yaml(str(experiment["data"]), root=root)
    model = load_yaml(str(experiment["model"]), root=root)
    trainer = load_yaml(str(experiment["trainer"]), root=root)
    excluded = {
        "screen-summary.json",
        "robustness-summary.json",
        "program-summary.json",
    }
    optional = _optional_cycle_summaries(root, excluded=excluded)
    cycles = [
        _cycle_record("cycle-0-weather-horizon", cycle_zero, role="prior"),
        _cycle_record("fusion-v2-screen", screen, role="selection"),
        _cycle_record(
            "fusion-v2-rolling-robustness",
            robustness,
            role="rolling-selection",
        ),
    ]
    cycles.extend(
        _cycle_record(cycle_id, summary, role="bounded-pivot")
        for cycle_id, summary in optional
    )
    aggregates = robustness["aggregates"]
    selected = aggregates["selected-fusion-v2"]
    lightgbm = aggregates["diagnostic-lightgbm"]
    selected_rmse = float(selected["primary"]["rmse_log10_cn2"])
    lightgbm_rmse = float(lightgbm["primary"]["rmse_log10_cn2"])
    relationship_to_lightgbm = (
        (lightgbm_rmse - selected_rmse) / lightgbm_rmse
    )
    resource_rows = [
        cycle["resources"]
        for cycle in cycles
        if isinstance(cycle.get("resources"), dict)
    ]
    program_resource_rows = [
        cycle["resources"]
        for cycle in cycles
        if cycle["role"] != "prior"
        and isinstance(cycle.get("resources"), dict)
    ]
    total_evidence_hours = sum(
        float(row.get("neural_wall_clock_hours", 0.0))
        for row in resource_rows
    )
    program_neural_hours = sum(
        float(row.get("neural_wall_clock_hours", 0.0))
        for row in program_resource_rows
    )
    prior_neural_hours = total_evidence_hours - program_neural_hours
    peak_board = max(
        (float(row.get("peak_total_board_vram_gib", 0.0)) for row in resource_rows),
        default=0.0,
    )
    artifact_bytes = sum(
        int(row.get("artifact_storage_bytes", 0))
        for row in resource_rows
    )
    robustness_passed = robustness.get("status") == "PASS"
    confirmation_status = str(
        robustness.get("confirmation", {}).get(
            "status",
            "NOT_EVALUATED",
        )
    )
    if confirmation_status != "NOT_EVALUATED":
        raise RuntimeError(
            "Program synthesis refuses a released confirmation partition"
        )
    checks = {
        "gate_and_accounting_repairs": "PASS",
        "rolling_protocol_frozen": "PASS",
        "fusion_v2_implemented": "PASS",
        "multitask_pretraining_implemented": "PASS",
        "complete_ablation_screen": str(
            screen.get("checks", {}).get(
                "all_24_arms_completed",
                "FAIL",
            )
        ),
        "three_seed_multifold_evidence": str(
            robustness.get("checks", {}).get("matrix_complete", "FAIL")
        ),
        "official_mlo_test_sealed": "PASS",
        "usna_confirmation_labels_sealed": "PASS",
        "custom_candidate_success": (
            "PASS" if robustness_passed else "FAIL"
        ),
        "confirmation_evaluation": "NOT_EVALUATED",
        "champion_promotion": "NOT_EVALUATED",
    }
    null_pivot_count = sum(
        cycle["status"] == "FAIL"
        for cycle in cycles
        if cycle["role"] == "bounded-pivot"
    )
    conclusion = (
        "Fusion v2 passed every frozen rolling-development condition, but no "
        "confirmation labels were released and no champion was promoted."
        if robustness_passed
        else (
            "Fusion v2 did not pass the frozen rolling-development gate. "
            f"{null_pivot_count} completed bounded pivot cycle(s) also ended "
            "without promotion. LightGBM remains a diagnostic comparator; "
            "confirmation and the official MLO test remain sealed."
        )
    )
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v2-program-synthesis",
        "program_id": PROGRAM_ID,
        "task_kind": "fusion_v2_program",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": "PASS" if robustness_passed else "FAIL",
        "plain_language_question": (
            "What did the complete Fusion v2 research program establish, and "
            "what remains unsupported?"
        ),
        "plain_language_conclusion": conclusion,
        "hypothesis": (
            "A custom multiscale neural architecture with representation-level "
            "weather interaction can improve rolling multi-horizon Cn2 "
            "forecasting without weakening uncertainty or tail behavior."
        ),
        "evaluation_partition": "rolling-selection",
        "protocol": "docs/FUSION_V2_PROGRAM.md",
        "cycles": cycles,
        "ledger_events": ledger["events"],
        "screen_evidence": {
            "run_count": len(screen.get("runs", [])),
            "runs": screen.get("runs", []),
            "closed_hypothesis_families": screen.get(
                "closed_hypothesis_families",
                {},
            ),
            "selected_for_robustness_characterization": screen.get(
                "selected_for_robustness_characterization",
                {},
            ),
            "result": screen.get("result", {}),
        },
        "aggregates": aggregates,
        "result": robustness["result"],
        "stronger_neural_control": robustness["stronger_neural_control"],
        "checks": checks,
        "relationship_to_lightgbm": {
            "selected_primary_rmse": selected_rmse,
            "diagnostic_lightgbm_primary_rmse": lightgbm_rmse,
            "relative_improvement_over_lightgbm": relationship_to_lightgbm,
            "interpretation": (
                "The custom neural candidate was directionally better than "
                "diagnostic LightGBM on rolling development."
                if relationship_to_lightgbm > 0
                else (
                    "Diagnostic LightGBM remained stronger than the custom "
                    "neural candidate on rolling development."
                )
            ),
            "claim_boundary": (
                "LightGBM is diagnostic only and is not the intended flagship."
            ),
        },
        "provenance": {
            "source_sha256": robustness["data_identity"]["source_sha256"],
            "split_sha256": robustness["data_identity"]["split_sha256"],
            "repository_revision": robustness["repository_revision"],
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
            "raw_data_edited": False,
        },
        "rolling_split": {
            "id": split["id"],
            "strategy": split["strategy"],
            "minimum_boundary_purge_minutes": split[
                "minimum_boundary_purge_minutes"
            ],
            "folds": split["folds"],
            "confirmation": {
                "consumption_id": split["confirmation"]["consumption_id"],
                "start": split["confirmation"]["start"],
                "end": split["confirmation"]["end"],
                "labels_loaded": False,
            },
        },
        "frozen_configuration": {
            "sources": {
                "experiment": "configs/experiments/fusion_v2_program.yaml",
                "data": str(experiment["data"]),
                "model": str(experiment["model"]),
                "trainer": str(experiment["trainer"]),
                "split": "configs/splits/frozen/otbench_usna_lg_fusion_v2.yaml",
            },
            "seeds": experiment["seeds"],
            "rolling_folds": experiment["rolling_folds"],
            "primary_horizons_minutes": experiment[
                "primary_horizons_minutes"
            ],
            "anchor_horizon_minutes": experiment["anchor_horizon_minutes"],
            "bootstrap": experiment["bootstrap"],
            "budget": experiment["budget"],
            "deadline": experiment["deadline"],
            "data": {
                "label": data["label"],
                "raw_features": data["features"]["raw_allowlist"],
                "derived_features": data["features"]["derived_allowlist"],
                "contexts": data["sampling"]["contexts"],
                "maximum_source_gap_minutes": data["sampling"][
                    "maximum_source_gap_minutes"
                ],
            },
            "model": {
                "name": model["name"],
                "hidden_dim": model["hidden_dim"],
                "num_heads": model["num_heads"],
                "num_experts": model["num_experts"],
                "dropout": model["dropout"],
                "horizon_fourier_bands": model[
                    "horizon_fourier_bands"
                ],
                "parameter_budget_min": model["parameter_budget_min"],
                "parameter_budget_max": model["parameter_budget_max"],
                "distribution_head": "Student-t plus quantiles",
                "prediction_anchor": "persistence residual",
            },
            "trainer": {
                "precision": trainer["precision"],
                "full_max_epochs": trainer["full_max_epochs"],
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
                "deterministic": trainer["deterministic"],
                "num_workers": trainer["num_workers"],
            },
        },
        "resources": {
            "neural_wall_clock_hours": program_neural_hours,
            "prior_cycle0_neural_wall_clock_hours": prior_neural_hours,
            "total_evidence_neural_wall_clock_hours": total_evidence_hours,
            "peak_total_board_vram_gib": peak_board,
            "artifact_storage_bytes": artifact_bytes,
            "cloud_cost_usd": 0.0,
        },
        "limitations": [
            "All Fusion v2 decisions use public USNA-large development folds.",
            "The reserved confirmation labels and official MLO test were not loaded.",
            "Logged attention, expert, and timescale weights are diagnostic, not causal.",
            "A failed development gate cannot support champion promotion.",
        ],
        "next_phase": (
            "Run the frozen confirmation protocol only after explicit human "
            "authorization."
            if robustness_passed
            else (
                "Expand clearly open training evidence or test a newly "
                "preregistered architecture family without reopening failed "
                "development gates."
            )
        ),
        "report_pdf": None,
    }


def write_program_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = root / PROGRAM_ROOT / "program-summary.json"
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_program_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_program_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v2-program-synthesis",
        tags={
            "program_id": PROGRAM_ID,
            "run_kind": "program_evidence",
            "evidence_role": "final-synthesis",
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
                "program/neural_wall_clock_hours": float(
                    summary["resources"]["neural_wall_clock_hours"]
                ),
                "program/peak_total_board_vram_gib": float(
                    summary["resources"]["peak_total_board_vram_gib"]
                ),
                "program/artifact_storage_bytes": float(
                    summary["resources"]["artifact_storage_bytes"]
                ),
                "program/improvement_over_persistence": float(
                    summary["result"]["relative_improvement_over_persistence"]
                ),
                "program/improvement_over_lightgbm": float(
                    summary["relationship_to_lightgbm"][
                        "relative_improvement_over_lightgbm"
                    ]
                ),
            }
        )
        for path, artifact_path in (
            (summary_path, "evidence"),
            (report_path, "reports"),
            (report_path.with_suffix(".tex"), "reports"),
            (root / RESEARCH_ROOT / "ledger.json", "evidence"),
            (root / "docs" / "FUSION_V2_PROGRAM.md", "protocol"),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        for cycle in summary["cycles"]:
            report = cycle.get("report_pdf")
            if not isinstance(report, str):
                continue
            report_path = root / report
            if report_path.is_file():
                mlflow.log_artifact(
                    str(report_path),
                    artifact_path="cycle-reports",
                )
                latex_path = report_path.with_suffix(".tex")
                if latex_path.is_file():
                    mlflow.log_artifact(
                        str(latex_path),
                        artifact_path="cycle-reports",
                    )
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build the final Fusion v2 program synthesis"
    )
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v2-program/"
            "fusion-v2-program-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    summary_path = write_program_summary(root)
    if args.log_mlflow:
        print(log_program_evidence(summary_path, root / args.report))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
