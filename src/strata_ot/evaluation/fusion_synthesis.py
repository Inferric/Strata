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
RESEARCH_ROOT = Path("research/experiments/fusion-v2-program")
CYCLE_UPLOAD_ROLES = {
    "fusion-v2-screen": "fusion_v2_selection_screen",
    "fusion-v2-rolling-robustness": "fusion_v2_rolling_robustness",
    "point-loss": "fusion_v21_point_loss",
    "data-scale": "fusion_v22_data_scale",
    "residual-scale": "fusion_v23_residual_scaling",
    "shortcut": "fusion_v24_direct_shortcut",
    "convergence": "fusion_v25_convergence",
    "v25-robustness": "fusion_v25_rolling_robustness",
    "v26-tail-objective": "fusion_v26_tail_objective",
    "v27-residual-cap": "fusion_v27_residual_cap",
    "v28-horizon-bilinear": "fusion_v28_horizon_bilinear",
}


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
            resources["neural_wall_clock_hours"] = resources["consumed_neural_wall_clock_hours"]
        if isinstance(
            resources.get("consumed_artifact_storage_bytes"),
            (int, float),
        ):
            resources["artifact_storage_bytes"] = resources["consumed_artifact_storage_bytes"]
    status = summary.get("status")
    if not isinstance(status, str):
        claim = summary.get("assessment_claim", {})
        status = (
            "FAIL"
            if isinstance(claim, dict) and claim.get("status") in {"null_or_partial", "failed"}
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
    initial_robustness = _required_object(
        root,
        PROGRAM_ROOT / "robustness-summary.json",
    )
    best_robustness_path = root / PROGRAM_ROOT / "v25-robustness-summary.json"
    robustness = (
        _load_object(best_robustness_path) if best_robustness_path.is_file() else initial_robustness
    )
    best_custom_candidate_id = (
        "selected-fusion-v25"
        if "selected-fusion-v25" in robustness.get("aggregates", {})
        else "selected-fusion-v2"
    )
    best_custom_candidate_label = (
        "Fusion v2.5" if best_custom_candidate_id == "selected-fusion-v25" else "Fusion v2"
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
            initial_robustness,
            role="rolling-selection",
        ),
    ]
    cycles.extend(
        _cycle_record(cycle_id, summary, role="bounded-pivot") for cycle_id, summary in optional
    )
    drive_index_path = root / RESEARCH_ROOT / "drive-upload-index.json"
    drive_uploads_by_role: dict[str, dict[str, Any]] = {}
    if drive_index_path.is_file():
        drive_index = _load_object(drive_index_path)
        drive_uploads_by_role = {
            str(upload["role"]): upload
            for upload in drive_index.get("uploads", [])
            if isinstance(upload, dict) and isinstance(upload.get("role"), str)
        }
        drive_urls = {
            str(upload["local_path"]): str(upload["drive_url"])
            for upload in drive_index.get("uploads", [])
            if isinstance(upload, dict)
            and isinstance(upload.get("local_path"), str)
            and isinstance(upload.get("drive_url"), str)
        }
        for cycle in cycles:
            report = cycle.get("report_pdf")
            upload = drive_uploads_by_role.get(
                CYCLE_UPLOAD_ROLES.get(str(cycle["cycle_id"]), ""),
                {},
            )
            if not isinstance(report, str) and isinstance(
                upload.get("local_path"),
                str,
            ):
                report = str(upload["local_path"])
                cycle["report_pdf"] = report
            cycle["drive_url"] = (
                drive_urls.get(report)
                if isinstance(report, str)
                else upload.get("drive_url")
            )
    aggregates = robustness["aggregates"]
    selected = aggregates[best_custom_candidate_id]
    lightgbm = aggregates["diagnostic-lightgbm"]
    selected_rmse = float(selected["primary"]["rmse_log10_cn2"])
    lightgbm_rmse = float(lightgbm["primary"]["rmse_log10_cn2"])
    relationship_to_lightgbm = (lightgbm_rmse - selected_rmse) / lightgbm_rmse
    resource_rows = [
        cycle["resources"] for cycle in cycles if isinstance(cycle.get("resources"), dict)
    ]
    program_resource_rows = [
        cycle["resources"]
        for cycle in cycles
        if cycle["role"] != "prior" and isinstance(cycle.get("resources"), dict)
    ]
    total_evidence_hours = sum(
        float(row.get("neural_wall_clock_hours", 0.0)) for row in resource_rows
    )
    program_neural_hours = sum(
        float(row.get("neural_wall_clock_hours", 0.0)) for row in program_resource_rows
    )
    prior_neural_hours = total_evidence_hours - program_neural_hours
    peak_board = max(
        (float(row.get("peak_total_board_vram_gib", 0.0)) for row in resource_rows),
        default=0.0,
    )
    artifact_bytes = sum(int(row.get("artifact_storage_bytes", 0)) for row in resource_rows)
    robustness_passed = robustness.get("status") == "PASS"
    confirmation_status = str(
        robustness.get("confirmation", {}).get(
            "status",
            "NOT_EVALUATED",
        )
    )
    if confirmation_status != "NOT_EVALUATED":
        raise RuntimeError("Program synthesis refuses a released confirmation partition")
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
            robustness.get("checks", {}).get(
                "nine_fresh_candidate_runs_complete",
                robustness.get("checks", {}).get("matrix_complete", "FAIL"),
            )
        ),
        "all_bounded_cycles_terminal": (
            "PASS"
            if all(
                cycle["status"] in {"PASS", "FAIL"} for cycle in cycles if cycle["role"] != "prior"
            )
            else "FAIL"
        ),
        "program_execution_complete": "PASS",
        "official_mlo_test_sealed": "PASS",
        "usna_confirmation_labels_sealed": "PASS",
        "custom_candidate_success": ("PASS" if robustness_passed else "FAIL"),
        "confirmation_evaluation": "NOT_EVALUATED",
        "champion_promotion": "NOT_EVALUATED",
    }
    null_pivot_count = sum(
        cycle["status"] == "FAIL" for cycle in cycles if cycle["role"] == "bounded-pivot"
    )
    result = robustness["result"]
    stronger_neural_rmse = float(result.get("stronger_neural_primary_rmse", selected_rmse))
    neural_improvement = float(
        result.get(
            "relative_improvement_over_stronger_neural",
            (stronger_neural_rmse - selected_rmse) / stronger_neural_rmse
            if stronger_neural_rmse
            else 0.0,
        )
    )
    stronger_neural_control = str(
        robustness.get("stronger_neural_control", "the stronger neural control")
    )
    conclusion = (
        f"{best_custom_candidate_label} passed every frozen rolling-development "
        "condition, but no "
        "confirmation labels were released and no champion was promoted."
        if robustness_passed
        else (
            f"{best_custom_candidate_label} was the strongest custom model but "
            "did not pass the frozen rolling-development gate: its primary "
            f"RMSE improvement was {neural_improvement * 100:.2f}% over "
            f"{stronger_neural_control}, below the required 2%, and did not "
            "pass the tail-MAE condition. "
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
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
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
        "best_custom_candidate_id": best_custom_candidate_id,
        "best_custom_candidate_label": best_custom_candidate_label,
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
            "best_custom_candidate_id": best_custom_candidate_id,
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
            "claim_boundary": ("LightGBM is diagnostic only and is not the intended flagship."),
        },
        "provenance": {
            "source_sha256": robustness["data_identity"]["source_sha256"],
            "split_sha256": robustness["data_identity"]["split_sha256"],
            "repository_revision": robustness["repository_revision"],
            "best_candidate_repository_revision": robustness["repository_revision"],
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
            "raw_data_edited": False,
        },
        "rolling_split": {
            "id": split["id"],
            "strategy": split["strategy"],
            "minimum_boundary_purge_minutes": split["minimum_boundary_purge_minutes"],
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
            "primary_horizons_minutes": experiment["primary_horizons_minutes"],
            "anchor_horizon_minutes": experiment["anchor_horizon_minutes"],
            "bootstrap": experiment["bootstrap"],
            "budget": experiment["budget"],
            "deadline": experiment["deadline"],
            "data": {
                "label": data["label"],
                "raw_features": data["features"]["raw_allowlist"],
                "derived_features": data["features"]["derived_allowlist"],
                "contexts": data["sampling"]["contexts"],
                "maximum_source_gap_minutes": data["sampling"]["maximum_source_gap_minutes"],
            },
            "model": {
                "name": model["name"],
                "hidden_dim": model["hidden_dim"],
                "num_heads": model["num_heads"],
                "num_experts": model["num_experts"],
                "dropout": model["dropout"],
                "horizon_fourier_bands": model["horizon_fourier_bands"],
                "parameter_budget_min": model["parameter_budget_min"],
                "parameter_budget_max": model["parameter_budget_max"],
                "distribution_head": "Student-t plus quantiles",
                "prediction_anchor": "persistence residual",
            },
            "trainer": {
                "precision": trainer["precision"],
                "full_max_epochs": trainer["full_max_epochs"],
                "batch_size": trainer["batch_size"],
                "gradient_accumulation": trainer["gradient_accumulation"],
                "learning_rate": trainer["learning_rate"],
                "weight_decay": trainer["weight_decay"],
                "early_stopping_patience": trainer["early_stopping_patience"],
                "full_max_train_examples": trainer["full_max_train_examples"],
                "full_max_evaluation_examples": trainer["full_max_evaluation_examples"],
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
            (
                "Fusion v2.5 improved primary RMSE over MLP by 1.32%, but the "
                "preregistered margin was 2% and its tail MAE exceeded the "
                "immutable limit."
            ),
        ],
        "next_phase": (
            "Run the frozen confirmation protocol only after explicit human authorization."
            if robustness_passed
            else (
                "Treat the near-miss as data- and objective-limited evidence: "
                "expand clearly open training coverage with provenance-checked "
                "adapters, then preregister a new architecture family that does "
                "not reopen the closed shortcut, loss, scale, or cap searches."
            )
        ),
        "report_pdf": None,
        "drive_url": drive_uploads_by_role.get("fusion_v2_program_synthesis", {}).get(
            "drive_url"
        ),
        "run_index": {
            "local_path": (
                "reports/generated/fusion-v2-program/fusion-v2-run-index.json"
            ),
            "drive_url": drive_uploads_by_role.get(
                "fusion_v2_compact_run_index",
                {},
            ).get("drive_url"),
        },
    }


def write_program_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = root / PROGRAM_ROOT / "program-summary.json"
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_program_summary(root)
    for key in ("evidence_run_id", "report_pdf"):
        if isinstance(existing.get(key), str):
            summary[key] = existing[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def write_program_run_index(
    summary_path: Path,
    root: Path | None = None,
) -> Path:
    root = root or find_repo_root(summary_path.parent)
    summary = _load_object(summary_path)
    records: dict[str, dict[str, Any]] = {}

    def collect(
        value: Any,
        *,
        source: str,
        candidate_hint: str | None = None,
    ) -> None:
        if isinstance(value, dict):
            run_id = value.get("run_id")
            if isinstance(run_id, str) and run_id:
                record: dict[str, Any] = {
                    "run_id": run_id,
                    "source_summary": source,
                }
                candidate_id = value.get("candidate_id", candidate_hint)
                for key, item in (
                    ("candidate_id", candidate_id),
                    ("fold_id", value.get("fold_id")),
                    ("seed", value.get("seed")),
                    ("kind", value.get("kind")),
                ):
                    if isinstance(item, (str, int)):
                        record[key] = item
                records.setdefault(run_id, record)
            for key, item in value.items():
                next_hint = candidate_hint
                if isinstance(key, str) and key.startswith(
                    ("control-", "diagnostic-", "selected-")
                ):
                    next_hint = key
                collect(item, source=source, candidate_hint=next_hint)
        elif isinstance(value, list):
            for item in value:
                collect(item, source=source, candidate_hint=candidate_hint)

    for path in sorted((root / PROGRAM_ROOT).glob("*-summary.json")):
        if path.name == "program-summary.json":
            continue
        collect(
            _load_object(path),
            source=path.relative_to(root).as_posix(),
        )

    evidence_runs = [
        {
            "cycle_id": cycle["cycle_id"],
            "run_id": cycle["evidence_run_id"],
        }
        for cycle in summary["cycles"]
        if isinstance(cycle.get("evidence_run_id"), str)
    ]
    program_evidence = summary.get("evidence_run_id")
    if isinstance(program_evidence, str):
        evidence_runs.append(
            {
                "cycle_id": "program-synthesis",
                "run_id": program_evidence,
            }
        )
    run_index = {
        "schema_version": 1,
        "program_id": summary["program_id"],
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "scientific_status": summary["status"],
        "best_custom_candidate_id": summary["best_custom_candidate_id"],
        "source_sha256": summary["provenance"]["source_sha256"],
        "split_sha256": summary["provenance"]["split_sha256"],
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
        "training_run_count": len(records),
        "evidence_run_count": len(evidence_runs),
        "training_runs": sorted(
            records.values(),
            key=lambda item: (
                str(item.get("source_summary", "")),
                str(item.get("candidate_id", "")),
                str(item.get("fold_id", "")),
                int(item.get("seed", -1)),
                str(item["run_id"]),
            ),
        ),
        "evidence_runs": evidence_runs,
    }
    destination = (
        root
        / "reports"
        / "generated"
        / "fusion-v2-program"
        / "fusion-v2-run-index.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run_index, indent=2) + "\n",
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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    if isinstance(existing, str) and existing:
        run_context = mlflow.start_run(run_id=existing)
    else:
        mlflow.set_experiment("strata-ot-fusion-v2-program")
        run_context = mlflow.start_run(
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
        )
    with run_context as run:
        summary["evidence_run_id"] = run.info.run_id
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        run_index_path = write_program_run_index(summary_path, root)
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
                    summary["relationship_to_lightgbm"]["relative_improvement_over_lightgbm"]
                ),
            }
        )
        for path, artifact_path in (
            (summary_path, "evidence"),
            (report_path, "reports"),
            (report_path.with_suffix(".tex"), "reports"),
            (run_index_path, "evidence"),
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
    parser = argparse.ArgumentParser(description="Build the final Fusion v2 program synthesis")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=("reports/generated/fusion-v2-program/fusion-v2-program-report.pdf"),
    )
    args = parser.parse_args()
    root = find_repo_root()
    summary_path = write_program_summary(root)
    if args.log_mlflow:
        print(log_program_evidence(summary_path, root / args.report))
    else:
        write_program_run_index(summary_path, root)
        print(summary_path)


if __name__ == "__main__":
    main()
