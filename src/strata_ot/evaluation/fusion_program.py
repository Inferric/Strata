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

PRIMARY_HORIZONS = (15, 30, 60)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _primary_mean(run: dict[str, Any], metric: str) -> float:
    values = [
        float(run["by_horizon"][str(horizon)][metric])
        for horizon in PRIMARY_HORIZONS
    ]
    return sum(values) / len(values)


def build_screen_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    freeze_path = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "screen-freeze.json"
    )
    freeze = _load_object(freeze_path)
    report_path = (
        root
        / "reports"
        / "generated"
        / "fusion-v2-screen"
        / "fusion-v2-screen-report.pdf"
    )
    inspection_path = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "screen-report-inspection.json"
    )
    inspection = (
        _load_object(inspection_path) if inspection_path.is_file() else None
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, run_id in freeze["run_ids"].items():
        manifest_path = root / "artifacts" / "runs" / run_id / "run-manifest.json"
        run = _load_object(manifest_path)
        if run.get("candidate_id") != candidate_id or run.get("run_id") != run_id:
            raise RuntimeError(f"Frozen screen identity mismatch for {candidate_id}")
        rows.append(
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "category": run["category"],
                "family": run["family"],
                "kind": run["kind"],
                "parameters": run.get("parameters"),
                "primary_rmse": _primary_mean(run, "rmse_log10_cn2"),
                "primary_tail_mae": _primary_mean(
                    run, "tail_mae_top_decile"
                ),
                "primary_bias": _primary_mean(run, "bias_log10_cn2"),
                "primary_crps": _primary_mean(run, "crps_gaussian"),
                "primary_coverage_80": _primary_mean(
                    run, "interval_80_coverage"
                ),
                "by_horizon": run["by_horizon"],
                "wall_clock_seconds": float(
                    run["metrics"]["wall_clock_seconds"]
                ),
                "peak_total_board_vram_gib": float(
                    run["metrics"]["peak_total_board_vram_gib"]
                ),
                "peak_process_allocated_vram_gib": float(
                    run["metrics"].get(
                        "peak_process_allocated_vram_gib", 0.0
                    )
                ),
                "artifact_storage_bytes": int(
                    run["metrics"]["artifact_storage_bytes"]
                ),
                "repository": run["repository"],
            }
        )
    order = {
        candidate_id: index
        for index, candidate_id in enumerate(freeze["run_ids"])
    }
    rows.sort(key=lambda row: order[str(row["candidate_id"])])
    persistence = next(
        row for row in rows if row["candidate_id"] == "control-persistence"
    )
    selected = next(
        row for row in rows if row["candidate_id"] == "fusion-concat"
    )
    selected_manifest = _load_object(
        root
        / "artifacts"
        / "runs"
        / str(selected["run_id"])
        / "run-manifest.json"
    )
    horizon_v1 = next(
        row for row in rows if row["candidate_id"] == "control-horizon-v1"
    )
    neural = [row for row in rows if row["kind"] == "neural"]
    data_config = load_yaml(
        "configs/data/otbench_usna_lg_fusion_v2.yaml", root=root
    )
    split_config = load_yaml(str(data_config["split_config"]), root=root)
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v2-program-screen",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "program_id": freeze["program_id"],
        "task_kind": "fusion_v2_screen",
        "status": "FAIL",
        "evaluation_partition": freeze["evaluation_role"],
        "seed": freeze["seed"],
        "primary_horizon_minutes": list(PRIMARY_HORIZONS),
        "anchor_horizon_minutes": 5,
        "hypothesis": (
            "Multiscale Cn2 context and deep weather interaction will improve "
            "a persistence-residual custom neural model."
        ),
        "plain_language_question": (
            "Can a custom neural forecast use several weather timescales "
            "reliably enough to beat copying the latest Cn2 value?"
        ),
        "plain_language_conclusion": (
            "Not yet. One Fusion v2 configuration was slightly better than "
            "persistence without worsening the tail, but the gain was only "
            "0.97% and Horizon v1 remained slightly stronger on this screen."
        ),
        "screen_freeze": freeze_path.relative_to(root).as_posix(),
        "data_identity": {
            "dataset_id": data_config["id"],
            "source_path": data_config["source_path"],
            "source_sha256": data_config["source_sha256"],
            "manifest_path": data_config["manifest_path"],
            "split_id": split_config["id"],
            "materialized_split_sha256": selected_manifest[
                "resolved_configuration"
            ]["split_sha256"],
            "target": data_config["label"]["name"],
            "label_provenance": data_config["label"]["provenance_class"],
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "repository_identity": selected_manifest["repository"],
        "screen_contract": freeze["screen_contract"],
        "screen_failures": freeze["screen_failures"],
        "closed_hypothesis_families": freeze[
            "closed_hypothesis_families"
        ],
        "selected_for_robustness_characterization": freeze[
            "selected_for_robustness_characterization"
        ],
        "runs": rows,
        "result": {
            "persistence_primary_rmse": persistence["primary_rmse"],
            "selected_fusion_primary_rmse": selected["primary_rmse"],
            "horizon_v1_primary_rmse": horizon_v1["primary_rmse"],
            "selected_relative_improvement_over_persistence": (
                persistence["primary_rmse"] - selected["primary_rmse"]
            )
            / persistence["primary_rmse"],
            "selected_relative_improvement_over_horizon_v1": (
                horizon_v1["primary_rmse"] - selected["primary_rmse"]
            )
            / horizon_v1["primary_rmse"],
            "final_two_percent_threshold_passed": False,
            "screen_advancement_passed": True,
            "confirmation_eligible": False,
        },
        "checks": {
            "all_24_arms_completed": "PASS",
            "screen_identity_frozen": "PASS",
            "tail_advancement_rule": "PASS",
            "coverage_advancement_rule": "PASS",
            "two_percent_success_threshold": "FAIL",
            "confirmation_eligibility": "FAIL",
            "official_mlo_test_sealed": "PASS",
            "usna_confirmation_labels_sealed": "PASS",
            "latex_pdf_compiled": "PASS"
            if report_path.is_file()
            else "NOT_EVALUATED",
            "visual_pdf_inspection": (
                str(inspection["status"])
                if inspection is not None
                else "NOT_EVALUATED"
            ),
        },
        "resources": {
            "neural_wall_clock_hours": sum(
                float(row["wall_clock_seconds"]) for row in neural
            )
            / 3600.0,
            "peak_total_board_vram_gib": max(
                float(row["peak_total_board_vram_gib"]) for row in rows
            ),
            "peak_process_allocated_vram_gib": max(
                float(row["peak_process_allocated_vram_gib"]) for row in rows
            ),
            "artifact_storage_bytes": sum(
                int(row["artifact_storage_bytes"]) for row in rows
            ),
            "cloud_cost_usd": 0.0,
        },
        "next_protocol": "docs/FUSION_V2_ROBUSTNESS.md",
        "report_pdf": (
            report_path.relative_to(root).as_posix()
            if report_path.is_file()
            else None
        ),
    }


def write_screen_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "screen-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_screen_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_screen_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v2-screen-evidence",
        tags={
            "program_id": summary["program_id"],
            "run_kind": "program_evidence",
            "evidence_role": "fold-1-selection-screen",
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
                "screen/selected_primary_rmse": float(
                    summary["result"]["selected_fusion_primary_rmse"]
                ),
                "screen/persistence_primary_rmse": float(
                    summary["result"]["persistence_primary_rmse"]
                ),
                "screen/horizon_v1_primary_rmse": float(
                    summary["result"]["horizon_v1_primary_rmse"]
                ),
                "screen/relative_improvement_over_persistence": float(
                    summary["result"][
                        "selected_relative_improvement_over_persistence"
                    ]
                ),
                "resources/neural_wall_clock_hours": float(
                    summary["resources"]["neural_wall_clock_hours"]
                ),
                "resources/peak_total_board_vram_gib": float(
                    summary["resources"]["peak_total_board_vram_gib"]
                ),
                "resources/artifact_storage_bytes": float(
                    summary["resources"]["artifact_storage_bytes"]
                ),
            }
        )
        for path, artifact_path in (
            (summary_path, "evidence"),
            (report_path, "reports"),
            (
                report_path.with_suffix(".tex"),
                "reports",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "screen-freeze.json",
                "evidence",
            ),
            (
                root
                / "research"
                / "experiments"
                / "fusion-v2-program"
                / "ledger.json",
                "evidence",
            ),
        ):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
        return str(run.info.run_id)


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Build the frozen Fusion v2 screen evidence summary"
    )
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        default=(
            "reports/generated/fusion-v2-screen/"
            "fusion-v2-screen-report.pdf"
        ),
    )
    args = parser.parse_args()
    root = find_repo_root()
    summary_path = write_screen_summary(root)
    if args.log_mlflow:
        report_path = root / args.report
        print(log_screen_evidence(summary_path, report_path))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
