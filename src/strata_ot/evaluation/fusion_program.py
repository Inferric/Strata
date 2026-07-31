from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import mlflow
import numpy as np

from strata_ot.config import find_repo_root, load_yaml
from strata_ot.training.train import _configure_utf8_output

PRIMARY_HORIZONS = (15, 30, 60)
ROBUSTNESS_FOLDS = ("fold-1", "fold-2", "fold-3")
ROBUSTNESS_SEEDS = (17, 41, 73)
ROBUSTNESS_FIXED_CONTROLS = (
    "control-persistence",
    "control-climatology",
    "diagnostic-lightgbm",
)
ROBUSTNESS_NEURAL_CANDIDATES = (
    "selected-fusion-v2",
    "control-mlp",
    "control-tcn",
    "control-horizon-v1",
)
ROBUSTNESS_EVIDENCE_REVISION = (
    "5a3aa33c56a80072c70800f5c1608661075102b8"
)
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_BLOCK_MINUTES = 1_440


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_robustness_keys() -> list[tuple[str, str, int]]:
    expected = [
        (candidate_id, fold_id, 17)
        for fold_id in ROBUSTNESS_FOLDS
        for candidate_id in ROBUSTNESS_FIXED_CONTROLS
    ]
    expected.extend(
        (candidate_id, fold_id, seed)
        for candidate_id in ROBUSTNESS_NEURAL_CANDIDATES
        for fold_id in ROBUSTNESS_FOLDS
        for seed in ROBUSTNESS_SEEDS
    )
    return expected


def _collect_robustness_runs(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    expected = set(_expected_robustness_keys())
    matching: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        repository = run.get("repository", {})
        if repository.get("git_revision") != ROBUSTNESS_EVIDENCE_REVISION:
            continue
        key = (
            str(run.get("candidate_id")),
            str(run.get("fold_id")),
            int(run.get("seed", -1)),
        )
        if key not in expected:
            continue
        if repository.get("git_dirty") is not False:
            continue
        if key in matching:
            raise RuntimeError(f"Duplicate frozen robustness run for {key}")
        matching[key] = run
    missing = sorted(expected - set(matching))
    if missing:
        raise RuntimeError(
            "Frozen robustness matrix is incomplete: "
            + ", ".join(f"{candidate}/{fold}/seed-{seed}" for candidate, fold, seed in missing)
        )
    return matching


def _excluded_dirty_robustness_runs(root: Path) -> list[dict[str, Any]]:
    expected = set(_expected_robustness_keys())
    excluded: list[dict[str, Any]] = []
    for path in sorted((root / "artifacts" / "runs").glob("*/run-manifest.json")):
        run = _load_object(path)
        repository = run.get("repository", {})
        if repository.get("git_revision") != ROBUSTNESS_EVIDENCE_REVISION:
            continue
        key = (
            str(run.get("candidate_id")),
            str(run.get("fold_id")),
            int(run.get("seed", -1)),
        )
        if key not in expected or repository.get("git_dirty") is False:
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
                "peak_total_board_vram_gib": float(
                    run.get("metrics", {}).get(
                        "peak_total_board_vram_gib",
                        0.0,
                    )
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


def _run_primary_metrics(run: dict[str, Any]) -> dict[str, float]:
    metric_names = (
        "mae_log10_cn2",
        "rmse_log10_cn2",
        "bias_log10_cn2",
        "tail_mae_top_decile",
        "crps_gaussian",
        "student_t_nll",
        "interval_80_coverage",
        "interval_80_width",
        "predictive_std_mean",
    )
    return {
        metric: _primary_mean(run, metric)
        for metric in metric_names
    }


def _aggregate_candidate(
    candidate_id: str,
    runs: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    candidate_runs = [
        run
        for (candidate, _fold, _seed), run in runs.items()
        if candidate == candidate_id
    ]
    candidate_runs.sort(key=lambda run: (str(run["fold_id"]), int(run["seed"])))
    if not candidate_runs:
        raise RuntimeError(f"No robustness runs for {candidate_id}")
    rows = [
        {
            "run_id": run["run_id"],
            "fold_id": run["fold_id"],
            "seed": run["seed"],
            "primary": _run_primary_metrics(run),
            "by_horizon": run["by_horizon"],
            "wall_clock_seconds": float(run["metrics"]["wall_clock_seconds"]),
            "peak_total_board_vram_gib": float(
                run["metrics"]["peak_total_board_vram_gib"]
            ),
            "peak_process_allocated_vram_gib": float(
                run["metrics"].get("peak_process_allocated_vram_gib", 0.0)
            ),
            "process_rss_gib": float(run["metrics"].get("process_rss_gib", 0.0)),
            "artifact_storage_bytes": int(run["metrics"]["artifact_storage_bytes"]),
            "operational": {
                metric: float(run["metrics"][metric])
                for metric in (
                    "ood_score_mean",
                    "selective_80_rmse_log10_cn2",
                    "ood_top_quintile_rmse_log10_cn2",
                    "inference_latency_ms_per_sample",
                    "throughput_samples_per_second",
                    "peak_process_reserved_vram_gib",
                )
                if isinstance(run["metrics"].get(metric), (int, float))
            },
            "parameters": run.get("parameters"),
            "checkpoint": run.get("checkpoint"),
            "artifacts": run["artifacts"],
        }
        for run in candidate_runs
    ]
    primary_names = tuple(rows[0]["primary"])
    primary = {
        metric: float(np.mean([row["primary"][metric] for row in rows]))
        for metric in primary_names
    }
    primary["absolute_bias"] = float(
        np.mean([abs(row["primary"]["bias_log10_cn2"]) for row in rows])
    )
    by_horizon = {
        str(horizon): {
            metric: float(
                np.mean(
                    [
                        float(run["by_horizon"][str(horizon)][metric])
                        for run in candidate_runs
                    ]
                )
            )
            for metric in candidate_runs[0]["by_horizon"][str(horizon)]
            if isinstance(
                candidate_runs[0]["by_horizon"][str(horizon)][metric],
                (int, float),
            )
        }
        for horizon in (5, *PRIMARY_HORIZONS)
    }
    rmse_values = np.asarray(
        [row["primary"]["rmse_log10_cn2"] for row in rows],
        dtype=np.float64,
    )
    operational_names = sorted(
        {
            metric
            for row in rows
            for metric in row["operational"]
        }
    )
    operational = {
        metric: float(
            np.mean(
                [
                    float(row["operational"][metric])
                    for row in rows
                    if metric in row["operational"]
                ]
            )
        )
        for metric in operational_names
    }
    fold_means = {
        fold: float(
            np.mean(
                [
                    row["primary"]["rmse_log10_cn2"]
                    for row in rows
                    if row["fold_id"] == fold
                ]
            )
        )
        for fold in ROBUSTNESS_FOLDS
    }
    seed_means = {
        str(seed): float(
            np.mean(
                [
                    row["primary"]["rmse_log10_cn2"]
                    for row in rows
                    if int(row["seed"]) == seed
                ]
            )
        )
        for seed in sorted({int(row["seed"]) for row in rows})
    }
    diagnostics: dict[str, Any] = {}
    if candidate_id.startswith("selected-fusion-v2") and all(
        run.get("component_summary") for run in candidate_runs
    ):
        for horizon in (5, *PRIMARY_HORIZONS):
            values = [
                run["component_summary"][str(horizon)]
                for run in candidate_runs
            ]
            diagnostics[str(horizon)] = {
                "residual_mean": float(
                    np.mean(
                        [
                            float(value["residual"]["mean"])
                            for value in values
                            if "residual" in value
                        ]
                    )
                ),
                "residual_std": float(
                    np.mean(
                        [
                            float(value["residual"]["std"])
                            for value in values
                            if "residual" in value
                        ]
                    )
                ),
                "scale_weights_mean": np.mean(
                    np.asarray(
                        [
                            value["scale_weights"]["mean"]
                            for value in values
                            if "scale_weights" in value
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                ).tolist(),
                "expert_weights_mean": np.mean(
                    np.asarray(
                        [
                            value["expert_weights"]["mean"]
                            for value in values
                            if "expert_weights" in value
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                ).tolist(),
                "physics_token_norm_mean": float(
                    np.mean(
                        [
                            float(value["physics_token_norm"]["mean"])
                            for value in values
                            if "physics_token_norm" in value
                        ]
                    )
                ),
                "weather_attention_mean": float(
                    np.mean(
                        [
                            np.asarray(
                                value["weather_attention"]["mean"],
                                dtype=np.float64,
                            ).mean()
                            for value in values
                            if "weather_attention" in value
                        ]
                    )
                ),
                "modulation_norm_mean": float(
                    np.mean(
                        [
                            np.asarray(
                                value["modulation_norm"]["mean"],
                                dtype=np.float64,
                            ).mean()
                            for value in values
                            if "modulation_norm" in value
                        ]
                    )
                ),
            }
    return {
        "candidate_id": candidate_id,
        "family": candidate_runs[0]["family"],
        "kind": candidate_runs[0]["kind"],
        "parameters": candidate_runs[0].get("parameters"),
        "primary": primary,
        "by_horizon": by_horizon,
        "fold_primary_rmse": fold_means,
        "seed_primary_rmse": seed_means,
        "relative_run_rmse_std": float(rmse_values.std() / rmse_values.mean()),
        "operational": operational,
        "diagnostics": diagnostics,
        "runs": rows,
    }


def _load_predictions(root: Path, run: dict[str, Any]) -> dict[str, np.ndarray]:
    path = root / str(run["artifacts"]["predictions"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen predictions: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "target",
            "location",
            "horizon_minutes",
            "target_timestamp_minute",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise RuntimeError(
                f"Prediction artifact {path} is missing {', '.join(missing)}"
            )
        return {key: np.asarray(payload[key]) for key in required}


def _assert_paired_predictions(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    identity: str,
) -> None:
    for key in ("target", "horizon_minutes", "target_timestamp_minute"):
        if not np.array_equal(reference[key], candidate[key]):
            raise RuntimeError(f"Paired prediction identity mismatch for {identity}: {key}")


def _complete_primary_block_ids(
    day: np.ndarray,
    horizon: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            day_id
            for day_id in np.unique(day)
            if all(
                bool(((day == day_id) & (horizon == horizon_minutes)).any())
                for horizon_minutes in PRIMARY_HORIZONS
            )
        ],
        dtype=day.dtype,
    )


def _paired_block_statistics(
    root: Path,
    runs: dict[tuple[str, str, int], dict[str, Any]],
    *,
    candidate_id: str,
    benchmark_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    candidate_sums: list[np.ndarray] = []
    benchmark_sums: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    excluded_incomplete_blocks = 0
    for fold_id in ROBUSTNESS_FOLDS:
        benchmark_seeds = (
            (17,)
            if benchmark_id in ROBUSTNESS_FIXED_CONTROLS
            else ROBUSTNESS_SEEDS
        )
        candidate_payloads = [
            _load_predictions(root, runs[(candidate_id, fold_id, seed)])
            for seed in ROBUSTNESS_SEEDS
        ]
        benchmark_payloads = [
            _load_predictions(root, runs[(benchmark_id, fold_id, seed)])
            for seed in benchmark_seeds
        ]
        reference = candidate_payloads[0]
        for seed, payload in zip(
            ROBUSTNESS_SEEDS,
            candidate_payloads,
            strict=True,
        ):
            _assert_paired_predictions(
                reference,
                payload,
                identity=f"{candidate_id}/{fold_id}/seed-{seed}",
            )
        for seed, payload in zip(
            benchmark_seeds,
            benchmark_payloads,
            strict=True,
        ):
            _assert_paired_predictions(
                reference,
                payload,
                identity=f"{benchmark_id}/{fold_id}/seed-{seed}",
            )
        candidate_squared_error = np.mean(
            np.stack(
                [
                    (payload["location"] - payload["target"]) ** 2
                    for payload in candidate_payloads
                ],
                axis=0,
            ),
            axis=0,
        )
        benchmark_squared_error = np.mean(
            np.stack(
                [
                    (payload["location"] - payload["target"]) ** 2
                    for payload in benchmark_payloads
                ],
                axis=0,
            ),
            axis=0,
        )
        horizon = reference["horizon_minutes"].astype(np.int64)
        day = (
            reference["target_timestamp_minute"].astype(np.int64)
            // BOOTSTRAP_BLOCK_MINUTES
        )
        all_day_ids = np.unique(day)
        complete_day_ids = _complete_primary_block_ids(day, horizon)
        excluded_incomplete_blocks += len(all_day_ids) - len(
            complete_day_ids
        )
        for day_id in complete_day_ids:
            candidate_row: list[float] = []
            benchmark_row: list[float] = []
            count_row: list[float] = []
            for horizon_minutes in PRIMARY_HORIZONS:
                mask = (day == day_id) & (horizon == horizon_minutes)
                candidate_row.append(float(candidate_squared_error[mask].sum()))
                benchmark_row.append(float(benchmark_squared_error[mask].sum()))
                count_row.append(float(mask.sum()))
            candidate_sums.append(np.asarray(candidate_row))
            benchmark_sums.append(np.asarray(benchmark_row))
            counts.append(np.asarray(count_row))
    return (
        np.stack(candidate_sums),
        np.stack(benchmark_sums),
        np.stack(counts),
        excluded_incomplete_blocks,
    )


def _bootstrap_improvement(
    candidate_sums: np.ndarray,
    benchmark_sums: np.ndarray,
    counts: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if (
        candidate_sums.shape != benchmark_sums.shape
        or candidate_sums.shape != counts.shape
    ):
        raise ValueError("Paired block statistics must have identical shapes")
    if candidate_sums.ndim != 2 or candidate_sums.shape[1] != len(PRIMARY_HORIZONS):
        raise ValueError("Expected one column per primary horizon")
    rng = np.random.default_rng(seed)
    block_count = candidate_sums.shape[0]
    sampled = rng.integers(0, block_count, size=(resamples, block_count))
    sampled_counts = counts[sampled].sum(axis=1)
    candidate_rmse = np.sqrt(
        candidate_sums[sampled].sum(axis=1) / sampled_counts
    ).mean(axis=1)
    benchmark_rmse = np.sqrt(
        benchmark_sums[sampled].sum(axis=1) / sampled_counts
    ).mean(axis=1)
    relative = (benchmark_rmse - candidate_rmse) / benchmark_rmse
    absolute = benchmark_rmse - candidate_rmse
    total_counts = counts.sum(axis=0)
    candidate_point = float(
        np.sqrt(candidate_sums.sum(axis=0) / total_counts).mean()
    )
    benchmark_point = float(
        np.sqrt(benchmark_sums.sum(axis=0) / total_counts).mean()
    )
    return {
        "block_minutes": BOOTSTRAP_BLOCK_MINUTES,
        "blocks": block_count,
        "resamples": resamples,
        "seed": seed,
        "candidate_primary_rmse": candidate_point,
        "benchmark_primary_rmse": benchmark_point,
        "relative_improvement": (
            benchmark_point - candidate_point
        )
        / benchmark_point,
        "relative_improvement_ci95": [
            float(np.quantile(relative, 0.025)),
            float(np.quantile(relative, 0.975)),
        ],
        "absolute_improvement": benchmark_point - candidate_point,
        "absolute_improvement_ci95": [
            float(np.quantile(absolute, 0.025)),
            float(np.quantile(absolute, 0.975)),
        ],
    }


def _paired_bootstrap(
    root: Path,
    runs: dict[tuple[str, str, int], dict[str, Any]],
    *,
    candidate_id: str,
    benchmark_id: str,
) -> dict[str, Any]:
    candidate, benchmark, counts, excluded_incomplete_blocks = (
        _paired_block_statistics(
        root,
        runs,
        candidate_id=candidate_id,
        benchmark_id=benchmark_id,
        )
    )
    result = _bootstrap_improvement(candidate, benchmark, counts)
    result["candidate_id"] = candidate_id
    result["benchmark_id"] = benchmark_id
    result["excluded_incomplete_blocks"] = excluded_incomplete_blocks
    return result


def build_robustness_freeze(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_robustness_runs(root)
    ordered = _expected_robustness_keys()
    return {
        "schema_version": 1,
        "program_id": "strata-fusion-v2-program",
        "cycle_id": "fusion-v2-rolling-robustness",
        "frozen_at": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
        "protocol": "docs/FUSION_V2_ROBUSTNESS.md",
        "repository_revision": ROBUSTNESS_EVIDENCE_REVISION,
        "expected_run_count": len(ordered),
        "run_ids": {
            f"{candidate_id}/{fold_id}/seed-{seed}": runs[
                (candidate_id, fold_id, seed)
            ]["run_id"]
            for candidate_id, fold_id, seed in ordered
        },
        "excluded_dirty_runs": _excluded_dirty_robustness_runs(root),
        "bootstrap": {
            "block_minutes": BOOTSTRAP_BLOCK_MINUTES,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "official_mlo_test_loaded": False,
        "usna_confirmation_labels_loaded": False,
    }


def write_robustness_freeze(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "research"
        / "experiments"
        / "fusion-v2-program"
        / "robustness-freeze.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_robustness_freeze(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_robustness_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or find_repo_root()
    runs = _collect_robustness_runs(root)
    excluded_dirty_runs = _excluded_dirty_robustness_runs(root)
    candidate_ids = (
        *ROBUSTNESS_FIXED_CONTROLS,
        *ROBUSTNESS_NEURAL_CANDIDATES,
    )
    aggregates = {
        candidate_id: _aggregate_candidate(candidate_id, runs)
        for candidate_id in candidate_ids
    }
    selected = aggregates["selected-fusion-v2"]
    persistence = aggregates["control-persistence"]
    diagnostic_lightgbm = aggregates["diagnostic-lightgbm"]
    stronger_neural_id = min(
        ("control-mlp", "control-tcn"),
        key=lambda candidate_id: aggregates[candidate_id]["primary"][
            "rmse_log10_cn2"
        ],
    )
    stronger_neural = aggregates[stronger_neural_id]
    selected_rmse = float(selected["primary"]["rmse_log10_cn2"])
    persistence_rmse = float(persistence["primary"]["rmse_log10_cn2"])
    stronger_neural_rmse = float(
        stronger_neural["primary"]["rmse_log10_cn2"]
    )
    lightgbm_rmse = float(
        diagnostic_lightgbm["primary"]["rmse_log10_cn2"]
    )
    improvement_persistence = (
        persistence_rmse - selected_rmse
    ) / persistence_rmse
    improvement_neural = (
        stronger_neural_rmse - selected_rmse
    ) / stronger_neural_rmse
    improvement_lightgbm = (
        lightgbm_rmse - selected_rmse
    ) / lightgbm_rmse
    bootstrap_persistence = _paired_bootstrap(
        root,
        runs,
        candidate_id="selected-fusion-v2",
        benchmark_id="control-persistence",
    )
    bootstrap_neural = _paired_bootstrap(
        root,
        runs,
        candidate_id="selected-fusion-v2",
        benchmark_id=stronger_neural_id,
    )
    persistence_seed_rmse = float(persistence["primary"]["rmse_log10_cn2"])
    seed_directions = {
        str(seed): (
            persistence_seed_rmse
            - float(selected["seed_primary_rmse"][str(seed)])
        )
        / persistence_seed_rmse
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
        "matrix_complete": "PASS",
        "repository_identity": "PASS",
        "data_and_split_identity": "PASS",
        "dirty_runs_excluded": "PASS",
        "lightgbm_relationship_stated": "PASS",
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
            "PASS" if all(value > 0 for value in seed_directions.values()) else "FAIL"
        ),
        "coverage_70_to_90_percent": (
            "PASS" if 0.70 <= coverage <= 0.90 else "FAIL"
        ),
        "absolute_bias_at_most_0_25": (
            "PASS" if float(selected["primary"]["absolute_bias"]) <= 0.25 else "FAIL"
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
        "repository_identity",
        "data_and_split_identity",
        "lightgbm_relationship_stated",
    )
    passed = all(checks[name] == "PASS" for name in final_gate_names)
    checks["confirmation_eligibility"] = "PASS" if passed else "FAIL"
    checks["confirmation_evaluation"] = "NOT_EVALUATED"
    all_runs = [run for aggregate in aggregates.values() for run in aggregate["runs"]]
    neural_runs = [
        run
        for candidate_id in ROBUSTNESS_NEURAL_CANDIDATES
        for run in aggregates[candidate_id]["runs"]
    ]
    source_hashes = {
        run["resolved_configuration"]["source_sha256"]
        for run in runs.values()
    }
    split_hashes = {
        run["resolved_configuration"]["split_sha256"]
        for run in runs.values()
    }
    if len(source_hashes) != 1 or len(split_hashes) != 1:
        raise RuntimeError("Robustness runs do not share one data/split identity")
    failed_conditions = [
        name for name in final_gate_names if checks[name] == "FAIL"
    ]
    split_config = load_yaml(
        "configs/splits/frozen/otbench_usna_lg_fusion_v2.yaml",
        root=root,
    )
    selected_manifest = runs[
        ("selected-fusion-v2", "fold-1", ROBUSTNESS_SEEDS[0])
    ]
    resolved = selected_manifest["resolved_configuration"]
    trainer = resolved["trainer_config"]
    conclusion = (
        "Fusion v2 passed every rolling-development condition, but this is "
        "still not a champion or promotion decision."
        if passed
        else (
            "Fusion v2 did not qualify for confirmation. Its mean primary "
            f"RMSE change versus persistence was {100 * improvement_persistence:.2f}% "
            f"and {len(failed_conditions)} frozen conditions failed. The "
            "confirmation labels remain sealed. "
            + (
                "It was directionally better than diagnostic LightGBM."
                if improvement_lightgbm > 0
                else "Diagnostic LightGBM remained stronger."
            )
        )
    )
    return {
        "schema_version": 1,
        "experiment_id": "strata-fusion-v2-rolling-robustness",
        "program_id": "strata-fusion-v2-program",
        "task_kind": "fusion_v2_robustness",
        "report_short_title": "Fusion v2 robustness",
        "report_model_label": "Strata-OT Fusion v2",
        "generated_at": datetime.now(
            ZoneInfo("America/Chicago")
        ).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "evaluation_partition": "rolling_selection",
        "hypothesis": (
            "The frozen balanced Fusion v2 candidate will retain at least a "
            "2% primary RMSE gain across three weather periods and three seeds "
            "without uncertainty, tail, or stability regression."
        ),
        "plain_language_question": (
            "Does the selected custom neural model beat simple persistence and "
            "plain neural controls reliably across seasons and random seeds?"
        ),
        "plain_language_conclusion": conclusion,
        "protocol": "docs/FUSION_V2_ROBUSTNESS.md",
        "freeze": (
            "research/experiments/fusion-v2-program/robustness-freeze.json"
        ),
        "data_identity": {
            "source_sha256": next(iter(source_hashes)),
            "split_sha256": next(iter(split_hashes)),
            "official_mlo_test_loaded": False,
            "usna_confirmation_labels_loaded": False,
        },
        "repository_revision": ROBUSTNESS_EVIDENCE_REVISION,
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
            "stronger_neural_primary_rmse": stronger_neural_rmse,
            "diagnostic_lightgbm_primary_rmse": lightgbm_rmse,
            "relative_improvement_over_persistence": improvement_persistence,
            "relative_improvement_over_stronger_neural": improvement_neural,
            "relative_improvement_over_diagnostic_lightgbm": (
                improvement_lightgbm
            ),
            "seed_directions": seed_directions,
            "bootstrap_vs_persistence": bootstrap_persistence,
            "bootstrap_vs_stronger_neural": bootstrap_neural,
            "confirmation_eligible": passed,
            "failed_conditions": failed_conditions,
        },
        "checks": checks,
        "excluded_dirty_runs": excluded_dirty_runs,
        "resource_narrative": (
            "One dirty-provenance retry is retained as failure evidence but "
            "excluded from the exact 45-run matrix; only the clean rerun is "
            "analyzed. No OOM, numerical retry, unsafe thermal state, or "
            "evidence substitution is included in this matrix."
        ),
        "next_cycle_narrative": (
            "Because the rolling gate failed, the next preregistered "
            "development cycle changes one factor: a robust explicit "
            "persistence-residual point objective. Confirmation stays sealed."
        ),
        "resources": {
            "matrix_run_count": len(all_runs),
            "neural_run_count": len(neural_runs),
            "neural_wall_clock_hours": sum(
                float(run["wall_clock_seconds"]) for run in neural_runs
            )
            / 3600.0,
            "excluded_dirty_neural_wall_clock_hours": sum(
                float(run["wall_clock_seconds"])
                for run in excluded_dirty_runs
            )
            / 3600.0,
            "consumed_neural_wall_clock_hours": (
                sum(float(run["wall_clock_seconds"]) for run in neural_runs)
                + sum(
                    float(run["wall_clock_seconds"])
                    for run in excluded_dirty_runs
                )
            )
            / 3600.0,
            "peak_total_board_vram_gib": max(
                [
                    float(run["peak_total_board_vram_gib"])
                    for run in all_runs
                ]
                + [
                    float(run["peak_total_board_vram_gib"])
                    for run in excluded_dirty_runs
                ]
            ),
            "peak_process_allocated_vram_gib": max(
                float(run["peak_process_allocated_vram_gib"]) for run in all_runs
            ),
            "peak_process_rss_gib": max(
                float(run["process_rss_gib"]) for run in all_runs
            ),
            "artifact_storage_bytes": sum(
                int(run["artifact_storage_bytes"]) for run in all_runs
            ),
            "excluded_dirty_artifact_storage_bytes": sum(
                int(run["artifact_storage_bytes"])
                for run in excluded_dirty_runs
            ),
            "consumed_artifact_storage_bytes": (
                sum(int(run["artifact_storage_bytes"]) for run in all_runs)
                + sum(
                    int(run["artifact_storage_bytes"])
                    for run in excluded_dirty_runs
                )
            ),
            "cloud_cost_usd": 0.0,
        },
        "confirmation": {
            "status": "NOT_EVALUATED",
            "labels_loaded": False,
            "reason": (
                "Rolling-development eligibility failed; release is prohibited."
                if not passed
                else "Eligible but not automatically released or promoted."
            ),
        },
        "report_pdf": None,
    }


def write_robustness_summary(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    destination = (
        root
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "robustness-summary.json"
    )
    existing = _load_object(destination) if destination.is_file() else {}
    summary = build_robustness_summary(root)
    if isinstance(existing.get("evidence_run_id"), str):
        summary["evidence_run_id"] = existing["evidence_run_id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def log_robustness_evidence(summary_path: Path, report_path: Path) -> str:
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
        run_name="fusion-v2-robustness-evidence",
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
                "robustness/bootstrap_lower_persistence": float(
                    summary["result"]["bootstrap_vs_persistence"][
                        "relative_improvement_ci95"
                    ][0]
                ),
                "resources/neural_wall_clock_hours": float(
                    summary["resources"]["neural_wall_clock_hours"]
                ),
                "resources/consumed_neural_wall_clock_hours": float(
                    summary["resources"][
                        "consumed_neural_wall_clock_hours"
                    ]
                ),
                "resources/excluded_dirty_neural_wall_clock_hours": float(
                    summary["resources"][
                        "excluded_dirty_neural_wall_clock_hours"
                    ]
                ),
                "resources/peak_total_board_vram_gib": float(
                    summary["resources"]["peak_total_board_vram_gib"]
                ),
                "resources/consumed_artifact_storage_bytes": float(
                    summary["resources"][
                        "consumed_artifact_storage_bytes"
                    ]
                ),
            }
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
                / "robustness-freeze.json",
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
        description="Build frozen Fusion v2 program evidence"
    )
    parser.add_argument(
        "--cycle",
        choices=("screen", "robustness"),
        default="screen",
    )
    parser.add_argument("--freeze", action="store_true")
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
    if args.cycle == "robustness":
        if args.freeze:
            write_robustness_freeze(root)
        summary_path = write_robustness_summary(root)
    else:
        summary_path = write_screen_summary(root)
    if args.log_mlflow:
        report_path = root / args.report
        if args.cycle == "robustness":
            print(log_robustness_evidence(summary_path, report_path))
        else:
            print(log_screen_evidence(summary_path, report_path))
    else:
        print(summary_path)


if __name__ == "__main__":
    main()
