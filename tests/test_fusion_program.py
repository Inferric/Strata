from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from strata_ot.evaluation.fusion_cycles import (
    POINT_WEIGHTS,
    _collect_point_loss_runs,
    _point_decision,
)
from strata_ot.evaluation.fusion_program import (
    ROBUSTNESS_EVIDENCE_REVISION,
    _bootstrap_improvement,
    _collect_robustness_runs,
    _complete_primary_block_ids,
    _excluded_dirty_robustness_runs,
    _expected_robustness_keys,
)
from strata_ot.evaluation.fusion_synthesis import build_program_summary
from strata_ot.training.fusion import (
    FusionLightningModule,
    _ensure_program_gpu_budget,
    _resolve_max_epochs,
    masked_tail_huber_loss,
)


def test_paired_block_bootstrap_resamples_time_not_seed_replicates() -> None:
    benchmark_sums = np.full((6, 3), 10.0)
    candidate_sums = np.full((6, 3), 6.4)
    counts = np.full((6, 3), 10.0)
    result = _bootstrap_improvement(
        candidate_sums,
        benchmark_sums,
        counts,
        resamples=200,
        seed=20260730,
    )
    assert result["blocks"] == 6
    assert result["relative_improvement"] == pytest.approx(0.2)
    assert result["relative_improvement_ci95"][0] > 0


def test_candidate_screen_epoch_cap_does_not_change_full_or_final_fit() -> None:
    candidate = {"screen_max_epochs": 9, "full_max_epochs": 10}
    trainer = {
        "screen_max_epochs": 3,
        "full_max_epochs": 12,
        "confirmation_max_epochs": 16,
    }
    assert _resolve_max_epochs(
        candidate,
        trainer,
        screen=True,
        final_fit=False,
    ) == 9
    assert _resolve_max_epochs(
        candidate,
        trainer,
        screen=False,
        final_fit=False,
    ) == 10
    assert _resolve_max_epochs(
        candidate,
        trainer,
        screen=False,
        final_fit=True,
    ) == 16


def test_paired_block_bootstrap_rejects_unpaired_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        _bootstrap_improvement(
            np.ones((2, 3)),
            np.ones((3, 3)),
            np.ones((2, 3)),
        )


def test_bootstrap_excludes_only_blocks_missing_a_primary_horizon() -> None:
    day = np.asarray([10, 10, 11, 11, 11, 12], dtype=np.int64)
    horizon = np.asarray([15, 30, 15, 30, 60, 60], dtype=np.int64)
    complete = _complete_primary_block_ids(day, horizon)
    np.testing.assert_array_equal(complete, np.asarray([11]))


class _FixedDistribution(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = batch["target"]
        location = torch.zeros_like(target)
        return {
            "location": location,
            "student_t_scale": torch.ones_like(target),
            "student_t_df": torch.full_like(target, 4.0),
            "quantiles": torch.stack(
                (location - 1.0, location, location + 1.0),
                dim=-1,
            ),
        }


def test_explicit_point_loss_changes_forecast_objective() -> None:
    batch = {"target": torch.tensor([0.5, -0.5])}
    unweighted = FusionLightningModule(
        _FixedDistribution(),
        learning_rate=3e-4,
        weight_decay=0.01,
        max_epochs=1,
        point_loss_weight=0.0,
    )._forecast_loss(batch, "test")
    weighted = FusionLightningModule(
        _FixedDistribution(),
        learning_rate=3e-4,
        weight_decay=0.01,
        max_epochs=1,
        point_loss_weight=1.0,
    )._forecast_loss(batch, "test")
    assert float(weighted) > float(unweighted)


def test_masked_tail_huber_changes_only_training_defined_tail() -> None:
    location = torch.zeros(4)
    target = torch.tensor([-1.0, 0.0, 0.9, 1.1])
    loss, fraction = masked_tail_huber_loss(
        location,
        target,
        threshold=0.8,
        beta=0.1,
    )
    expected = torch.nn.functional.smooth_l1_loss(
        location[2:],
        target[2:],
        beta=0.1,
        reduction="sum",
    ) / len(target)
    assert float(loss) == pytest.approx(float(expected))
    assert float(fraction) == pytest.approx(0.5)


def test_tail_huber_weight_changes_forecast_objective() -> None:
    batch = {"target": torch.tensor([0.5, 1.0])}
    control = FusionLightningModule(
        _FixedDistribution(),
        learning_rate=3e-4,
        weight_decay=0.01,
        max_epochs=1,
        tail_huber_weight=0.0,
        tail_huber_threshold=0.75,
    )._forecast_loss(batch, "test")
    weighted = FusionLightningModule(
        _FixedDistribution(),
        learning_rate=3e-4,
        weight_decay=0.01,
        max_epochs=1,
        tail_huber_weight=1.0,
        tail_huber_threshold=0.75,
    )._forecast_loss(batch, "test")
    assert float(weighted) > float(control)


def test_point_loss_screen_requires_point_and_constraint_gates() -> None:
    control = {
        "primary": {
            "rmse_log10_cn2": 0.30,
            "tail_mae_top_decile": 0.20,
            "crps_gaussian": 0.15,
        }
    }
    intervention = {
        "primary": {
            "rmse_log10_cn2": 0.29,
            "tail_mae_top_decile": 0.20,
            "crps_gaussian": 0.15,
            "interval_80_coverage": 0.80,
            "bias_log10_cn2": 0.01,
        }
    }
    passing = _point_decision(control, intervention)
    assert passing["advances"] is True
    intervention["primary"]["tail_mae_top_decile"] = 0.21
    failing = _point_decision(control, intervention)
    assert failing["advances"] is False
    assert failing["checks"]["tail_no_two_percent_regression"] == "FAIL"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_point_collection_keeps_screen_and_ignores_later_full_run(
    tmp_path: Path,
) -> None:
    metrics = {
        "rmse_log10_cn2": 0.3,
        "mae_log10_cn2": 0.2,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.2,
        "crps_gaussian": 0.15,
        "interval_80_coverage": 0.8,
    }
    for index, (candidate_id, weight) in enumerate(POINT_WEIGHTS.items()):
        artifact = (
            tmp_path / "artifacts" / "runs" / candidate_id / "predictions.npz"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence", encoding="utf-8")
        checkpoint = tmp_path / "checkpoints" / f"{candidate_id}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
        _write_json(
            artifact.parent / "run-manifest.json",
            {
                "run_id": f"screen-{index}",
                "candidate_id": candidate_id,
                "fold_id": "fold-1",
                "seed": 17,
                "stopped_for_budget": False,
                "by_horizon": {
                    str(horizon): metrics
                    for horizon in (5, 15, 30, 60)
                },
                "artifacts": {
                    "predictions": artifact.relative_to(tmp_path).as_posix()
                },
                "checkpoint": str(checkpoint),
                "resolved_configuration": {
                    "screen": True,
                    "candidate": {"point_loss_weight": weight},
                },
                "repository": {
                    "git_revision": "a" * 40,
                    "git_dirty": False,
                },
            },
        )
    _write_json(
        tmp_path
        / "artifacts"
        / "runs"
        / "later-full"
        / "run-manifest.json",
        {
            "run_id": "later-full",
            "candidate_id": "point-huber-0p25",
            "fold_id": "fold-1",
            "seed": 17,
            "resolved_configuration": {
                "screen": False,
                "candidate": {"point_loss_weight": 0.25},
            },
        },
    )
    collected = _collect_point_loss_runs(tmp_path)
    assert collected["point-huber-0p25"]["run_id"] == "screen-1"


def _synthesis_fixture(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "configs" / "experiments" / "fusion_v2_program.yaml",
        {
            "data": "configs/data/fusion.yaml",
            "model": "configs/model/fusion.yaml",
            "trainer": "configs/trainer/fusion.yaml",
            "seeds": [17, 41, 73],
            "rolling_folds": ["fold-1", "fold-2", "fold-3"],
            "primary_horizons_minutes": [15, 30, 60],
            "anchor_horizon_minutes": 5,
            "bootstrap": {
                "block_minutes": 1440,
                "resamples": 2000,
                "seed": 20260730,
            },
            "budget": {"max_total_gpu_hours": 8.0},
            "deadline": {
                "timezone": "America/Chicago",
                "no_new_work_after": "2026-07-31T06:06:00-05:00",
            },
        },
    )
    _write_json(
        tmp_path / "configs" / "data" / "fusion.yaml",
        {
            "label": {
                "name": "Cn2_3m",
                "representation": "log10_cn2_m_minus_2_over_3",
            },
            "features": {
                "raw_allowlist": ["T_3m", "P_3m"],
                "derived_allowlist": ["dew_point"],
            },
            "sampling": {
                "contexts": {
                    "short": {"rows": 6, "spacing_minutes": 5},
                    "medium": {"rows": 12, "spacing_minutes": 15},
                    "slow": {"rows": 24, "spacing_minutes": 60},
                },
                "maximum_source_gap_minutes": 2,
            },
        },
    )
    _write_json(
        tmp_path / "configs" / "model" / "fusion.yaml",
        {
            "name": "fusion",
            "hidden_dim": 192,
            "num_heads": 6,
            "num_experts": 4,
            "dropout": 0.1,
            "horizon_fourier_bands": 8,
            "parameter_budget_min": 2_000_000,
            "parameter_budget_max": 8_000_000,
        },
    )
    _write_json(
        tmp_path / "configs" / "trainer" / "fusion.yaml",
        {
            "precision": "bf16-mixed",
            "full_max_epochs": 12,
            "batch_size": 128,
            "gradient_accumulation": 2,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "early_stopping_patience": 4,
            "full_max_train_examples": 50_000,
            "full_max_evaluation_examples": 12_000,
            "full_max_seconds": 1800,
            "deterministic": True,
            "num_workers": 0,
        },
    )
    _write_json(
        tmp_path
        / "configs"
        / "splits"
        / "frozen"
        / "otbench_usna_lg_fusion_v2.yaml",
        {
            "id": "split",
            "strategy": "purged_rolling_origin",
            "minimum_boundary_purge_minutes": 2880,
            "folds": [],
            "confirmation": {
                "consumption_id": "sealed",
                "start": "2022-03-01",
                "end": "2022-04-30",
            },
        },
    )
    _write_json(
        tmp_path
        / "artifacts"
        / "experiments"
        / "mlo-weather-horizon-v1"
        / "summary.json",
        {
            "task_kind": "multi_horizon_forecast",
            "assessment_claim": {"status": "null_or_partial"},
            "plain_language_conclusion": "Cycle 0 was partial.",
            "total_neural_gpu_hours": 0.25,
            "peak_vram_gb": 1.0,
        },
    )
    program = (
        tmp_path
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
    )
    _write_json(
        program / "screen-summary.json",
        {
            "task_kind": "fusion_v2_screen",
            "status": "FAIL",
            "plain_language_conclusion": "Screen complete.",
            "checks": {"matrix_complete": "PASS"},
            "resources": {
                "neural_wall_clock_hours": 0.2,
                "peak_total_board_vram_gib": 2.0,
                "artifact_storage_bytes": 100,
            },
        },
    )
    primary = {
        "rmse_log10_cn2": 0.25,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.18,
        "crps_gaussian": 0.14,
        "interval_80_coverage": 0.8,
    }
    aggregate = {
        "primary": primary,
        "runs": [],
    }
    _write_json(
        program / "robustness-summary.json",
        {
            "task_kind": "fusion_v2_robustness",
            "status": "FAIL",
            "plain_language_conclusion": "Robustness gate failed.",
            "repository_revision": "a" * 40,
            "stronger_neural_control": "control-tcn",
            "aggregates": {
                "selected-fusion-v2": aggregate,
                "diagnostic-lightgbm": {
                    "primary": {**primary, "rmse_log10_cn2": 0.24},
                    "runs": [],
                },
            },
            "result": {
                "relative_improvement_over_persistence": 0.01,
            },
            "checks": {"matrix_complete": "PASS"},
            "data_identity": {
                "source_sha256": "b" * 64,
                "split_sha256": "c" * 64,
            },
            "confirmation": {
                "status": "NOT_EVALUATED",
                "labels_loaded": False,
            },
            "resources": {
                "neural_wall_clock_hours": 1.0,
                "peak_total_board_vram_gib": 3.0,
                "artifact_storage_bytes": 200,
            },
        },
    )
    _write_json(
        tmp_path
        / "research"
        / "experiments"
        / "strata-fusion-v2-program"
        / "ledger.json",
        {"events": [{"sequence": 0, "result": "PASS"}]},
    )


def test_program_synthesis_preserves_null_and_sealed_boundaries(
    tmp_path: Path,
) -> None:
    _synthesis_fixture(tmp_path)
    summary = build_program_summary(tmp_path)
    assert summary["status"] == "FAIL"
    assert summary["checks"]["confirmation_evaluation"] == "NOT_EVALUATED"
    assert summary["provenance"]["official_mlo_test_loaded"] is False
    assert summary["relationship_to_lightgbm"][
        "relative_improvement_over_lightgbm"
    ] < 0
    assert summary["resources"]["neural_wall_clock_hours"] == pytest.approx(1.2)
    assert summary["resources"][
        "total_evidence_neural_wall_clock_hours"
    ] == pytest.approx(1.45)
    assert summary["frozen_configuration"]["model"]["hidden_dim"] == 192
    assert summary["frozen_configuration"]["bootstrap"]["resamples"] == 2000


def test_program_synthesis_refuses_released_confirmation(tmp_path: Path) -> None:
    _synthesis_fixture(tmp_path)
    path = (
        tmp_path
        / "artifacts"
        / "experiments"
        / "strata-fusion-v2-program"
        / "robustness-summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confirmation"]["status"] = "PASS"
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match="refuses a released confirmation"):
        build_program_summary(tmp_path)


def test_robustness_collection_excludes_dirty_retry_but_keeps_clean_matrix(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    for index, (candidate_id, fold_id, seed) in enumerate(
        _expected_robustness_keys()
    ):
        _write_json(
            runs_root / f"clean-{index:02d}" / "run-manifest.json",
            {
                "run_id": f"clean-{index:02d}",
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "seed": seed,
                "repository": {
                    "git_revision": ROBUSTNESS_EVIDENCE_REVISION,
                    "git_dirty": False,
                    "working_tree_status": "",
                },
            },
        )
    candidate_id, fold_id, seed = _expected_robustness_keys()[0]
    _write_json(
        runs_root / "dirty-retry" / "run-manifest.json",
        {
            "run_id": "dirty-retry",
            "candidate_id": candidate_id,
            "fold_id": fold_id,
            "seed": seed,
            "repository": {
                "git_revision": ROBUSTNESS_EVIDENCE_REVISION,
                "git_dirty": True,
                "working_tree_status": " M docs/goals/auto-run-1.md",
            },
        },
    )
    collected = _collect_robustness_runs(tmp_path)
    excluded = _excluded_dirty_robustness_runs(tmp_path)
    assert len(collected) == 45
    assert excluded == [
        {
            "run_id": "dirty-retry",
            "candidate_id": candidate_id,
            "fold_id": fold_id,
            "seed": seed,
            "reason": "repository_worktree_dirty",
            "working_tree_status": " M docs/goals/auto-run-1.md",
            "wall_clock_seconds": 0.0,
            "peak_total_board_vram_gib": 0.0,
            "artifact_storage_bytes": 0,
        }
    ]


def test_program_gpu_budget_counts_all_neural_attempts_and_reserves_run(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "artifacts" / "runs"
    _write_json(
        runs / "fusion" / "run-manifest.json",
        {
            "candidate_id": "selected-fusion-v2",
            "kind": "neural",
            "metrics": {"wall_clock_seconds": 3600},
        },
    )
    _write_json(
        runs / "mlp" / "run-manifest.json",
        {
            "candidate_id": "control-mlp",
            "kind": "neural",
            "metrics": {"wall_clock_seconds": 7200},
        },
    )
    _write_json(
        runs / "baseline" / "run-manifest.json",
        {
            "candidate_id": "control-persistence",
            "kind": "baseline",
            "metrics": {"wall_clock_seconds": 9000},
        },
    )
    consumed = _ensure_program_gpu_budget(
        tmp_path,
        candidate_ids={"selected-fusion-v2", "control-mlp"},
        run_limit_seconds=1800,
        budget_hours=4.0,
    )
    assert consumed == 10_800
    with pytest.raises(RuntimeError, match="cannot admit another bounded run"):
        _ensure_program_gpu_budget(
            tmp_path,
            candidate_ids={"selected-fusion-v2", "control-mlp"},
            run_limit_seconds=1800,
            budget_hours=3.4,
        )
