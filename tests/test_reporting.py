from pathlib import Path

from strata_ot.reporting.render import _environment, latex_escape, latex_number


def test_latex_escape() -> None:
    escaped = latex_escape("Cn2_run & 50%")
    assert r"\_" in escaped
    assert r"\&" in escaped
    assert r"\%" in escaped


def test_latex_number_handles_missing_values() -> None:
    assert latex_number(None) == "--"
    assert latex_number(0.125) == "0.1250"


def test_experiment_template_renders_complete_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    template = _environment(root / "reports" / "templates").get_template(
        "experiment_report.tex.j2"
    )
    neural_metrics = {
        "rmse_log10_cn2": 0.4,
        "mae_log10_cn2": 0.3,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.2,
        "gaussian_nll": 1.0,
        "crps_gaussian": 0.2,
        "interval_80_coverage": 0.8,
        "interval_80_width": 1.0,
        "predictive_std_mean": 0.4,
        "calibration_slope": 1.0,
        "wall_clock_seconds": 60,
        "peak_vram_gb": 1.0,
        "inference_latency_ms_per_sample": 0.1,
        "throughput_samples_per_second": 10_000,
    }
    neural_run = {
        "name": "candidate-17",
        "run_id": "run",
        "kind": "neural",
        "model": "strata_ot_surface",
        "seed": 17,
        "parameters": 3_000_000,
        "metrics": neural_metrics,
    }
    rendered = template.render(
        summary={
            "experiment_id": "template-test",
            "generated_at": "2026-07-29T00:00:00Z",
            "code_revision": "test",
            "evaluation_gate_id": "gate",
            "repository_identity": {"source_digest_sha256": "a" * 64},
            "source_snapshot": {"sha256": "b" * 64},
            "data_identity": {
                "dataset_manifest_sha256": "c" * 64,
                "split_sha256": "d" * 64,
            },
            "environment": {
                "device": "cpu",
                "gpu": {"total_memory_gb": 0},
                "python": "3.12",
                "torch": "2",
                "lightning": "2",
                "cuda_runtime": None,
            },
            "hypothesis": "A complete report template renders from recorded evidence.",
            "dataset_id": "dataset",
            "split_id": "split",
            "target_transform": "log10",
            "data_coverage": {
                "coverage": {
                    "rows": 3,
                    "source_rows": 3,
                    "temporal_start": "start",
                    "temporal_end": "end",
                },
                "geometry": {
                    "site": "site",
                    "latitude": 0,
                    "longitude": 0,
                    "observation_height_m": 15,
                },
                "qc": {"upstream_commit": "e" * 40},
            },
            "data_qc_details": {
                "manifest_verified": True,
                "label": {
                    "detail": "direct observation",
                    "name": "Cn2",
                    "representation": "log10",
                    "raw_units": "m^(-2/3)",
                },
                "coverage": {
                    partition: {
                        "start": "start",
                        "end": "end",
                        "maximum_gap_minutes": 5,
                    }
                    for partition in ("train", "validation", "test")
                },
                "upstream_task": {
                    "features": 87,
                    "rows_after_dropna": {
                        "train": 100,
                        "validation": 40,
                        "test": 40,
                    },
                },
                "split": {
                    "post_purge_rows": {
                        "train": 90,
                        "validation": 30,
                        "test": 30,
                    },
                    "context_32_sequence_counts": {
                        "train": 59,
                        "validation": 1,
                        "test": 1,
                    },
                    "sealed_sha256": "f" * 64,
                },
            },
            "runs": [neural_run],
            "best_overall_run": neural_run,
            "best_neural_run": neural_run,
            "model_aggregates": [
                {
                    "model": "mlp",
                    "seeds": 1,
                    "mean_rmse_log10_cn2": 0.4,
                    "std_rmse_log10_cn2": None,
                    "relative_seed_std": None,
                },
                {
                    "model": "strata_ot_surface",
                    "seeds": 1,
                    "mean_rmse_log10_cn2": 0.4,
                    "std_rmse_log10_cn2": None,
                    "relative_seed_std": None,
                }
            ],
            "strata_interval_coverages": [0.8, 0.8],
            "peak_vram_gb": 1.0,
            "total_neural_gpu_hours": 1 / 60,
            "best_checkpoint": {
                "display_path": "checkpoint.ckpt",
                "run_id": "run",
                "seed": 17,
                "sha256": "0" * 64,
                "bytes": 1024,
            },
            "report_figures": [],
            "resolved_configuration": {"data": {}, "model": {}, "trainer": {}},
            "checks": {"latex_pdf_compiled": True},
            "gate_result": {
                "passed": False,
                "failures": ["minimum_seeds"],
                "best_eligible_model": "strata_ot_surface",
                "best_mean_rmse_log10_cn2": 0.4,
            },
            "failure_records": [],
            "recommended_next_experiment": "Run a bounded confirmatory experiment.",
        }
    )
    assert "minimum\\_seeds" in rendered
    assert "Exact resolved configuration" in rendered


def test_completed_report_templates_do_not_render_open_gate_states() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "experiment_report.tex.j2",
        "forecast_report.tex.j2",
        "horizon_report.tex.j2",
        "fusion_screen_report.tex.j2",
        "fusion_robustness_report.tex.j2",
        "fusion_point_loss_report.tex.j2",
        "fusion_program_report.tex.j2",
    ):
        source = (root / "reports" / "templates" / name).read_text(
            encoding="utf-8"
        )
        assert r"\newcommand{\open}" not in source
        assert r"\open" not in source
        assert "OPEN" not in source


def test_fusion_robustness_template_renders_terminal_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    template = _environment(root / "reports" / "templates").get_template(
        "fusion_robustness_report.tex.j2"
    )
    primary = {
        "rmse_log10_cn2": 0.25,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.17,
        "crps_gaussian": 0.14,
        "interval_80_coverage": 0.8,
    }
    run = {
        "run_id": "run",
        "fold_id": "fold-1",
        "seed": 17,
        "primary": primary,
        "wall_clock_seconds": 60,
    }
    aggregate = {
        "candidate_id": "selected-fusion-v2",
        "runs": [run],
        "primary": primary,
        "relative_run_rmse_std": 0.02,
        "by_horizon": {
            "5": {
                "rmse_log10_cn2": 0.2,
                "mae_log10_cn2": 0.15,
                "bias_log10_cn2": 0.01,
                "tail_mae_top_decile": 0.16,
                "crps_gaussian": 0.12,
                "student_t_nll": -0.2,
                "interval_80_coverage": 0.8,
                "interval_80_width": 0.4,
            }
        },
        "operational": {
            "selective_80_rmse_log10_cn2": 0.2,
            "ood_top_quintile_rmse_log10_cn2": 0.3,
            "ood_score_mean": 1.0,
            "inference_latency_ms_per_sample": 0.1,
            "throughput_samples_per_second": 10_000,
        },
        "diagnostics": {
            "15": {
                "residual_mean": 0.01,
                "residual_std": 0.02,
                "scale_weights_mean": [0.5, 0.3, 0.2],
                "physics_token_norm_mean": 2.0,
            }
        },
    }
    rendered = template.render(
            summary={
                "generated_at": "2026-07-30T20:00:00-05:00",
                "report_short_title": "Fusion v2 robustness",
                "report_model_label": "Strata-OT Fusion v2",
            "plain_language_question": "Does the model remain stable?",
            "plain_language_conclusion": "No; one frozen condition failed.",
            "hypothesis": "The candidate will improve every rolling period.",
            "status": "FAIL",
            "resources": {
                "matrix_run_count": 45,
                "neural_wall_clock_hours": 1.0,
                "excluded_dirty_neural_wall_clock_hours": 0.1,
                "consumed_neural_wall_clock_hours": 1.1,
                "peak_total_board_vram_gib": 8.0,
                "peak_process_allocated_vram_gib": 1.0,
                "peak_process_rss_gib": 3.0,
                "artifact_storage_bytes": 1000,
                "excluded_dirty_artifact_storage_bytes": 100,
                "consumed_artifact_storage_bytes": 1100,
                "cloud_cost_usd": 0.0,
            },
            "result": {
                "selected_primary_rmse": 0.25,
                "persistence_primary_rmse": 0.26,
                "stronger_neural_primary_rmse": 0.255,
                "diagnostic_lightgbm_primary_rmse": 0.245,
                "relative_improvement_over_persistence": 0.03,
                "relative_improvement_over_stronger_neural": 0.01,
                "relative_improvement_over_diagnostic_lightgbm": -0.02,
                "bootstrap_vs_persistence": {
                    "relative_improvement_ci95": [-0.01, 0.05],
                    "blocks": 90,
                    "excluded_incomplete_blocks": 3,
                },
                "bootstrap_vs_stronger_neural": {
                    "relative_improvement_ci95": [-0.02, 0.03]
                },
                "seed_directions": {"17": 0.03, "41": -0.01, "73": 0.02},
            },
            "stronger_neural_control": "control-tcn",
            "protocol": "docs/protocol.md",
            "freeze": "research/freeze.json",
            "data_identity": {
                "source_sha256": "a" * 64,
                "split_sha256": "b" * 64,
            },
            "repository_revision": "c" * 40,
            "rolling_split": {
                "minimum_boundary_purge_minutes": 2880,
                "folds": [
                    {
                        "id": "fold-1",
                        "train": {
                            "start": "2020-01-01",
                            "end": "2020-03-15",
                        },
                        "calibration": {"start": "2020-03-18"},
                        "selection": {
                            "start": "2020-04-01",
                            "end": "2020-04-30",
                        },
                    }
                ],
            },
                "frozen_configuration": {
                    "candidate": {"id": "selected-fusion-v2"},
                    "parameters": 3_611_530,
                    "feature_names": ["T_3m", "P_3m"],
                "trainer": {
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
                },
            },
                "aggregates": {
                candidate: (
                    {**aggregate, "operational": {}}
                    if candidate == "control-persistence"
                    else aggregate
                )
                for candidate in (
                    "control-persistence",
                    "diagnostic-lightgbm",
                    "control-mlp",
                    "control-tcn",
                    "control-horizon-v1",
                    "selected-fusion-v2",
                    )
                },
                "selected_aggregate": aggregate,
            "checks": {
                "matrix_complete": "PASS",
                "confirmation_evaluation": "NOT_EVALUATED",
            },
                "excluded_dirty_runs": [],
                "resource_narrative": "No unsafe resource event occurred.",
                "next_cycle_narrative": "Confirmation remains sealed.",
            }
        )
    assert "NOT\\_EVALUATED" in rendered
    assert "No; one frozen condition failed." in rendered
    assert "TimeXer" in rendered


def test_fusion_point_loss_template_renders_terminal_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    template = _environment(root / "reports" / "templates").get_template(
        "fusion_point_loss_report.tex.j2"
    )
    primary = {
        "rmse_log10_cn2": 0.29,
        "mae_log10_cn2": 0.21,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.19,
        "crps_gaussian": 0.14,
        "interval_80_coverage": 0.8,
        "interval_80_width": 0.4,
        "student_t_nll": -0.2,
    }
    run = {
        "candidate_id": "point-huber-0p00",
        "run_id": "run",
        "weight": 0.0,
        "primary": primary,
        "operational": {
            "selective_80_rmse_log10_cn2": 0.2,
            "ood_top_quintile_rmse_log10_cn2": 0.3,
            "ood_score_mean": 1.0,
            "inference_latency_ms_per_sample": 0.1,
            "throughput_samples_per_second": 10_000,
        },
        "by_horizon": {"15": primary, "30": primary, "60": primary},
        "wall_clock_seconds": 60,
    }
    rendered = template.render(
        summary={
            "generated_at": "2026-07-30T20:00:00-05:00",
            "plain_language_question": "Does point loss improve location?",
            "plain_language_conclusion": (
                "No explicit point-loss weight passed the frozen screen."
            ),
            "hypothesis": "Point pressure will improve the residual.",
            "status": "FAIL",
            "protocol": "docs/FUSION_V21_POINT_LOSS.md",
            "repository_revision": "a" * 40,
            "data_identity": {
                "source_sha256": "b" * 64,
                "split_sha256": "c" * 64,
            },
            "runs": [run],
            "decisions": {
                "point-huber-0p25": {
                    "relative_improvement_over_control": 0.01,
                    "checks": {
                        "two_percent_rmse_improvement": "FAIL",
                        "tail_no_two_percent_regression": "PASS",
                        "crps_no_two_percent_regression": "PASS",
                        "coverage_70_to_90_percent": "PASS",
                    },
                    "advances": False,
                }
            },
            "checks": {
                "four_arms_completed": "PASS",
                "family_advancement": "FAIL",
                "confirmation_evaluation": "NOT_EVALUATED",
            },
            "resources": {
                "neural_wall_clock_hours": 0.1,
                "peak_total_board_vram_gib": 8.0,
                "peak_process_allocated_vram_gib": 1.0,
                "peak_process_rss_gib": 3.0,
                "artifact_storage_bytes": 1000,
                "cloud_cost_usd": 0.0,
            },
        }
    )
    assert "NOT\\_EVALUATED" in rendered
    assert "No explicit point-loss weight passed" in rendered
    assert "Smooth L1" in rendered


def test_fusion_program_template_renders_complete_claim_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    template = _environment(root / "reports" / "templates").get_template(
        "fusion_program_report.tex.j2"
    )
    primary = {
        "rmse_log10_cn2": 0.25,
        "bias_log10_cn2": 0.01,
        "tail_mae_top_decile": 0.17,
        "crps_gaussian": 0.14,
        "interval_80_coverage": 0.8,
    }
    rendered = template.render(
        summary={
            "generated_at": "2026-07-30T21:00:00-05:00",
            "plain_language_question": "What did the program establish?",
            "plain_language_conclusion": "No champion was promoted.",
            "protocol": "docs/FUSION_V2_PROGRAM.md",
            "provenance": {
                "source_sha256": "a" * 64,
                "split_sha256": "b" * 64,
                "repository_revision": "c" * 40,
            },
                "rolling_split": {
                    "minimum_boundary_purge_minutes": 2880,
                "folds": [
                    {
                        "id": "fold-1",
                        "train": {
                            "start": "2020-01-01",
                            "end": "2020-03-15",
                        },
                        "calibration": {"start": "2020-03-18"},
                        "selection": {
                            "start": "2020-04-01",
                            "end": "2020-04-30",
                        },
                        }
                    ],
                },
                "frozen_configuration": {
                    "seeds": [17, 41, 73],
                    "rolling_folds": ["fold-1", "fold-2", "fold-3"],
                    "primary_horizons_minutes": [15, 30, 60],
                    "anchor_horizon_minutes": 5,
                    "bootstrap": {
                        "block_minutes": 1440,
                        "resamples": 2000,
                        "seed": 20260730,
                    },
                    "data": {
                        "raw_features": ["T_3m", "P_3m"],
                        "contexts": {
                            "short": {
                                "rows": 6,
                                "spacing_minutes": 5,
                            },
                            "medium": {
                                "rows": 12,
                                "spacing_minutes": 15,
                            },
                            "slow": {
                                "rows": 24,
                                "spacing_minutes": 60,
                            },
                        },
                    },
                    "model": {
                        "hidden_dim": 192,
                        "num_heads": 6,
                        "num_experts": 4,
                        "dropout": 0.1,
                        "horizon_fourier_bands": 8,
                        "distribution_head": "Student-t plus quantiles",
                        "prediction_anchor": "persistence residual",
                    },
                    "trainer": {
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
                    },
                    "deadline": {
                        "no_new_work_after": "2026-07-31T06:06:00-05:00",
                        "timezone": "America/Chicago",
                    },
                },
                "resources": {
                "neural_wall_clock_hours": 1.0,
                "prior_cycle0_neural_wall_clock_hours": 0.25,
                "total_evidence_neural_wall_clock_hours": 1.25,
                "peak_total_board_vram_gib": 8.0,
                "artifact_storage_bytes": 1000,
                "cloud_cost_usd": 0.0,
            },
            "cycles": [
                {
                    "cycle_id": "cycle-0",
                    "role": "prior",
                    "status": "FAIL",
                    "conclusion": "Partial evidence only.",
                }
            ],
            "screen_evidence": {
                "run_count": 1,
                "runs": [
                    {
                        "candidate_id": "fusion-concat",
                        "category": "fusion",
                        "family": "fusion_v2",
                        "primary_rmse": 0.30,
                        "primary_tail_mae": 0.18,
                        "primary_bias": 0.01,
                        "primary_coverage_80": 0.8,
                    }
                ],
                "closed_hypothesis_families": {
                    "context": "Three frozen arms did not advance."
                },
            },
            "ledger_events": [
                {
                    "sequence": 0,
                    "result": "PASS",
                    "kind": "machinery_repair",
                    "notes": "Terminal gate states and resource accounting fixed.",
                }
            ],
            "aggregates": {
                    "selected-fusion-v2": {
                        "runs": [{"run_id": "run"}],
                        "primary": primary,
                        "by_horizon": {
                            "5": {
                                "rmse_log10_cn2": 0.2,
                                "mae_log10_cn2": 0.15,
                                "bias_log10_cn2": 0.01,
                                "tail_mae_top_decile": 0.16,
                                "crps_gaussian": 0.12,
                                "interval_80_coverage": 0.8,
                            }
                        },
                        "diagnostics": {
                            "5": {
                                "residual_mean": 0.01,
                                "residual_std": 0.1,
                                "scale_weights_mean": [0.4, 0.35, 0.25],
                                "expert_weights_mean": [
                                    0.25,
                                    0.25,
                                    0.25,
                                    0.25,
                                ],
                                "weather_attention_mean": 0.1,
                                "physics_token_norm_mean": 1.0,
                                "modulation_norm_mean": 0.2,
                            }
                        },
                        "operational": {
                        "selective_80_rmse_log10_cn2": 0.2,
                        "ood_top_quintile_rmse_log10_cn2": 0.3,
                        "ood_score_mean": 1.0,
                        "inference_latency_ms_per_sample": 0.1,
                        "throughput_samples_per_second": 10_000,
                    },
                }
            },
            "result": {
                "selected_primary_rmse": 0.25,
                "persistence_primary_rmse": 0.26,
                "stronger_neural_primary_rmse": 0.255,
                "relative_improvement_over_persistence": 0.03,
                "relative_improvement_over_stronger_neural": 0.01,
                "bootstrap_vs_persistence": {
                    "relative_improvement_ci95": [-0.01, 0.05]
                },
                "bootstrap_vs_stronger_neural": {
                    "relative_improvement_ci95": [-0.02, 0.03]
                },
            },
            "relationship_to_lightgbm": {
                "interpretation": "LightGBM remained stronger.",
                "relative_improvement_over_lightgbm": -0.02,
                "claim_boundary": "LightGBM is diagnostic only.",
            },
            "checks": {
                "custom_candidate_success": "FAIL",
                "confirmation_evaluation": "NOT_EVALUATED",
            },
            "limitations": ["Confirmation labels remained sealed."],
            "next_phase": "Preregister a new architecture family.",
        }
    )
    assert "NOT\\_EVALUATED" in rendered
    assert "No champion was promoted." in rendered
    assert "LightGBM is diagnostic only." in rendered
