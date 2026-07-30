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
    ):
        source = (root / "reports" / "templates" / name).read_text(
            encoding="utf-8"
        )
        assert r"\newcommand{\open}" not in source
        assert r"\open" not in source
        assert "OPEN" not in source
