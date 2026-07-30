from __future__ import annotations

import numpy as np
import pytest

from strata_ot.evaluation.evaluate import evaluate_gates, gate_status
from strata_ot.evaluation.metrics import regression_metrics


def test_probabilistic_metrics_are_finite() -> None:
    target = np.array([-15.0, -14.8, -14.5, -13.9])
    prediction = target + np.array([0.1, -0.1, 0.2, -0.2])
    metrics = regression_metrics(target, prediction, np.full(4, 0.3))
    assert 0 <= metrics["interval_80_coverage"] <= 1
    assert metrics["crps_gaussian"] >= 0
    assert metrics["rmse_log10_cn2"] > 0


def test_gate_requires_two_seeds_for_same_model() -> None:
    summary = {
        "checks": {"required": True},
        "runs": [
            {
                "name": "climatology",
                "kind": "baseline",
                "metrics": {"rmse_log10_cn2": 1.0},
            },
            {
                "name": "surface-17",
                "kind": "neural",
                "model": "surface",
                "seed": 17,
                "metrics": {
                    "rmse_log10_cn2": 0.7,
                    "peak_vram_gb": 2,
                    "wall_clock_seconds": 10,
                },
            },
            {
                "name": "mlp-41",
                "kind": "neural",
                "model": "mlp",
                "seed": 41,
                "metrics": {
                    "rmse_log10_cn2": 0.8,
                    "peak_vram_gb": 1,
                    "wall_clock_seconds": 10,
                },
            },
        ],
    }
    gates = {
        "id": "gate",
        "required_checks": ["required"],
        "promotion": {
            "minimum_seeds": 2,
            "require_improvement_over_climatology": True,
            "require_improvement_over_persistence": False,
            "max_relative_seed_std": 0.15,
            "max_peak_vram_gb": 15.5,
            "max_local_gpu_hours": 4,
        },
    }
    result = evaluate_gates(summary, gates)
    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert result["condition_statuses"]["minimum_seeds"] == "FAIL"
    assert "minimum_seeds" in result["failures"]
    assert "OPEN" not in set(result["condition_statuses"].values())


def test_gate_status_has_only_terminal_or_unevaluated_states() -> None:
    assert gate_status(True) == "PASS"
    assert gate_status(False) == "FAIL"
    assert gate_status(None) == "NOT_EVALUATED"
    assert gate_status(False, evaluated=False) == "NOT_EVALUATED"
    with pytest.raises(ValueError, match="Invalid gate status"):
        gate_status("OPEN")
