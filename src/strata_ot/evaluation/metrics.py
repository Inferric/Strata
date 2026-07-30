from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr


def regression_metrics(
    target: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    location = np.asarray(location, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(location)
    target = target[valid]
    location = location[valid]
    if not len(target):
        raise ValueError("No finite values available for evaluation")
    error = location - target
    metrics = {
        "mae_log10_cn2": float(np.mean(np.abs(error))),
        "rmse_log10_cn2": float(np.sqrt(np.mean(np.square(error)))),
        "bias_log10_cn2": float(np.mean(error)),
    }
    prediction_variance = float(np.var(location))
    if prediction_variance > 1e-12:
        calibration_slope = float(np.cov(location, target, ddof=0)[0, 1]) / prediction_variance
        metrics["calibration_slope"] = calibration_slope
        metrics["calibration_intercept"] = float(
            np.mean(target) - calibration_slope * np.mean(location)
        )
    target_standard_deviation = float(np.std(target))
    prediction_standard_deviation = float(np.std(location))
    if target_standard_deviation > 1e-12 and prediction_standard_deviation > 1e-12:
        metrics["correlation"] = float(np.corrcoef(target, location)[0, 1])
    threshold = np.quantile(target, 0.90)
    tail = target >= threshold
    metrics["tail_mae_top_decile"] = float(np.mean(np.abs(error[tail])))

    if scale is not None:
        sigma = np.asarray(scale, dtype=np.float64)[valid]
        sigma = np.clip(sigma, 1e-6, None)
        z = (target - location) / sigma
        metrics["gaussian_nll"] = float(
            np.mean(np.log(sigma) + 0.5 * np.square(z) + 0.5 * math.log(2 * math.pi))
        )
        # Closed-form CRPS for a Gaussian predictive distribution.
        phi = np.exp(-0.5 * np.square(z)) / math.sqrt(2 * math.pi)
        crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi))
        metrics["crps_gaussian"] = float(np.mean(crps))
        half_width = 1.2815515655446004 * sigma
        covered = np.abs(error) <= half_width
        metrics["interval_80_coverage"] = float(np.mean(covered))
        metrics["interval_80_width"] = float(np.mean(2 * half_width))
        metrics["predictive_std_mean"] = float(np.mean(sigma))
    return metrics
