from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from strata_ot.config import find_repo_root


def evaluate_gates(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks = dict(summary.get("checks", {}))
    required = [str(item) for item in gates.get("required_checks", [])]
    failures = [name for name in required if checks.get(name) is not True]
    runs = summary.get("runs", [])
    neural_runs = [run for run in runs if run.get("kind") == "neural"]
    promotion = gates["promotion"]
    minimum_seeds = int(promotion["minimum_seeds"])
    seeds_by_model: dict[str, set[int]] = defaultdict(set)
    metrics_by_model: dict[str, list[float]] = defaultdict(list)
    coverage_by_model: dict[str, list[float]] = defaultdict(list)
    bias_by_model: dict[str, list[float]] = defaultdict(list)
    for run in neural_runs:
        model = str(run.get("model", "unknown"))
        if run.get("seed") is not None:
            seeds_by_model[model].add(int(run["seed"]))
        rmse = run.get("metrics", {}).get("rmse_log10_cn2")
        if rmse is not None:
            metrics_by_model[model].append(float(rmse))
        coverage = run.get("metrics", {}).get("interval_80_coverage")
        if coverage is not None:
            coverage_by_model[model].append(float(coverage))
        bias = run.get("metrics", {}).get("bias_log10_cn2")
        if bias is not None:
            bias_by_model[model].append(float(bias))
    eligible_models = [
        model
        for model, seeds in seeds_by_model.items()
        if len(seeds) >= minimum_seeds and len(metrics_by_model[model]) >= minimum_seeds
    ]
    if not eligible_models:
        failures.append("minimum_seeds")
    baseline_runs = {
        str(run.get("name")): run
        for run in runs
        if run.get("kind") == "baseline"
    }
    best_model = min(
        eligible_models,
        key=lambda model: sum(metrics_by_model[model]) / len(metrics_by_model[model]),
        default=None,
    )
    best_mean_rmse = (
        sum(metrics_by_model[best_model]) / len(metrics_by_model[best_model])
        if best_model is not None
        else None
    )
    if promotion.get("require_improvement_over_climatology"):
        climatology = baseline_runs.get("climatology", {}).get("metrics", {}).get(
            "rmse_log10_cn2"
        )
        if (
            climatology is None
            or best_mean_rmse is None
            or best_mean_rmse >= float(climatology)
        ):
            failures.append("improvement_over_climatology")
    if promotion.get("require_improvement_over_persistence"):
        persistence = baseline_runs.get("persistence", {}).get("metrics", {}).get(
            "rmse_log10_cn2"
        )
        if (
            persistence is None
            or best_mean_rmse is None
            or best_mean_rmse >= float(persistence)
        ):
            failures.append("improvement_over_persistence")
    if best_model is not None:
        values = metrics_by_model[best_model]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        relative_std = variance**0.5 / max(abs(mean), 1e-12)
        if relative_std > float(promotion["max_relative_seed_std"]):
            failures.append("seed_stability")
        mean_coverage = (
            sum(coverage_by_model[best_model]) / len(coverage_by_model[best_model])
            if coverage_by_model[best_model]
            else None
        )
        minimum_coverage = promotion.get("min_interval_80_coverage")
        maximum_coverage = promotion.get("max_interval_80_coverage")
        if (
            minimum_coverage is not None
            and (mean_coverage is None or mean_coverage < float(minimum_coverage))
        ):
            failures.append("interval_undercoverage")
        if (
            maximum_coverage is not None
            and (mean_coverage is None or mean_coverage > float(maximum_coverage))
        ):
            failures.append("interval_overcoverage")
        mean_bias = (
            sum(bias_by_model[best_model]) / len(bias_by_model[best_model])
            if bias_by_model[best_model]
            else None
        )
        maximum_bias = promotion.get("max_absolute_bias_log10_cn2")
        if (
            maximum_bias is not None
            and (mean_bias is None or abs(mean_bias) > float(maximum_bias))
        ):
            failures.append("absolute_bias")
    else:
        relative_std = None
        mean_coverage = None
        mean_bias = None
    if any(
        float(run.get("metrics", {}).get("peak_vram_gb", 0))
        > float(promotion["max_peak_vram_gb"])
        for run in neural_runs
    ):
        failures.append("peak_vram")
    total_gpu_hours = sum(
        float(run.get("metrics", {}).get("wall_clock_seconds", 0)) / 3600
        for run in neural_runs
    )
    if total_gpu_hours > float(promotion["max_local_gpu_hours"]):
        failures.append("local_gpu_hours")
    result = {
        "gate_id": gates["id"],
        "passed": not failures,
        "failures": sorted(set(failures)),
        "evaluated_run_count": len(runs),
        "best_eligible_model": best_model,
        "best_mean_rmse_log10_cn2": best_mean_rmse,
        "best_relative_seed_std": relative_std,
        "best_mean_interval_80_coverage": mean_coverage,
        "best_mean_bias_log10_cn2": mean_bias,
        "total_neural_gpu_hours": total_gpu_hours,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate immutable Strata-OT gates")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--gates", default="configs/evaluation/gates/public_v1.yaml")
    args = parser.parse_args()
    root = find_repo_root()
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    gates_path = Path(args.gates)
    if not gates_path.is_absolute():
        gates_path = root / gates_path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    result = evaluate_gates(summary, gates)
    output = summary_path.with_name("gate-result.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
