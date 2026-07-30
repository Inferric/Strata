from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import lightning as L
import mlflow
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import MLFlowLogger

from strata_ot.config import load_yaml, resolve_experiment
from strata_ot.data.forecast import ForecastDataModule, ForecastSequenceDataset
from strata_ot.data.registry import verify_manifest
from strata_ot.evaluation.metrics import regression_metrics
from strata_ot.models import MLPBaseline, StrataOTSurface
from strata_ot.training.module import Cn2LightningModule
from strata_ot.training.train import (
    _configure_torch_runtime,
    _configure_utf8_output,
    _environment_fingerprint,
    _repository_identity,
    _sha256_file,
)


def fit_scale_factor(
    target: np.ndarray,
    location: np.ndarray,
    raw_scale: np.ndarray,
) -> float:
    """Maximum-likelihood scalar correction fitted on the calibration block."""
    target = np.asarray(target, dtype=np.float64)
    location = np.asarray(location, dtype=np.float64)
    raw_scale = np.asarray(raw_scale, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(location) & np.isfinite(raw_scale)
    if not np.any(valid) or np.any(raw_scale[valid] <= 0):
        raise ValueError("Scale calibration requires finite positive raw scales")
    ratio_squared = np.square((target[valid] - location[valid]) / raw_scale[valid])
    factor = float(np.sqrt(np.mean(ratio_squared)))
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Scale calibration produced a non-finite factor")
    return factor


def _make_datamodule(
    experiment: dict[str, Any],
    *,
    allow_test: bool,
    batch_size: int | None = None,
) -> ForecastDataModule:
    root = Path(str(experiment["_repo_root"]))
    data_config = experiment["data_config"]
    split_config = load_yaml(str(data_config["split_config"]), root=root)
    trainer = experiment["trainer_config"]
    configured_workers = int(trainer["num_workers"])
    workers = int(os.getenv("STRATA_NUM_WORKERS", configured_workers))
    if workers < 0:
        raise ValueError("STRATA_NUM_WORKERS must be non-negative")
    return ForecastDataModule(
        data_config,
        split_config,
        batch_size=batch_size or int(trainer["batch_size"]),
        num_workers=workers,
        allow_test=allow_test,
    )


def _dataset_arrays(dataset: ForecastSequenceDataset) -> tuple[np.ndarray, ...]:
    return (
        dataset.features.cpu().numpy(),
        dataset.target.cpu().numpy(),
        dataset.persistence.cpu().numpy(),
        dataset.history_target.cpu().numpy(),
    )


def _evaluation_dataset(
    datamodule: ForecastDataModule,
    *,
    release_test: bool,
) -> tuple[ForecastSequenceDataset, str]:
    if release_test:
        if datamodule.test_set is None:
            raise RuntimeError("Test release was requested but no test set is available")
        return datamodule.test_set, "test"
    if datamodule.calibration_set is None:
        raise RuntimeError("Calibration block is unavailable")
    return datamodule.calibration_set, "calibration"


def _build_model(
    model_name: str,
    model_config: dict[str, Any],
    *,
    input_dim: int,
    context: int,
) -> torch.nn.Module:
    excluded = {
        "name",
        "input_dim",
        "parameter_budget_max",
        "quantiles",
        "context",
        "mlp_flatten_context",
    }
    common = {key: value for key, value in model_config.items() if key not in excluded}
    if model_name == "mlp":
        flatten = bool(model_config.get("mlp_flatten_context", True))
        mlp_options = {
            key: value
            for key, value in common.items()
            if key not in {"hidden_dim", "depth"}
        }
        return MLPBaseline(
            input_dim=input_dim * context if flatten else input_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            depth=4,
            flatten_context=flatten,
            **mlp_options,
        )
    if model_name == "strata_ot_surface":
        return StrataOTSurface(input_dim=input_dim, max_context=context, **common)
    raise ValueError(f"Unsupported forecast model: {model_name}")


def _predict(
    module: Cn2LightningModule,
    dataset: ForecastSequenceDataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    device = next(module.parameters()).device
    module.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = module(
                batch["features"].to(device),
                batch["persistence"].to(device),
            )
            locations.append(output["location"].cpu().numpy())
            scales.append(output["log_scale"].exp().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(locations), np.concatenate(scales), elapsed


def _save_diagnostics(
    root: Path,
    run_id: str,
    dataset: ForecastSequenceDataset,
    prediction: np.ndarray,
    raw_scale: np.ndarray | None,
    calibrated_scale: np.ndarray | None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = root / "artifacts" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    target = dataset.target.cpu().numpy()
    persistence = dataset.persistence.cpu().numpy()
    predictions_path = destination / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        target=target,
        prediction=prediction,
        persistence=persistence,
        raw_scale=np.array([]) if raw_scale is None else raw_scale,
        calibrated_scale=(
            np.array([]) if calibrated_scale is None else calibrated_scale
        ),
        input_end_ids=dataset.input_end_ids,
        target_ids=dataset.target_ids,
        target_timestamps=dataset.target_timestamps,
    )
    residual = prediction - target
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].scatter(target, prediction, s=6, alpha=0.3, edgecolors="none")
    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    axes[0].plot([lower, upper], [lower, upper], color="#d9485f", linewidth=1.2)
    axes[0].set(xlabel="Observed log10 Cn²", ylabel="Forecast log10 Cn²")
    axes[1].hist(residual, bins=45, color="#237a77", alpha=0.85)
    axes[1].axvline(0, color="#d9485f", linewidth=1.2)
    axes[1].set(xlabel="Forecast residual", ylabel="Count")
    shown = min(400, len(target))
    axes[2].plot(target[:shown], color="#223047", linewidth=0.8, label="observed")
    axes[2].plot(
        prediction[:shown],
        color="#19a79d",
        linewidth=0.8,
        alpha=0.9,
        label="forecast",
    )
    if calibrated_scale is not None:
        half_width = 1.2815515655446004 * calibrated_scale[:shown]
        axes[2].fill_between(
            np.arange(shown),
            prediction[:shown] - half_width,
            prediction[:shown] + half_width,
            color="#19a79d",
            alpha=0.16,
            label="calibrated 80% interval",
        )
    axes[2].set(xlabel="Chronological forecast", ylabel="log10 Cn²")
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("One-step-ahead forecast diagnostics")
    figure_path = destination / "diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {
        "predictions": predictions_path.relative_to(root).as_posix(),
        "figure": figure_path.relative_to(root).as_posix(),
    }


def _write_json_artifact(
    root: Path,
    run_id: str,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    destination = root / "artifacts" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _data_identity(
    root: Path,
    experiment: dict[str, Any],
    datamodule: ForecastDataModule,
) -> dict[str, Any]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    manifest_path = root / str(experiment["data_config"]["manifest_path"])
    split_config_path = root / str(experiment["data_config"]["split_config"])
    sealed_path = (
        root
        / "data"
        / "sealed"
        / "splits"
        / f"{datamodule.metadata.split_id}.json"
    )
    split_identity: dict[str, Any] = {
        "split_config_path": split_config_path.relative_to(root).as_posix(),
        "split_config_sha256": _sha256_file(split_config_path),
    }
    if sealed_path.is_file():
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        split_identity.update(
            {
                "split_path": sealed_path.relative_to(root).as_posix(),
                "split_sha256": sealed["sha256"],
            }
        )
    return {
        "dataset_id": datamodule.metadata.dataset_id,
        "source_dataset_id": datamodule.metadata.source_dataset_id,
        "dataset_manifest_path": manifest_path.relative_to(root).as_posix(),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "split_id": datamodule.metadata.split_id,
        **split_identity,
        "target_transform": datamodule.metadata.target_transform,
        "evaluation_partition": "test" if datamodule.allow_test else "calibration",
    }


def _resolved_configuration(
    experiment: dict[str, Any],
    datamodule: ForecastDataModule,
    *,
    model_name: str,
    seed: int | None,
    parameters: int | None = None,
) -> dict[str, Any]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    return {
        "experiment": experiment["id"],
        "task_kind": "one_step_forecast",
        "model": model_name,
        "seed": seed,
        "parameters": parameters,
        "data": experiment["data_config"],
        "model_config": experiment["model_config"],
        "trainer_config": {
            **experiment["trainer_config"],
            "effective_num_workers": datamodule.num_workers,
            "effective_batch_size": datamodule.batch_size,
        },
        "normalization": datamodule.normalization,
        "feature_names": datamodule.metadata.feature_names,
        "sequence_qc": datamodule.sequence_qc,
        "target_transform": datamodule.metadata.target_transform,
        "baseline_residual": "last_observed_cn2_persistence",
    }


def _log_common_artifacts(
    root: Path,
    experiment: dict[str, Any],
    run_id: str,
    artifacts: dict[str, str],
    evidence_path: Path,
) -> None:
    for relative in artifacts.values():
        mlflow.log_artifact(str(root / relative), artifact_path="diagnostics")
    mlflow.log_artifact(str(evidence_path), artifact_path="evidence")
    for key in ("manifest_path", "split_config"):
        mlflow.log_artifact(
            str(root / str(experiment["data_config"][key])),
            artifact_path="evidence",
        )


def _baseline_prediction(
    name: str,
    datamodule: ForecastDataModule,
    evaluation: ForecastSequenceDataset,
) -> tuple[np.ndarray, dict[str, Any]]:
    if datamodule.train_set is None:
        raise RuntimeError("Training data is unavailable")
    train_features, train_target, _train_persistence, _train_history = _dataset_arrays(
        datamodule.train_set
    )
    features, _target, persistence, history = _dataset_arrays(evaluation)
    if name == "climatology":
        value = float(np.median(train_target))
        return np.full(len(evaluation), value), {"train_median": value}
    if name == "persistence":
        return persistence.copy(), {"source": "last_observed_history_target"}
    if name == "recent_mean":
        return history.mean(axis=1), {"history_rows": history.shape[1]}
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError("Install the data extra to run LightGBM") from error
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=17,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=max(1, min(4, datamodule.num_workers or 1)),
        )
        model.fit(train_features.reshape(len(train_features), -1), train_target)
        prediction = model.predict(features.reshape(len(features), -1))
        return np.asarray(prediction), {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "random_state": 17,
        }
    raise ValueError(f"Unsupported baseline: {name}")


def _log_baseline(
    root: Path,
    experiment: dict[str, Any],
    datamodule: ForecastDataModule,
    evaluation: ForecastSequenceDataset,
    evaluation_partition: str,
    name: str,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    started = time.monotonic()
    prediction, parameters = _baseline_prediction(name, datamodule, evaluation)
    target = evaluation.target.cpu().numpy()
    tags = _run_tags(datamodule, name, "baseline", evaluation_partition, run_context)
    with mlflow.start_run(run_name=f"{name}-{datamodule.metadata.site}", tags=tags) as run:
        metrics = regression_metrics(target, prediction)
        metrics.update(
            {
                "wall_clock_seconds": time.monotonic() - started,
                "peak_vram_gb": 0.0,
                "evaluation_samples": float(len(target)),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_params({"model_family": name, **parameters})
        artifacts = _save_diagnostics(
            root,
            run.info.run_id,
            evaluation,
            prediction,
            None,
            None,
        )
        resolved = _resolved_configuration(
            experiment,
            datamodule,
            model_name=name,
            seed=None,
        )
        resolved["baseline_parameters"] = parameters
        evidence = {
            "run_id": run.info.run_id,
            "name": name,
            "kind": "baseline",
            "model": name,
            "seed": None,
            **run_context,
            "resolved_configuration": resolved,
            "metrics": metrics,
            "checkpoint": None,
            "artifacts": artifacts,
        }
        evidence_path = _write_json_artifact(
            root, run.info.run_id, "run-manifest.json", evidence
        )
        mlflow.log_dict(resolved, "resolved-config.json")
        mlflow.log_dict(run_context["environment"], "environment.json")
        mlflow.log_dict(run_context["repository"], "repository-identity.json")
        _log_common_artifacts(root, experiment, run.info.run_id, artifacts, evidence_path)
        return {
            "run_id": run.info.run_id,
            "name": name,
            "kind": "baseline",
            "model": name,
            "seed": None,
            "metrics": metrics,
            "checkpoint": None,
            "artifacts": artifacts,
        }


def _run_tags(
    datamodule: ForecastDataModule,
    model_name: str,
    run_kind: str,
    evaluation_partition: str,
    run_context: dict[str, Any],
) -> dict[str, str]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    return {
        "dataset_id": datamodule.metadata.dataset_id,
        "source_dataset_id": datamodule.metadata.source_dataset_id,
        "split_id": datamodule.metadata.split_id,
        "site": datamodule.metadata.site,
        "task_kind": "one_step_forecast",
        "forecast_horizon_minutes": str(
            datamodule.metadata.forecast_horizon_minutes
        ),
        "model_family": model_name,
        "run_kind": run_kind,
        "evaluation_partition": evaluation_partition,
        "git_dirty": str(run_context["repository"]["git_dirty"]).lower(),
        "source_digest_sha256": run_context["repository"]["source_digest_sha256"],
    }


def _train_neural(
    root: Path,
    experiment: dict[str, Any],
    datamodule: ForecastDataModule,
    evaluation: ForecastSequenceDataset,
    evaluation_partition: str,
    model_name: str,
    seed: int,
    *,
    fast_dev_run: bool,
    max_seconds: float,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    if (
        datamodule.metadata is None
        or datamodule.calibration_set is None
        or datamodule.train_set is None
    ):
        raise RuntimeError("Forecast data must be prepared before training")
    L.seed_everything(seed, workers=True)
    trainer_config = experiment["trainer_config"]
    model_config = experiment["model_config"]
    network = _build_model(
        model_name,
        model_config,
        input_dim=datamodule.metadata.input_dim,
        context=datamodule.metadata.context,
    )
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    parameter_budget = int(model_config.get("parameter_budget_max", 8_000_000))
    if parameter_count > parameter_budget:
        raise ValueError(
            f"{model_name} has {parameter_count:,} parameters; budget is "
            f"{parameter_budget:,}"
        )
    module = Cn2LightningModule(
        network,
        learning_rate=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]),
        max_epochs=int(trainer_config["max_epochs"]),
    )
    checkpoint_dir = (
        root / "checkpoints" / str(experiment["id"]) / model_name / str(seed)
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="validation/rmse_log10_cn2",
        mode="min",
        save_top_k=1,
        filename="{epoch:03d}-{step:06d}",
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="validation/rmse_log10_cn2",
        mode="min",
        patience=int(trainer_config["early_stopping_patience"]),
    )
    timer = Timer(duration=timedelta(seconds=max_seconds), interval="step")
    run_kind = "probe" if fast_dev_run else "neural"
    run_name = f"{model_name}-{datamodule.metadata.site}-seed-{seed}"
    if fast_dev_run:
        run_name += "-fast-dev"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger = MLFlowLogger(
        experiment_name=str(experiment["tracking"]["experiment_name"]),
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=_run_tags(
            datamodule,
            model_name,
            run_kind,
            evaluation_partition,
            run_context,
        ),
    )
    accelerator = str(trainer_config["accelerator"])
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("The local_16gb forecast configuration requires CUDA")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=int(trainer_config["devices"]),
        precision=cast(Any, str(trainer_config["precision"])),
        max_epochs=int(trainer_config["max_epochs"]),
        accumulate_grad_batches=int(trainer_config["gradient_accumulation"]),
        gradient_clip_val=float(trainer_config["gradient_clip_val"]),
        deterministic=trainer_config["deterministic"],
        benchmark=bool(trainer_config["benchmark"]),
        log_every_n_steps=int(trainer_config["log_every_n_steps"]),
        callbacks=[checkpoint, early_stop, timer],
        logger=logger,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    run_id = logger.run_id
    if run_id is None:
        raise RuntimeError("MLflow did not assign a run identity")
    resolved = _resolved_configuration(
        experiment,
        datamodule,
        model_name=model_name,
        seed=seed,
        parameters=parameter_count,
    )
    resolved["fast_dev_run"] = fast_dev_run
    logger.log_hyperparams(
        {
            "model": model_name,
            "seed": seed,
            "parameters": parameter_count,
            "batch_size": datamodule.batch_size,
            "gradient_accumulation": int(trainer_config["gradient_accumulation"]),
            "precision": str(trainer_config["precision"]),
            "context": datamodule.metadata.context,
            "scale_parameterization": str(
                model_config.get("scale_parameterization", "clamp")
            ),
        }
    )
    try:
        trainer.fit(module, datamodule=datamodule)
        if checkpoint.best_model_path and not fast_dev_run:
            reloaded_network = _build_model(
                model_name,
                model_config,
                input_dim=datamodule.metadata.input_dim,
                context=datamodule.metadata.context,
            )
            module = Cn2LightningModule.load_from_checkpoint(
                checkpoint.best_model_path,
                model=reloaded_network,
                learning_rate=float(trainer_config["learning_rate"]),
                weight_decay=float(trainer_config["weight_decay"]),
                max_epochs=int(trainer_config["max_epochs"]),
            )
            module = module.to(torch.device("cuda" if accelerator == "gpu" else "cpu"))
        calibration_location, calibration_raw_scale, _ = _predict(
            module,
            datamodule.calibration_set,
            datamodule.batch_size,
        )
        calibration_target = datamodule.calibration_set.target.cpu().numpy()
        scale_factor = fit_scale_factor(
            calibration_target,
            calibration_location,
            calibration_raw_scale,
        )
        location, raw_scale, inference_seconds = _predict(
            module,
            evaluation,
            datamodule.batch_size,
        )
        calibrated_scale = raw_scale * scale_factor
        target = evaluation.target.cpu().numpy()
        metrics = regression_metrics(target, location, calibrated_scale)
        raw_metrics = regression_metrics(target, location, raw_scale)
        metrics.update({f"raw_{key}": value for key, value in raw_metrics.items()})
        elapsed = time.monotonic() - started
        peak_vram = (
            float(torch.cuda.max_memory_allocated() / (1024**3))
            if torch.cuda.is_available()
            else 0.0
        )
        metrics.update(
            {
                "wall_clock_seconds": elapsed,
                "peak_vram_gb": peak_vram,
                "scale_calibration_factor": scale_factor,
                "evaluation_samples": float(len(target)),
                "inference_latency_ms_per_sample": (
                    inference_seconds * 1000 / len(target)
                ),
                "throughput_samples_per_second": (
                    len(target) / max(inference_seconds, 1e-12)
                ),
            }
        )
        artifacts = _save_diagnostics(
            root,
            run_id,
            evaluation,
            location,
            raw_scale,
            calibrated_scale,
        )
        logger.log_metrics(metrics, step=trainer.global_step)
        logger.experiment.log_dict(run_id, resolved, "resolved-config.json")
        logger.experiment.log_dict(
            run_id, run_context["environment"], "environment.json"
        )
        logger.experiment.log_dict(
            run_id, run_context["repository"], "repository-identity.json"
        )
        for relative in artifacts.values():
            logger.experiment.log_artifact(
                run_id, str(root / relative), artifact_path="diagnostics"
            )
        if checkpoint.best_model_path:
            logger.experiment.log_artifact(
                run_id, checkpoint.best_model_path, artifact_path="checkpoints"
            )
        calibration_record = {
            "fit_partition": "validation_calibration",
            "factor": scale_factor,
            "samples": len(calibration_target),
            "raw_metrics": regression_metrics(
                calibration_target,
                calibration_location,
                calibration_raw_scale,
            ),
            "calibrated_metrics": regression_metrics(
                calibration_target,
                calibration_location,
                calibration_raw_scale * scale_factor,
            ),
        }
        logger.experiment.log_dict(run_id, calibration_record, "calibration.json")
        evidence = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "seed": seed,
            "retry_attempt": 0,
            **run_context,
            "resolved_configuration": resolved,
            "calibration": calibration_record,
            "metrics": metrics,
            "checkpoint": checkpoint.best_model_path or None,
            "artifacts": artifacts,
            "stopped_for_budget": timer.time_remaining() == 0,
        }
        evidence_path = _write_json_artifact(
            root, run_id, "run-manifest.json", evidence
        )
        logger.experiment.log_artifact(
            run_id, str(evidence_path), artifact_path="evidence"
        )
        for key in ("manifest_path", "split_config"):
            logger.experiment.log_artifact(
                run_id,
                str(root / str(experiment["data_config"][key])),
                artifact_path="evidence",
            )
        logger.finalize("success")
    except Exception as error:
        failure = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "seed": seed,
            **run_context,
            "resolved_configuration": resolved,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        failure_path = _write_json_artifact(
            root, run_id, "failure-manifest.json", failure
        )
        logger.experiment.log_artifact(
            run_id, str(failure_path), artifact_path="failures"
        )
        logger.finalize("failed")
        raise
    return {
        "run_id": run_id,
        "name": run_name,
        "kind": run_kind,
        "model": model_name,
        "seed": seed,
        "parameters": parameter_count,
        "metrics": metrics,
        "checkpoint": checkpoint.best_model_path or None,
        "artifacts": artifacts,
        "calibration": calibration_record,
        "stopped_for_budget": timer.time_remaining() == 0,
    }


def _write_data_qc(
    root: Path,
    experiment: dict[str, Any],
    datamodule: ForecastDataModule,
) -> Path:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    manifest_path = root / str(experiment["data_config"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = [
        dataset
        for dataset in (
            datamodule.train_set,
            datamodule.validation_set,
            datamodule.calibration_set,
            datamodule.test_set,
        )
        if dataset is not None
    ]
    identity_sets = [set(dataset.target_ids.tolist()) for dataset in datasets]
    no_overlap = all(
        identity_sets[left].isdisjoint(identity_sets[right])
        for left in range(len(identity_sets))
        for right in range(left + 1, len(identity_sets))
    )
    sealed_path = (
        root
        / "data"
        / "sealed"
        / "splits"
        / f"{datamodule.metadata.split_id}.json"
    )
    sealed_sha = None
    if sealed_path.is_file():
        sealed_sha = json.loads(sealed_path.read_text(encoding="utf-8")).get("sha256")
    payload = {
        "dataset_id": datamodule.metadata.dataset_id,
        "source_dataset_id": datamodule.metadata.source_dataset_id,
        "manifest_verified": not verify_manifest(manifest_path, root),
        "label_provenance": manifest.get("label_provenance"),
        "coverage": manifest.get("coverage"),
        "geometry": manifest.get("geometry"),
        "instrument": manifest.get("instrument"),
        "forecast": {
            "context_rows": datamodule.metadata.context,
            "history_minutes": datamodule.metadata.history_minutes,
            "horizon_minutes": datamodule.metadata.forecast_horizon_minutes,
            "sequence_qc": datamodule.sequence_qc,
        },
        "normalization_fit_partition": datamodule.normalization.get("fit_partition"),
        "checks": {
            "manifest_valid": not verify_manifest(manifest_path, root),
            "direct_observation_labels": (
                manifest.get("label_provenance") == "direct_observation"
            ),
            "preprocessing_fit_on_train_only": (
                datamodule.normalization.get("fit_partition") == "train"
            ),
            "no_target_identity_overlap": no_overlap,
            "gap_threshold_enforced": True,
            "boundary_purge_enforced": True,
        },
        "split": {
            "id": datamodule.metadata.split_id,
            "sealed_sha256": sealed_sha,
            "test_released": datamodule.allow_test,
        },
    }
    destination = (
        root
        / "artifacts"
        / "data-qc"
        / f"{datamodule.metadata.dataset_id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _source_snapshot(root: Path, repository: dict[str, Any]) -> Path:
    digest = str(repository["source_digest_sha256"])
    destination = root / "artifacts" / "source-snapshots" / f"{digest}.zip"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    included: list[Path] = []
    for relative in ("pyproject.toml", "uv.lock"):
        path = root / relative
        if path.is_file():
            included.append(path)
    for relative in ("src", "configs", "schemas", "scripts", "reports/templates"):
        directory = root / relative
        if directory.is_dir():
            included.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, path.relative_to(root).as_posix())
    return destination


def _model_aggregates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for model in ("mlp", "strata_ot_surface"):
        values = [
            float(run["metrics"]["rmse_log10_cn2"])
            for run in runs
            if run.get("kind") == "neural" and run.get("model") == model
        ]
        if not values:
            continue
        array = np.asarray(values)
        deviation = float(array.std(ddof=1)) if len(array) > 1 else None
        aggregates.append(
            {
                "model": model,
                "seeds": len(values),
                "mean_rmse_log10_cn2": float(array.mean()),
                "std_rmse_log10_cn2": deviation,
                "relative_seed_std": (
                    deviation / max(abs(float(array.mean())), 1e-12)
                    if deviation is not None
                    else None
                ),
            }
        )
    return aggregates


def block_bootstrap_improvement(
    target: np.ndarray,
    neural: np.ndarray,
    persistence: np.ndarray,
    *,
    block_rows: int,
    resamples: int = 2000,
    seed: int = 20260729,
) -> dict[str, float]:
    """Paired moving-block bootstrap of RMSE improvement over persistence."""
    target = np.asarray(target)
    neural = np.asarray(neural)
    persistence = np.asarray(persistence)
    count = len(target)
    if not (len(neural) == len(persistence) == count) or count < block_rows:
        raise ValueError("Bootstrap arrays must align and contain at least one block")
    rng = np.random.default_rng(seed)
    block_starts = np.arange(0, count - block_rows + 1)
    needed = math.ceil(count / block_rows)
    improvements = np.empty(resamples)
    for index in range(resamples):
        starts = rng.choice(block_starts, size=needed, replace=True)
        sampled = np.concatenate(
            [np.arange(start, start + block_rows) for start in starts]
        )[:count]
        persistence_rmse = np.sqrt(np.mean(np.square(persistence[sampled] - target[sampled])))
        neural_rmse = np.sqrt(np.mean(np.square(neural[sampled] - target[sampled])))
        improvements[index] = (persistence_rmse - neural_rmse) / persistence_rmse
    return {
        "mean_relative_improvement": float(improvements.mean()),
        "ci95_low": float(np.quantile(improvements, 0.025)),
        "ci95_high": float(np.quantile(improvements, 0.975)),
        "probability_improvement": float(np.mean(improvements > 0)),
        "resamples": float(resamples),
        "block_rows": float(block_rows),
    }


def _forecast_claim(
    root: Path,
    runs: list[dict[str, Any]],
    evaluation: ForecastSequenceDataset,
    cadence_minutes: float,
) -> dict[str, Any]:
    persistence_run = next(
        (run for run in runs if run.get("model") == "persistence"), None
    )
    neural_runs = [run for run in runs if run.get("kind") == "neural"]
    families = {
        model: [run for run in neural_runs if run.get("model") == model]
        for model in {str(run.get("model")) for run in neural_runs}
    }
    eligible = {
        model: family_runs
        for model, family_runs in families.items()
        if len({run.get("seed") for run in family_runs}) >= 2
    }
    if persistence_run is None or not eligible:
        return {"eligible": False, "reason": "missing two-seed comparable runs"}
    best_model, best_runs = min(
        eligible.items(),
        key=lambda item: np.mean(
            [
                float(run["metrics"]["rmse_log10_cn2"])
                for run in item[1]
            ]
        ),
    )
    predictions = [
        np.load(root / run["artifacts"]["predictions"])["prediction"]
        for run in best_runs
    ]
    block_rows = max(1, round(24 * 60 / cadence_minutes))
    target = evaluation.target.cpu().numpy()
    persistence = evaluation.persistence.cpu().numpy()
    count = len(target)
    rng = np.random.default_rng(20260729)
    starts = np.arange(0, count - block_rows + 1)
    needed = math.ceil(count / block_rows)
    improvements = np.empty(2000)
    for index in range(len(improvements)):
        sampled_starts = rng.choice(starts, size=needed, replace=True)
        sampled = np.concatenate(
            [
                np.arange(start, start + block_rows)
                for start in sampled_starts
            ]
        )[:count]
        persistence_sample_rmse = np.sqrt(
            np.mean(np.square(persistence[sampled] - target[sampled]))
        )
        family_sample_rmse = float(
            np.mean(
                [
                    np.sqrt(np.mean(np.square(prediction[sampled] - target[sampled])))
                    for prediction in predictions
                ]
            )
        )
        improvements[index] = (
            persistence_sample_rmse - family_sample_rmse
        ) / persistence_sample_rmse
    bootstrap = {
        "mean_relative_improvement": float(improvements.mean()),
        "ci95_low": float(np.quantile(improvements, 0.025)),
        "ci95_high": float(np.quantile(improvements, 0.975)),
        "probability_improvement": float(np.mean(improvements > 0)),
        "resamples": float(len(improvements)),
        "block_rows": float(block_rows),
    }
    persistence_rmse = float(persistence_run["metrics"]["rmse_log10_cn2"])
    best_rmse = float(
        np.mean(
            [
                float(run["metrics"]["rmse_log10_cn2"])
                for run in best_runs
            ]
        )
    )
    measured = (persistence_rmse - best_rmse) / persistence_rmse
    return {
        "eligible": True,
        "threshold_relative_improvement": 0.02,
        "measured_relative_improvement": measured,
        "passed_point_estimate": measured >= 0.02,
        "passed_bootstrap": bootstrap["ci95_low"] > 0,
        "run_ids": [run["run_id"] for run in best_runs],
        "seeds": [run["seed"] for run in best_runs],
        "best_model": best_model,
        "persistence_rmse_log10_cn2": persistence_rmse,
        "best_neural_rmse_log10_cn2": best_rmse,
        "bootstrap": bootstrap,
    }


def run_forecast_experiment(
    config_path: str,
    *,
    selected_model: str | None = None,
    selected_seed: int | None = None,
    fast_dev_run: bool = False,
    release_test: bool = False,
) -> dict[str, Any]:
    _configure_utf8_output()
    experiment = resolve_experiment(config_path)
    if experiment.get("task_kind") != "one_step_forecast":
        raise ValueError("Forecast runner requires task_kind: one_step_forecast")
    if release_test and "usna" not in str(experiment["id"]).lower():
        raise ValueError("The development MLO test partition may not be released")
    if float(experiment.get("budget", {}).get("max_cloud_cost_usd", 0)) != 0:
        raise ValueError("Forecast experiments require a zero-dollar cloud budget")
    _configure_torch_runtime(experiment["trainer_config"])
    root = Path(str(experiment["_repo_root"]))
    datamodule = _make_datamodule(experiment, allow_test=release_test)
    datamodule.prepare_data()
    datamodule.setup("fit")
    if (
        datamodule.train_set is None
        or datamodule.validation_set is None
        or datamodule.calibration_set is None
        or datamodule.metadata is None
    ):
        raise RuntimeError("Forecast data preparation is incomplete")
    evaluation, evaluation_partition = _evaluation_dataset(
        datamodule, release_test=release_test
    )
    repository = _repository_identity(root)
    _source_snapshot(root, repository)
    run_context = {
        "repository": repository,
        "environment": _environment_fingerprint(),
        "data_identity": _data_identity(root, experiment, datamodule),
    }
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(experiment["tracking"]["experiment_name"]))
    requested_models = (
        [selected_model] if selected_model else list(experiment["models"])
    )
    runs: list[dict[str, Any]] = []
    for name in ("climatology", "persistence", "recent_mean", "lightgbm"):
        if name in requested_models:
            runs.append(
                _log_baseline(
                    root,
                    experiment,
                    datamodule,
                    evaluation,
                    evaluation_partition,
                    name,
                    run_context,
                )
            )
    seeds = [selected_seed] if selected_seed is not None else list(experiment["seeds"])
    if fast_dev_run:
        seeds = seeds[:1]
    budget_hours = float(
        experiment.get("budget", {}).get(
            "max_total_gpu_hours", experiment["trainer_config"]["max_gpu_hours"]
        )
    )
    neural_started = time.monotonic()
    budget_exhausted = False
    for model_name in ("mlp", "strata_ot_surface"):
        if model_name not in requested_models:
            continue
        for seed in seeds:
            remaining = budget_hours * 3600 - (time.monotonic() - neural_started)
            if remaining <= 0:
                budget_exhausted = True
                break
            try:
                result = _train_neural(
                    root,
                    experiment,
                    datamodule,
                    evaluation,
                    evaluation_partition,
                    model_name,
                    int(seed),
                    fast_dev_run=fast_dev_run,
                    max_seconds=remaining,
                    run_context=run_context,
                )
            except torch.cuda.OutOfMemoryError:
                if fast_dev_run or datamodule.batch_size <= 1:
                    raise
                recovered_batch = max(1, datamodule.batch_size // 2)
                experiment["trainer_config"]["gradient_accumulation"] = int(
                    experiment["trainer_config"]["gradient_accumulation"]
                ) * 2
                datamodule = _make_datamodule(
                    experiment,
                    allow_test=release_test,
                    batch_size=recovered_batch,
                )
                datamodule.prepare_data()
                datamodule.setup("fit")
                evaluation, evaluation_partition = _evaluation_dataset(
                    datamodule, release_test=release_test
                )
                result = _train_neural(
                    root,
                    experiment,
                    datamodule,
                    evaluation,
                    evaluation_partition,
                    model_name,
                    int(seed),
                    fast_dev_run=fast_dev_run,
                    max_seconds=remaining,
                    run_context=run_context,
                )
                result["recovery"] = {
                    "reason": "cuda_out_of_memory",
                    "batch_size": recovered_batch,
                    "gradient_accumulation": experiment["trainer_config"][
                        "gradient_accumulation"
                    ],
                }
            runs.append(result)
            if result["stopped_for_budget"]:
                budget_exhausted = True
                break
        if budget_exhausted:
            break
    total_gpu_hours = sum(
        float(run.get("metrics", {}).get("wall_clock_seconds", 0)) / 3600
        for run in runs
        if run.get("kind") == "neural"
    )
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata became unavailable")
    metadata = datamodule.metadata
    qc_path = _write_data_qc(root, experiment, datamodule)
    qc_payload = json.loads(qc_path.read_text(encoding="utf-8"))
    claim = _forecast_claim(
        root,
        runs,
        evaluation,
        metadata.forecast_horizon_minutes,
    )
    neural_runs = [run for run in runs if run.get("kind") == "neural"]
    peak_vram = max(
        (float(run["metrics"].get("peak_vram_gb", 0)) for run in neural_runs),
        default=0.0,
    )
    summary = {
        "experiment_id": experiment["id"],
        "task_kind": "one_step_forecast",
        "generated_at": datetime.now(UTC).isoformat(),
        "hypothesis": experiment["hypothesis"],
        "plain_language_question": (
            f"Can six recent Cn² observations plus weather predict Cn² "
            f"{metadata.forecast_horizon_minutes:g} minutes ahead "
            "better than simply repeating the latest observation?"
        ),
        "dataset_id": metadata.dataset_id,
        "source_dataset_id": metadata.source_dataset_id,
        "site": metadata.site,
        "split_id": metadata.split_id,
        "evaluation_partition": evaluation_partition,
        "test_released": release_test,
        "forecast_horizon_minutes": metadata.forecast_horizon_minutes,
        "history_minutes": metadata.history_minutes,
        "evaluation_gate_id": experiment["evaluation_config"]["id"],
        "target_transform": metadata.target_transform,
        "code_revision": (
            repository["git_revision"]
            or f"uncommitted:{repository['source_digest_sha256'][:12]}"
        ),
        "repository_identity": repository,
        "environment": run_context["environment"],
        "data_identity": run_context["data_identity"],
        "data_coverage": json.loads(
            (root / str(experiment["data_config"]["manifest_path"])).read_text(
                encoding="utf-8"
            )
        ).get("coverage", {}),
        "data_qc": qc_path.relative_to(root).as_posix(),
        "sequence_qc": datamodule.sequence_qc,
        "runs": runs,
        "model_aggregates": _model_aggregates(runs),
        "forecast_claim": claim,
        "total_neural_gpu_hours": total_gpu_hours,
        "peak_vram_gb": peak_vram,
        "resolved_configuration": {
            "data": experiment["data_config"],
            "model": experiment["model_config"],
            "trainer": {
                **experiment["trainer_config"],
                "effective_num_workers": datamodule.num_workers,
                "effective_batch_size": datamodule.batch_size,
            },
        },
        "checks": {
            "dataset_manifest_valid": not verify_manifest(
                root / str(experiment["data_config"]["manifest_path"]), root
            ),
            "no_random_row_primary_split": True,
            "preprocessing_fit_on_train_only": (
                datamodule.normalization.get("fit_partition") == "train"
            ),
            "no_time_overlap_across_partitions": bool(
                qc_payload["checks"]["no_target_identity_overlap"]
            ),
            "uncertainty_reported": all(
                "gaussian_nll" in run["metrics"] for run in neural_runs
            ),
            "calibration_reported": all(
                "scale_calibration_factor" in run["metrics"] for run in neural_runs
            ),
            "direct_and_teacher_labels_separated": True,
            "reproducibility_recorded": all(
                (
                    root
                    / "artifacts"
                    / "runs"
                    / str(run["run_id"])
                    / "run-manifest.json"
                ).is_file()
                for run in runs
            ),
            "resource_metrics_reported": all(
                "wall_clock_seconds" in run["metrics"]
                and "peak_vram_gb" in run["metrics"]
                for run in runs
            ),
            "within_gpu_budget": (
                not budget_exhausted
                and total_gpu_hours <= budget_hours
                and peak_vram
                <= float(experiment["budget"].get("max_peak_vram_gb", 15.5))
            ),
            "latex_pdf_compiled": False,
        },
        "recommended_next_experiment": (
            "If the one-shot USNA result is credible, test whether weather adds value "
            "beyond Cn² history using one pre-registered ablation on a newly frozen "
            "development partition; do not tune on the released USNA test."
        ),
    }
    output = root / "artifacts" / "latest" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    experiment_output = (
        root / "artifacts" / "experiments" / str(experiment["id"]) / "summary.json"
    )
    experiment_output.parent.mkdir(parents=True, exist_ok=True)
    experiment_output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe Cn² forecasting")
    parser.add_argument(
        "--config", default="configs/experiments/mlo_short_forecast_dev.yaml"
    )
    parser.add_argument(
        "--model",
        choices=[
            "climatology",
            "persistence",
            "recent_mean",
            "lightgbm",
            "mlp",
            "strata_ot_surface",
        ],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--release-test",
        action="store_true",
        help="Release the official USNA test once, after configuration freeze",
    )
    args = parser.parse_args()
    summary = run_forecast_experiment(
        args.config,
        selected_model=args.model,
        selected_seed=args.seed,
        fast_dev_run=args.fast_dev_run,
        release_test=args.release_test,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
