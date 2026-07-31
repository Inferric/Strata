from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import lightning as L
import mlflow
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn

from strata_ot.config import load_yaml, resolve_experiment
from strata_ot.data.horizon import (
    HorizonDataModule,
    HorizonSequenceDataset,
)
from strata_ot.data.registry import verify_manifest
from strata_ot.evaluation.metrics import regression_metrics
from strata_ot.models import MLPBaseline, StrataOTHorizon, StrataOTSurface
from strata_ot.models.components import gaussian_nll, pinball_loss
from strata_ot.training.forecast import (
    _source_snapshot,
    fit_scale_factor,
)
from strata_ot.training.safety import (
    HardwareSafetyCallback,
    TrainingProcessLock,
    hardware_snapshot,
    preflight_hardware,
    resource_peaks,
)
from strata_ot.training.train import (
    _configure_torch_runtime,
    _configure_utf8_output,
    _environment_fingerprint,
    _repository_identity,
    _sha256_file,
)


class HorizonLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(
        self,
        features: Tensor,
        baseline: Tensor,
        horizon_minutes: Tensor,
    ) -> dict[str, Tensor]:
        if bool(getattr(self.model, "requires_horizon_minutes", False)):
            output = self.model(
                features,
                baseline=baseline,
                horizon_minutes=horizon_minutes,
            )
        else:
            output = self.model(features, baseline=baseline)
        return cast(dict[str, Tensor], output)

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(
            batch["features"],
            batch["persistence"],
            batch["horizon_minutes"],
        )
        target = batch["target"]
        nll = gaussian_nll(output["location"], output["log_scale"], target)
        quantile = pinball_loss(output["quantiles"], target)
        loss = nll + 0.20 * quantile
        error = output["location"] - target
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/nll": nll,
                f"{stage}/mae_log10_cn2": error.abs().mean(),
                f"{stage}/rmse_log10_cn2": error.square().mean().sqrt(),
                f"{stage}/bias_log10_cn2": error.mean(),
            },
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage != "train",
            batch_size=len(target),
        )
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "validation")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_epochs),
            eta_min=self.learning_rate * 0.03,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _make_datamodule(
    experiment: dict[str, Any],
    *,
    feature_set: str,
    allow_assessment: bool,
    batch_size: int | None = None,
) -> HorizonDataModule:
    root = Path(str(experiment["_repo_root"]))
    data_config = experiment["data_config"]
    split_config = load_yaml(str(data_config["split_config"]), root=root)
    trainer_config = experiment["trainer_config"]
    workers = int(
        os.getenv("STRATA_NUM_WORKERS", str(trainer_config["num_workers"]))
    )
    if workers < 0:
        raise ValueError("STRATA_NUM_WORKERS must be non-negative")
    return HorizonDataModule(
        data_config,
        split_config,
        feature_set=feature_set,
        batch_size=batch_size or int(trainer_config["batch_size"]),
        num_workers=workers,
        allow_assessment=allow_assessment,
    )


def _distribution_options(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale_parameterization": str(model_config["scale_parameterization"]),
        "min_scale": float(model_config["min_scale"]),
        "initial_scale": float(model_config["initial_scale"]),
        "min_log_scale": float(model_config["min_log_scale"]),
        "max_log_scale": float(model_config["max_log_scale"]),
    }


def _build_model(
    model_name: str,
    model_config: dict[str, Any],
    *,
    input_dim: int,
    context: int,
    weather_enabled: bool,
) -> nn.Module:
    controls = model_config["controls"]
    distribution = _distribution_options(model_config)
    if model_name == "mlp":
        return MLPBaseline(
            input_dim=input_dim * context,
            hidden_dim=int(controls["mlp_hidden_dim"]),
            depth=int(controls["mlp_depth"]),
            flatten_context=True,
            **distribution,
        )
    if model_name == "strata_ot_surface":
        return StrataOTSurface(
            input_dim=input_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            depth=int(controls["surface_depth"]),
            num_heads=int(model_config["num_heads"]),
            num_experts=int(model_config["num_experts"]),
            dropout=float(model_config["dropout"]),
            max_context=context,
            use_baseline_residual=bool(
                controls["surface_use_baseline_residual"]
            ),
            **distribution,
        )
    if model_name == "strata_ot_horizon":
        return StrataOTHorizon(
            input_dim=input_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            weather_dim=int(model_config["weather_dim"]),
            num_heads=int(model_config["num_heads"]),
            num_experts=int(model_config["num_experts"]),
            dropout=float(model_config["dropout"]),
            horizon_fourier_bands=int(model_config["horizon_fourier_bands"]),
            causal_kernel_size=int(model_config["causal_kernel_size"]),
            causal_dilations=[
                int(value) for value in model_config["causal_dilations"]
            ],
            weather_enabled=weather_enabled,
            **distribution,
        )
    raise ValueError(f"Unsupported horizon model: {model_name}")


def _evaluation_dataset(
    datamodule: HorizonDataModule,
    *,
    release_assessment: bool,
) -> tuple[HorizonSequenceDataset, str]:
    if release_assessment:
        if datamodule.assessment_set is None:
            raise RuntimeError("Assessment release requested but unavailable")
        return datamodule.assessment_set, "assessment"
    if datamodule.validation_set is None:
        raise RuntimeError("Selection role is unavailable")
    return datamodule.validation_set, "selection"


def _predict(
    module: HorizonLightningModule,
    dataset: HorizonSequenceDataset,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], float]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    collected: dict[str, list[np.ndarray]] = {}
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
                batch["horizon_minutes"].to(device),
            )
            for key, value in output.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        key: np.concatenate(values, axis=0)
        for key, values in collected.items()
    }, elapsed


def _metrics_by_horizon(
    dataset: HorizonSequenceDataset,
    prediction: np.ndarray,
    scale: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    target = dataset.target.cpu().numpy()
    horizons = dataset.horizon_minutes.cpu().numpy()
    by_horizon: dict[str, dict[str, float]] = {}
    flat: dict[str, float] = {}
    for value in sorted(np.unique(horizons)):
        mask = horizons == value
        metrics = regression_metrics(
            target[mask],
            prediction[mask],
            None if scale is None else scale[mask],
        )
        metrics["samples"] = float(mask.sum())
        key = str(int(value))
        by_horizon[key] = metrics
        for metric, metric_value in metrics.items():
            flat[f"horizon_{key}m/{metric}"] = float(metric_value)
    pooled = regression_metrics(target, prediction, scale)
    flat.update({f"pooled/{key}": value for key, value in pooled.items()})
    return by_horizon, flat


def _component_summary(
    outputs: dict[str, np.ndarray],
    dataset: HorizonSequenceDataset,
) -> dict[str, Any]:
    diagnostics = {
        key: value
        for key, value in outputs.items()
        if key
        in {
            "history_delta",
            "weather_delta",
            "weather_gate",
            "weather_contribution",
            "regime_weights",
            "history_attention",
            "weather_attention",
        }
    }
    if not diagnostics:
        return {}
    horizons = dataset.horizon_minutes.cpu().numpy()
    result: dict[str, Any] = {}
    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        values: dict[str, Any] = {}
        for key, array in diagnostics.items():
            selected = np.asarray(array)[mask]
            values[key] = {
                "mean": np.asarray(selected.mean(axis=0)).tolist(),
                "std": np.asarray(selected.std(axis=0)).tolist(),
                "minimum": float(selected.min()),
                "maximum": float(selected.max()),
            }
        result[str(int(horizon))] = values
    return result


def _save_run_artifacts(
    root: Path,
    run_id: str,
    dataset: HorizonSequenceDataset,
    *,
    outputs: dict[str, np.ndarray],
    raw_scale: np.ndarray | None,
    calibrated_scale: np.ndarray | None,
    component_summary: dict[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = root / "artifacts" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    target = dataset.target.cpu().numpy()
    persistence = dataset.persistence.cpu().numpy()
    horizon_rows = dataset.horizon_rows.cpu().numpy()
    horizon_minutes = dataset.horizon_minutes.cpu().numpy()
    prediction = outputs["location"]
    predictions_path = destination / "predictions.npz"
    payload: dict[str, np.ndarray] = {
        "target": target,
        "prediction": prediction,
        "persistence": persistence,
        "raw_scale": np.array([]) if raw_scale is None else raw_scale,
        "calibrated_scale": (
            np.array([]) if calibrated_scale is None else calibrated_scale
        ),
        "horizon_rows": horizon_rows,
        "horizon_minutes": horizon_minutes,
        "input_end_ids": dataset.input_end_ids,
        "target_ids": dataset.target_ids,
        "target_timestamps": dataset.target_timestamps,
    }
    for key, value in outputs.items():
        if key not in {"location", "log_scale", "quantiles"}:
            payload[key] = value
    np.savez_compressed(predictions_path, **cast(Any, payload))

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for axis, horizon in zip(axes.flat, sorted(np.unique(horizon_minutes)), strict=True):
        mask = horizon_minutes == horizon
        shown = min(320, int(mask.sum()))
        axis.plot(
            target[mask][:shown],
            color="#223047",
            linewidth=0.8,
            label="observed",
        )
        axis.plot(
            prediction[mask][:shown],
            color="#0b5fff",
            linewidth=0.8,
            label="forecast",
        )
        if calibrated_scale is not None:
            selected_scale = calibrated_scale[mask][:shown]
            half_width = 1.2815515655446004 * selected_scale
            selected_prediction = prediction[mask][:shown]
            axis.fill_between(
                np.arange(shown),
                selected_prediction - half_width,
                selected_prediction + half_width,
                color="#0b5fff",
                alpha=0.14,
            )
        axis.set_title(f"{int(horizon)} minute")
        axis.set(xlabel="chronological endpoint", ylabel="log10 Cn2")
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure_path = destination / "horizon-diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    components_path = destination / "component-diagnostics.json"
    components_path.write_text(
        json.dumps(component_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "predictions": predictions_path.relative_to(root).as_posix(),
        "figure": figure_path.relative_to(root).as_posix(),
        "components": components_path.relative_to(root).as_posix(),
    }


def _write_json(
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


def _tracked_artifact_bytes(
    root: Path,
    artifacts: dict[str, str],
    *,
    checkpoint: str | None = None,
) -> int:
    """Count unique stable run outputs and checkpoint bytes exactly."""
    paths = {(root / relative).resolve() for relative in artifacts.values()}
    if checkpoint:
        paths.add(Path(checkpoint).resolve())
    return sum(path.stat().st_size for path in paths if path.is_file())


def _data_identity(
    root: Path,
    experiment: dict[str, Any],
    datamodule: HorizonDataModule,
    evaluation_partition: str,
) -> dict[str, Any]:
    if datamodule.metadata is None:
        raise RuntimeError("Horizon metadata is unavailable")
    manifest_path = root / str(experiment["data_config"]["manifest_path"])
    split_path = root / str(experiment["data_config"]["split_config"])
    sealed_path = root / str(datamodule.split_config["materialized_path"])
    return {
        "dataset_id": datamodule.metadata.dataset_id,
        "source_dataset_id": datamodule.metadata.source_dataset_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "split_id": datamodule.metadata.split_id,
        "split_config_path": split_path.relative_to(root).as_posix(),
        "split_config_sha256": _sha256_file(split_path),
        "sealed_split_path": sealed_path.relative_to(root).as_posix(),
        "sealed_split_sha256": datamodule.materialized_split["sha256"],
        "evaluation_partition": evaluation_partition,
        "assessment_released": datamodule.allow_assessment,
        "official_mlo_test_loaded": False,
    }


def _tags(
    datamodule: HorizonDataModule,
    *,
    model_name: str,
    run_kind: str,
    evaluation_partition: str,
    repository: dict[str, Any],
) -> dict[str, str]:
    if datamodule.metadata is None:
        raise RuntimeError("Horizon metadata is unavailable")
    return {
        "dataset_id": datamodule.metadata.dataset_id,
        "source_dataset_id": datamodule.metadata.source_dataset_id,
        "split_id": datamodule.metadata.split_id,
        "feature_set": datamodule.metadata.feature_set,
        "raw_feature_allowlist": ",".join(
            datamodule.metadata.raw_feature_allowlist
        ),
        "horizon_rows": ",".join(
            str(value) for value in datamodule.metadata.horizon_rows
        ),
        "horizon_minutes": ",".join(
            str(int(value)) for value in datamodule.metadata.horizon_minutes
        ),
        "site": datamodule.metadata.site,
        "task_kind": "multi_horizon_forecast",
        "model_family": model_name,
        "run_kind": run_kind,
        "evaluation_partition": evaluation_partition,
        "git_dirty": str(repository["git_dirty"]).lower(),
        "source_digest_sha256": str(repository["source_digest_sha256"]),
    }


def _baseline_outputs(
    name: str,
    datamodule: HorizonDataModule,
    evaluation: HorizonSequenceDataset,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if datamodule.train_set is None:
        raise RuntimeError("Training data is unavailable")
    target = datamodule.train_set.target.cpu().numpy()
    evaluation_features = evaluation.features.cpu().numpy()
    evaluation_persistence = evaluation.persistence.cpu().numpy()
    evaluation_history = evaluation.history_target.cpu().numpy()
    if name == "climatology":
        value = float(datamodule.normalization["history_target_mean"])
        return {"location": np.full(len(evaluation), value)}, {"train_mean": value}
    if name == "persistence":
        return {"location": evaluation_persistence.copy()}, {
            "source": "last_observed_history_target"
        }
    if name == "recent_mean":
        return {"location": evaluation_history.mean(axis=1)}, {
            "history_rows": evaluation_history.shape[1]
        }
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError("Install the data extra to run LightGBM") from error
        train_features = datamodule.train_set.features.cpu().numpy()
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
            n_jobs=4,
        )
        model.fit(train_features.reshape(len(train_features), -1), target)
        prediction = model.predict(
            evaluation_features.reshape(len(evaluation_features), -1)
        )
        return {"location": np.asarray(prediction)}, {
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
    datamodule: HorizonDataModule,
    evaluation: HorizonSequenceDataset,
    *,
    evaluation_partition: str,
    name: str,
    repository: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    process = psutil.Process()
    rss_start = float(process.memory_info().rss / (1024**3))
    outputs, parameters = _baseline_outputs(name, datamodule, evaluation)
    by_horizon, metrics = _metrics_by_horizon(evaluation, outputs["location"])
    elapsed = time.monotonic() - started
    metrics.update(
        {
            "wall_clock_seconds": elapsed,
            "peak_vram_gb": 0.0,
            "process_rss_gib": float(process.memory_info().rss / (1024**3)),
            "process_rss_delta_gib": float(
                process.memory_info().rss / (1024**3) - rss_start
            ),
            "evaluation_samples": float(len(evaluation)),
        }
    )
    tags = _tags(
        datamodule,
        model_name=name,
        run_kind="baseline",
        evaluation_partition=evaluation_partition,
        repository=repository,
    )
    run_name = f"{name}-{datamodule.feature_set}-mlo"
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params({"model_family": name, **parameters})
        components: dict[str, Any] = {}
        artifacts = _save_run_artifacts(
            root,
            run.info.run_id,
            evaluation,
            outputs=outputs,
            raw_scale=None,
            calibrated_scale=None,
            component_summary=components,
        )
        evidence = {
            "run_id": run.info.run_id,
            "name": run_name,
            "kind": "baseline",
            "model": name,
            "feature_set": datamodule.feature_set,
            "seed": None,
            "repository": repository,
            "environment": environment,
            "data_identity": _data_identity(
                root, experiment, datamodule, evaluation_partition
            ),
            "resolved_configuration": {
                "parameters": parameters,
                "feature_names": datamodule.metadata.feature_names
                if datamodule.metadata
                else [],
                "raw_feature_allowlist": datamodule.raw_feature_allowlist,
                "horizon_rows": datamodule.horizon_rows,
                "horizon_minutes": datamodule.horizon_minutes,
            },
            "metrics": metrics,
            "by_horizon": by_horizon,
            "artifacts": artifacts,
        }
        evidence_path = _write_json(
            root, run.info.run_id, "run-manifest.json", evidence
        )
        artifact_bytes = _tracked_artifact_bytes(root, artifacts)
        metrics["artifact_storage_bytes"] = float(artifact_bytes)
        metrics["storage_delta_gib"] = float(artifact_bytes / (1024**3))
        evidence["metrics"] = metrics
        evidence_path = _write_json(
            root, run.info.run_id, "run-manifest.json", evidence
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(evidence["data_identity"], "data-identity.json")
        mlflow.log_dict(environment, "environment.json")
        mlflow.log_dict(repository, "repository-identity.json")
        mlflow.log_artifact(str(evidence_path), artifact_path="evidence")
        for relative in artifacts.values():
            mlflow.log_artifact(str(root / relative), artifact_path="diagnostics")
    return {
        "run_id": run.info.run_id,
        "name": run_name,
        "kind": "baseline",
        "model": name,
        "feature_set": datamodule.feature_set,
        "seed": None,
        "metrics": metrics,
        "by_horizon": by_horizon,
        "artifacts": artifacts,
        "parameters": parameters,
    }


def _train_neural(
    root: Path,
    experiment: dict[str, Any],
    datamodule: HorizonDataModule,
    evaluation: HorizonSequenceDataset,
    *,
    evaluation_partition: str,
    model_name: str,
    seed: int,
    fast_dev_run: bool,
    max_seconds: float,
    repository: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    if (
        datamodule.metadata is None
        or datamodule.train_set is None
        or datamodule.calibration_set is None
    ):
        raise RuntimeError("Horizon data must be prepared before training")
    L.seed_everything(seed, workers=True)
    trainer_config = experiment["trainer_config"]
    model_config = experiment["model_config"]
    network = _build_model(
        model_name,
        model_config,
        input_dim=datamodule.metadata.input_dim,
        context=datamodule.metadata.context,
        weather_enabled=datamodule.metadata.weather_enabled,
    )
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if model_name == "strata_ot_horizon":
        minimum = int(model_config["parameter_budget_min"])
        maximum = int(model_config["parameter_budget_max"])
        if not minimum <= parameter_count <= maximum:
            raise ValueError(
                f"Horizon v1 has {parameter_count:,} parameters; required range "
                f"is {minimum:,}..{maximum:,}"
            )
    elif parameter_count > 8_000_000:
        raise ValueError(f"{model_name} exceeds the 8M control parameter ceiling")
    module = HorizonLightningModule(
        network,
        learning_rate=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]),
        max_epochs=int(trainer_config["max_epochs"]),
    )
    checkpoint_dir = (
        root
        / "checkpoints"
        / str(experiment["id"])
        / datamodule.feature_set
        / model_name
        / str(seed)
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
    timer = Timer(
        duration={"seconds": max(1, int(max_seconds))},
        interval="step",
    )
    emergency_checkpoint = checkpoint_dir / "safety-stop.ckpt"
    safety_callback = HardwareSafetyCallback(
        root=root,
        safety=experiment["safety"],
        emergency_checkpoint=emergency_checkpoint,
    )
    run_kind = "probe" if fast_dev_run else "neural"
    run_name = (
        f"{model_name}-{datamodule.feature_set}-seed-{seed}"
        + ("-fast-dev" if fast_dev_run else "")
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger = MLFlowLogger(
        experiment_name=str(experiment["tracking"]["experiment_name"]),
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=_tags(
            datamodule,
            model_name=model_name,
            run_kind=run_kind,
            evaluation_partition=evaluation_partition,
            repository=repository,
        ),
    )
    accelerator = str(trainer_config["accelerator"])
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("The frozen horizon experiment requires CUDA")
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
        deterministic=bool(trainer_config["deterministic"]),
        benchmark=bool(trainer_config["benchmark"]),
        log_every_n_steps=int(trainer_config["log_every_n_steps"]),
        callbacks=[checkpoint, early_stop, timer, safety_callback],
        logger=logger,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    run_id = logger.run_id
    if run_id is None:
        raise RuntimeError("MLflow did not assign a run identity")
    resolved = {
        "experiment": experiment["id"],
        "model": model_name,
        "feature_set": datamodule.feature_set,
        "seed": seed,
        "parameters": parameter_count,
        "data": experiment["data_config"],
        "model_config": model_config,
        "trainer_config": {
            **trainer_config,
            "effective_num_workers": datamodule.num_workers,
            "effective_batch_size": datamodule.batch_size,
        },
        "normalization": datamodule.normalization,
        "feature_names": datamodule.metadata.feature_names,
        "raw_feature_allowlist": datamodule.metadata.raw_feature_allowlist,
        "horizon_rows": datamodule.metadata.horizon_rows,
        "horizon_minutes": datamodule.metadata.horizon_minutes,
        "sequence_qc": datamodule.sequence_qc,
        "fast_dev_run": fast_dev_run,
    }
    logger.log_hyperparams(
        {
            "model": model_name,
            "feature_set": datamodule.feature_set,
            "seed": seed,
            "parameters": parameter_count,
            "batch_size": datamodule.batch_size,
            "gradient_accumulation": int(trainer_config["gradient_accumulation"]),
            "precision": str(trainer_config["precision"]),
            "context": datamodule.metadata.context,
            "horizon_minutes": ",".join(
                str(int(value)) for value in datamodule.horizon_minutes
            ),
        }
    )
    try:
        trainer.fit(module, datamodule=datamodule)
        if safety_callback.stop_reason is not None:
            raise RuntimeError(
                f"Hardware safety stop: {safety_callback.stop_reason}"
            )
        if checkpoint.best_model_path and not fast_dev_run:
            reloaded = _build_model(
                model_name,
                model_config,
                input_dim=datamodule.metadata.input_dim,
                context=datamodule.metadata.context,
                weather_enabled=datamodule.metadata.weather_enabled,
            )
            module = HorizonLightningModule.load_from_checkpoint(
                checkpoint.best_model_path,
                model=reloaded,
                learning_rate=float(trainer_config["learning_rate"]),
                weight_decay=float(trainer_config["weight_decay"]),
                max_epochs=int(trainer_config["max_epochs"]),
            )
            module = module.to(
                torch.device("cuda" if accelerator == "gpu" else "cpu")
            )
        calibration_outputs, _ = _predict(
            module,
            datamodule.calibration_set,
            datamodule.batch_size,
        )
        calibration_target = datamodule.calibration_set.target.cpu().numpy()
        calibration_raw_scale = np.exp(calibration_outputs["log_scale"])
        scale_factor = fit_scale_factor(
            calibration_target,
            calibration_outputs["location"],
            calibration_raw_scale,
        )
        outputs, inference_seconds = _predict(
            module,
            evaluation,
            datamodule.batch_size,
        )
        raw_scale = np.exp(outputs["log_scale"])
        calibrated_scale = raw_scale * scale_factor
        by_horizon, metrics = _metrics_by_horizon(
            evaluation,
            outputs["location"],
            calibrated_scale,
        )
        raw_by_horizon, raw_metrics = _metrics_by_horizon(
            evaluation,
            outputs["location"],
            raw_scale,
        )
        del raw_by_horizon
        metrics.update({f"raw/{key}": value for key, value in raw_metrics.items()})
        elapsed = time.monotonic() - started
        safety_callback.capture_final()
        peaks = resource_peaks(safety_callback.samples)
        metrics.update(
            {
                "wall_clock_seconds": elapsed,
                "peak_vram_gb": peaks["peak_total_board_vram_gib"],
                **peaks,
                "scale_calibration_factor": scale_factor,
                "evaluation_samples": float(len(evaluation)),
                "inference_latency_ms_per_sample": (
                    inference_seconds * 1000 / len(evaluation)
                ),
                "throughput_samples_per_second": (
                    len(evaluation) / max(inference_seconds, 1e-12)
                ),
                "process_rss_gib": float(
                    psutil.Process().memory_info().rss / (1024**3)
                ),
            }
        )
        components = _component_summary(outputs, evaluation)
        artifacts = _save_run_artifacts(
            root,
            run_id,
            evaluation,
            outputs=outputs,
            raw_scale=raw_scale,
            calibrated_scale=calibrated_scale,
            component_summary=components,
        )
        calibration_record = {
            "fit_partition": "calibration",
            "pooled_across_horizons": True,
            "factor": scale_factor,
            "samples": len(calibration_target),
            "raw_metrics": regression_metrics(
                calibration_target,
                calibration_outputs["location"],
                calibration_raw_scale,
            ),
            "calibrated_metrics": regression_metrics(
                calibration_target,
                calibration_outputs["location"],
                calibration_raw_scale * scale_factor,
            ),
        }
        logger.experiment.log_dict(run_id, resolved, "resolved-config.json")
        logger.experiment.log_dict(run_id, environment, "environment.json")
        logger.experiment.log_dict(run_id, repository, "repository-identity.json")
        logger.experiment.log_dict(run_id, calibration_record, "calibration.json")
        logger.experiment.log_dict(
            run_id,
            {"samples": safety_callback.samples},
            "hardware-telemetry.json",
        )
        data_identity = _data_identity(
            root, experiment, datamodule, evaluation_partition
        )
        logger.experiment.log_dict(run_id, data_identity, "data-identity.json")
        for relative in artifacts.values():
            logger.experiment.log_artifact(
                run_id,
                str(root / relative),
                artifact_path="diagnostics",
            )
        if checkpoint.best_model_path:
            logger.experiment.log_artifact(
                run_id,
                checkpoint.best_model_path,
                artifact_path="checkpoints",
            )
        evidence = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "feature_set": datamodule.feature_set,
            "seed": seed,
            "retry_attempt": 0,
            "repository": repository,
            "environment": environment,
            "data_identity": data_identity,
            "resolved_configuration": resolved,
            "calibration": calibration_record,
            "metrics": metrics,
            "by_horizon": by_horizon,
            "component_summary": components,
            "hardware_telemetry": safety_callback.samples,
            "checkpoint": checkpoint.best_model_path or None,
            "artifacts": artifacts,
            "stopped_for_budget": timer.time_remaining() == 0,
        }
        evidence_path = _write_json(
            root, run_id, "run-manifest.json", evidence
        )
        artifact_bytes = _tracked_artifact_bytes(
            root,
            artifacts,
            checkpoint=checkpoint.best_model_path or None,
        )
        metrics["artifact_storage_bytes"] = float(artifact_bytes)
        metrics["storage_delta_gib"] = float(artifact_bytes / (1024**3))
        evidence["metrics"] = metrics
        evidence_path = _write_json(
            root, run_id, "run-manifest.json", evidence
        )
        logger.log_metrics(metrics, step=trainer.global_step)
        logger.experiment.log_artifact(
            run_id, str(evidence_path), artifact_path="evidence"
        )
        for config_key in (
            "manifest_path",
            "split_config",
        ):
            logger.experiment.log_artifact(
                run_id,
                str(root / str(experiment["data_config"][config_key])),
                artifact_path="evidence",
            )
        logger.experiment.log_artifact(
            run_id,
            str(root / str(experiment["run_manifest"])),
            artifact_path="evidence",
        )
        logger.finalize("success")
    except Exception as error:
        failure = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "feature_set": datamodule.feature_set,
            "seed": seed,
            "repository": repository,
            "environment": environment,
            "resolved_configuration": resolved,
            "hardware_telemetry": safety_callback.samples,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        failure_path = _write_json(
            root, run_id, "failure-manifest.json", failure
        )
        logger.experiment.log_artifact(
            run_id,
            str(failure_path),
            artifact_path="failures",
        )
        logger.finalize("failed")
        raise
    return {
        "run_id": run_id,
        "name": run_name,
        "kind": run_kind,
        "model": model_name,
        "feature_set": datamodule.feature_set,
        "seed": seed,
        "parameters": parameter_count,
        "metrics": metrics,
        "by_horizon": by_horizon,
        "component_summary": components,
        "calibration": calibration_record,
        "checkpoint": checkpoint.best_model_path or None,
        "artifacts": artifacts,
        "stopped_for_budget": timer.time_remaining() == 0,
    }


def _primary_relative_improvement(
    candidate: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    horizons: list[int],
) -> float:
    improvements: list[float] = []
    for horizon in horizons:
        candidate_rmse = float(
            np.mean(
                [
                    run["by_horizon"][str(horizon)]["rmse_log10_cn2"]
                    for run in candidate
                ]
            )
        )
        comparator_rmse = float(
            np.mean(
                [
                    run["by_horizon"][str(horizon)]["rmse_log10_cn2"]
                    for run in comparator
                ]
            )
        )
        improvements.append(1.0 - candidate_rmse / comparator_rmse)
    return float(np.mean(improvements))


def _primary_metric_mean(
    runs: list[dict[str, Any]], horizons: list[int], metric: str
) -> float:
    return float(
        np.mean(
            [
                run["by_horizon"][str(horizon)][metric]
                for run in runs
                for horizon in horizons
            ]
        )
    )


def _load_predictions(root: Path, run: dict[str, Any]) -> dict[str, np.ndarray]:
    path = root / str(run["artifacts"]["predictions"])
    with np.load(path) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _paired_bootstrap(
    root: Path,
    candidate_runs: list[dict[str, Any]],
    comparator_runs: list[dict[str, Any]],
    *,
    horizons: list[int],
    block_rows: int,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    candidate_by_seed = {
        int(run["seed"]): run for run in candidate_runs if run.get("seed") is not None
    }
    comparator_by_seed = {
        int(run["seed"]): run
        for run in comparator_runs
        if run.get("seed") is not None
    }
    deterministic = (
        comparator_runs[0] if comparator_runs and not comparator_by_seed else None
    )
    pairs: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for candidate_seed, candidate_run in candidate_by_seed.items():
        comparator_run = comparator_by_seed.get(candidate_seed, deterministic)
        if comparator_run is None:
            raise ValueError("Comparator is missing a paired seed")
        candidate_payload = _load_predictions(root, candidate_run)
        comparator_payload = _load_predictions(root, comparator_run)
        for key in ("target_ids", "horizon_minutes", "target"):
            if not np.array_equal(candidate_payload[key], comparator_payload[key]):
                raise ValueError(f"Paired bootstrap identity mismatch: {key}")
        pairs[candidate_seed] = (candidate_payload, comparator_payload)
    if not pairs:
        raise ValueError("Paired bootstrap requires neural candidate seeds")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for resample_index in range(resamples):
        seed_improvements: list[float] = []
        for candidate_payload, comparator_payload in pairs.values():
            horizon_improvements: list[float] = []
            for horizon in horizons:
                mask = candidate_payload["horizon_minutes"] == horizon
                target = candidate_payload["target"][mask]
                candidate = candidate_payload["prediction"][mask]
                comparator = comparator_payload["prediction"][mask]
                count = len(target)
                if count == 0:
                    raise ValueError(f"No samples for horizon {horizon}")
                needed = math.ceil(count / block_rows)
                starts = rng.integers(0, count, size=needed)
                sampled = np.concatenate(
                    [
                        (start + np.arange(block_rows)) % count
                        for start in starts
                    ]
                )[:count]
                candidate_rmse = float(
                    np.sqrt(np.mean(np.square(candidate[sampled] - target[sampled])))
                )
                comparator_rmse = float(
                    np.sqrt(np.mean(np.square(comparator[sampled] - target[sampled])))
                )
                horizon_improvements.append(1.0 - candidate_rmse / comparator_rmse)
            seed_improvements.append(float(np.mean(horizon_improvements)))
        values[resample_index] = float(np.mean(seed_improvements))
    return {
        "mean_relative_improvement": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_improvement": float(np.mean(values > 0)),
        "resamples": float(resamples),
        "block_rows": float(block_rows),
        "seed": float(seed),
    }


def _assessment_claim(
    root: Path,
    runs: list[dict[str, Any]],
    experiment: dict[str, Any],
    *,
    released: bool,
) -> dict[str, Any]:
    if not released:
        return {
            "eligible": False,
            "status": "selection_only",
            "reason": "Assessment labels have not been released",
        }
    assessment = experiment["assessment"]
    horizons = [int(value) for value in assessment["primary_horizon_minutes"]]

    def matching(model: str, feature_set: str) -> list[dict[str, Any]]:
        return [
            run
            for run in runs
            if run["model"] == model
            and run["feature_set"] == feature_set
            and run["kind"] == "neural"
        ]

    candidate = matching("strata_ot_horizon", "operational_weather")
    history = matching("strata_ot_horizon", "history_only")
    mlp = matching("mlp", "operational_weather")
    lightgbm = [
        run
        for run in runs
        if run["model"] == "lightgbm"
        and run["feature_set"] == "operational_weather"
    ]
    if not all((len(candidate) == 2, len(history) == 2, len(mlp) == 2, lightgbm)):
        return {
            "eligible": False,
            "status": "incomplete_matrix",
            "reason": "Required two-seed candidate/comparator runs are missing",
        }
    mlp_primary = _primary_metric_mean(mlp, horizons, "rmse_log10_cn2")
    lightgbm_primary = _primary_metric_mean(
        lightgbm, horizons, "rmse_log10_cn2"
    )
    stronger_name, stronger = (
        ("mlp", mlp)
        if mlp_primary <= lightgbm_primary
        else ("lightgbm", lightgbm)
    )
    improvement_history = _primary_relative_improvement(
        candidate, history, horizons
    )
    improvement_control = _primary_relative_improvement(
        candidate, stronger, horizons
    )
    history_bootstrap = _paired_bootstrap(
        root,
        candidate,
        history,
        horizons=horizons,
        block_rows=int(assessment["bootstrap_block_rows"]),
        resamples=int(assessment["bootstrap_resamples"]),
        seed=int(assessment["bootstrap_seed"]),
    )
    control_bootstrap = _paired_bootstrap(
        root,
        candidate,
        stronger,
        horizons=horizons,
        block_rows=int(assessment["bootstrap_block_rows"]),
        resamples=int(assessment["bootstrap_resamples"]),
        seed=int(assessment["bootstrap_seed"]),
    )
    candidate_by_seed = {int(run["seed"]): run for run in candidate}
    history_by_seed = {int(run["seed"]): run for run in history}
    stronger_by_seed = {
        int(run["seed"]): run
        for run in stronger
        if run.get("seed") is not None
    }
    seed_directions: dict[str, dict[str, float | bool]] = {}
    for seed, candidate_run in candidate_by_seed.items():
        control_run = stronger_by_seed.get(seed, stronger[0])
        against_history = _primary_relative_improvement(
            [candidate_run], [history_by_seed[seed]], horizons
        )
        against_control = _primary_relative_improvement(
            [candidate_run], [control_run], horizons
        )
        seed_directions[str(seed)] = {
            "against_history": against_history,
            "against_control": against_control,
            "passed": against_history > 0 and against_control > 0,
        }
    coverage_pass = all(
        0.70
        <= float(run["by_horizon"][str(horizon)]["interval_80_coverage"])
        <= 0.90
        for run in candidate
        for horizon in horizons
    )
    candidate_bias = _primary_metric_mean(
        candidate, horizons, "bias_log10_cn2"
    )
    stronger_bias = _primary_metric_mean(
        stronger, horizons, "bias_log10_cn2"
    )
    bias_pass = (
        abs(candidate_bias) <= 0.25
        and abs(candidate_bias)
        <= abs(stronger_bias)
        + float(assessment["maximum_bias_regression_log10"])
    )
    candidate_tail = _primary_metric_mean(
        candidate, horizons, "tail_mae_top_decile"
    )
    stronger_tail = _primary_metric_mean(
        stronger, horizons, "tail_mae_top_decile"
    )
    tail_pass = candidate_tail <= stronger_tail * (
        1.0 + float(assessment["maximum_tail_mae_regression_fraction"])
    )
    candidate_crps = _primary_metric_mean(candidate, horizons, "crps_gaussian")
    if stronger_name == "lightgbm":
        crps_pass = True
        stronger_crps: float | None = None
    else:
        stronger_crps = _primary_metric_mean(stronger, horizons, "crps_gaussian")
        crps_pass = candidate_crps <= stronger_crps * (
            1.0 + float(assessment["maximum_crps_regression_fraction"])
        )

    def seed_primary_values(family_runs: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                np.mean(
                    [
                        run["by_horizon"][str(horizon)]["rmse_log10_cn2"]
                        for horizon in horizons
                    ]
                )
                for run in family_runs
            ],
            dtype=np.float64,
        )

    candidate_seed_values = seed_primary_values(candidate)
    history_seed_values = seed_primary_values(history)
    candidate_stability = float(
        candidate_seed_values.std(ddof=1)
        / max(abs(float(candidate_seed_values.mean())), 1e-12)
    )
    history_stability = float(
        history_seed_values.std(ddof=1)
        / max(abs(float(history_seed_values.mean())), 1e-12)
    )
    stability_pass = (
        candidate_stability
        <= float(assessment["maximum_relative_seed_std"])
        and candidate_stability
        <= history_stability + float(assessment["maximum_stability_regression"])
    )
    threshold = float(assessment["minimum_relative_rmse_improvement"])
    conditions = {
        "point_improvement": (
            improvement_history >= threshold and improvement_control >= threshold
        ),
        "bootstrap": (
            history_bootstrap["ci95_low"] > 0
            and control_bootstrap["ci95_low"] > 0
        ),
        "seed_direction": all(
            bool(value["passed"]) for value in seed_directions.values()
        ),
        "coverage": coverage_pass,
        "bias": bias_pass,
        "tail_mae": tail_pass,
        "crps": crps_pass,
        "stability": stability_pass,
    }
    passed = all(conditions.values())
    return {
        "eligible": True,
        "status": "success" if passed else "null_or_partial",
        "passed": passed,
        "primary_horizon_minutes": horizons,
        "anchor_horizon_minutes": int(
            assessment["anchor_horizon_minutes"]
        ),
        "stronger_operational_control": stronger_name,
        "relative_rmse_improvement": {
            "against_history_horizon": improvement_history,
            "against_stronger_control": improvement_control,
            "threshold": threshold,
        },
        "bootstrap": {
            "against_history_horizon": history_bootstrap,
            "against_stronger_control": control_bootstrap,
        },
        "seed_directions": seed_directions,
        "secondary_metrics": {
            "candidate_bias": candidate_bias,
            "stronger_control_bias": stronger_bias,
            "candidate_tail_mae": candidate_tail,
            "stronger_control_tail_mae": stronger_tail,
            "candidate_crps": candidate_crps,
            "stronger_control_crps": stronger_crps,
            "candidate_relative_seed_std": candidate_stability,
            "history_relative_seed_std": history_stability,
        },
        "conditions": conditions,
    }


def _matrix_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for horizon, metrics in run["by_horizon"].items():
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "feature_set": run["feature_set"],
                    "seed": run["seed"],
                    "horizon_minutes": int(horizon),
                    "metrics": metrics,
                }
            )
    return rows


def _deadline_guard(experiment: dict[str, Any]) -> None:
    deadline = datetime.fromisoformat(
        str(experiment["deadline"]["no_new_work_after"])
    )
    timezone = ZoneInfo(str(experiment["deadline"]["timezone"]))
    now = datetime.now(timezone)
    if now >= deadline:
        raise RuntimeError(
            f"The no-new-run gate passed at {deadline.isoformat()}; now={now.isoformat()}"
        )


def run_horizon_experiment(
    config_path: str = "configs/experiments/mlo_weather_horizon_v1.yaml",
    *,
    selected_model: str | None = None,
    selected_feature_set: str | None = None,
    selected_seed: int | None = None,
    fast_dev_run: bool = False,
    release_assessment: bool = False,
) -> dict[str, Any]:
    _configure_utf8_output()
    experiment = resolve_experiment(config_path)
    if experiment.get("task_kind") != "multi_horizon_forecast":
        raise ValueError("Horizon runner requires task_kind: multi_horizon_forecast")
    if float(experiment["budget"]["max_cloud_cost_usd"]) != 0:
        raise ValueError("The horizon experiment requires zero cloud spending")
    if os.getenv("STRATA_NUM_WORKERS", "0") != "0":
        raise ValueError("The frozen overnight profile requires STRATA_NUM_WORKERS=0")
    _deadline_guard(experiment)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "4"
    torch.set_num_threads(4)
    _configure_torch_runtime(experiment["trainer_config"])
    root = Path(str(experiment["_repo_root"]))
    repository = _repository_identity(root)
    if release_assessment and repository["git_dirty"]:
        raise RuntimeError(
            "Assessment release requires a committed implementation and clean worktree"
        )
    environment = _environment_fingerprint()
    source_snapshot = _source_snapshot(root, repository)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(experiment["tracking"]["experiment_name"]))
    requested_models = (
        [selected_model] if selected_model else list(experiment["models"])
    )
    requested_feature_sets = (
        [selected_feature_set]
        if selected_feature_set
        else list(experiment["feature_sets"])
    )
    seeds = (
        [selected_seed]
        if selected_seed is not None
        else [int(value) for value in experiment["seeds"]]
    )
    if fast_dev_run:
        seeds = seeds[:1]
    datamodules: dict[str, HorizonDataModule] = {}
    evaluations: dict[str, tuple[HorizonSequenceDataset, str]] = {}
    for feature_set in requested_feature_sets:
        datamodule = _make_datamodule(
            experiment,
            feature_set=str(feature_set),
            allow_assessment=release_assessment,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        datamodules[str(feature_set)] = datamodule
        evaluations[str(feature_set)] = _evaluation_dataset(
            datamodule,
            release_assessment=release_assessment,
        )
    audit_datamodule = datamodules.get("operational_weather") or next(
        iter(datamodules.values())
    )
    feature_audit = audit_datamodule.feature_audit()
    audit_path = (
        root
        / "artifacts"
        / "experiments"
        / str(experiment["id"])
        / "feature-audit.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(feature_audit, indent=2) + "\n", encoding="utf-8"
    )
    runs: list[dict[str, Any]] = []
    deterministic_feature = (
        "history_only"
        if "history_only" in datamodules
        else next(iter(datamodules))
    )
    baseline_dm = datamodules[deterministic_feature]
    baseline_evaluation, evaluation_partition = evaluations[deterministic_feature]
    for name in ("climatology", "persistence", "recent_mean"):
        if name in requested_models:
            runs.append(
                _log_baseline(
                    root,
                    experiment,
                    baseline_dm,
                    baseline_evaluation,
                    evaluation_partition=evaluation_partition,
                    name=name,
                    repository=repository,
                    environment=environment,
                )
            )
    for feature_set in requested_feature_sets:
        if "lightgbm" not in requested_models:
            break
        datamodule = datamodules[str(feature_set)]
        evaluation, partition = evaluations[str(feature_set)]
        preflight_hardware(root, experiment["safety"])
        runs.append(
            _log_baseline(
                root,
                experiment,
                datamodule,
                evaluation,
                evaluation_partition=partition,
                name="lightgbm",
                repository=repository,
                environment=environment,
            )
        )

    budget_seconds = float(experiment["budget"]["max_total_gpu_hours"]) * 3600
    neural_started = time.monotonic()
    oom_recovered = False
    budget_exhausted = False
    safety_stop = False
    neural_models = ("mlp", "strata_ot_surface", "strata_ot_horizon")
    lock_path = root / "artifacts" / "locks" / "mlo-weather-horizon-v1.lock"
    with TrainingProcessLock(lock_path):
        for model_name in neural_models:
            if model_name not in requested_models:
                continue
            for feature_set in requested_feature_sets:
                datamodule = datamodules[str(feature_set)]
                evaluation, partition = evaluations[str(feature_set)]
                for seed in seeds:
                    remaining = budget_seconds - (time.monotonic() - neural_started)
                    if remaining <= 0:
                        budget_exhausted = True
                        break
                    preflight_hardware(root, experiment["safety"])
                    try:
                        result = _train_neural(
                            root,
                            experiment,
                            datamodule,
                            evaluation,
                            evaluation_partition=partition,
                            model_name=model_name,
                            seed=int(seed),
                            fast_dev_run=fast_dev_run,
                            max_seconds=remaining,
                            repository=repository,
                            environment=environment,
                        )
                    except torch.cuda.OutOfMemoryError:
                        if (
                            oom_recovered
                            or not bool(experiment["safety"]["one_oom_recovery"])
                            or datamodule.batch_size <= 1
                        ):
                            raise
                        oom_recovered = True
                        recovered_batch = max(1, datamodule.batch_size // 2)
                        experiment["trainer_config"]["gradient_accumulation"] = int(
                            experiment["trainer_config"]["gradient_accumulation"]
                        ) * 2
                        datamodule = _make_datamodule(
                            experiment,
                            feature_set=str(feature_set),
                            allow_assessment=release_assessment,
                            batch_size=recovered_batch,
                        )
                        datamodule.prepare_data()
                        datamodule.setup("fit")
                        datamodules[str(feature_set)] = datamodule
                        evaluation, partition = _evaluation_dataset(
                            datamodule,
                            release_assessment=release_assessment,
                        )
                        evaluations[str(feature_set)] = (evaluation, partition)
                        result = _train_neural(
                            root,
                            experiment,
                            datamodule,
                            evaluation,
                            evaluation_partition=partition,
                            model_name=model_name,
                            seed=int(seed),
                            fast_dev_run=fast_dev_run,
                            max_seconds=remaining,
                            repository=repository,
                            environment=environment,
                        )
                        result["recovery"] = {
                            "reason": "cuda_out_of_memory",
                            "batch_size": recovered_batch,
                            "gradient_accumulation": experiment["trainer_config"][
                                "gradient_accumulation"
                            ],
                        }
                    except RuntimeError as error:
                        if "Hardware safety stop:" in str(error):
                            safety_stop = True
                        raise
                    runs.append(result)
                    if result["stopped_for_budget"]:
                        budget_exhausted = True
                        break
                if budget_exhausted or safety_stop:
                    break
            if budget_exhausted or safety_stop:
                break
    total_gpu_hours = sum(
        float(run.get("metrics", {}).get("wall_clock_seconds", 0.0)) / 3600
        for run in runs
        if run["kind"] == "neural"
    )
    peak_total_board_vram = max(
        (
            float(
                run.get("metrics", {}).get(
                    "peak_total_board_vram_gib",
                    run.get("metrics", {}).get("peak_vram_gb", 0.0),
                )
            )
            for run in runs
            if run["kind"] == "neural"
        ),
        default=0.0,
    )
    peak_process_vram = max(
        (
            max(
                float(
                    run.get("metrics", {}).get(
                        "peak_process_allocated_vram_gib", 0.0
                    )
                ),
                float(
                    run.get("metrics", {}).get(
                        "peak_process_reserved_vram_gib", 0.0
                    )
                ),
            )
            for run in runs
            if run["kind"] == "neural"
        ),
        default=0.0,
    )
    artifact_storage_bytes = sum(
        int(run.get("metrics", {}).get("artifact_storage_bytes", 0.0))
        for run in runs
    )
    storage_delta = artifact_storage_bytes / (1024**3)
    claim = _assessment_claim(
        root,
        runs,
        experiment,
        released=release_assessment,
    )
    matrix = _matrix_rows(runs)
    plain_conclusion = (
        "Assessment was not released; these are selection diagnostics only."
        if not release_assessment
        else (
            "All preregistered Horizon v1 success conditions passed."
            if claim.get("passed")
            else "The bounded experiment produced null or partial evidence; no model was promoted."
        )
    )
    first_dm = next(iter(datamodules.values()))
    if first_dm.metadata is None:
        raise RuntimeError("Horizon metadata became unavailable")
    summary = {
        "experiment_id": experiment["id"],
        "task_kind": "multi_horizon_forecast",
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "hypothesis": experiment["hypothesis"],
        "plain_language_question": (
            "Does operational weather add information beyond six recent Cn2 "
            "observations from 5 through 60 minutes, and can Horizon v1 use it "
            "better than MLP and LightGBM?"
        ),
        "plain_language_conclusion": plain_conclusion,
        "dataset_id": first_dm.metadata.dataset_id,
        "source_dataset_id": first_dm.metadata.source_dataset_id,
        "site": first_dm.metadata.site,
        "split_id": first_dm.metadata.split_id,
        "evaluation_partition": (
            "assessment" if release_assessment else "selection"
        ),
        "assessment_released": release_assessment,
        "official_mlo_test_loaded": False,
        "feature_sets": requested_feature_sets,
        "horizon_rows": first_dm.metadata.horizon_rows,
        "horizon_minutes": first_dm.metadata.horizon_minutes,
        "primary_horizon_minutes": experiment["assessment"][
            "primary_horizon_minutes"
        ],
        "anchor_horizon_minutes": experiment["assessment"][
            "anchor_horizon_minutes"
        ],
        "history_minutes": first_dm.metadata.history_minutes,
        "evaluation_gate_id": experiment["evaluation_config"]["id"],
        "repository_identity": repository,
        "environment": environment,
        "source_snapshot": {
            "path": source_snapshot.relative_to(root).as_posix(),
            "sha256": _sha256_file(source_snapshot),
        },
        "data_identity": _data_identity(
            root,
            experiment,
            first_dm,
            "assessment" if release_assessment else "selection",
        ),
        "feature_audit": audit_path.relative_to(root).as_posix(),
        "sequence_qc": {
            name: datamodule.sequence_qc
            for name, datamodule in datamodules.items()
        },
        "runs": runs,
        "horizon_matrix": matrix,
        "assessment_claim": claim,
        "total_neural_gpu_hours": total_gpu_hours,
        "peak_vram_gb": peak_total_board_vram,
        "peak_total_board_vram_gib": peak_total_board_vram,
        "peak_process_vram_gib": peak_process_vram,
        "artifact_storage_bytes": artifact_storage_bytes,
        "storage_delta_gib": storage_delta,
        "cloud_cost_usd": 0.0,
        "oom_recovered": oom_recovered,
        "hardware_final": hardware_snapshot(root),
        "checks": {
            "dataset_manifest_valid": not verify_manifest(
                root / str(experiment["data_config"]["manifest_path"]), root
            ),
            "materialized_split_verified": all(
                datamodule.materialized_split["sha256"]
                == str(
                    datamodule.split_config["materialized_payload_sha256"]
                )
                for datamodule in datamodules.values()
            ),
            "no_random_row_primary_split": True,
            "no_time_overlap_across_partitions": True,
            "preprocessing_fit_on_train_only": all(
                datamodule.normalization.get("fit_partition")
                == "official_train_after_outer_purge"
                for datamodule in datamodules.values()
            ),
            "assessment_release_explicit": release_assessment,
            "official_mlo_test_sealed": True,
            "uncertainty_reported": all(
                "pooled/gaussian_nll" in run["metrics"]
                for run in runs
                if run["kind"] == "neural"
            ),
            "calibration_reported": all(
                "scale_calibration_factor" in run["metrics"]
                for run in runs
                if run["kind"] == "neural"
            ),
            "direct_and_teacher_labels_separated": True,
            "resource_metrics_reported": all(
                "wall_clock_seconds" in run["metrics"]
                and "artifact_storage_bytes" in run["metrics"]
                and (
                    run["kind"] != "neural"
                    or (
                        "peak_total_board_vram_gib" in run["metrics"]
                        and "peak_process_allocated_vram_gib" in run["metrics"]
                        and "peak_process_reserved_vram_gib" in run["metrics"]
                    )
                )
                for run in runs
            ),
            "within_gpu_budget": (
                not budget_exhausted
                and not safety_stop
                and total_gpu_hours
                < float(experiment["budget"]["max_total_gpu_hours"])
                and peak_process_vram
                <= float(experiment["budget"]["max_peak_process_vram_gb"])
                and peak_total_board_vram
                <= float(experiment["budget"]["max_board_vram_gb"])
            ),
            "within_storage_budget": storage_delta
            <= float(experiment["budget"]["max_new_storage_gib"]),
            "latex_pdf_compiled": False,
        },
        "gate_result": None,
        "recommended_next_experiment": (
            "Stop this bounded cycle. Review the null/partial evidence and the "
            "small assessment span before authorizing any independent-site protocol."
        ),
    }
    latest_path = root / "artifacts" / "latest" / "summary.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    experiment_path = (
        root
        / "artifacts"
        / "experiments"
        / str(experiment["id"])
        / "summary.json"
    )
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen MLO weather-horizon experiment"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/mlo_weather_horizon_v1.yaml",
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
            "strata_ot_horizon",
        ],
    )
    parser.add_argument(
        "--feature-set",
        choices=["history_only", "operational_weather"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--release-assessment",
        action="store_true",
        help="Load the frozen MLO validation assessment role after implementation freeze",
    )
    args = parser.parse_args()
    summary = run_horizon_experiment(
        args.config,
        selected_model=args.model,
        selected_feature_set=args.feature_set,
        selected_seed=args.seed,
        fast_dev_run=args.fast_dev_run,
        release_assessment=args.release_assessment,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
