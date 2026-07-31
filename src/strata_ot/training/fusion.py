from __future__ import annotations

import argparse
import json
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
from scipy.special import gammaln
from torch import Tensor, nn
from torch.nn import functional as F

from strata_ot.config import load_yaml, resolve_experiment
from strata_ot.data.fusion import FusionDataModule, FusionSequenceDataset
from strata_ot.evaluation.metrics import regression_metrics
from strata_ot.models import (
    FusionMLPControl,
    FusionTCNControl,
    HorizonV1FusionAdapter,
    StrataOTFusionV2,
)
from strata_ot.models.components import pinball_loss
from strata_ot.models.fusion_controls import (
    flatten_fusion_batch,
    flatten_fusion_feature_names,
)
from strata_ot.models.strata_fusion import student_t_nll
from strata_ot.training.forecast import fit_scale_factor
from strata_ot.training.horizon import _tracked_artifact_bytes
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
)

PREDICTION_OUTPUT_KEYS = (
    "location",
    "log_scale",
    "student_t_scale",
    "student_t_df",
    "quantiles",
    "embedding",
    "residual",
    "scale_weights",
    "horizon_attention",
    "weather_attention",
    "modulation_norm",
    "physics_token_norm",
    "expert_weights",
)


class FusionLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        pretraining: str = "none",
        mask_fraction: float = 0.15,
        reconstruction_weight: float = 1.0,
        future_weather_weight: float = 0.5,
        cadence_cn2_weight: float = 0.5,
        point_loss_weight: float = 0.0,
        point_loss_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.pretraining = pretraining
        self.mask_fraction = mask_fraction
        self.reconstruction_weight = reconstruction_weight
        self.future_weather_weight = future_weather_weight
        self.cadence_cn2_weight = cadence_cn2_weight
        self.point_loss_weight = point_loss_weight
        self.point_loss_beta = point_loss_beta
        self.phase = "forecast"

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return cast(dict[str, Tensor], self.model(batch))

    def _pretraining_loss(self, batch: dict[str, Tensor]) -> Tensor:
        if not isinstance(self.model, StrataOTFusionV2):
            raise RuntimeError("Atmospheric pretraining is only defined for Fusion v2")
        source = batch["short_weather"]
        observed = batch["short_present"].bool()
        mask = (torch.rand_like(source) < self.mask_fraction) & observed
        if not bool(mask.any()):
            mask[..., 0] = observed[..., 0]
        masked = source.masked_fill(mask, 0.0)
        output = self.model(batch, masked_short_weather=masked)
        reconstruction_error = (
            output["weather_reconstruction"] - source
        ).square()
        reconstruction = reconstruction_error[mask].mean()
        loss = self.reconstruction_weight * reconstruction
        logs = {"pretrain/reconstruction": reconstruction}
        if self.pretraining == "full_multitask":
            future_present = batch["future_weather_present"].bool()
            future_error = (
                output["future_weather_prediction"] - batch["future_weather"]
            ).square()
            future = future_error[future_present].mean()
            cadence = (
                output["cadence_cn2_prediction"] - batch["target"]
            ).square().mean()
            loss = (
                loss
                + self.future_weather_weight * future
                + self.cadence_cn2_weight * cadence
            )
            logs.update(
                {
                    "pretrain/future_weather": future,
                    "pretrain/cadence_cn2": cadence,
                }
            )
        self.log_dict(
            {**logs, "pretrain/loss": loss},
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["target"]),
        )
        return cast(Tensor, loss)

    def _forecast_loss(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(batch)
        target = batch["target"]
        nll = student_t_nll(
            output["location"],
            output["student_t_scale"],
            output["student_t_df"],
            target,
        )
        quantile = pinball_loss(output["quantiles"], target)
        point = F.smooth_l1_loss(
            output["location"],
            target,
            beta=self.point_loss_beta,
        )
        loss = nll + 0.20 * quantile + self.point_loss_weight * point
        error = output["location"] - target
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/student_t_nll": nll,
                f"{stage}/pinball": quantile,
                f"{stage}/point_huber": point,
                f"{stage}/rmse_log10_cn2": error.square().mean().sqrt(),
                f"{stage}/mae_log10_cn2": error.abs().mean(),
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
        if self.phase == "pretrain":
            return self._pretraining_loss(batch)
        return self._forecast_loss(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        if self.phase == "pretrain":
            return self._pretraining_loss(batch)
        return self._forecast_loss(batch, "validation")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def _candidate(experiment: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        dict(candidate)
        for candidate in experiment["candidates"]
        if candidate["id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate Fusion candidate: {candidate_id}")
    return matches[0]


def _build_model(
    candidate: dict[str, Any],
    model_config: dict[str, Any],
    *,
    weather_dim: int,
    feature_names: list[str],
) -> nn.Module:
    family = str(candidate["family"])
    hidden_dim = int(candidate.get("hidden_dim", model_config["hidden_dim"]))
    dropout = float(candidate.get("dropout", model_config["dropout"]))
    if family == "fusion_v2":
        physics_indices = [
            index
            for index, name in enumerate(feature_names)
            if name
            in {
                "dew_point",
                "vapor_pressure",
                "total_refractivity",
                "air_surface_temperature_difference",
                "dry_refractivity",
                "wet_refractivity",
            }
        ]
        physics_start = min(physics_indices) if physics_indices else None
        return StrataOTFusionV2(
            weather_dim=weather_dim,
            hidden_dim=hidden_dim,
            num_heads=int(model_config["num_heads"]),
            num_experts=int(model_config["num_experts"]),
            dropout=dropout,
            fusion=str(candidate["fusion"]),
            active_scales=tuple(str(value) for value in candidate["active_scales"]),
            horizon_fourier_bands=int(model_config["horizon_fourier_bands"]),
            physics_start=physics_start,
        )
    if family == "mlp":
        return FusionMLPControl(
            weather_dim=weather_dim,
            hidden_dim=int(model_config["controls"]["mlp_hidden_dim"]),
            depth=int(model_config["controls"]["mlp_depth"]),
            dropout=dropout,
        )
    if family == "tcn":
        return FusionTCNControl(
            weather_dim=weather_dim,
            hidden_dim=int(model_config["controls"]["tcn_hidden_dim"]),
            dropout=dropout,
        )
    if family == "horizon_v1":
        return HorizonV1FusionAdapter(
            weather_dim=weather_dim,
            hidden_dim=192,
            dropout=dropout,
        )
    raise ValueError(f"Candidate family is not neural: {family}")


def _device_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _predict(
    model: nn.Module,
    dataset: FusionSequenceDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            moved = _device_batch(batch, device)
            output = cast(dict[str, Tensor], model(moved))
            for key in PREDICTION_OUTPUT_KEYS:
                value = output.get(key)
                if value is not None:
                    collected.setdefault(key, []).append(value.float().cpu().numpy())
            for key in (
                "target",
                "persistence",
                "horizon_minutes",
                "target_timestamp_minute",
                "origin_timestamp_minute",
            ):
                collected.setdefault(key, []).append(batch[key].numpy())
    elapsed = time.perf_counter() - started
    return {
        key: np.concatenate(values, axis=0) for key, values in collected.items()
    }, elapsed


def _student_t_nll_numpy(
    target: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: np.ndarray,
) -> float:
    z = (target - location) / scale
    log_probability = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
        - np.log(scale)
        - ((degrees_of_freedom + 1.0) / 2.0)
        * np.log1p(z**2 / degrees_of_freedom)
    )
    return float(-np.mean(log_probability))


def _metrics(
    payload: dict[str, np.ndarray],
    *,
    calibrated_std: np.ndarray,
    ood_score: np.ndarray | None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    target = payload["target"]
    prediction = payload["location"]
    horizons = payload["horizon_minutes"].astype(int)
    by_horizon: dict[str, dict[str, float]] = {}
    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        values = regression_metrics(
            target[mask], prediction[mask], calibrated_std[mask]
        )
        values["student_t_nll"] = _student_t_nll_numpy(
            target[mask],
            prediction[mask],
            payload["student_t_scale"][mask],
            payload["student_t_df"][mask],
        )
        by_horizon[str(int(horizon))] = values
    pooled = regression_metrics(target, prediction, calibrated_std)
    pooled["student_t_nll"] = _student_t_nll_numpy(
        target,
        prediction,
        payload["student_t_scale"],
        payload["student_t_df"],
    )
    if ood_score is not None:
        threshold = float(np.quantile(ood_score, 0.8))
        retained = ood_score <= threshold
        high = ood_score > threshold
        pooled["ood_score_mean"] = float(ood_score.mean())
        pooled["selective_80_rmse_log10_cn2"] = float(
            np.sqrt(np.mean((target[retained] - prediction[retained]) ** 2))
        )
        pooled["ood_top_quintile_rmse_log10_cn2"] = float(
            np.sqrt(np.mean((target[high] - prediction[high]) ** 2))
        )
    return by_horizon, pooled


def _fit_ood(train_embedding: np.ndarray) -> dict[str, np.ndarray]:
    mean = train_embedding.mean(axis=0)
    std = train_embedding.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return {"mean": mean, "std": std}


def _ood_score(embedding: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (embedding - fit["mean"]) / fit["std"]
    return np.sqrt(np.mean(standardized**2, axis=-1))


def _component_summary(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    diagnostic_keys = (
        "residual",
        "scale_weights",
        "horizon_attention",
        "weather_attention",
        "modulation_norm",
        "physics_token_norm",
        "expert_weights",
    )
    horizons = payload["horizon_minutes"].astype(int)
    result: dict[str, Any] = {}
    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        values: dict[str, Any] = {}
        for key in diagnostic_keys:
            if key not in payload:
                continue
            selected = payload[key][mask]
            values[key] = {
                "mean": np.asarray(selected.mean(axis=0)).tolist(),
                "std": np.asarray(selected.std(axis=0)).tolist(),
                "minimum": float(selected.min()),
                "maximum": float(selected.max()),
            }
        result[str(int(horizon))] = values
    return result


def _save_artifacts(
    root: Path,
    run_id: str,
    payload: dict[str, np.ndarray],
    *,
    ood_score: np.ndarray | None,
    component_summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = root / "artifacts" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    predictions = destination / "predictions.npz"
    saved_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"embedding", "horizon_attention", "weather_attention"}
    }
    if ood_score is not None:
        saved_payload["ood_score"] = ood_score
    np.savez_compressed(predictions, **cast(Any, saved_payload))
    components = destination / "component-diagnostics.json"
    components.write_text(
        json.dumps(component_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    extra_path = destination / "model-diagnostics.json"
    extra_path.write_text(
        json.dumps(extra or {}, indent=2) + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    horizons = payload["horizon_minutes"].astype(int)
    for axis, horizon in zip(axes.flat, sorted(np.unique(horizons)), strict=True):
        mask = horizons == horizon
        shown = min(400, int(mask.sum()))
        axis.plot(
            payload["target"][mask][:shown],
            color="#223047",
            linewidth=0.7,
            label="observed",
        )
        axis.plot(
            payload["location"][mask][:shown],
            color="#0B5FFF",
            linewidth=0.7,
            label="forecast",
        )
        axis.set_title(f"{horizon} minute")
        axis.set(xlabel="chronological sample", ylabel="log10 Cn2")
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure_path = destination / "horizon-diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {
        "predictions": predictions.relative_to(root).as_posix(),
        "components": components.relative_to(root).as_posix(),
        "model_diagnostics": extra_path.relative_to(root).as_posix(),
        "figure": figure_path.relative_to(root).as_posix(),
    }


def _dataset_matrix(
    dataset: FusionSequenceDataset,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    horizons: list[np.ndarray] = []
    for batch in loader:
        features.append(flatten_fusion_batch(batch).numpy())
        targets.append(batch["target"].numpy())
        horizons.append(batch["horizon_minutes"].numpy())
    return (
        np.concatenate(features),
        np.concatenate(targets),
        np.concatenate(horizons),
    )


def _baseline_run(
    root: Path,
    experiment: dict[str, Any],
    candidate: dict[str, Any],
    datamodule: FusionDataModule,
    evaluation: FusionSequenceDataset,
    *,
    fold_id: str,
    seed: int,
    repository: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    if datamodule.train_set is None or datamodule.calibration_set is None:
        raise RuntimeError("Fusion baseline data roles are unavailable")
    family = str(candidate["family"])
    started = time.monotonic()
    hardware_samples = [hardware_snapshot(root)]
    evaluation_x, target, horizons = _dataset_matrix(
        evaluation, batch_size=datamodule.batch_size
    )
    train_x, train_target, _ = _dataset_matrix(
        datamodule.train_set, batch_size=datamodule.batch_size
    )
    extra: dict[str, Any] = {}
    if family == "persistence":
        prediction = np.asarray(
            [evaluation[index]["persistence"].item() for index in range(len(evaluation))]
        )
    elif family == "climatology":
        if datamodule.normalization is None:
            raise RuntimeError("Training-only normalization is unavailable")
        prediction = np.full(len(evaluation), datamodule.normalization.target_mean)
    elif family == "lightgbm":
        from lightgbm import LGBMRegressor
        from sklearn.inspection import permutation_importance

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            reg_lambda=1.0,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(train_x, train_target)
        prediction = np.asarray(model.predict(evaluation_x))
        names = flatten_fusion_feature_names(datamodule.feature_names)
        if len(names) != evaluation_x.shape[1]:
            raise RuntimeError("Semantic LightGBM feature map does not match matrix")
        importance = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(importance)[::-1][:25]
        permutation_rows = np.linspace(
            0,
            len(train_x) - 1,
            min(1500, len(train_x)),
            dtype=np.int64,
        )
        permutation = permutation_importance(
            model,
            train_x[permutation_rows],
            train_target[permutation_rows],
            n_repeats=2,
            random_state=20260730,
            n_jobs=4,
            scoring="neg_root_mean_squared_error",
        )
        extra["lightgbm_diagnostic_only"] = True
        extra["permutation_fit_role"] = "train"
        extra["permutation_samples"] = len(permutation_rows)
        extra["top_gain_features"] = [
            {"name": names[index], "importance": float(importance[index])}
            for index in order
        ]
        perm_order = np.argsort(permutation.importances_mean)[::-1][:25]
        extra["top_permutation_features"] = [
            {
                "name": names[index],
                "importance": float(permutation.importances_mean[index]),
            }
            for index in perm_order
        ]
    else:
        raise ValueError(f"Unsupported baseline family: {family}")
    calibration_x, calibration_target, _ = _dataset_matrix(
        datamodule.calibration_set,
        batch_size=datamodule.batch_size,
    )
    if family == "persistence":
        calibration_prediction = np.asarray(
            [
                datamodule.calibration_set[index]["persistence"].item()
                for index in range(len(datamodule.calibration_set))
            ]
        )
    elif family == "climatology":
        calibration_prediction = np.full(
            len(datamodule.calibration_set),
            float(np.mean(train_target)),
        )
    else:
        calibration_prediction = np.asarray(model.predict(calibration_x))
    residual_quantile = float(
        np.quantile(
            np.abs(calibration_target - calibration_prediction),
            0.8,
        )
    )
    calibrated_std = np.full(
        len(target), max(residual_quantile / 1.2815515655446004, 1e-3)
    )
    payload = {
        "location": prediction,
        "target": target,
        "persistence": np.asarray(
            [evaluation[index]["persistence"].item() for index in range(len(evaluation))]
        ),
        "horizon_minutes": horizons,
        "target_timestamp_minute": np.asarray(
            [
                evaluation[index]["target_timestamp_minute"].item()
                for index in range(len(evaluation))
            ]
        ),
        "origin_timestamp_minute": np.asarray(
            [
                evaluation[index]["origin_timestamp_minute"].item()
                for index in range(len(evaluation))
            ]
        ),
        "student_t_scale": calibrated_std,
        "student_t_df": np.full(len(target), 30.0),
    }
    by_horizon, metrics = _metrics(
        payload,
        calibrated_std=calibrated_std,
        ood_score=None,
    )
    metrics["wall_clock_seconds"] = time.monotonic() - started
    hardware_samples.append(hardware_snapshot(root))
    metrics.update(resource_peaks(hardware_samples))
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(str(experiment["tracking"]["experiment_name"]))
    tags = {
        "program_id": str(experiment["id"]),
        "candidate_id": str(candidate["id"]),
        "ablation_category": str(candidate["category"]),
        "model_family": family,
        "fold_id": fold_id,
        "seed": str(seed),
        "evaluation_partition": (
            "confirmation" if fold_id == "final" else "selection"
        ),
        "task_kind": "rolling_multi_horizon_forecast",
    }
    with mlflow.start_run(run_name=f"{candidate['id']}-{fold_id}-{seed}", tags=tags) as run:
        component_summary: dict[str, Any] = {}
        artifacts = _save_artifacts(
            root,
            run.info.run_id,
            payload,
            ood_score=None,
            component_summary=component_summary,
            extra=extra,
        )
        telemetry_path = (
            root
            / "artifacts"
            / "runs"
            / run.info.run_id
            / "hardware-telemetry.json"
        )
        telemetry_path.write_text(
            json.dumps({"samples": hardware_samples}, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts["hardware_telemetry"] = telemetry_path.relative_to(root).as_posix()
        artifact_bytes = _tracked_artifact_bytes(root, artifacts)
        metrics["artifact_storage_bytes"] = float(artifact_bytes)
        metrics["storage_delta_gib"] = artifact_bytes / (1024**3)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(extra, "model-diagnostics.json")
        mlflow.log_dict({"samples": hardware_samples}, "hardware-telemetry.json")
        for relative in artifacts.values():
            mlflow.log_artifact(str(root / relative), artifact_path="diagnostics")
        result = {
            "run_id": run.info.run_id,
            "candidate_id": candidate["id"],
            "category": candidate["category"],
            "family": family,
            "fold_id": fold_id,
            "seed": seed,
            "kind": "diagnostic" if family == "lightgbm" else "baseline",
            "evaluation_partition": tags["evaluation_partition"],
            "metrics": metrics,
            "by_horizon": by_horizon,
            "calibration": {
                "fit_partition": "calibration",
                "method": "symmetric_absolute_residual",
                "q80": residual_quantile,
            },
            "component_summary": component_summary,
            "artifacts": artifacts,
            "resolved_configuration": {
                "candidate": candidate,
                "fold_id": fold_id,
                "seed": seed,
                "feature_names": datamodule.feature_names,
                "flattened_feature_names": (
                    names if family == "lightgbm" else None
                ),
                "normalization": {
                    "fit_fold": datamodule.normalization.fit_fold,
                    "fit_role": datamodule.normalization.fit_role,
                }
                if datamodule.normalization is not None
                else None,
                "split_sha256": datamodule.materialized_split["sha256"],
                "source_sha256": experiment["data_config"]["source_sha256"],
            },
            "hardware_telemetry": hardware_samples,
            "repository": repository,
            "environment": environment,
        }
        evidence_path = root / "artifacts" / "runs" / run.info.run_id / "run-manifest.json"
        evidence_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(evidence_path), artifact_path="evidence")
    return result


def _neural_run(
    root: Path,
    experiment: dict[str, Any],
    candidate: dict[str, Any],
    datamodule: FusionDataModule,
    evaluation: FusionSequenceDataset | None,
    *,
    fold_id: str,
    seed: int,
    screen: bool,
    repository: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    if (
        datamodule.train_set is None
        or datamodule.calibration_set is None
        or datamodule.normalization is None
    ):
        raise RuntimeError("Fusion neural data roles are unavailable")
    L.seed_everything(seed, workers=True)
    trainer_config = experiment["trainer_config"]
    model_config = experiment["model_config"]
    network = _build_model(
        candidate,
        model_config,
        weather_dim=len(datamodule.feature_names),
        feature_names=datamodule.feature_names,
    )
    parameters = sum(parameter.numel() for parameter in network.parameters())
    if candidate["family"] == "fusion_v2" and not int(
        model_config["parameter_budget_min"]
    ) <= parameters <= int(model_config["parameter_budget_max"]):
        raise ValueError(f"Fusion v2 parameter budget violation: {parameters}")
    final_fit = fold_id == "final"
    max_epochs = int(
        trainer_config["confirmation_max_epochs"]
        if final_fit
        else (
            trainer_config["screen_max_epochs"]
            if screen
            else trainer_config["full_max_epochs"]
        )
    )
    pretraining = str(candidate["pretraining"])
    policy = model_config["pretraining_policy"]
    module = FusionLightningModule(
        network,
        learning_rate=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]),
        max_epochs=max_epochs,
        pretraining=pretraining,
        mask_fraction=float(policy["mask_fraction"]),
        reconstruction_weight=float(policy["reconstruction_weight"]),
        future_weather_weight=float(policy["future_weather_weight"]),
        cadence_cn2_weight=float(policy["cadence_cn2_weight"]),
        point_loss_weight=float(candidate.get("point_loss_weight", 0.0)),
        point_loss_beta=float(candidate.get("point_loss_beta", 0.1)),
    )
    checkpoint_dir = (
        root
        / "checkpoints"
        / str(experiment["id"])
        / str(candidate["id"])
        / fold_id
        / str(seed)
    )
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=None if final_fit else "validation/rmse_log10_cn2",
        mode="min",
        save_top_k=1,
        filename="{epoch:03d}-{step:06d}",
        auto_insert_metric_name=False,
    )
    callbacks: list[L.Callback] = [checkpoint]
    if not final_fit:
        callbacks.append(
            EarlyStopping(
                monitor="validation/rmse_log10_cn2",
                mode="min",
                patience=int(trainer_config["early_stopping_patience"]),
            )
        )
    max_seconds = float(
        trainer_config["screen_max_seconds"]
        if screen
        else trainer_config["full_max_seconds"]
    )
    timer = Timer(duration={"seconds": int(max_seconds)}, interval="step")
    safety_callback = HardwareSafetyCallback(
        root=root,
        safety=experiment["safety"],
        emergency_checkpoint=checkpoint_dir / "safety-stop.ckpt",
    )
    callbacks.extend((timer, safety_callback))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    tags = {
        "program_id": str(experiment["id"]),
        "candidate_id": str(candidate["id"]),
        "ablation_category": str(candidate["category"]),
        "model_family": str(candidate["family"]),
        "fold_id": fold_id,
        "seed": str(seed),
        "physics_tokens": str(candidate["physics_tokens"]),
        "fusion": str(candidate["fusion"]),
        "active_scales": ",".join(str(value) for value in candidate["active_scales"]),
        "pretraining": pretraining,
        "calibration": str(candidate["calibration"]),
        "point_loss_weight": str(candidate.get("point_loss_weight", 0.0)),
        "evaluation_partition": (
            "confirmation" if fold_id == "final" else "selection"
        ),
        "task_kind": "rolling_multi_horizon_forecast",
        "source_digest_sha256": str(repository["source_digest_sha256"]),
    }
    logger = MLFlowLogger(
        experiment_name=str(experiment["tracking"]["experiment_name"]),
        tracking_uri=tracking_uri,
        run_name=f"{candidate['id']}-{fold_id}-{seed}",
        tags=tags,
    )
    accelerator = str(trainer_config["accelerator"])
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Fusion v2 requires the configured local CUDA device")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    pretraining_checkpoint: str | None = None
    if pretraining != "none":
        module.phase = "pretrain"
        pretrain_epochs = int(
            policy["screen_epochs"] if screen else policy["full_epochs"]
        )
        pretrain_trainer = L.Trainer(
            accelerator=accelerator,
            devices=int(trainer_config["devices"]),
            precision=cast(Any, str(trainer_config["precision"])),
            max_epochs=pretrain_epochs,
            accumulate_grad_batches=int(trainer_config["gradient_accumulation"]),
            gradient_clip_val=float(trainer_config["gradient_clip_val"]),
            deterministic=bool(trainer_config["deterministic"]),
            benchmark=bool(trainer_config["benchmark"]),
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
        )
        pretrain_trainer.fit(module, train_dataloaders=datamodule.train_dataloader())
        pretraining_path = checkpoint_dir / "pretraining.ckpt"
        pretraining_path.parent.mkdir(parents=True, exist_ok=True)
        pretrain_trainer.save_checkpoint(str(pretraining_path))
        pretraining_checkpoint = str(pretraining_path)
    module.phase = "forecast"
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=int(trainer_config["devices"]),
        precision=cast(Any, str(trainer_config["precision"])),
        max_epochs=max_epochs,
        accumulate_grad_batches=int(trainer_config["gradient_accumulation"]),
        gradient_clip_val=float(trainer_config["gradient_clip_val"]),
        deterministic=bool(trainer_config["deterministic"]),
        benchmark=bool(trainer_config["benchmark"]),
        log_every_n_steps=int(trainer_config["log_every_n_steps"]),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    run_id = logger.run_id
    if run_id is None:
        raise RuntimeError("MLflow did not assign the Fusion run identity")
    try:
        if final_fit:
            trainer.fit(
                module,
                train_dataloaders=datamodule.train_dataloader(),
            )
        else:
            trainer.fit(module, datamodule=datamodule)
        if safety_callback.stop_reason:
            raise RuntimeError(f"Hardware safety stop: {safety_callback.stop_reason}")
        if checkpoint.best_model_path:
            reloaded_network = _build_model(
                candidate,
                model_config,
                weather_dim=len(datamodule.feature_names),
                feature_names=datamodule.feature_names,
            )
            module = FusionLightningModule.load_from_checkpoint(
                checkpoint.best_model_path,
                model=reloaded_network,
            )
        model = module.model.to(device)
        calibration_payload, _ = _predict(
            model,
            datamodule.calibration_set,
            batch_size=datamodule.batch_size,
            device=device,
        )
        if evaluation is None:
            evaluation = datamodule.release_confirmation_data()
        calibration_method = str(candidate["calibration"])
        raw_scale = calibration_payload["student_t_scale"]
        scale_factor = (
            fit_scale_factor(
                calibration_payload["target"],
                calibration_payload["location"],
                raw_scale,
            )
            if calibration_method != "none"
            else 1.0
        )
        conformal_q80 = float(
            np.quantile(
                np.abs(
                    calibration_payload["target"]
                    - calibration_payload["location"]
                ),
                0.8,
            )
        )
        evaluation_payload, inference_seconds = _predict(
            model,
            evaluation,
            batch_size=datamodule.batch_size,
            device=device,
        )
        if calibration_method in {"conformal", "student_t_conformal"}:
            calibrated_std = np.full(
                len(evaluation),
                max(conformal_q80 / 1.2815515655446004, 1e-3),
            )
        else:
            calibrated_std = (
                evaluation_payload["student_t_scale"] * scale_factor
            )
        train_payload, _ = _predict(
            model,
            datamodule.train_set,
            batch_size=datamodule.batch_size,
            device=device,
        )
        ood_fit = _fit_ood(train_payload["embedding"])
        ood_score = _ood_score(evaluation_payload["embedding"], ood_fit)
        by_horizon, metrics = _metrics(
            evaluation_payload,
            calibrated_std=calibrated_std,
            ood_score=ood_score,
        )
        safety_callback.capture_final()
        peaks = resource_peaks(safety_callback.samples)
        elapsed = time.monotonic() - started
        metrics.update(
            {
                "wall_clock_seconds": elapsed,
                **peaks,
                "peak_vram_gb": peaks["peak_total_board_vram_gib"],
                "inference_latency_ms_per_sample": (
                    inference_seconds * 1000 / len(evaluation)
                ),
                "throughput_samples_per_second": (
                    len(evaluation) / max(inference_seconds, 1e-12)
                ),
                "process_rss_gib": (
                    psutil.Process().memory_info().rss / (1024**3)
                ),
                "scale_calibration_factor": scale_factor,
                "conformal_q80": conformal_q80,
            }
        )
        components = _component_summary(evaluation_payload)
        artifacts = _save_artifacts(
            root,
            run_id,
            evaluation_payload,
            ood_score=ood_score,
            component_summary=components,
            extra={
                "ood_fit_mean": ood_fit["mean"].tolist(),
                "ood_fit_std": ood_fit["std"].tolist(),
            },
        )
        checkpoint_path = checkpoint.best_model_path or None
        artifact_bytes = _tracked_artifact_bytes(
            root,
            artifacts,
            checkpoint=checkpoint_path,
        )
        if pretraining_checkpoint and Path(pretraining_checkpoint).is_file():
            artifact_bytes += Path(pretraining_checkpoint).stat().st_size
        metrics["artifact_storage_bytes"] = float(artifact_bytes)
        metrics["storage_delta_gib"] = artifact_bytes / (1024**3)
        calibration_record = {
            "fit_partition": "calibration",
            "method": calibration_method,
            "scale_factor": scale_factor,
            "conformal_q80": conformal_q80,
            "samples": len(calibration_payload["target"]),
        }
        resolved = {
            "candidate": candidate,
            "model_config": model_config,
            "trainer_config": trainer_config,
            "fold_id": fold_id,
            "seed": seed,
            "screen": screen,
            "parameters": parameters,
            "training_example_cap": int(
                candidate.get(
                    "max_train_examples",
                    (
                        trainer_config["screen_max_train_examples"]
                        if screen
                        else trainer_config["full_max_train_examples"]
                    ),
                )
            ),
            "training_examples_actual": len(datamodule.train_set),
            "calibration_examples_actual": len(datamodule.calibration_set),
            "evaluation_examples_actual": (
                len(evaluation) if evaluation is not None else 0
            ),
            "feature_names": datamodule.feature_names,
            "normalization": {
                "fit_fold": datamodule.normalization.fit_fold,
                "fit_role": datamodule.normalization.fit_role,
                "target_mean": datamodule.normalization.target_mean,
                "target_std": datamodule.normalization.target_std,
            },
            "split_sha256": datamodule.materialized_split["sha256"],
            "source_sha256": experiment["data_config"]["source_sha256"],
        }
        result = {
            "run_id": run_id,
            "candidate_id": candidate["id"],
            "category": candidate["category"],
            "family": candidate["family"],
            "fold_id": fold_id,
            "seed": seed,
            "kind": "neural",
            "evaluation_partition": tags["evaluation_partition"],
            "parameters": parameters,
            "metrics": metrics,
            "by_horizon": by_horizon,
            "calibration": calibration_record,
            "component_summary": components,
            "checkpoint": checkpoint_path,
            "pretraining_checkpoint": pretraining_checkpoint,
            "artifacts": artifacts,
            "resolved_configuration": resolved,
            "hardware_telemetry": safety_callback.samples,
            "repository": repository,
            "environment": environment,
            "stopped_for_budget": timer.time_remaining() == 0,
        }
        evidence_path = root / "artifacts" / "runs" / run_id / "run-manifest.json"
        evidence_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        logger.log_hyperparams(
            {
                "candidate_id": candidate["id"],
                "fold_id": fold_id,
                "seed": seed,
                "parameters": parameters,
                "physics_tokens": candidate["physics_tokens"],
                "fusion": candidate["fusion"],
                "pretraining": pretraining,
                "calibration": calibration_method,
                "point_loss_weight": float(
                    candidate.get("point_loss_weight", 0.0)
                ),
                "training_example_cap": int(
                    candidate.get(
                        "max_train_examples",
                        (
                            trainer_config["screen_max_train_examples"]
                            if screen
                            else trainer_config["full_max_train_examples"]
                        ),
                    )
                ),
                "training_examples_actual": len(datamodule.train_set),
            }
        )
        logger.log_metrics(metrics, step=trainer.global_step)
        logger.experiment.log_dict(run_id, resolved, "resolved-config.json")
        logger.experiment.log_dict(run_id, calibration_record, "calibration.json")
        logger.experiment.log_dict(
            run_id,
            {"samples": safety_callback.samples},
            "hardware-telemetry.json",
        )
        for relative in artifacts.values():
            logger.experiment.log_artifact(
                run_id, str(root / relative), artifact_path="diagnostics"
            )
        if checkpoint_path:
            logger.experiment.log_artifact(
                run_id, checkpoint_path, artifact_path="checkpoints"
            )
        if pretraining_checkpoint:
            logger.experiment.log_artifact(
                run_id, pretraining_checkpoint, artifact_path="pretraining"
            )
        logger.experiment.log_artifact(
            run_id, str(evidence_path), artifact_path="evidence"
        )
        logger.finalize("success")
        return result
    except Exception as error:
        failure = {
            "run_id": run_id,
            "candidate_id": candidate["id"],
            "fold_id": fold_id,
            "seed": seed,
            "repository": repository,
            "environment": environment,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        failure_path = root / "artifacts" / "runs" / run_id / "failure-manifest.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        logger.experiment.log_artifact(
            run_id, str(failure_path), artifact_path="failures"
        )
        logger.finalize("failed")
        raise


def _ensure_program_gpu_budget(
    root: Path,
    *,
    candidate_ids: set[str],
    run_limit_seconds: float,
    budget_hours: float,
) -> float:
    consumed_seconds = 0.0
    for path in (root / "artifacts" / "runs").glob(
        "*/run-manifest.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("kind") != "neural"
            or payload.get("candidate_id") not in candidate_ids
        ):
            continue
        consumed_seconds += float(
            payload.get("metrics", {}).get(
                "wall_clock_seconds",
                0.0,
            )
        )
    budget_seconds = budget_hours * 3600.0
    if consumed_seconds + run_limit_seconds > budget_seconds:
        raise RuntimeError(
            "Fusion program GPU-hour budget cannot admit another bounded "
            f"run: {consumed_seconds / 3600.0:.3f} h consumed, "
            f"{run_limit_seconds / 3600.0:.3f} h reserved, "
            f"{budget_hours:.3f} h ceiling"
        )
    return consumed_seconds


def run_fusion_candidate(
    config_path: str = "configs/experiments/fusion_v2_program.yaml",
    *,
    candidate_id: str,
    fold_id: str = "fold-1",
    seed: int = 17,
    screen: bool = True,
    release_confirmation: bool = False,
) -> dict[str, Any]:
    _configure_utf8_output()
    os.environ["STRATA_NUM_WORKERS"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    torch.set_num_threads(4)
    experiment = resolve_experiment(config_path)
    _configure_torch_runtime(experiment["trainer_config"])
    root = Path(str(experiment["_repo_root"]))
    candidate = _candidate(experiment, candidate_id)
    if release_confirmation:
        if fold_id != "final":
            raise ValueError("Confirmation release requires fold_id=final")
        if screen:
            raise ValueError("Confirmation release requires the frozen full-fit policy")
        if candidate["family"] in {"persistence", "climatology", "lightgbm"}:
            raise ValueError(
                "One-shot confirmation release requires a frozen neural candidate"
            )
        repository = _repository_identity(root)
        if repository["git_dirty"]:
            raise RuntimeError(
                "Confirmation release requires committed protocol and implementation"
            )
    elif fold_id == "final":
        raise ValueError("Final-fold evaluation requires --release-confirmation")
    else:
        repository = _repository_identity(root)
    deadline = datetime.fromisoformat(str(experiment["deadline"]["no_new_work_after"]))
    now = datetime.now(ZoneInfo(str(experiment["deadline"]["timezone"])))
    if now >= deadline:
        raise RuntimeError("Fusion program no-new-run deadline has passed")
    trainer_config = experiment["trainer_config"]
    max_train_examples = int(
        candidate.get(
            "max_train_examples",
            (
                trainer_config["screen_max_train_examples"]
                if screen
                else trainer_config["full_max_train_examples"]
            ),
        )
    )
    max_evaluation_examples = int(
        trainer_config["screen_max_evaluation_examples"]
        if screen
        else trainer_config["full_max_evaluation_examples"]
    )
    if candidate["family"] not in {
        "persistence",
        "climatology",
        "lightgbm",
    }:
        _ensure_program_gpu_budget(
            root,
            candidate_ids={
                str(item["id"]) for item in experiment["candidates"]
            },
            run_limit_seconds=float(
                trainer_config[
                    "screen_max_seconds"
                    if screen
                    else "full_max_seconds"
                ]
            ),
            budget_hours=float(
                experiment["budget"]["max_total_gpu_hours"]
            ),
        )
    preflight_hardware(root, experiment["safety"])
    workers = int(os.getenv("STRATA_NUM_WORKERS", "0"))
    datamodule = FusionDataModule(
        experiment["data_config"],
        load_yaml(
            str(experiment["data_config"]["split_config"]),
            root=root,
        ),
        fold_id=fold_id,
        physics_tokens=str(candidate["physics_tokens"]),
        batch_size=int(trainer_config["batch_size"]),
        num_workers=workers,
        max_train_examples=max_train_examples,
        max_evaluation_examples=max_evaluation_examples,
        release_confirmation=release_confirmation,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    evaluation = None if release_confirmation else datamodule.selection_set
    if not release_confirmation and evaluation is None:
        raise RuntimeError("Requested Fusion evaluation role is unavailable")
    environment = _environment_fingerprint()
    if candidate["family"] in {"persistence", "climatology", "lightgbm"}:
        assert evaluation is not None
        return _baseline_run(
            root,
            experiment,
            candidate,
            datamodule,
            evaluation,
            fold_id=fold_id,
            seed=seed,
            repository=repository,
            environment=environment,
        )
    with TrainingProcessLock(
        root / "artifacts" / "locks" / "fusion-training.lock"
    ):
        try:
            return _neural_run(
                root,
                experiment,
                candidate,
                datamodule,
                evaluation,
                fold_id=fold_id,
                seed=seed,
                screen=screen,
                repository=repository,
                environment=environment,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            recovered_batch_size = max(1, datamodule.batch_size // 2)
            if recovered_batch_size == datamodule.batch_size:
                raise RuntimeError("OOM after one recovery was unavailable") from error
            datamodule.batch_size = recovered_batch_size
            experiment["trainer_config"]["gradient_accumulation"] = (
                int(experiment["trainer_config"]["gradient_accumulation"]) * 2
            )
            result = _neural_run(
                root,
                experiment,
                candidate,
                datamodule,
                evaluation,
                fold_id=fold_id,
                seed=seed,
                screen=screen,
                repository=repository,
                environment=environment,
            )
            result["oom_recovery"] = {
                "attempts": 1,
                "batch_size": recovered_batch_size,
                "gradient_accumulation": experiment["trainer_config"][
                    "gradient_accumulation"
                ],
            }
            return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one preregistered Strata-OT Fusion v2 candidate"
    )
    parser.add_argument(
        "--config", default="configs/experiments/fusion_v2_program.yaml"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", default="fold-1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--release-confirmation", action="store_true")
    args = parser.parse_args()
    result = run_fusion_candidate(
        args.config,
        candidate_id=args.candidate,
        fold_id=args.fold,
        seed=args.seed,
        screen=not args.full,
        release_confirmation=args.release_confirmation,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
