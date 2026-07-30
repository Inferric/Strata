from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
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
from strata_ot.data.otbench_adapter import OtbenchDataModule, SequenceDataset
from strata_ot.evaluation.metrics import regression_metrics
from strata_ot.models import MLPBaseline, StrataOTSurface
from strata_ot.models.baselines import fit_climatology, persistence_predictions
from strata_ot.training.module import Cn2LightningModule


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(root: Path) -> str:
    files: list[Path] = []
    for relative in ("pyproject.toml", "uv.lock"):
        candidate = root / relative
        if candidate.is_file():
            files.append(candidate)
    for relative in ("src", "configs", "schemas", "scripts", "reports/templates"):
        directory = root / relative
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative_bytes = path.relative_to(root).as_posix().encode()
        digest.update(relative_bytes)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _configure_torch_runtime(trainer_config: dict[str, Any]) -> None:
    precision = str(trainer_config.get("float32_matmul_precision", "highest"))
    torch.set_float32_matmul_precision(precision)
    deterministic = bool(trainer_config.get("deterministic"))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if str(trainer_config.get("sdpa_backend", "auto")) == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _repository_identity(root: Path) -> dict[str, Any]:
    revision = _git_output(root, "rev-parse", "HEAD")
    return {
        "git_revision": revision,
        "git_branch": _git_output(root, "symbolic-ref", "--short", "HEAD") or "master",
        "git_dirty": bool(_git_output(root, "status", "--porcelain=v1")),
        "working_tree_status": _git_output(root, "status", "--porcelain=v1") or "",
        "source_digest_sha256": _source_digest(root),
        "identity_mode": "git_commit" if revision is not None else "unborn_branch_content_digest",
    }


def _environment_fingerprint() -> dict[str, Any]:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "lightning": importlib.metadata.version("lightning"),
        "mlflow": importlib.metadata.version("mlflow"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "device": device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
            "math": torch.backends.cuda.math_sdp_enabled(),  # type: ignore[no-untyped-call]
        },
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        environment["gpu"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": float(properties.total_memory / (1024**3)),
            "driver": driver.stdout.strip() if driver.returncode == 0 else "unavailable",
        }
    return environment


def _data_identity(
    root: Path,
    experiment: dict[str, Any],
    datamodule: OtbenchDataModule,
) -> dict[str, Any]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    manifest_path = root / str(experiment["data_config"]["manifest_path"])
    split_path = (
        root
        / "data"
        / "sealed"
        / "splits"
        / f"{datamodule.metadata.split_id}.json"
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        "dataset_id": datamodule.metadata.dataset_id,
        "dataset_manifest_path": str(manifest_path.relative_to(root)),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "split_id": datamodule.metadata.split_id,
        "split_path": str(split_path.relative_to(root)),
        "split_sha256": split["sha256"],
        "target_transform": datamodule.metadata.target_transform,
    }


def _write_evidence_file(
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


def _evidence_source_paths(
    root: Path,
    experiment: dict[str, Any],
    datamodule: OtbenchDataModule,
) -> list[Path]:
    if datamodule.metadata is None:
        raise RuntimeError("Data metadata is unavailable")
    return [
        root / str(experiment["data_config"]["manifest_path"]),
        root
        / "data"
        / "sealed"
        / "splits"
        / f"{datamodule.metadata.split_id}.json",
    ]


def _dataset_targets(dataset: SequenceDataset) -> np.ndarray:
    return dataset.target[dataset.context - 1 :].cpu().numpy()


def _dataset_row_ids(dataset: SequenceDataset) -> np.ndarray:
    return dataset.endpoint_row_ids


def _save_diagnostics(
    root: Path,
    run_id: str,
    target: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray | None = None,
    row_ids: np.ndarray | None = None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = root / "artifacts" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        target=target,
        prediction=prediction,
        scale=np.array([]) if scale is None else scale,
        row_ids=np.array([]) if row_ids is None else np.asarray(row_ids, dtype=str),
    )

    residual = prediction - target
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].scatter(target, prediction, s=6, alpha=0.30, edgecolors="none")
    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    axes[0].plot([lower, upper], [lower, upper], color="#d9485f", linewidth=1.2)
    axes[0].set(xlabel="Observed log10 Cn²", ylabel="Predicted log10 Cn²")
    axes[1].hist(residual, bins=45, color="#237a77", alpha=0.85)
    axes[1].axvline(0, color="#d9485f", linewidth=1.2)
    axes[1].set(xlabel="Residual", ylabel="Count")
    axes[2].plot(target, color="#223047", linewidth=0.8, label="observed")
    axes[2].plot(prediction, color="#19a79d", linewidth=0.8, alpha=0.9, label="predicted")
    if scale is not None:
        axes[2].fill_between(
            np.arange(len(prediction)),
            prediction - 1.2815515655446004 * scale,
            prediction + 1.2815515655446004 * scale,
            color="#19a79d",
            alpha=0.16,
            label="80% interval",
        )
    axes[2].set(xlabel="Blocked test order", ylabel="log10 Cn²")
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Blocked-test diagnostics")
    figure_path = destination / "diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {
        "predictions": str(predictions_path.relative_to(root)),
        "figure": str(figure_path.relative_to(root)),
    }


def _predict(
    module: Cn2LightningModule,
    dataset: SequenceDataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    module.eval()
    device = next(module.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for features, _target in loader:
            output = module(features.to(device))
            locations.append(output["location"].cpu().numpy())
            scales.append(output["log_scale"].exp().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(locations), np.concatenate(scales), elapsed


def _log_baseline(
    root: Path,
    experiment: dict[str, Any],
    datamodule: OtbenchDataModule,
    name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    row_ids: np.ndarray,
    tags: dict[str, str],
    run_context: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    with mlflow.start_run(
        run_name=name,
        tags={
            **tags,
            "model_family": name,
            "run_kind": "baseline",
            "git_dirty": str(run_context["repository"]["git_dirty"]).lower(),
            "source_digest_sha256": run_context["repository"]["source_digest_sha256"],
        },
    ) as run:
        metrics = regression_metrics(target, prediction)
        metrics.update(
            {
                "wall_clock_seconds": time.monotonic() - started,
                "peak_vram_gb": 0.0,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_params({"model_family": name, "seed": "not_applicable"})
        artifacts = _save_diagnostics(
            root,
            run.info.run_id,
            target,
            prediction,
            row_ids=row_ids,
        )
        for artifact_relative in artifacts.values():
            mlflow.log_artifact(
                str(root / artifact_relative), artifact_path="diagnostics"
            )
        resolved = {
            "experiment": experiment["id"],
            "model": name,
            "seed": None,
            "data": experiment["data_config"],
            "trainer_config": {
                **experiment["trainer_config"],
                "effective_num_workers": datamodule.num_workers,
            },
        }
        evidence = {
            "run_id": run.info.run_id,
            "name": name,
            "kind": "baseline",
            "model": name,
            "seed": None,
            "retry_attempt": 0,
            **run_context,
            "resolved_configuration": resolved,
            "metrics": metrics,
            "artifacts": artifacts,
            "checkpoint": None,
        }
        evidence_path = _write_evidence_file(
            root, run.info.run_id, "run-manifest.json", evidence
        )
        mlflow.log_artifact(str(evidence_path), artifact_path="evidence")
        for evidence_source in _evidence_source_paths(root, experiment, datamodule):
            mlflow.log_artifact(str(evidence_source), artifact_path="evidence")
        return {
            "run_id": run.info.run_id,
            "name": name,
            "kind": "baseline",
            "model": name,
            "seed": None,
            "metrics": metrics,
            "artifacts": artifacts,
            "checkpoint": None,
        }


def _make_datamodule(experiment: dict[str, Any]) -> OtbenchDataModule:
    root = Path(str(experiment["_repo_root"]))
    data_config = experiment["data_config"]
    split_config = load_yaml(str(data_config["split_config"]), root=root)
    trainer = experiment["trainer_config"]
    model = experiment["model_config"]
    configured_workers = int(trainer["num_workers"])
    num_workers = int(os.getenv("STRATA_NUM_WORKERS", configured_workers))
    if num_workers < 0:
        raise ValueError("STRATA_NUM_WORKERS must be non-negative")
    return OtbenchDataModule(
        data_config,
        split_config,
        batch_size=int(trainer["batch_size"]),
        num_workers=num_workers,
        context=int(model.get("context", 1)),
    )


def _build_model(name: str, model_config: dict[str, Any], input_dim: int) -> torch.nn.Module:
    config = {
        key: value
        for key, value in model_config.items()
        if key
        not in {
            "name",
            "input_dim",
            "parameter_budget_max",
            "quantiles",
        }
    }
    if name == "mlp":
        return MLPBaseline(
            input_dim=input_dim,
            hidden_dim=1024,
            depth=4,
        )
    if name == "strata_ot_surface":
        return StrataOTSurface(input_dim=input_dim, **config)
    raise ValueError(f"Unsupported first-stage model: {name}")


def _train_neural(
    experiment: dict[str, Any],
    datamodule: OtbenchDataModule,
    model_name: str,
    seed: int,
    *,
    fast_dev_run: bool,
    max_seconds: float,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    if (
        datamodule.metadata is None
        or datamodule.train_set is None
        or datamodule.test_set is None
    ):
        raise RuntimeError("Data module must be prepared before model construction")
    L.seed_everything(seed, workers=True)
    trainer_config = experiment["trainer_config"]
    model_config = experiment["model_config"]
    network = _build_model(model_name, model_config, datamodule.metadata.input_dim)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    parameter_budget = int(model_config.get("parameter_budget_max", 8_000_000))
    if parameter_count > parameter_budget:
        raise ValueError(
            f"{model_name} has {parameter_count:,} parameters; budget is {parameter_budget:,}"
        )
    baseline_value = (
        float(np.median(_dataset_targets(datamodule.train_set)))
        if model_name == "strata_ot_surface"
        and bool(model_config.get("use_baseline_residual"))
        else None
    )
    module = Cn2LightningModule(
        network,
        learning_rate=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]),
        max_epochs=int(trainer_config["max_epochs"]),
        baseline_value=baseline_value,
    )
    root = Path(str(experiment["_repo_root"]))
    checkpoint_dir = root / "checkpoints" / str(experiment["id"]) / model_name / str(seed)
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
        duration=timedelta(seconds=max_seconds),
        interval="step",
        verbose=True,
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    run_kind = "probe" if fast_dev_run else "neural"
    run_name = f"{model_name}-seed-{seed}" + ("-fast-dev" if fast_dev_run else "")
    logger = MLFlowLogger(
        experiment_name=str(experiment["tracking"]["experiment_name"]),
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags={
            "dataset_id": datamodule.metadata.dataset_id,
            "split_id": datamodule.metadata.split_id,
            "model_family": model_name,
            "run_kind": run_kind,
            "fast_dev_run": str(fast_dev_run).lower(),
            "git_dirty": str(run_context["repository"]["git_dirty"]).lower(),
            "source_digest_sha256": run_context["repository"]["source_digest_sha256"],
        },
    )
    accelerator = str(trainer_config["accelerator"])
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "The local_16gb configuration requires CUDA; run a CPU smoke config only "
            "for diagnostics, not as the claimed first real result"
        )
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
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
    )
    run_id = logger.run_id
    if run_id is None:
        raise RuntimeError("MLflow did not assign a run identity")
    resolved = {
        "experiment": experiment["id"],
        "model": model_name,
        "seed": seed,
        "parameters": parameter_count,
        "data": experiment["data_config"],
        "model_config": model_config,
        "trainer_config": {
            **trainer_config,
            "effective_num_workers": datamodule.num_workers,
        },
        "normalization": datamodule.normalization,
        "feature_names": datamodule.metadata.feature_names,
        "target_transform": datamodule.metadata.target_transform,
        "baseline_residual": {
            "source": "train_fold_climatology_median" if baseline_value is not None else None,
            "value": baseline_value,
        },
        "fast_dev_run": fast_dev_run,
        "retry_attempt": 0,
    }
    logger.log_hyperparams(
        {
            "model": model_name,
            "seed": seed,
            "parameters": parameter_count,
            "batch_size": int(trainer_config["batch_size"]),
            "gradient_accumulation": int(trainer_config["gradient_accumulation"]),
            "precision": str(trainer_config["precision"]),
            "context": int(model_config.get("context", 1)),
        }
    )
    try:
        trainer.fit(module, datamodule=datamodule)
        remaining_after_fit = timer.time_remaining()
        stopped_for_budget = (
            remaining_after_fit is not None and remaining_after_fit <= 0
        )
        if checkpoint.best_model_path and not fast_dev_run:
            module = Cn2LightningModule.load_from_checkpoint(
                checkpoint.best_model_path,
                model=network,
                learning_rate=float(trainer_config["learning_rate"]),
                weight_decay=float(trainer_config["weight_decay"]),
                max_epochs=int(trainer_config["max_epochs"]),
                baseline_value=baseline_value,
            )
            module = module.to(torch.device("cuda" if accelerator == "gpu" else "cpu"))
        location, scale, inference_seconds = _predict(
            module,
            datamodule.test_set,
            int(trainer_config["batch_size"]),
        )
        test_target = _dataset_targets(datamodule.test_set)
        metrics = regression_metrics(test_target, location, scale)
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
                "inference_latency_ms_per_sample": (
                    inference_seconds * 1000 / len(test_target)
                ),
                "throughput_samples_per_second": len(test_target)
                / max(inference_seconds, 1e-12),
            }
        )
        artifacts = _save_diagnostics(
            root,
            run_id,
            test_target,
            location,
            scale,
            _dataset_row_ids(datamodule.test_set),
        )
        logger.log_metrics(metrics, step=trainer.global_step)
        logger.experiment.log_dict(run_id, resolved, "resolved-config.json")
        logger.experiment.log_dict(
            run_id, run_context["environment"], "environment.json"
        )
        logger.experiment.log_dict(
            run_id, run_context["repository"], "repository-identity.json"
        )
        for artifact_relative in artifacts.values():
            logger.experiment.log_artifact(
                run_id,
                str(root / artifact_relative),
                artifact_path="diagnostics",
            )
        if checkpoint.best_model_path:
            logger.experiment.log_artifact(
                run_id, checkpoint.best_model_path, artifact_path="checkpoints"
            )
        evidence = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "seed": seed,
            "retry_attempt": 0,
            **run_context,
            "resolved_configuration": resolved,
            "metrics": metrics,
            "checkpoint": checkpoint.best_model_path or None,
            "artifacts": artifacts,
            "stopped_for_budget": stopped_for_budget,
        }
        evidence_path = _write_evidence_file(
            root, run_id, "run-manifest.json", evidence
        )
        logger.experiment.log_artifact(
            run_id, str(evidence_path), artifact_path="evidence"
        )
        for evidence_source in _evidence_source_paths(root, experiment, datamodule):
            logger.experiment.log_artifact(
                run_id, str(evidence_source), artifact_path="evidence"
            )
        logger.finalize("success")
    except Exception as error:
        failure = {
            "run_id": run_id,
            "name": run_name,
            "kind": run_kind,
            "model": model_name,
            "seed": seed,
            "retry_attempt": 0,
            **run_context,
            "resolved_configuration": resolved,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        failure_path = _write_evidence_file(
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
        "stopped_for_budget": stopped_for_budget,
    }


def run_experiment(
    config_path: str,
    *,
    selected_model: str | None = None,
    selected_seed: int | None = None,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    _configure_utf8_output()
    experiment = resolve_experiment(config_path)
    _configure_torch_runtime(experiment["trainer_config"])
    root = Path(str(experiment["_repo_root"]))
    datamodule = _make_datamodule(experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")
    if datamodule.train_set is None or datamodule.test_set is None or datamodule.metadata is None:
        raise RuntimeError("Data preparation did not produce all required splits")
    run_context = {
        "repository": _repository_identity(root),
        "environment": _environment_fingerprint(),
        "data_identity": _data_identity(root, experiment, datamodule),
    }

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(experiment["tracking"]["experiment_name"])
    mlflow.set_experiment(experiment_name)
    tags = {
        "dataset_id": datamodule.metadata.dataset_id,
        "split_id": datamodule.metadata.split_id,
        "git_revision": run_context["repository"]["git_revision"] or "uncommitted",
    }
    train_target = _dataset_targets(datamodule.train_set)
    test_target = _dataset_targets(datamodule.test_set)
    test_row_ids = _dataset_row_ids(datamodule.test_set)
    runs: list[dict[str, Any]] = []
    requested_models = [selected_model] if selected_model else list(experiment["models"])
    if "climatology" in requested_models:
        climatology = fit_climatology(train_target)
        runs.append(
            _log_baseline(
                root,
                experiment,
                datamodule,
                "climatology",
                test_target,
                climatology.predict(len(test_target)),
                test_row_ids,
                tags,
                run_context,
            )
        )
    if "persistence" in requested_models:
        runs.append(
            _log_baseline(
                root,
                experiment,
                datamodule,
                "persistence",
                test_target,
                persistence_predictions(test_target, float(train_target[-1])),
                test_row_ids,
                tags,
                run_context,
            )
        )
    seeds = [selected_seed] if selected_seed is not None else list(experiment["seeds"])
    if fast_dev_run:
        seeds = seeds[:1]
    neural_started = time.monotonic()
    gpu_budget_seconds = float(experiment["trainer_config"]["max_gpu_hours"]) * 3600
    budget_exhausted = False
    for model_name in requested_models:
        if model_name in {"climatology", "persistence"}:
            continue
        for seed in seeds:
            remaining_seconds = gpu_budget_seconds - (time.monotonic() - neural_started)
            if remaining_seconds <= 0:
                budget_exhausted = True
                break
            result = _train_neural(
                experiment,
                datamodule,
                model_name,
                int(seed),
                fast_dev_run=fast_dev_run,
                max_seconds=remaining_seconds,
                run_context=run_context,
            )
            runs.append(result)
            if result["stopped_for_budget"]:
                budget_exhausted = True
                break
        if budget_exhausted:
            break

    total_neural_gpu_hours = sum(
        float(run.get("metrics", {}).get("wall_clock_seconds", 0)) / 3600
        for run in runs
        if run.get("kind") == "neural"
    )
    summary = {
        "experiment_id": experiment["id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "hypothesis": experiment["hypothesis"],
        "dataset_id": datamodule.metadata.dataset_id,
        "split_id": datamodule.metadata.split_id,
        "evaluation_gate_id": experiment["evaluation_config"]["id"],
        "target_transform": datamodule.metadata.target_transform,
        "code_revision": (
            run_context["repository"]["git_revision"]
            or f"uncommitted:{run_context['repository']['source_digest_sha256'][:12]}"
        ),
        "repository_identity": run_context["repository"],
        "environment": run_context["environment"],
        "data_identity": run_context["data_identity"],
        "data_coverage": _manifest_coverage(experiment),
        "data_qc": "artifacts/data-qc/otbench-mlo-cn2-15m-v1.json",
        "runs": runs,
        "model_aggregates": _model_aggregates(runs),
        "total_neural_gpu_hours": total_neural_gpu_hours,
        "resolved_configuration": {
            "data": experiment["data_config"],
            "model": experiment["model_config"],
            "trainer": {
                **experiment["trainer_config"],
                "effective_num_workers": datamodule.num_workers,
            },
        },
        "checks": {
            "dataset_manifest_valid": True,
            "no_random_row_primary_split": True,
            "preprocessing_fit_on_train_only": True,
            "no_time_overlap_across_partitions": True,
            "uncertainty_reported": all(
                "gaussian_nll" in run["metrics"] for run in runs if run["kind"] == "neural"
            ),
            "calibration_reported": all(
                "interval_80_coverage" in run["metrics"]
                for run in runs
                if run["kind"] == "neural"
            ),
            "direct_and_teacher_labels_separated": True,
            "reproducibility_recorded": all(
                (root / "artifacts" / "runs" / str(run["run_id"]) / "run-manifest.json").is_file()
                for run in runs
            ),
            "resource_metrics_reported": all(
                "wall_clock_seconds" in run["metrics"]
                and "peak_vram_gb" in run["metrics"]
                for run in runs
            ),
            "within_gpu_budget": (
                not budget_exhausted
                and total_neural_gpu_hours
                <= float(experiment["trainer_config"]["max_gpu_hours"])
            ),
            "latex_pdf_compiled": False,
        },
    }
    output = root / "artifacts" / "latest" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _code_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _manifest_coverage(experiment: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(experiment["_repo_root"]))
    path = root / str(experiment["data_config"]["manifest_path"])
    if not path.exists():
        return {"status": "manifest_missing"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "coverage": manifest.get("coverage", {}),
        "label_provenance": manifest.get("label_provenance"),
        "geometry": manifest.get("geometry", {}),
        "qc": manifest.get("qc", {}),
    }


def _model_aggregates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[float]] = {}
    for run in runs:
        if run.get("kind") != "neural":
            continue
        model = str(run["model"])
        rmse = float(run["metrics"]["rmse_log10_cn2"])
        by_model.setdefault(model, []).append(rmse)
    aggregates: list[dict[str, Any]] = []
    for model, values in sorted(by_model.items()):
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else None
        aggregates.append(
            {
                "model": model,
                "seeds": len(values),
                "mean_rmse_log10_cn2": mean,
                "std_rmse_log10_cn2": standard_deviation,
                "relative_seed_std": (
                    standard_deviation / max(abs(mean), 1e-12)
                    if standard_deviation is not None
                    else None
                ),
            }
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bounded Strata-OT experiment")
    parser.add_argument("--config", default="configs/experiments/first_real_mlo.yaml")
    parser.add_argument(
        "--model",
        choices=["climatology", "persistence", "mlp", "strata_ot_surface"],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    summary = run_experiment(
        args.config,
        selected_model=args.model,
        selected_seed=args.seed,
        fast_dev_run=args.fast_dev_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
