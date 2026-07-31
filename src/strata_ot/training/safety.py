from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import lightning as L
import psutil  # type: ignore[import-untyped]
import torch


def _gib(value: float) -> float:
    return value / (1024**3)


def hardware_snapshot(root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(root.anchor or str(root))
    snapshot: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "cpu_percent": float(psutil.cpu_percent(interval=1.0)),
        "free_ram_gib": _gib(float(memory.available)),
        "process_rss_gib": _gib(float(psutil.Process().memory_info().rss)),
        "workspace_drive_free_gib": _gib(float(disk.free)),
        "gpu": None,
    }
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        first = completed.stdout.strip().splitlines()[0]
        values = [value.strip() for value in first.split(",")]
        if len(values) == 3:
            snapshot["gpu"] = {
                "free_memory_gib": float(values[0]) / 1024,
                "used_memory_gib": float(values[1]) / 1024,
                "temperature_c": float(values[2]),
                "process_peak_allocated_gib": (
                    _gib(float(torch.cuda.max_memory_allocated()))
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "process_peak_reserved_gib": (
                    _gib(float(torch.cuda.max_memory_reserved()))
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
    return snapshot


def resource_peaks(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate board and process CUDA peaks without conflating them."""
    gpu_samples = [
        sample["gpu"]
        for sample in samples
        if isinstance(sample.get("gpu"), dict)
    ]
    return {
        "peak_total_board_vram_gib": max(
            (float(gpu["used_memory_gib"]) for gpu in gpu_samples),
            default=0.0,
        ),
        "peak_process_allocated_vram_gib": max(
            (float(gpu["process_peak_allocated_gib"]) for gpu in gpu_samples),
            default=0.0,
        ),
        "peak_process_reserved_vram_gib": max(
            (float(gpu["process_peak_reserved_gib"]) for gpu in gpu_samples),
            default=0.0,
        ),
    }


def preflight_hardware(root: Path, safety: dict[str, Any]) -> dict[str, Any]:
    snapshot = hardware_snapshot(root)
    cpu_limit = float(safety["start_max_cpu_percent"])
    if float(snapshot["cpu_percent"]) > cpu_limit:
        sustained = float(safety["start_cpu_sustained_seconds"])
        deadline = time.monotonic() + sustained
        while time.monotonic() < deadline:
            time.sleep(min(15.0, max(0.1, deadline - time.monotonic())))
            refreshed = hardware_snapshot(root)
            if float(refreshed["cpu_percent"]) <= cpu_limit:
                snapshot = refreshed
                break
        else:
            raise RuntimeError(
                f"CPU remained above {cpu_limit:g}% for {sustained:g} seconds"
            )
    if float(snapshot["free_ram_gib"]) < float(safety["start_min_free_ram_gib"]):
        raise RuntimeError("Insufficient free RAM for a new training run")
    gpu = snapshot.get("gpu")
    if not isinstance(gpu, dict):
        raise RuntimeError("nvidia-smi GPU telemetry is required for the bounded run")
    if float(gpu["free_memory_gib"]) < float(safety["start_min_free_gpu_gib"]):
        raise RuntimeError("Insufficient free GPU memory for a new training run")
    if float(gpu["temperature_c"]) > float(
        safety["start_max_gpu_temperature_c"]
    ):
        raise RuntimeError("GPU temperature exceeds the new-run threshold")
    return snapshot


class TrainingProcessLock:
    """Exclusive lock preventing a second Strata horizon training process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> TrainingProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"Another horizon training lock exists: {self.path}"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_unix": time.time()}, handle)
        self.acquired = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False


class HardwareSafetyCallback(L.Callback):
    """Periodic hardware guard that checkpoints once before a graceful stop."""

    def __init__(
        self,
        *,
        root: Path,
        safety: dict[str, Any],
        emergency_checkpoint: Path,
    ) -> None:
        super().__init__()
        self.root = root
        self.safety = safety
        self.emergency_checkpoint = emergency_checkpoint
        self.samples: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self._last_checked = 0.0
        self._hot_since: float | None = None
        self._checkpoint_written = False

    def _request_stop(self, trainer: L.Trainer, reason: str) -> None:
        if self.stop_reason is not None:
            return
        self.stop_reason = reason
        if not self._checkpoint_written:
            self.emergency_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self.emergency_checkpoint))
            self._checkpoint_written = True
        trainer.should_stop = True

    def _check(self, trainer: L.Trainer) -> None:
        now = time.monotonic()
        interval = float(self.safety["monitor_interval_seconds"])
        if self._last_checked and now - self._last_checked < interval:
            return
        self._last_checked = now
        snapshot = hardware_snapshot(self.root)
        self.samples.append(snapshot)
        if float(snapshot["free_ram_gib"]) < float(
            self.safety["stop_min_free_ram_gib"]
        ):
            self._request_stop(trainer, "free_ram_below_limit")
            return
        gpu = snapshot.get("gpu")
        if not isinstance(gpu, dict):
            self._request_stop(trainer, "gpu_telemetry_unavailable")
            return
        if float(gpu["used_memory_gib"]) > float(
            self.safety["stop_max_board_vram_gib"]
        ):
            self._request_stop(trainer, "board_vram_above_limit")
            return
        process_peak = max(
            float(gpu["process_peak_allocated_gib"]),
            float(gpu["process_peak_reserved_gib"]),
        )
        if process_peak > float(self.safety["stop_max_process_vram_gib"]):
            self._request_stop(trainer, "process_vram_above_limit")
            return
        temperature = float(gpu["temperature_c"])
        if temperature > float(self.safety["stop_max_gpu_temperature_c"]):
            if self._hot_since is None:
                self._hot_since = now
            elif now - self._hot_since >= float(
                self.safety["stop_gpu_temperature_sustained_seconds"]
            ):
                self._request_stop(trainer, "gpu_temperature_sustained")
        else:
            self._hot_since = None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        self._check(trainer)

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        self._last_checked = 0.0
        self._check(trainer)

    def capture_final(self) -> dict[str, Any]:
        """Record an unconditional final sample for short and fast-dev runs."""
        snapshot = hardware_snapshot(self.root)
        self.samples.append(snapshot)
        return snapshot
