from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor, nn

from strata_ot.models.components import gaussian_nll, pinball_loss


class Cn2LightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        baseline_value: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.baseline_value = baseline_value

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        baseline = (
            features.new_full((features.shape[0],), self.baseline_value)
            if self.baseline_value is not None
            else None
        )
        output: dict[str, Tensor] = self.model(features, baseline=baseline)
        return output

    def _step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        features, target = batch
        output = self(features)
        nll = gaussian_nll(output["location"], output["log_scale"], target)
        quantile = pinball_loss(output["quantiles"], target)
        loss = nll + 0.20 * quantile
        error = output["location"] - target
        metrics: dict[str, Any] = {
            f"{stage}/loss": loss,
            f"{stage}/nll": nll,
            f"{stage}/mae_log10_cn2": error.abs().mean(),
            f"{stage}/rmse_log10_cn2": error.square().mean().sqrt(),
            f"{stage}/bias_log10_cn2": error.mean(),
        }
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage != "train",
            batch_size=len(target),
        )
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "validation")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, "test")

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
