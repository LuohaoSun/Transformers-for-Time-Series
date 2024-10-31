# Author: Sun LuoHao
# All rights reserved
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from functools import partial
from typing import Mapping, Iterable, Callable, Any
from torch import Tensor
from .framework_base import FrameworkBase
from .callbacks.forecasting_callbacks import ComputeMetricsAndLog
from .callbacks.autoencoding_callbacks import ViAndLog
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
import torch


class ForecastingLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., Tensor],
        optimizer: Callable[[Iterable[Parameter]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def model_step(self, batch: Iterable[Tensor]) -> Mapping[str, Tensor]:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat}

    def configure_optimizers(
        self,
    ) -> Mapping[str, Optimizer | LRScheduler | None]:
        optimizer = self.optimizer(self.model.parameters())
        lr_scheduler = self.lr_scheduler(optimizer) if self.lr_scheduler else None
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


class ForecastingTrainer:

    def __init__(
        self,
        max_epochs: int,
        # optimizer params:
        lr: float = 1e-3,
        loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "mse",
        optimizer: str = "adam",
        # callbacks params:
        vi_every_n_epochs: int = 20,
        # Trainer params:
        accelerator: str = "auto",
        devices: list[int] | str | int = "auto",
        precision: int | str = 32,
        log_every_n_steps: int = 1,
        default_root_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.partial_optimizer = self.get_partial_optimizer(optimizer, lr)

        callbacks = [
            ViAndLog(every_n_epochs=vi_every_n_epochs),
            ComputeMetricsAndLog(),
        ]
        self.trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,  # type: ignore
            log_every_n_steps=log_every_n_steps,
            default_root_dir=default_root_dir,
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloaders: DataLoader | L.LightningDataModule | None = None,
        val_dataloaders: DataLoader | L.LightningDataModule | None = None,
        datamodule: L.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        lit_model = ForecastingLitModule(
            model, self.loss_fn, self.partial_optimizer, None
        )
        self.trainer.fit(
            lit_model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )

    def get_loss_fn(
        self, loss_fn: str | Callable[..., Tensor]
    ) -> Callable[..., Tensor]:
        if isinstance(loss_fn, Callable):
            return loss_fn
        elif loss_fn.lower() in ["mse", "l2", "l2loss"]:
            return F.mse_loss
        elif loss_fn.lower() in ["mae", "l1", "l1loss"]:
            return F.l1_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def get_partial_optimizer(
        self, optimizer: str, lr: float
    ) -> Callable[[Iterable[Parameter]], Optimizer]:
        if optimizer.lower() == "adam":
            return partial(torch.optim.Adam, lr=lr)
        elif optimizer.lower() == "sgd":
            return partial(torch.optim.SGD, lr=lr)
        elif optimizer.lower() == "adamw":
            return partial(torch.optim.AdamW, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
