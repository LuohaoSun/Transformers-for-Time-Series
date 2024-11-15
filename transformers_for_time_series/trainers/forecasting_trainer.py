# Author: Sun LuoHao
# All rights reserved
import time
from functools import partial
from typing import Callable, Iterable

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from ..callbacks.default_callbacks import (
    EarlyStopping,
    LogLoss,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from ..callbacks.forecasting_callbacks import ComputeMetricsAndLog, ViAndLog
from ..utils import get_loss_fn
from .trainer_base import TrainerBase


class ForecastingTrainer(TrainerBase):
    """
    NOTE: a batch must be a tuple of (input: Tensor, target: Tensor), where the
    input will be directly passed to the model, and the target will be used to
    compute the loss.
    NOTE: the input/target tensor must be of shape (batch_size, seq_len, n_features).
    """

    def __init__(
        self,
        max_epochs: int,
        # optimizer params:
        lr: float = 1e-3,
        loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "mse",
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        # callbacks params:
        vi_every_n_epochs: int = 20,
        early_stopping_patience: int = 10,
        # Trainer params:
        gradient_clip_algorithm: str | None = None,
        gradient_clip_val: float | None = None,
        accelerator: str = "auto",
        devices: list[int] | str | int = "auto",
        precision: int | str = 32,
        log_every_n_steps: int = 1,
        log_save_dir: str = ".",
        log_save_name: str | None = None,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.log_save_dir = log_save_dir
        time_stamp = time.strftime("%Y年%m月%d日%H时%M分%S秒", time.localtime())
        self.version = log_save_name and f"{log_save_name}-{time_stamp}"
        self.vi_every_n_epochs = vi_every_n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def configure_trainer(self) -> L.Trainer:
        logger = TensorBoardLogger(
            save_dir=self.log_save_dir,
            name="forecasting trainer logs",
            version=self.version,
        )
        callbacks = [
            LogLoss(),
            ViAndLog(every_n_epochs=self.vi_every_n_epochs),
            ComputeMetricsAndLog(),
            EarlyStopping(monitor="val_loss", patience=self.early_stopping_patience),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            RichProgressBar(),
            RichModelSummary(max_depth=3),
        ]
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,  # type: ignore
            log_every_n_steps=self.log_every_n_steps,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            gradient_clip_val=self.gradient_clip_val,
        )

        return trainer

    def configure_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return get_loss_fn(self.loss_fn)

    def configure_optimizer(self) -> Callable[[Iterable[Parameter]], Optimizer]:
        if self.optimizer.lower() == "adam":
            return partial(torch.optim.Adam, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "sgd":
            return partial(torch.optim.SGD, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == "adamw":
            return partial(
                torch.optim.AdamW, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
