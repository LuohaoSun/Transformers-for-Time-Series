# Author: Sun LuoHao
# All rights reserved
from operator import call
import lightning as L
import torch.nn as nn
import torch

from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from abc import ABC, abstractmethod
from rich import print
from .default_callbacks import *


class FrameworkBase(L.LightningModule, ABC):
    """
    用于时间序列任务的基础模型类，封装了backbone-head的结构，并实现了forward, configure_optimizers, trainer, logger.
    子类通过实现loss, training_step, val_step, test_step, predict_step等方法定制各种下游任务模型。
    properties:
        backbone: nn.Module
        head: nn.Module

    methods:
        forward: Tensor -> Tensor
        fit: L.LightningDataModule -> None. Model trains itself.
        test: L.LightningDataModule -> None. Model tests itself.
        run_training: L.LightningDataModule -> None. Same as fit.
        run_testing: L.LightningDataModule -> None. Same as test.
    """

    def __init__(
        self,
        # model components
        backbone: nn.Module,
        head: nn.Module,
        # callbacks
        callbacks: list[L.Callback],
        # training params
        lr: float,
        max_epochs: int,
        max_steps: int,
    ) -> None:

        super().__init__()
        self.backbone = backbone
        self.head = head

        self.callbacks = callbacks

        self.lr = lr
        self.max_epochs = max_epochs
        self.max_steps = max_steps

    @property
    def framework_trainer(self) -> L.Trainer:
        framework_trainer = L.Trainer(
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            callbacks=self.callbacks,
            accelerator="auto",
        )
        return framework_trainer

    def forward(self, x: Tensor) -> Tensor:
        """
        input: (batch_size, in_seq_len, in_features)
        output: (batch_size, out_seq_len, out_features)
        """
        x = self.backbone(x)
        x = self.head(x)
        return x

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        # models of industrial usage do not need much hyperparameter tuning,
        # so I simply use Adam optimizer with default parameters here.
        # Flexibility can be added in the future.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

    def fit(self, datamodule: L.LightningDataModule):
        self.framework_trainer.fit(self, datamodule)
        return

    def test(self, datamodule: L.LightningDataModule):
        return self.framework_trainer.test(self, datamodule)

    def run_training(self, datamodule: L.LightningDataModule):
        return self.fit(datamodule)

    def run_testing(self, datamodule: L.LightningDataModule):
        return self.test(datamodule)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)  # type: ignore
        msg = f"""
            Hyperparameters:
            {self.hparams}
            =======================================
            =          Training Started.          =
            =======================================
            """
        print(msg)
        return super().on_train_start()

    @abstractmethod
    def loss(self, output: Tensor, target: Tensor) -> Tensor: ...

    @abstractmethod
    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]: ...

    @abstractmethod
    def validation_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]: ...

    @abstractmethod
    def test_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]: ...

    @abstractmethod
    def predict_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]: ...
