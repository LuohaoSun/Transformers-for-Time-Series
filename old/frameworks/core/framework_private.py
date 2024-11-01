# Author: Sun LuoHao
# All rights reserved
from typing import Any, Callable, Dict, Iterable, Mapping, Union, final

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...utils import get_loss_fn
from ..callbacks.default_callbacks import get_default_callbacks

"""
框架的私有属性和方法，主要在调用公共方法fit时初始化。
"""
class FrameworkBasePrivate(L.LightningModule):
    # ==============================
    # 以下是框架私有方法和属性：
    _framework_optimizer: Optimizer
    _framework_logger: TensorBoardLogger
    _framework_trainer: L.Trainer
    _framework_loss: Callable
    _framework_callbacks: list[L.Callback]
    _framework_backbone: nn.Module
    _framework_neck: nn.Module
    _framework_head: nn.Module

    def _configure_framework(
        self,
        datamodule: L.LightningDataModule | None,
        # logging params:
        log_every_n_steps: int,
        # training params:
        lr: float,
        max_epochs: int,
        max_steps: int,
        early_stopping_patience: int,
        accelerator: str,
        compile_model: bool,
        custom_loss_fn: Callable | str | None,
        **trainer_kwargs,
    ) -> L.LightningModule:
        """
        configuration of the following properties before training:
        - self._framework_optimizer
        - self._framework_logger
        - self._framework_callbacks
        - self._framework_trainer
        - self._framework_loss
        """
        # save hyperparameters:
        if datamodule is not None:
            self.hparams.update(datamodule.hparams)
        self.hparams.update(
            {
                "lr": lr,
                "max_epochs": max_epochs,
                "max_steps": max_steps,
            }
        )

        self._framework_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._framework_logger = TensorBoardLogger(
            save_dir=".", name="lightning_logs", default_hp_metric=False
        )
        self._framework_callbacks = (
            get_default_callbacks(early_stopping_patience) + self.get_task_callbacks()
        )
        self._framework_trainer = L.Trainer(
            max_epochs=max_epochs,
            max_steps=max_steps,
            callbacks=self._framework_callbacks,
            logger=self._framework_logger,
            accelerator=accelerator,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=False,
            **trainer_kwargs,
        )
        self._framework_loss = (
            get_loss_fn(custom_loss_fn) if custom_loss_fn else self.loss
        )

        if compile_model:
            self = torch.compile(self)

        return self  # type: ignore
