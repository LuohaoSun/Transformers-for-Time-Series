# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Mapping, Iterable
from torch import Tensor
from abc import ABC, abstractmethod
from ..framework_base.framework_base import FrameworkBase
from .classification_callbacks import ComputeAndLogMetrics2Tensorboard


class ClassificationFramework(FrameworkBase, ABC):

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        head: nn.Module,
        num_classes: int,
        # training params
        lr: float,
        max_epochs: int,
        max_steps: int,
    ) -> None:

        super().__init__(
            backbone=backbone,
            head=head,
            additional_callbacks=[ComputeAndLogMetrics2Tensorboard(num_classes)],
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
        )

        self.num_classes = num_classes

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(output, target)

    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        return {"loss": loss, "y": y, "y_hat": y_hat}

    def validation_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        return self.training_step(batch, batch_idx)

    def test_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        return self.training_step(batch, batch_idx)
