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
from .forecasting_callbacks import ComputeAndLogMetrics2Tensorboard


class ForecastingFramework(FrameworkBase, ABC):
    """
    input: Tensor of shape (b, l_in, d_in)
    output: mask of shape (b, l_out, d_out)
    in a forecasting task, the d_out could be different from d_in.
    """

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        head: nn.Module,
        # training params
        lr: float,
        max_epochs: int,
        max_steps: int,
        loss_type: str = "mse",  # TODO: add more loss types
    ) -> None:

        super().__init__(
            backbone=backbone,
            head=head,
            additional_callbacks=[ComputeAndLogMetrics2Tensorboard()],
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
        )
        self.loss_type = loss_type

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(output, target)

    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:
        '''
        batch: (x, y)
        x: (b, l_in, d_in)
        y: (b, l_out, d_out)
        '''

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
