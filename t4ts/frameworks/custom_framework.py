# Author: Sun LuoHao
# All rights reserved

from typing import Any, Callable, Iterable, Mapping

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..callbacks.classification_callbacks import \
    ComputeAndLogMetrics2Tensorboard
from .framework_base import FrameworkBase


class CustomFramework(FrameworkBase):

    def __init__(
        self,
        # backbone:
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        loss_fn: Callable[..., Any],
        task_callbacks: list[L.Callback] = [],
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.loss_fn = loss_fn
        self.task_callbacks = task_callbacks

    def get_task_callbacks(self) -> list[L.Callback]:
        return self.task_callbacks

    def loss(self, *args, **kwargs) -> Tensor:
        return self.loss_fn(*args, **kwargs)

    def framework_forward(
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
    ) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        return: (batch_size, out_seq_len, num_classes)
        """
        assert len(x.shape) == 3, f"Expected 3D input, got {x.shape}"
        x = backbone(x)
        x = neck(x)
        x = head(x)
        return x

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable[..., Any]
    ) -> Mapping[str, Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)

        return {"loss": loss, "y": y, "y_hat": y_hat}
