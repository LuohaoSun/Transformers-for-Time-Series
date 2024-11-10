# Author: Sun LuoHao
# All rights reserved

from typing import Any, Callable, Iterable, Mapping

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..callbacks.classification_callbacks import ComputeAndLogMetrics2Tensorboard
from .framework_base import FrameworkBase


class ClassificationFramework(FrameworkBase):
    def __init__(
        self,
        # backbone:
        backbone: nn.Module,
        backbone_out_features: int,
        # task params:
        out_seq_len: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.out_seq_len = out_seq_len
        self.num_classes = num_classes
        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.neck = nn.Identity()
        self.head = nn.Linear(backbone_out_features, num_classes)

    def get_task_callbacks(self) -> list[L.Callback]:
        return [ComputeAndLogMetrics2Tensorboard(self.num_classes)]

    def loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(y_hat, y)

    def framework_forward(
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
    ) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        return: (batch_size, out_seq_len, num_classes)
        """
        assert len(x.shape) == 3, f"Expected 3D input, got {x.shape}"
        out_seq_len = self.out_seq_len if self.out_seq_len > 0 else x.size(1)
        x = backbone(x)[:, -out_seq_len, :]
        x = head(x)
        return x

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable[..., Any]
    ) -> Mapping[str, Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)

        return {"loss": loss, "y": y, "y_hat": y_hat}
