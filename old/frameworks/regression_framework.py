# Author: Sun LuoHao
# All rights reserved

from typing import Any, Callable, Iterable, Mapping

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..callbacks.regression_callbacks import ComputeAndLogMetrics2Tensorboard
from .framework_base import FrameworkBase


class RegressionFramework(FrameworkBase):
    def __init__(
        self,
        # backbone:
        backbone: nn.Module,
        backbone_out_features: int,
        # task params:
        out_seq_len: int,
        num_features: int,
    ) -> None:
        super().__init__()
        # TODO: 回归任务和分类任务相似，可以考虑合并（回归框架直接继承分类框架），callback同理
        self.hparams.update({"backbone": backbone})
        self.out_seq_len = out_seq_len
        self.num_features = num_features

        self.backbone = backbone
        self.head = nn.Linear(backbone_out_features, num_features)

    def get_task_callbacks(self) -> list[L.Callback]:
        return [ComputeAndLogMetrics2Tensorboard(self.num_features)]

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: (b, l_out, d_out)
        target: (b, l_out, d_out)
        """
        return F.mse_loss(input, target)

    def framework_forward(
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
    ) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        return: (batch_size, out_seq_len, num_classes)
        """
        out_seq_len = self.out_seq_len if self.out_seq_len > 0 else x.size(1)
        x = backbone(x)[:, -out_seq_len, :]
        x = head(x)
        return x

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Mapping[str, Tensor]:
        x, y = batch
        assert len(x.shape) == 3, f"Expected 3D input, got {x.shape}"
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)

        return {"loss": loss, "y": y, "y_hat": y_hat}
