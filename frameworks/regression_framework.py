# Author: Sun LuoHao
# All rights reserved

import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Mapping, Iterable, Callable, Any
from torch import Tensor

from .framework_base import FrameworkBase
from .callbacks.regression_callbacks import (
    ComputeAndLogMetrics2Tensorboard,
)


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
        # 必须使用logger=False，否则会报错.
        self.out_seq_len = out_seq_len
        self.num_features = num_features
        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.head = nn.Linear(backbone_out_features, num_features)

    @property
    def _task_callbacks(self) -> list[L.Callback]:
        return [ComputeAndLogMetrics2Tensorboard(self.num_features)]

    @property
    def _loss(self) -> Callable:
        return nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        return: (batch_size, out_seq_len, num_classes)
        """
        out_seq_len = self.out_seq_len if self.out_seq_len > 0 else x.size(1)
        x = self.backbone(x)[:, -out_seq_len, :]
        x = self.head(x)
        return x

    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        x, y = batch
        assert len(x.shape) == 3, f"Expected 3D input, got {x.shape}"
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
