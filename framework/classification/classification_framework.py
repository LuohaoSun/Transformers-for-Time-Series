# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Mapping, Iterable
from torch import Tensor
from ..framework_base.framework_base import FrameworkBase
from .classification_functionalities import ComputeAndLogMetrics2Tensorboard


class ClassificationFramework(FrameworkBase):

    def __init__(
        self,
        # backbone:
        backbone: nn.Module,
        # task params:
        hidden_features: int,
        out_seq_len: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        # TODO: remove arg hidden_features
        # 必须使用logger=False，否则会报错.
        self.out_seq_len = out_seq_len
        self.num_classes = num_classes
        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.head = nn.Linear(hidden_features, num_classes)
        self.loss = nn.CrossEntropyLoss()

    @property
    def task_functionalities(self) -> list[L.Callback]:
        return [ComputeAndLogMetrics2Tensorboard(self.num_classes)]

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
        if len(y.size()) == 2:
            y = y.unsqueeze(1)
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
