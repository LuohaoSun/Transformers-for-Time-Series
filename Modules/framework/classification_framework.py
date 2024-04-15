# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from abc import ABC, abstractmethod
from torchmetrics import Accuracy, F1Score, Precision, Recall
from .framework_base import FrameworkBase

class ClassificationFramework(FrameworkBase, ABC):
    '''
    用于时间序列多分类任务的基础模型类，针对性地实现了损失、训练、验证、预测、可视化等，且forward方法已经在父类中实现。

    HOW TO USE:
    1. 子类实现backbone, head属性定制模型参数。
    2. 子类实现num_classes属性定制任务参数。
    3. 子类实现configure_optimizers方法定制训练参数。
    NOTE: 子类不需要实现forward方法！

    input: (batch_size, in_seq_len, in_features)
    output: (batch_size, num_classes)
    '''

    @abstractmethod
    def __init__(self,
                 # model params
                 backbone: nn.Module,
                 head: nn.Module,
                 num_classes: int,
                 # training params
                 lr: float,
                 max_epochs: int,
                 max_steps: int,
                 ) -> None:
        super().__init__(backbone, head, lr, max_epochs, max_steps)
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(output, target)

    def compute_and_log_metrics(self, y_hat: Tensor, y: Tensor, prefix: str, **log_kwargs) -> None:

        metrics = {
            'accuracy': self.accuracy(y_hat, y),
            # 'f1': self.f1(y_hat, y),
            # 'precision': self.precision(y_hat, y),
            # 'recall': self.recall(y_hat, y)
        }
        for metric_name, metric_value in metrics.items():
            self.log(f'{prefix}_{metric_name}', metric_value, **log_kwargs)

    def training_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Any]:
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        self.compute_and_log_metrics(y_hat, y, 'train', on_step=True, on_epoch=False,
                                     prog_bar=True, logger=True)

        return {'loss': loss, 'y': y, 'y_hat': y_hat}

    def validation_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.compute_and_log_metrics(y_hat, y, 'val',
                                     prog_bar=True, logger=True)

        return {'loss': loss, 'y': y, 'y_hat': y_hat}

    def test_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.compute_and_log_metrics(y_hat, y, 'test', on_step=False, on_epoch=True,
                                     prog_bar=True, logger=True)

        return {'loss': loss, 'y': y, 'y_hat': y_hat}

    def predict_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        return {'y': y, 'y_hat': y_hat}

