# Author: Sun LuoHao
# All rights reserved

import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Mapping, Iterable, Tuple, Callable, Optional
from torch import Tensor

from ..frameworks.utils import get_loss_fn

from .framework_base import FrameworkBase
from .callbacks.autoencoding_callbacks import ViAndLog
from .callbacks.classification_callbacks import ComputeAndLogMetrics2Tensorboard


class AnomalyDetectionFramework(FrameworkBase):
    """
    用于时间序列随机遮蔽重构任务的基础模型类，
    针对性地实现了损失、训练、验证、预测、可视化等，且forward方法已经在父类中实现。
    """

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        backbone_out_seq_len: int,
        backbone_out_features: int,
        # task params
        out_seq_len: int,
        out_features: int,
        detection_level: str,  # 'step' or 'sequence'
        threshold: float,
        # logging params
        vi_every_n_epochs: int = 10,
        figsize: Tuple[int, int] = (7, 3),
        # additional params
        custom_neck: Optional[nn.Module] = None,
        custom_head: Optional[nn.Module] = None,
    ) -> None:
        """
        Anomaly Detection Framework  is a framework composed of the autoencoding and classification frameworks.
        Methods:
            - detect_anomaly: Detects anomalies in the input sequence.
            - reconstruct: Reconstructs the input sequence. (Same as forward method)
        Args:
            backbone (nn.Module): The backbone module.
            head (nn.Module): The head module.

            detection_level (str): The level at which the anomaly detection should be performed.
                NOTE: different levels require different Tensor shapes for the output. Refer to the detect_anomaly method.
            every_n_epochs (int): log the visualization figures every n epochs.
            figsize (Tuple[int, int]): The size of the figure.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone = backbone

        if custom_neck is not None:
            self.neck = custom_neck
        elif backbone_out_seq_len != out_seq_len:
            self.neck = nn.Linear(backbone_out_seq_len, out_seq_len)
        else:
            self.neck = nn.Identity()

        if custom_head is not None:
            self.head = custom_head
        else:
            self.head = nn.Linear(backbone_out_features, out_features)

        self.detection_level = detection_level
        self.threshold = threshold
        self.every_n_epochs = vi_every_n_epochs
        self.figsize = figsize

    def loss(self, x_hat: Tensor, x: Tensor) -> Tensor:
        # TODO: customize the loss function
        return F.mse_loss(x_hat, x)

    def get_task_callbacks(self):
        return [
            # ComputeAndLogMetrics2Tensorboard(num_classes=2),
            ViAndLog(self.every_n_epochs, self.figsize),
        ]

    def detect_anomaly(
        self,
        x: Tensor,
        detection_level: str | None = None,
        threshold: float | None = None,
    ) -> Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, n_features)
        detection_level: str. This will override the detection_level set in the framework if not None.
        threshold: float. This will override the threshold set in the framework if not None.
        Returns:
            - Tensor of shape (batch_size, 1, 1) if detection_level is 'sequence'
            - Tensor of shape (batch_size, seq_len, 1) if detection_level is 'step'
            where each value is 1 if the anomaly is detected, 0 otherwise.
        """
        threshold = threshold or self.threshold
        detection_level = detection_level or self.detection_level
        if (detection_level) == "step":
            return self._detect_anomaly_step(x, threshold)
        elif detection_level == "sequence":
            return self._detect_anomaly_sequence(x, threshold)
        else:
            raise ValueError(
                f"detection_level should be either 'step' or 'sequence', got {detection_level}"
            )

    def _compute_anomaly_score_step(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """
        x (Tensor): shape (batch_size, seq_len, n_features)
        Returns: Tensor of shape (batch_size, seq_len, 1) with the anomaly score for each step
        """
        anomaly_score = F.mse_loss(x_hat, x, reduction="none")
        anomaly_score = anomaly_score.mean(dim=-1, keepdim=True)
        anomaly_score = (F.sigmoid(anomaly_score) - 0.5) * 2
        return anomaly_score

    def _compute_anomaly_score_sequence(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """
        x (Tensor): shape (batch_size, seq_len, n_features)
        Returns: Tensor of shape (batch_size, 1, 1) with the anomaly score for the entire sequence
        """
        anomaly_score = F.mse_loss(x_hat, x, reduction="none")
        anomaly_score = anomaly_score.mean(dim=[-1, -2], keepdim=True)
        anomaly_score = (F.sigmoid(anomaly_score) - 0.5) * 2
        return anomaly_score

    def framework_forward(
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
    ) -> Tensor:
        """
        Only reconstructs the input sequence.
        input: (batch_size, seq_len, n_features)
        output: (batch_size, seq_len, n_features)
        """
        x = backbone(x)
        x = neck(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = head(x)
        return x

    def reconstruct(self, x: Tensor) -> Tensor:
        return self(x)

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable
    ) -> Mapping[str, Tensor]:
        # TODO: add callbacks for anomaly detection metrics
        x, y = batch

        x_hat = self.forward(x)
        loss = loss_fn(x_hat, x)

        anomaly_score_step = self._compute_anomaly_score_step(x_hat, x)
        anomaly_score_sequence = self._compute_anomaly_score_sequence(x_hat, x)
        y_hat = torch.tensor(
            anomaly_score_step > self.threshold
            if self.detection_level == "step"
            else anomaly_score_sequence > self.threshold
        )  # y_hat is the predicted anomaly indicator

        return {
            "loss": loss,
            "original": x,
            "output": x_hat,
            "anomaly_score_step": anomaly_score_step,
            "anomaly_score_sequence": anomaly_score_sequence,
            "y": y,
            "y_hat": y_hat,
        }
