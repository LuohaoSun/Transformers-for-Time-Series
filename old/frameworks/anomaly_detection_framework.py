# Author: Sun LuoHao
# All rights reserved

from typing import Callable, Iterable, Mapping, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..callbacks.autoencoding_callbacks import ViAndLog
from ..callbacks.classification_callbacks import \
    ComputeAndLogMetrics2Tensorboard
from ..utils import get_loss_fn
from .framework_base import FrameworkBase


class AnomalyDetectionFramework(FrameworkBase):

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        backbone_out_seq_len: int,
        backbone_out_features: int,
        # task params
        out_seq_len: int,
        out_features: int,
        threshold: float,
        # logging params
        vi_every_n_epochs: int = 10,
        figsize: Tuple[int, int] = (7, 3),
        # additional params
        custom_neck: Optional[nn.Module] = None,
        custom_head: Optional[nn.Module] = None,
    ) -> None:
        """
        Anomaly Detection Framework is a framework composed of the autoencoding and classification frameworks.
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

        self.head = custom_head or nn.Linear(backbone_out_features, out_features)

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
        raise NotImplementedError()

    def _compute_anomaly_score_step(self, x_hat: Tensor, x: Tensor) -> Tensor:
        """
        x (Tensor): shape (batch_size, seq_len, n_features)
        Returns: Tensor of shape (batch_size, seq_len, 1) with the anomaly score for each step
        """
        anomaly_score = F.mse_loss(x_hat, x, reduction="none")
        anomaly_score = anomaly_score.mean(dim=-1, keepdim=True)
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
        x = neck(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
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
        y_hat = anomaly_score_step > self.threshold

        return {
            "loss": loss,
            "original": x,
            "reconstructed": x_hat,
            "anomaly_score": anomaly_score_step,
            "groud_truth": y,
            "anomaly_indicator": y_hat,
        }
