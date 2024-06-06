# Author: Sun LuoHao
# All rights reserved
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Mapping, Iterable
from torch import Tensor
from .framework_base import FrameworkBase
from .functionalities.forecasting_callbacks import ComputeMetricsAndLog
from .functionalities.autoencoding_callbacks import ViAndLog2Tensorboard


class ForecastingFramework(FrameworkBase):
    """
    input: Tensor of shape (b, l_in, d_in)
    output: mask of shape (b, l_out, d_out)
    in a forecasting task, the d_out could be different from d_in.
    """

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        backbone_out_features: int,
        # task params
        out_seq_len: int,
        out_features: int,
        # logging params
        evry_n_epochs: int = 20,
        fig_size: tuple[int, int] = (10, 5),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.out_seq_len = out_seq_len
        self.out_features = out_features
        self.every_n_epochs = evry_n_epochs
        self.fig_size = fig_size

        self.backbone = backbone
        self.head = nn.Linear(backbone_out_features, out_features)

    @property
    def task_functionalities(self) -> list[L.Callback]:
        return [
            ComputeMetricsAndLog(),
            ViAndLog2Tensorboard(
                every_n_epochs=self.every_n_epochs,
                fig_size=self.fig_size,
            ),  # 把autoencoding的callback借过来
        ]

    @property
    def _loss(self) -> nn.Module:
        return nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (b, l_in, d_in)
        return: (b, l_out, d_out)
        """
        out_seq_len = self.out_seq_len if self.out_seq_len > 0 else x.size(1)
        x = self.backbone(x)[:, -out_seq_len:, :]
        x = self.head(x)
        return x

    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:
        """
        batch: (x, y)
        x: (b, l_in, d_in)
        y: (b, l_out, d_out)
        """

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat}

    def validation_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        return self.training_step(batch, batch_idx)

    def test_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:

        return self.training_step(batch, batch_idx)
