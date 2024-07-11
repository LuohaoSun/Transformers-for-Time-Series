# Author: Sun LuoHao
# All rights reserved
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Mapping, Iterable, Callable, Any
from torch import Tensor
from .framework_base import FrameworkBase
from .callbacks.forecasting_callbacks import ComputeMetricsAndLog
from .callbacks.autoencoding_callbacks import ViAndLog


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
        # visualization params
        vi_every_n_epochs: int = 20,
        vi_fig_size: tuple[int, int] = (10, 5),
        # custom model components
        custom_neck: nn.Module | None = None,
        custom_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.out_seq_len = out_seq_len
        self.out_features = out_features
        self.every_n_epochs = vi_every_n_epochs
        self.fig_size = vi_fig_size

        self.backbone = backbone
        self.neck = nn.Identity() if custom_neck is None else custom_neck
        self.head = (
            nn.Linear(backbone_out_features, out_features)
            if custom_head is None
            else custom_head
        )

    def get_task_callbacks(self) -> list[L.Callback]:
        return [
            ComputeMetricsAndLog(),
            ViAndLog(
                every_n_epochs=self.every_n_epochs,
                fig_size=self.fig_size,
            ),  # 把autoencoding的callback借过来
        ]

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
        x: (b, l_in, d_in)
        return: (b, l_out, d_out)
        """
        out_seq_len = self.out_seq_len if self.out_seq_len > 0 else x.size(1)
        x = backbone(x)[:, -out_seq_len:, :]
        x = head(x)
        return x

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable[..., Any]
    ) -> Mapping[str, Tensor]:
        """
        batch: (x, y)
        x: (b, l_in, d_in)
        y: (b, l_out, d_out)
        """

        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)

        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat}
