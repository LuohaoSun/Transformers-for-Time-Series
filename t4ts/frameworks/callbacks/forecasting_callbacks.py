import lightning as L
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from rich import print
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    R2Score,
    ExplainedVariance,
)
from matplotlib.figure import Figure
from typing import Mapping, Iterable
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from ...utils.visualization import SeriesPlotter

__all__ = [
    "ComputeMetricsAndLog",
    "ViAndLog",
]


class ComputeMetricsAndLog(L.Callback):
    # TODO: add more metrics
    # TODO: add on_epoch_end logging
    def __init__(self) -> None:

        super().__init__()

        self.metric_funcs = nn.ModuleDict(
            {
                "RMSE": RootMeanSquaredError(),
                "MAE": MeanAbsoluteError(),
                "mape": MeanAbsolutePercentageError(),
                "r2": R2Score(),
                "ev": ExplainedVariance(),
            }
        )

    def __call__(
        self, trainer: L.Trainer, outputs: Mapping[str, Tensor], stage: str
    ) -> None:

        y_hat, y = outputs["y_hat"], outputs["y"]
        self.metric_funcs.to(y_hat.device)
        metrics = {
            f"{stage}_{m_name}": m_func(y_hat, y)
            for m_name, m_func in self.metric_funcs.items()
        }
        logger: TensorBoardLogger = trainer.logger  # type: ignore
        logger.log_metrics(metrics, step=trainer.global_step)


class RootMeanSquaredError(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.sqrt(self.mse(y_hat, y))


class ViAndLog(L.Callback):

    def __init__(
        self,
        every_n_epochs: int = 20,
        fig_size: tuple[int, int] = (4, 2),
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.figsize = fig_size

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Iterable[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        series_dict = {"ground truth": outputs["y"], "predicted": outputs["y_hat"]}
        img = SeriesPlotter.plot_series(series_dict)

        tb_writer: SummaryWriter = trainer.logger.experiment  # type: ignore
        tb_writer.add_figure(f"epoch {trainer.current_epoch}", img)
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
