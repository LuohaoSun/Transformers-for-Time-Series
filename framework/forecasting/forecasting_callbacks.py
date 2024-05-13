import lightning as L
import torch.nn as nn
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from rich import print
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, R2Score, ExplainedVariance


class ComputeAndLogMetrics2Tensorboard(L.Callback):
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