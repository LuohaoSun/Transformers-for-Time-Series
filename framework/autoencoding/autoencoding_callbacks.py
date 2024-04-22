import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from rich import print
from torchmetrics import Accuracy, F1Score, Precision, Recall
from traitlets import default
from ..framework_base.default_callbacks import get_default_callbacks


def get_autoencoding_callbacks(
    every_n_epochs=20,
    figsize=(10, 5),
    dpi=300,
) -> list[L.Callback]:
    default_callbacks = get_default_callbacks()
    autoencoder_visulization = AutoEncoderVisulization(
        every_n_epochs=20,
        figsize=(10, 5),
        dpi=300,
    )
    return [autoencoder_visulization] + default_callbacks


class AutoEncoderVisulization(L.Callback):
    """
    a callback for auto encoding framework, used to plot original and reconstructed series.
    rely on matplotlib and tensorboard.
    """

    def __init__(
        self,
        every_n_epochs: int = 20,
        figsize: tuple[int, int] = (10, 5),
        dpi: int = 300,
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.figsize = figsize
        self.dpi = dpi

    def __call__(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Iterable[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            experiment: SummaryWriter = trainer.logger.experiment  # type: ignore

            for key, value in outputs.items():
                if key == "loss":
                    continue
                img = self.plot_series({key: value})
                experiment.add_figure(tag=f"{key}-{trainer.current_epoch}", figure=img)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Iterable[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def plot_series(self, series: Tensor | list[Tensor] | dict[str, Tensor]) -> Figure:
        """
        all series should have the same shape if multiple series are passed.
        series: (batch_size, series_length) or (batch_size, series_length, n_features)
        only first nodes are plotted when the series has a third dimension.
        returns the figure path
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)  # FIXME: hard coded figure size
        if isinstance(series, dict):
            for series_name, series_values in series.items():
                self.sub_plot(series_values, series_name)
        elif isinstance(series, list):
            for series_values in series:
                self.sub_plot(series_values)
        else:
            self.sub_plot(series)
        # plt.legend()
        plt.xlabel("Time steps")
        plt.ylabel("Value")
        img = plt.gcf()
        plt.close()
        return img

    def sub_plot(self, series: Tensor, series_name: str | None = None):
        """
        series: (batch_size, series_length) or (batch_size, series_length, num_nodes)
        only first plot_nodes nodes are plotted when the series has a third dimension.
        """
        series_name = series_name if series_name is not None else "Series"
        # plot a single series
        if (
            len(series.shape) == 3
        ):  # Series has shape (batch_size, series_length, num_nodes)
            x = np.arange(series.shape[1])
            y = series[0, :, 0].cpu().detach().numpy()
        elif len(series.shape) == 2:  # Series has shape (batch_size, series_length)
            x = np.arange(series.shape[1])
            y = series[0, :].cpu().detach().numpy()
        else:
            UserWarning(
                f"""
                Unsupported series shape.
                Expected (batch_size, series_length)
                or (batch_size, series_length, num_nodes),
                but got shape {series.shape} instead.
                Skipping this series.
                """
            )
            return
        plt.plot(x, y, label=series_name)
