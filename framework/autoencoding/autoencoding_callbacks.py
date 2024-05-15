import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Mapping, Iterable
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


class ViAndLog2Tensorboard(L.Callback):
    """
    a callback for auto encoding framework.
    Used to visualize series returned in xxx_steps(e.g. original and reconstructed series),
    and save the figure to tensorboard.
    Rely on matplotlib and tensorboard.
    """

    def __init__(
        self,
        every_n_epochs: int = 20,
        fig_size: tuple[int, int] = (4, 2),
        dpi: int = 300,
    ) -> None:
        self.every_n_epochs = every_n_epochs
        self.figsize = fig_size
        self.dpi = dpi

    def __call__(
        self,
        trainer: L.Trainer,
        outputs: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:

        if trainer.current_epoch % self.every_n_epochs != 0:
            return  # only applied to the every_n_epochs-th epoch.
        if batch_idx != 0:
            return  # only applied to the first batch of

        def get_series_names(
            outputs: Mapping[str, Tensor],
        ) -> list[str]:
            return [name for name in outputs if name != "loss"]

        def get_image_names(series_names: list[str]) -> list[str]:
            return [
                f"{series_name}_epoch_{trainer.current_epoch}"
                for series_name in series_names
            ]

        def get_images(
            outputs: Mapping[str, Tensor], series_names: list[str]
        ) -> list[Figure]:
            return [
                self.plot_series(outputs[series_name]) for series_name in series_names
            ]

        def log_images(images: list[Figure], image_names: list[str]) -> None:
            tb_summary_writer: SummaryWriter = trainer.logger.experiment  # type: ignore
            for image_name, image in zip(image_names, images):
                tb_summary_writer.add_figure(image_name, image)

        series_names = get_series_names(outputs)
        image_names = get_image_names(series_names)
        images = get_images(outputs, series_names)
        log_images(images, image_names)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Iterable[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self(trainer, outputs, batch_idx)

    def plot_series(self, series: Tensor | list[Tensor] | dict[str, Tensor]) -> Figure:
        """
        TODO: 写得太难看了，需要重构
        all series should have the same shape if multiple series are passed.
        series: (batch_size, series_length) or (batch_size, series_length, n_features)
        only first nodes are plotted when the series has a third dimension.
        returns the figure path
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
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
        TODO: 写得太难看了，需要重构
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
