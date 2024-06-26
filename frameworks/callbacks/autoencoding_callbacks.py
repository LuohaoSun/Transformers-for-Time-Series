import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from typing import Mapping, Iterable
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from utils.visualization import SeriesPlotter


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
        if trainer.current_epoch % self.every_n_epochs != 0 or batch_idx != 0:
            return
        series_dict = {k: v for k, v in outputs.items() if k != "loss"}
        img = SeriesPlotter.plot_series(series_dict)

        tb_writer: SummaryWriter = trainer.logger.experiment  # type: ignore
        tb_writer.add_figure(f"epoch {trainer.current_epoch}", img)
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
