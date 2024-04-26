import lightning as L
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from rich import print
from torchmetrics import Accuracy, F1Score, Precision, Recall


class ComputeAndLogMetrics2Tensorboard(L.Callback):
    # TODO: add more metrics
    # TODO: add on_epoch_end logging
    def __init__(self, num_classes: int) -> None:

        super().__init__()

        self.metric_funcs = nn.ModuleDict(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes),
                "precision": Precision(task="multiclass", num_classes=num_classes),
                "recall": Recall(task="multiclass", num_classes=num_classes),
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

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:

        self(trainer, outputs, "train")

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:

        self(trainer, outputs, "val")

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:

        self(trainer, outputs, "test")

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
