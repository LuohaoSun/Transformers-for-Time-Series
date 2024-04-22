import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from rich import print
from torchmetrics import Accuracy, F1Score, Precision, Recall
from ..framework_base.default_callbacks import get_default_callbacks


def get_classification_callbacks(num_classes: int) -> list[L.Callback]:
    return [Log2Tensorboard(num_classes=num_classes)] + get_default_callbacks()


class Log2Tensorboard(L.Callback):
    # TODO: add more metrics
    # TODO: add on_epoch_end logging
    def __init__(self, num_classes: int) -> None:

        super().__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:

        metrics = {"train_loss": outputs["loss"]}
        metrics.update(self.compute_metrics(outputs["y_hat"], outputs["y"]))
        logger: TensorBoardLogger = trainer.logger  # type: ignore
        logger.log_metrics(metrics, step=trainer.global_step)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:

        metrics = {"val_loss": outputs["loss"]}
        metrics.update(self.compute_metrics(outputs["y_hat"], outputs["y"]))
        logger: TensorBoardLogger = trainer.logger  # type: ignore
        logger.log_metrics(metrics, step=trainer.global_step)

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:

        metrics = {"test_loss": outputs["loss"]}
        metrics.update(self.compute_metrics(outputs["y_hat"], outputs["y"]))
        logger: TensorBoardLogger = trainer.logger  # type: ignore
        logger.log_metrics(metrics, step=trainer.global_step)

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def compute_metrics(self, y_hat: Tensor, y: Tensor) -> Dict[str, float]:

        return {
            "accuracy": self.accuracy(y_hat, y).item(),
        }
