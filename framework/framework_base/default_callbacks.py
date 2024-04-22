import lightning as L
import torch.nn as nn
import torch
import subprocess
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from abc import ABC, abstractmethod
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from rich import print


def get_default_callbacks() -> list[L.Callback]:
    return [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        RichModelSummary(max_depth=-1),
        RichProgressBar(),
        LaunchTensorboard(),
        LoadCheckpoint(),
    ]


class LaunchTensorboard(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.proc: Optional[subprocess.Popen] = None
        pass

    def on_train_start(self, trainer, pl_module) -> None:
        with subprocess.Popen(
            ["tensorboard", "--logdir=lightning_logs"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            print(
                f"""
            =======================================
            =        Tensorboard Activated.       =
            =======================================
            Open http://localhost:6006/ to view the training process.
            tensorboard PID: {proc.pid}
                        """
            )
            self.proc = proc
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.proc:
            self.proc.terminate()
        else:
            pass
        return


class LoadCheckpoint(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        return

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:

        checkpoint_callback: ModelCheckpoint = trainer.checkpoint_callback  # type: ignore
        best_val_loss = checkpoint_callback.best_model_score
        best_val_loss_epoch = (
            checkpoint_callback.best_model_path.split(  # FIXME: Windows path separator
                "/"
            )[-1]
            .split("=")[1]
            .split("-")[0]
        )

        # FIXME: log_hyperparams does not work
        trainer.logger.log_hyperparams(  # type: ignore
            pl_module.hparams, {"hp_metric": best_val_loss}  # type: ignore
        )  # type: ignore

        pl_module = pl_module.__class__.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        msg = f"""
            Best validation loss: {best_val_loss} at epoch {best_val_loss_epoch}
            Checkpoint saved at {checkpoint_callback.best_model_path}
            Best Model Loaded from Checkpoint.        
            =======================================
            =          Training Finished.         =
            =======================================
            """
        print(msg)
        return super().on_train_end(trainer, pl_module)
