import re
import socket
import subprocess
from typing import Mapping

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils import find_process_by_port, print_dict

__all__ = ["get_default_callbacks"]


def get_default_callbacks(early_stopping_patience: int) -> list[L.Callback]:
    return [
        # lightning built-in callbacks
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        EarlyStopping(monitor="val_loss", patience=early_stopping_patience),
        # framework default callbacks
        PrintTrainingMsg(),
        LogGraph(),
        LogHyperparams(),
        LogLoss(),
        LoadCheckpoint(),
        LaunchTensorboard(),
        PrintTestMsg(),
    ]


class PrintTrainingMsg(L.Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        msg = f"""
=======================================
=          Training Started.          =
=======================================
"""
        print(msg)
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        msg = f"""
=======================================
=          Training Finished.         =
=======================================
"""
        print(msg)
        return super().on_train_end(trainer, pl_module)


class PrintTestMsg(L.Callback):
    def on_test_start(self, trainer, pl_module) -> None:
        msg = f"""
=======================================
=            Test Started.            =
=======================================
"""
        print(msg)
        return super().on_test_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        msg = f"""
=======================================
=            Test Finished.           =
=======================================
"""
        print(msg)
        return super().on_test_end(trainer, pl_module)


class LaunchTensorboard(L.Callback):
    proc: subprocess.Popen

    def on_train_start(self, trainer, pl_module) -> None:
        if (ps_using_6006 := find_process_by_port(6006)) is not None:
            msg = f"PID {ps_using_6006.pid} is using port 6006. No new tensorboard activated."
        else:
            # FIXME: hard-coded logdir
            self.proc = subprocess.Popen(
                ["tensorboard", "--logdir=lightning_logs"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            msg = f"""
TENSORBOARD ACTIVATED AT http://localhost:6006/
tensorboard PID: {self.proc.pid}
"""
        print(msg)
        return super().on_train_start(trainer, pl_module)


#     def on_train_end(self, trainer, pl_module) -> None:
#         if hasattr(self, "proc"):
#             self.proc.terminate()
#             msg = f"""
# =======================================
# =      Tensorboard Deactivated.       =
# =======================================
# """
#             print(msg)
#         return


class LogGraph(L.Callback):
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.current_epoch == 0 and batch_idx == 0:
            tb_writer: SummaryWriter = trainer.logger.experiment  # type: ignore
            tb_writer.add_graph(pl_module, batch[0][0:1, ...])

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class LogHyperparams(L.Callback):
    """
    注意：
    FrameworkBase只能对hparams属性进行修改，不应当进行任何log操作。
    所有的log操作应当在此LogHyperparams中进行。
    """

    hparams_ignore_patterns = [
        "backbone.*",
        "neck.*",
        "head.*",
        "task.*",
        "vi.*",
        "loss.*",
        "framework.*",
        "model.*",
        "data.*",
        ".*dir",
        ".*path",
        ".*split",
        "num_workers",
    ]

    def __init__(self):
        super().__init__()
        self.regexes = [re.compile(pattern) for pattern in self.hparams_ignore_patterns]

    def on_train_end(self, trainer, pl_module) -> None:
        checkpoint_callback: ModelCheckpoint = trainer.checkpoint_callback  # type: ignore
        best_val_loss = float(checkpoint_callback.best_model_score.item())  # type: ignore
        tb_logger: TensorBoardLogger = trainer.logger  # type: ignore

        hparams_to_log = self.hparams_filter(pl_module.hparams)
        tb_logger.log_hyperparams(
            hparams_to_log, metrics={"Best Validation Loss": best_val_loss}
        )
        msg = f"""
HYPERPARAMETERS LOGGED.
Best validation loss: {best_val_loss}
Hyperparameters:"""
        print(msg)
        print_dict(hparams_to_log, header=("Hyperparameter", "Value"))

    def hparams_filter(self, hparams):
        hparams_filtered = {}
        for key, value in hparams.items():
            if not any(regex.match(key) for regex in self.regexes):
                hparams_filtered[key] = value
        return hparams_filtered


class LogLoss(L.Callback):
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch,
        batch_idx,
    ) -> None:
        pl_module.log(
            "train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True
        )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch,
        batch_idx,
    ) -> None:
        pl_module.log(
            "val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True
        )

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch,
        batch_idx,
    ) -> None:
        pl_module.log("test_loss", outputs["loss"], on_step=False, on_epoch=True)

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)


class LoadCheckpoint(L.Callback):
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        ckpt_info: tuple[str, float, int] = self.get_ckpt_info_from_trainer(trainer)
        self.load_ckpt_from_path(pl_module, ckpt_info[0])
        self.print_success_msg(ckpt_info)

        return super().on_train_end(trainer, pl_module)

    def get_ckpt_info_from_trainer(self, trainer: L.Trainer) -> tuple[str, float, int]:
        checkpoint_callback: ModelCheckpoint = trainer.checkpoint_callback  # type: ignore
        best_model_path = checkpoint_callback.best_model_path
        best_val_loss = float(checkpoint_callback.best_model_score.item())  # type: ignore
        best_val_loss_epoch = int(best_model_path.split("epoch=")[1].split("-")[0])

        return best_model_path, best_val_loss, best_val_loss_epoch

    def load_ckpt_from_path(
        self, pl_module: L.LightningModule, best_model_path: str
    ) -> L.LightningModule:
        pl_module = pl_module.__class__.load_from_checkpoint(best_model_path)

        return pl_module

    def print_success_msg(self, ckpt_info: tuple[str, float, int]) -> None:
        best_model_path, best_val_loss, best_val_loss_epoch = ckpt_info
        msg = f"""
CKECKPOINT LOADED.
Best validation loss: {best_val_loss} at epoch {best_val_loss_epoch}
Checkpoint saved at: {best_model_path}
"""
        print(msg)

        return
