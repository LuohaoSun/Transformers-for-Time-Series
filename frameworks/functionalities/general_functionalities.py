import lightning as L
import subprocess
import yaml
from typing import Mapping
from torch import Tensor
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from rich import print
from torch.utils.tensorboard.writer import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

# TODO: rename to universal_functionalities.py

__all__ = ["get_general_functionalities"]


def get_general_functionalities() -> list[L.Callback]:
    return [
        # lightning built-in callbacks
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        # framework default callbacks
        LogGraph(),
        LogHyperparams(),
        LogLoss(),
        LoadCheckpoint(),
        LaunchTensorboard(),
    ]


class PrintTrainingMsg(L.Callback):
    def __init__(self) -> None:
        super().__init__()

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


class LaunchTensorboard(L.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.proc: subprocess.Popen
        pass

    def on_train_start(self, trainer, pl_module) -> None:
        self.proc = subprocess.Popen(
            ["tensorboard", "--logdir=lightning_logs"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        msg = f"""
=======================================
=        Tensorboard Activated.       =
=======================================
Open http://localhost:6006/ to view the training process.
tensorboard PID: {self.proc.pid}
"""
        print(msg)
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        # kill_proc = True if input("Terminate tensorboard? ([Y]/n): ") != "n" else False
        # if kill_proc:
        self.proc.terminate()
        msg = f"""
=======================================
=      Tensorboard Dectivated.        =
=======================================
"""
        # else:
        #         msg = f"""
        # kill the tensorboard with
        # kill {self.proc.pid}
        # if you dont need it.
        # """
        return print(msg)


class LogGraph(L.Callback):
    def __init__(self) -> None:
        super().__init__()

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
            tb_writer.add_graph(pl_module, batch[0])

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class LogHyperparams(L.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_end(self, trainer, pl_module) -> None:
        checkpoint_callback: ModelCheckpoint = trainer.checkpoint_callback  # type: ignore
        best_val_loss = float(checkpoint_callback.best_model_score.item())  # type: ignore
        tb_logger: TensorBoardLogger = trainer.logger  # type: ignore
        # hparam_dict = {k: str(v) for k, v in pl_module.hparams.items()}
        if len(pl_module.hparams) == 0:
            self.no_hyperparameters_found()
        else:
            self.hyperparameters_logged(tb_logger, pl_module.hparams, best_val_loss)
        return super().on_train_end(trainer, pl_module)

    def no_hyperparameters_found(self):
        msg = f"""
=======================================
=       No Hyperparameters Found.     =
=======================================
Use `self.save_hyperparameters()` in the __init__() method of your model to log hyperparameters.
"""
        print(msg)

    def hyperparameters_logged(self, logger, hparam_dict, best_val_loss):
        logger.log_hyperparams(hparam_dict, {"Best Validation Loss": best_val_loss})
        msg = f"""
=======================================
=       Hyperparameters Logged.       =
=======================================
Hyperparameters:\n{hparam_dict}
Best validation loss: {best_val_loss}
"""
        print(msg)


class LogLoss(L.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch,
        batch_idx,
    ) -> None:

        pl_module.log(
            "train_loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True
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
            "val_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True
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

    def __init__(self) -> None:
        super().__init__()
        return

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
=======================================
=        Checkpoint Loaded.          =
=======================================
Best validation loss: {best_val_loss} at epoch {best_val_loss_epoch}
Checkpoint saved at: {best_model_path}
"""
        print(msg)

        return
