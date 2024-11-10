# Author: Sun LuoHao
# All rights reserved
from typing import Any, Callable, Dict, Iterable, Mapping, Union, final

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader

"""
框架的公共方法，主要是训练、测试、预测等。
"""


class FrameworkBasePublic(L.LightningModule):
    # ==============================
    # 以下是框架公共方法：
    def fit(
        self,
        # trainer.fit params:
        datamodule: L.LightningDataModule | None = None,
        train_dataloaders: list[DataLoader] | DataLoader | None = None,
        val_dataloaders: list[DataLoader] | DataLoader | None = None,
        ckpt_path: str | None = None,
        # logging params:
        log_every_n_steps: int = 1,
        # training params:
        lr: float = 1e-3,
        max_epochs: int = 1,
        max_steps: int = -1,
        early_stopping_patience: int = 1000,
        accelerator: str = "auto",
        compile_model: bool = True,
        custom_loss_fn: Callable | str | None = None,
        **trainer_kwargs,
    ) -> L.LightningModule:
        """
        Initialize the framework and start training.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for training.
            ckpt_path (str | None): The path to the checkpoint for resuming training.
            lr (float): The learning rate for the optimizer.
            max_epochs (int): The maximum number of epochs for training.
            max_steps (int): The maximum number of steps for training.
            loss_fn (Callable | str | None): The loss function for training. If None, the default loss function defined in the task_framework will be used.
            accelerator (str): The accelerator for training.
            compile_model (bool): Compiles the model for faster execution using pytorch>=2.0.
            **trainer_kwargs: Additional arguments for the trainer.
            TODO: implement optimizer and lr_scheduler configuration.
        """

        self._configure_framework(
            datamodule=datamodule,
            log_every_n_steps=log_every_n_steps,
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            early_stopping_patience=early_stopping_patience,
            accelerator=accelerator,
            compile_model=compile_model,
            custom_loss_fn=custom_loss_fn,
            **trainer_kwargs,
        )

        if max_epochs == 0 or max_steps == 0:
            # skip training
            return self

        self._framework_trainer.fit(
            model=self,
            datamodule=datamodule,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            ckpt_path=ckpt_path,
        )

        return self

    def test(
        self,
        datamodule: L.LightningDataModule,
        dataloaders: list[DataLoader] | DataLoader | None = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        if not hasattr(self, "_framework_trainer") or self._framework_trainer is None:
            raise RuntimeError(
                "Trainer is not initialized. Please call fit() before test()."
            )

        self._framework_trainer.test(
            self,
            datamodule=datamodule,
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )

    def predict(self, x: Tensor) -> Tensor:
        return self(x)
