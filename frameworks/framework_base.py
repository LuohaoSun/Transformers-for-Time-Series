# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn as nn
import torch

from lightning.pytorch.loggers import TensorBoardLogger
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from .callbacks.default_callbacks import get_default_callbacks
from .utils import get_loss_fn


class FrameworkBase(L.LightningModule, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.loss = self._loss

    @property
    @abstractmethod
    def _task_callbacks(self) -> list[L.Callback]:
        """
        The task-specific functionalities for the framework.
        """
        ...

    @property
    def _default_callbacks(self) -> list[L.Callback]:
        """
        The default callbacks for the framework.
        """
        return get_default_callbacks()

    @property
    def _framework_callbacks(self) -> list[L.Callback]:
        """
        The general functionalities for the framework.
        """
        return self._default_callbacks + self._task_callbacks

    @property
    @abstractmethod
    def _loss(self) -> Callable:
        """
        The loss function for the task.
        This will be used if the loss_fn is not provided in the fit method.
        In other words, this will be overriden by the loss_fn in the fit method.
        """
        ...

    @abstractmethod
    def training_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:
        """
        Performs a single training step.

        Args:
            batch (Iterable[Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            Mapping[str, Tensor]: The mapping of output tensors.
        """
        ...

    @abstractmethod
    def validation_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:
        """
        Performs a single validation step.

        Args:
            batch (Iterable[Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            Mapping[str, Tensor]: The mapping of output tensors.
        """
        ...

    @abstractmethod
    def test_step(
        self, batch: Iterable[Tensor], batch_idx: int
    ) -> Mapping[str, Tensor]:
        """
        Performs a single testing step.

        Args:
            batch (Iterable[Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            Mapping[str, Tensor]: The mapping of output tensors.
        """
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass on the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_seq_len, in_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, out_seq_len, out_features).
        """
        ...

    def _load_checkpoint(self, ckpt_path: str):
        self = self.__class__.load_from_checkpoint(checkpoint_path=ckpt_path)
        return self

    def setup(self, stage: str) -> None:
        """
        This method is called before the training and testing steps.
        """
        if self._compile_model:
            self = torch.compile(self)
        return

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configures the optimizer for training.

        Returns:
            Dict[str, Union[Optimizer, LRScheduler]]: The optimizer configuration.
        """

        return {"optimizer": self._framework_optimizer}

    def fit(
        self,
        # data params:
        datamodule: L.LightningDataModule,
        train_dataloaders: list[DataLoader] | DataLoader | None = None,
        val_dataloaders: list[DataLoader] | DataLoader | None = None,
        ckpt_path: str | None = None,
        # logging params:
        log_every_n_steps: int = 1,
        # training params:
        lr: float = 1e-3,
        max_epochs: int = 1,
        max_steps: int = -1,
        loss_fn: Callable | str | None = None,
        accelerator: str = "auto",
        compile_model: bool = True,
        optimizer: Optimizer | str | None = None,
        lr_scheduler: LRScheduler | str | None = None,
        **trainer_kwargs,
    ) -> L.LightningModule:
        """
        Trains the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for training.
            lr (float): The learning rate for the optimizer.
            max_epochs (int): The maximum number of epochs for training.
            max_steps (int): The maximum number of steps for training.
            loss_fn (Callable | str | None): The loss function for training. If None, the default loss function defined in the task_framework will be used.
            accelerator (str): The accelerator for training.
            compile_model (bool): Compiles the model for faster execution using pytorch>=2.0.
            **trainer_kwargs: Additional arguments for the trainer.
            TODO: implement optimizer and lr_scheduler configuration.
        """
        # save hyperparameters:
        self.hparams.update(datamodule.hparams)
        self.hparams.update(
            {
                "lr": lr,
                "max_epochs": max_epochs,
                "max_steps": max_steps,
            }
        )

        # configure framework properties:
        self._compile_model = compile_model
        self.loss = get_loss_fn(loss_fn) if loss_fn is not None else self.loss
        self._framework_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._framework_logger = TensorBoardLogger(
            save_dir=".", name="lightning_logs", default_hp_metric=False
        )
        self._framework_trainer = L.Trainer(
            max_epochs=max_epochs,
            max_steps=max_steps,
            callbacks=self._framework_callbacks,
            logger=self._framework_logger,
            accelerator=accelerator,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=False,
            **trainer_kwargs,
        )

        # fit the model:
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

        self._framework_trainer.test(
            self,
            datamodule=datamodule,
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )

    def predict(self, x: Tensor) -> Tensor:
        return self(x)
