# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn as nn
import torch

from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from abc import ABC, abstractmethod
from rich import print
from .general_functionalities import get_general_functionalities


class FrameworkBase(L.LightningModule, ABC):

    def __init__(self) -> None:
        super().__init__()

    @property
    def functionalities(self) -> list[L.Callback]:
        """
        The general functionalities for the framework.
        """
        return get_general_functionalities() + self.task_functionalities

    @property
    @abstractmethod
    def task_functionalities(self) -> list[L.Callback]:
        """
        The task-specific functionalities for the framework.
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

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configures the optimizer for training.

        Returns:
            Dict[str, Union[Optimizer, LRScheduler]]: The optimizer configuration.
        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": self.framework_optimizer}

    def fit(
        self,
        # datamodule:
        datamodule: L.LightningDataModule,
        # training params:
        lr: float = 1e-3,
        max_epochs: int = 1,
        max_steps: int = -1,
        accelerator: str = "auto",
        **trainer_kwargs,
    ) -> None:
        """
        Trains the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for training.
            lr (float): The learning rate for the optimizer.
            max_epochs (int): The maximum number of epochs for training.
            max_steps (int): The maximum number of steps for training.
            accelerator (str): The accelerator for training.
            **trainer_kwargs: Additional arguments for the trainer.
            TODO: add lr_scheduler
        """
        self.hparams.update(
            {
                "lr": lr,
                "max_epochs": max_epochs,
                "max_steps": max_steps,
                "accelerator": accelerator,
                **trainer_kwargs,
            }
        )
        self.framework_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.framework_trainer = L.Trainer(
            max_epochs=max_epochs,
            max_steps=max_steps,
            callbacks=self.functionalities,
            accelerator=accelerator,
            **trainer_kwargs,
        )
        self.framework_trainer.fit(self, datamodule)
        

    def test(self, datamodule: L.LightningDataModule) -> None:
        """
        Tests the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for testing.
        """
        self.framework_trainer.test(self, datamodule)

    def load_checkpoint(self, ckpt_path: str):
        self = self.__class__.load_from_checkpoint(checkpoint_path=ckpt_path)
        return self
