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
from .default_callbacks import *
from .default_callbacks import get_default_callbacks


class FrameworkBase(L.LightningModule, ABC):
    """
    Base model class for time series tasks.

    This class provides the following functionalities:
    1. Defines the inheritance specification of L.LightningModule in the project. Subclasses need to implement methods such as loss, training_step, val_step, test_step.
    2. Encapsulates the structure of backbone-head and implements methods such as forward, configure_optimizers.
    3. Defines methods such as fit, test, run_training, run_testing for easy invocation.

    Properties:
        backbone (nn.Module): The backbone module.
        head (nn.Module): The head module.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs forward pass on the input tensor.

        fit(datamodule: L.LightningDataModule) -> None:
            Trains the model using the provided LightningDataModule.

        test(datamodule: L.LightningDataModule) -> None:
            Tests the model using the provided LightningDataModule.

        run_training(datamodule: L.LightningDataModule) -> None:
            Same as fit. Trains the model using the provided LightningDataModule.

        run_testing(datamodule: L.LightningDataModule) -> None:
            Same as test. Tests the model using the provided LightningDataModule.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        additional_callbacks: list[L.Callback],
        lr: float,
        max_epochs: int,
        max_steps: int,
    ) -> None:
        """
        Initializes the FrameworkBase.

        Args:
            backbone (nn.Module): The backbone module.
            head (nn.Module): The head module.
            additional_callbacks (list[L.Callback]): Additional callbacks for the trainer.
            lr (float): The learning rate for the optimizer.
            max_epochs (int): The maximum number of epochs for training.
            max_steps (int): The maximum number of steps for training.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head

        self.framework_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.framework_trainer = L.Trainer(
            max_epochs=max_epochs,
            max_steps=max_steps,
            callbacks=get_default_callbacks() + additional_callbacks,
            accelerator="auto",
        )


    @abstractmethod
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the loss between the model's output and the target.

        Args:
            output (Tensor): The model's output tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The calculated loss tensor.
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass on the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_seq_len, in_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, out_seq_len, out_features).
        """
        x = self.backbone(x)
        x = self.head(x)
        return x

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configures the optimizer for training.

        Returns:
            Dict[str, Union[Optimizer, LRScheduler]]: The optimizer configuration.
        """
        return {"optimizer": self.framework_optimizer}

    def fit(self, datamodule: L.LightningDataModule) -> None:
        """
        Trains the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for training.
        """
        self.framework_trainer.fit(self, datamodule)

    def test(self, datamodule: L.LightningDataModule) -> None:
        """
        Tests the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for testing.
        """
        self.framework_trainer.test(self, datamodule)

    def run_training(self, datamodule: L.LightningDataModule) -> None:
        """
        Same as fit. Trains the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for training.
        """
        return self.fit(datamodule)

    def run_testing(self, datamodule: L.LightningDataModule) -> None:
        """
        Same as test. Tests the model using the provided LightningDataModule.

        Args:
            datamodule (L.LightningDataModule): The LightningDataModule for testing.
        """
        return self.test(datamodule)
