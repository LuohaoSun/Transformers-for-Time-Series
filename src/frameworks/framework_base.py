# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn as nn
import torch

from lightning.pytorch.loggers import TensorBoardLogger
from typing import (
    Mapping,
    Union,
    Callable,
    Dict,
    Any,
    Iterable,
    final,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from .callbacks.default_callbacks import get_default_callbacks
from .utils import get_loss_fn
from rich import print


class FrameworkBase(L.LightningModule, ABC):
    # ==============================
    # 以下是子类需要实现的方法和属性：

    @property
    @abstractmethod
    def backbone(self) -> nn.Module: ...

    @property
    @abstractmethod
    def neck(self) -> nn.Module: ...

    @property
    @abstractmethod
    def head(self) -> nn.Module: ...

    @abstractmethod
    def framework_forward(
        self,
        x: Tensor,
        backbone: Callable[..., Tensor],
        neck: Callable[..., Tensor],
        head: Callable[..., Tensor],
    ) -> Tensor:
        """
        The forward pass of the model.
        This will be called in the forward method of the class.
        Refer to {self.__setattr__} method for why this design is used.
        """
        ...

    @abstractmethod
    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the loss between the input and target tensors.
        This will be overridden if a custom loss function is provided in the fit method.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The loss tensor.
        """
        ...

    @abstractmethod
    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Mapping[str, Tensor]:
        """
        Performs a single model step (training_step, validation_step, test_step).

        Args:
            batch (Iterable[Tensor]): The input batch.

        Returns:
            Mapping[str, Tensor]: The mapping of output tensors. MUST contain the key "loss".
        """
        ...

    @abstractmethod
    def get_task_callbacks(self) -> list[L.Callback]:
        """
        The task-specific functionalities for the framework.
        """
        ...

    # ==============================
    # 以下是框架私有方法和属性：
    _framework_optimizer: Optimizer
    _framework_logger: TensorBoardLogger
    _framework_trainer: L.Trainer
    _framework_loss: Callable[..., Tensor]
    _framework_callbacks: list[L.Callback]
    _framework_backbone: nn.Module
    _framework_neck: nn.Module
    _framework_head: nn.Module

    def _configure_framework(
        self,
        datamodule: L.LightningDataModule | None,
        # logging params:
        log_every_n_steps: int,
        # training params:
        lr: float,
        max_epochs: int,
        max_steps: int,
        early_stopping_patience: int,
        accelerator: str,
        compile_model: bool,
        custom_loss_fn: Callable | str | None,
        **trainer_kwargs,
    ) -> L.LightningModule:
        """
        configuration of the following properties before training:
        - self._framework_optimizer
        - self._framework_logger
        - self._framework_callbacks
        - self._framework_trainer
        - self._framework_loss
        """
        # save hyperparameters:
        if datamodule is not None:
            self.hparams.update(datamodule.hparams)
        self.hparams.update(
            {
                "lr": lr,
                "max_epochs": max_epochs,
                "max_steps": max_steps,
            }
        )

        self._framework_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._framework_logger = TensorBoardLogger(
            save_dir=".", name="lightning_logs", default_hp_metric=False
        )
        self._framework_callbacks = (
            get_default_callbacks(early_stopping_patience) + self.get_task_callbacks()
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
        self._framework_loss = (
            get_loss_fn(custom_loss_fn) if custom_loss_fn else self.loss
        )

        if compile_model:
            self = torch.compile(self)

        return self  # type: ignore

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
        Trains the model using the provided LightningDataModule.

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

    # ==============================
    # 以下是重写的父类方法：
    @final
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass on the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_seq_len, in_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, out_seq_len, out_features).
        """
        return self.framework_forward(x, self._framework_backbone, self.neck, self.head)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configures the optimizer for training.

        Returns:
            Dict[str, Union[Optimizer, LRScheduler]]: The optimizer configuration.
        """

        return {"optimizer": self._framework_optimizer}

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
        return self.model_step(batch, self._framework_loss)

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
        return self.model_step(batch, self._framework_loss)

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
        return self.model_step(batch, self._framework_loss)

    # 运算符重载（用于支持abstractmethod与hparam更新）：
    def __setattr__(self, name: str, value: Any) -> None:
        # override __setattr__ to:
        #   - update hyperparameters from backbone model
        #   - use @abstractmethod which could force subclasses to implement the properties
        #   - 兼容nn.Module的属性设置
        if name == "backbone":
            self._framework_backbone = value
            if hasattr(value, "hparams") and len(value.hparams) > 0:
                self.hparams.update(value.hparams)
            else:
                print(
                    f"No hyperparameters found in the {name}. Save your hyperparameters in {name}.hparams."
                )
        elif name == "neck":
            self._framework_neck = value
        elif name == "head":
            self._framework_head = value
        else:
            super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        if name == "backbone":
            return self._framework_backbone
        elif name == "neck":
            return self._framework_neck
        elif name == "head":
            return self._framework_head
        else:
            return super().__getattribute__(name)

    # Don't work in runtime, only for type checking. Use __setattr__ and __getattribute__ instead.
    @backbone.setter
    def backbone(self, backbone: nn.Module): ...
    @neck.setter
    def neck(self, neck: nn.Module): ...
    @head.setter
    def head(self, head: nn.Module): ...
    @backbone.getter
    def backbone(self) -> nn.Module: ...
    @neck.getter
    def neck(self) -> nn.Module: ...
    @head.getter
    def head(self) -> nn.Module: ...
