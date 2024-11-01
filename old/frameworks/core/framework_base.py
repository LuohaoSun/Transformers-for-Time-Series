# Author: Sun LuoHao
# All rights reserved
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Mapping, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from rich import print
from torch import Tensor

from .framework_override import FrameworkBaseOverride
from .framework_private import FrameworkBasePrivate
from .framework_public import FrameworkBasePublic


class FrameworkBase(FrameworkBaseOverride, FrameworkBasePrivate, FrameworkBasePublic, ABC):
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
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
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
        Performs a single model step.

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
