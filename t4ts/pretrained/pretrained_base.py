import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Mapping, Iterable, Tuple, Callable, Optional, final
from torch import Tensor
from abc import ABC, abstractmethod
from ..utils import get_loss_fn


class PretrainedBase(L.LightningModule, ABC):

    def __init__(self, task: str) -> None:
        super().__init__()
        # 预训练模型可能没有需要训练的参数，但是为了避免optimizer的报错，需要设置一个参数
        self.task = task
        self._sent_2_optimizer = nn.Parameter(torch.tensor(0.0))

    @final
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        original output of the model, i.e., embeddings, if available.
        """
        if self.task == "forecasting":
            return self.forecast(x)
        elif self.task == "reconstruction":
            return self.reconstruct(x)
        elif self.task == "embedding":
            return self.embed(x)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @abstractmethod
    def forecast(self, x: Tensor) -> Tensor:
        """
        forecast the future sequence
        """
        ...

    @abstractmethod
    def reconstruct(self, x: Tensor) -> Tensor:
        """
        reconstruct the input sequence
        """
        ...

    @abstractmethod
    def embed(self, x: Tensor) -> Tensor:
        """
        embed the input sequence
        """
        ...
