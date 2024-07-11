import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Mapping, Iterable, Tuple, Callable, Optional
from torch import Tensor
from abc import ABC, abstractmethod
from utils import get_loss_fn


class PretrainedBase(L.LightningModule, ABC):

    def __init__(self) -> None:
        super().__init__()
        # 预训练模型可能没有需要训练的参数，但是为了避免optimizer的报错，需要设置一个参数
        self._sent_2_optimizer = nn.Parameter(torch.tensor(0.0))

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        original output of the model, i.e., embeddings, if available.
        """
        pass

