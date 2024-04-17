from typing import Callable
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class Callable2Module(nn.Module):
    def __init__(self, callable: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.callable = callable

    def forward(self, x: Tensor):
        return self.callable(x)


def get_activation_fn(activation: str | Callable[[Tensor], Tensor]) -> nn.Module:
    if callable(activation):
        if isinstance(activation, nn.Module):
            return activation
        else:
            return Callable2Module(activation)
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            f"activation function {activation} not implemented")
