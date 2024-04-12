from typing import Callable
from torch import Tensor
from torch import nn
import torch.nn.functional as F

def get_activation_fn(activation: str | nn.Module) -> nn.Module:
    if callable(activation):
        return activation
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'linear':
        return nn.Identity()
    else:
        raise NotImplementedError(
            f"activation function {activation} not implemented")