from typing import Callable

import torch.nn.functional as F
from torch import Tensor, nn


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
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation.lower() == "softmax":
        return nn.Softmax(dim=-1)
    elif activation.lower() == "log_softmax":
        return nn.LogSoftmax(dim=-1)
    elif activation.lower() == "leaky_relu":
        return nn.LeakyReLU()
    elif activation.lower() == "elu":
        return nn.ELU()
    elif activation.lower() == "selu":
        return nn.SELU()
    elif activation.lower() == "swish":
        return nn.SiLU()
    elif activation.lower() == "mish":
        return nn.Mish()
    elif activation.lower() == "hardswish":
        return nn.Hardswish()
    elif activation.lower() == "hardtanh":
        return nn.Hardtanh()
    elif activation.lower() == "softplus":
        return nn.Softplus()
    elif activation.lower() == "softsign":
        return nn.Softsign()
    elif activation.lower() == "hardshrink":
        return nn.Hardshrink()
    elif activation.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif activation.lower() == "relu6":
        return nn.ReLU6()
    elif activation.lower() == "silu":
        return nn.SiLU()
    elif activation.lower() == "prelu":
        return nn.PReLU()
    elif activation.lower() == "rrelu":
        return nn.RReLU()
    elif activation.lower() == "glu":
        return nn.GLU()
    elif activation.lower() == "log_sigmoid":
        return nn.LogSigmoid()
    elif activation.lower() == "softmin":
        return nn.Softmin()
    elif activation.lower() == "softshrink":
        return nn.Softshrink()
    elif activation.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif activation.lower() == "linear":
        return nn.Identity()
    else:
        raise NotImplementedError(f"activation function {activation} not implemented")
