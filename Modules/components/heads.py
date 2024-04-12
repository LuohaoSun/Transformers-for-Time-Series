import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from typing import Callable, Optional
from abc import ABC
from Modules.components.activations import get_activation_fn


class Head(L.LightningModule, ABC):
    def __init__(self, in_features, out_features):
        super(Head, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


class LinearHead(Head):
    def __init__(self, in_features, out_features):
        '''
        this head fetches the last time step of the input tensor.
        '''
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, steps, in_features)
        output: Tensor with shape (Batch, out_features)
        '''
        return self.linear(x[:, -1, :])
    
class MeanLinearHead(Head):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, steps, in_features)
        output: Tensor with shape (Batch, out_features)
        '''
        return self.linear(x.mean(dim=1))


class MLPHead(Head):
    def __init__(self, in_features: int, out_features: int,
                 activation: str | Callable[[Tensor], Tensor],
                 dropout: float, hidden_features: Optional[list[int]] = None):
        super().__init__(in_features, out_features)

        # TODO: assign output time steps
        # TODO: multiple hidden layers
        self.fc1 = nn.Linear(in_features, in_features*4)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features*4, out_features)

    def forward(self, x: Tensor) -> Tensor:
        '''
        input: Tensor with shape (Batch, steps, in_features)
        output: Tensor with shape (Batch, steps, out_features)
        '''
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class UpSampleHead(Head):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)


class ConvHead(Head):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, stride: int, padding: int,
                 activation: str | Callable[[Tensor], Tensor],
                 dropout: float):
        super().__init__(in_features, out_features)
        self.conv = nn.Conv1d(in_features, out_features,
                              kernel_size, stride, padding)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        '''
        input: Tensor with shape (Batch, steps, in_features)
        output: Tensor with shape (Batch, steps, out_features)
        '''
        x = self.conv(x.permute(0, 2, 1))
        x = self.activation(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)
