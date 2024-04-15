import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from typing import Callable, Optional
from Modules.components.activations import get_activation_fn
'''
所有头部均未添加softmax层，因为nn.CrossEntropyLoss会自动添加softmax层。
如果使用其他的损失函数，需要自定义头部。
'''


class LinearHead(L.LightningModule):
    def __init__(self, in_features, num_classes):
        '''
        this head fetches the last time step of the input tensor.
        '''
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, num_classes)
        '''
        return self.linear(x[:, -1, :])


class MeanLinearHead(L.LightningModule):
    def __init__(self, in_features, num_classes):
        '''
        this head fetches all time steps of the input tensor and 
        calculate the mean on the time dimension, then feed the mean tensor to a linear layer.
        '''
        super().__init__(in_features, num_classes)
        self.in_features = in_features
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, num_classes)
        '''
        return self.linear(x.mean(dim=1))
    
class LSTMHead(L.LightningModule):
    def __init__(self, in_features, num_classes, hidden_size, num_layers, dropout):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, num_classes)
        '''
        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])
    
class GRUHead(L.LightningModule):
    def __init__(self, in_features, num_classes, hidden_size, num_layers, dropout):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gru = nn.GRU(in_features, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, num_classes)
        '''
        x, _ = self.gru(x)
        return self.linear(x[:, -1, :])
