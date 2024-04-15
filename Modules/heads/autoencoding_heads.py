import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from Modules.components.activations import get_activation_fn
'''
所有头部均未添加softmax层，因为nn.CrossEntropyLoss会自动添加softmax层。
如果使用其他的损失函数，需要自定义头部。
'''


class LinearHead(L.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, in_seq_len, out_features)
        '''
        return self.linear(x)
    
class MLPHead(L.LightningModule):
    def __init__(self, in_features, out_features, hidden_features, activation):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            get_activation_fn(activation),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, in_seq_len, out_features)
        '''
        return self.mlp(x)

    
class LSTMHead(L.LightningModule):
    def __init__(self, in_features, out_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, out_features)
        '''
        x, _ = self.lstm(x)
        return self.linear(x)
    
class GRUHead(L.LightningModule):
    def __init__(self, in_features, out_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gru = nn.GRU(in_features, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor):
        '''
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, out_features)
        '''
        x, _ = self.gru(x)
        return self.linear(x)
    

