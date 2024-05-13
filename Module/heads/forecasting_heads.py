import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from Module.components.activations import get_activation_fn

class LinearHead(L.LightningModule):
    '''
    project the last out_seq_len of the input sequence to out_features
    '''
    def __init__(self, in_features, out_features, out_seq_len):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_seq_len = out_seq_len
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor):
        """
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, in_seq_len, out_features)
        """
        return self.linear(x)
    
class MLPHead(L.LightningModule):
    '''
    project the last out_seq_len of the input sequence to out_features
    '''
    def __init__(self, in_features, out_features, out_seq_len, hidden_features, activation):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_seq_len = out_seq_len
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            get_activation_fn(activation),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: Tensor):
        """
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, in_seq_len, out_features)
        """
        return self.mlp(x)
    
