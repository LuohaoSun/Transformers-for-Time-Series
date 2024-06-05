import torch
import torch.nn as nn
import lightning as L
from .components import positional_embedding as PE
from .components import token_embedding as TE
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable


class LSTMBackbone(L.LightningModule):
    __doc__ = f"""
    LSTM Backbone.
    The output only contains the 'outputs' of the nn.LSTM Outputs, without '(h, c)'. 
    See {nn.LSTM} for more details.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        return: (batch_size, in_seq_len, hidden_features)
        """
        x, _ = self.lstm(x)
        return x
