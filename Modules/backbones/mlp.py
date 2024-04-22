import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from Modules.components import positional_embedding as PE
from Modules.components import token_embedding as TE
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable
from Modules.components.activations import get_activation_fn
from Modules.components import token_embedding as TE


class MLPBackbone(L.LightningModule):
    """
    吗乐批骨干。
    this simple MLP backbone encode each channel of the input sequence independently while sharing the same weights.
    i.e. (Batch, in_seq_len, in_channels) -> (Batch, out_seq_len, in_channels)
    """

    def __init__(
        self,
        in_seq_len: int,
        hidden_len: tuple[int, ...],
        out_seq_len: int,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_seq_len, hidden_len[0])
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_len[i], hidden_len[i + 1]),
                    get_activation_fn(activation),
                )
                for i in range(len(hidden_len) - 1)
            ]
        )
        self.proj = nn.Linear(hidden_len[-1], out_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, out_seq_len, in_features)
        """
        x = self.embed(x.permute(0, 2, 1))
        x = self.layers(x)
        x = self.proj(x)
        return x.permute(0, 2, 1)
