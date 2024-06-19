from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import copy
from .components import positional_embedding as PE
from .components import token_embedding as TE
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable
from .components.activations import get_activation_fn


class MLPBackbone(L.LightningModule):
    """
    吗乐批骨干。
    This is a simple MLP backbone for sequence data, which flattens the input sequence and applies a series of linear layers.
    """

    def __init__(
        self,
        in_seq_len: int,
        in_features: int,
        hidden_features: list[int],
        activation: str | Callable[[Tensor], Tensor] = "relu",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Linear(in_seq_len * in_features, hidden_features[0])
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_features[i], hidden_features[i + 1]),
                    get_activation_fn(activation),
                )
                for i in range(len(hidden_features) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, 1, out_features)
        """
        x = x.flatten(1)
        x = self.embed(x)
        x = self.layers(x)
        return x.unsqueeze(1)


class ResMLPBackbone(L.LightningModule):
    """
    MLP backbone with residual connections.
    """

    def __init__(
        self,
        in_seq_len: int,
        in_features: int,
        res_block_eatures: list[int],
        num_res_blocks: int,
        activation: str | Callable[[Tensor], Tensor] = "relu",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Linear(in_seq_len * in_features, res_block_eatures[0])
        res_block = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(res_block_eatures[i], res_block_eatures[i + 1]),
                    get_activation_fn(activation),
                )
                for i in range(len(res_block_eatures) - 1)
            ]
        )
        self.layers = nn.Sequential(
            *[copy.deepcopy(res_block) for _ in range(num_res_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input: Tensor with shape (Batch, in_seq_len, in_features)
        output: Tensor with shape (Batch, 1, out_features)
        """
        x = x.flatten(1)
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return x.unsqueeze(1)
