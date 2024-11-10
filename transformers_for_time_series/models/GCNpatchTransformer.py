from typing import Any, Callable, Dict, Iterable, Mapping, Union

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from .layers import positional_embedding as PE
from .layers import token_embedding as TE


class GCNPatchTransformer(L.LightningModule):
    def __init__(
        self,
        adj: Tensor | str,
        in_features: int,
        patch_size: int,
        d_model: int,
        out_features: int,
        num_layers: int,
        recurrent_times: int = 1,
        dropout: float = 0.1,
        nhead: int = 4,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
    ) -> None:
        """
        Initializes a PatchTST module.

        Args:
            in_features (int): Number of input features.
            patch_size (int): Size of the patches. <= 16 is recommended.
            d_model (int): Dimensionality of the model, converted from in_features*patch_size.
            out_features (int): Number of output features.
            num_layers (int): Number of transformer layers.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            nhead (int, optional): Number of attention heads. Defaults to 4.
            activation (str or Callable[[Tensor], Tensor], optional): Activation function or name. Defaults to 'gelu'.
            additional_tokens_at_last (int, optional): Number of additional tokens to be added at the end of the sequence.
                These tokens can be used for classification, regression or other tasks. Defaults to 0.
            norm_first (bool, optional): Whether to apply layer normalization before the attention layer. Defaults to True.
        TODO: save_hyperparameters to be added.
        """
        super().__init__()

        self.d_model = d_model
        self.recurrent_times = recurrent_times
        self.patch_emb = TE.GCNPatchEmbedding(
            adj,
            in_features,
            patch_size,
            d_model,
            out_features,
            activation,
        )
        self.patch_size = patch_size
        self.pos_emb = PE.SinPosEmbedding(d_model=d_model, max_len=1024)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation=activation,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps, in_features)
        """
        assert (
            x.shape[1] % self.patch_size == 0
        ), "x.shape[1] must be divisible by patch_size"
        x = self.patch_emb(x)
        x = self.pos_emb(x)
        for _ in range(self.recurrent_times):
            x = self.transformer_encoder(x)
        x = self.patch_emb.unpatch(x)

        return x
