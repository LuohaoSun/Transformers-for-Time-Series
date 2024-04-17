import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from Modules.components import positional_embedding as PE
from Modules.components import token_embedding as TE
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable
from Modules.components.activations import get_activation_fn


class PatchTSTEncoder(nn.Module):
    # TODO: use:
    #               [cls] token for classification,
    #               [sep] token for regression,
    #               [mask] token for masked reconstruction
    #               [pad] token for padding
    # TODO: use the official implementation including the channel-mixing
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 patch_size: int,
                 patch_stride: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 nhead: int = 4,
                 activation: str | Callable[[Tensor], Tensor] = 'gelu',
                 norm_first: bool = True) -> None:
        """
        Initializes a PatchTST module.

        Args:
            in_features (int): Number of input features.
            d_model (int): Dimensionality of the model.
            patch_size (int): Size of the patches. <= 16 is recommended.
            patch_stride (int): Stride of the patches. If 0, patch_stride = patch_size, recommended.
            num_layers (int): Number of transformer layers.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            nhead (int, optional): Number of attention heads. Defaults to 4.
            activation (str or Callable[[Tensor], Tensor], optional): Activation function or name. Defaults to 'gelu'.
            norm_first (bool, optional): Whether to apply layer normalization before the attention layer. Defaults to True.

        """
        super().__init__()

        self.token_emb = TE.PatchEmbedding(
            in_features=in_features,
            d_model=d_model,
            patch_size=patch_size,
            patch_stride=patch_stride if patch_stride > 0 else patch_size
        )
        self.pos_emb = PE.SinPosEmbedding(
            d_model=d_model,
            max_len=1024
        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation=activation,
            norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps//patch_stride, d_model)
        '''
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        return x


class PatchTSTBackbone(L.LightningModule):
    # TODO: use:
    #               [cls] token for classification,
    #               [sep] token for regression,
    #               [mask] token for masked reconstruction
    #               [pad] token for padding
    def __init__(self,
                 in_features: int,
                 d_model: int,
                 patch_size: int,
                 patch_stride: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 nhead: int = 4,
                 activation: str | Callable[[Tensor], Tensor] = 'gelu',
                 additional_tokens_at_last: int = 0,
                 norm_first: bool = True) -> None:
        """
        Initializes a PatchTST module.

        Args:
            in_features (int): Number of input features.
            d_model (int): Dimensionality of the model.
            patch_size (int): Size of the patches. <= 16 is recommended.
            patch_stride (int): Stride of the patches. If 0, patch_stride = patch_size, recommended.
            num_layers (int): Number of transformer layers.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            nhead (int, optional): Number of attention heads. Defaults to 4.
            activation (str or Callable[[Tensor], Tensor], optional): Activation function or name. Defaults to 'gelu'.
            additional_tokens_at_last (int, optional): Number of additional tokens to be added at the end of the sequence. 
                These tokens can be used for classification, regression or other tasks. Defaults to 0.
            norm_first (bool, optional): Whether to apply layer normalization before the attention layer. Defaults to True.

        """
        super().__init__()

        self.token_emb = TE.PatchEmbedding(
            in_features=in_features,
            d_model=d_model,
            patch_size=patch_size,
            patch_stride=patch_stride if patch_stride > 0 else patch_size
        )
        self.pos_emb = PE.SinPosEmbedding(
            d_model=d_model,
            max_len=1024
        )
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation=activation,
            norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers
        )

        self.atal = additional_tokens_at_last
        if additional_tokens_at_last > 0:
            self.additional_tokens = nn.Parameter(
                torch.randn(additional_tokens_at_last, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps//patch_stride + atal, d_model)
        '''
        x = self.token_emb(x.permute(0, 2, 1))
        if self.atal > 0:
            x = torch.cat([x, self.additional_tokens[None, :,
                          :].repeat(x.shape[0], 1, 1)], dim=1)
        x = self.pos_emb(x.permute(0, 2, 1))
        x = self.transformer_encoder(x)

        return x
