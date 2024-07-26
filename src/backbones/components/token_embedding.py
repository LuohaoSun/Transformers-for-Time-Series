import torch
import math
import torch.nn as nn
from torch import Tensor


class Conv1dEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input: (batch_size, in_seq_len, in_features)
        output: (batch_size, out_seq_len, out_features)
        """
        x = self.cnn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_features: int, d_model: int, patch_size: int, patch_stride: int
    ):
        super().__init__()
        patch_stride = patch_stride if patch_stride > 0 else patch_size
        self.token_emb = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps//patch_stride, d_model)
        """
        x = self.token_emb(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
