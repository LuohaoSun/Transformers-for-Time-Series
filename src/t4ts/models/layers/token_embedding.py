import math

import torch
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
        self, in_features: int, patch_size: int, d_model: int, out_features: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        self.d_model = d_model
        self.out_features = out_features
        
        self.patcher = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if out_features > 0:
            self.unpatcher = nn.Linear(d_model, out_features * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps//patch_stride, d_model)
        """
        x = self.patcher(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def unpatch(self, x: Tensor) -> Tensor:
        # (b, steps//patch_size, d_model) -> (b, steps//patch_size, out_features*patch_size)
        x = self.unpatcher(x)
        # (b, steps//patch_size, out_features*patch_size) -> (b, steps//patch_size, patch_size, out_features)
        x = x.unflatten(-1, (self.patch_size, self.out_features))
        # (b, steps//patch_size, patch_size, out_features) -> (b, steps, out_features)
        x = x.flatten(1, 2)
        return x
