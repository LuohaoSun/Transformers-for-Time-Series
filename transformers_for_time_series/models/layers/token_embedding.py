import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .gnns import GCN


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
            bias=False,
        )
        if out_features > 0:
            self.unpatcher = nn.Linear(d_model, out_features * patch_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.patch(x)

    def patch(self, x: Tensor) -> Tensor:
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

    def auto_encode(self, x: Tensor) -> Tensor:
        x = self.patch(x)
        x = self.unpatch(x)
        return x


class GCNPatchEmbedding(nn.Module):
    def __init__(
        self,
        adj: Tensor | str,
        in_features: int,
        patch_size: int,
        d_model: int,
        out_features: int = 0,
        activation: str | Callable = "gelu",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_features = in_features
        self.d_model = d_model
        self.out_features = out_features

        if isinstance(adj, str):
            self.gcn = GCN.from_adj_file(
                adj,
                patch_size,
                patch_size * 4,
                patch_size,
                activation,
            )
        else:
            self.gcn = GCN(
                adj,
                patch_size,
                patch_size * 4,
                patch_size,
                activation,
            )
        self.patcher = nn.Linear(patch_size * in_features, d_model, bias=False)
        if out_features > 0:
            self.unpatcher = nn.Linear(d_model, out_features * patch_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.patch(x)

    def patch(self, x: Tensor) -> Tensor:
        """
        x: Tensor with shape (batch_size, steps, in_features)
        returns: Tensor with shape (batch_size, steps//patch_size, d_model)
        """
        # (b, l, d_in) -> (b, l//p, p, d_in)
        x = x.unflatten(1, (self.patch_size, -1))
        # (b, l//p, p, d_in) -> (b, l//p, d_in, p), p as gcn in_features
        x = x.permute(0, 1, 3, 2)
        # (b, l//p, d_in, p) -> (b*l//p, d_in, p)
        s1, s2, s3, s4 = x.shape
        x = self.gcn(x.view(s1 * s2, s3, s4))
        # (b*l//p, d_in, p) -> (b, l//p, d_in, p)
        x = x.view(s1, s2, s3, s4)
        # (b, l//p, d_in, p) -> (b, l//p, p*d_in)
        x = x.flatten(2, 3)
        # (b, l//p, p*d_in) -> (b, l//p, d_model)
        x = self.patcher(x)
        return x

    def unpatch(self, x: Tensor) -> Tensor:
        # (b, steps//patch_size, d_model) -> (b, steps//patch_size, out_features*patch_size)
        x = self.unpatcher(x)
        # (b, steps//patch_size, out_features*patch_size) -> (b, steps//patch_size, patch_size, out_features)
        x = x.unflatten(-1, (self.patch_size, self.out_features))
        # (b, steps//patch_size, patch_size, out_features) -> (b, steps, out_features)
        x = x.flatten(1, 2)
        return x

    def auto_encode(self, x: Tensor) -> Tensor:
        x = self.patch(x)
        x = self.unpatch(x)
        return x
