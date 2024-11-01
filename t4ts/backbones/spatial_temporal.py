from typing import Any

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor


class STBackbone(nn.Module):
    """
    this module provides a simple way to build a spatial-temporal model.
    Only support static graph.
    Args:
        gnn (nn.Module): gnn module from torch_geometric.
        backbone (nn.Module): a time series backbone, e.g., PatchTST.
        adj (torch.Tensor): adjacency matrix (fixed).
    """

    def __init__(
        self,
        gnn: nn.Module,
        backbone: nn.Module,
        adj: torch.Tensor,
        add_self_loop: bool = True,
    ):
        super(STBackbone, self).__init__()
        self.gnn = gnn
        self.backbone = backbone
        self.edge_index = self._convert_to_edge_index(adj, add_self_loop)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape (batch_size, seq_len, in_channels)
        Returns:
            Tensor: output tensor with shape (batch_size, seq_len, out_channels)
        """
        b, n, d = x.size()
        # (batch_size, seq_len, in_channels) -> (batch_size * seq_len, in_channels)
        x = x.view(b * n, d)
        # (batch_size * seq_len, in_channels) -> (batch_size * seq_len, d_model)
        x = self.gnn(x, self.edge_index)
        # (batch_size * seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = x.view(b, n, -1)

        x = self.backbone(x)
        return x

    @torch.no_grad()
    def _convert_to_edge_index(
        self, adj: torch.Tensor, add_self_loop: bool = True
    ) -> torch.Tensor:
        edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
        if add_self_loop:
            row, col = torch.arange(adj.size(0)), torch.arange(adj.size(0))
            edge_index = torch.cat([edge_index, torch.stack([row, col])], dim=1)
        return edge_index
