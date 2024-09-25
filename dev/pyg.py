import imp
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)
        self.gcn = gnn.GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=out_channels,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
