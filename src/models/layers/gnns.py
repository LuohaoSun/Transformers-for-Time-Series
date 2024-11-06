"""
为模型添加静态图模型，增强空间建模能力。
"""

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import overload
from .activations import get_activation_fn


class GCN(nn.Module):
    """
    Graph Convolutional Network.
    args:
        adj (torch.Tensor): Adjacency matrix of the graph.
        input_dim (int): Dimension of the input features. Default is 1.
        hidden_dim (int): Dimension of the hidden features. Default is 4.
        output_dim (int): Dimension of the output features. Default is 1.
        Note that the output_dim should be the same as the input_dim in this implementation.
    """

    def __init__(self, adj, input_dim=1, hidden_dim=4, output_dim=1, activation=F.relu):
        super(GCN, self).__init__()
        assert input_dim == output_dim
        self.residual = input_dim == output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.gc1 = GraphConv(adj, input_dim, hidden_dim)
        self.activation = get_activation_fn(activation)
        self.gc2 = GraphConv(adj, hidden_dim, output_dim)

    @classmethod
    def from_adj_file(
        cls, adj_file_path, input_dim=1, hidden_dim=4, output_dim=1, activation=F.relu
    ):
        adj = pd.read_csv(adj_file_path)
        adj = torch.tensor(adj.values, dtype=torch.float32)
        return cls(adj, input_dim, hidden_dim, output_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        if len(input_shape) == 2:
            x = x.unsqueeze(dim=-1)

        res = x
        x = self.norm(x)

        # Apply the first graph convolutional layer
        x = self.gc1(x)
        x = self.activation(x)

        # Apply the second graph convolutional layer
        x = self.gc2(x)
        x = self.activation(x)

        # Add residual connection
        if self.residual:
            x = x + res

        if len(input_shape) == 2:
            x = x.squeeze(dim=-1)

        return x


class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.

    Args:
        adj (torch.Tensor): Adjacency matrix of the graph.
        input_dim (int): Dimension of the input features. Default is 1.
        output_dim (int): Dimension of the output features. Default is 1.
    """

    def __init__(self, adj, input_dim=1, output_dim=1) -> None:
        super().__init__()

        self.DAD = self.calculate_laplacian_with_self_loop(adj)
        self.num_nodes = self.DAD.shape[0]
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the layer.
        """
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, 0)

    def forward(self, x):
        """
        Perform forward pass of the layer.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output features.
        """
        self.DAD = self.DAD.to(x.device)
        x = torch.matmul(self.DAD, x)
        x = torch.matmul(x, self.weights)
        x = x + self.biases
        return x

    def calculate_laplacian_with_self_loop(self, matrix):
        """
        Calculate the normalized Laplacian matrix with self-loop.

        Args:
            matrix (torch.Tensor): Adjacency matrix of the graph.

        Returns:
            torch.Tensor: Normalized Laplacian matrix.
        """
        matrix = matrix  # .to(torch.device('cpu'))         # 这个函数不兼容mps设备
        if matrix[0, 0] == 0:  # 判断是否存在自环,没有的话加上
            matrix = matrix + torch.eye(matrix.size(0))
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian


class MHGCN(nn.Module):
    pass


class GAT(nn.Module):
    pass


class Graphormer(nn.Module):
    pass
