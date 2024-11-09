import torch

from ...external.time_series_lib.models import PatchTST as TSLPatchTST
from ...external.time_series_lib.models import iTransformer as TSLiTransformer


class Configs:
    task_name: str = "long_term_forecast"
    seq_len: int
    pred_len: int
    d_model: int
    dropout: float
    n_heads: int
    d_ff: int
    activation: str
    e_layers: int
    enc_in: int
    embed: str = "fixed"
    freq: str = "h"


class iTransformer(TSLiTransformer.Model):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        dropout: float,
        n_heads: int,
        d_ff: int,
        activation: str,
        e_layers: int,
        enc_in: int,
    ):
        configs = Configs()
        configs.seq_len = seq_len
        configs.pred_len = pred_len
        configs.d_model = d_model
        configs.n_heads = n_heads
        configs.d_ff = d_ff
        configs.e_layers = e_layers
        configs.enc_in = enc_in
        configs.dropout = dropout
        configs.activation = activation
        super().__init__(configs=configs)

    def forward(self, x: torch.Tensor):
        return super().forward(x, None, None, None)


class PatchTST(TSLPatchTST.Model):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        dropout: float,
        n_heads: int,
        d_ff: int,
        activation: str,
        e_layers: int,
        enc_in: int,
        patch_len: int,
        stride: int,
    ):
        configs = Configs()
        configs.seq_len = seq_len
        configs.pred_len = pred_len
        configs.d_model = d_model
        configs.n_heads = n_heads
        configs.d_ff = d_ff
        configs.activation = activation
        configs.dropout = dropout
        configs.e_layers = e_layers
        configs.enc_in = enc_in
        super().__init__(configs=configs, patch_len=patch_len, stride=stride)

    def forward(self, x: torch.Tensor):
        return super().forward(x, None, None, None)
