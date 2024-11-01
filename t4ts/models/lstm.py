import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from .components import token_embedding as TE


class LSTMBackbone(L.LightningModule):
    __doc__ = f"""
    LSTM Backbone.
    The output only contains the 'outputs' of the nn.LSTM Outputs, without '(h, c)'. 
    See {nn.LSTM} for more details.
    TODO: Add support for customizing the token embedding.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        auto_recurrsive_steps: int = 0,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.auto_recurrsive_steps = auto_recurrsive_steps
        self.token_embedding = TE.Conv1dEmbedding(
            in_features=in_features,
            d_model=hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_features,
            hidden_size=hidden_features // (2 if bidirectional else 1),
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, in_seq_len, in_features)
        returns: (batch_size, out_seq_len, out_features) where
            - out_seq_len = in_seq_len + auto_recurrsive_steps
            - out_features = hidden_features if not bidirectional else hidden_features * 2
        """
        x = self.token_embedding(x)
        x, hidden_state = self.lstm(x)
        for _ in range(self.auto_recurrsive_steps):
            step_in = x[:, -1:, :]
            step_out, hidden_state = self.lstm(step_in, hidden_state)
            x = torch.cat([x, step_out], dim=1)
        return x


class LSTMModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = LSTMBackbone(
            in_features=in_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.backbone(x))
