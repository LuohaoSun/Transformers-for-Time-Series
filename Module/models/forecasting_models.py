import torch
import torch.nn as nn
from framework.forecasting.forecasting_framework import ForecastingFramework
from Module.components.token_embedding import *
from Module.heads.forecasting_heads import *


class LSTM(ForecastingFramework):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        lr=1e-3,
        max_epochs=10,
        max_steps=-1,
        loss_type="mse",
    ):

        super().__init__(
            backbone=self.configure_backbone(
                input_size,
                hidden_size,
                num_layers,
                bidirectional,
            ),
            head=self.configure_head(hidden_size, input_size),
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            loss_type=loss_type,
        )

    def configure_backbone(
        self, input_size, hidden_size, num_layers, bidirectional
    ) -> nn.Module:
        lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        return lstm

    def configure_head(self, hidden_size, input_size) -> nn.Module:
        return LinearHead(in_features=hidden_size, out_features=input_size)
