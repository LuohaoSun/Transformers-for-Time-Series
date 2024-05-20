import torch
import torch.nn as nn
from framework.forecasting.forecasting_framework import ForecastingFramework
from ..components.token_embedding import *
from model.forecasting_models.forecasting_heads import *


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
        self.save_hyperparameters()
        super().__init__(
            backbone=self.configure_backbone(
                input_size,
                hidden_size // 2 if bidirectional else hidden_size,
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
        class backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            def forward(self,x:Tensor) -> Tensor:
                return self.lstm(x)[0]
        return backbone()

    def configure_head(self, hidden_size, input_size) -> nn.Module:
        return LinearHead(in_features=hidden_size, out_features=input_size)
