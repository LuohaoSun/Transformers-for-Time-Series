import torch
import torch.nn as nn
from frameworks.forecasting_framework import ForecastingFramework
from ..components.token_embedding import *
from modules.forecasting.forecasting_heads import *


class LSTM(ForecastingFramework):
    def __init__(
        self,
        # model parameters
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        # training parameters
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

            def forward(self, x: Tensor) -> Tensor:
                return self.lstm(x)[0]

        return backbone()

    def configure_head(self, hidden_size, input_size) -> nn.Module:
        return LinearHead(in_features=hidden_size, out_features=input_size)


class PatchTST(ForecastingFramework):
    def __init__(
        self,
        # model parameters
        in_features,
        d_model,
        in_seq_len,
        out_seq_len,
        patch_size=8,
        patch_stride=8,
        num_layers=2,
        dropout=0.1,
        nhead=4,
        activation="gelu",
        norm_first=True,
        # logging parameters
        evry_n_epochs=1,
        fig_size=(10, 5),
        # training parameters
        lr=1e-3,
        max_epochs=10,
        max_steps=-1,
        loss_type="mse",
    ) -> None:
        self.save_hyperparameters()

        super().__init__(
            backbone=self.configure_backbone(),
            head=self.configure_head(),
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            loss_type=loss_type,
            evry_n_epochs=evry_n_epochs,
            fig_size=fig_size,
        )

    def configure_backbone(self):
        from ..backbones.patchtst import PatchTSTBackbone

        return PatchTSTBackbone(
            in_features=self.hparams["in_features"],
            d_model=self.hparams["d_model"],
            patch_size=self.hparams["patch_size"],
            patch_stride=self.hparams["patch_stride"],
            num_layers=self.hparams["num_layers"],
            dropout=self.hparams["dropout"],
            nhead=self.hparams["nhead"],
            activation=self.hparams["activation"],
            additional_tokens_at_last=self.hparams["out_seq_len"],
            norm_first=self.hparams["norm_first"],
        )

    def configure_head(self):
        return LinearHead(
            in_features=self.hparams["d_model"],
            out_features=self.hparams["in_features"],
            out_seq_len=self.hparams["out_seq_len"],
        )
        
        
        

