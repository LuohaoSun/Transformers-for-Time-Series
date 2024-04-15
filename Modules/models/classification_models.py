
import torch.nn as nn
import torch
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from Modules.framework import ClassificationFramework
from Modules.components.activations import get_activation_fn
from Modules.backbones.patchtst import PatchTSTEncoder
from Modules.heads.classification_heads import *


class CustomClassificationModel(ClassificationFramework):
    '''
    custom your own classification model(Beta).
    '''

    def __init__(self,
                 backbone: nn.Module,
                 head: nn.Module,
                 num_classes: int,
                 lr: float,
                 max_epochs: int,
                 max_steps: int,
                 ) -> None:
        '''
        DO NOT IMPLEMENT forward() method in this class, pass the backbone and head is enough.
        note that the inference of the model is y = head(backbone(x)).
        and 
        x is of shape (batch_size, in_seq_len, in_features).
        y is of shape (batch_size, num_classes).
        '''
        super().__init__(backbone,
                         head,
                         num_classes,
                         lr,
                         max_epochs,
                         max_steps)
        self.save_hyperparameters()


class SimpleConv1dClassificationModel(ClassificationFramework):
    def __init__(self,
                 # model params
                 in_features: int,
                 num_classes: int,
                 hidden_features: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 pool_size: int,
                 activation: str | Callable[[Tensor], Tensor],
                 # training params
                 lr: float,
                 max_epochs: int,
                 max_steps: int = -1,
                 ) -> None:
        self.save_hyperparameters()
        backbone = self.configure_backbone()
        head = self.configure_head()
        super().__init__(backbone,
                         head,
                         num_classes,
                         lr,
                         max_epochs,
                         max_steps)

    def configure_backbone(self) -> nn.Module:
        model = self

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = nn.Conv1d(
                    in_channels=model.hparams['in_features'],
                    out_channels=model.hparams['hidden_features'],
                    kernel_size=model.hparams['kernel_size'],
                    stride=model.hparams['stride'],
                    padding=model.hparams['padding']
                )
                self.activation = get_activation_fn(
                    model.hparams['activation'])
                self.pool = nn.MaxPool1d(
                    kernel_size=model.hparams['pool_size'],
                    stride=model.hparams['pool_size']
                )

            def forward(self, x: Tensor) -> Tensor:
                x = self.conv1d(x.permute(0, 2, 1))
                x = self.activation(x)
                x = self.pool(x).permute(0, 2, 1)
                return x

        return Backbone()

    def configure_head(self) -> nn.Module:
        head = MeanLinearHead(
            in_features=self.hparams['hidden_features'],
            out_features=self.hparams['num_classes']
        )
        return head


class PatchTSTClassificationModel(ClassificationFramework):
    def __init__(self,
                 # model params
                 in_features: int,
                 d_model: int,
                 patch_size: int,
                 patch_stride: int,
                 num_layers: int,
                 dropout: float,
                 nhead: int,
                 activation: str | Callable[[Tensor], Tensor],
                 norm_first: bool,
                 num_classes: int,
                 # training params
                 lr: float,
                 max_epochs: int,
                 max_steps: int = -1,
                 ) -> None:

        self.save_hyperparameters()
        backbone = self.configure_backbone()
        head = self.configure_head()

        super().__init__(backbone,
                         head,
                         num_classes,
                         lr,
                         max_epochs,
                         max_steps)

    def configure_backbone(self) -> nn.Module:
        backbone = PatchTSTEncoder(
            in_features=self.hparams['in_features'],
            d_model=self.hparams['d_model'],
            patch_size=self.hparams['patch_size'],
            patch_stride=self.hparams['patch_stride'],
            num_layers=self.hparams['num_layers'],
            dropout=self.hparams['dropout'],
            nhead=self.hparams['nhead'],
            activation=self.hparams['activation'],
            norm_first=self.hparams['norm_first']
        )
        return backbone

    def configure_head(self) -> nn.Module:
        head = MeanLinearHead(
            in_features=self.hparams['d_model'],
            num_classes=self.hparams['num_classes']
        )
        return head
