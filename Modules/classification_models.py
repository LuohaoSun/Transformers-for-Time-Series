import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor

from Modules.components.framework import ClassificationFramework
from Modules.components.activations import get_activation_fn
from Modules.components.patchtst import PatchTSTEncoder
from Modules.components.heads import *


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
        super().__init__(backbone,
                         head,
                         num_classes,
                         lr,
                         max_epochs,
                         max_steps)
        self.save_hyperparameters()


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
            out_features=self.hparams['num_classes']
        )
        return head
