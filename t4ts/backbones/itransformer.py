from typing import Any, Callable, Dict, Iterable, Mapping, Union

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from .components import positional_embedding as PE
from .components import token_embedding as TE


class iTransformerBackbone(L.LightningModule):
    '''
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Paper: https://arxiv.org/abs/2310.06625 
    Note: This is a simplified version of the original implementation.
    '''
    def __init__(
        self,
       
        
    ) -> None:
        '''
        Args:
        
        '''
        super().__init__()
