import torch
import torch.nn as nn
import lightning as L
from .components import positional_embedding as PE
from .components import token_embedding as TE
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable

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
