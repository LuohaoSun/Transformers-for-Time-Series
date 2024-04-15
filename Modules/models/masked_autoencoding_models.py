from typing import Callable
from torch import Tensor
from Modules.framework.masked_autoencoding_framework import MaskedAutoEncodingFramework
from Modules.backbones import patchtst
from Modules.heads import autoencoding_heads
from Modules.components.activations import get_activation_fn


class PatchTSTAutoEncodingModel(MaskedAutoEncodingFramework):
    '''
    PatchTST自编码模型，用于时间序列随机遮蔽重构任务。
    '''

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
                 # training params
                 lr: float,
                 max_epochs: int,
                 max_steps: int,
                 mask_ratio: float,
                 mask_length: int,
                 loss_type: str = 'hybrid',  # 'full', 'masked', 'hybrid'
                 ) -> None:
        '''
        Args:
            in_features (int): Number of input features.
            d_model (int): Dimensionality of the model.
            patch_size (int): Size of the patches. <= 16 is recommended.
            patch_stride (int): Stride of the patches. If 0, patch_stride = patch_size, recommended.
            num_layers (int): Number of transformer layers.
            dropout (float, optional): Dropout rate.
            nhead (int, optional): Number of attention heads.
            activation (str or Callable[[Tensor], Tensor], optional): Activation function or name.
            additional_tokens_at_last (int, optional): Number of additional tokens to be added at the end of the sequence. 
                These tokens can be used for classification, regression or other tasks.
            norm_first (bool, optional): Whether to apply layer normalization before the attention layer.

            lr (float): Learning rate.
            max_epochs (int): Maximum number of epochs.
            max_steps (int): Maximum number of steps.
            mask_ratio (float): The ratio of masked tokens in the input sequence.
            mask_length (int, optional): The length of the masked tokens. Defaults to 1. 
                A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
            loss_type (str, optional): The type of loss to be used. Can be 'full', 'masked', or 'hybrid'. Defaults to 'hybrid'.
        '''
        self.save_hyperparameters()
        activation = get_activation_fn(activation)
        super().__init__(
            backbone=patchtst.PatchTSTBackbone(
                in_features, d_model, patch_size, patch_stride, num_layers, dropout, nhead, activation, norm_first),
            head=autoencoding_heads.MLPHead(
                d_model, d_model*4, in_features, activation),
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            mask_ratio=mask_ratio,
            mask_length=mask_length,
            loss_type=loss_type)
