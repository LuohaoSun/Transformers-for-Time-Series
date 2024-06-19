from typing import Callable, Tuple
from torch import Tensor, nn
from frameworks.autoencoding_framework import AutoEncodingFramework
from ..backbones import patchtst, mlp
from modules.autoencoding import autoencoding_heads
from ..components.activations import get_activation_fn

class MLPAutoEncoder(AutoEncodingFramework):
    def __init__(
        self,
        # model params
        encoder_in_seq_len: int,
        encoder_hidden_len: tuple[int, ...],
        encoder_out_seq_len: int,
        activation: str | Callable[[Tensor], Tensor],
        # logging params
        every_n_epochs: int,
        figsize: Tuple[int, int],
        dpi: int,
        # training params
        lr: float,
        max_epochs: int,
        max_steps: int = -1,
        mask_ratio: float = 0,
        mask_length: int = 1,
        loss_type: str = "full",  # 'full', 'masked', 'hybrid'
    ) -> None:
        """
        decoder is just the encoder in reverse order.
        """
        self.save_hyperparameters()
        backbone = mlp.MLPBackbone(
            in_seq_len=encoder_in_seq_len,
            hidden_len=encoder_hidden_len,
            out_seq_len=encoder_out_seq_len,
            activation=activation,
        )
        head = mlp.MLPBackbone(
            in_seq_len=encoder_out_seq_len,
            hidden_len=encoder_hidden_len[::-1],
            out_seq_len=encoder_in_seq_len,
            activation=activation,
        )
        super().__init__(
            # model params
            backbone=backbone,
            head=head,
            # logging params
            vi_every_n_epochs=every_n_epochs,
            figsize=figsize,
            dpi=dpi,
            # training params
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            mask_ratio=mask_ratio,
            mask_length=mask_length,
            loss_type=loss_type,
        )


class PatchTSTAutoEncodingModel(AutoEncodingFramework):
    """
    PatchTST自编码模型，用于时间序列随机遮蔽重构任务。
    """

    def __init__(
        self,
        # model params
        in_features: int,
        d_model: int,
        patch_size: int,
        patch_stride: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        activation: str | Callable[[Tensor], Tensor],
        additional_tokens_at_last: int,
        norm_first: bool,
        # logging params
        every_n_epochs: int,
        figsize: Tuple[int, int],
        dpi: int,
        # training params
        lr: float,
        max_epochs: int,
        max_steps: int,
        mask_ratio: float,
        mask_length: int = 1,
        loss_type: str = "full",  # 'full', 'masked', 'hybrid'
    ) -> None:
        """
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
        """
        self.save_hyperparameters()
        activation = get_activation_fn(activation)
        backbone = patchtst.PatchTSTBackbone(
            in_features,
            d_model,
            patch_size,
            patch_stride,
            num_layers,
            dropout,
            nhead,
            activation,
            additional_tokens_at_last,
            norm_first,
        )
        head = self.MLPDecoder(d_model, patch_stride, in_features, activation)

        super().__init__(
            backbone=backbone,
            head=head,
            # logging params
            vi_every_n_epochs=every_n_epochs,
            figsize=figsize,
            dpi=dpi,
            # training params
            lr=lr,
            max_epochs=max_epochs,
            max_steps=max_steps,
            mask_ratio=mask_ratio,
            mask_length=mask_length,
            loss_type=loss_type,
        )

    class MLPDecoder(nn.Module):
        """
        decode a (batch_size, in_seq_len//patch_stride, d_model) tensor to a (batch_size, in_seq_len, in_features) tensor
        """

        def __init__(self, d_model, patch_stride, in_features, activation):
            super().__init__()
            self.in_features = in_features
            self.decoder = nn.Sequential(
                nn.Linear(d_model, patch_stride * in_features * 4),
                get_activation_fn(activation),
                nn.Linear(
                    patch_stride * in_features * 4,
                    patch_stride * in_features,
                ),
            )

        def forward(self, x: Tensor) -> Tensor:
            # (b, l//s, d) -> (b, l//s, s*d_in):
            x = self.decoder(x)
            # (b, l//s, s*d_in) -> (b, l, d_in):
            x = x.view(x.size(0), -1, self.in_features)
            return x
