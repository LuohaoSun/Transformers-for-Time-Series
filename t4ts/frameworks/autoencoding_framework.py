# Author: Sun LuoHao
# All rights reserved

from typing import Callable, Iterable, Mapping, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..frameworks.utils import get_loss_fn
from .callbacks.autoencoding_callbacks import ViAndLog
from .framework_base import FrameworkBase


class RandomMasker(L.LightningModule):
    def __init__(self, mask_ratio, mask_length=1) -> None:
        """
        mask the input tensor and store the mask in property.
        mask_ratio (float) [0, 1): The ratio of masked tokens in the input sequence.
            A mask_ration = 0 will implement Identity input-output.
        mask_length (int, optional): The length of the masked tokens. Defaults to 1.
            A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
        TODO: learnable mask token
        TODO: customize loss params
        FIXME: not test for mask_ratio == 0
        """
        super().__init__()
        assert 0 <= mask_ratio < 1
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.mask: Tensor = torch.zeros(1, 1, 1, dtype=torch.bool)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        tensor: (batch_size, seq_len, n_features)
        """
        mask = self.create_mask(tensor)
        masked_tensor, mask = self.mask_tensor(tensor, mask)
        self.mask = mask
        return masked_tensor

    def create_mask(self, tensor: Tensor) -> Tensor:
        """
        input: Tensor of shape (b, l, d)
        output: mask of shape (b, l, d) where 1 denotes mask and 0 denotes non-mask
        """
        assert (
            tensor.shape[1] % self.mask_length == 0
        ), f"mask_length should be a divisor of sequence length, but got {self.mask_length} and {tensor.shape[1]}"
        mask = (
            torch.rand(tensor.shape[0], tensor.shape[1] // self.mask_length)
            < self.mask_ratio
        )
        mask = mask.repeat_interleave(self.mask_length, dim=1)
        mask = mask.to(self.device).unsqueeze(dim=-1).repeat(1, 1, tensor.shape[-1])
        return mask

    def mask_tensor(self, tensor: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # here I simply set the masked values to 0, which may not be the best choice.
        masked_tensor = tensor.where(~mask, 0)
        return masked_tensor, mask


class MaskedLoss(L.LightningModule):
    def __init__(
        self,
        loss_type: str = "hybrid",  # full, masked, hybrid
        hybrid_ratio: Iterable[float] = [0.1, 0.9],
    ) -> None:
        """
        full loss: compute loss within all outputs(include both masked and non-masked)
        masked loss: compute loss within only masked inputs and outputs
        hybrid: combine both loss with a fixed ratio.
        """
        super().__init__()
        self.loss_type = loss_type
        self.hybrid_ratio = hybrid_ratio

    def forward(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        if self.loss_type == "full":
            return self.full_mse_loss(x, x_hat)
        elif self.loss_type == "masked":
            return self.masked_mse_loss(x, x_hat, mask)
        elif self.loss_type == "hybrid":
            return self.hybrid_mse_loss(x, x_hat, mask)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def full_mse_loss(self, x: Tensor, x_hat: Tensor, mask=None) -> Tensor:
        return F.mse_loss(x, x_hat)

    def masked_mse_loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        # FIXME: not test for mask = ones or zeros
        masked_values = x[mask]
        reconstructed_values = x_hat[mask]
        return F.mse_loss(masked_values, reconstructed_values)

    def hybrid_mse_loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        full_weight, masked_weight = self.hybrid_ratio
        full_loss = self.full_mse_loss(x, x_hat)
        masked_loss = self.masked_mse_loss(x, x_hat, mask)
        return full_weight * full_loss + masked_weight * masked_loss


class AutoEncodingFramework(FrameworkBase):
    """
    用于时间序列随机遮蔽重构任务的基础模型类，
    针对性地实现了损失、训练、验证、预测、可视化等，且forward方法已经在父类中实现。
    """

    def __init__(
        self,
        # model params
        backbone: nn.Module,
        backbone_out_seq_len: int,
        backbone_out_features: int,
        # task params
        out_seq_len: int,
        out_features: int,
        mask_ratio: float = 0,
        mask_length: int = 1,
        loss_type: str = "full",  # 'full', 'masked', 'hybrid'
        # logging params
        vi_every_n_epochs: int = 1,
        figsize: Tuple[int, int] = (8, 8),
        # additional params
        custom_head: Optional[nn.Module] = None,
    ) -> None:
        """
        Initializes the Framework class.

        Args:
            backbone (nn.Module): The backbone module.
            head (nn.Module): The head module.

            every_n_epochs (int): log the visualization figures every n epochs.
            figsize (Tuple[int, int]): The size of the figure.
            dpi (int): The dpi of the figure.

            mask_ratio (float): The ratio of masked tokens in the input sequence. Defaults to 0.
            mask_length (int, optional): The length of the masked tokens. Defaults to 1.
                A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
            loss_type (str, optional): The type of loss to be used. Can be 'full', 'masked', or 'hybrid'.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.neck = nn.Linear(backbone_out_seq_len, out_seq_len)
        self.head = (
            nn.Linear(backbone_out_features, out_features)
            if custom_head is None
            else custom_head
        )

        assert (
            mask_ratio > 0 or loss_type == "full"
        ), "mask_ratio should be greater than 0 when loss_type is not 'full'"
        self.random_masker = RandomMasker(mask_ratio, mask_length)
        self.masked_loss = MaskedLoss(loss_type=loss_type)
        self.every_n_epochs = vi_every_n_epochs
        self.figsize = figsize

    def loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        # 每次调用都会获取最新的mask
        mask = self.random_masker.mask
        return self.masked_loss(x, x_hat, mask)

    def get_task_callbacks(self):
        return [ViAndLog(self.every_n_epochs, self.figsize)]

    def encode(self, x: Tensor) -> Tensor:
        """
        input: (batch_size, seq_len, n_features)
        output: (batch_size, seq_len, hidden_features)
        """
        return self.backbone(x)

    def decode(self, x: Tensor) -> Tensor:
        """
        input: (batch_size, seq_len, hidden_features)
        output: (batch_size, seq_len, n_features)
        """
        x = self.neck(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.head(x)

    def framework_forward(
        self, x: Tensor, backbone: nn.Module, neck: nn.Module, head: nn.Module
    ) -> Tensor:
        """
        input: (batch_size, seq_len, n_features)
        output: (batch_size, seq_len, n_features)
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def model_step(
        self, batch: Iterable[Tensor], loss_fn: Callable
    ) -> Mapping[str, Tensor]:
        x, y = batch
        masked_input = self.random_masker(x)
        mask = self.random_masker.mask

        x_hat = self.forward(masked_input)
        loss = loss_fn(x, x_hat)

        if self.random_masker.mask_ratio > 0:
            return {
                "loss": loss,
                "original": x,
                "masked_input": masked_input,
                "mask": mask,
                "output": x_hat,
                "masked_values": x.where(mask, 0),
                "reconstructed_values": x_hat.where(mask, 0),
            }
        else:
            return {
                "loss": loss,
                "original": x,
                "output": x_hat,
            }
