# Author: Sun LuoHao
# All rights reserved

import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import tensorboard
from lightning.pytorch.callbacks import Callback
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable, Tuple
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from abc import ABC, abstractmethod
from .framework_base import FrameworkBase


class RandomMask(L.LightningModule):
    def __init__(self, mask_ratio, mask_length) -> None:
        '''
        mask the input tensor and store the mask in property.
        mask_ratio (float) [0, 1): The ratio of masked tokens in the input sequence.
            A mask_ration = 0 will implement Identity input-output.
        mask_length (int, optional): The length of the masked tokens. Defaults to 1. 
            A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
        TODO: learnable mask token
        FIXME: not test for mask_ratio == 0
        '''
        super().__init__()
        assert 0 <= mask_ratio < 1
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.mask: Tensor

    def forward(self, tensor: Tensor) -> Tensor:
        '''
        tensor: (batch_size, seq_len, n_features)
        '''
        mask = self.create_mask(tensor)
        masked_tensor, mask = self.mask_tensor(tensor, mask)
        self.mask = mask
        return masked_tensor

    def create_mask(self, tensor: Tensor) -> Tensor:
        '''
        input: Tensor of shape (b, l, d)
        output: mask of shape (b, l, d) where 1 denotes mask and 0 denotes non-mask
        '''
        assert tensor.shape[1] % self.mask_length == 0, \
            f"mask_length should be a divisor of sequence length, but got {self.mask_length} and {tensor.shape[1]}"
        mask = torch.rand(tensor.shape[0],
                          tensor.shape[1]//self.mask_length) < self.mask_ratio
        mask = mask.repeat_interleave(self.mask_length, dim=1)
        mask = mask.unsqueeze(dim=-1).to(self.device)
        return mask

    def mask_tensor(self, tensor: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # here I simply set the masked values to 0, which may not be the best choice.
        masked_tensor = tensor.where(~mask, 0)
        return masked_tensor, mask


class MaskedLoss(L.LightningModule):
    def __init__(self,
                 loss_type: str = 'hybrid',  # full, masked, hybrid
                 hybrid_ratio: Iterable[float] = [0.1, 0.9]
                 ) -> None:
        '''
        full loss: compute loss within all outputs(include both masked and non-masked)
        masked loss: compute loss within only masked inputs and outputs
        hybrid: combine both loss with a fixed ratio.
        '''
        super().__init__()
        self.loss_type = loss_type
        self.hybrid_ratio = hybrid_ratio

    def forward(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        if self.loss_type == 'full':
            return self.full_mse_loss(x, x_hat)
        elif self.loss_type == 'masked':
            return self.masked_mse_loss(x, x_hat, mask)
        elif self.loss_type == 'hybrid':
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
        return full_weight*full_loss + masked_weight*masked_loss


class PlotSeriesCallbcak():
    '''
    a callback for auto encoding framework, used to plot original and reconstructed series.
    rely on matplotlib and tensorboard.
    '''

    def __init__(self,
                 every_n_epochs: int = 20,
                 figsize: tuple[int, int] = (10, 5),
                 dpi: int = 300
                 ) -> None:
        self.every_n_epochs = every_n_epochs
        self.figsize = figsize
        self.dpi = dpi

    def __call__(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Mapping[str, Tensor],
        batch: Iterable[Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            experiment: SummaryWriter = trainer.logger.experiment       # type: ignore

            for key, value in outputs.items():
                if key == 'loss':
                    continue
                img = self.plot_series({key: value})
                experiment.add_figure(tag=key, figure=img)

    def plot_series(self, series: Tensor | list[Tensor] | dict[str, Tensor]) -> Figure:
        '''
        all series should have the same shape if multiple series are passed.
        series: (batch_size, series_length) or (batch_size, series_length, n_features)
        only first nodes are plotted when the series has a third dimension.
        returns the figure path
        '''
        plt.figure(figsize=self.figsize,
                   dpi=self.dpi)    # FIXME: hard coded figure size
        if isinstance(series, dict):
            for series_name, series_values in series.items():
                self.sub_plot(series_values, series_name)
        elif isinstance(series, list):
            for series_values in series:
                self.sub_plot(series_values)
        else:
            self.sub_plot(series)
        # plt.legend()
        plt.xlabel("Time steps")
        plt.ylabel("Value")
        img = plt.gcf()
        plt.close()
        return img

    def sub_plot(self, series: Tensor, series_name: str | None = None):
        '''
        series: (batch_size, series_length) or (batch_size, series_length, num_nodes)
        only first plot_nodes nodes are plotted when the series has a third dimension.
        '''
        series_name = series_name if series_name is not None else 'Series'
        # plot a single series
        if len(series.shape) == 3:  # Series has shape (batch_size, series_length, num_nodes)
            x = np.arange(series.shape[1])
            y = series[0, :, 0].cpu().detach().numpy()
        elif len(series.shape) == 2:  # Series has shape (batch_size, series_length)
            x = np.arange(series.shape[1])
            y = series[0, :].cpu().detach().numpy()
        else:
            UserWarning(
                f'''
                Unsupported series shape.
                Expected (batch_size, series_length)
                or (batch_size, series_length, num_nodes),
                but got shape {series.shape} instead.
                Skipping this series.
                '''
            )
            return
        plt.plot(x, y, label=series_name)


class AutoEncodingFramework(FrameworkBase, ABC):
    '''
    用于时间序列随机遮蔽重构任务的基础模型类，
    针对性地实现了损失、训练、验证、预测、可视化等，且forward方法已经在父类中实现。
    '''

    @abstractmethod
    def __init__(self,
                 # model params
                 backbone: nn.Module,
                 head: nn.Module,
                 # training params
                 lr: float,
                 max_epochs: int,
                 max_steps: int,
                 mask_ratio: float = 0,
                 mask_length: int = 1,
                 loss_type: str = 'full',  # 'full', 'masked', 'hybrid'
                 ) -> None:
        """
        Initializes the Framework class.

        Args:
            backbone (nn.Module): The backbone module.
            head (nn.Module): The head module.
            mask_ratio (float): The ratio of masked tokens in the input sequence. Defaults to 0.
            mask_length (int, optional): The length of the masked tokens. Defaults to 1. 
                A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
            loss_type (str, optional): The type of loss to be used. Can be 'full', 'masked', or 'hybrid'. 
        """
        super().__init__(backbone, head, lr, max_epochs, max_steps)
        self.random_mask = RandomMask(mask_ratio, mask_length)
        # TODO: customize loss params
        self.loss_func = MaskedLoss(loss_type=loss_type)
        assert mask_ratio > 0 or loss_type == 'full', "mask_ratio should be greater than 0 when loss_type is not 'full'"
        self.mask_ratio = mask_ratio
        # TODO: customize plotter params
        self.series_plotter = PlotSeriesCallbcak()

    def loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        '''
        input: (batch_size, seq_len, n_features)
        output: scalar
        '''
        mask = self.random_mask.mask
        return self.loss_func(x, x_hat, mask)

    def training_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        step_output = self.test_step(batch, batch_idx)
        loss = step_output['loss']
        self.log('train_loss', loss)
        return step_output

    def validation_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        step_output = self.test_step(batch, batch_idx)
        loss = step_output['loss']
        self.log('val_loss', loss)
        return step_output

    def test_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        x, y = batch
        masked_input = self.random_mask(x)
        mask = self.random_mask.mask
        x_hat = self.forward(masked_input)

        loss = self.loss(x, x_hat)

        if self.mask_ratio > 0:
            return {'loss': loss,
                    'original': x,
                    'masked_input': masked_input,
                    'mask': mask,
                    'output': x_hat,
                    'masked_values': x.where(mask, 0),
                    'reconstructed_values': x_hat.where(mask, 0)
                    }
        else:
            return {'loss': loss,
                    'original': x,
                    'output': x_hat,
                    }

    def predict_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        raise NotImplementedError()

    def encode(self, x: Tensor) -> Tensor:
        '''
        input: (batch_size, seq_len, n_features)
        output: (batch_size, seq_len, hidden_features)
        '''
        return self.backbone(x)

    def decode(self, x: Tensor) -> Tensor:
        '''
        input: (batch_size, seq_len, hidden_features)
        output: (batch_size, seq_len, n_features)
        '''
        return self.head(x)

    def on_validation_batch_end(self, outputs: Mapping[str, Tensor], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.series_plotter(self.trainer, self, outputs,
                            batch, batch_idx, dataloader_idx)
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
