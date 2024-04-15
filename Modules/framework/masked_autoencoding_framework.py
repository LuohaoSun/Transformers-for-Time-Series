# Author: Sun LuoHao
# All rights reserved
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from typing import Mapping, Union, Optional, Callable, Dict, Any, Iterable
from torch import Tensor
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from abc import ABC, abstractmethod
from .framework_base import FrameworkBase

class MaskedAutoEncodingFramework(FrameworkBase, ABC):
    '''
    用于时间序列随机遮蔽重构任务的基础模型类，
    针对性地实现了损失、训练、验证、预测、可视化等，且forward方法已经在父类中实现。

    HOW TO USE:
    1. 子类实现backbone, head属性定制模型参数。
    2. 子类实现mask_ratio, learnable_mask属性定制任务参数。
    3. 子类实现configure_optimizers方法定制训练参数。
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
                 mask_ratio: float,
                 mask_length: int = 1,
                 loss_type: str = 'hybrid',  # 'full', 'masked', 'hybrid'
                 ) -> None:
        """
        Initializes the Framework class.

        Args:
            backbone (nn.Module): The backbone module.
            head (nn.Module): The head module.
            mask_ratio (float): The ratio of masked tokens in the input sequence.
            mask_length (int, optional): The length of the masked tokens. Defaults to 1. 
                A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
            loss_type (str, optional): The type of loss to be used. Can be 'full', 'masked', or 'hybrid'. Defaults to 'hybrid'.
        """
        super().__init__(backbone, head, lr, max_epochs, max_steps)
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.loss_type = loss_type

    def loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
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
        masked_values = x[mask]
        reconstructed_values = x_hat[mask]
        return F.mse_loss(masked_values, reconstructed_values)

    def hybrid_mse_loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        return 0.1*self.full_mse_loss(x, x_hat) + 0.9*self.masked_mse_loss(x, x_hat, mask)

    def mask_timeseries(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        '''
        tensor: (batch_size, seq_len, n_features)
        '''

        assert tensor.shape[1] % self.mask_length == 0, \
            f"mask_length should be a divisor of sequence length, but got {
                self.mask_length} and {tensor.shape[1]}"
        mask = torch.rand(tensor.shape[0],
                          tensor.shape[1]//self.mask_length) < self.mask_ratio
        mask = mask.repeat_interleave(self.mask_length, dim=1)
        mask = mask.unsqueeze(dim=-1).to(self.device)

        # here I simply set the masked values to 0, which may not be the best choice.
        masked_tensor = tensor.where(~mask, 0)
        return masked_tensor, mask

    def training_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Any]:
        x, y = batch
        masked_input, mask = self.mask_timeseries(x)
        x_hat = self.forward(masked_input)

        loss = self.loss(x, x_hat, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        x, y = batch
        masked_input, mask = self.mask_timeseries(x)
        x_hat = self.forward(masked_input)

        loss = self.loss(x, x_hat, mask)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss,
                'original': x,
                'masked_input': masked_input,
                'mask': mask,
                'output': x_hat,
                'masked_values': x.where(mask, 0),
                'reconstructed_values': x_hat.where(mask, 0)
                }

    def test_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        x, y = batch
        masked_input, mask = self.mask_timeseries(x)
        x_hat = self.forward(masked_input)

        loss = self.loss(x, x_hat, mask)
        self.log('test_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return {'loss': loss,
                'original': x,
                'masked_input': masked_input,
                'mask': mask,
                'output': x_hat,
                'masked_values': x.where(mask, 0),
                'reconstructed_values': x_hat.where(mask, 0)
                }

    def predict_step(self, batch: Iterable[Tensor], batch_idx: int) -> Mapping[str, Tensor]:
        return self.test_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs: Mapping[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        '''
        log visualizations of the time series data.
        '''
        if self.trainer.current_epoch % 20 == 0 and batch_idx == 0:     # FIXME: hard coded epoch interval
            run: neptune.Run = self.logger.experiment  # type: ignore
            for key, value in outputs.items():
                img = self.plot_series({key: value})
                run[f'image/{key}'].append(img)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def plot_series(self, series: Tensor | list[Tensor] | dict[str, Tensor]) -> Figure:
        '''
        all series should have the same shape if multiple series are passed.
        series: (batch_size, series_length) or (batch_size, series_length, n_features)
        only first nodes are plotted when the series has a third dimension.
        returns the figure path
        '''
        plt.figure(figsize=(7, 3), dpi=300)    # FIXME: hard coded figure size
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

