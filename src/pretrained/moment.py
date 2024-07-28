"""
These is a simplified application of the pretrained model from https://github.com/moment-timeseries-foundation-model/moment
forecasting and classification method of the original model implementation is dropped due to the finetuning requirement.

to use MOMENT, run:
```
pip install momentfm
```
"""

from momentfm import MOMENTPipeline

import torch
import torch.nn as nn
import lightning as L

from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable, Optional, List
from .pretrained_base import PretrainedBase

MODEL = "AutonLab/MOMENT-1-large"
D_MODEL = 1024


class MOMENT(PretrainedBase):
    def __init__(self, task: str):  # forecasting, reconstruction, embedding
        """
        MOMENT supports the following tasks:
            - zero-shot reconstruction
            - embedding.
        NOTE: MOMENT cannot be trained, but you can use its embeddings to fine-tune your own head.
        model input:
            - Time-series: Tensor of shape (batch_size, seq_len = 512, channels)
            - Input mask(optional): Tensor of shape (batch_size, seq_len = 512)。长度与时间步相同。
            它用来指示模型应该关注哪些时间步。例如，如果时间序列数据中包含填充（padding）的步骤，可以使用输入掩码来告诉模型忽略这些填充步骤。
            在掩码中，被填充的位置会被标记为零。
            - mask(optional): Tensor of shape (batch_size, seq_len = 512)。长度与时间步相同。
            它用来表示数据中缺失或未观察到的值。模型使用所谓的掩码标记（mask tokens）来替换包含任何缺失时间步的块。
            这样，MOMENT模型在重建过程中可以考虑到这些缺失值。
        model output:
            1. reconstruction: Tensor of shape (batch_size, seq_len = 512, channels)
            2. embedding: Tensor of shape (batch_size, seq_len = 512, d = 1024 for MOMENT-1-large)
        """
        super().__init__(task)
        moment = MOMENTPipeline.from_pretrained(
            MODEL,
            model_kwargs={"task_name": task},
        )
        moment.init()
        self.moment = moment

    def forecast(self, x: Tensor) -> Tensor | None:
        return None

    def reconstruct(self, x: Tensor) -> Tensor:
        assert self.task == "reconstruction"
        x = x.permute(0, 2, 1)
        output = self.moment(x)
        reconstruction: Tensor = output.reconstruction  # type: ignore
        return reconstruction.permute(0, 2, 1)

    def embed(self, x: Tensor) -> Tensor:
        assert self.task == "embedding"
        x = x.permute(0, 2, 1)
        output = self.moment(x)
        embeddings: Tensor = output.embeddings  # type: ignore
        return embeddings.permute(0, 2, 1)

    def _pre_process_input(self, x: Tensor) -> Tensor:
        """
        MOMENT takes 3 inputs:
        An input time series of length 512 timesteps and  channels, and
        Two optional masks, both of length 512.
        The input mask is utilized to regulate the time steps or patches that the model should attend to. For instance, in the case of shorter time series, you may opt not to attend to padding. To implement this, you can provide an input mask with zeros in the padded locations.
        The second mask, referred to simply as mask, denotes masked or unobserved values. We employ mask tokens to replace all patches containing any masked time step (for further details, refer to Section 3.2 in our paper). MOMENT can attend to these mask tokens during reconstruction.
        By default, all time steps are observed and attended to.

        This function pads the input tensor to the required length and generates the masks.
        """
        ...
