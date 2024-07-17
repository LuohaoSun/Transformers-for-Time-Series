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
    def __init__(
        self, task: str  # forecasting, classification, reconstruction, embedding
    ):
        super().__init__(task)
        moment = MOMENTPipeline.from_pretrained(
            MODEL,
            model_kwargs={"task_name": task},
        )
        moment.init()
        self.moment = moment

    def forward(self, x: Tensor):
        x = x.permute(0, 2, 1)
        output = self.moment(x)
        reconstruction: Tensor = output.reconstruction  #type: ignore
        anomaly_scores = output.anomaly_scores
        return reconstruction.permute(0, 2, 1)

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
