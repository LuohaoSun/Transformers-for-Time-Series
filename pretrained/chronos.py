"""
Chronos is a family of pretrained time series forecasting models based on language model architectures. 
A time series is transformed into a sequence of tokens via scaling and quantization, and 
a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic 
forecasts are obtained by sampling multiple future trajectories given the historical context. 
Chronos models have been trained on a large corpus of publicly available time series data, as well as 
synthetic data generated using Gaussian processes.
For more details, please refer to https://github.com/amazon-science/chronos-forecasting

To use Chronos, you need to install the Chronos library:
pip install git+https://github.com/amazon-science/chronos-forecasting.git
"""

from math import sin
from chronos import ChronosPipeline

from cycler import V
import torch
import torch.nn as nn
import lightning as L
import warnings
import pandas as pd
from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable, Optional, List
from .pretrained_base import PretrainedBase


class Chronos(PretrainedBase):
    def __init__(
        self,
        task: str,  # "forecasting" or "embedding"
        out_seq_len: int = 1,
        size: str = "small",  # "tiny", "mini", "small", "base", "large"
        device_map="cpu",  # use "cpu" for CPU inference, "cuda" for nVidia and "mps" for Apple Silicon
    ) -> None:
        """
        Model	            Parameters	Based on            d_model
        chronos-t5-tiny	    8M	        t5-efficient-tiny
        chronos-t5-mini	    20M	        t5-efficient-mini
        chronos-t5-small	46M	        t5-efficient-small
        chronos-t5-base	    200M	    t5-efficient-base
        chronos-t5-large	710M	    t5-efficient-large
        """
        super().__init__()
        self.task = task
        self.out_seq_len = out_seq_len
        self.chronos = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map=device_map,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, n_features)
        returns: Tensor of shape (batch_size, out_seq_len, n_features)
        """
        if self.task == "forecasting":
            return self._chronos_forecast_3d(x, self.out_seq_len)
        elif self.task == "embedding":
            return self._chronos_embed_3d(x)[:, -self.out_seq_len :, :]
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _chronos_embed_3d(self, x: Tensor) -> Tensor:
        """
        NOTE: The vanilla Chronos only supports (b, l).
        This method is adjusted to support (b, l, d) input by channel-independent processing, i.e.
            (b, l, d) -> (b, d, l) -> (bd, l) -> Chronos -> (bd, l) -> (b, d, l) -> (b, l, d)
        x: Tensor of shape (batch_size, seq_len, n_features)
        returns: Tensor of shape (batch_size, seq_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape
        x = x.transpose(1, 2).reshape(batch_size * n_features, seq_len)
        x = self._chronos_embed_2d(x)
        x = x.reshape(batch_size, n_features, seq_len).transpose(1, 2)
        return x

    def _chronos_forecast_3d(self, x: Tensor, out_seq_len) -> Tensor:
        """
        Forecasting using the Chronos model.
        NOTE: The vanilla Chronos only supports (b, l).
        This method is adjusted to support (b, l, d) input by channel-independent processing, i.e.
            (b, l, d) -> (b, d, l) -> (bd, l) -> Chronos -> (bd, l_out) -> (b, d, l_out) -> (b, l_out, d)
        x: Tensor of shape (batch_size, seq_len, n_features)
        returns: Tensor of shape (batch_size, seq_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape
        x = x.transpose(1, 2).reshape(batch_size * n_features, seq_len)
        x = self._chronos_forecast_2d(x, out_seq_len)
        x = x.reshape(batch_size, n_features, out_seq_len).transpose(1, 2)
        return x

    def _chronos_embed_2d(self, x: Tensor) -> Tensor:
        """
        Forecasting using the Chronos model.
        x: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len)
        TODO: the embeddings output is moved to cpu by default. see {self.chronos.embed}
        """
        embeddings, tokenizer_state = self.chronos.embed(x)
        return embeddings

    def _chronos_forecast_2d(
        self, x: Tensor, prediction_length: int, **kwargs
    ) -> Tensor:
        """
        Forecasting using the Chronos model.
        Chronos uses sampling to get results from tokens.
        x: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, num_samples, seq_len)
        forecast = self.chronos.predict(x, prediction_length, **kwargs, num_samples=1)
        forecast = forecast.mean(dim=1)
        return forecast
