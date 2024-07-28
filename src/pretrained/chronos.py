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

from chronos import ChronosPipeline

import torch
import torch.nn as nn
import lightning as L

from torch import Tensor
from typing import Any, Dict, Iterable, Mapping, Union, Callable, Optional, List
from .pretrained_base import PretrainedBase


class Chronos(nn.Module):
    def __init__(
        self,
        task: str,  # "forecasting" or "embedding"
        out_seq_len: int = 0,
        num_samples: int = 1,
        size: str = "small",  # "tiny", "mini", "small", "base", "large"
        device_map="cpu",  # use "cpu" for CPU inference, "cuda" for nVidia and "mps" for Apple Silicon
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> None:
        """
        NOTE: original Chronos model only supports CPU inference.
        You can modify the source code to support GPU inference.
        A bigger num_samples leads to higher precision but also increases memory usage.
        Set a smaller value to reduce memory usage.

        Model	            Parameters	Based on            Storage
        chronos-t5-tiny	    8M	        t5-efficient-tiny   30MB
        chronos-t5-mini	    20M	        t5-efficient-mini   80MB
        chronos-t5-small	46M	        t5-efficient-small  200MB
        chronos-t5-base	    200M	    t5-efficient-base   800MB
        chronos-t5-large	710M	    t5-efficient-large  2.8GB
        """
        super().__init__()

        self.task = task
        self.out_seq_len = out_seq_len
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.limit_prediction_length = limit_prediction_length

        self.chronos = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map=device_map,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    def forward(
        self,
        x: torch.Tensor,
        out_seq_len: int = 0,
        num_samples: int = 0,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, n_features)
        returns: Tensor of shape (batch_size, out_seq_len, n_features)
        """
        num_samples = num_samples or self.num_samples
        out_seq_len = out_seq_len or self.out_seq_len
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        top_p = top_p or self.top_p
        limit_prediction_length = (
            limit_prediction_length or self.limit_prediction_length
        )
        if self.task == "forecasting":
            return self._chronos_forecast_3d(
                x,
                self.out_seq_len,
                num_samples,
                temperature,
                top_k,
                top_p,
                limit_prediction_length,
            )
        elif self.task == "embedding":
            return self._chronos_embed_3d(x)[:, :-out_seq_len, :]
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

    def _chronos_forecast_3d(
        self,
        x: torch.Tensor,
        prediction_length: int,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> torch.Tensor:
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
        x = self._chronos_forecast_2d(
            x,
            prediction_length,
            num_samples,
            temperature,
            top_k,
            top_p,
            limit_prediction_length,
        )
        x = x.reshape(batch_size, n_features, prediction_length).transpose(1, 2)
        return x

    def _chronos_embed_2d(self, x: Tensor) -> Tensor:
        """
        Forecasting using the Chronos model.
        x: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len)
        TODO: the embeddings output is moved to cpu by default. see {self.chronos.embed}
        """
        embeddings, tokenizer_state = self.chronos.embed(x)
        return embeddings.to(x.device)

    def _chronos_forecast_2d(
        self,
        x: torch.Tensor,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> torch.Tensor:
        """
        Forecasting using the Chronos model.
        Chronos uses sampling to get results from tokens.
        x: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, num_samples, seq_len)
        forecast = self.chronos.predict(
            x,
            prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            limit_prediction_length=limit_prediction_length,
        )
        forecast = forecast.mean(dim=1)
        return forecast.to(x.device)
