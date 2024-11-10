# TimeGPT is a production ready, generative pretrained transformer for time series.
# It's capable of accurately predicting various domains such as retail, electricity,
# finance, and IoT with just a few lines of code 🚀.
# For more details, please refer to https://github.com/Nixtla/nixtla?tab=readme-ov-file

# To use TimeGPT, you need to install the Nixtla library:
# pip install nixtla

import warnings
from typing import Any, Callable, Dict, Iterable, Mapping, Union

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from nixtla.nixtla_client import NixtlaClient
from torch import Tensor

from .pretrained_base import PretrainedBase


class TimeGPT(PretrainedBase):
    def __init__(
        self,
        task_type: str,  # "forecasting", "anomaly_detection", or "cross_validation"
        output_len: int,
        api_key: str,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        if task_type != "forecasting":
            raise NotImplementedError(
                "TimeGPT is currently only implemented for forecasting tasks."
            )

        self.timegpt = NixtlaClient(api_key=api_key)
        self.output_len = output_len

    def forward(self, x: Tensor) -> Tensor:
        return x

    def _chronos_forecast_3d(self, x: Tensor) -> Tensor:
        x_df = self._tensor_to_df(x)
        y_df = self.timegpt.forecast(x_df, h=self.output_len)
        y = self._df_to_tensor(y_df)
        return y

    def detect_anomalies(self, x: Tensor) -> Tensor:
        x_df = self._tensor_to_df(x)
        y_df = self.timegpt.detect_anomalies(x_df)
        y = self._df_to_tensor(y_df)
        return y

    def on_train_start(self) -> None:
        warnings.warn(
            "TimeGPT is a pre-trained model for time series and is not trainable. "
        )
        return super().on_train_start()

    def _tensor_to_df(self, x: Tensor) -> pd.DataFrame:
        """
        Note that the input of timeGPT is a pandas DataFrame:
        Expected to contain at least the following columns:
            - time_col:
                Column name in `df` that contains the time indices of the time series. This is typically a datetime
                column with regular intervals, e.g., hourly, daily, monthly data points.
            - target_col:
                Column name in `df` that contains the target variable of the time series, i.e., the variable we
                wish to predict or analyze.
            Additionally, you can pass multiple time series (stacked in the dataframe) considering an additional column:
            - id_col:
                Column name in `df` that identifies unique time series. Each unique value in this column
                corresponds to a unique time series.
        """
        raise NotImplementedError()
        return pd.DataFrame(x.cpu().detach().numpy())

    def _df_to_tensor(self, df: pd.DataFrame) -> Tensor:
        """
        Convert a pandas DataFrame to a tensor.
        """
        raise NotImplementedError()
        return torch.tensor(df.values)
