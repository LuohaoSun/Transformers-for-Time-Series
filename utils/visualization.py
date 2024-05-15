import numpy as np
from torch import Tensor
from typing import Union, List, Dict
from matplotlib.figure import Figure

import matplotlib.pyplot as plt


class SeriesPlotter:
    """
    Plot series in a single plot.
    All series are plotted in the same plot. Same shape is expected for all series if multiple series are passed.
    """

    @classmethod
    def plot_and_show(
        cls,
        series: Union[Tensor, List[Tensor], Dict[str, Tensor]],
        figsize=(10, 6),
    ):
        img = cls.plot_series(series, figsize)
        img.show()

    @classmethod
    def plot_series(
        cls,
        series: Union[Tensor, List[Tensor], Dict[str, Tensor]],
        figsize=(10, 6),
    ) -> Figure:
        """
        Plots the series.

        Args:
            series: The series to plot. It can be a single tensor, a list of tensors, or a dictionary of tensors.

        Returns:
            The matplotlib figure object.
        """
        plt.figure(figsize=figsize, dpi=300)

        if isinstance(series, dict):
            for series_name, series_values in series.items():
                cls.sub_plot(series_values, series_name)
        elif isinstance(series, list):
            for series_values in series:
                cls.sub_plot(series_values)
        else:
            cls.sub_plot(series)

        plt.xlabel("Time steps")
        plt.ylabel("Value")
        img = plt.gcf()
        plt.close()
        return img

    @classmethod
    def sub_plot(cls, series: Tensor, series_name: str = "Series"):
        """
        Plots a single series.

        Args:
            series: The series to plot.
            series_name: The name of the series (default: "Series").
        """
        if len(series.shape) == 3:
            # Series has shape (batch_size, series_length, num_nodes)
            x = np.arange(series.shape[1])
            y = series[0, :, 0].cpu().detach().numpy()
        elif len(series.shape) == 2:
            # Series has shape (batch_size, series_length)
            x = np.arange(series.shape[1])
            y = series[0, :].cpu().detach().numpy()
        else:
            # Unsupported series shape
            print(f"Unsupported series shape: {series.shape}. Skipping this series.")
            return

        plt.plot(x, y, label=series_name)
