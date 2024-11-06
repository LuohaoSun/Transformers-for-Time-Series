from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.figure import Figure
from torch import Tensor


class SeriesPlotter:
    """
    Plot series in a single plot.
    All series are plotted in the same plot. Same shape is expected for all series if multiple series are passed.
    """

    @classmethod
    def plot_and_show(
        cls,
        series: Union[Tensor, List[Tensor], Dict[str, Tensor]],
        figsize=(5, 3),
    ) -> Figure:
        img = cls._plot_series(series, figsize)
        plt.show()
        # plt.close()
        return img

    @classmethod
    def plot_series(
        cls,
        series: Union[Tensor, List[Tensor], Dict[str, Tensor]],
        figsize=(5, 3),
        line_styles: List[str] | None = None,
        markers: List[str] | None = None,  # could be
        colors: List[str] | None = None,
    ) -> Figure:
        """
        Plots the series.

        Args:
            series: The series to plot. It can be a single tensor, a list of tensors, or a dictionary of tensors.
            figsize: (width, height)
            line_styles: Could be "-", "--", "-.", ":"
            markers: Could be
            colors:

        Returns:
            The matplotlib figure object.
        """
        img = cls._plot_series(series, figsize)
        plt.close()
        return img

    @classmethod
    def _plot_series(
        cls,
        series: Union[Tensor, List[Tensor], Dict[str, Tensor]],
        figsize,
    ) -> Figure:

        plt.figure(figsize=figsize, dpi=100)

        if isinstance(series, dict):
            for series_name, series_values in series.items():
                cls._sub_plot(series_values, series_name)
        elif isinstance(series, list):
            for series_values in series:
                cls._sub_plot(series_values)
        else:
            cls._sub_plot(series)

        plt.xlabel("Time steps")
        plt.ylabel("Value")
        plt.legend()
        img = plt.gcf()
        return img

    @staticmethod
    def _sub_plot(series: Tensor, series_name: str = "Series"):
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
