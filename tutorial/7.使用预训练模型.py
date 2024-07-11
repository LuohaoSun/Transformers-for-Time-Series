# 本项目提供多种预训练时间序列大模型，并统一了调用接口。
# 用户可以直接调用这些模型进行预测，也可以基于这些模型进行二次开发。
import sys

sys.path.append(".")
sys.path.append("..")
import torch
from math import sin, pi


def main():

    # Example usage
    from utils.visualization import SeriesPlotter
    from pretrained.Chronos import Chronos

    model = Chronos(size="tiny")
    sin_wave = (
        torch.tensor([sin(x / pi) for x in range(100)]).unsqueeze(-1).unsqueeze(0)
    )
    context = sin_wave[:, :50, :]
    groud_truth = sin_wave[:, 50:, :]
    forecast = model.forecast(context, 50)
    SeriesPlotter.plot_and_show(
        {
            "forecast": torch.cat([context, forecast], dim=1),
            "groud_truth": torch.cat([context, groud_truth], dim=1),
        }
    )


main()
