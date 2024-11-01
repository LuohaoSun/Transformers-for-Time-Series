import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset


class sin_dataset_univariate(Dataset):
    def __init__(self, sample_lenght, length, freq, phase, amplitude):
        self.sin_wave = create_sin_wave(freq, phase, amplitude, length)


def create_sin_wave(freq, phase, amplitude, length):
    """
    创建一个正弦波信号。

    参数:
    - freq: 频率，单位为Hz。
    - phase: 相位，单位为弧度。
    - amplitude: 振幅。
    - length: 信号长度，单位为秒。

    返回:
    - t: 时间数组。
    - y: 正弦波信号数组。
    """
    # 采样率，足够高以满足奈奎斯特准则
    sampling_rate = 2 * freq * 10
    # 生成时间数组
    t = np.linspace(0, length, int(sampling_rate * length), endpoint=False)
    # 生成正弦波信号
    y = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, y


# 使用示例
freq = 5  # 频率为5Hz
phase = np.pi / 4  # 相位为π/4弧度
amplitude = 1  # 振幅为1
length = 2  # 长度为2秒

t, y = create_sin_wave(freq, phase, amplitude, length)

# 绘制正弦波
plt.plot(t, y)
plt.title("Sin Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
