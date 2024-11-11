import numpy as np
import torch

from .components.forecasting_datamodule import ForecastingDataModule


def get_forecasting_datamodule(
    input_length,
    output_length,
    batch_size,
    file_path="data/ca_2021_15min_resample/ca_his_2021.h5",
    train_val_test_split=(0.7, 0.1, 0.2),
    stride=1,
    num_workers=4,
    normalization="zscore",
):
    """
    获取地铁客流预测数据集。
    """
    datamodule = ForecastingDataModule(
        file_path=file_path,
        stride=stride,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        train_val_test_split=train_val_test_split,
        num_workers=num_workers,
        normalization=normalization,
    )
    return datamodule


def get_adj_matrix(file_path="data/ca_2021_15min_resample/ca_rn_adj.npy"):
    adj = np.load(file_path)
    adj_matrix = torch.from_numpy(adj)
    return adj_matrix
