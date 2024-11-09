import torch
import pandas as pd
from .components.forecasting_datamodule import ForecastingDataModule


def get_forecasting_datamodule(
    input_length,
    output_length,
    batch_size,
    csv_file_path="data/pems_bay/pems_bay.csv",
    train_val_test_split=(0.7, 0.1, 0.2),
    stride=1,
    num_workers=4,
    normalization="01",
):
    """
    获取地铁客流预测数据集。
    """
    datamodule = ForecastingDataModule(
        csv_file_path=csv_file_path,
        stride=stride,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        train_val_test_split=train_val_test_split,
        num_workers=num_workers,
        normalization=normalization,
    )
    return datamodule


def get_adj_matrix(csv_file_path="data/pems_bay/adj_pems_bay.csv"):

    adj = pd.read_csv(csv_file_path)
    adj_matrix = torch.from_numpy(adj.values)
    return adj_matrix
