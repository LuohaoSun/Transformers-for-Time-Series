from t4ts.utils.data import ForecastingDataModule
import torch


class BJInflowDataModule(ForecastingDataModule):
    def __init__(
        self,
        input_length,
        output_length,
        batch_size,
        train_val_test_split=(0.7, 0.1, 0.2),
        stride=1,
        num_workers=4,
        normalization="01",
    ):

        super().__init__(
            csv_file_path="data/BJinflow/in_10min_trans.csv",
            stride=stride,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
            train_val_test_split=train_val_test_split,
            num_workers=num_workers,
            normalization=normalization,
        )

def get_adj_matrix():
    """
    获取邻接矩阵。

    返回值：
    adj_matrix (torch.Tensor)：邻接矩阵 (276, 276)。
    """
    import pandas as pd
    import numpy as np
    adj=pd.read_csv('data/BJinflow/adjacency_with_label.csv')
    adj_matrix=torch.from_numpy(adj.values)
    return adj_matrix