import torch
from torch.utils.data import DataLoader
from utils.Dataset import TimeSeriesDataset


def get_datasets(input_len, output_len, division=[6,2,2]):
    """
    获取数据集的函数。

    参数：
    input_len：输入序列的长度。
    output_len：输出序列的长度。
    division：数据集划分比例，默认为[6,2,2]。

    返回值：
    training_dataset：训练数据集。
    val_dataset：验证数据集。
    test_dataset：测试数据集。
    """
    training_dataset = TimeSeriesDataset(
    data_path='data/BJinflow/in_10min_trans.csv', division=division,
    input_len=input_len, output_len=output_len, flag='training')

    val_dataset=TimeSeriesDataset(
    data_path='data/BJinflow/in_10min_trans.csv', division=division,
    input_len=input_len, output_len=output_len, flag='validation')

    test_dataset = TimeSeriesDataset(
    data_path='data/BJinflow/in_10min_trans.csv', division=division,
    input_len=input_len, output_len=output_len, flag='test')

    return training_dataset,val_dataset,test_dataset

def get_dataloaders(input_len, output_len, batch_size=64, division=[6,2,2]):
    """
    获取训练、验证和测试数据加载器。

    参数：
    input_len (int)：输入序列的长度。
    output_len (int)：输出序列的长度。
    batch_size (int, optional)：每个批次的样本数，默认为64。
    division (list, optional)：数据集划分比例，默认为[6,2,2]，表示训练集、验证集和测试集的比例。

    返回值：
    training_dataloader (DataLoader)：训练数据加载器。
    val_dataloader (DataLoader)：验证数据加载器。
    test_dataloader (DataLoader)：测试数据加载器。
    """
    training_dataset,val_dataset,test_dataset=get_datasets(input_len, output_len, division)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return training_dataloader,val_dataloader,test_dataloader

def get_adj_matrix():
    """
    获取邻接矩阵。

    返回值：
    adj_matrix (torch.Tensor)：邻接矩阵 (276, 276)。
    """
    import numpy as np
    import pandas as pd
    adj=pd.read_csv('data/BJinflow/adjacency_with_label.csv')
    adj_matrix=torch.from_numpy(adj.values)
    return adj_matrix

if __name__=='__main__':
    adj=get_adj_matrix()
    print(adj.shape)