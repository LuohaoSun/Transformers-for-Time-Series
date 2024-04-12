import torch
from torch.utils.data import DataLoader, Dataset
from data.data_augmentation import NoisyDataset, MaskingDataset, ShiftDataset, SmoothingDataset, MixupDataset, ReversedDataset
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
SAMPLE_PREDICT_DIR = 'data/fault_predict/test2'

def get_dataloaders(batch_size: int | list[int] = [2000, 2000, 2000], shuffle=True):
    """
    获取数据加载器的函数。

    参数：
    - batch_size：每个批次的样本数量，默认为64。
    - shuffle：是否对数据进行洗牌，默认为True。

    返回值：
    - training_dataloader：训练数据加载器。
    - val_dataloader：验证数据加载器。
    - prediction_dataloader：预测数据加载器。注意，预测数据集的标签全部为0（假的）。
    """
    if isinstance(batch_size, int):
        batch_size = [batch_size]*3
    training_dataset, val_dataset, prediction_dataset = get_datasets()
    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size[0], shuffle=shuffle, persistent_workers=True,  num_workers=2)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size[1], persistent_workers=True, num_workers=2)
    prediction_dataloader = DataLoader(
        prediction_dataset, batch_size=batch_size[2], persistent_workers=True, num_workers=2)

    return training_dataloader, val_dataloader, prediction_dataloader


def get_datasets(division=[9, 1]):
    """
    Get the training, validation, and prediction datasets.

    Args:
        division (list, optional): The division ratio between training and validation datasets. Defaults to [9, 1].

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, and prediction dataset.
        note that the labels of the prediction dataset are all 0 (fake).
    """
    sample_0_dir = 'data/fault_predict/train/0'
    sample_1_dir = 'data/fault_predict/train/1'
    sample_2_dir = 'data/fault_predict/train/2'
    sample_3_dir = 'data/fault_predict/train/3'
    sample_predict_dir = SAMPLE_PREDICT_DIR

    sample_0 = [torch.tensor(np.loadtxt(sample_0_dir+'/'+file),
                             dtype=torch.float32) for file in os.listdir(sample_0_dir)]
    sample_1 = [torch.tensor(np.loadtxt(sample_1_dir+'/'+file),
                             dtype=torch.float32) for file in os.listdir(sample_1_dir)]
    sample_2 = [torch.tensor(np.loadtxt(sample_2_dir+'/'+file),
                             dtype=torch.float32) for file in os.listdir(sample_2_dir)]
    sample_3 = [torch.tensor(np.loadtxt(sample_3_dir+'/'+file),
                             dtype=torch.float32) for file in os.listdir(sample_3_dir)]
    sample_prediction = [torch.tensor(np.loadtxt(
        sample_predict_dir+'/'+file), dtype=torch.float32) for file in sorted(os.listdir(sample_predict_dir), key=lambda x: int(x.split('.')[0]))]

    # Calculate the division ratio
    total_length = len(sample_0) + len(sample_1) + \
        len(sample_2) + len(sample_3)
    # NOTE: here suppose lengthes of all samples with various labels are equal!
    training_length = int(len(sample_0) * division[0] / sum(division))
    val_length = total_length - training_length

    # Split the datasets
    training_dataset = List2Dataset(
        sample_0[:training_length], sample_1[:training_length], sample_2[:training_length], sample_3[:training_length])
    val_dataset = List2Dataset(sample_0[training_length:], sample_1[training_length:],
                               sample_2[training_length:], sample_3[training_length:])
    prediction_dataset = List2Dataset(sample_prediction)

    return training_dataset, val_dataset, prediction_dataset


def get_augmented_datasets(dataset: Dataset, noise_stddev=0.1, mask_length=[1000, 3000], mask_value=0.0, max_shift_length=32, max_smoothing_window=32):
    """
    获取增强的数据集。

    Args:
        dataset (Dataset): 原始数据集。
        noise_stddev (float, optional): 噪声标准差。默认为1e-2。
        mask_length (list, optional): 掩码长度范围。默认为[32, 128]。
        mask_value (float, optional): 掩码值。默认为0.0。
        max_shift_length (int, optional): 最大平移长度。默认为10。
        smoothing_window (int, optional): 平滑窗口大小。默认为5。

    Returns:
        tuple: 包含噪声数据集、掩码数据集、平移数据集和平滑数据集的元组。
    """
    noisy_dataset = NoisyDataset(dataset, noise_stddev)
    masking_dataset = MaskingDataset(dataset, mask_length, mask_value)
    shift_dataset = ShiftDataset(dataset, max_shift_length)
    smoothing_dataset = SmoothingDataset(dataset, max_smoothing_window)
    mixup_dataset = MixupDataset(dataset)
    reversed_dataset = ReversedDataset(dataset)
    return noisy_dataset, masking_dataset, shift_dataset, smoothing_dataset, mixup_dataset, reversed_dataset


class List2Dataset(Dataset):
    """
    A custom dataset class that converts multiple lists into a single dataset. 
    Note that the labels are the index of the args.

    Args:
        *sample_lists: Variable number of lists containing the samples.

    Attributes:
        sample_lists (tuple): Tuple containing the input sample lists.
        sample_length (list): List containing the length of each sample list.

    Methods:
        __getitem__(self, index): Retrieves the sample at the given index.
        __len__(self): Returns the total number of samples in the dataset.
    """

    def __init__(self, *sample_lists, normlize='zscore'):
        super().__init__()
        self.sample_lists = sample_lists
        self.sample_length = [len(sample_list) for sample_list in sample_lists]
        self.normlize = normlize

    def __getitem__(self, index):
        sample_list_index = 0
        while index >= self.sample_length[sample_list_index]:
            index -= self.sample_length[sample_list_index]
            sample_list_index += 1

        one_hot_index = torch.tensor(sample_list_index)
        one_hot = F.one_hot(one_hot_index, num_classes=4)   # FIXME: hard code

        x = self.sample_lists[sample_list_index][index]
        y = one_hot.to(torch.float32)
        if self.normlize is None:
            pass
        elif self.normlize == 'minmax':
            x = (x - x.min()) / (x.max() - x.min())
        elif self.normlize == 'zscore':
            x = (x - x.mean()) / x.std()
        elif self.normlize == 'mean':
            x = x / x.mean()
        elif self.normlize == 'max':
            x = x / x.max()
        elif self.normlize == 'log':
            x = torch.log(x + 1)
        elif self.normlize == 'l2':
            x = F.normalize(x, p=2, dim=0)
        else:
            raise ValueError('normlize method not supported')
        return x, y

    def __len__(self):
        return sum(self.sample_length)


if __name__ == '__main__':
    training_dataset, val_dataset, pred_dataset = get_datasets()
    print(len(training_dataset))
    x, y = training_dataset[0]
    print(x, y)
