import torch
import numpy as np
from torch.utils.data import Dataset
'''
data augmentation for 1D sequence data.
'''

class NoisyDataset(Dataset):
    """
    数据增强：给一维序列添加噪声
    """
    def __init__(self, dataset: Dataset, noise_stddev=1e-2) -> None:
        super().__init__()
        self.dataset = dataset
        self.noise_stddev = noise_stddev

    def __getitem__(self, index):
        x, y = self.dataset[index]
        noise = torch.randn_like(x,dtype=torch.float32) * self.noise_stddev
        noisy_x = x + noise
        return noisy_x, y

    def __len__(self):
        return len(self.dataset)

class MaskingDataset(Dataset):
    """
    数据增强：对一维序列进行掩码处理
    """
    def __init__(self, dataset: Dataset, mask_length: list[int] = [32, 128], mask_value: float = 0.0):
        super().__init__()
        self.dataset = dataset
        self.mask_length = mask_length
        self.mask_value = mask_value

    def __getitem__(self, index):
        x, y = self.dataset[index]

        mask_length = torch.randint(*self.mask_length, (1,)).item()
        start = torch.randint(0, max(1, x.shape[0] - mask_length + 1), (1,)).item()
        end = start + mask_length
        masked_x = x.clone()
        masked_x[start:end] = self.mask_value

        return masked_x, y

    def __len__(self):
        return len(self.dataset)

class ShiftDataset(Dataset):
    """
    数据增强：对一维序列进行平移操作
    """
    def __init__(self, dataset: Dataset, max_shift_length: int):
        super().__init__()
        self.dataset = dataset
        self.max_shift_length = max_shift_length

    def __getitem__(self, index):
        x, y = self.dataset[index]

        shift_length = torch.randint(0, self.max_shift_length, (1,)).item()
        shifted_x = torch.roll(x, shifts=shift_length, dims=0)
        if shift_length > 0:
            shifted_x[:shift_length] = 0
        else:
            shifted_x[shift_length:] = 0

        return shifted_x, y

    def __len__(self):
        return len(self.dataset)

class SmoothingDataset(Dataset):
    """
    数据增强：对一维序列进行平滑处理
    """
    def __init__(self, dataset: Dataset, max_window_size: int = 5, padding=True):
        super().__init__()
        self.dataset = dataset
        self.max_window_size = max_window_size
        self.padding = padding

    def __getitem__(self, index):
        x, y = self.dataset[index]
        window_size = torch.randint(1, self.max_window_size, (1,)).item()
        smoothed_x = x.unfold(0, window_size, 1).mean(1)
        if self.padding:
            pad = window_size - 1
            smoothed_x = torch.cat([torch.zeros(pad), smoothed_x])
            assert smoothed_x.shape[0] == x.shape[0]
        return smoothed_x, y

    def __len__(self):
        return len(self.dataset)

class RandomResizedCropDataset(Dataset):
    """
    数据增强：对一维序列进行随机裁剪
    """
    def __init__(self, dataset: Dataset, size: int = 4096, scale: tuple[float, float] = (0.08, 1.0), ratio: tuple[float, float] = (3./4., 4./3.)):
        super().__init__()
        self.dataset = dataset
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.numpy()
        x = np.resize(x, self.size)
        return torch.tensor(x, dtype=torch.float32), y

    def __len__(self):
        return len(self.dataset)
    
class RandomFlipDataset(Dataset):
    """
    数据增强：对一维序列进行随机翻转
    """
    def __init__(self, dataset: Dataset, p: float = 0.5):
        super().__init__()
        self.dataset = dataset
        self.p = p

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if torch.rand(1).item() < self.p:
            x = torch.flip(x, [0])
        return x, y

    def __len__(self):
        return len(self.dataset)
    
class MixupDataset(Dataset):
    """
    数据增强：对一维序列进行mixup
    """
    def __init__(self, dataset: Dataset, alpha: float = 1.0):
        super().__init__()
        self.dataset = dataset
        self.alpha = alpha

    def __getitem__(self, index):
        x1, y1 = self.dataset[index]
        index2 = np.random.randint(0, len(self.dataset))
        x2, y2 = self.dataset[index2]
        lam = np.random.beta(self.alpha, self.alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    def __len__(self):
        return len(self.dataset)
    
class ReversedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x * (-1), y

    def __len__(self):
        return len(self.dataset)
