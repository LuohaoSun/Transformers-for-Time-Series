from typing import Tuple
import lightning as L
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import Tensor
from rich.progress import Progress


class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int, stride: int):
        """
        Args:
            data: np.ndarray, the input data. Each row is a time step, and each column is a feature.
            input_length: int, the length of the input sequence.
            output_length: int, the length of the output sequence.
            stride: int, the stride of the sliding window.
        NOTE: __getitem__ returns only ONE Tensor of shape (len, channel), 
            i.e., the windowed sequence of shape (window_size, num_cols).
        """
        self.samples: list[Tensor] = []
        progress = Progress()
        task = progress.add_task("[cyan]Creating samples...", total=(len(data) - window_size) // stride)
        
        for i in range(0, len(data) - window_size, stride):
            sample = data[i : i + window_size]
            sample = torch.tensor(sample, dtype=torch.float32)
            self.samples.append(sample)
            
            progress.update(task, advance=1)
        
        progress.stop()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Tensor:
        return self.samples[index]

    @classmethod
    def from_csv(
        cls,
        csv_file_path: str,
        window_size: int,
        stride: int,
        **pd_read_csv_kwargs,
    ) -> "SlidingWindowDataset":
        """Create a dataset from a CSV file.

        Args:
            csv_file_path (str): The path to the CSV file.
            window_size (int): The size of the sliding window.
            stride (int): The stride of the sliding window.
            **pd_read_csv_kwargs: Additional keyword arguments to pass to `pd.read_csv`.

        Returns:
            Dataset: The created dataset.
        """
        data = pd.read_csv(csv_file_path, **pd_read_csv_kwargs).to_numpy()
        return cls(data, window_size, stride)

    @classmethod
    def from_directory(
        cls,
        directory: str,
        window_size: int,
        stride: int,
        **pd_read_csv_kwargs,
    ) -> ConcatDataset:
        """
        Creates a dataset by reading multiple CSV files from a directory and applying sliding window transformation.

        Args:
            directory (str): The directory path containing the CSV files.
            window_size (int): The size of the sliding window.
            stride (int): The stride value for the sliding window.
            **pd_read_csv_kwargs: Additional keyword arguments to be passed to the `pd.read_csv` function.

        Returns:
            Dataset: The concatenated dataset created from the CSV files.

        """
        import os

        csv_files_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        datasets = [
            cls.from_csv(f, window_size, stride, **pd_read_csv_kwargs)
            for f in csv_files_paths
        ]
        return ConcatDataset(datasets)
