from typing import Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ForecastingDataModule(L.LightningDataModule):
    """
    Convert a .csv file to a LightningDataModule for forecasting tasks.

    __init__: load a csv file and convert it to a LightningDataModule.
    from_datasets: load a list of datasets and convert it to a LightningDataModule. See:
    {L.LightningDataModule.from_datasets}

    """

    def __init__(
        self,
        file_path: str,
        stride: int,
        input_length: int,
        output_length: int,
        batch_size: int,
        train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        num_workers: int = 0,
        normalization: str = "01",  # 01, zscore, minmax, none
        fillna: bool = True,
        dtype: type = np.float32,
    ) -> None:
        """
        Args:
            file_path: str, the path of the .csv or .h5 file. Each row is a time step, and each column is a feature.
            stride: int, the stride of the sliding window.
            input_length: int, the length of the input sequence.
            output_length: int, the length of the output sequence.
            batch_size: int, the batch size.
            train_val_test_split: Tuple[float, float, float], the ratio of train, validation and test set.
            num_workers: int, the number of workers for DataLoader.
            normalization: str, the normalization method. 01, zscore, minmax, none.
        """
        super().__init__()
        self.file_path = file_path
        self.stride = stride
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.normalization = normalization
        self.fillna = fillna
        self.dtype = dtype

    def prepare_data(self) -> None:
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        if self.file_path.endswith(".csv"):
            csv_data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".h5"):
            csv_data = pd.read_hdf(self.file_path)
        else:
            raise ValueError("Invalid file extension")
        if self.fillna:
            csv_data = csv_data.ffill()

        np_data = csv_data.to_numpy(dtype=self.dtype)
        if self.normalization == "01":
            np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
        elif self.normalization == "zscore":
            np_data = (np_data - np_data.mean()) / np_data.std()
        elif self.normalization == "minmax":
            np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
        assert np.isnan(np_data).sum() == 0
        self.dataset = ForecastingDataset(
            np_data,
            self.stride,
            self.input_length,
            self.output_length,
        )
        return

    def setup(self, stage: str | None = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        assert sum(self.train_val_test_split) - 1 < 1e-6
        assert stage in (None, "fit", "validate", "test", "predict")

        if stage == "fit" and not hasattr(self, "train_dataset"):
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, self.train_val_test_split
            )
            assert len(self.train_dataset) > 0
            assert len(self.val_dataset) > 0
        elif stage == "test":
            assert len(self.test_dataset) > 0
        else:
            raise ValueError("Invalid stage")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class ForecastingDataset(Dataset):
    """
    Split a (total length, dim) series to many (input_length, dim) and (output_length, dim) pairs.
    """

    def __init__(
        self,
        series: np.ndarray | pd.DataFrame,
        stride: int,
        input_length: int,
        output_length: int,
    ) -> None:
        super().__init__()
        self.samples = []
        for i in range(0, len(series) - input_length - output_length, stride):
            sample_x = series[i : i + input_length, :]
            sample_x = torch.tensor(sample_x, dtype=torch.float32)
            sample_y = series[i + input_length : i + input_length + output_length, :]
            sample_y = torch.tensor(sample_y, dtype=torch.float32)
            self.samples.append((sample_x, sample_y))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == "__main__":
    pass
