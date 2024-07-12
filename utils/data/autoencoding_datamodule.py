from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightning as L
import pandas as pd


class AutoEncodingDataModule(L.LightningDataModule):
    __doc__ = f"""
    __init__: load a csv file and convert it to a LightningDataModule.
    from_datasets: load a list of datasets and convert it to a LightningDataModule. See:
    > {L.LightningDataModule.from_datasets.__doc__}

    """

    def __init__(
        self,
        csv_file_path: str,
        input_length: int,
        batch_size: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 1,
        ignore_last_cols: int = 0,
    ) -> None:
        super().__init__()
        self.file_path = csv_file_path
        self.input_length = input_length
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers

    def prepare_data(self) -> None:


        return

    def setup(self, stage: str | None = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        assert sum(self.train_val_test_split) - 1 < 1e-6
        assert stage in (None, "fit", "validate", "test", "predict")
        pass

    def train_dataloader(self):
        raise NotImplementedError
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    @classmethod
    def from_directory(
        cls,
        directory: str,
        input_length: int,
        batch_size: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 1,
        ignore_last_cols: int = 0,
    ) -> "AutoEncodingDataModule":
        """
        read a directory of multipl csv files and convert it to a LightningDataModule.
        """
        ...


class AutoEncodingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        input_length: int,
        stride: int,
        ignore_last_cols: int,
    ):
        self.data = data
        self.input_length = input_length
        self.ignore_last_cols = ignore_last_cols

        self.samples = []

        for i in tqdm(range(0, len(data) - input_length + 1, stride)):
            x = data.iloc[i : i + input_length]
            x = torch.tensor(x.values, dtype=torch.float32)
            self.samples.append((x, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @classmethod
    def from_csv(
        cls,
        csv_file_path: str,
        input_length: int,
        stride: int,
        ignore_last_cols: int,
    ) -> "AutoEncodingDataset":
        data = pd.read_csv(csv_file_path)
        return cls(data, input_length, stride, ignore_last_cols)
