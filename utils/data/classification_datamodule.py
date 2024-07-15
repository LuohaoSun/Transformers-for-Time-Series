from typing import Tuple
import lightning as L
import pandas as pd
from rich import print
import torch
from tqdm import tqdm

from typing import IO, Any, Dict, Iterable, Optional, Union, cast
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split


class ClassificationDataModule(L.LightningDataModule):
    __doc__ = f"""
    __init__: load a csv file and convert it to a LightningDataModule.
    from_datasets: load a list of datasets and convert it to a LightningDataModule. See:
    > {L.LightningDataModule.from_datasets.__doc__}

    """

    def __init__(
        self,
        csv_file_path: str,
        batch_size: int,
        input_length: int = 1,
        output_length: int = 1,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.csv_data = pd.read_csv(csv_file_path)
        self.dataset = ClassificationDataset(
            self.csv_data, self.input_length, self.output_length
        )
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        assert self.num_workers > 0, "num_workers must be greater than 0"
        assert (
            sum(self.train_val_test_split) - 1 < 1e-6
        ), "train_val_test_split must sum to 1"

    @property
    def num_classes(self):
        return self.dataset.num_classes

    def prepare_data(self) -> None:
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed

        return

    def setup(self, stage: str | None = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        assert stage in (None, "fit", "validate", "test", "predict")
        if not hasattr(self, "train_dataset"):
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [
                    int(len(self.dataset) * self.train_val_test_split[0]),
                    int(len(self.dataset) * self.train_val_test_split[1]),
                    int(len(self.dataset) * self.train_val_test_split[2]),
                ],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class ClassificationDataset1(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        input_length: int = 1,
        output_length: int = 1,
        stride: int = 1,
    ):
        """
        each row of the data is a single step of the time series. The LAST column is the label and the rest are the features.
        input_length: used to determine the input sequence length.
        output_length: used to determine the output sequence length. If output_length < input_length, the output is the labels of the last output_length steps in order.
        """
        self.samples = []
        self.num_classes = len(data.iloc[:, -1].unique())

        for i in tqdm(range(len(data) - input_length - output_length + 1, stride)):
            sample_data = data.iloc[i : i + input_length + output_length]
            x = torch.tensor(sample_data.iloc[:, :-1].values, dtype=torch.float32)
            y = torch.tensor(
                sample_data.iloc[-output_length:, -1].values, dtype=torch.long
            )
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ClassificationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    

if __name__ == "__main__":
    datamodule = ClassificationDataModule("", 1, 1, 1)
    pass
