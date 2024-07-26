import os
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
from ...utils.data.sliding_window_dataset import SlidingWindowDataset


class AutoEncodingDataModule(L.LightningDataModule):
    __doc__ = f"""
    from_datasets: load a list of datasets and convert it to a LightningDataModule. See:
    > {L.LightningDataModule.from_datasets.__doc__}
    """

    def __init__(
        self,
        data_path: str,
        windows_size: int,
        stride: int,
        ignore_last_cols: int,
        batch_size: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 1,
    ) -> None:
        """
        Initializes the AutoencodingDataModule.

        Args:
            path (str): The path to the data. Could be a .csv file or a directory containing multiple .csv files.
            windows_size (int): The size of the sliding window.
            stride (int): The stride of the sliding window.
            ignore_last_cols (int): The number of columns to ignore from the end of each sample.
            batch_size (int): The batch size for training and validation dataloaders.
            train_val_test_split (Tuple[float, float, float], optional): The split ratio for train, validation, and test sets. Defaults to (0.7, 0.2, 0.1).
            num_workers (int, optional): The number of workers for data loading. Defaults to 1.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.windows_size = windows_size
        self.stride = stride
        self.ignore_last_cols = ignore_last_cols
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.dataset = AutoEncodingDataset(
            self.data_path, self.windows_size, self.stride, self.ignore_last_cols
        )

    def setup(self, stage: str | None = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        assert stage in (None, "fit", "validate", "test", "predict")
        if not hasattr(self, "train_set"):
            self.train_set, self.val_set, self.test_set = random_split(
                self.dataset, self.train_val_test_split
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class AutoEncodingDataset(Dataset):
    def __init__(
        self, path: str, windows_size: int, stride: int, ignore_last_cols: int
    ) -> None:
        super().__init__()
        if os.path.isfile(path):
            self.sliding_window_dataset = SlidingWindowDataset.from_csv(
                path, windows_size, stride
            )
        elif os.path.isdir(path):
            self.sliding_window_dataset = SlidingWindowDataset.from_directory(
                path, windows_size, stride
            )
        else:
            raise ValueError(f"Invalid path: {path}")

        self.ignore_last_cols = ignore_last_cols

    def __len__(self):
        return len(self.sliding_window_dataset)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        sequence = self.sliding_window_dataset[index]
        if self.ignore_last_cols > 0:
            sequence = sequence[:, : -self.ignore_last_cols]
        return sequence, sequence
