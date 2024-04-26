from typing import Tuple
import lightning as L
import pandas as pd

from typing import IO, Any, Dict, Iterable, Optional, Union, cast

from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset


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
        output_length: int,
        batch_size: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.file_path = csv_file_path
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed

        return

    def setup(self, stage: str | None = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        assert sum(self.train_val_test_split) - 1 < 1e-6
        assert stage in (None, "fit", "validate", "test", "predict")
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


if __name__ == "__main__":
    datamodule = AutoEncodingDataModule("", 1, 1, 1)
    pass
