from typing import Any, Dict, Optional, Tuple, Union, Callable
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torch.functional as F
import numpy as np
import os
import torch
import tqdm
from rich import print
from rich.progress import Progress


class FaultPredictionDataModule(LightningDataModule):
    """DataModule for fault prediction task."""

    def __init__(
        self,
        train_data_dir: str = "data/bearing_fault_prediction/raw/",
        train_val_test_split: Tuple[int, int, int] = (2800, 400, 800),
        batch_size: int = 40,
        num_workers: int = 7,
        transforms: Union[None, Callable[[torch.Tensor], torch.Tensor]] = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data_dir = train_data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms if transforms else sample_transform()
        self.pin_memory = pin_memory

    @property
    def num_classes(self) -> int:
        return 4

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.batch_size, 4096, 1)

    def prepare_data(self) -> None:

        if not os.path.exists(self.train_data_dir):
            raise FileNotFoundError(
                f"Training data directory { self.train_data_dir} does not exist."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size

        if not hasattr(self, "data_train"):
            self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_datasets()

    def prepare_datasets(self):
        print("Preparing datasets...")
        # read 4 types of samples:
        sample_i_datasets = []
        with Progress() as progress:
            task = progress.add_task("[green]Reading samples...", total=4)
            for i in range(4):
                sample_i_dir = self.train_data_dir + str(i) + "/"
                sample_i_files = os.listdir(sample_i_dir)
                progress.update(task, advance=1, description=f"Reading samples {i}")
                sample_i = [
                    np.loadtxt(sample_i_dir + "/" + file) for file in sample_i_files
                ]
                sample_i_dataset = List2Dataset(sample_i, i, self.transforms)
                sample_i_datasets.append(sample_i_dataset)
        train_dataset = ConcatDataset(sample_i_datasets)
        # random split train, val and test dataset:
        train_size, val_size, test_size = self.train_val_test_split
        train_dataset, val_dataset, test_dataset = random_split(
            train_dataset, [train_size, val_size, test_size]
        )
        # TODO: make 4 types of samples balanced in 3 datasets
        print("datasets prepared.")
        return train_dataset, val_dataset, test_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class List2Dataset(Dataset):
    def __init__(
        self,
        sample_list: list[np.ndarray] | list[Tensor],
        label: int | str | Tensor,
        transform: (
            Callable[[Tensor], Tensor] | Callable[[np.ndarray], np.ndarray] | None
        ) = None,
    ) -> None:
        super().__init__()
        if isinstance(label, str):
            label = torch.tensor(int(label), dtype=torch.long)
        elif isinstance(label, int):
            label = torch.tensor(label, dtype=torch.long)
        if isinstance(sample_list[0], np.ndarray):
            sample_list: list[Tensor] = [
                torch.tensor(x, dtype=torch.float32) for x in sample_list
            ]
        if len(sample_list[0].shape) < 3:
            sample_list = [x.unsqueeze(dim=-1) for x in sample_list]
        if transform:
            sample_list = [transform(x) for x in sample_list]  # type: ignore
        self.samples = [(x, label) for x in sample_list]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class sample_transform:
    def __init__(
        self, transform: Union[str, Callable[[Tensor], Tensor]] = "z_score"
    ) -> None:
        print(f"{transform} sample transform initialized.")
        if isinstance(transform, str):

            if transform == "p2":
                self.transform = self.p2
            elif transform == "z_score":
                self.transform = self.z_score
            else:
                raise ValueError(
                    f'Invalid transform name: {transform}. Use "p2" or "z_score" instead.'
                )
        elif callable(transform):
            self.transform = transform
        else:
            raise ValueError("Invalid transform type. Use str or callable instead.")

    @property
    def z_score(self) -> Callable[[Tensor], Tensor]:
        return lambda sample: (sample - sample.mean()) / sample.std()

    @property
    def p2(self) -> Callable[[Tensor], Tensor]:
        return lambda sample: torch.nn.functional.normalize(sample, p=2, dim=-1)

    def __call__(self, sample: Tensor) -> Tensor:
        """
        input: sample: torch.Tensor, shape=(seq_len)
        output: sample: torch.Tensor, shape=(seq_len)
        """
        return self.transform(sample)

    def __repr__(self) -> str:
        raise NotImplementedError("sample transform not implemented yet.")
