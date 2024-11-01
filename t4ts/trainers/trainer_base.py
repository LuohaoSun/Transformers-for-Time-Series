# Author: Sun LuoHao
# All rights reserved
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, Iterable, Mapping, Tuple

import lightning as L
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


class LitModuleWrapper(L.LightningModule):

    def __init__(
        self,
        loss_fn: Callable[
            [Annotated[Tensor, "output"], Annotated[Tensor, "target"]], Tensor
        ],
        optimizer: Callable[[Iterable[Parameter]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def wrap_model(self, model: nn.Module) -> L.LightningModule:
        self.model = model
        return self

    def model_step(
        self, batch: Tuple[Annotated[Tensor, "input"], Annotated[Tensor, "target"]]
    ) -> Mapping[str, Tensor]:
        input, target = batch
        output = self.model(input)
        loss = self.loss_fn(output, target)
        return {"loss": loss, "input": input, "target": target, "output": output}

    def training_step(
        self,
        batch: Tuple[Annotated[Tensor, "input"], Annotated[Tensor, "target"]],
        batch_idx: int,
    ) -> Mapping[str, Tensor]:
        return self.model_step(batch)

    def validation_step(
        self,
        batch: Tuple[Annotated[Tensor, "input"], Annotated[Tensor, "target"]],
        batch_idx: int,
    ) -> Mapping[str, Tensor]:
        return self.model_step(batch)

    def test_step(
        self,
        batch: Tuple[Annotated[Tensor, "input"], Annotated[Tensor, "target"]],
        batch_idx: int,
    ) -> Mapping[str, Tensor]:
        return self.model_step(batch)

    def configure_optimizers(
        self,
    ) -> Mapping[str, Optimizer | LRScheduler]:
        optimizer = self.optimizer(self.model.parameters())
        return {"optimizer": optimizer}


class TrainerBase(ABC):
    """
    NOTE: a batch must be a tuple of (input: Tensor, target: Tensor), where the
    input will be directly passed to the model, and the target will be used to
    compute the loss.
    """

    trainer: L.Trainer
    lit_model_wrapper: LitModuleWrapper

    def __new__(cls, *args: Any, **kwargs: Any) -> "TrainerBase":
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        self.lit_model_wrapper = LitModuleWrapper(
            loss_fn=self.configure_loss_fn(),
            optimizer=self.configure_optimizer(),
            lr_scheduler=self.configure_lr_scheduler(),
        )
        self.trainer = self.configure_trainer()

    @abstractmethod
    def configure_loss_fn(
        self,
    ) -> Callable[[Annotated[Tensor, "output"], Annotated[Tensor, "target"]], Tensor]:
        """
        NOTE: (output, target) -> loss
        """
        ...

    @abstractmethod
    def configure_optimizer(self) -> Callable[[Iterable[Parameter]], Optimizer]: ...

    def configure_lr_scheduler(self) -> Callable[[Optimizer], LRScheduler] | None:
        return None

    @abstractmethod
    def configure_trainer(self) -> L.Trainer: ...

    def fit(
        self,
        model: nn.Module,
        train_dataloaders: DataLoader | L.LightningDataModule | None = None,
        val_dataloaders: DataLoader | L.LightningDataModule | None = None,
        datamodule: L.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        lit_model = self.lit_model_wrapper.wrap_model(model)
        self.trainer.fit(
            lit_model,
            train_dataloaders,
            val_dataloaders,
            datamodule,
            ckpt_path,
        )

    def test(
        self,
        model: nn.Module,
        dataloaders: DataLoader | L.LightningDataModule | None = None,
        ckpt_path: str = "best",
        verbose: bool = True,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:

        lit_model = self.lit_model_wrapper.wrap_model(model)
        self.trainer.test(lit_model, dataloaders, ckpt_path, verbose, datamodule)
