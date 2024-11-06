import sys

sys.path.append(".")
sys.path.append("src/time_series_lib")

import time

import tensorboard
import torch.nn as nn
from lightning.pytorch import seed_everything
from src.time_series_lib.models.PatchTST import Model
from src.trainers.forecasting_trainer import ForecastingTrainer
from exp.prepare_data import datamodule
from exp.exp_config import *

seed_everything(42)


class Configs:
    task_name = "long_term_forecast"
    seq_len = DATA_INPUT_LENGTH
    pred_len = DATA_OUTPUT_LENGTH
    d_model = MODEL_D_MODEL
    dropout = MODEL_DROPOUT
    factor = 1
    n_heads = MODEL_NHEAD
    d_ff = 4 * MODEL_D_MODEL
    activation = MODEL_ACTIVATION
    e_layers = MODEL_NUM_LAYERS
    enc_in = DATA_FEATURES


class TSLModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(TSLModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, None, None, None)


model = Model(
    configs=Configs,
    patch_len=MODEL_PATCH_SIZE,
    stride=MODEL_PATCH_STRIDE,
)
model = TSLModelWrapper(model)

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
trainer = ForecastingTrainer(
    max_epochs=TRAINER_MAX_EPOCHS,
    lr=TRAINER_LR,
    weight_decay=TRAINER_WEIGHT_DECAY,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    gradient_clip_algorithm=TRAINER_GRADIENT_CLIP_ALGORITHM,
    gradient_clip_val=TRAINER_GRADIENT_CLIP_VAL,
    version=f"PatchTST-{DATA_INPUT_LENGTH}-{DATA_OUTPUT_LENGTH}-{current_time}",
)

trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
