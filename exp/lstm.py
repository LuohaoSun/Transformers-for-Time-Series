import sys

sys.path.append(".")

import time

import tensorboard
from lightning.pytorch import seed_everything
from src.models.lstm import LSTMModel
from src.trainers.forecasting_trainer import ForecastingTrainer
from exp.prepare_data import datamodule
from exp.exp_config import *

seed_everything(42)

model = LSTMModel(
    in_features=DATA_FEATURES,
    hidden_features=MODEL_D_MODEL,
    out_features=DATA_FEATURES,
    num_layers=MODEL_NUM_LAYERS,
    bidirectional=True,
)

trainer = ForecastingTrainer(
    max_epochs=TRAINER_MAX_EPOCHS,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    lr=TRAINER_LR,
    weight_decay=TRAINER_WEIGHT_DECAY,
    gradient_clip_algorithm=TRAINER_GRADIENT_CLIP_ALGORITHM,
    gradient_clip_val=TRAINER_GRADIENT_CLIP_VAL,
    version=f"LSTM-{DATA_INPUT_LENGTH}-{DATA_OUTPUT_LENGTH}-{int(time.time())}",
)
trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
