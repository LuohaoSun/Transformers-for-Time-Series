import sys

sys.path.append(".")

import time

import tensorboard
from lightning.pytorch import seed_everything
from src.models.lstm import LSTMModel
from src.trainers.forecasting_trainer import ForecastingTrainer
from src.utils.data import ForecastingDataModule
from exp.exp_config import *

seed_everything(42)

datamodule = ForecastingDataModule(
    csv_file_path=DATA_PATH,
    stride=DATA_STRIDE,
    input_length=DATA_INPUT_LENGTH,
    output_length=DATA_OUTPUT_LENGTH,
    batch_size=DATA_BATCH_SIZE,
    num_workers=DATA_NUM_WORKERS,
    train_val_test_split=DATA_TRAIN_VAL_TEST_SPLIT,
    normalization=DATA_NORMALIZATION,
)

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
