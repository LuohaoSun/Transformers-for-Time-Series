import sys

sys.path.append(".")
sys.path.append("src/time_series_lib")


import tensorboard
from lightning.pytorch import seed_everything

seed_everything(42)

from exp.exp_config import *
from exp.prepare_data import datamodule
from transformers_for_time_series.models.itransformer import iTransformer
from transformers_for_time_series.trainers.forecasting_trainer import ForecastingTrainer

model = iTransformer(
    seq_len=DATA_INPUT_LENGTH,
    pred_len=DATA_OUTPUT_LENGTH,
    d_model=MODEL_D_MODEL,
    dropout=MODEL_DROPOUT,
    n_heads=MODEL_NHEAD,
    d_ff=MODEL_D_MODEL * 4,
    activation=MODEL_ACTIVATION,
    e_layers=MODEL_NUM_LAYERS,
    enc_in=DATA_FEATURES,
)

trainer = ForecastingTrainer(
    max_epochs=TRAINER_MAX_EPOCHS,
    lr=TRAINER_LR,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    gradient_clip_algorithm=TRAINER_GRADIENT_CLIP_ALGORITHM,
    gradient_clip_val=TRAINER_GRADIENT_CLIP_VAL,
    weight_decay=TRAINER_WEIGHT_DECAY,
    log_save_name=f"iTransformer-{DATA_INPUT_LENGTH}-{DATA_OUTPUT_LENGTH}",
)

trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
