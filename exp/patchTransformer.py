import sys
import torch.nn.functional as F

sys.path.append(".")

import tensorboard
from lightning.pytorch import seed_everything

seed_everything(42)
from exp.exp_config import *
from exp.prepare_data import datamodule
from transformers_for_time_series.models.patchTransformer import PatchTransformer
from transformers_for_time_series.trainers.forecasting_trainer import ForecastingTrainer

model = PatchTransformer(
    in_features=DATA_FEATURES,
    patch_size=MODEL_PATCH_SIZE,
    d_model=MODEL_D_MODEL,
    out_features=DATA_FEATURES,
    num_layers=MODEL_NUM_LAYERS,
    nhead=MODEL_NHEAD,
    dropout=MODEL_DROPOUT,
    norm_first=True,
)


trainer = ForecastingTrainer(
    max_epochs=TRAINER_MAX_EPOCHS,
    precision=TRAINER_PRECISION,
    lr=TRAINER_LR,
    weight_decay=TRAINER_WEIGHT_DECAY,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    gradient_clip_algorithm=TRAINER_GRADIENT_CLIP_ALGORITHM,
    gradient_clip_val=TRAINER_GRADIENT_CLIP_VAL,
    log_save_name=f"PatchTransformer-{DATA_INPUT_LENGTH}-{DATA_OUTPUT_LENGTH}",
)
trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
