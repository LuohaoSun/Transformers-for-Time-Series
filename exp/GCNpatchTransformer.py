import sys

sys.path.append(".")

import tensorboard
from lightning.pytorch import seed_everything

seed_everything(42)
from exp.exp_config import *
from exp.prepare_data import datamodule
from transformers_for_time_series.models.GCNpatchTransformer import GCNPatchTransformer
from transformers_for_time_series.trainers.forecasting_trainer import ForecastingTrainer

model = GCNPatchTransformer(
    adj=DATA_ADJ_PATH,
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
    lr=TRAINER_LR,
    early_stopping_patience=TRAINER_EARLY_STOPPING_PATIENCE,
    weight_decay=TRAINER_WEIGHT_DECAY,
    gradient_clip_algorithm=TRAINER_GRADIENT_CLIP_ALGORITHM,
    gradient_clip_val=TRAINER_GRADIENT_CLIP_VAL,
    log_save_name=f"GCNPatchTransformer-{DATA_INPUT_LENGTH}-{DATA_OUTPUT_LENGTH}",
)
trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
