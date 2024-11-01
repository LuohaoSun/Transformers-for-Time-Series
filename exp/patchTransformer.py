import sys

sys.path.append(".")

import time

import tensorboard

from src.t4ts.models.patchTransformer import PatchTransformer
from src.t4ts.trainers.forecasting_trainer import ForecastingTrainer
from src.t4ts.utils.data import ForecastingDataModule

INPUT_LENGTH = 256  # 输入序列长度
OUTPUT_LENGTH = 256  # 输出序列长度
NUM_STATIONS = 276  # 地铁站点数（特征数）

datamodule = ForecastingDataModule(
    csv_file_path="data/BJinflow/in_10min_trans.csv",
    stride=1,
    input_length=INPUT_LENGTH,
    output_length=OUTPUT_LENGTH,
    batch_size=64,
    num_workers=7,
    train_val_test_split=(0.6, 0.2, 0.2),
    normalization="zscore",
)

model = PatchTransformer(
    in_features=NUM_STATIONS,
    patch_size=4,
    d_model=512,
    out_features=NUM_STATIONS,
    num_layers=2,
    nhead=2,
    dropout=0.1,
    norm_first=True,
)

current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
trainer = ForecastingTrainer(
    max_epochs=1000,
    early_stopping_patience=50,
    version=f"PatchTransformer-{INPUT_LENGTH}-{OUTPUT_LENGTH}-{current_time}",
)
trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
