import sys

sys.path.append(".")

import tensorboard

from t4ts.backbones.lstm import LSTMModel
from t4ts.frameworks.forecasting_trainer import ForecastingTrainer
from t4ts.utils.data import ForecastingDataModule

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
    train_val_test_split=(0.7, 0.2, 0.1),
    normalization="zscore",
)

model = LSTMModel(
    in_features=NUM_STATIONS,
    hidden_features=64,
    out_features=NUM_STATIONS,
    num_layers=2,
    bidirectional=True,
)

trainer = ForecastingTrainer(
    max_epochs=1000,
    early_stopping_patience=10,
    version=f"LSTM-{INPUT_LENGTH}-{OUTPUT_LENGTH}",
)
trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
