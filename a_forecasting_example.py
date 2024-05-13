# Step 1. Choose a dataset (customized dataset from framework.forecasting.forecasting_datamodule)
from framework.forecasting.forecasting_datamodule import ForecastingDataModule

datamodule = ForecastingDataModule(
    csv_file_path="data/BJinflow/in_10min_trans.csv",
    stride=1,
    input_length=24,
    output_length=24,
    batch_size=32,
    train_val_test_split=(0.6, 0.2, 0.2),
    num_workers=7,
)

# Step 2. Choose a model (ForecastingFramework from models.forecasting_models)
from Module.models.forecasting_models import LSTM

model = LSTM(
    input_size=276,  # here we know the input size is 276
    hidden_size=64,
    num_layers=2,
    bidirectional=True,
    lr=1e-3,
    max_epochs=10,
    loss_type="mse",
)

# Step 3. model trains itself with the datamodule
model.fit(datamodule)
