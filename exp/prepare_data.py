from lightning.pytorch import seed_everything

from exp.exp_config import *
from src.data.PeMS_Bay import get_forecasting_datamodule, get_adj_matrix

seed_everything(42)

datamodule = get_forecasting_datamodule(
    csv_file_path=DATA_PATH,
    stride=DATA_STRIDE,
    input_length=DATA_INPUT_LENGTH,
    output_length=DATA_OUTPUT_LENGTH,
    batch_size=DATA_BATCH_SIZE,
    num_workers=DATA_NUM_WORKERS,
    train_val_test_split=DATA_TRAIN_VAL_TEST_SPLIT,
    normalization=DATA_NORMALIZATION,
)

adj = get_adj_matrix()
