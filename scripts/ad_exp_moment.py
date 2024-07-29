# Description: An example of how to use the AutoEncodingFramework

import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = ".cache/huggingface"
sys.path.append("./")
import torch.nn as nn
from src.utils.visualization import SeriesPlotter


def main():
    # Step 1. Choose a dataset
    from src.utils.data import AutoEncodingDataModule

    datamodule = AutoEncodingDataModule(
        data_path="data/bearing_fault_prediction/0",
        windows_size=512,
        stride=512,
        ignore_last_cols=0,
        batch_size=8,
        train_val_test_split=(0.7, 0.1, 0.2),
        num_workers=4,
    )

    # Step 2. Choose a backbone
    from src.pretrained.moment import MOMENT

    backbone = MOMENT(task="reconstruction")

    # Step 3. Choose a framework
    from src.frameworks import AnomalyDetectionFramework

    framework = AnomalyDetectionFramework(
        backbone=backbone,
        backbone_out_seq_len=512,
        backbone_out_features=1,
        out_seq_len=512,
        out_features=1,
        threshold=0.9,
        vi_every_n_epochs=10,
        custom_neck=nn.Identity(),
        custom_head=nn.Identity(),
    )

    # Step 4. fit and test
    framework.fit(datamodule, max_epochs=0, accelerator="mps")
    framework.test(datamodule)


if __name__ == "__main__":
    main()
