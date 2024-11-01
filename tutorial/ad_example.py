# Description: An example of how to use the AutoEncodingFramework
import sys

sys.path.append("./")
from t4ts.utils.visualization import SeriesPlotter


def main():

    # Step 1. Choose a dataset
    from t4ts.utils.data import AutoEncodingDataModule
    datamodule=AutoEncodingDataModule(
        data_path="data/bearing_fault_prediction/0",
        windows_size=128,
        stride=128,
        ignore_last_cols=0,
        batch_size=32,
        train_val_test_split=(0.7, 0.1, 0.2),
        num_workers=4,
    )

    # Step 2. Choose a backbone
    from t4ts.models import PatchTSTBackbone

    backbone = PatchTSTBackbone(
        in_features=1,
        d_model=64,
        patch_size=8,
        patch_stride=8,
        num_layers=4,
    )

    # Step 3. Choose a framework
    from t4ts.frameworks import AnomalyDetectionFramework

    framework = AnomalyDetectionFramework(
        backbone=backbone,
        backbone_out_seq_len=128//8,
        backbone_out_features=64,
        out_seq_len=128,
        out_features=1,
        threshold=0.9,
        detection_level="step",
        vi_every_n_epochs=10,
    )

    # Step 4. fit and test
    framework.fit(datamodule, max_epochs=100)
    framework.test(datamodule)



if __name__ == "__main__":
    main()
