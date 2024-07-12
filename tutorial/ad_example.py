# Description: An example of how to use the AutoEncodingFramework
import sys

sys.path.append("./")
from utils.visualization import SeriesPlotter


def main():

    # Step 1. Choose a dataset
    from data.bearing_fault_prediction.datamodule import (
        FaultPredictionDataModule,
    )

    datamodule = FaultPredictionDataModule(batch_size=40)

    # Step 2. Choose a backbone
    from backbones import PatchTSTBackbone

    backbone = PatchTSTBackbone(
        in_features=1,
        d_model=64,
        patch_size=16,
        patch_stride=16,
        num_layers=2,
    )

    # Step 3. Choose a framework
    from frameworks import AnomalyDetectionFramework

    framework = AnomalyDetectionFramework(
        backbone=backbone,
        backbone_out_seq_len=4096 // 16,
        backbone_out_features=64,
        out_seq_len=4096,
        out_features=1,
        threshold=0.99,
        detection_level="step",
        vi_every_n_epochs=10,
    )

    # Step 4. fit and test
    framework.fit(datamodule, max_epochs=100)
    framework.test(datamodule)



if __name__ == "__main__":
    main()
