# Description: An example of how to use the AutoEncodingFramework
import sys

sys.path.append("./")
from src.utils.visualization import SeriesPlotter


def main():

    # Step 1. Choose a dataset
    from data.bearing_fault_prediction.datamodule import (
        FaultPredictionDataModule,
    )

    datamodule = FaultPredictionDataModule(batch_size=40)

    # Step 2. Choose a backbone
    from src.backbones import PatchTSTBackbone

    backbone = PatchTSTBackbone(
        in_features=1,
        d_model=64,
        patch_size=16,
        patch_stride=16,
        num_layers=2,
    )

    # Step 3. Choose a framework
    from src.frameworks import AutoEncodingFramework

    framework = AutoEncodingFramework(
        backbone=backbone,
        backbone_out_seq_len=4096 // 16,
        backbone_out_features=64,
        out_seq_len=4096,
        out_features=1,
        mask_ratio=0.1,  # here we implement a 10%-ratio masked-autoencoding task
        mask_length=16,  # the length of the masked tokens
    )

    # Step 4. fit and test
    framework.fit(datamodule, max_epochs=10)
    framework.test(datamodule)

    # Step 5. model predicts
    sample_data = datamodule.test_dataset[0][0].unsqueeze(0)

    encoded = framework.encode(sample_data)  # same as model.backbone(sample_data)
    decoded = framework.decode(encoded)  # same as model.head(encoded)
    SeriesPlotter.plot_and_show(
        {"origin": sample_data, "decoded": decoded}, figsize=(30, 2)
    )


if __name__ == "__main__":
    main()
