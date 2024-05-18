# Description: An example of how to use the AutoEncodingFramework
import sys

sys.path.append("./")
from utils.visualization import SeriesPlotter


def main():
    import torch

    # Step 1. Choose a dataset (L.LightningDataModule)
    from data.bearing_fault_prediction.raw.fault_prediction_datamodule import (
        FaultPredictionDataModule,
    )

    datamodule = FaultPredictionDataModule(batch_size=40)

    # Step 2. Choose a model (AutoEncodingFramework from models.autoencoding_models)
    from model.autoencoding_models.autoencoding_models import PatchTSTAutoEncodingModel

    model = PatchTSTAutoEncodingModel(
        in_features=1,
        d_model=32,
        patch_size=16,
        patch_stride=16,
        num_layers=2,
        dropout=0,
        nhead=2,
        activation="relu",
        additional_tokens_at_last=0,
        norm_first=True,
        # logging params
        every_n_epochs=10,
        figsize=(30, 5),
        dpi=300,
        # training params
        mask_ratio=0,
        lr=1e-5,
        max_epochs=11,
        max_steps=-1,
    )

    # Step 3. model trains itself with the datamodule
    model.fit(datamodule)
    model.test(datamodule)

    # Step 4. model predicts
    sample_data = torch.randn(4, 4096, 1)

    encoded = model.encode(sample_data)  # same as model.backbone(sample_data)
    decoded = model.decode(encoded)  # same as model.head(encoded)
    SeriesPlotter.plot_and_show([sample_data, decoded], figsize=(30, 2))


if __name__ == "__main__":
    main()
