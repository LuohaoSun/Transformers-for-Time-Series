"""
a .py version of the notebook example.ipynb
"""


def main():
    import torch

    # Step 1. Choose a dataset (L.LightningDataModule)
    from data.bearing_fault_prediction.raw.fault_prediction_datamodule import (
        FaultPredictionDataModule,
    )

    data_module = FaultPredictionDataModule()

    # Step 2. Choose a model (ClassificationFramework from models.classification_models)
    from model.classification_models.classification_models import SimpleConv1dClassificationModel

    model = SimpleConv1dClassificationModel(
        in_features=1,
        num_classes=4,
        hidden_features=512,
        kernel_size=16,
        stride=8,
        padding=0,
        pool_size=8,
        activation="softplus",
        lr=1e-3,
        max_epochs=10,
    )

    # for a higher resolution model, use the following model
    # from Modules.models.classification_models import PatchTSTClassificationModel
    # model = PatchTSTClassificationModel(
    #     in_features=1,
    #     d_model=64,
    #     num_classes=4,
    #     patch_size=64,
    #     patch_stride=32,
    #     dropout=0.1,
    #     nhead=2,
    #     num_layers=2,
    #     norm_first=True,
    #     activation='gelu',

    #     lr=1e-3,
    #     max_epochs=50,

    # )

    # Step 3. model trains itself with the datamodule
    model.fit(data_module)
    model.test(data_module)

    # Step 4. model predicts
    y = model(torch.rand(32, 4096, 1))
    print(torch.softmax(y, dim=-1))


if __name__ == "__main__":
    main()
