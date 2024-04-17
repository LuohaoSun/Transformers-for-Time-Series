'''
a .py version of the notebook example.ipynb
'''
from data.bearing_fault_prediction.raw.fault_prediction_datamodule import FaultPredictionDataModule
from Modules.models.classification_models import *
import torch

if __name__ == '__main__':

    data_module = FaultPredictionDataModule()
    model = SimpleConv1dClassificationModel(
        in_features=1,
        num_classes=4,
        hidden_features=64,
        kernel_size=16,
        stride=8,
        padding=4,
        pool_size=64,
        activation='relu',

        lr=1e-3,
        max_epochs=50,
    )

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

    model.fit(data_module)

    model.test(data_module)

    y = model(torch.rand(32, 4096, 1))
    print(torch.softmax(y, dim=-1))
