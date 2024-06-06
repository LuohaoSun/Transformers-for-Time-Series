import torch
from torch import nn
import sys

sys.path.append("./")


def main():
    # 第1步：根据数据集创建datamodule。此处你需要指定数据集的参数，例如batch_size、子集划分等。
    from data.bearing_fault_prediction import FaultPredictionDataModule

    datamodule = FaultPredictionDataModule(
        train_val_test_split=(2800, 400, 800), batch_size=40, num_workers=4
    )

    # 第2步：根据喜好选择骨干模型。此处你需要指定模型超参数，例如d_model、num_layers等。
    from backbones import PatchTSTBackbone

    backbone = PatchTSTBackbone(
        in_features=1,
        d_model=64,
        patch_size=16,
        patch_stride=16,
        num_layers=2,
    )

    # 第3步：根据任务选择framework。此处你需要指定任务参数，例如out_seq_len、num_classes等。
    from frameworks import ClassificationFramework

    framework = ClassificationFramework(
        backbone=backbone,
        backbone_out_features=64,
        out_seq_len=1,
        num_classes=4,
    )

    # 第4步：训练和测试。此处你需要指定优化算法的参数，主要是学习率，训练代数。对于更复杂的情况，还包括优化算法、损失函数和学习率调度器的选择。
    framework.fit(datamodule)
    framework.test(datamodule)


if __name__ == "__main__":
    main()
