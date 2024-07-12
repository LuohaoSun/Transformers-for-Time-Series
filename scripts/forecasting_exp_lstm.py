def main():

    import sys
    import os

    sys.path.append(".")
    sys.path.append("..")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 使用metroBJ数据集测试各个模型的序列预测性能
    import torch
    import torch.nn as nn

    # 1. 读取数据
    INPUT_LENGTH = 48  # 输入序列长度
    OUTPUT_LENGTH = 12  # 输出序列长度
    NUM_STATIONS = 276  # 地铁站点数（特征数）

    from data.BJinflow import BJInflowDataModule

    datamodule = BJInflowDataModule(
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        batch_size=64,
        train_val_test_split=(
            0.7,
            0.27,
            0.03,
        ),  # chronos将(b, l, d) 转换为(bd, l)来处理，非常消耗资源，只能把batch_size, testset尽量调小。
    )

    # 2. 选择骨干模型
    from backbones import LSTMBackbone

    DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"

    lstm_backbone = LSTMBackbone(
        in_features=NUM_STATIONS,
        hidden_features=64,
        auto_recurrsive_steps=OUTPUT_LENGTH,
        num_layers=2,
    )

    # 3. 选择任务框架
    from frameworks import ForecastingFramework

    chronos_framework = ForecastingFramework(
        backbone=lstm_backbone,
        backbone_out_features=64,
        out_seq_len=OUTPUT_LENGTH,
        out_features=NUM_STATIONS,
    )

    # 4. 训练模型
    chronos_framework.fit(
        datamodule=datamodule, max_epochs=100, accelerator=DEVICE
    )  # 仅用于初始化Trainer
    chronos_framework.test(datamodule=datamodule)


if __name__ == "__main__":
    main()
