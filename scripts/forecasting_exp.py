def main():

    import sys
    import os

    sys.path.append(".")
    sys.path.append("..")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
        batch_size=32,
        train_val_test_split=(0.7, 0.2, 0.1),
    )

    # 2. 选择骨干模型
    from pretrained import Chronos

    DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"

    chronos_backbone = Chronos(
        size="tiny",
        task="forecasting",
        out_seq_len=OUTPUT_LENGTH,
        device_map=DEVICE,
    )

    # 3. 选择任务框架
    from frameworks import ForecastingFramework

    chronos_framework = ForecastingFramework(
        backbone=chronos_backbone,
        custom_head=nn.Identity(),  # 因为要使用backbone的0-shot能力，因此不需要额外的头部
        # 以下参数其实没用：
        backbone_out_features=NUM_STATIONS,
        out_seq_len=OUTPUT_LENGTH,
        out_features=NUM_STATIONS,
    )

    # 4. 训练模型
    chronos_framework.fit(
        datamodule=datamodule, max_epochs=0, accelerator=DEVICE
    )  # 仅用于初始化Trainer
    chronos_framework.test(datamodule=datamodule)


if __name__ == "__main__":
    main()
