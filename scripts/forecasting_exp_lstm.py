"""
使用LSTM骨干进行地铁客流量预测
"""

import sys


def main():

    sys.path.append(".")
    sys.path.append("..")

    # 1. 读取数据
    INPUT_LENGTH = 48  # 输入序列长度
    OUTPUT_LENGTH = 48  # 输出序列长度
    NUM_STATIONS = 276  # 地铁站点数（特征数）

    from data.BJinflow import BJInflowDataModule

    datamodule = BJInflowDataModule(
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        batch_size=64,
        train_val_test_split=(0.7, 0.2, 0.1),
        normalization="zscore",
    )

    # 2. 选择骨干模型
    from t4ts.backbones import LSTMBackbone

    DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"

    lstm_backbone = LSTMBackbone(
        in_features=NUM_STATIONS,
        hidden_features=64,
        auto_recurrsive_steps=OUTPUT_LENGTH,
        num_layers=2,
    )

    # 3. 选择任务框架
    from t4ts.frameworks import ForecastingFramework

    framework = ForecastingFramework(
        backbone=lstm_backbone,
        backbone_out_features=64,
        out_seq_len=OUTPUT_LENGTH,
        out_features=NUM_STATIONS,
    )

    # 4. 训练模型
    framework.fit(
        datamodule=datamodule,
        log_every_n_steps=10,
        max_epochs=2000,
        early_stopping_patience=100,
        accelerator=DEVICE,
        compile_model=False,
    )
    framework.test(datamodule=datamodule)


if __name__ == "__main__":
    main()
