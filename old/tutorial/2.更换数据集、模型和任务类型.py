import sys

sys.path.append(".")
sys.path.append("..")


def main():
    """
    **Forecasting Example**

    作为一个简单框架使用示例，我们在北京地铁进站客流数据集上使用一个简单的LSTM骨干进行序列预测任务。
    这个任务中，数据集、模型、任务类型均与分类任务有所不同，用于展示如何在不同情况下使用框架。

    北京地铁进站客流数据集中，每个时间步包含276个站点的进站客流数据，我们的任务是根据历史客流数据预测未来客流数据。
    """

    INPUT_LENGTH = 24   # 输入序列长度
    OUTPUT_LENGTH = 24  # 输出序列长度
    NUM_STATIONS = 276  # 地铁站点数（特征数）

    # 第1步：根据数据集创建datamodule。此处指定batch_size为32，训练集、验证集和测试集的划分比例为7:1:2。
    from data.BJinflow import BJInflowDataModule

    datamodule = BJInflowDataModule(
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        batch_size=32,
        train_val_test_split=(0.7, 0.1, 0.2),
    )

    # 第2步：根据喜好选择骨干模型。此处使用LSTM骨干，指定输入特征数为276，隐藏特征数为128，自回归步数为6，层数为2。
    from t4ts.models import LSTMBackbone

    backbone = LSTMBackbone(
        in_features=NUM_STATIONS,
        hidden_features=128,
        auto_recurrsive_steps=OUTPUT_LENGTH,
        num_layers=2,
    )

    # 第3步：根据任务选择framework。此处你需要指定任务参数，例如out_seq_len、num_classes等。
    from t4ts.frameworks import ForecastingFramework

    framework = ForecastingFramework(
        backbone=backbone,
        backbone_out_features=128,
        out_seq_len=OUTPUT_LENGTH,
        out_features=NUM_STATIONS,
    )

    # 第4步：训练和测试。此处你需要指定优化算法的参数，主要是学习率，训练代数。对于更复杂的情况，还包括优化算法、损失函数和学习率调度器的选择。
    framework.fit(datamodule, max_epochs=10, lr=1e-3)
    framework.test(datamodule)

    # 查看预测曲线
    from t4ts.utils.visualization import SeriesPlotter

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    y_hat = framework.predict(x)
    SeriesPlotter.plot_and_show({"y": y, "y_hat": y_hat})


if __name__ == "__main__":
    main()
