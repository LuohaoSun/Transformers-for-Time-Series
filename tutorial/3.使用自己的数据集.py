import sys

sys.path.append(".")
sys.path.append("..")


def main():
    
    # 在【快速上手】中，我们已经介绍了轴承数据集：
    from data.bearing_fault_prediction import FaultPredictionDataModule

    datamodule1 = FaultPredictionDataModule(
        train_val_test_split=(2800, 400, 800), batch_size=40
    )

    # 在【更换数据集、模型和任务类别】中，我们介绍了北京地铁进站客流数据集：
    from data.BJinflow import BJInflowDataModule

    data_module2 = BJInflowDataModule(
        input_length=24,
        output_length=24,
        batch_size=32,
        train_val_test_split=(0.7, 0.1, 0.2),
    )

    # 上述数据集均为可直接使用的Datamodule，但在实际应用中，我们自己的数据集是由.csv文件或其他格式的原始数据组成的。
    # 为了方便从新的原始数据集创建新的datamodule，本项目提供了一些工具，你只需要按照要求的格式提供原始数据集，即可创建新的datamodule。
    # 作为一个简单的示例，我们使用PEMS-BAY数据集的.csv格式原始数据，创建一个用于序列预测的Datamodule实例：
    from src.utils.data import ForecastingDataModule

    data_path = "data/pems_bay/pems_bay.csv"
    datamodule3 = ForecastingDataModule(
        csv_file_path=data_path,  # 关于所需的原始数据格式，请参考ForecastingDataModule的说明。
        input_length=12,
        output_length=12,
        batch_size=32,
        train_val_test_split=(0.7, 0.1, 0.2),
        stride=1,
    )

    # 然后你可以使用这个datamodule进行训练和测试，如：
    # framework.fit(datamodule3, max_epochs=10, lr=1e-3)
    # framework.test(datamodule3)

    # 此外，framework同时兼容Dataloader，你可以使用自己的dataloader替代datamodule，如：
    # dataloader1, dataloader2 = ...
    # framework.fit(train_dataloaders=dataloader1, val_dataloaders=dataloader2)
    # framework.test(dataloader=dataloader)
    
    # TODO: add training and testing code here
