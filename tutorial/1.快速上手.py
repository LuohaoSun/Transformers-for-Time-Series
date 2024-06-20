import sys

sys.path.append(".")
sys.path.append("..")


def main():
    """
    **Classification Example**

    作为一个简单框架使用示例，我们在轴承故障数据集上使用一个简单的MLP骨干进行分类任务。

    轴承故障数据集中，每个样本包含了一个轴承的一维振动信号，我们的任务是根据声音信号判断轴承的故障类别。
    - x: (batch_size, 4096, 1) 包含4096个时间步
    - y: (batch_size, 1, 1) 包含4种故障类别

    MLP骨干的结构如下：
    - 输入层：4096个神经元
    - 3个全连接层：每层包含256个神经元

    Framework包括了一个全连接层作为分类器，输出4个类别。
    """
    # 第1步：根据数据集创建datamodule。此处你需要指定数据集的参数，例如batch_size、子集划分等。
    from data.bearing_fault_prediction import FaultPredictionDataModule

    datamodule = FaultPredictionDataModule(
        train_val_test_split=(2800, 400, 800), batch_size=40
    )

    # 第2步：根据喜好选择骨干模型。此处你需要指定模型超参数，例如d_model、num_layers等。
    from backbones import MLPBackbone, ResMLPBackbone

    # backbone = MLPBackbone(
    #     in_seq_len=4096,
    #     in_features=1,
    #     hidden_features=[256, 256, 256],
    #     activation="relu",
    # )
    backbone = ResMLPBackbone(
        in_seq_len=4096,
        in_features=1,
        hidden_features=128,
        res_block_features=512,
        num_res_blocks=10,
    )

    # 第3步：根据任务选择framework。此处你需要指定任务参数，例如out_seq_len、num_classes等。
    from frameworks import ClassificationFramework

    framework = ClassificationFramework(
        backbone=backbone,
        backbone_out_features=128,
        out_seq_len=1,
        num_classes=4,
    )

    # 第4步：训练和测试。此处你需要指定优化算法的参数，主要是学习率，训练代数。对于更复杂的情况，还包括优化算法、损失函数和学习率调度器的选择。
    framework.fit(datamodule, max_epochs=10, lr=1e-3)
    framework.test(datamodule)


if __name__ == "__main__":
    main()
    __doc__ = """
    ### 简介

    **通用步骤**

    1. 选择数据集。
    2. 根据需求选择模型。
    3. 根据任务类型创建framework。
    4. 训练和测试模型。

    **以下是上述步骤的详细说明**

    1. 选择数据集。
        - 本项目的数据集都使用`LightningDataModule`加载。一个`LightningDataModule`实例包括了训练、验证、测试所需的所有数据加载器，无需重复分别实现。
        - 本项目自带了一些测试数据集，位于`data/`路径下。
        - 使用项目自带的工具可以方便地将原始数据转换`LightningDataModule`，详见更换数据集的教程。
        
    2. 根据需求选择模型。
        - 本项目自带的模型位于`models`模块。
        - 你可以自定义自己的模型，详见自定义模型的教程。

    3. 根据任务类型创建framework。
        - 本项目的`frameworks`包含特定任务所需的一切功能，并根据任务类型进行了分类。
            
        - 本项目的framework位于`frameworks`模块。

    4. 训练和测试模型。
        - 只需简单调用framework的`fit()`方法即可训练和测试模型，训练过程所需的所有功能已为你实现好。以下列出了其中一些功能：
            1. 自动打印超参数、模型结构、训练过程等任何信息
            2. 自动启动Tensorboard（在浏览器中打开`http://localhost:6060`)
            3. 自动保存模型图（在`Tensorboard-Graph`中查看）
            4. 自动计算并保存任务相关指标（在`Tensorboard-Scalar`中查看）
            5. 自动完成并保存任务相关可视化（在`Tensorboard-Image`中查看）
            6. 自动保存超参数及其对应指标（在`Tensorboard-Hparams`中查看）
            7. 基于验证集损失自动保存checkpoint(在`lightning_logs/`路径下)
            8. 训练完成后自动从checkpoint加载训练过程中的最佳模型
        - 完成训练后，最优权重已自动从checkpoint加载。同样地，只需简单调用`framework.test()`即可开始模型测试。
        - 本项目的framework继承自`L.LightningModule`。它与标准的`nn.Module`模块一样，你可以直接调用`framework(input)`或`framework.forward(input)`来执行前向传播。
    """
