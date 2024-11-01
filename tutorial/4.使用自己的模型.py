import sys

sys.path.append(".")
sys.path.append("..")


def main():
    # 在【快速上手】中，我们已经使用了MLP骨干进行了一个简单的分类任务：
    from t4ts.models import MLPBackbone

    backbone1 = MLPBackbone(
        in_seq_len=4096,
        in_features=1,
        hidden_features=[256, 256, 256],
        activation="relu",
    )

    # 在【更换数据集、模型和任务类别】中，我们使用了LSTM骨干进行了一个简单的序列预测任务：
    from t4ts.models import LSTMBackbone

    backbone2 = LSTMBackbone(
        in_features=276,
        hidden_features=128,
        auto_recurrsive_steps=24,
        num_layers=2,
    )

    # 上述模型均为本项目提供的骨干模型，但在实际应用中，我们可能需要根据任务的特点自定义骨干模型。
    # 为了方便使用现有的任务框架，在定义自定义骨干模型时，你只需要遵守一个简单的规则：
    # 模型输入输出张量的形状必须是(batch_size, seq_len, features)。例如：
    import torch
    from torch import nn

    class GRUBackbone(nn.Module):
        def __init__(self, in_features, hidden_features, num_layers):
            super(GRUBackbone, self).__init__()
            self.gru = nn.GRU(
                input_size=in_features,
                hidden_size=hidden_features,
                num_layers=num_layers,
                batch_first=True,
            )

        def forward(self, x):
            # x: (batch_size, seq_len, features)
            # output: (batch_size, seq_len, hidden_features)
            output, _ = self.gru(x)
            return output

    backbone3 = GRUBackbone(in_features=276, hidden_features=128, num_layers=2)

    # 你可以根据需要自定义任意的骨干模型，只要遵守上述规则即可。它会兼容任何任务框架，你只需要同之前一样使用即可：
    from t4ts.frameworks import ForecastingFramework

    framework = ForecastingFramework(
        backbone=backbone3,
        backbone_out_features=128,
        out_seq_len=24,
        out_features=276,
    )
    
    # TODO: add training and testing code here
