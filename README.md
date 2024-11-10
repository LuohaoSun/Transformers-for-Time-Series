基于 Lightning 构建的时间序列深度学习模型 Trainers, 方便地进行模型训练、评估和预测。
本项目还包括一些常用的数据集和模型集成。

## How to deploy

1. clone the repo

```bash
git clone https://github.com/LuohaoSun/Transformers-for-Time-Series.git
```

2. clone the submodules

```bash
git submodule update --init --recursive
```

3. install the dependencies

```bash
pip install -r requirements.txt
```

4. run the code

## How to train your own model

```python
model = ...
train_dataloader, val_dataloader = ...
from src.trainers.forecasting_trainer import ForecastingTrainer
trainer = ForecastingTrainer(max_epochs=100, lr=1e-3, ...)
trainer.train(model, train_dataloader, val_dataloader)
```
