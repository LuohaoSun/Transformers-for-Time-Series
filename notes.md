### 关于超参数

1. 框架中不进行任何log超参数到logger的操作，仅处理self.hparams
2. 要将超参数log到tensorboard，使用对应的callback, 从self.hparams中过滤出需要log的超参数
3. callback只从self.hparams中获取超参数，不从其他地方获取
4. 由于恢复checkpoint需要保存__init__中所有参数（包括backbone），所以backbone需要保存到self.hparams中。然而，backbone本身不需要log到tensorboard，而是需要将backbone的超参数log到tensorboard。所以，backbone的超参数需要保存到self.hparams中。
5. 基于以上考虑，backbone需要在FrameworkBase中进行规范，可以考虑abstractmethod的方式，也可以考虑直接在__init__中定义。

### 关于不同任务框架共用属性
1. training_step, validation_step, test_step, predict_step, configure_optimizers, loss 