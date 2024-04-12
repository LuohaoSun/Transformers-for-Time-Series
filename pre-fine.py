from Modules.patchtst import PatchTSTRandomMaskedReconstructionModel, PatchTSTClassificationModel
from data.bearing_fault_prediction.raw.fault_prediction_datamodule import FaultPredictionDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from torch.utils.data import random_split

import torch

import argparse
import copy
torch.manual_seed(0)


# 参数 ===============================================
parser = argparse.ArgumentParser()
parser.add_argument('--label-ratio', type=float, default=0.05)
args = parser.parse_args()
LABEL_RATIO = args.label_ratio

# 预训练 ===============================================
print(
    '''
    ===========================
    = Starting Pretraining... =
    ===========================
    '''
)

pretrain_datamodule = FaultPredictionDataModule(
    train_val_test_split=(2800, 400, 800),
    batch_size=40,
    num_workers=4,
    pin_memory=True,
)

patchTST = PatchTSTRandomMaskedReconstructionModel(
    in_features=1,
    d_model=128,
    patch_size=16,
    patch_stride=16,
    num_layers=4,
    dropout=0.1,
    nhead=8,
    activation='relu',
    norm_first=False,

    mask_ratio=0.2,
    learnable_mask=False,

    lr=1e-3
)

logger = NeptuneLogger(
    project='bearing-fault-classification',
    name='pretrain',
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNDgzNmZlMC02ZDgyLTQyZDAtOWI4Zi0yMzdiOGU4OTk2N2IifQ=='
)

callbacks = [
    RichProgressBar(),
    ModelCheckpoint(monitor='val_loss', save_top_k=1)
]

trainer = Trainer(
    max_steps=4000,
    accelerator='auto',
    logger=logger,   # type: ignore
    callbacks=callbacks  # type: ignore
)

trainer.fit(patchTST, pretrain_datamodule)
best_model_path = trainer.checkpoint_callback.best_model_path
pretrained_backbone = copy.deepcopy(
    PatchTSTRandomMaskedReconstructionModel.load_from_checkpoint(
        best_model_path).backbone.state_dict())


# 微调 =================================================
def finetune(label_ration):
    print(
        '''
        ===========================
        = Starting Finetuning...  =
        ===========================
        '''
    )

    # finetune_datamodule = FaultPredictionDataModule(
    #     train_val_test_split=(2800, 400, 800),
    #     batch_size=40,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    # finetune_datamodule.data_train = random_split(
        # pretrain_datamodule.data_train, [label_ration, 1-label_ration])[0]
    finetune_datamodule = FaultPredictionDataModule(
        train_val_test_split=(int(label_ration * 2800),
                            400, int((1-label_ration)*2800)+800),
        batch_size=40,
        num_workers=4,
        pin_memory=True,
    )

    patchTST = PatchTSTClassificationModel(
        in_features=1,
    d_model=128,
    patch_size=16,
    patch_stride=16,
    num_layers=4,
    dropout=0.1,
    nhead=8,
    activation='relu',
    norm_first=False,

        num_classes=4,
        lr=1e-3
    )
    patchTST.backbone.load_state_dict(pretrained_backbone)
    logger = NeptuneLogger(
        project='bearing-fault-classification',
        name='finetune',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNDgzNmZlMC02ZDgyLTQyZDAtOWI4Zi0yMzdiOGU4OTk2N2IifQ=='
    )

    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(monitor='val_loss', save_top_k=1)
    ]

    trainer = Trainer(
        max_steps=200,
        accelerator='auto',
        logger=logger,   # type: ignore
        callbacks=callbacks  # type: ignore
    )

    trainer.fit(patchTST, finetune_datamodule)


    # 测试 =================================================
    best_model_path = trainer.checkpoint_callback.best_model_path
    finetuned_model = PatchTSTClassificationModel.load_from_checkpoint(
        best_model_path)
    trainer.test(model=finetuned_model, datamodule=finetune_datamodule)
    
    del finetuned_model
    del patchTST
    del finetune_datamodule
    torch.cuda.empty_cache()

for label_ratio in [0.05, 0.1, 0.2, 0.5, 1.0]:
    finetune(label_ratio)