import torch.nn as nn

from typing import Callable



def get_loss_fn(loss_fn: Callable | str) -> Callable:

    if isinstance(loss_fn, Callable):
        return loss_fn
    elif loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_fn == "nll":
        return nn.NLLLoss()
    elif loss_fn == "l1" or loss_fn == "mae":
        return nn.L1Loss()
    elif loss_fn == "kl_div":
        return nn.KLDivLoss()
    elif loss_fn == "bce":
        return nn.BCELoss()
    elif loss_fn == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_fn == "margin_ranking":
        return nn.MarginRankingLoss()
    elif loss_fn == "hinge_embedding":
        return nn.HingeEmbeddingLoss()
    elif loss_fn == "multi_margin":
        return nn.MultiMarginLoss()
    elif loss_fn == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_fn == "soft_margin":
        return nn.SoftMarginLoss()
    elif loss_fn == "multi_label_soft_margin":
        return nn.MultiLabelSoftMarginLoss()
    elif loss_fn == "cosine_embedding":
        return nn.CosineEmbeddingLoss()
    elif loss_fn == "multi_label_margin":
        return nn.MultiLabelMarginLoss()
    elif loss_fn == "triplet_margin":
        return nn.TripletMarginLoss()
    else:
        raise ValueError(f"loss_fn {loss_fn} not supported.")