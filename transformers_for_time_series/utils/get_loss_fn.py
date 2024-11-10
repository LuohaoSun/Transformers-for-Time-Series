from typing import Callable

import torch.nn as nn


def get_loss_fn(loss_fn: Callable | str) -> Callable:
    if isinstance(loss_fn, Callable):
        return loss_fn
    elif loss_fn.lower() in ["mse", "l2", "l2loss"]:
        return nn.MSELoss()
    elif loss_fn.lower() == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_fn.lower() == "nll":
        return nn.NLLLoss()
    elif loss_fn.lower() in ["mae", "l1", "l1loss"]:
        return nn.L1Loss()
    elif loss_fn.lower() == "kl_div":
        return nn.KLDivLoss()
    elif loss_fn.lower() == "bce":
        return nn.BCELoss()
    elif loss_fn.lower() == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_fn.lower() == "margin_ranking":
        return nn.MarginRankingLoss()
    elif loss_fn.lower() == "hinge_embedding":
        return nn.HingeEmbeddingLoss()
    elif loss_fn.lower() == "multi_margin":
        return nn.MultiMarginLoss()
    elif loss_fn.lower() == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_fn.lower() == "soft_margin":
        return nn.SoftMarginLoss()
    elif loss_fn.lower() == "multi_label_soft_margin":
        return nn.MultiLabelSoftMarginLoss()
    elif loss_fn.lower() == "cosine_embedding":
        return nn.CosineEmbeddingLoss()
    elif loss_fn.lower() == "multi_label_margin":
        return nn.MultiLabelMarginLoss()
    elif loss_fn.lower() == "triplet_margin":
        return nn.TripletMarginLoss()
    else:
        raise ValueError(f"loss_fn {loss_fn} not supported.")
