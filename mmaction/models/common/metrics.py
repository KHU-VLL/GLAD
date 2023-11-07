import torch
from mmaction.core import mean_class_accuracy


def calc_mca(cls_score, labels, max_label=-1):
    mca = mean_class_accuracy(
        cls_score.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        max_label
    )
    if isinstance(cls_score, torch.Tensor):
        return torch.tensor(mca, device=cls_score.device)
    else:
        return type(cls_score)(mca)
