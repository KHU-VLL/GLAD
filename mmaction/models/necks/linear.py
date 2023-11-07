# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import NECKS, build_loss


@NECKS.register_module()
class Linear(nn.Module):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.init_std = init_std
        self.loss = build_loss(loss)
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, f, labels=None, domains=None, **kwargs):
        if labels is not None:  # if train
            # [N, in_channels]
            cls_score = self.fc_cls(f)
            # [N, num_classes]
            losses:dict = self.loss(cls_score, labels, domains, **kwargs)
            return f, losses
        else:
            return f, {}
